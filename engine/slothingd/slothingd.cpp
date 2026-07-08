// slothingd: a minimal Unix-socket daemon that reranks libchewing candidate
// lists with a small local LLM, using llama.cpp's C API directly (no
// llama-server, no Python bindings).
//
// Protocol: one JSON object per connection on stdin-style read, one JSON
// object written back, then the connection is closed.
//
// Request (rerank):
//           {"positions": [["你","妳","擬"], ["好","號","毫"], ...], "n": 3,
//            "context": "已經送出的前文"}
//           ("n" = max alternatives wanted, optional, defaults to 1;
//            "context" = text before the cursor, optional, tail-truncated)
// Request (decode, libchewing-free; needs -t phonetic_table.tsv):
//           {"syllables": ["ㄨㄛˇ","ㄗㄞˋ",...], "n": 1, "context": "..."}
//           Syllables may omit tones ("ㄨㄛ") -- legality then unions all
//           tones of that base syllable. Each syllable becomes one grammar
//           position of its phonetically-legal characters; the model decodes
//           the sentence directly (z2t / toneless training tasks).
// Response: {"sentences": ["你好", "妳好"]}
//           (deduped, best/greedy first; only fully grammar-complete sentences)
//        or {"error": "message"}

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

// A client that connects but never sends a complete request must not be able
// to wedge the single-threaded accept loop; bound each read with this.
constexpr int kClientReadTimeoutSec = 5;

void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [-s /path/to/socket] [-c context_size]\n"
            "          [-t phonetic_table.tsv]   enable decode mode\n",
            argv0);
}

// Default socket path: a per-user private location so that (a) another local
// user cannot connect and read what is being typed, and (b) it matches the
// path the fcitx5 engine derives. Falls back to /tmp only if XDG_RUNTIME_DIR
// is unset. Keep this identical to slothingdSocketPath() in eim.cpp and the
// derivation in packaging/run-slothingd.sh.
std::string default_socket_path() {
    if (const char * env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    if (const char * xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::string(xdg) + "/slothingd.sock";
    }
    return "/tmp/slothingd.sock";
}

// Phonetic table for decode mode: bopomofo syllable -> the characters legal
// for that reading (model/phonetic_table.tsv, built by
// model/build_phonetic_table.py from the SAME reading source as SlothLM's
// z2t/toneless training data). `tonal` is keyed by the exact syllable;
// `toneless` unions every tone of a base syllable, for tone-free typing.
struct PhoneticTable {
    std::unordered_map<std::string, std::vector<std::string>> tonal;
    std::unordered_map<std::string, std::vector<std::string>> toneless;
    bool loaded = false;
};

// Split a UTF-8 string into single characters (bopomofo symbols and Han
// characters are one codepoint each).
std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t len = 1;
        unsigned char c = s[i];
        if ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

bool is_tone_mark(const std::string & c) {
    return c == "ˊ" || c == "ˇ" || c == "ˋ" || c == "˙";
}

std::string strip_tones(const std::string & syl) {
    std::string out;
    for (const auto & c : utf8_chars(syl)) {
        if (!is_tone_mark(c)) {
            out += c;
        }
    }
    return out;
}

bool load_phonetic_table(const std::string & path, PhoneticTable & table) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::unordered_map<std::string, std::unordered_set<std::string>> seen;
    std::string line;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        std::string syl = line.substr(0, tab);
        std::vector<std::string> chars = utf8_chars(line.substr(tab + 1));
        table.tonal[syl] = chars;
        // toneless union preserves per-tone order, dedupes across tones
        std::string base = strip_tones(syl);
        auto & dst = table.toneless[base];
        auto & dup = seen[base];
        for (auto & ch : chars) {
            if (dup.insert(ch).second) {
                dst.push_back(ch);
            }
        }
    }
    table.loaded = !table.tonal.empty();
    return table.loaded;
}

// Quote a candidate string as a GBNF string literal.
std::string gbnf_quote(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    out += '"';
    return out;
}

// Builds a GBNF grammar that accepts exactly one candidate per position, in
// order, e.g. root ::= p0 p1\np0 ::= "你" | "妳"\np1 ::= "好" | "號"\n
std::string build_grammar(const std::vector<std::vector<std::string>> & positions) {
    std::string grammar = "root ::=";
    for (size_t i = 0; i < positions.size(); i++) {
        grammar += " p" + std::to_string(i);
    }
    grammar += "\n";
    for (size_t i = 0; i < positions.size(); i++) {
        grammar += "p" + std::to_string(i) + " ::=";
        for (size_t j = 0; j < positions[i].size(); j++) {
            grammar += (j == 0 ? " " : " | ") + gbnf_quote(positions[i][j]);
        }
        grammar += "\n";
    }
    return grammar;
}

std::string build_user_message(const std::vector<std::vector<std::string>> & positions,
                               const std::string & context) {
    std::string msg;
    if (!context.empty()) {
        // Text already committed before the cursor; lets the model pick
        // candidates that continue the document, not just an isolated
        // sentence (e.g. 在 vs 再 given what came before). Phrased as a
        // fill-in-the-blank continuation, which small models follow better
        // than a meta description of "preceding text".
        msg += context + "＿＿＿\n";
    }
    for (size_t i = 0; i < positions.size(); i++) {
        if (i > 0) {
            msg += " ";
        }
        msg += "第" + std::to_string(i + 1) + "字選(";
        for (size_t j = 0; j < positions[i].size(); j++) {
            if (j > 0) {
                msg += "/";
            }
            msg += positions[i][j];
        }
        msg += ")";
    }
    return msg;
}

// Reads a full request off `fd` (client writes then shuts down its write
// side, or sends a trailing newline) and returns it as a string.
std::string read_request(int fd) {
    std::string data;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        data.append(buf, n);
        if (!data.empty() && data.back() == '\n') {
            break;
        }
    }
    return data;
}

void write_all(int fd, const std::string & data) {
    size_t off = 0;
    while (off < data.size()) {
        // MSG_NOSIGNAL: a client that closed early (e.g. timed out) must yield
        // EPIPE here, never a process-killing SIGPIPE. (SIGPIPE is also
        // ignored globally in main() as a belt-and-braces measure.)
        ssize_t n = send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) {
            break;
        }
        off += n;
    }
}

} // namespace

int main(int argc, char ** argv) {
    // Never let a write to a disconnected client kill the daemon.
    signal(SIGPIPE, SIG_IGN);

    std::string model_path;
    std::string socket_path = default_socket_path();
    std::string table_path;
    int n_ctx = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            table_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    PhoneticTable table;
    if (!table_path.empty()) {
        if (!load_phonetic_table(table_path, table)) {
            fprintf(stderr, "error: failed to load phonetic table '%s'\n",
                    table_path.c_str());
            return 1;
        }
        fprintf(stderr, "slothingd: decode mode on (%zu syllables, %zu toneless)\n",
                table.tonal.size(), table.toneless.size());
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU-only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    unlink(socket_path.c_str());
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "error: socket() failed: %s\n", strerror(errno));
        return 1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    if (bind(listen_fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: bind(%s) failed: %s\n", socket_path.c_str(), strerror(errno));
        return 1;
    }
    if (listen(listen_fd, 8) != 0) {
        fprintf(stderr, "error: listen() failed: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "slothingd: listening on %s\n", socket_path.c_str());

    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

    while (true) {
        int fd = accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }

        // Bound the read so a client that connects and stalls without sending
        // a full request cannot wedge this single-threaded loop forever.
        struct timeval rtv;
        rtv.tv_sec = kClientReadTimeoutSec;
        rtv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));

        std::string req_str = read_request(fd);
        std::string resp_str;
        try {
            json req = json::parse(req_str);
            std::vector<std::vector<std::string>> positions;
            bool decode_mode = req.contains("syllables");
            std::vector<std::string> syllables;
            if (decode_mode) {
                if (!table.loaded) {
                    throw std::runtime_error(
                        "decode mode needs -t phonetic_table.tsv");
                }
                syllables = req.at("syllables").get<std::vector<std::string>>();
                if (syllables.empty()) {
                    throw std::runtime_error("syllables must be non-empty");
                }
                for (const auto & syl : syllables) {
                    // exact (tonal) lookup first; a syllable with no tone mark
                    // falls back to the toneless union of all its tones.
                    const std::vector<std::string> * cands = nullptr;
                    auto it = table.tonal.find(syl);
                    if (it != table.tonal.end()) {
                        cands = &it->second;
                    } else {
                        auto it2 = table.toneless.find(strip_tones(syl));
                        if (it2 != table.toneless.end()) {
                            cands = &it2->second;
                        }
                    }
                    if (cands) {
                        positions.push_back(*cands);
                    } else {
                        // Not a bopomofo syllable: a code-switch English token
                        // (React, Python, ...). Pass it through as a literal
                        // single-candidate position so the grammar keeps it
                        // verbatim while the model decodes the zhuyin around it.
                        positions.push_back({syl});
                    }
                }
            } else {
                positions =
                    req.at("positions").get<std::vector<std::vector<std::string>>>();
                if (positions.empty()) {
                    throw std::runtime_error("positions must be non-empty");
                }
            }
            int n_alternatives = req.value("n", 1);
            if (n_alternatives < 1) {
                n_alternatives = 1;
            }
            if (n_alternatives > 8) {
                n_alternatives = 8;
            }
            std::string context = req.value("context", std::string());
            // Keep the prompt bounded even if a client sends a huge context.
            constexpr size_t kMaxContextBytes = 240;
            if (context.size() > kMaxContextBytes) {
                size_t cut = context.size() - kMaxContextBytes;
                // don't split a UTF-8 sequence at the cut point
                while (cut < context.size() &&
                       (static_cast<unsigned char>(context[cut]) & 0xC0) == 0x80) {
                    cut++;
                }
                context.erase(0, cut);
            }

            // start every request from a clean KV cache: each call is an
            // independent classification, not a multi-turn conversation.
            llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

            // Decode mode uses the exact z2t/toneless training format
            // (system 注音轉繁體中文。, user = space-joined syllables);
            // rerank keeps the SELECT format. Context, when present, is
            // prepended as a fill-in-the-blank continuation in both.
            std::vector<llama_chat_message> messages;
            std::string user_msg;
            if (decode_mode) {
                messages.push_back({"system", "注音轉繁體中文。"});
                if (!context.empty()) {
                    user_msg += context + "＿＿＿\n";
                }
                for (size_t i = 0; i < syllables.size(); i++) {
                    if (i > 0) {
                        user_msg += " ";
                    }
                    user_msg += syllables[i];
                }
            } else {
                messages.push_back({"system", "選字。"});
                user_msg = build_user_message(positions, context);
            }
            messages.push_back({"user", user_msg.c_str()});

            std::vector<char> formatted(n_ctx);
            int len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                 formatted.data(), formatted.size());
            if (len > (int) formatted.size()) {
                formatted.resize(len);
                len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                 formatted.data(), formatted.size());
            }
            if (len < 0) {
                throw std::runtime_error("failed to apply chat template");
            }
            std::string prompt(formatted.begin(), formatted.begin() + len);

            std::string grammar_str = build_grammar(positions);

            const int n_prompt_tokens =
                -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
            std::vector<llama_token> prompt_tokens(n_prompt_tokens);
            llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                           prompt_tokens.size(), true, true);
            if (prompt_tokens.empty()) {
                throw std::runtime_error("empty prompt");
            }

            // Decode the prompt except its last token once; each alternative
            // pass then re-decodes just that last token to refresh the
            // logits, generates, and trims the KV cache back. This makes
            // extra alternatives cost only their own generated tokens, not a
            // full prompt re-eval.
            const int n_prefix = (int) prompt_tokens.size() - 1;
            if (n_prefix > 0) {
                llama_batch prefix_batch =
                    llama_batch_get_one(prompt_tokens.data(), n_prefix);
                if (llama_decode(ctx, prefix_batch) != 0) {
                    throw std::runtime_error("llama_decode failed (prompt)");
                }
            }
            llama_token last_prompt_token = prompt_tokens.back();

            // Token budget: the longest legal sentence is one where every
            // position takes its longest-tokenizing candidate. Cap at that
            // (plus slack for EOG / template quirks) rather than a fixed
            // tokens-per-position guess, which truncated long-phrase
            // candidates mid-grammar. A generation that still hits this cap
            // without the grammar completing is discarded below.
            size_t budget = 0;
            for (const auto & pos : positions) {
                int longest = 0;
                for (const auto & cand : pos) {
                    int t = -llama_tokenize(vocab, cand.c_str(), cand.size(),
                                            NULL, 0, false, true);
                    if (t > longest) {
                        longest = t;
                    }
                }
                budget += (size_t) longest;
            }
            const int max_new_tokens = (int) budget + 8;

            std::vector<std::string> sentences;
            for (int pass = 0; pass < n_alternatives; pass++) {
                // drop everything after the prompt prefix from the KV cache
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prefix, -1);

                llama_sampler * smpl =
                    llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(
                    smpl, llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
                if (pass == 0) {
                    // the primary answer stays deterministic
                    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
                } else {
                    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
                    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1000 + pass));
                }

                std::string sentence;
                bool complete = false; // grammar reached an accepting state (EOG)
                llama_batch batch = llama_batch_get_one(&last_prompt_token, 1);
                for (int step = 0; step < max_new_tokens; step++) {
                    if (llama_decode(ctx, batch) != 0) {
                        llama_sampler_free(smpl);
                        throw std::runtime_error("llama_decode failed");
                    }
                    llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
                    if (llama_vocab_is_eog(vocab, new_token)) {
                        complete = true;
                        break;
                    }
                    char buf[256];
                    int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
                    if (n < 0) {
                        llama_sampler_free(smpl);
                        throw std::runtime_error("failed to detokenize");
                    }
                    sentence.append(buf, n);
                    batch = llama_batch_get_one(&new_token, 1);
                }
                llama_sampler_free(smpl);

                // Only surface a fully-formed sentence: a run that exhausted
                // the token budget without the grammar completing is a
                // truncated fragment, never a valid answer.
                if (complete && !sentence.empty() &&
                    std::find(sentences.begin(), sentences.end(), sentence) ==
                        sentences.end()) {
                    sentences.push_back(std::move(sentence));
                }
            }

            json resp;
            resp["sentences"] = sentences;
            resp_str = resp.dump();
        } catch (const std::exception & e) {
            json resp;
            resp["error"] = e.what();
            resp_str = resp.dump();
        }

        resp_str += "\n";
        write_all(fd, resp_str);
        close(fd);
    }

    llama_free(ctx);
    llama_model_free(model);
    close(listen_fd);
    return 0;
}
