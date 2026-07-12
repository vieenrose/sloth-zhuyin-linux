// slothingd_slothe: a Unix-socket daemon that serves the validated ternary IME
// encoder (libslothe / ggml) over the same protocol as slothingd, so the
// existing fcitx5 and IBus socket clients run on the 25M ternary model.
//
// Unlike slothingd (which drove a *generative* llama.cpp model with a
// grammar-constrained sampler), this daemon runs the *bidirectional-encoder*
// forward pass from slothe.{h,cpp} and decodes with the legality-masked
// argmax scorer ported from slothe_score.cpp. The forward is already validated
// against the PyTorch golden reference (blk0 ~7e-5, argmax 67/68).
//
// Protocol (decode mode only): one JSON object per connection, one JSON object
// back, then the connection closes.
//
// Request:  {"syllables": ["ㄕㄣ","ㄊㄧˇ",...], "n": 1, "context": "..."}
//           Syllables may carry tones ("ㄊㄧˇ") or omit them ("ㄊㄧ"); a
//           toneless syllable's legality unions all tones of that base
//           syllable. A token that is not a bopomofo syllable (a code-switch
//           English word) passes through verbatim as a single literal
//           candidate.
// Response: {"sentences": ["身體總是在喪失水分", ...]}
//           (greedy/best first, deduped) or {"error": "message"}.
//
// NOTE on "context": the deployed GGUF was exported with char_hints=FALSE, so
// the encoder cannot consume preceding text via character hints. The "context"
// field is therefore accepted but IGNORED here (a known capability gap versus
// the older 11.6M hinted ONNX model). Sending it is not an error.

#include "slothe.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
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
            "usage: %s -m model.gguf -t phonetic_table.tsv -v syl_vocab.json\n"
            "          -j char2id.json [-s /path/to/socket]\n",
            argv0);
}

// Default socket path: a per-user private location so that (a) another local
// user cannot connect and read what is being typed, and (b) it matches the
// path the fcitx5 engine derives. Falls back to /tmp only if XDG_RUNTIME_DIR
// is unset. Kept identical to slothingd.cpp's derivation so the same clients
// connect unchanged.
std::string default_socket_path() {
    if (const char * env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    if (const char * xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::string(xdg) + "/slothingd.sock";
    }
    return "/tmp/slothingd.sock";
}

// ---- Phonetic table + UTF-8 helpers (verbatim from slothingd.cpp) ----
// bopomofo syllable -> the characters legal for that reading. `tonal` is keyed
// by the exact syllable; `toneless` unions every tone of a base syllable, in
// per-tone file order, deduped across tones.
struct PhoneticTable {
    std::unordered_map<std::string, std::vector<std::string>> tonal;
    std::unordered_map<std::string, std::vector<std::string>> toneless;
    bool loaded = false;
};

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

bool has_tone_mark(const std::string & syl) {
    for (const auto & c : utf8_chars(syl)) {
        if (is_tone_mark(c)) {
            return true;
        }
    }
    return false;
}

// Candidate reading chars for one syllable. Mirrors slothe_score.cpp's
// candidates() EXACTLY, which is the validated decode logic:
//   * The syllable is "tone-bearing" iff it carries a tone MARK (ˊˇˋ˙).
//     Tone 1 is unmarked, so an unmarked syllable is treated as toneless.
//   * tone-bearing + found  -> its exact tonal candidate list;
//   * otherwise (toneless, OR tone-bearing but not found) -> the ordered union
//     of every tone of the base (file order, deduped across tones);
//   * nothing legal -> the token itself as a single literal candidate (this is
//     how code-switch English tokens pass through verbatim).
// NOTE: deciding by tone-mark presence (not by whether `tonal` happens to hold
// an unmarked tone-1 key) is what keeps this faithful to the golden reference;
// looking `tonal` up first would wrongly restrict unmarked syllables to tone 1.
const std::vector<std::string> & syllable_candidates(const PhoneticTable & table,
                                                     const std::string & syl,
                                                     std::vector<std::string> & literal_scratch) {
    if (has_tone_mark(syl)) {
        auto it = table.tonal.find(syl);
        if (it != table.tonal.end()) {
            return it->second;
        }
    }
    auto it2 = table.toneless.find(strip_tones(syl));
    if (it2 != table.toneless.end()) {
        return it2->second;
    }
    literal_scratch.assign(1, syl);
    return literal_scratch;
}

// ---- syl_vocab.json / char2id.json: {"utf8key": int, ...} ----
bool load_int_map(const std::string & path, std::unordered_map<std::string, int> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    json j;
    try {
        f >> j;
    } catch (const std::exception &) {
        return false;
    }
    if (!j.is_object()) {
        return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.value().is_number_integer() || it.value().is_number_unsigned()) {
            out[it.key()] = it.value().get<int>();
        }
    }
    return !out.empty();
}

// ---- request read / response write (verbatim from slothingd.cpp) ----
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
    std::string vocab_path;
    std::string char2id_path;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            table_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            char2id_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || table_path.empty() || vocab_path.empty() ||
        char2id_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    PhoneticTable table;
    if (!load_phonetic_table(table_path, table)) {
        fprintf(stderr, "error: failed to load phonetic table '%s'\n", table_path.c_str());
        return 1;
    }
    std::unordered_map<std::string, int> syl_vocab;
    if (!load_int_map(vocab_path, syl_vocab)) {
        fprintf(stderr, "error: failed to load syl_vocab '%s'\n", vocab_path.c_str());
        return 1;
    }
    std::unordered_map<std::string, int> char2id;
    if (!load_int_map(char2id_path, char2id)) {
        fprintf(stderr, "error: failed to load char2id '%s'\n", char2id_path.c_str());
        return 1;
    }

    slothe_model * model = slothe_load(model_path.c_str());
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }
    const int n_char = slothe_n_char(model);
    fprintf(stderr,
            "slothingd_slothe: loaded model (n_char=%d, %zu syllables, "
            "%zu toneless, syl_vocab=%zu, char2id=%zu)\n",
            n_char, table.tonal.size(), table.toneless.size(), syl_vocab.size(),
            char2id.size());

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
    fprintf(stderr, "slothingd_slothe: listening on %s\n", socket_path.c_str());

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
            if (!req.contains("syllables")) {
                throw std::runtime_error("request needs a 'syllables' array");
            }
            std::vector<std::string> syllables =
                req.at("syllables").get<std::vector<std::string>>();
            if (syllables.empty()) {
                throw std::runtime_error("syllables must be non-empty");
            }
            int n_alternatives = req.value("n", 1);
            if (n_alternatives < 1) {
                n_alternatives = 1;
            }
            if (n_alternatives > 8) {
                n_alternatives = 8;
            }
            // "context" is accepted for protocol compatibility but ignored:
            // the deployed GGUF has char_hints=FALSE, so the encoder cannot
            // consume preceding text. (Known gap vs the 11.6M hinted ONNX.)
            (void) req;

            const int T = (int) syllables.size();

            // syl_ids[i] = syl_vocab.get(syllable, 1)  (1 == <unk>)
            std::vector<int32_t> syl_ids(T);
            for (int i = 0; i < T; i++) {
                auto it = syl_vocab.find(syllables[i]);
                syl_ids[i] = (int32_t)(it == syl_vocab.end() ? 1 : it->second);
            }

            // encoder forward
            std::vector<float> logits((size_t) T * n_char);
            slothe_logits(model, syl_ids.data(), T, logits.data());

            // Per position: rank the phonetically-legal candidates that exist
            // in char2id by their logit; keep the ordered list so we can build
            // n-best by flipping the tightest positions to 2nd best.
            struct Ranked {
                std::vector<std::string> chars; // sorted by logit desc
                std::vector<float> vals;         // parallel logits
            };
            std::vector<Ranked> ranked(T);
            std::vector<std::string> literal_scratch;
            for (int i = 0; i < T; i++) {
                const std::vector<std::string> & cands =
                    syllable_candidates(table, syllables[i], literal_scratch);
                const float * row = &logits[(size_t) i * n_char];
                std::vector<std::pair<float, std::string>> scored;
                for (const auto & ch : cands) {
                    auto it = char2id.find(ch);
                    if (it == char2id.end()) {
                        continue; // char absent from char2id: skip (mirror gate)
                    }
                    scored.emplace_back(row[it->second], ch);
                }
                if (scored.empty()) {
                    // No candidate is representable: emit a literal placeholder
                    // so positions stay aligned. Matches the gate's '?' fallback
                    // but here we keep the first raw candidate if any exists.
                    if (!cands.empty()) {
                        ranked[i].chars.push_back(cands.front());
                        ranked[i].vals.push_back(0.0f);
                    } else {
                        ranked[i].chars.push_back("?");
                        ranked[i].vals.push_back(0.0f);
                    }
                    continue;
                }
                std::stable_sort(scored.begin(), scored.end(),
                                 [](const std::pair<float, std::string> & a,
                                    const std::pair<float, std::string> & b) {
                                     return a.first > b.first;
                                 });
                for (auto & p : scored) {
                    ranked[i].chars.push_back(p.second);
                    ranked[i].vals.push_back(p.first);
                }
            }

            // Primary (greedy) sentence: top-1 at every position.
            std::vector<size_t> pick(T, 0);
            auto join_pick = [&](const std::vector<size_t> & sel) {
                std::string s;
                for (int i = 0; i < T; i++) {
                    s += ranked[i].chars[sel[i]];
                }
                return s;
            };

            std::vector<std::string> sentences;
            sentences.push_back(join_pick(pick));

            if (n_alternatives > 1) {
                // Positions that actually have a 2nd candidate, ranked by how
                // close their top-1 and top-2 logits are (smallest margin =
                // most likely to be wrong = flip first). Simple single-flip
                // n-best: cheap, deterministic, and dedup keeps it honest.
                std::vector<std::pair<float, int>> margins;
                for (int i = 0; i < T; i++) {
                    if (ranked[i].chars.size() >= 2) {
                        margins.emplace_back(ranked[i].vals[0] - ranked[i].vals[1], i);
                    }
                }
                std::stable_sort(margins.begin(), margins.end(),
                                 [](const std::pair<float, int> & a,
                                    const std::pair<float, int> & b) {
                                     return a.first < b.first;
                                 });
                for (const auto & mp : margins) {
                    if ((int) sentences.size() >= n_alternatives) {
                        break;
                    }
                    std::vector<size_t> alt = pick;
                    alt[mp.second] = 1; // flip that one position to 2nd best
                    std::string s = join_pick(alt);
                    if (std::find(sentences.begin(), sentences.end(), s) ==
                        sentences.end()) {
                        sentences.push_back(std::move(s));
                    }
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

    slothe_free(model);
    close(listen_fd);
    return 0;
}
