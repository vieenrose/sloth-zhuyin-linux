// slothingd: a minimal Unix-socket daemon that reranks libchewing candidate
// lists with a small local LLM, using llama.cpp's C API directly (no
// llama-server, no Python bindings).
//
// Protocol: one JSON object per connection on stdin-style read, one JSON
// object written back, then the connection is closed.
//
// Request:  {"positions": [["你","妳","擬"], ["好","號","毫"], ...]}
// Response: {"sentence": "你好"}
//        or {"error": "message"}

#include "llama.h"
#include "nlohmann/json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m model.gguf [-s /path/to/socket] [-c context_size]\n", argv0);
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

std::string build_user_message(const std::vector<std::vector<std::string>> & positions) {
    std::string msg;
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
        ssize_t n = write(fd, data.data() + off, data.size() - off);
        if (n <= 0) {
            break;
        }
        off += n;
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string socket_path = "/tmp/slothingd.sock";
    int n_ctx = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
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

        std::string req_str = read_request(fd);
        std::string resp_str;
        try {
            json req = json::parse(req_str);
            std::vector<std::vector<std::string>> positions =
                req.at("positions").get<std::vector<std::vector<std::string>>>();
            if (positions.empty()) {
                throw std::runtime_error("positions must be non-empty");
            }

            // start every request from a clean KV cache: each call is an
            // independent classification, not a multi-turn conversation.
            llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

            std::vector<llama_chat_message> messages;
            messages.push_back({"system", "選字。"});
            std::string user_msg = build_user_message(positions);
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
            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

            const int n_prompt_tokens =
                -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
            std::vector<llama_token> prompt_tokens(n_prompt_tokens);
            llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                           prompt_tokens.size(), true, true);

            // safety cap: real answers finish once the grammar completes, but
            // guard against a runaway loop if the model refuses to converge.
            const int max_new_tokens = (int) positions.size() * 4 + 8;

            std::string sentence;
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
            for (int step = 0; step < max_new_tokens; step++) {
                if (llama_decode(ctx, batch) != 0) {
                    throw std::runtime_error("llama_decode failed");
                }
                llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, new_token)) {
                    break;
                }
                char buf[256];
                int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    throw std::runtime_error("failed to detokenize");
                }
                sentence.append(buf, n);
                batch = llama_batch_get_one(&new_token, 1);
            }

            llama_sampler_free(smpl);

            json resp;
            resp["sentence"] = sentence;
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
