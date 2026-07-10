// slothingd protocol client — frontend-free, shared by the fcitx5 and IBus
// engines (extracted from eim.cpp so both speak the daemon with one codec).
// Blocking Unix-socket JSON requests with a hard timeout; callers that must
// not block the UI thread run these on a worker and publish the fd into an
// atomic slot so teardown can shutdown() it.
#ifndef _SLOTHING_COMMON_DAEMON_H_
#define _SLOTHING_COMMON_DAEMON_H_

#include "nlohmann/json.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace slothing {

using json = nlohmann::json;

constexpr int kSlothingdTimeoutMs = 3000;

// Keep identical to default_socket_path() in slothingd and
// packaging/run-slothingd.sh.
inline std::string slothingdSocketPath() {
    if (const char *env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    if (const char *xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::string(xdg) + "/slothingd.sock";
    }
    return "/tmp/slothingd.sock";
}

enum class DaemonError { None, Connect, Io, Empty };

// Sends a request to slothingd and returns its "sentences". Publishes the
// connected fd into `fdSlot` so the owner can shutdown() it to unblock
// this thread at teardown.
inline std::vector<std::string>
sendDaemonRequest(const std::string &req, std::atomic<int> &fdSlot,
                  DaemonError &err, json *fullOut = nullptr) {
    err = DaemonError::Empty;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        err = DaemonError::Connect;
        return {};
    }
    struct timeval tv;
    tv.tv_sec = kSlothingdTimeoutMs / 1000;
    tv.tv_usec = (kSlothingdTimeoutMs % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::string sockPath = slothingdSocketPath();
    strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);
    if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) !=
        0) {
        close(fd);
        err = DaemonError::Connect;
        return {};
    }
    fdSlot.store(fd);
    auto finish = [&](std::vector<std::string> result) {
        fdSlot.store(-1);
        close(fd);
        return result;
    };

    size_t off = 0;
    while (off < req.size()) {
        ssize_t w = send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (w <= 0) {
            err = DaemonError::Io;
            return finish({});
        }
        off += static_cast<size_t>(w);
    }
    shutdown(fd, SHUT_WR);

    std::string resp;
    char buf[4096];
    ssize_t r;
    while ((r = read(fd, buf, sizeof(buf))) > 0) {
        resp.append(buf, static_cast<size_t>(r));
    }
    if (resp.empty()) {
        err = DaemonError::Io;
        return finish({});
    }
    try {
        json parsed = json::parse(resp);
        if (parsed.contains("sentences")) {
            if (fullOut) {
                *fullOut = parsed;
            }
            auto sentences = parsed["sentences"].get<std::vector<std::string>>();
            if (!sentences.empty()) {
                err = DaemonError::None;
            }
            return finish(std::move(sentences));
        }
    } catch (const std::exception &) {
    }
    return finish({});
}

inline std::string buildDecodeRequest(const std::vector<std::string> &syllables,
                                      int n, const std::string &context) {
    json req;
    req["syllables"] = syllables;
    req["n"] = n;
    if (!context.empty()) {
        req["context"] = context;
    }
    return req.dump() + "\n";
}

inline std::vector<std::string>
queryDecoder(const std::vector<std::string> &syllables, int n,
             const std::string &context, std::atomic<int> &fdSlot,
             DaemonError &err, json *fullOut = nullptr) {
    if (syllables.empty()) {
        err = DaemonError::Empty;
        return {};
    }
    return sendDaemonRequest(buildDecodeRequest(syllables, n, context), fdSlot,
                             err, fullOut);
}

// Decode conditioned on user picks: {"hints": {"2": "在"}}. Synchronous but
// ~1 ms; a dead daemon fails the connect instantly.
inline std::vector<std::string>
queryDecoderWithHints(const std::vector<std::string> &syllables,
                      const std::map<int, std::string> &hints,
                      json *fullOut = nullptr) {
    json req;
    req["syllables"] = syllables;
    req["n"] = 1;
    json h = json::object();
    for (const auto &[i, ch] : hints) {
        h[std::to_string(i)] = ch;
    }
    req["hints"] = h;
    std::atomic<int> fd{-1};
    DaemonError err = DaemonError::None;
    return sendDaemonRequest(req.dump() + "\n", fd, err, fullOut);
}

// Fire-and-forget: persist the user's corrections in the daemon's learn
// store (detached thread; a dead daemon just drops it).
inline void sendLearn(const json &learnBody) {
    std::thread([req = json{{"learn", learnBody}}.dump() + "\n"]() {
        std::atomic<int> fd{-1};
        DaemonError err = DaemonError::None;
        sendDaemonRequest(req, fd, err);
    }).detach();
}

// Model-ranked 2-char phrase candidates for positions (at, at+1) with their
// joint probabilities. Synchronous but tiny (one encoder forward, ~2 ms).
inline std::vector<std::pair<double, std::string>>
queryPhrasesScored(const std::vector<std::string> &syllables, int at, int n) {
    json req;
    req["syllables"] = syllables;
    req["phrase_at"] = at;
    req["n"] = n;
    std::atomic<int> fd{-1};
    DaemonError err = DaemonError::None;
    json full;
    auto phrases = sendDaemonRequest(req.dump() + "\n", fd, err, &full);
    std::vector<std::pair<double, std::string>> out;
    for (size_t i = 0; i < phrases.size(); i++) {
        double p = 0.0;
        if (full.contains("scores") && i < full["scores"].size()) {
            p = full["scores"][i].get<double>();
        }
        out.emplace_back(p, std::move(phrases[i]));
    }
    return out;
}

} // namespace slothing

#endif // _SLOTHING_COMMON_DAEMON_H_
