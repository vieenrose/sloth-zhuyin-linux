/*
 * SPDX-FileCopyrightText: 2010~2017 CSSlayer <wengxt@gmail.com>
 * SPDX-FileCopyrightText: 2026 sloth-zhuyin-linux
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
// Slothing engine: a libchewing-free zhuyin input method. Keystrokes are
// parsed by the ZhuyinBuffer FSM (zhuyin.h); the typed syllables are decoded
// to Traditional Chinese by the local SlothLM model via slothingd's decode
// mode, and presented as an editable segment list. No libchewing.
#include "eim.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcitx-utils/log.h>
#include <fcitx-utils/standardpath.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/text.h>
#include <fcitx/userinterfacemanager.h>
#include <fstream>
#include <thread>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;

namespace fcitx {

FCITX_DEFINE_LOG_CATEGORY(slothing_log, "chewing");
#define SLOTHING_DEBUG() FCITX_LOGC(slothing_log, Debug)

namespace {

constexpr int kSlothingdTimeoutMs = 3000;

// Digit/label sets for the selection keys (index by ChewingSelectionKey).
const char *const builtin_selectkeys[] = {
    "1234567890", "asdfghjkl;", "asdfzxcv89",
    "asdfjkl789", "aoeuhtn789", "1234qweras", "dstnaeo789",
};

// Slice UTF-8 chars [from, to) of s (char indices, not bytes).
std::string utf8CharSlice(const std::string &s, int from, int to) {
    auto begin = s.begin();
    size_t fromByte = utf8::ncharByteLength(begin, from);
    size_t toByte = utf8::ncharByteLength(begin, to);
    return s.substr(fromByte, toByte - fromByte);
}

// Keep identical to default_socket_path() in slothingd.cpp and
// packaging/run-slothingd.sh.
std::string slothingdSocketPath() {
    if (const char *env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    if (const char *xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::string(xdg) + "/slothingd.sock";
    }
    return "/tmp/slothingd.sock";
}

// Verifies `sentence` is exactly one candidate from each entry of `positions`,
// in order -- so it can only be a combination of legal characters, never
// arbitrary model output. Longest-match per position.
bool matchesPositions(const std::string &sentence,
                      const std::vector<std::vector<std::string>> &positions) {
    size_t off = 0;
    for (const auto &cands : positions) {
        size_t best = 0;
        for (const auto &c : cands) {
            if (c.size() > best && sentence.compare(off, c.size(), c) == 0) {
                best = c.size();
            }
        }
        if (best == 0) {
            return false;
        }
        off += best;
    }
    return off == sentence.size();
}

std::string buildDecodeRequest(const std::vector<std::string> &syllables, int n,
                               const std::string &context) {
    json req;
    req["syllables"] = syllables;
    req["n"] = n;
    if (!context.empty()) {
        req["context"] = context;
    }
    return req.dump() + "\n";
}

enum class RerankError { None, Connect, Io, Empty };

// Sends a request to slothingd and returns its "sentences". Publishes the
// connected fd into `fdSlot` so the destructor can shutdown() it to unblock
// this thread at teardown.
std::vector<std::string> sendDaemonRequest(const std::string &req,
                                           std::atomic<int> &fdSlot,
                                           RerankError &err,
                                           json *fullOut = nullptr) {
    err = RerankError::Empty;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        err = RerankError::Connect;
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
        err = RerankError::Connect;
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
            err = RerankError::Io;
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
        err = RerankError::Io;
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
                err = RerankError::None;
            }
            return finish(std::move(sentences));
        }
    } catch (const std::exception &) {
    }
    return finish({});
}

std::vector<std::string> queryDecoder(const std::vector<std::string> &syllables,
                                      int n, const std::string &context,
                                      std::atomic<int> &fdSlot,
                                      RerankError &err,
                                      json *fullOut = nullptr) {
    if (syllables.empty()) {
        err = RerankError::Empty;
        return {};
    }
    return sendDaemonRequest(buildDecodeRequest(syllables, n, context), fdSlot,
                             err, fullOut);
}

// Model-ranked 2-char phrase candidates for positions (at, at+1). Synchronous
// but tiny (one encoder forward, ~2 ms); a dead daemon fails the connect
// immediately, so the UI thread is never held hostage.
// Decode conditioned on user picks: {"hints": {"2": "在"}}. Synchronous but
// ~1 ms; a dead daemon fails the connect instantly.
std::vector<std::string> queryDecoderWithHints(
    const std::vector<std::string> &syllables,
    const std::map<int, std::string> &hints) {
    json req;
    req["syllables"] = syllables;
    req["n"] = 1;
    json h = json::object();
    for (const auto &[i, ch] : hints) {
        h[std::to_string(i)] = ch;
    }
    req["hints"] = h;
    std::atomic<int> fd{-1};
    RerankError err = RerankError::None;
    return sendDaemonRequest(req.dump() + "\n", fd, err);
}

// Fire-and-forget: persist the user's corrections in the daemon's learn
// store (detached thread; a dead daemon just drops it).
void sendLearn(const json &learnBody) {
    std::thread([req = json{{"learn", learnBody}}.dump() + "\n"]() {
        std::atomic<int> fd{-1};
        RerankError err = RerankError::None;
        sendDaemonRequest(req, fd, err);
    }).detach();
}

std::vector<std::pair<double, std::string>>
queryPhrasesScored(const std::vector<std::string> &syllables, int at, int n) {
    json req;
    req["syllables"] = syllables;
    req["phrase_at"] = at;
    req["n"] = n;
    std::atomic<int> fd{-1};
    RerankError err = RerankError::None;
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

} // namespace

// One alternative for the focused segment in segment-conversion.
class SegmentCandidateWord : public CandidateWord {
public:
    SegmentCandidateWord(ChewingEngine *engine, int index, std::string text)
        : CandidateWord(Text(std::move(text))), engine_(engine), index_(index) {
    }
    void select(InputContext *inputContext) const override {
        engine_->pickSegment(inputContext, index_);
    }

private:
    ChewingEngine *engine_;
    int index_;
};

// 微軟新注音-style ` symbol menu: categories -> symbols (same set as the
// web demo's SYMBOLS).
struct SymbolCat {
    const char *name;
    const char *syms;
};
inline const std::vector<SymbolCat> &symbolCats() {
    static const std::vector<SymbolCat> cats = {
        {"常用", "，、。．？！；：…—～‧「」『』（）"},
        {"括號", "（）「」『』〔〕【】《》〈〉｛｝︵︶﹁﹂"},
        {"數學", "＋－×÷＝≠≒±√＜＞≦≧∞∩∪∫∵∴"},
        {"單位", "℃℉°′″＄％＠＃＆＊§￥"},
        {"箭號", "↑↓←→↖↗↙↘"},
        {"圖形", "○●△▲☆★◇◆□■▽▼◎⊙※"},
    };
    return cats;
}
std::vector<std::string> splitUtf8(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t len = 1;
        unsigned char c = s[i];
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

class SymbolCandidateWord : public CandidateWord {
public:
    SymbolCandidateWord(ChewingEngine *engine, std::string sym)
        : CandidateWord(Text(sym)), engine_(engine), sym_(std::move(sym)) {}
    void select(InputContext *inputContext) const override {
        engine_->pickSymbol(inputContext, sym_);
    }

private:
    ChewingEngine *engine_;
    std::string sym_;
};

// A 2-char phrase alternative covering the focused segment and the next one
// (per-phrase Down-rank; one pick fixes a whole word).
class PhraseCandidateWord : public CandidateWord {
public:
    PhraseCandidateWord(ChewingEngine *engine, int start, std::string phrase)
        : CandidateWord(Text(phrase)), engine_(engine), start_(start),
          phrase_(std::move(phrase)) {}
    void select(InputContext *inputContext) const override {
        engine_->pickPhrase(inputContext, start_, phrase_);
    }

private:
    ChewingEngine *engine_;
    int start_;
    std::string phrase_;
};

ChewingEngine::ChewingEngine(Instance *instance) : instance_(instance) {
    dispatcher_.attach(&instance_->eventLoop());
    reloadConfig();
    loadPhoneticTable();
}

ChewingEngine::~ChewingEngine() { stopWorker(); }

namespace {
// Display string for a token list: zh -> bopomofo (or a supplied char),
// en -> the literal with spaces around it. Doubles collapsed at the end.
struct JoinResult {
    std::string text;
    size_t cursorBytes = 0;
};
bool isAsciiRun(const std::string &v);
inline bool isAsciiRunFwd(const std::string &v) { return isAsciiRun(v); }
// Join per-token display strings with the web demo's spacing rules (ASCII
// English runs spaced, fullwidth punctuation hugging, zh plain), inserting
// `tail` at token index `cursorTok` (-1 = end) and reporting the caret's
// byte offset (after the tail).
JoinResult joinDisplay(const std::vector<slothing::SegTok> &toks,
                       const std::vector<std::string> &disp, int cursorTok,
                       const std::string &tail) {
    JoinResult r;
    auto append = [&](const std::string &piece, bool zh) {
        if (piece.empty()) return;
        if (zh) {
            r.text += piece;
        } else if (isAsciiRunFwd(piece)) {
            if (!r.text.empty() && r.text.back() != ' ') r.text += ' ';
            r.text += piece;
            r.text += ' ';
        } else { // fullwidth punctuation
            if (!r.text.empty() && r.text.back() == ' ') r.text.pop_back();
            r.text += piece;
        }
    };
    const int n = static_cast<int>(toks.size());
    const int cur = (cursorTok < 0 || cursorTok > n) ? n : cursorTok;
    for (int i = 0; i <= n; i++) {
        if (i == cur) {
            r.text += tail; // already display-formed (mixed zh/en possible)
            r.cursorBytes = r.text.size();
        }
        if (i == n) break;
        const std::string &d =
            (i < static_cast<int>(disp.size()) && !disp[i].empty()) ? disp[i]
                                                                    : toks[i].v;
        append(d, toks[i].zh);
    }
    if (!r.text.empty() && r.text.back() == ' ') {
        r.text.pop_back();
        if (r.cursorBytes > r.text.size()) r.cursorBytes = r.text.size();
    }
    return r;
}
bool isAsciiRun(const std::string &v) {
    for (unsigned char c : v) {
        if (c >= 0x80) return false;
    }
    return true;
}
std::string toksDisplay(const std::vector<slothing::SegTok> &toks) {
    std::string out;
    for (const auto &t : toks) {
        if (t.zh) {
            out += t.v;
        } else if (isAsciiRun(t.v)) {
            out += " " + t.v + " ";
        } else { // fullwidth punctuation: no spaces, eat one before it
            if (!out.empty() && out.back() == ' ') out.pop_back();
            out += t.v;
        }
    }
    return out;
}
// ASCII -> fullwidth (Ｆｕｌｌ　ｗｉｄｔｈ), UTF-8 encoded.
std::string toFullWidth(char c) {
    unsigned int cp;
    if (c == ' ') {
        cp = 0x3000;
    } else if (c > 32 && c < 127) {
        cp = 0xFF01 + (static_cast<unsigned char>(c) - 33);
    } else {
        return std::string(1, c);
    }
    std::string out;
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
    return out;
}
std::string tidySpaces(std::string s) {
    std::string out;
    for (char c : s) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out += c;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}
} // namespace

void ChewingEngine::commitRun() {
    if (rawKeys_.empty() || !segmenter_) {
        return;
    }
    // forced-English: the run is one literal token, never segmented
    std::vector<slothing::SegTok> toks;
    if (enMode_) {
        toks.push_back({false, rawKeys_});
    } else {
        toks = segmenter_->segment(rawKeys_);
    }
    if (tokCursor_ < 0 ||
        tokCursor_ >= static_cast<int>(committedToks_.size())) {
        for (auto &t : toks) {
            committedToks_.push_back(std::move(t));
        }
        tokCursor_ = -1;
    } else {
        committedToks_.insert(committedToks_.begin() + tokCursor_,
                              toks.begin(), toks.end());
        tokCursor_ += static_cast<int>(toks.size());
    }
    rawKeys_.clear();
}

void ChewingEngine::stopWorker() {
    workerStop_.store(true);
    int fd = inflightFd_.exchange(-1);
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    workerStop_.store(false);
}

void ChewingEngine::reloadConfig() { readAsIni(config_, "conf/chewing.conf"); }

void ChewingEngine::loadPhoneticTable() {
    std::string path;
    if (const char *env = std::getenv("SLOTHING_PHONETIC_TABLE")) {
        path = env;
    } else {
        path = StandardPath::global().locate(StandardPath::Type::Data,
                                             "slothing/phonetic_table.tsv");
    }
    if (path.empty()) {
        SLOTHING_DEBUG() << "phonetic table not found; decode unavailable";
        return;
    }
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        std::string syl = line.substr(0, tab);
        const std::string &rest = line.substr(tab + 1);
        std::vector<std::string> chars;
        for (size_t i = 0; i < rest.size();) {
            unsigned char c = rest[i];
            size_t len = (c & 0xF8) == 0xF0   ? 4
                         : (c & 0xF0) == 0xE0 ? 3
                         : (c & 0xE0) == 0xC0 ? 2
                                              : 1;
            chars.push_back(rest.substr(i, len));
            i += len;
        }
        if (!chars.empty()) {
            phoneticTable_.emplace(std::move(syl), std::move(chars));
        }
    }
    SLOTHING_DEBUG() << "loaded phonetic table: " << phoneticTable_.size()
                    << " syllables";
    // Build the keystream segmenter: valid TONELESS bases from the table.
    std::set<std::string> validBase;
    for (const auto &[syl, chars] : phoneticTable_) {
        std::string base;
        for (size_t i = 0; i < syl.size();) {
            size_t len = 1;
            unsigned char c = syl[i];
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            std::string ch = syl.substr(i, len);
            if (ch != "ˊ" && ch != "ˇ" && ch != "ˋ" && ch != "˙") {
                base += ch;
            }
            i += len;
        }
        if (!base.empty()) {
            validBase.insert(std::move(base));
        }
    }
    segmenter_ = std::make_unique<slothing::Segmenter>(std::move(validBase));
}

void ChewingEngine::reset(const InputMethodEntry &, InputContextEvent &event) {
    // chewing behavior: losing focus / switching IM COMMITS the pending
    // text rather than dropping it (no data loss on stray clicks).
    if (auto *ic = event.inputContext()) {
        std::string pending;
        if (convertState_ == ConvertState::Choosing) {
            pending = composedSentence();
        } else if (!composingEmpty()) {
            commitRun();
            pending = (!livePreedit_.empty() && liveToks_ == committedToks_)
                          ? livePreedit_
                          : tidySpaces(toksDisplay(committedToks_));
        }
        if (!pending.empty()) {
            ic->commitString(pending);
        }
    }
    stopWorker();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_.clear();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    rawKeys_.clear();
    committedToks_.clear();
    tokCursor_ = -1;
    pendingFocus_ = -1;
    livePreedit_.clear();
    liveDisp_.clear();
    liveToks_.clear();
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    segSel_.clear();
    segFocus_ = 0;
    auto *ic = event.inputContext();
    ic->inputPanel().reset();
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::renderComposing(InputContext *ic) {
    ic->inputPanel().reset();
    // Live conversion (微軟新注音 style): decoded Chinese for the finalized
    // tokens + the current run's live segmentation (bopomofo / letters) at
    // the token cursor. Falls back to raw bopomofo when the decode isn't
    // fresh.
    const bool fresh = !liveDisp_.empty() && liveToks_ == committedToks_;
    std::string tail;
    if (!rawKeys_.empty()) {
        tail = enMode_ ? rawKeys_
                       : (segmenter_ ? tidySpaces(toksDisplay(
                                           segmenter_->segment(rawKeys_)))
                                     : rawKeys_);
    }
    // Stale-preserving display (chewing/新注音 never regress to bopomofo):
    // while a new decode is in flight, reuse the previous conversion for
    // tokens unchanged from the start (prefix) or the end (suffix — covers
    // mid-sentence edits); only new tokens show bopomofo until decoded.
    std::vector<std::string> disp;
    if (fresh) {
        disp = liveDisp_;
    } else if (!liveDisp_.empty()) {
        const auto &cur = committedToks_;
        const auto &old = liveToks_;
        disp.assign(cur.size(), std::string());
        size_t pre = 0;
        while (pre < cur.size() && pre < old.size() && cur[pre] == old[pre]) {
            disp[pre] = liveDisp_[pre];
            pre++;
        }
        size_t suf = 0;
        while (suf < cur.size() - pre && suf < old.size() - pre &&
               cur[cur.size() - 1 - suf] == old[old.size() - 1 - suf]) {
            disp[cur.size() - 1 - suf] = liveDisp_[old.size() - 1 - suf];
            suf++;
        }
    }
    JoinResult jr = joinDisplay(committedToks_, disp, tokCursor_, tail);
    const std::string &pre = jr.text;
    if (!pre.empty()) {
        const auto useClient = ic->capabilityFlags().test(CapabilityFlag::Preedit);
        Text preedit(pre, useClient ? TextFormatFlag::Underline
                                    : TextFormatFlag::NoFlag);
        preedit.setCursor(static_cast<int>(jr.cursorBytes));
        if (useClient) {
            ic->inputPanel().setClientPreedit(preedit);
        } else {
            ic->inputPanel().setPreedit(preedit);
        }
    }
    if (!convertNotice_.empty()) {
        ic->inputPanel().setAuxDown(Text(convertNotice_));
    }
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::updateUI(InputContext *ic) {
    if (convertState_ == ConvertState::Choosing) {
        return; // renderSegments owns the panel
    }
    if (convertState_ == ConvertState::Converting) {
        ic->inputPanel().reset();
        const auto useClient = ic->capabilityFlags().test(CapabilityFlag::Preedit);
        Text preedit(tidySpaces(toksDisplay(committedToks_)),
                     useClient ? TextFormatFlag::Underline
                               : TextFormatFlag::NoFlag);
        if (useClient) {
            ic->inputPanel().setClientPreedit(preedit);
        } else {
            ic->inputPanel().setPreedit(preedit);
        }
        std::string aux = "轉換中";
        for (int i = 0; i <= convertTicks_ % 3; i++) {
            aux += "·";
        }
        if (convertTicks_ >= 2) {
            aux += " " + std::to_string(convertTicks_ / 2) + "s";
        }
        ic->inputPanel().setAuxDown(Text(aux));
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
    }
    renderComposing(ic);
}

void ChewingEngine::cancelConversion(InputContext *ic, std::string notice) {
    stopWorker();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_ = std::move(notice);
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    segSel_.clear();
    segFocus_ = 0;
    // keep buffer_ so the user can keep editing / retry
    updateUI(ic);
}

void ChewingEngine::scheduleLiveDecode(InputContext *ic) {
    if (committedToks_.empty() || phoneticTable_.empty() || !segmenter_) {
        livePreedit_.clear();
        liveDisp_.clear();
        liveToks_.clear();
        return;
    }
    bool anyZh = false;
    for (const auto &t : committedToks_) {
        anyZh |= t.zh;
    }
    if (!anyZh) { // pure English so far: literal preedit, no decode needed
        liveDisp_.clear();
        for (const auto &t : committedToks_) {
            liveDisp_.push_back(t.v);
        }
        liveToks_ = committedToks_;
        livePreedit_ =
            joinDisplay(committedToks_, liveDisp_, -1, std::string()).text;
        return;
    }
    stopWorker();
    const uint64_t generation = ++liveGeneration_;
    auto icRef = ic->watch();
    worker_ = std::thread([this, icRef, generation,
                           toks = committedToks_]() {
        // Decode each contiguous zh run separately (the daemon input is
        // syllables-only; en runs are passthrough); one display per token.
        std::vector<std::string> disp;
        size_t i = 0;
        bool allOk = true;
        while (i < toks.size() && !workerStop_.load()) {
            if (!toks[i].zh) {
                disp.push_back(toks[i].v);
                i++;
                continue;
            }
            std::vector<std::string> run;
            const size_t start = i;
            while (i < toks.size() && toks[i].zh) {
                run.push_back(toks[i].v);
                i++;
            }
            RerankError err = RerankError::None;
            auto sentences = queryDecoder(run, 1, "", inflightFd_, err);
            if (!sentences.empty() &&
                utf8::lengthValidated(sentences[0]) == run.size()) {
                const std::string &sent = sentences[0];
                for (size_t k = 0, off = 0; k < run.size(); k++) {
                    size_t len = utf8::ncharByteLength(sent.begin() + off, 1);
                    disp.push_back(sent.substr(off, len));
                    off += len;
                }
            } else { // daemon down / bad reply: bopomofo for this run
                allOk = false;
                for (size_t k = start; k < start + run.size(); k++) {
                    disp.push_back(toks[k].v);
                }
            }
        }
        if (workerStop_.load()) {
            return;
        }
        dispatcher_.schedule([this, icRef, generation, toks = std::move(toks),
                              disp = std::move(disp), allOk]() {
            if (generation != liveGeneration_ ||
                convertState_ != ConvertState::Composing) {
                return;
            }
            auto *ic = icRef.get();
            if (!ic) {
                return;
            }
            if (allOk) {
                liveDisp_ = disp;
                liveToks_ = toks;
                livePreedit_ =
                    joinDisplay(toks, disp, -1, std::string()).text;
            } // partial failure: keep bopomofo fallback (silent)
            renderComposing(ic);
        });
    });
}

void ChewingEngine::startDecode(InputContext *ic) {
    commitRun();
    if (composingEmpty() || phoneticTable_.empty()) {
        convertNotice_ = phoneticTable_.empty() ? "無音表" : "";
        updateUI(ic);
        return;
    }

    // One segment per token. zh -> its legal chars (unknown = typo syllable:
    // single bopomofo-literal candidate, still commit-able); en -> literal.
    std::vector<std::vector<std::string>> positions;
    std::vector<std::pair<int, int>> intervals;
    std::vector<std::string> syllables; // zh syllable per token ("" for en)
    int at = 0;
    for (const auto &t : committedToks_) {
        int span = 1;
        if (t.zh) {
            auto it = phoneticTable_.find(t.v);
            if (it != phoneticTable_.end() && !it->second.empty()) {
                positions.push_back(it->second);
            } else {
                positions.push_back({t.v}); // unknown reading: literal
                span = static_cast<int>(utf8::lengthValidated(t.v));
            }
            syllables.push_back(t.v);
        } else {
            positions.push_back({t.v});
            span = static_cast<int>(utf8::lengthValidated(t.v));
            syllables.push_back("");
        }
        intervals.emplace_back(at, at + span);
        at += span;
    }

    // Document context before the cursor, when the app exposes it.
    std::string context;
    if (ic->capabilityFlags().test(CapabilityFlag::SurroundingText) &&
        ic->surroundingText().isValid()) {
        const auto &st = ic->surroundingText();
        const std::string &t = st.text();
        auto tLen = utf8::lengthValidated(t);
        if (tLen != utf8::INVALID_LENGTH && st.cursor() <= tLen) {
            context = t.substr(0, utf8::ncharByteLength(t.begin(), st.cursor()));
        }
    }

    stopWorker();
    convertState_ = ConvertState::Converting;
    convertBuffer_.clear();
    convertPositions_ = positions;
    convertIntervals_ = intervals;
    convertSyllables_ = syllables;
    convertToks_ = committedToks_;
    const uint64_t generation = ++convertGeneration_;
    const int n = std::max(*config_.LlmCandidateCount, 4);
    auto icRef = ic->watch();
    convertNotice_.clear();
    convertTicks_ = 0;
    convertIc_ = ic->watch();
    updateUI(ic);

    convertTimer_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 500000, 10000,
        [this](EventSourceTime *source, uint64_t) {
            if (convertState_ == ConvertState::Converting) {
                if (auto *timerIc = convertIc_.get()) {
                    convertTicks_++;
                    updateUI(timerIc);
                }
                source->setNextInterval(500000);
                source->setOneShot();
            }
            return true;
        });

    const bool pureZh = std::all_of(
        committedToks_.begin(), committedToks_.end(),
        [this](const slothing::SegTok &t) {
            return t.zh && phoneticTable_.count(t.v);
        });

    worker_ = std::thread([this, icRef, generation, n, pureZh,
                           toks = std::vector<slothing::SegTok>(committedToks_),
                           positions = std::move(positions),
                           context = std::move(context)]() {
        RerankError err = RerankError::None;
        std::vector<std::string> verified;
        std::vector<std::vector<std::string>> ranked;
        if (pureZh) {
            // single request, n-best sentences (the original path); the
            // reply also carries per-position candidates ranked by model
            // score — the candidate UI shows those instead of table order
            std::vector<std::string> syls;
            for (const auto &t : toks) {
                syls.push_back(t.v);
            }
            json full;
            for (auto &sentence :
                 queryDecoder(syls, n, context, inflightFd_, err, &full)) {
                if (!matchesPositions(sentence, positions)) {
                    continue;
                }
                if (std::find(verified.begin(), verified.end(), sentence) ==
                    verified.end()) {
                    verified.push_back(std::move(sentence));
                }
            }
            if (full.contains("candidates")) {
                try {
                    auto r = full["candidates"]
                                 .get<std::vector<std::vector<std::string>>>();
                    if (r.size() == positions.size()) {
                        bool ok = true;
                        for (size_t i = 0; i < r.size(); i++) {
                            if (r[i].size() != positions[i].size()) {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            ranked = std::move(r);
                        }
                    }
                } catch (const std::exception &) {
                }
            }
        } else {
            // mixed zh/en: best decode per zh run, en literal in between
            std::string sentence;
            bool ok = true;
            size_t i = 0;
            while (i < toks.size() && !workerStop_.load()) {
                if (!toks[i].zh) {
                    sentence += toks[i].v;
                    i++;
                    continue;
                }
                std::vector<std::string> run;
                const size_t start = i;
                while (i < toks.size() && toks[i].zh) {
                    run.push_back(toks[i].v);
                    i++;
                }
                auto sentences = queryDecoder(run, 1, "", inflightFd_, err);
                if (!sentences.empty() &&
                    utf8::lengthValidated(sentences[0]) == run.size()) {
                    sentence += sentences[0];
                } else {
                    ok = false;
                    break;
                }
                (void)start;
            }
            if (ok && !sentence.empty()) {
                verified.push_back(std::move(sentence));
            }
        }
        if (workerStop_.load()) {
            return;
        }
        std::string failNotice;
        if (verified.empty()) {
            switch (err) {
            case RerankError::Connect:
                failNotice = "slothingd 未執行";
                break;
            case RerankError::Io:
                failNotice = "slothingd 無回應";
                break;
            default:
                failNotice = "無法解碼";
                break;
            }
        }
        dispatcher_.schedule([this, icRef, generation,
                              verified = std::move(verified),
                              ranked = std::move(ranked),
                              failNotice = std::move(failNotice)]() {
            if (generation != convertGeneration_ ||
                convertState_ != ConvertState::Converting) {
                return;
            }
            auto *ic = icRef.get();
            if (!ic) {
                convertState_ = ConvertState::Composing;
                return;
            }
            if (verified.empty()) {
                cancelConversion(ic, failNotice);
                return;
            }
            if (!ranked.empty()) {
                convertPositions_ = ranked; // model-score candidate order
            }
            convertBuffer_ = verified.front();
            showConversionChoices(ic, verified);
        });
    });
}

void ChewingEngine::showConversionChoices(
    InputContext *ic, const std::vector<std::string> &sentences) {
    convertState_ = ConvertState::Choosing;
    convertTimer_.reset();

    phraseCands_.clear();
    candSpan_ = 2;
    userFixed_.clear();
    segSel_.assign(convertPositions_.size(), 0);
    const std::string &best = sentences.empty() ? convertBuffer_ : sentences[0];
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int from = convertIntervals_[i].first, to = convertIntervals_[i].second;
        std::string span = utf8CharSlice(best, from, to);
        for (size_t j = 0; j < convertPositions_[i].size(); j++) {
            if (convertPositions_[i][j] == span) {
                segSel_[i] = static_cast<int>(j);
                break;
            }
        }
    }
    segFocus_ = 0;
    if (pendingFocus_ >= 0 &&
        pendingFocus_ < static_cast<int>(convertPositions_.size())) {
        segFocus_ = pendingFocus_; // ↓ at the cursor's segment
    } else {
        for (size_t i = 0; i < convertPositions_.size(); i++) {
            if (convertPositions_[i].size() > 1) {
                segFocus_ = static_cast<int>(i);
                break;
            }
        }
    }
    pendingFocus_ = -1;
    initialSel_ = segSel_;
    renderSegments(ic);
}

std::string ChewingEngine::composedSentence() const {
    std::string out;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int sel = (i < segSel_.size()) ? segSel_[i] : 0;
        if (sel >= 0 && sel < static_cast<int>(convertPositions_[i].size())) {
            const bool en = i < convertToks_.size() && !convertToks_[i].zh &&
                            isAsciiRun(convertToks_[i].v);
            if (en) out += " ";
            if (!en && i < convertToks_.size() && !convertToks_[i].zh &&
                !out.empty() && out.back() == ' ') {
                out.pop_back(); // fullwidth punct hugs the previous word
            }
            out += convertPositions_[i][sel];
            if (en) out += " ";
        }
    }
    return tidySpaces(out);
}

void ChewingEngine::renderSegments(InputContext *ic) {
    ic->inputPanel().reset();
    const auto useClient = ic->capabilityFlags().test(CapabilityFlag::Preedit);
    const auto base =
        useClient ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
    Text preedit;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        const std::string &seg = convertPositions_[i][segSel_[i]];
        if (static_cast<int>(i) == segFocus_) {
            preedit.append(seg, {TextFormatFlag::HighLight, base});
        } else {
            preedit.append(seg, base);
        }
    }
    if (useClient) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }

    auto list = std::make_unique<CommonCandidateList>();
    list->setPageSize(*config_.PageSize);
    auto layout = *config_.CandidateLayout;
    list->setLayoutHint(layout == CandidateLayoutHint::NotSet
                            ? CandidateLayoutHint::Horizontal
                            : layout);
    const char *selkeys =
        builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
    KeyList selectionKeys;
    std::vector<std::string> labels;
    for (int i = 0; i < 10 && selkeys[i]; i++) {
        selectionKeys.push_back(Key(std::string(1, selkeys[i])));
        labels.push_back(std::string(1, selkeys[i]) + ".");
    }
    list->setSelectionKey(selectionKeys);
    list->setLabels(labels);
    // One span view at a time (chewing-style): 詞 view shows model-ranked
    // 2-char phrases covering focus..focus+1; 單字 view shows the ranked
    // chars. ↓/↑ cycle the views.
    if (!phraseCands_.count(segFocus_)) {
        // words COVERING the focused char (chewing/新注音 semantics): both
        // the (focus-1, focus) and (focus, focus+1) windows, merged by the
        // model's joint probability.
        std::vector<std::string> syls;
        int focusInRun = -1, runStartTok = 0;
        for (int k = 0; k < static_cast<int>(convertSyllables_.size()); k++) {
            if (convertSyllables_[k].empty()) {
                if (k > segFocus_) break;
                syls.clear();
                runStartTok = k + 1;
                continue;
            }
            if (k == segFocus_) focusInRun = static_cast<int>(syls.size());
            syls.push_back(convertSyllables_[k]);
        }
        std::vector<std::pair<double, std::pair<int, std::string>>> merged;
        if (focusInRun >= 0) {
            for (int w : {focusInRun - 1, focusInRun}) {
                if (w < 0 || w + 1 >= static_cast<int>(syls.size())) {
                    continue;
                }
                for (auto &[p, ph] : queryPhrasesScored(syls, w, 6)) {
                    merged.push_back({p, {runStartTok + w, ph}});
                }
            }
        }
        std::sort(merged.begin(), merged.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        std::vector<std::pair<int, std::string>> out;
        for (auto &[p, sp] : merged) {
            if (out.size() >= 8) break;
            out.push_back(sp);
        }
        phraseCands_[segFocus_] = std::move(out);
    }
    const auto &phrases = phraseCands_[segFocus_];
    const bool wordView = candSpan_ == 2 && !phrases.empty();
    if (wordView) {
        for (const auto &[start, ph] : phrases) {
            list->append(std::make_unique<PhraseCandidateWord>(this, start, ph));
        }
        list->setGlobalCursorIndex(0);
    } else {
        const auto &cands = convertPositions_[segFocus_];
        for (size_t j = 0; j < cands.size(); j++) {
            list->append(std::make_unique<SegmentCandidateWord>(
                this, static_cast<int>(j), cands[j]));
        }
        const int curIdx = segSel_[segFocus_];
        list->setGlobalCursorIndex(curIdx);
        if (*config_.PageSize > 0) {
            list->setPage(curIdx / *config_.PageSize);
        }
    }
    ic->inputPanel().setCandidateList(std::move(list));
    ic->inputPanel().setAuxDown(
        Text(std::string(candSpan_ == 2 ? "【詞】" : "【單字】") +
             "　↓ 換長度　←→ 移動　1-9 選　⏎ 確認　Esc 取消"));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::pickPhrase(InputContext *ic, int start,
                               const std::string &phrase) {
    const int i = start;
    if (i < 0 || i + 1 >= static_cast<int>(convertPositions_.size())) {
        return;
    }
    // split the 2-char phrase into its utf8 chars
    auto c0len = utf8::ncharByteLength(phrase.begin(), 1);
    std::string c0 = phrase.substr(0, c0len);
    std::string c1 = phrase.substr(c0len);
    auto setSel = [this](int pos, const std::string &ch) {
        const auto &cands = convertPositions_[pos];
        for (size_t j = 0; j < cands.size(); j++) {
            if (cands[j] == ch) {
                segSel_[pos] = static_cast<int>(j);
                return;
            }
        }
    };
    setSel(i, c0);
    setSel(i + 1, c1);
    userFixed_.insert(i);
    userFixed_.insert(i + 1);
    rescoreChoosing(ic);
    // advance focus past the phrase, to the next ambiguous segment
    candSpan_ = 2;
    segFocus_ = i + 1;
    for (int k = i + 2; k < static_cast<int>(convertPositions_.size()); k++) {
        if (convertPositions_[k].size() > 1) {
            segFocus_ = k;
            break;
        }
    }
    renderSegments(ic);
}

void ChewingEngine::rescoreChoosing(InputContext *ic) {
    (void)ic;
    // pure-zh sentences only (position mapping is 1:1 with the daemon)
    for (const auto &s : convertSyllables_) {
        if (s.empty()) {
            return;
        }
    }
    if (userFixed_.empty()) {
        return;
    }
    std::map<int, std::string> hints;
    for (int i : userFixed_) {
        if (i >= 0 && i < static_cast<int>(convertPositions_.size())) {
            hints[i] = convertPositions_[i][segSel_[i]];
        }
    }
    auto sentences = queryDecoderWithHints(convertSyllables_, hints);
    if (sentences.empty() ||
        !matchesPositions(sentences[0], convertPositions_)) {
        return; // daemon down / non-hint model: keep current selections
    }
    // adopt the re-scored chars for every segment the user hasn't touched
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        if (userFixed_.count(static_cast<int>(i))) {
            continue;
        }
        std::string span =
            utf8CharSlice(sentences[0], convertIntervals_[i].first,
                          convertIntervals_[i].second);
        for (size_t j = 0; j < convertPositions_[i].size(); j++) {
            if (convertPositions_[i][j] == span) {
                segSel_[i] = static_cast<int>(j);
                break;
            }
        }
    }
}

void ChewingEngine::pickSegment(InputContext *ic, int candIdx) {
    if (segFocus_ < 0 ||
        segFocus_ >= static_cast<int>(convertPositions_.size())) {
        return;
    }
    if (candIdx >= 0 &&
        candIdx < static_cast<int>(convertPositions_[segFocus_].size())) {
        segSel_[segFocus_] = candIdx;
        userFixed_.insert(segFocus_);
        rescoreChoosing(ic); // 新注音-style: the pick re-scores the rest
    }
    candSpan_ = 2;
    for (int i = segFocus_ + 1; i < static_cast<int>(convertPositions_.size());
         i++) {
        if (convertPositions_[i].size() > 1) {
            segFocus_ = i;
            break;
        }
    }
    renderSegments(ic);
}

void ChewingEngine::acceptConversion(InputContext *ic,
                                     const std::string &sentence) {
    ic->commitString(sentence);
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    liveGeneration_++;
    livePreedit_.clear();
    liveDisp_.clear();
    liveToks_.clear();
    rawKeys_.clear();
    committedToks_.clear();
    tokCursor_ = -1;
    pendingFocus_ = -1;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    segSel_.clear();
    segFocus_ = 0;
    updateUI(ic);
}

void ChewingEngine::renderSymbols(InputContext *ic) {
    ic->inputPanel().reset();
    const auto &cats = symbolCats();
    std::string aux;
    for (size_t i = 0; i < cats.size(); i++) {
        if (static_cast<int>(i) == symCat_) {
            aux += "【" + std::string(cats[i].name) + "】";
        } else {
            aux += " " + std::string(cats[i].name) + " ";
        }
    }
    ic->inputPanel().setAuxUp(Text(aux));
    auto list = std::make_unique<CommonCandidateList>();
    list->setPageSize(*config_.PageSize);
    list->setLayoutHint(CandidateLayoutHint::Horizontal);
    const char *selkeys =
        builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
    KeyList selectionKeys;
    std::vector<std::string> labels;
    for (int i = 0; i < 10 && selkeys[i]; i++) {
        selectionKeys.push_back(Key(std::string(1, selkeys[i])));
        labels.push_back(std::string(1, selkeys[i]) + ".");
    }
    list->setSelectionKey(selectionKeys);
    list->setLabels(labels);
    for (auto &sym : splitUtf8(cats[symCat_].syms)) {
        list->append(std::make_unique<SymbolCandidateWord>(this, sym));
    }
    ic->inputPanel().setCandidateList(std::move(list));
    ic->inputPanel().setAuxDown(Text("←→ 分類　1-9 選取　PgUp/PgDn 翻頁　Esc/` 關閉"));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::pickSymbol(InputContext *ic, const std::string &sym) {
    symbolMode_ = false;
    if (composingEmpty() && convertState_ == ConvertState::Composing) {
        ic->commitString(sym);
        ic->inputPanel().reset();
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
    }
    commitRun();
    const int n = static_cast<int>(committedToks_.size());
    const int cur = (tokCursor_ < 0 || tokCursor_ > n) ? n : tokCursor_;
    committedToks_.insert(committedToks_.begin() + cur, {false, sym});
    if (tokCursor_ >= 0) {
        tokCursor_ = cur + 1;
    }
    scheduleLiveDecode(ic);
    renderComposing(ic);
}

void ChewingEngine::keyEvent(const InputMethodEntry &, KeyEvent &keyEvent) {
    // Lone-Shift release toggles forced-English mode (微軟/web-demo idiom):
    // press Shift, release without another key -> 中/英 switch.
    if (keyEvent.isRelease()) {
        if ((keyEvent.key().sym() == FcitxKey_Shift_L ||
             keyEvent.key().sym() == FcitxKey_Shift_R) &&
            shiftAlone_) {
            shiftAlone_ = false;
            enMode_ = !enMode_;
            if (auto *ic = keyEvent.inputContext()) {
                convertNotice_ = enMode_ ? "英文模式（Shift 切回）" : "";
                if (convertState_ == ConvertState::Composing) {
                    renderComposing(ic);
                }
            }
            keyEvent.filterAndAccept();
        }
        return;
    }
    if (keyEvent.key().sym() == FcitxKey_Shift_L ||
        keyEvent.key().sym() == FcitxKey_Shift_R) {
        shiftAlone_ = true;
        return;
    }
    shiftAlone_ = false;
    auto *ic = keyEvent.inputContext();

    // -- Choosing: modal segment editing -----------------------------------
    if (convertState_ == ConvertState::Choosing) {
        keyEvent.filterAndAccept();
        if (segFocus_ < 0 || convertPositions_.empty()) {
            cancelConversion(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Escape)) {
            cancelConversion(ic);
            renderComposing(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Return)) {
            // learn the user's corrections (changed zh segments + adjacent
            // changed pairs as phrases) before committing
            json chars = json::array(), phrases = json::array();
            for (int i : userFixed_) {
                if (i < 0 || i >= static_cast<int>(segSel_.size()) ||
                    i >= static_cast<int>(convertSyllables_.size()) ||
                    convertSyllables_[i].empty()) {
                    continue;
                }
                chars.push_back({convertSyllables_[i],
                                 convertPositions_[i][segSel_[i]]});
                if (userFixed_.count(i + 1) &&
                    i + 1 < static_cast<int>(segSel_.size()) &&
                    i + 1 < static_cast<int>(convertSyllables_.size()) &&
                    !convertSyllables_[i + 1].empty()) {
                    phrases.push_back(
                        {convertSyllables_[i] + " " + convertSyllables_[i + 1],
                         convertPositions_[i][segSel_[i]] +
                             convertPositions_[i + 1][segSel_[i + 1]]});
                }
            }
            if (!chars.empty() || !phrases.empty()) {
                sendLearn({{"chars", chars}, {"phrases", phrases}});
            }
            acceptConversion(ic, composedSentence());
            return;
        }
        const int nseg = static_cast<int>(convertPositions_.size());
        auto &foc = segSel_[segFocus_];
        const int ncand = static_cast<int>(convertPositions_[segFocus_].size());
        if (keyEvent.key().check(FcitxKey_Right)) {
            for (int i = segFocus_ + 1; i < nseg; i++) {
                if (convertPositions_[i].size() > 1) {
                    segFocus_ = i;
                    candSpan_ = 2;
                    break;
                }
            }
        } else if (keyEvent.key().check(FcitxKey_Left)) {
            for (int i = segFocus_ - 1; i >= 0; i--) {
                if (convertPositions_[i].size() > 1) {
                    segFocus_ = i;
                    candSpan_ = 2;
                    break;
                }
            }
        } else if (keyEvent.key().check(FcitxKey_Next) ||
                   keyEvent.key().check(FcitxKey_Prior)) {
            // page the candidate list (chewing's PgUp/PgDn)
            if (auto list = ic->inputPanel().candidateList()) {
                if (auto *pageable = list->toPageable()) {
                    if (keyEvent.key().check(FcitxKey_Next)) {
                        if (pageable->hasNext()) pageable->next();
                    } else if (pageable->hasPrev()) {
                        pageable->prev();
                    }
                    ic->updateUserInterface(
                        UserInterfaceComponent::InputPanel);
                }
            }
            return;
        } else if (keyEvent.key().check(FcitxKey_Down) ||
                   keyEvent.key().check(FcitxKey_Up) ||
                   (keyEvent.key().check(FcitxKey_space) &&
                    *config_.SpaceAsSelection)) {
            // chewing-style span cycling: 詞 view <-> 單字 view
            candSpan_ = (candSpan_ == 2) ? 1 : 2;
        } else if (keyEvent.key().isSimple()) {
            const char *selkeys =
                builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
            char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            // number keys select from the VISIBLE list (phrase candidates
            // are prepended before the per-char ones), so route through it
            const auto list = ic->inputPanel().candidateList();
            const int nvis = list ? list->size() : ncand;
            for (int i = 0; i < 10 && selkeys[i]; i++) {
                if (selkeys[i] == sym && i < nvis) {
                    if (list) {
                        list->candidate(i).select(ic);
                    } else {
                        pickSegment(ic, i);
                    }
                    return;
                }
            }
        }
        renderSegments(ic);
        return;
    }

    // -- Converting: only Esc acts ------------------------------------------
    if (convertState_ == ConvertState::Converting) {
        if (keyEvent.key().check(FcitxKey_Escape)) {
            keyEvent.filterAndAccept();
            cancelConversion(ic);
            renderComposing(ic);
        } else {
            keyEvent.filterAndAccept(); // swallow while decoding
        }
        return;
    }

    // -- Symbol menu ---------------------------------------------------------
    if (symbolMode_) {
        keyEvent.filterAndAccept();
        if (keyEvent.key().check(FcitxKey_Escape) ||
            keyEvent.key().check(FcitxKey_grave)) {
            symbolMode_ = false;
            if (convertState_ == ConvertState::Choosing) {
                renderSegments(ic);
            } else {
                renderComposing(ic);
            }
            return;
        }
        const int ncat = static_cast<int>(symbolCats().size());
        if (keyEvent.key().check(FcitxKey_Left)) {
            symCat_ = (symCat_ - 1 + ncat) % ncat;
            renderSymbols(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Right)) {
            symCat_ = (symCat_ + 1) % ncat;
            renderSymbols(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Next) ||
            keyEvent.key().check(FcitxKey_Prior)) {
            if (auto list = ic->inputPanel().candidateList()) {
                if (auto *pageable = list->toPageable()) {
                    if (keyEvent.key().check(FcitxKey_Next)) {
                        if (pageable->hasNext()) pageable->next();
                    } else if (pageable->hasPrev()) {
                        pageable->prev();
                    }
                    ic->updateUserInterface(
                        UserInterfaceComponent::InputPanel);
                }
            }
            return;
        }
        if (keyEvent.key().isSimple()) {
            const char *selkeys =
                builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
            char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            const auto list = ic->inputPanel().candidateList();
            for (int i = 0; i < 10 && selkeys[i]; i++) {
                if (selkeys[i] == sym && list &&
                    i < list->size()) {
                    list->candidate(i).select(ic);
                    return;
                }
            }
        }
        return;
    }

    // -- Composing ----------------------------------------------------------
    convertNotice_.clear();

    // The convert key force-decodes into the segment UI (kept for muscle
    // memory; live conversion normally makes it unnecessary).
    if (keyEvent.key().normalize().checkKeyList(*config_.ConvertKey)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            startDecode(ic);
            return;
        }
        return; // let Ctrl+Return etc. reach the app
    }

    // ←/→ move the insertion cursor over the composed tokens (chewing-style
    // preedit editing); Home/End jump. The current run is finalized first.
    if (keyEvent.key().check(FcitxKey_Left) ||
        keyEvent.key().check(FcitxKey_Right) ||
        keyEvent.key().check(FcitxKey_Home) ||
        keyEvent.key().check(FcitxKey_End)) {
        if (composingEmpty()) {
            return; // nothing composed: let the app have the key
        }
        keyEvent.filterAndAccept();
        commitRun();
        const int n = static_cast<int>(committedToks_.size());
        int cur = tokCursor_ < 0 ? n : tokCursor_;
        if (keyEvent.key().check(FcitxKey_Left)) {
            cur = std::max(0, cur - 1);
        } else if (keyEvent.key().check(FcitxKey_Right)) {
            cur = std::min(n, cur + 1);
        } else if (keyEvent.key().check(FcitxKey_Home)) {
            cur = 0;
        } else {
            cur = n;
        }
        tokCursor_ = (cur >= n) ? -1 : cur;
        scheduleLiveDecode(ic);
        renderComposing(ic);
        return;
    }

    // ↓ while composing: per-character selection at the cursor's segment
    // (微軟新注音 idiom).
    if (keyEvent.key().check(FcitxKey_Down)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            commitRun();
            // candidates open for the char BEFORE the cursor (新注音); at
            // the end of the buffer that's the last character.
            const int n = static_cast<int>(committedToks_.size());
            pendingFocus_ =
                tokCursor_ < 0 ? n - 1
                               : std::max(0, std::min(tokCursor_ - 1, n - 1));
            startDecode(ic);
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_Escape)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            rawKeys_.clear();
            committedToks_.clear();
            tokCursor_ = -1;
            livePreedit_.clear();
            liveDisp_.clear();
            liveToks_.clear();
            renderComposing(ic);
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_BackSpace)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            if (!rawKeys_.empty()) {
                // pop one UTF-8 character (enMode may hold fullwidth chars)
                while (!rawKeys_.empty() &&
                       (static_cast<unsigned char>(rawKeys_.back()) & 0xC0) ==
                           0x80) {
                    rawKeys_.pop_back(); // continuation bytes
                }
                if (!rawKeys_.empty()) {
                    rawKeys_.pop_back(); // lead / ASCII byte
                }
            } else {
                const int n = static_cast<int>(committedToks_.size());
                int cur = tokCursor_ < 0 ? n : tokCursor_;
                if (cur > 0) {
                    committedToks_.erase(committedToks_.begin() + (cur - 1));
                    if (tokCursor_ >= 0) {
                        tokCursor_ = cur - 1;
                    }
                }
                scheduleLiveDecode(ic);
            }
            renderComposing(ic);
        }
        return;
    }

    // Enter: commit. If the live decode is fresh, commit instantly;
    // otherwise decode-then-commit.
    if (keyEvent.key().check(FcitxKey_Return)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            commitRun();
            if (!livePreedit_.empty() && liveToks_ == committedToks_) {
                acceptConversion(ic, livePreedit_);
                return;
            }
            startDecode(ic);
        }
        return;
    }

    if (keyEvent.key().isSimple()) {
        char c = static_cast<char>(keyEvent.key().sym() & 0xff);

        // Shift+Space: 全形/半形 (微軟/chewing convention)
        if (c == ' ' && keyEvent.key().states().test(KeyState::Shift)) {
            keyEvent.filterAndAccept();
            fullWidth_ = !fullWidth_;
            convertNotice_ = fullWidth_ ? "全形" : "半形";
            if (convertState_ == ConvertState::Composing) {
                renderComposing(ic);
            }
            return;
        }

        // Forced-English mode (lone Shift): literal input, no zhuyin
        // parsing / segmentation; space commits the word. Case preserved,
        // fullwidth honoured.
        if (enMode_) {
            keyEvent.filterAndAccept();
            if (c == ' ') {
                if (!rawKeys_.empty()) {
                    committedToks_.push_back({false, rawKeys_});
                    rawKeys_.clear();
                    scheduleLiveDecode(ic);
                } else {
                    committedToks_.push_back(
                        {false, fullWidth_ ? toFullWidth(' ')
                                           : std::string(" ")});
                    scheduleLiveDecode(ic);
                }
            } else if (c >= 33 && c < 127) {
                rawKeys_ += fullWidth_ ? toFullWidth(c) : std::string(1, c);
            }
            renderComposing(ic);
            return;
        }
        // Capitals are unambiguous ENGLISH evidence: they have no zhuyin
        // mapping (dachen keys are lowercase), so they stay verbatim in the
        // raw run and the segmenter routes them (and usually the letters
        // around them) into an English token. Case is preserved (Claude).

        // Space finalizes the current run (also serves as tone 1 — the
        // segmenter already treats a bare valid base as a tone-1 syllable).
        if (c == ' ') {
            if (!rawKeys_.empty()) {
                keyEvent.filterAndAccept();
                commitRun();
                scheduleLiveDecode(ic);
                renderComposing(ic);
            } else if (!committedToks_.empty()) {
                keyEvent.filterAndAccept(); // swallow: don't type into app
            }
            return;
        }

        // A tone key completes the last syllable -> finalize + live decode.
        if (slothing::segToneMark(c) && !rawKeys_.empty()) {
            keyEvent.filterAndAccept();
            rawKeys_ += c;
            commitRun();
            scheduleLiveDecode(ic);
            renderComposing(ic);
            return;
        }

        // ` opens the categorized symbol menu (微軟/自然 convention)
        if (c == '`') {
            keyEvent.filterAndAccept();
            symbolMode_ = true;
            renderSymbols(ic);
            return;
        }

        // Punctuation, 微軟新注音/chewing conventions: Shift+,.…/;1 -> full-
        // width marks, \\ -> 、. Appended as a literal token (commits with
        // the sentence); committed directly when nothing is being composed.
        static const std::unordered_map<char, const char *> kPunct = {
            {'<', "，"}, {'>', "。"}, {'?', "？"}, {'!', "！"},
            {':', "："}, {'"', "；"}, {'(', "（"}, {')', "）"},
            {'\\', "、"},
        };
        const char rawSym = static_cast<char>(keyEvent.key().sym() & 0xff);
        if (auto pit = kPunct.find(rawSym); pit != kPunct.end()) {
            keyEvent.filterAndAccept();
            if (composingEmpty()) {
                ic->commitString(pit->second);
                return;
            }
            commitRun();
            committedToks_.push_back({false, pit->second});
            scheduleLiveDecode(ic);
            renderComposing(ic);
            return;
        }

        // Any zhuyin-mappable or alphanumeric key extends the raw run; the
        // segmenter re-decides zh/en live (auto code-switch, no mode key).
        const bool feeds = slothing::dachenMap().count(c) ||
                           (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9');
        if (feeds) {
            keyEvent.filterAndAccept();
            rawKeys_ += c;
            renderComposing(ic);
            return;
        }
    }

    // Anything else: if we're composing, swallow printable noise so stray
    // keys don't corrupt the run; let real control keys pass.
    if (!composingEmpty()) {
        if (keyEvent.key().isSimple()) {
            keyEvent.filterAndAccept();
        }
    }
}

} // namespace fcitx

FCITX_ADDON_FACTORY(fcitx::ChewingEngineFactory);
