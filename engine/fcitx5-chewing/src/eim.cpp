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
                                           RerankError &err) {
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
                                      RerankError &err) {
    if (syllables.empty()) {
        err = RerankError::Empty;
        return {};
    }
    return sendDaemonRequest(buildDecodeRequest(syllables, n, context), fdSlot,
                             err);
}

// Model-ranked 2-char phrase candidates for positions (at, at+1). Synchronous
// but tiny (one encoder forward, ~2 ms); a dead daemon fails the connect
// immediately, so the UI thread is never held hostage.
std::vector<std::string> queryPhrases(const std::vector<std::string> &syllables,
                                      int at, int n) {
    json req;
    req["syllables"] = syllables;
    req["phrase_at"] = at;
    req["n"] = n;
    std::atomic<int> fd{-1};
    RerankError err = RerankError::None;
    return sendDaemonRequest(req.dump() + "\n", fd, err);
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

// A 2-char phrase alternative covering the focused segment and the next one
// (per-phrase Down-rank; one pick fixes a whole word).
class PhraseCandidateWord : public CandidateWord {
public:
    PhraseCandidateWord(ChewingEngine *engine, std::string phrase)
        : CandidateWord(Text(phrase + "（詞）")), engine_(engine),
          phrase_(std::move(phrase)) {}
    void select(InputContext *inputContext) const override {
        engine_->pickPhrase(inputContext, phrase_);
    }

private:
    ChewingEngine *engine_;
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
std::string toksDisplay(const std::vector<slothing::SegTok> &toks) {
    std::string out;
    for (const auto &t : toks) {
        if (t.zh) {
            out += t.v;
        } else {
            out += " " + t.v + " ";
        }
    }
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
    for (auto &t : segmenter_->segment(rawKeys_)) {
        committedToks_.push_back(std::move(t));
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
    stopWorker();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_.clear();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    rawKeys_.clear();
    committedToks_.clear();
    livePreedit_.clear();
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
    // the tail. Falls back to raw bopomofo when the decode isn't fresh.
    std::string pre;
    if (!livePreedit_.empty() && liveToks_ == committedToks_) {
        pre = livePreedit_;
    } else {
        pre = toksDisplay(committedToks_);
    }
    if (!rawKeys_.empty() && segmenter_) {
        pre += toksDisplay(segmenter_->segment(rawKeys_));
    }
    pre = tidySpaces(pre);
    if (!pre.empty()) {
        const auto useClient = ic->capabilityFlags().test(CapabilityFlag::Preedit);
        Text preedit(pre, useClient ? TextFormatFlag::Underline
                                    : TextFormatFlag::NoFlag);
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
        liveToks_.clear();
        return;
    }
    bool anyZh = false;
    for (const auto &t : committedToks_) {
        anyZh |= t.zh;
    }
    if (!anyZh) { // pure English so far: literal preedit, no decode needed
        livePreedit_ = tidySpaces(toksDisplay(committedToks_));
        liveToks_ = committedToks_;
        return;
    }
    stopWorker();
    const uint64_t generation = ++liveGeneration_;
    auto icRef = ic->watch();
    worker_ = std::thread([this, icRef, generation,
                           toks = committedToks_]() {
        // Decode each contiguous zh run separately (the daemon input is
        // syllables-only; en runs are passthrough) and stitch the preedit.
        std::string pre;
        size_t i = 0;
        bool allOk = true;
        while (i < toks.size() && !workerStop_.load()) {
            if (!toks[i].zh) {
                pre += " " + toks[i].v + " ";
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
                pre += sentences[0];
            } else { // daemon down / bad reply: bopomofo for this run
                allOk = false;
                for (size_t k = start; k < start + run.size(); k++) {
                    pre += toks[k].v;
                }
            }
        }
        if (workerStop_.load()) {
            return;
        }
        dispatcher_.schedule([this, icRef, generation, toks = std::move(toks),
                              pre = tidySpaces(pre), allOk]() {
            if (generation != liveGeneration_ ||
                convertState_ != ConvertState::Composing) {
                return;
            }
            auto *ic = icRef.get();
            if (!ic) {
                return;
            }
            if (allOk) {
                livePreedit_ = pre;
                liveToks_ = toks;
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
        if (pureZh) {
            // single request, n-best sentences (the original path)
            std::vector<std::string> syls;
            for (const auto &t : toks) {
                syls.push_back(t.v);
            }
            for (auto &sentence :
                 queryDecoder(syls, n, context, inflightFd_, err)) {
                if (!matchesPositions(sentence, positions)) {
                    continue;
                }
                if (std::find(verified.begin(), verified.end(), sentence) ==
                    verified.end()) {
                    verified.push_back(std::move(sentence));
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
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        if (convertPositions_[i].size() > 1) {
            segFocus_ = static_cast<int>(i);
            break;
        }
    }
    renderSegments(ic);
}

std::string ChewingEngine::composedSentence() const {
    std::string out;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int sel = (i < segSel_.size()) ? segSel_[i] : 0;
        if (sel >= 0 && sel < static_cast<int>(convertPositions_[i].size())) {
            const bool en = i < convertToks_.size() && !convertToks_[i].zh;
            if (en) out += " ";
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
    // Per-phrase Down-rank: model-ranked 2-char phrases covering this and
    // the next segment, fetched lazily once per focus position.
    const bool zhPair =
        segFocus_ + 1 < static_cast<int>(convertSyllables_.size()) &&
        !convertSyllables_[segFocus_].empty() &&
        !convertSyllables_[segFocus_ + 1].empty();
    if (zhPair && !phraseCands_.count(segFocus_)) {
        // phrase scoring wants a pure-syllable context; use the zh run
        // around the focus (en tokens have no syllable)
        std::vector<std::string> syls;
        int focusInRun = -1;
        for (int k = 0; k < static_cast<int>(convertSyllables_.size()); k++) {
            if (convertSyllables_[k].empty()) {
                if (k > segFocus_) break;
                syls.clear();
                continue;
            }
            if (k == segFocus_) focusInRun = static_cast<int>(syls.size());
            syls.push_back(convertSyllables_[k]);
        }
        phraseCands_[segFocus_] =
            focusInRun >= 0 ? queryPhrases(syls, focusInRun, 5)
                            : std::vector<std::string>();
    }
    int nPhrase = 0;
    if (auto it = phraseCands_.find(segFocus_); it != phraseCands_.end()) {
        for (const auto &ph : it->second) {
            list->append(std::make_unique<PhraseCandidateWord>(this, ph));
            nPhrase++;
        }
    }
    const auto &cands = convertPositions_[segFocus_];
    for (size_t j = 0; j < cands.size(); j++) {
        list->append(std::make_unique<SegmentCandidateWord>(
            this, static_cast<int>(j), cands[j]));
    }
    list->setGlobalCursorIndex(nPhrase + segSel_[segFocus_]);
    ic->inputPanel().setCandidateList(std::move(list));
    ic->inputPanel().setAuxDown(Text("←→ 選詞　↑↓ 換字　⏎ 確認　Esc 取消"));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::pickPhrase(InputContext *ic, const std::string &phrase) {
    const int i = segFocus_;
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
    // advance focus past the phrase, to the next ambiguous segment
    segFocus_ = i + 1;
    for (int k = i + 2; k < static_cast<int>(convertPositions_.size()); k++) {
        if (convertPositions_[k].size() > 1) {
            segFocus_ = k;
            break;
        }
    }
    renderSegments(ic);
}

void ChewingEngine::pickSegment(InputContext *ic, int candIdx) {
    if (segFocus_ < 0 ||
        segFocus_ >= static_cast<int>(convertPositions_.size())) {
        return;
    }
    if (candIdx >= 0 &&
        candIdx < static_cast<int>(convertPositions_[segFocus_].size())) {
        segSel_[segFocus_] = candIdx;
    }
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
    liveToks_.clear();
    rawKeys_.clear();
    committedToks_.clear();
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    segSel_.clear();
    segFocus_ = 0;
    updateUI(ic);
}

void ChewingEngine::keyEvent(const InputMethodEntry &, KeyEvent &keyEvent) {
    if (keyEvent.isRelease()) {
        return;
    }
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
                    break;
                }
            }
        } else if (keyEvent.key().check(FcitxKey_Left)) {
            for (int i = segFocus_ - 1; i >= 0; i--) {
                if (convertPositions_[i].size() > 1) {
                    segFocus_ = i;
                    break;
                }
            }
        } else if (keyEvent.key().check(FcitxKey_Down) ||
                   (keyEvent.key().check(FcitxKey_space) &&
                    *config_.SpaceAsSelection)) {
            foc = (foc + 1) % ncand;
        } else if (keyEvent.key().check(FcitxKey_Up)) {
            foc = (foc - 1 + ncand) % ncand;
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

    // ↓ while composing: per-character selection (微軟新注音 idiom).
    if (keyEvent.key().check(FcitxKey_Down)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            startDecode(ic);
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_Escape)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            rawKeys_.clear();
            committedToks_.clear();
            livePreedit_.clear();
            liveToks_.clear();
            renderComposing(ic);
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_BackSpace)) {
        if (!composingEmpty()) {
            keyEvent.filterAndAccept();
            if (!rawKeys_.empty()) {
                rawKeys_.pop_back();
            } else {
                committedToks_.pop_back();
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
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }

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

        // Any zhuyin-mappable or alphanumeric key extends the raw run; the
        // segmenter re-decides zh/en live (auto code-switch, no mode key).
        const bool feeds = slothing::dachenMap().count(c) ||
                           (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
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
