/*
 * SPDX-FileCopyrightText: 2010~2017 CSSlayer <wengxt@gmail.com>
 * SPDX-FileCopyrightText: 2026 sloth-zhuyin-linux
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
// Slothing engine, fcitx5 adapter: keystrokes are segmented by the shared DP
// segmenter (engine/common), the typed syllables are decoded to Traditional
// Chinese by the local SlothLM model via slothingd's decode mode, and the
// shared state machine (engine/common/core.h) drives the chewing-parity
// interaction. This file only decodes fcitx keys, runs the async decode
// workers, and paints. No libchewing.
#include "eim.h"
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

using json = nlohmann::json;
using slothing::DaemonError;
using slothing::isAsciiRun;
using slothing::joinDisplay;
using slothing::punctMap;
using slothing::splitUtf8;
using slothing::symbolCats;
using slothing::tidySpaces;
using slothing::toFullWidth;
using slothing::toksDisplay;

namespace fcitx {

FCITX_DEFINE_LOG_CATEGORY(slothing_log, "chewing");
#define SLOTHING_DEBUG() FCITX_LOGC(slothing_log, Debug)

namespace {

// Digit/label sets for the selection keys (index by ChewingSelectionKey).
const char *const builtin_selectkeys[] = {
    "1234567890", "asdfghjkl;", "asdfzxcv89",
    "asdfjkl789", "aoeuhtn789", "1234qweras", "dstnaeo789",
};

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
    loadAssoc();
}

ChewingEngine::~ChewingEngine() { stopWorker(); }

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
        std::vector<std::string> chars = splitUtf8(rest);
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
        for (const auto &ch : splitUtf8(syl)) {
            if (ch != "ˊ" && ch != "ˇ" && ch != "ˋ" && ch != "˙") {
                base += ch;
            }
        }
        if (!base.empty()) {
            validBase.insert(std::move(base));
        }
    }
    segmenter_ = std::make_unique<slothing::Segmenter>(std::move(validBase));
}

void ChewingEngine::loadAssoc() {
    std::string dictTsv;
    std::string path;
    if (const char *env = std::getenv("SLOTHING_ASSOC_TABLE")) {
        path = env;
    } else {
        path = StandardPath::global().locate(StandardPath::Type::Data,
                                             "slothing/assoc_tc.tsv");
    }
    if (!path.empty()) {
        std::ifstream f(path);
        dictTsv.assign(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
    }
    const char *xdg = std::getenv("XDG_DATA_HOME");
    const char *home = std::getenv("HOME");
    std::string dir = (xdg ? std::string(xdg)
                           : std::string(home ? home : ".") + "/.local/share") +
                      "/slothing";
    // same per-user store the daemon's learn.tsv lives beside
    assoc_.load(dictTsv, dir + "/assoc_user.tsv");
}

// Every commit feeds 聯想 (personal bigrams + the prediction tail).
void ChewingEngine::commitAndRecord(InputContext *ic, const std::string &s) {
    if (s.empty()) {
        return;
    }
    ic->commitString(s);
    assoc_.record(s);
}

// 聯想 aux row after a commit: 聯: ⇧1 腦 ⇧2 子 … — ⇧1-9 selects (digits stay
// typeable, 微軟 convention); any other key dismisses (handled in keyEvent).
void ChewingEngine::renderPredictions(InputContext *ic) {
    predicting_ = false;
    if (!*config_.Association) {
        return;
    }
    auto preds = assoc_.predictions();
    if (preds.empty()) {
        return;
    }
    ic->inputPanel().reset();
    std::string aux = "聯:";
    for (size_t i = 0; i < preds.size() && i < 9; i++) {
        aux += " ⇧" + std::to_string(i + 1) + " " + preds[i];
    }
    ic->inputPanel().setAuxUp(Text(aux));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    predicting_ = true;
}

std::string ChewingEngine::surroundingContext(InputContext *ic) const {
    std::string ctx;
    if (ic->capabilityFlags().test(CapabilityFlag::SurroundingText) &&
        ic->surroundingText().isValid()) {
        const auto &st = ic->surroundingText();
        const std::string &t = st.text();
        auto tLen = utf8::lengthValidated(t);
        if (tLen != utf8::INVALID_LENGTH && st.cursor() <= tLen) {
            ctx = t.substr(0, utf8::ncharByteLength(t.begin(), st.cursor()));
        }
    }
    return ctx;
}

void ChewingEngine::reset(const InputMethodEntry &, InputContextEvent &event) {
    // chewing behavior: losing focus / switching IM COMMITS the pending
    // text rather than dropping it (no data loss on stray clicks).
    if (auto *ic = event.inputContext()) {
        std::string pending;
        if (convertState_ == ConvertState::Choosing) {
            pending = choosing_.composedSentence();
        } else if (!comp_.empty()) {
            comp_.commitRun(segmenter_.get(), enMode_);
            pending = (!livePreedit_.empty() && liveToks_ == comp_.toks)
                          ? livePreedit_
                          : tidySpaces(toksDisplay(comp_.toks));
        }
        if (!pending.empty()) {
            commitAndRecord(ic, pending);
        }
    }
    assoc_.clearTail(); // focus change: stale predictions must not carry over
    predicting_ = false;
    stopWorker();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_.clear();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    comp_.clear();
    choosing_.clear();
    pendingFocus_ = -1;
    livePreedit_.clear();
    liveDisp_.clear();
    liveToks_.clear();
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    convertToks_.clear();
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
    // fresh; the shared stale-preserving rule keeps unchanged tokens
    // converted (never regress to bopomofo).
    std::string tail;
    if (!comp_.rawKeys.empty()) {
        tail = enMode_ ? comp_.rawKeys
                       : (segmenter_ ? tidySpaces(toksDisplay(
                                           segmenter_->segment(comp_.rawKeys)))
                                     : comp_.rawKeys);
    }
    std::vector<std::string> disp =
        slothing::staleDisplay(comp_.toks, liveToks_, liveDisp_);
    slothing::JoinResult jr =
        joinDisplay(comp_.toks, disp, comp_.tokCursor, tail);
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
        Text preedit(tidySpaces(toksDisplay(comp_.toks)),
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
    choosing_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    convertToks_.clear();
    // keep the composing buffer so the user can keep editing / retry
    updateUI(ic);
}

void ChewingEngine::scheduleLiveDecode(InputContext *ic) {
    if (comp_.toks.empty() || phoneticTable_.empty() || !segmenter_) {
        livePreedit_.clear();
        liveDisp_.clear();
        liveToks_.clear();
        return;
    }
    bool anyZh = false;
    for (const auto &t : comp_.toks) {
        anyZh |= t.zh;
    }
    if (!anyZh) { // pure English so far: literal preedit, no decode needed
        liveDisp_.clear();
        for (const auto &t : comp_.toks) {
            liveDisp_.push_back(t.v);
        }
        liveToks_ = comp_.toks;
        livePreedit_ =
            joinDisplay(comp_.toks, liveDisp_, -1, std::string()).text;
        return;
    }
    std::string ctx = surroundingContext(ic);
    stopWorker();
    const uint64_t generation = ++liveGeneration_;
    auto icRef = ic->watch();
    worker_ = std::thread([this, icRef, generation, ctx = std::move(ctx),
                           toks = comp_.toks]() {
        // Decode each contiguous zh run separately (the daemon input is
        // syllables-only; en runs are passthrough); one display per token.
        std::vector<std::string> disp;
        std::string runCtx = ctx; // grows with decoded runs
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
            DaemonError err = DaemonError::None;
            auto sentences =
                slothing::queryDecoder(run, 1, runCtx, inflightFd_, err);
            if (!sentences.empty() &&
                utf8::lengthValidated(sentences[0]) == run.size()) {
                const std::string &sent = sentences[0];
                runCtx += sent;
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

void ChewingEngine::startDecode(InputContext *ic, bool commitDirect) {
    comp_.commitRun(segmenter_.get(), enMode_);
    if (comp_.empty() || phoneticTable_.empty()) {
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
    for (const auto &t : comp_.toks) {
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

    std::string context = surroundingContext(ic);

    stopWorker();
    convertState_ = ConvertState::Converting;
    convertBuffer_.clear();
    convertPositions_ = positions;
    convertIntervals_ = intervals;
    convertSyllables_ = syllables;
    convertToks_ = comp_.toks;
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
        comp_.toks.begin(), comp_.toks.end(),
        [this](const slothing::SegTok &t) {
            return t.zh && phoneticTable_.count(t.v);
        });

    worker_ = std::thread([this, icRef, generation, n, pureZh, commitDirect,
                           toks = std::vector<slothing::SegTok>(comp_.toks),
                           positions = std::move(positions),
                           context = std::move(context)]() {
        DaemonError err = DaemonError::None;
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
            for (auto &sentence : slothing::queryDecoder(
                     syls, n, context, inflightFd_, err, &full)) {
                if (!slothing::matchesPositions(sentence, positions)) {
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
            // mixed zh/en: decode each zh run, keep en literals — build a
            // per-token display and join with joinDisplay so English runs get
            // the same spacing as the preedit (a raw `sentence += tok.v`
            // dropped the spaces the user typed between words: "web app" ->
            // "webapp").
            std::vector<std::string> disp;
            bool ok = true;
            size_t i = 0;
            while (i < toks.size() && !workerStop_.load()) {
                if (!toks[i].zh) {
                    disp.push_back(toks[i].v);
                    i++;
                    continue;
                }
                std::vector<std::string> run;
                while (i < toks.size() && toks[i].zh) {
                    run.push_back(toks[i].v);
                    i++;
                }
                auto sentences =
                    slothing::queryDecoder(run, 1, "", inflightFd_, err);
                if (!sentences.empty() &&
                    utf8::lengthValidated(sentences[0]) == run.size()) {
                    const std::string &sent = sentences[0];
                    for (size_t k = 0, off = 0; k < run.size(); k++) {
                        size_t len = utf8::ncharByteLength(sent.begin() + off, 1);
                        disp.push_back(sent.substr(off, len));
                        off += len;
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if (ok && !disp.empty()) {
                std::string sentence =
                    joinDisplay(toks, disp, -1, std::string()).text;
                if (!sentence.empty()) {
                    verified.push_back(std::move(sentence));
                }
            }
        }
        if (workerStop_.load()) {
            return;
        }
        std::string failNotice;
        if (verified.empty()) {
            switch (err) {
            case DaemonError::Connect:
                failNotice = "slothingd 未執行";
                break;
            case DaemonError::Io:
                failNotice = "slothingd 無回應";
                break;
            default:
                failNotice = "無法解碼";
                break;
            }
        }
        dispatcher_.schedule([this, icRef, generation, commitDirect,
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
            if (commitDirect) { // Enter: one keypress commits (新注音)
                acceptConversion(ic, verified.front());
                return;
            }
            showConversionChoices(ic, verified);
        });
    });
}

void ChewingEngine::showConversionChoices(
    InputContext *ic, const std::vector<std::string> &sentences) {
    convertState_ = ConvertState::Choosing;
    convertTimer_.reset();
    const std::string &best = sentences.empty() ? convertBuffer_ : sentences[0];
    choosing_.begin(convertPositions_, convertIntervals_, convertSyllables_,
                    convertToks_, best, pendingFocus_);
    pendingFocus_ = -1;
    renderSegments(ic);
}

void ChewingEngine::renderSegments(InputContext *ic) {
    ic->inputPanel().reset();
    const auto useClient = ic->capabilityFlags().test(CapabilityFlag::Preedit);
    const auto base =
        useClient ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
    Text preedit;
    for (size_t i = 0; i < choosing_.positions.size(); i++) {
        const std::string &seg = choosing_.positions[i][choosing_.segSel[i]];
        if (static_cast<int>(i) == choosing_.segFocus) {
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

    if (!choosing_.candListOpen) {
        ic->inputPanel().setAuxDown(
            Text("←→ 移動　↓ 選字　⏎ 上字　Esc 取消"));
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
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
    const auto &phrases = choosing_.ensurePhrases();
    {
        const auto &cands = choosing_.positions[choosing_.segFocus];
        for (size_t j = 0; j < cands.size(); j++) {
            list->append(std::make_unique<SegmentCandidateWord>(
                this, static_cast<int>(j), cands[j]));
        }
        const int curIdx = choosing_.chCursor;
        list->setGlobalCursorIndex(curIdx);
        if (*config_.PageSize > 0) {
            list->setPage(curIdx / *config_.PageSize);
        }
    }
    if (!phrases.empty()) { // 詞 options: ⇧1-9 or ←→+⏎ (highlight marked)
        std::string aux = "詞:";
        for (size_t j = 0; j < phrases.size() && j < 9; j++) {
            const bool hl = static_cast<int>(j) == choosing_.phraseHl;
            aux += std::string(" ") + (hl ? "【" : "") + "⇧" +
                   std::to_string(j + 1) + " " + phrases[j].second +
                   (hl ? "】" : "");
        }
        ic->inputPanel().setAuxUp(Text(aux));
    }
    ic->inputPanel().setCandidateList(std::move(list));
    ic->inputPanel().setAuxDown(
        Text("1-9 選字　⇧1-9 選詞　←→ 移動　⏎ 確認　Esc 取消"));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::pickPhrase(InputContext *ic, int start,
                               const std::string &phrase) {
    choosing_.pickPhrase(start, phrase);
    renderSegments(ic);
}

void ChewingEngine::pickSegment(InputContext *ic, int candIdx) {
    choosing_.pickSegment(candIdx);
    renderSegments(ic);
}

void ChewingEngine::acceptConversion(InputContext *ic,
                                     const std::string &sentence) {
    commitAndRecord(ic, sentence);
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    liveGeneration_++;
    livePreedit_.clear();
    liveDisp_.clear();
    liveToks_.clear();
    comp_.clear();
    choosing_.clear();
    pendingFocus_ = -1;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertSyllables_.clear();
    convertToks_.clear();
    updateUI(ic);
    predictChain_ = 0;
    renderPredictions(ic); // 聯想 flips on after the commit (微軟-style)
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
    if (comp_.empty() && convertState_ == ConvertState::Composing) {
        commitAndRecord(ic, sym);
        ic->inputPanel().reset();
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
    }
    comp_.commitRun(segmenter_.get(), enMode_);
    comp_.insertToken({false, sym});
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
        if (choosing_.segFocus < 0 || choosing_.empty()) {
            cancelConversion(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Escape)) {
            if (choosing_.candListOpen) { // close the window first (chewing)
                choosing_.candListOpen = false;
                renderSegments(ic);
                return;
            }
            cancelConversion(ic);
            renderComposing(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Return)) {
            if (choosing_.candListOpen) {
                // Enter confirms the highlight (詞 or 字)
                choosing_.confirmHighlight();
                renderSegments(ic);
                return;
            }
            // learn the user's corrections before committing
            json payload = choosing_.learnPayload();
            if (!payload["chars"].empty() || !payload["phrases"].empty()) {
                slothing::sendLearn(payload);
            }
            acceptConversion(ic, choosing_.composedSentence());
            return;
        }
        if (keyEvent.key().check(FcitxKey_Right) ||
            keyEvent.key().check(FcitxKey_Left)) {
            const int d = keyEvent.key().check(FcitxKey_Right) ? 1 : -1;
            if (choosing_.candListOpen) {
                // ←→ walk 詞 options + chars as one loop
                choosing_.moveHighlight(d);
            } else {
                choosing_.moveFocus(d);
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
            if (!choosing_.candListOpen) { // reopen at the focused char
                choosing_.reopen();
            } else { // ↓/↑/space page the char list
                if (auto list = ic->inputPanel().candidateList()) {
                    if (auto *pageable = list->toPageable()) {
                        if (keyEvent.key().check(FcitxKey_Up)) {
                            if (pageable->hasPrev()) pageable->prev();
                        } else if (pageable->hasNext()) {
                            pageable->next();
                        }
                        ic->updateUserInterface(
                            UserInterfaceComponent::InputPanel);
                    }
                }
                return;
            }
        } else if (choosing_.candListOpen &&
                   keyEvent.key().states().test(KeyState::Shift) &&
                   keyEvent.key().isSimple()) {
            // ⇧1-9 picks a 詞 (word) option from the aux row
            const char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            static const char *shifted = "!@#$%^&*(";
            const auto &ph = choosing_.ensurePhrases();
            for (int i = 0; i < 9; i++) {
                if (shifted[i] == sym && i < static_cast<int>(ph.size())) {
                    pickPhrase(ic, ph[i].first, ph[i].second);
                    return;
                }
            }
        } else if (keyEvent.key().isSimple()) {
            const char *selkeys =
                builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
            char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            // number keys select from the VISIBLE list
            const auto list = ic->inputPanel().candidateList();
            const int nvis =
                list ? list->size()
                     : static_cast<int>(
                           choosing_.positions[choosing_.segFocus].size());
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
    // 聯想 predictions showing (post-commit, empty buffer): ⇧1-9 selects; any
    // other key dismisses the row and is then processed normally.
    if (predicting_ && convertState_ == ConvertState::Composing) {
        if (keyEvent.key().states().test(KeyState::Shift) &&
            keyEvent.key().isSimple()) {
            const char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            static const char *shifted = "!@#$%^&*(";
            auto preds = assoc_.predictions();
            for (int i = 0; i < 9; i++) {
                if (shifted[i] == sym &&
                    i < static_cast<int>(preds.size())) {
                    keyEvent.filterAndAccept();
                    commitAndRecord(ic, preds[i]);
                    if (++predictChain_ < 5) {
                        renderPredictions(ic);
                    } else {
                        predicting_ = false;
                        predictChain_ = 0;
                        ic->inputPanel().reset();
                        ic->updatePreedit();
                        ic->updateUserInterface(
                            UserInterfaceComponent::InputPanel);
                    }
                    return;
                }
            }
        }
        predicting_ = false;
        ic->inputPanel().reset();
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        // fall through: the key is processed normally
    }
    convertNotice_.clear();

    // The convert key force-decodes into the segment UI (kept for muscle
    // memory; live conversion normally makes it unnecessary).
    if (keyEvent.key().normalize().checkKeyList(*config_.ConvertKey)) {
        if (!comp_.empty()) {
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
        if (comp_.empty()) {
            return; // nothing composed: let the app have the key
        }
        keyEvent.filterAndAccept();
        if (!comp_.rawKeys.empty()) {
            return; // chewing: arrows ignored while composing a syllable
        }
        using Move = slothing::ComposingCore::Move;
        comp_.moveCursor(keyEvent.key().check(FcitxKey_Left)    ? Move::Left
                         : keyEvent.key().check(FcitxKey_Right) ? Move::Right
                         : keyEvent.key().check(FcitxKey_Home)  ? Move::Home
                                                                : Move::End);
        scheduleLiveDecode(ic);
        renderComposing(ic);
        return;
    }

    // ↓ while composing: per-character selection at the cursor's segment
    // (微軟新注音 idiom).
    if (keyEvent.key().check(FcitxKey_Down)) {
        if (!comp_.empty()) {
            keyEvent.filterAndAccept();
            comp_.commitRun(segmenter_.get(), enMode_);
            // chewing: candidates for the char AT the cursor; at the end of
            // the buffer, the last character.
            const int n = static_cast<int>(comp_.toks.size());
            pendingFocus_ =
                comp_.tokCursor < 0
                    ? n - 1
                    : std::max(0, std::min(comp_.tokCursor, n - 1));
            startDecode(ic);
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_Escape)) {
        // chewing: Esc clears only the pending bopomofo, never converted text
        if (!comp_.rawKeys.empty()) {
            keyEvent.filterAndAccept();
            comp_.rawKeys.clear();
            renderComposing(ic);
        } else if (!comp_.empty()) {
            keyEvent.filterAndAccept(); // swallow, keep the sentence
        }
        return;
    }

    if (keyEvent.key().check(FcitxKey_BackSpace)) {
        if (!comp_.empty()) {
            keyEvent.filterAndAccept();
            const bool hadRaw = !comp_.rawKeys.empty();
            comp_.backspace();
            if (!hadRaw) {
                scheduleLiveDecode(ic);
            }
            renderComposing(ic);
        }
        return;
    }

    // Enter: commit. If the live decode is fresh, commit instantly;
    // otherwise decode-then-commit.
    if (keyEvent.key().check(FcitxKey_Return)) {
        if (!comp_.empty()) {
            keyEvent.filterAndAccept();
            comp_.commitRun(segmenter_.get(), enMode_);
            if (!livePreedit_.empty() && liveToks_ == comp_.toks) {
                acceptConversion(ic, livePreedit_);
                return;
            }
            startDecode(ic, /*commitDirect=*/true);
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

        // Forced-English mode (lone Shift): PASSTHROUGH, 微軟/chewing
        // style — keys go straight to the app, no preedit, no Enter.
        // (With text composed, flush it first so ordering is preserved.)
        if (enMode_) {
            keyEvent.filterAndAccept();
            if (c == ' ' || (c >= 33 && c < 127)) {
                if (!comp_.empty()) {
                    comp_.commitRun(segmenter_.get(), enMode_);
                    ic->commitString(
                        (!livePreedit_.empty() && liveToks_ == comp_.toks)
                            ? livePreedit_
                            : tidySpaces(toksDisplay(comp_.toks)));
                    comp_.clear();
                    livePreedit_.clear();
                    liveDisp_.clear();
                    liveToks_.clear();
                    renderComposing(ic);
                }
                commitAndRecord(ic, fullWidth_ ? toFullWidth(c)
                                               : std::string(1, c));
            }
            return;
        }
        // Capitals are unambiguous ENGLISH evidence: they have no zhuyin
        // mapping (dachen keys are lowercase), so they stay verbatim in the
        // raw run and the segmenter routes them (and usually the letters
        // around them) into an English token. Case is preserved.

        // Space finalizes the current run (also serves as tone 1 — the
        // segmenter already treats a bare valid base as a tone-1 syllable).
        if (c == ' ') {
            if (!comp_.rawKeys.empty()) {
                keyEvent.filterAndAccept();
                comp_.commitRun(segmenter_.get(), enMode_);
                scheduleLiveDecode(ic);
                renderComposing(ic);
            } else if (!comp_.toks.empty()) {
                keyEvent.filterAndAccept(); // swallow: don't type into app
            }
            return;
        }

        // A tone key completes the last syllable -> finalize + live decode.
        if (slothing::segToneMark(c) && !comp_.rawKeys.empty()) {
            keyEvent.filterAndAccept();
            comp_.rawKeys += c;
            comp_.commitRun(segmenter_.get(), enMode_);
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

        // Punctuation, 微軟新注音/chewing conventions. Appended as a literal
        // token (commits with the sentence); committed directly when nothing
        // is being composed.
        const char rawSym = static_cast<char>(keyEvent.key().sym() & 0xff);
        if (auto pit = punctMap().find(rawSym); pit != punctMap().end()) {
            keyEvent.filterAndAccept();
            if (comp_.empty()) {
                commitAndRecord(ic, pit->second);
                renderPredictions(ic); // tail cleared by punct -> hides
                return;
            }
            comp_.commitRun(segmenter_.get(), enMode_);
            comp_.insertToken({false, pit->second});
            scheduleLiveDecode(ic);
            renderComposing(ic);
            return;
        }

        // Any zhuyin-mappable or alphanumeric key extends the raw run; the
        // segmenter re-decides zh/en live (auto code-switch, no mode key).
        const bool feeds = slothing::dachenMap().count(c) ||
                           (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '\'';
        if (feeds) {
            keyEvent.filterAndAccept();
            comp_.rawKeys += c;
            renderComposing(ic);
            return;
        }
    }

    // Anything else: if we're composing, swallow printable noise so stray
    // keys don't corrupt the run; let real control keys pass.
    if (!comp_.empty()) {
        if (keyEvent.key().isSimple()) {
            keyEvent.filterAndAccept();
        }
    }
}

} // namespace fcitx

FCITX_ADDON_FACTORY(fcitx::ChewingEngineFactory);
