// SlothingSession — the frontend-free IME driver behind the JNI. It is to
// Android what SlothingImpl (engine/ibus-slothing/src/main.cpp) is to IBus:
// it owns the shared state machine (Segmenter + ComposingCore + ChoosingCore),
// runs the same chewing-parity flow, and drives decode through an injected
// Decoder instead of the Unix socket. The one difference from the Linux
// frontends is that it PAINTS NOTHING: every method either mutates state or
// returns a plain view struct, and the InputMethodService (Kotlin) renders it.
//
// Requires the decode-port delta to engine/common/core.h (see decoder.h):
//   ChoosingCore gains `void setDecoder(Decoder*)` and routes rescore() /
//   ensurePhrases() through it. Everything else in core.h is used unchanged.
//
// Threading contract (enforced by the JNI + Kotlin, guarded here by mu_):
//   * Cheap, UI-thread ops lock mu_ for their (short) duration.
//   * Heavy, worker-thread ops that run during a MODAL phase (Converting /
//     Choosing) also hold mu_ for the whole call — the UI thread only reads,
//     and Kotlin awaits them before repainting, so there is no contention.
//   * decodeLive() is the one heavy op that runs WHILE the user keeps typing:
//     it snapshots under mu_, releases mu_ for the ONNX forward, re-locks, and
//     applies only if liveGen_ is unchanged (stale results are dropped).
#ifndef _SLOTHING_ANDROID_SESSION_H_
#define _SLOTHING_ANDROID_SESSION_H_

#include "core.h"     // ComposingCore, ChoosingCore, staleDisplay, matchesPositions
#include "decoder.h"  // Decoder (injected)
#include "display.h"  // joinDisplay, toksDisplay, tidySpaces, utf8*, punctMap
#include "segment.h"  // Segmenter, SegTok
#include "zhuyin.h"   // dachenMap, segToneMark

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace slothing {

// ---- view structs handed to the JNI (mapped to Kotlin data classes) --------

struct PreeditView {
    std::string text;
    int cursor = 0;   // codepoint index of the caret within text
    int hlStart = -1; // focused-segment highlight (codepoint range), -1 = none
    int hlEnd = -1;
};

struct CandidateView {
    int focus = 0;                    // segFocus
    int cursor = 0;                   // chCursor within items
    bool open = false;                // candListOpen
    std::vector<std::string> items;   // positions[segFocus]
};

struct PhraseView {
    int start = 0;       // token index the phrase covers (pickPhrase arg)
    std::string text;    // the 2-char 詞
};

// What a key did, so Kotlin knows whether to repaint / decode / drain commit.
enum class KeyOutcome {
    Ignored,   // not consumed — let the app handle the key
    Consumed,  // state changed; repaint preedit; no decode
    NeedLive,  // a run finalized; repaint then schedule a live decode
    Committed  // text was staged for commit; drain getCommit() + repaint
};

class SlothingSession {
public:
    // dec is owned by the session. table = the phonetic_table.tsv bytes
    // (syllable \t chars), same file the Linux addons install; it seeds both
    // the reading map and the Segmenter's valid bases. assocTsv = the 聯想
    // dictionary (head char \t completions, from model/assoc_tc.tsv);
    // assocUserPath = the personal bigram store ("" = in-memory only).
    SlothingSession(std::unique_ptr<Decoder> dec, const std::string &table,
                    const std::string &assocTsv = std::string(),
                    std::string assocUserPath = std::string())
        : dec_(std::move(dec)) {
        loadPhoneticTable(table);
        loadAssoc(assocTsv);
        loadAssocUser(std::move(assocUserPath));
        choosing_.setDecoder(dec_.get()); // decode-port delta on ChoosingCore
    }

    bool ready() const { return segmenter_ != nullptr && dec_ != nullptr; }

    // Debug/benchmark only: decode a syllable run straight through the injected
    // Decoder (bypasses the keyboard + segmenter), returning the best sentence.
    // Used by the on-device accuracy self-test to score against the reference.
    std::string decodeBest(const std::vector<std::string> &syls) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!dec_) return {};
        auto r = dec_->decode(syls, 1, std::string());
        return r.sentences.empty() ? std::string() : r.sentences[0];
    }

    // ---- context from the app (InputConnection.getTextBeforeCursor) --------
    void setContext(std::string ctx) {
        std::lock_guard<std::mutex> lk(mu_);
        context_ = std::move(ctx);
    }

    // ---- mode toggles ------------------------------------------------------
    void setEnglishMode(bool on) {
        std::lock_guard<std::mutex> lk(mu_);
        enMode_ = on;
    }
    void setFullWidth(bool on) {
        std::lock_guard<std::mutex> lk(mu_);
        fullWidth_ = on;
    }

    // ---- composing input (all cheap, UI thread) ----------------------------

    // A printable key. In Composing: zhuyin/alnum extend the raw run (the
    // segmenter re-decides zh/en live); punctuation / ` / lone-Shift English
    // are handled inline. Tone keys and space go through toneOrSpace().
    KeyOutcome feedKey(uint32_t cp) {
        std::lock_guard<std::mutex> lk(mu_);
        if (cp > 0x7f || cp == 0) {
            return KeyOutcome::Ignored; // soft keyboard only sends ASCII here
        }
        const char c = static_cast<char>(cp);

        // lone-Shift English: passthrough (flush composed first for ordering).
        if (enMode_) {
            if (c == ' ' || (c >= 33 && c < 127)) {
                if (!comp_.empty()) {
                    comp_.commitRun(segmenter_.get(), enMode_);
                    stageCommit(composedFallback());
                    comp_.clear();
                    clearLive();
                }
                stageCommit(fullWidth_ ? toFullWidth(c) : std::string(1, c));
                bumpLive();
                return KeyOutcome::Committed;
            }
            return KeyOutcome::Consumed;
        }

        // ` opens the symbol menu (Kotlin renders symbolCats()).
        if (c == '`') {
            symbolMode_ = true;
            return KeyOutcome::Consumed;
        }

        // 微軟新注音/chewing punctuation.
        if (auto it = punctMap().find(c); it != punctMap().end()) {
            if (comp_.empty()) {
                stageCommit(it->second);
                return KeyOutcome::Committed;
            }
            comp_.commitRun(segmenter_.get(), enMode_);
            comp_.insertToken({false, it->second});
            bumpLive();
            return KeyOutcome::NeedLive;
        }

        // Any zhuyin-mappable or alphanumeric key extends the raw run.
        const bool feeds = dachenMap().count(c) || (c >= 'a' && c <= 'z') ||
                           (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                           c == '\'';
        if (feeds) {
            comp_.rawKeys += c;
            return KeyOutcome::Consumed; // preedit shows the segmenter tail
        }
        return comp_.empty() ? KeyOutcome::Ignored : KeyOutcome::Consumed;
    }

    // A tone key (3/4/6/7) or space: finalize the current run. Space with an
    // empty buffer is not consumed (let the app get a real space).
    KeyOutcome toneOrSpace(uint32_t cp) {
        std::lock_guard<std::mutex> lk(mu_);
        const char c = static_cast<char>(cp);
        if (c == ' ' && shiftDownForFullwidth_) { /* handled by setFullWidth */ }
        if (segToneMark(c) && !comp_.rawKeys.empty()) {
            comp_.rawKeys += c;
            comp_.commitRun(segmenter_.get(), enMode_);
            bumpLive();
            return KeyOutcome::NeedLive;
        }
        if (c == ' ') {
            if (!comp_.rawKeys.empty()) {
                comp_.commitRun(segmenter_.get(), enMode_);
                bumpLive();
                return KeyOutcome::NeedLive;
            }
            return comp_.toks.empty() ? KeyOutcome::Ignored
                                      : KeyOutcome::Consumed;
        }
        return KeyOutcome::Consumed;
    }

    KeyOutcome backspace() {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.empty()) {
            return KeyOutcome::Ignored;
        }
        const bool hadRaw = !comp_.rawKeys.empty();
        comp_.backspace();
        if (hadRaw) {
            return KeyOutcome::Consumed; // popped a raw byte; no re-decode
        }
        bumpLive();
        return KeyOutcome::NeedLive; // removed a finalized token
    }

    // dir: -1 Left, +1 Right, -2 Home, +2 End. Arrows are ignored mid-syllable
    // (chewing). Returns NeedLive when the caret actually moved.
    KeyOutcome moveCursor(int dir) {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.empty()) {
            return KeyOutcome::Ignored;
        }
        if (!comp_.rawKeys.empty()) {
            return KeyOutcome::Consumed;
        }
        using M = ComposingCore::Move;
        comp_.moveCursor(dir == -1   ? M::Left
                         : dir == 1  ? M::Right
                         : dir == -2 ? M::Home
                                     : M::End);
        bumpLive();
        return KeyOutcome::NeedLive;
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        hardClear();
    }

    // Focus-out / onFinishInput: chewing commits pending text rather than
    // dropping it. Stages the commit; Kotlin drains getCommit().
    void flush() {
        std::lock_guard<std::mutex> lk(mu_);
        std::string pending;
        if (state_ == State::Choosing) {
            pending = choosing_.composedSentence();
        } else if (!comp_.empty()) {
            comp_.commitRun(segmenter_.get(), enMode_);
            pending = composedFallback();
        }
        if (!pending.empty()) {
            stageCommit(pending);
        }
        hardClear();
    }

    // ---- live (modeless) conversion ---------------------------------------

    // Resolve the live display without the model when possible (empty or all
    // English). Returns true if resolved synchronously; false means the caller
    // must run decodeLive() on a worker.
    bool refreshLiveFast() {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.toks.empty() || !segmenter_) {
            clearLive();
            return true;
        }
        bool anyZh = false;
        for (const auto &t : comp_.toks) anyZh |= t.zh;
        if (!anyZh) {
            liveDisp_.clear();
            for (const auto &t : comp_.toks) liveDisp_.push_back(t.v);
            liveToks_ = comp_.toks;
            livePreedit_ =
                joinDisplay(comp_.toks, liveDisp_, -1, std::string()).text;
            return true;
        }
        return false; // needs the model
    }

    // HEAVY, worker thread. Snapshot -> unlock -> ONNX -> re-lock -> apply iff
    // the generation is unchanged. Returns true if a fresh live display landed.
    // When the whole buffer is one zh run (the common case), also fetches the
    // n-best sentences for the always-visible mobile suggestion strip
    // (Gboard/iOS convention: candidates appear automatically as you type).
    bool decodeLive() {
        std::vector<SegTok> toks;
        std::string ctx;
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (state_ != State::Composing) return false;
            toks = comp_.toks;
            ctx = context_;
            gen = liveGen_;
        }
        std::vector<std::string> disp;
        std::vector<std::string> nbest; // whole-buffer alternatives (pure zh)
        std::vector<std::string> lastCands; // ranked chars for the LAST zh token
        int lastIdx = -1;
        std::string runCtx = ctx;
        bool allOk = true;
        for (size_t i = 0; i < toks.size();) {
            if (!toks[i].zh) {
                disp.push_back(toks[i].v);
                i++;
                continue;
            }
            std::vector<std::string> run;
            const size_t start = i;
            while (i < toks.size() && toks[i].zh) run.push_back(toks[i++].v);
            const bool wholeBuffer = (start == 0 && i == toks.size());
            auto res = dec_->decode(run, wholeBuffer ? kLiveNBest : 1, runCtx);
            if (!res.sentences.empty() &&
                utf8Length(res.sentences[0]) == run.size()) {
                const std::string &s = res.sentences[0];
                if (wholeBuffer) nbest = res.sentences;
                // mobile convention (Gboard/iOS/Rime): candidates for the last
                // word in the buffer show AUTOMATICALLY — capture the ranked
                // chars of the final zh token from the same decode reply
                if (i == toks.size() && !res.candidates.empty()) {
                    lastCands = res.candidates.back();
                    if (static_cast<int>(lastCands.size()) > kLastCands)
                        lastCands.resize(kLastCands);
                    lastIdx = static_cast<int>(toks.size()) - 1;
                }
                runCtx += s;
                for (size_t k = 0, off = 0; k < run.size(); k++) {
                    size_t len = utf8SeqLen(s[off]);
                    disp.push_back(s.substr(off, len));
                    off += len;
                }
            } else {
                allOk = false;
                for (size_t k = start; k < start + run.size(); k++)
                    disp.push_back(toks[k].v);
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (gen != liveGen_ || state_ != State::Composing || !allOk) {
            return false; // stale or partial: keep whatever we had
        }
        // re-apply the user's in-composition picks (and prune stale ones)
        for (auto it = liveFixed_.begin(); it != liveFixed_.end();) {
            const int idx = it->first;
            if (idx < static_cast<int>(toks.size()) && toks[idx].zh &&
                toks[idx].v == it->second.first &&
                idx < static_cast<int>(disp.size())) {
                disp[idx] = it->second.second;
                ++it;
            } else {
                it = liveFixed_.erase(it); // token gone or resegmented
            }
        }
        liveDisp_ = std::move(disp);
        liveToks_ = std::move(toks);
        livePreedit_ = joinDisplay(liveToks_, liveDisp_, -1, std::string()).text;
        liveSents_ = std::move(nbest);
        liveLastCands_ = std::move(lastCands);
        liveLastIdx_ = lastIdx;
        return true;
    }

    std::string getLive() {
        std::lock_guard<std::mutex> lk(mu_);
        return (!livePreedit_.empty() && liveToks_ == comp_.toks) ? livePreedit_
                                                                  : std::string();
    }

    // n-best sentence suggestions for the always-visible strip (fresh only;
    // [0] is the sentence already shown inline in the preedit). Empty when the
    // buffer is empty, stale, or mixed zh/en.
    std::vector<std::string> getLiveSuggestions() {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.toks.empty() || liveToks_ != comp_.toks) return {};
        return liveSents_;
    }

    // Auto candidates for the LAST word in the buffer (mobile convention) —
    // model-ranked chars of the final zh token, from the live decode. Fresh
    // only; empty when the last token is English or the buffer is empty.
    std::vector<std::string> getLastWordCands() {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.toks.empty() || liveToks_ != comp_.toks || liveLastIdx_ < 0)
            return {};
        return liveLastCands_;
    }

    // The char currently displayed for the last word (for the selected chip).
    std::string getLastWordCurrent() {
        std::lock_guard<std::mutex> lk(mu_);
        if (comp_.toks.empty() || liveToks_ != comp_.toks || liveLastIdx_ < 0 ||
            liveLastIdx_ >= static_cast<int>(liveDisp_.size()))
            return {};
        return liveDisp_[liveLastIdx_];
    }

    // Tap on a last-word candidate chip: replace that character IN PLACE and
    // keep composing (no modal window). The pick sticks across further typing
    // (re-applied after every live decode) and is learned on commit.
    bool pickLastWord(const std::string &ch) {
        std::lock_guard<std::mutex> lk(mu_);
        if (ch.empty() || state_ != State::Composing || liveLastIdx_ < 0 ||
            liveLastIdx_ >= static_cast<int>(comp_.toks.size()) ||
            liveToks_ != comp_.toks ||
            liveLastIdx_ >= static_cast<int>(liveDisp_.size()))
            return false;
        liveFixed_[liveLastIdx_] = {comp_.toks[liveLastIdx_].v, ch};
        liveDisp_[liveLastIdx_] = ch;
        livePreedit_ =
            joinDisplay(liveToks_, liveDisp_, -1, std::string()).text;
        return true;
    }

    // Tap on a suggestion chip: commit that sentence outright (Gboard-style
    // tap-to-commit; equivalent to Enter when it's suggestion [0]).
    void commitSentence(const std::string &s) {
        std::lock_guard<std::mutex> lk(mu_);
        if (s.empty()) return;
        stageCommit(s);
        hardClear();
    }

    // 符 symbol-strip tap: literal token into the composition (commits with
    // the sentence), or staged for direct commit when nothing is composed —
    // mirrors the desktop engines' pickSymbol.
    KeyOutcome insertSymbol(const std::string &sym) {
        std::lock_guard<std::mutex> lk(mu_);
        if (sym.empty() || state_ != State::Composing) {
            return KeyOutcome::Consumed;
        }
        if (comp_.empty()) {
            stageCommit(sym);
            return KeyOutcome::Committed;
        }
        comp_.commitRun(segmenter_.get(), enMode_);
        comp_.insertToken({false, sym});
        bumpLive();
        return KeyOutcome::NeedLive;
    }

    // ---- convert / choose --------------------------------------------------

    // ↓ / commit-convert. HEAVY, worker thread (modal). focus = the token the
    // ↓ landed on (pendingFocus), or -1 for the first ambiguous. When
    // commitDirect (Enter), a single successful decode is committed outright
    // (新注音) instead of opening the window — stages the commit, returns
    // false, and Kotlin drains getCommit(). Returns true when the window opened.
    bool beginConvert(int focus, bool commitDirect) {
        std::lock_guard<std::mutex> lk(mu_);
        comp_.commitRun(segmenter_.get(), enMode_);
        if (comp_.empty() || phoneticTable_.empty()) {
            notice_ = phoneticTable_.empty() ? "無音表" : "";
            return false;
        }
        std::vector<std::vector<std::string>> positions;
        std::vector<std::pair<int, int>> intervals;
        std::vector<std::string> syllables;
        int at = 0;
        for (const auto &t : comp_.toks) {
            int span = 1;
            if (t.zh) {
                auto it = phoneticTable_.find(t.v);
                if (it != phoneticTable_.end() && !it->second.empty()) {
                    positions.push_back(it->second);
                } else {
                    positions.push_back({t.v});
                    span = static_cast<int>(utf8Length(t.v));
                }
                syllables.push_back(t.v);
            } else {
                positions.push_back({t.v});
                span = static_cast<int>(utf8Length(t.v));
                syllables.push_back("");
            }
            intervals.emplace_back(at, at + span);
            at += span;
        }
        const bool pureZh =
            std::all_of(comp_.toks.begin(), comp_.toks.end(),
                        [this](const SegTok &t) {
                            return t.zh && phoneticTable_.count(t.v);
                        });

        std::vector<std::string> verified;
        std::vector<std::vector<std::string>> ranked;
        if (pureZh) {
            std::vector<std::string> syls;
            for (const auto &t : comp_.toks) syls.push_back(t.v);
            auto res = dec_->decode(syls, 5, context_);
            for (auto &s : res.sentences) {
                if (matchesPositions(s, positions) &&
                    std::find(verified.begin(), verified.end(), s) ==
                        verified.end()) {
                    verified.push_back(s);
                }
            }
            if (res.candidates.size() == positions.size()) {
                bool ok = true;
                for (size_t i = 0; i < res.candidates.size(); i++) {
                    if (res.candidates[i].size() != positions[i].size()) {
                        ok = false;
                        break;
                    }
                }
                if (ok) ranked = std::move(res.candidates);
            }
        } else {
            // mixed zh/en: decode each zh run, keep en literals, join with the
            // preedit spacing rules (so "web app" keeps its space).
            std::vector<std::string> disp;
            bool ok = true;
            for (size_t i = 0; i < comp_.toks.size();) {
                if (!comp_.toks[i].zh) {
                    disp.push_back(comp_.toks[i].v);
                    i++;
                    continue;
                }
                std::vector<std::string> run;
                while (i < comp_.toks.size() && comp_.toks[i].zh)
                    run.push_back(comp_.toks[i++].v);
                auto res = dec_->decode(run, 1, "");
                if (!res.sentences.empty() &&
                    utf8Length(res.sentences[0]) == run.size()) {
                    const std::string &s = res.sentences[0];
                    for (size_t k = 0, off = 0; k < run.size(); k++) {
                        size_t len = utf8SeqLen(s[off]);
                        disp.push_back(s.substr(off, len));
                        off += len;
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if (ok && !disp.empty()) {
                std::string s =
                    joinDisplay(comp_.toks, disp, -1, std::string()).text;
                if (!s.empty()) verified.push_back(std::move(s));
            }
        }

        if (verified.empty()) {
            notice_ = "無法解碼";
            return false;
        }
        if (!ranked.empty()) positions = ranked;
        if (commitDirect) {
            stageCommit(verified.front());
            hardClear();
            return false; // committed, no window
        }
        state_ = State::Choosing;
        choosing_.begin(positions, intervals, syllables, comp_.toks,
                        verified.front(), focus);
        choosing_.ensurePhrases(); // pre-warm phrases for the initial focus
        return true;
    }

    // Commit the already-shown live conversion without re-decoding (Enter when
    // liveToks == comp.toks). Returns true if it committed.
    bool commitLive() {
        std::lock_guard<std::mutex> lk(mu_);
        comp_.commitRun(segmenter_.get(), enMode_);
        if (!livePreedit_.empty() && liveToks_ == comp_.toks) {
            // in-composition picks are explicit user corrections: learn them
            // (same semantics as the Choosing window's learn-diff)
            if (!liveFixed_.empty() && dec_) {
                json chars = json::array();
                for (auto &kv : liveFixed_)
                    chars.push_back({kv.second.first, kv.second.second});
                dec_->learn(json{{"chars", chars}, {"phrases", json::array()}});
            }
            stageCommit(livePreedit_);
            hardClear();
            return true;
        }
        return false; // caller should beginConvert(commitDirect=true) instead
    }

    // ---- choosing-window navigation (cheap, UI thread) ---------------------
    // Phrases for the focused segment must be pre-warmed (beginConvert /
    // ensurePhrases) before these run, so ChoosingCore::ensurePhrases() inside
    // moveHighlight/confirm is a cache hit and never decodes on the UI thread.

    void moveHighlight(int dir) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.moveHighlight(dir);
    }
    void moveFocus(int dir) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.moveFocus(dir);
    }
    void reopenCandList() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.reopen();
    }
    void closeCandList() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.candListOpen = false;
    }

    // Pre-warm phrases for the current focus. HEAVY (phrasesScored), worker.
    void ensurePhrases() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.ensurePhrases();
    }

    // Pick a char / phrase. HEAVY (rescore), worker, modal.
    void pickSegment(int idx) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.pickSegment(idx);
    }
    void pickPhrase(int start, const std::string &phrase) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ == State::Choosing) choosing_.pickPhrase(start, phrase);
    }

    // Enter in Choosing. If the window is open, confirm the highlight (may
    // rescore — HEAVY, worker) and stay; else commit the sentence, run learn,
    // and clear. Returns true when it committed (Kotlin drains getCommit()).
    bool confirmChoosing() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::Choosing) return false;
        if (choosing_.candListOpen) {
            choosing_.confirmHighlight();
            return false; // still choosing
        }
        json payload = choosing_.learnPayload();
        if (!payload["chars"].empty() || !payload["phrases"].empty()) {
            dec_->learn(payload);
        }
        stageCommit(choosing_.composedSentence());
        hardClear();
        return true;
    }

    // Esc in Choosing: close the window first, then cancel back to Composing.
    // Returns true if it consumed the key.
    bool escapeChoosing() {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::Choosing) return false;
        if (choosing_.candListOpen) {
            choosing_.candListOpen = false;
            return true;
        }
        // cancel: drop the conversion, keep nothing staged (chewing keeps the
        // composed reading, but on-device we return to Composing preedit).
        state_ = State::Composing;
        choosing_.clear();
        return true;
    }

    // ---- read-only views (cheap, UI thread) --------------------------------

    PreeditView getPreedit() {
        std::lock_guard<std::mutex> lk(mu_);
        PreeditView v;
        if (state_ == State::Choosing) {
            std::string pre;
            size_t hlFromByte = 0, hlToByte = 0;
            bool hasHl = false;
            for (size_t i = 0; i < choosing_.positions.size(); i++) {
                const std::string &seg =
                    choosing_.positions[i][choosing_.segSel[i]];
                if (static_cast<int>(i) == choosing_.segFocus) {
                    hlFromByte = pre.size();
                    hlToByte = pre.size() + seg.size();
                    hasHl = true;
                }
                pre += seg;
            }
            v.text = pre;
            if (hasHl) {
                v.hlStart = cpIndex(pre, hlFromByte);
                v.hlEnd = cpIndex(pre, hlToByte);
                v.cursor = v.hlStart;
            } else {
                v.cursor = cpIndex(pre, pre.size());
            }
            return v;
        }
        // Composing (Converting shows the same plain preedit while decoding).
        std::string tail;
        if (!comp_.rawKeys.empty()) {
            tail = enMode_ ? comp_.rawKeys
                           : (segmenter_ ? tidySpaces(toksDisplay(
                                               segmenter_->segment(comp_.rawKeys)))
                                         : comp_.rawKeys);
        }
        std::vector<std::string> disp =
            staleDisplay(comp_.toks, liveToks_, liveDisp_);
        JoinResult jr = joinDisplay(comp_.toks, disp, comp_.tokCursor, tail);
        v.text = jr.text;
        v.cursor = cpIndex(jr.text, jr.cursorBytes);
        return v;
    }

    CandidateView getCandidates() {
        std::lock_guard<std::mutex> lk(mu_);
        CandidateView v;
        if (state_ != State::Choosing || choosing_.positions.empty()) return v;
        v.focus = choosing_.segFocus;
        v.cursor = choosing_.chCursor;
        v.open = choosing_.candListOpen;
        v.items = choosing_.positions[choosing_.segFocus];
        return v;
    }

    // Cheap read of the pre-warmed phrase cache for the focused segment.
    std::vector<PhraseView> getPhrases() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<PhraseView> out;
        if (state_ != State::Choosing) return out;
        auto it = choosing_.phraseCands.find(choosing_.segFocus);
        if (it == choosing_.phraseCands.end()) return out;
        for (auto &[start, text] : it->second) out.push_back({start, text});
        return out;
    }

    // Drain the staged commit (text to send via InputConnection.commitText).
    std::string getCommit() {
        std::lock_guard<std::mutex> lk(mu_);
        std::string s = std::move(pendingCommit_);
        pendingCommit_.clear();
        return s;
    }

    // ---- 聯想 next-word prediction (mobile convention: after a commit the
    // strip flips from candidates to predictions) ----------------------------

    // Predictions for what follows the last committed character: the user's
    // own bigrams first (learned from every commit), then the dictionary
    // completions (assoc_tc.tsv, 微軟新注音-style 聯想). Empty while composing.
    std::vector<std::string> getPredictions() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<std::string> out;
        if (!comp_.empty() || predictTail_.empty()) return out;
        // personal bigrams, count-ranked
        std::vector<std::pair<int, std::string>> mine;
        for (auto &kv : assocUser_)
            if (kv.first.first == predictTail_)
                mine.push_back({kv.second, kv.first.second});
        std::sort(mine.begin(), mine.end(),
                  [](auto &a, auto &b) { return a.first > b.first; });
        for (auto &p : mine) {
            if (out.size() >= kPredictions) break;
            out.push_back(p.second);
        }
        auto it = assoc_.find(predictTail_);
        if (it != assoc_.end())
            for (auto &c : it->second) {
                if (out.size() >= kPredictions) break;
                if (std::find(out.begin(), out.end(), c) == out.end())
                    out.push_back(c);
            }
        return out;
    }

    // A prediction chip was tapped and committed by the frontend: learn the
    // transition and move the tail so predictions chain.
    void predicted(const std::string &text) {
        std::lock_guard<std::mutex> lk(mu_);
        recordAssocLocked(text);
    }

    // Field switch: predictions from the previous field are stale.
    void clearPredictions() {
        std::lock_guard<std::mutex> lk(mu_);
        predictTail_.clear();
    }

    std::string getNotice() {
        std::lock_guard<std::mutex> lk(mu_);
        return notice_;
    }
    int state() {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(state_);
    }
    bool symbolMode() {
        std::lock_guard<std::mutex> lk(mu_);
        return symbolMode_;
    }

private:
    enum class State { Composing = 0, Converting = 1, Choosing = 2 };

    void loadPhoneticTable(const std::string &tsv) {
        std::set<std::string> validBase;
        size_t i = 0;
        while (i < tsv.size()) {
            size_t nl = tsv.find('\n', i);
            std::string line =
                tsv.substr(i, nl == std::string::npos ? std::string::npos : nl - i);
            i = (nl == std::string::npos) ? tsv.size() : nl + 1;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string syl = line.substr(0, tab);
            std::vector<std::string> chars = splitUtf8(line.substr(tab + 1));
            if (chars.empty()) continue;
            std::string base;
            for (const auto &ch : splitUtf8(syl)) {
                if (ch != "ˊ" && ch != "ˇ" && ch != "ˋ" && ch != "˙")
                    base += ch;
            }
            if (!base.empty()) validBase.insert(base);
            phoneticTable_.emplace(std::move(syl), std::move(chars));
        }
        if (!validBase.empty())
            segmenter_ = std::make_unique<Segmenter>(std::move(validBase));
    }

    // codepoint index of a byte offset within s.
    static int cpIndex(const std::string &s, size_t byteOff) {
        return static_cast<int>(utf8Length(s.substr(0, std::min(byteOff, s.size()))));
    }

    // The non-model fallback commit for the composed buffer (live if fresh,
    // else the plain token display).
    std::string composedFallback() {
        return (!livePreedit_.empty() && liveToks_ == comp_.toks)
                   ? livePreedit_
                   : tidySpaces(toksDisplay(comp_.toks));
    }

    void stageCommit(const std::string &s) {
        pendingCommit_ += s;
        recordAssocLocked(s); // feed 聯想: personal bigrams + prediction tail
    }

    // ---- 聯想 internals (callers hold mu_) ---------------------------------

    static bool isCjk(const std::string &cp) {
        // CJK Unified Ideographs (U+4E00..U+9FFF): E4B880..E9BFBF in UTF-8
        return cp.size() == 3 &&
               ((static_cast<unsigned char>(cp[0]) == 0xE4 &&
                 static_cast<unsigned char>(cp[1]) >= 0xB8) ||
                (static_cast<unsigned char>(cp[0]) >= 0xE5 &&
                 static_cast<unsigned char>(cp[0]) <= 0xE9));
    }

    // Count adjacent CJK bigrams across [predictTail_] + s, advance the tail.
    void recordAssocLocked(const std::string &s) {
        std::string prev = predictTail_;
        bool dirty = false;
        for (const auto &cp : splitUtf8(s)) {
            if (!isCjk(cp)) {
                prev.clear(); // punctuation/latin breaks adjacency
                continue;
            }
            if (!prev.empty()) {
                assocUser_[{prev, cp}]++;
                dirty = true;
            }
            prev = cp;
        }
        predictTail_ = prev; // last CJK char, or "" if s ended non-CJK
        if (dirty) saveAssocUser();
    }

    void loadAssoc(const std::string &tsv) {
        size_t i = 0;
        while (i < tsv.size()) {
            size_t nl = tsv.find('\n', i);
            std::string line = tsv.substr(
                i, nl == std::string::npos ? std::string::npos : nl - i);
            i = (nl == std::string::npos) ? tsv.size() : nl + 1;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string head = line.substr(0, tab);
            std::vector<std::string> comps;
            std::string rest = line.substr(tab + 1);
            size_t j = 0;
            while (j < rest.size()) {
                size_t sp = rest.find(' ', j);
                std::string c = rest.substr(
                    j, sp == std::string::npos ? std::string::npos : sp - j);
                if (!c.empty()) comps.push_back(c);
                if (sp == std::string::npos) break;
                j = sp + 1;
            }
            if (!comps.empty()) assoc_.emplace(std::move(head), std::move(comps));
        }
    }

    void loadAssocUser(std::string path) {
        assocUserFile_ = std::move(path);
        if (assocUserFile_.empty()) return;
        std::ifstream f(assocUserFile_);
        std::string line;
        while (std::getline(f, line)) {
            auto t1 = line.find('\t');
            if (t1 == std::string::npos) continue;
            auto t2 = line.find('\t', t1 + 1);
            if (t2 == std::string::npos) continue;
            int n = std::atoi(line.c_str() + t2 + 1);
            if (n > 0)
                assocUser_[{line.substr(0, t1),
                            line.substr(t1 + 1, t2 - t1 - 1)}] = n;
        }
    }

    void saveAssocUser() {
        if (assocUserFile_.empty()) return;
        std::ofstream f(assocUserFile_);
        for (auto &kv : assocUser_)
            f << kv.first.first << "\t" << kv.first.second << "\t" << kv.second
              << "\n";
    }
    void bumpLive() { liveGen_++; }
    void clearLive() {
        livePreedit_.clear();
        liveDisp_.clear();
        liveToks_.clear();
        liveSents_.clear();
        liveLastCands_.clear();
        liveLastIdx_ = -1;
        liveFixed_.clear();
    }
    void hardClear() {
        state_ = State::Composing;
        comp_.clear();
        choosing_.clear();
        clearLive();
        notice_.clear();
        symbolMode_ = false;
        liveGen_++;
    }

    std::unique_ptr<Decoder> dec_;
    std::unique_ptr<Segmenter> segmenter_;
    std::unordered_map<std::string, std::vector<std::string>> phoneticTable_;

    // 聯想: dictionary head-char -> completions, personal bigram counts, and
    // the last committed CJK char that predictions key off.
    std::unordered_map<std::string, std::vector<std::string>> assoc_;
    std::map<std::pair<std::string, std::string>, int> assocUser_;
    std::string assocUserFile_;
    std::string predictTail_;

    ComposingCore comp_;
    ChoosingCore choosing_;
    State state_ = State::Composing;

    bool enMode_ = false;
    bool fullWidth_ = false;
    bool symbolMode_ = false;
    bool shiftDownForFullwidth_ = false; // reserved for Shift+Space handling

    std::string context_;
    std::string notice_;
    std::string pendingCommit_;

    static constexpr int kLiveNBest = 5;
    static constexpr int kLastCands = 8;
    static constexpr size_t kPredictions = 8;

    std::string livePreedit_;
    std::vector<std::string> liveDisp_;
    std::vector<SegTok> liveToks_;
    std::vector<std::string> liveSents_; // n-best for the suggestion strip
    std::vector<std::string> liveLastCands_; // ranked chars, last zh token
    int liveLastIdx_ = -1;
    // in-composition picks: token index -> (syllable, chosen char); re-applied
    // after every live decode, learned on commit
    std::map<int, std::pair<std::string, std::string>> liveFixed_;
    uint64_t liveGen_ = 0;

    std::mutex mu_;
};

} // namespace slothing

#endif // _SLOTHING_ANDROID_SESSION_H_
