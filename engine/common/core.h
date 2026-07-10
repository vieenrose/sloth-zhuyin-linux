// The frontend-free Slothing IME state machine, shared by the fcitx5 and
// IBus engines (extracted from eim.cpp so the chewing-parity behavior is
// written once and unit-testable offline — see core_test.cpp).
//
//   ComposingCore  — the raw keystream + finalized token buffer with an
//                    insertion cursor, and the stale-preserving live-decode
//                    display (chewing/新注音 never regress to bopomofo).
//   ChoosingCore   — the segment-conversion window: per-position candidate
//                    lists, the combined ←→ highlight loop over 詞 chips and
//                    chars, pick-closes-window, hint-conditioned re-scoring
//                    of untouched positions, and the learn-diff payload.
//
// Frontends own only: key decoding into simple calls, async decode workers,
// and painting (preedit text, lookup table, aux rows) from this state.
#ifndef _SLOTHING_COMMON_CORE_H_
#define _SLOTHING_COMMON_CORE_H_

#include "daemon.h"
#include "display.h"
#include "segment.h"
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace slothing {

// Verifies `sentence` is exactly one candidate from each entry of `positions`,
// in order -- so it can only be a combination of legal characters, never
// arbitrary model output. Longest-match per position.
inline bool
matchesPositions(const std::string &sentence,
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

// ---- Composing ------------------------------------------------------------

struct ComposingCore {
    // Raw keystream of the current run (re-segmented live into zh/en tokens)
    // plus the finalized tokens of earlier runs (a run ends on a tone key or
    // space).
    std::string rawKeys;
    std::vector<SegTok> toks;
    // Insertion cursor over toks (token granularity): index into the token
    // list where the current run / new input lands. -1 = end.
    int tokCursor = -1;

    bool empty() const { return rawKeys.empty() && toks.empty(); }

    // Finalize the current run into tokens at the cursor. forcedEnglish: the
    // run is one literal token, never segmented (lone-Shift mode).
    void commitRun(const Segmenter *segmenter, bool forcedEnglish) {
        if (rawKeys.empty() || !segmenter) {
            return;
        }
        std::vector<SegTok> run;
        if (forcedEnglish) {
            run.push_back({false, rawKeys});
        } else {
            run = segmenter->segment(rawKeys);
        }
        insertTokens(std::move(run));
        rawKeys.clear();
    }

    void insertTokens(std::vector<SegTok> run) {
        if (tokCursor < 0 || tokCursor >= static_cast<int>(toks.size())) {
            for (auto &t : run) {
                toks.push_back(std::move(t));
            }
            tokCursor = -1;
        } else {
            toks.insert(toks.begin() + tokCursor, run.begin(), run.end());
            tokCursor += static_cast<int>(run.size());
        }
    }

    void insertToken(SegTok t) { insertTokens({std::move(t)}); }

    // Backspace: pop one UTF-8 char of the raw run, else the token before the
    // cursor. Returns false if there was nothing to remove.
    bool backspace() {
        if (!rawKeys.empty()) {
            // pop one UTF-8 character (enMode may hold fullwidth chars)
            while (!rawKeys.empty() &&
                   (static_cast<unsigned char>(rawKeys.back()) & 0xC0) == 0x80) {
                rawKeys.pop_back(); // continuation bytes
            }
            if (!rawKeys.empty()) {
                rawKeys.pop_back(); // lead / ASCII byte
            }
            return true;
        }
        const int n = static_cast<int>(toks.size());
        int cur = tokCursor < 0 ? n : tokCursor;
        if (cur > 0) {
            toks.erase(toks.begin() + (cur - 1));
            if (tokCursor >= 0) {
                tokCursor = cur - 1;
            }
            return true;
        }
        return false;
    }

    enum class Move { Left, Right, Home, End };
    void moveCursor(Move m) {
        const int n = static_cast<int>(toks.size());
        int cur = tokCursor < 0 ? n : tokCursor;
        switch (m) {
        case Move::Left: cur = std::max(0, cur - 1); break;
        case Move::Right: cur = std::min(n, cur + 1); break;
        case Move::Home: cur = 0; break;
        case Move::End: cur = n; break;
        }
        tokCursor = (cur >= n) ? -1 : cur;
    }

    void clear() {
        rawKeys.clear();
        toks.clear();
        tokCursor = -1;
    }
};

// Stale-preserving per-token display (chewing/新注音 never regress to
// bopomofo): while a new decode is in flight, reuse the previous conversion
// for tokens unchanged from the start (prefix) or the end (suffix — covers
// mid-sentence edits); only new tokens show bopomofo until decoded.
inline std::vector<std::string>
staleDisplay(const std::vector<SegTok> &cur, const std::vector<SegTok> &old,
             const std::vector<std::string> &oldDisp) {
    if (!oldDisp.empty() && cur == old) {
        return oldDisp;
    }
    std::vector<std::string> disp;
    if (oldDisp.empty()) {
        return disp;
    }
    disp.assign(cur.size(), std::string());
    size_t pre = 0;
    while (pre < cur.size() && pre < old.size() && cur[pre] == old[pre]) {
        disp[pre] = oldDisp[pre];
        pre++;
    }
    size_t suf = 0;
    while (suf < cur.size() - pre && suf < old.size() - pre &&
           cur[cur.size() - 1 - suf] == old[old.size() - 1 - suf]) {
        disp[cur.size() - 1 - suf] = oldDisp[old.size() - 1 - suf];
        suf++;
    }
    return disp;
}

// ---- Choosing (segment-conversion window) ----------------------------------

class ChoosingCore {
public:
    // The decode's per-syllable candidate lists, interval spans, syllables,
    // and the token list the conversion was started for.
    std::vector<std::vector<std::string>> positions;
    std::vector<std::pair<int, int>> intervals;
    std::vector<std::string> syllables; // zh syllable per token ("" for en)
    std::vector<SegTok> toks;
    // One selected candidate index per interval, and which segment the
    // arrows act on.
    std::vector<int> segSel;
    std::vector<int> initialSel;
    // Segments the user explicitly picked (hints for re-scoring; also the
    // only positions the learn store records).
    std::set<int> userFixed;
    int segFocus = 0;
    // Candidate window visibility (chewing: a pick CLOSES the window; ↓
    // reopens it; Esc closes it before cancelling the whole conversion).
    bool candListOpen = true;
    // ←→ highlight: phraseHl >= 0 = 詞 chip index; -1 = highlight is in the
    // char list at chCursor. Enter confirms whichever is highlighted.
    int phraseHl = -1;
    int chCursor = 0; // cursor within positions[segFocus]
    // Model-ranked 2-char phrase candidates per focus position, fetched
    // lazily from the daemon while Choosing.
    std::map<int, std::vector<std::pair<int, std::string>>> phraseCands;

    // Seed selections from the best decoded sentence; focus lands on
    // pendingFocus (↓ at the cursor's segment) or the first ambiguous one.
    void begin(std::vector<std::vector<std::string>> pos,
               std::vector<std::pair<int, int>> ivals,
               std::vector<std::string> syls, std::vector<SegTok> tokens,
               const std::string &best, int pendingFocus) {
        positions = std::move(pos);
        intervals = std::move(ivals);
        syllables = std::move(syls);
        toks = std::move(tokens);
        phraseCands.clear();
        candListOpen = true;
        phraseHl = -1;
        userFixed.clear();
        segSel.assign(positions.size(), 0);
        for (size_t i = 0; i < positions.size(); i++) {
            std::string span =
                utf8CharSlice(best, intervals[i].first, intervals[i].second);
            for (size_t j = 0; j < positions[i].size(); j++) {
                if (positions[i][j] == span) {
                    segSel[i] = static_cast<int>(j);
                    break;
                }
            }
        }
        segFocus = 0;
        if (pendingFocus >= 0 &&
            pendingFocus < static_cast<int>(positions.size())) {
            segFocus = pendingFocus;
        } else {
            for (size_t i = 0; i < positions.size(); i++) {
                if (positions[i].size() > 1) {
                    segFocus = static_cast<int>(i);
                    break;
                }
            }
        }
        initialSel = segSel;
        chCursor = segSel[segFocus];
    }

    void clear() {
        positions.clear();
        intervals.clear();
        syllables.clear();
        toks.clear();
        segSel.clear();
        initialSel.clear();
        userFixed.clear();
        phraseCands.clear();
        segFocus = 0;
        phraseHl = -1;
        chCursor = 0;
        candListOpen = true;
    }

    bool empty() const { return positions.empty(); }

    std::string composedSentence() const {
        std::string out;
        for (size_t i = 0; i < positions.size(); i++) {
            int sel = (i < segSel.size()) ? segSel[i] : 0;
            if (sel >= 0 && sel < static_cast<int>(positions[i].size())) {
                const bool en =
                    i < toks.size() && !toks[i].zh && isAsciiRun(toks[i].v);
                if (en) out += " ";
                if (!en && i < toks.size() && !toks[i].zh && !out.empty() &&
                    out.back() == ' ') {
                    out.pop_back(); // fullwidth punct hugs the previous word
                }
                out += positions[i][sel];
                if (en) out += " ";
            }
        }
        return tidySpaces(out);
    }

    // Words COVERING the focused char (chewing/新注音 semantics): both the
    // (focus-1, focus) and (focus, focus+1) windows, merged by the model's
    // joint probability. Fetched lazily; cached per focus.
    const std::vector<std::pair<int, std::string>> &ensurePhrases() {
        if (!phraseCands.count(segFocus)) {
            std::vector<std::string> syls;
            int focusInRun = -1, runStartTok = 0;
            for (int k = 0; k < static_cast<int>(syllables.size()); k++) {
                if (syllables[k].empty()) {
                    if (k > segFocus) break;
                    syls.clear();
                    runStartTok = k + 1;
                    continue;
                }
                if (k == segFocus) focusInRun = static_cast<int>(syls.size());
                syls.push_back(syllables[k]);
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
                      [](const auto &a, const auto &b) {
                          return a.first > b.first;
                      });
            std::vector<std::pair<int, std::string>> out;
            for (auto &[p, sp] : merged) {
                if (out.size() >= 8) break;
                out.push_back(sp);
            }
            phraseCands[segFocus] = std::move(out);
        }
        return phraseCands[segFocus];
    }

    // Re-decode the sentence conditioned on the user's picks (hint-aware
    // model) and update the segments the user has NOT touched; the reply's
    // hint-conditioned candidate ranking replaces the lists (later ↓ lists
    // reflect the picks).
    void rescore() {
        // pure-zh sentences only (position mapping is 1:1 with the daemon)
        for (const auto &s : syllables) {
            if (s.empty()) {
                return;
            }
        }
        if (userFixed.empty()) {
            return;
        }
        std::map<int, std::string> hints;
        for (int i : userFixed) {
            if (i >= 0 && i < static_cast<int>(positions.size())) {
                hints[i] = positions[i][segSel[i]];
            }
        }
        json full;
        auto sentences = queryDecoderWithHints(syllables, hints, &full);
        if (sentences.empty() || !matchesPositions(sentences[0], positions)) {
            return; // daemon down / non-hint model: keep current selections
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
                        // remap current selections to the new ordering first
                        for (size_t i = 0; i < positions.size(); i++) {
                            const std::string cur = positions[i][segSel[i]];
                            for (size_t j = 0; j < r[i].size(); j++) {
                                if (r[i][j] == cur) {
                                    segSel[i] = static_cast<int>(j);
                                    break;
                                }
                            }
                        }
                        positions = std::move(r);
                        phraseCands.clear(); // phrase ranks are stale too
                    }
                }
            } catch (const std::exception &) {
            }
        }
        // adopt the re-scored chars for every segment the user hasn't touched
        for (size_t i = 0; i < positions.size(); i++) {
            if (userFixed.count(static_cast<int>(i))) {
                continue;
            }
            std::string span = utf8CharSlice(sentences[0], intervals[i].first,
                                             intervals[i].second);
            for (size_t j = 0; j < positions[i].size(); j++) {
                if (positions[i][j] == span) {
                    segSel[i] = static_cast<int>(j);
                    break;
                }
            }
        }
    }

    // Number key / click on a char candidate. chewing: the pick CLOSES the
    // candidate window; the cursor stays put. 新注音-style: the pick
    // re-scores the rest.
    void pickSegment(int candIdx) {
        if (segFocus < 0 || segFocus >= static_cast<int>(positions.size())) {
            return;
        }
        if (candIdx >= 0 &&
            candIdx < static_cast<int>(positions[segFocus].size())) {
            segSel[segFocus] = candIdx;
            userFixed.insert(segFocus);
            rescore();
        }
        candListOpen = false;
        chCursor = segSel[segFocus];
    }

    // ⇧1-9 / Enter on a highlighted 詞: set both chars of the 2-char phrase,
    // re-score, close the window; focus stays on the word.
    void pickPhrase(int start, const std::string &phrase) {
        const int i = start;
        if (i < 0 || i + 1 >= static_cast<int>(positions.size())) {
            return;
        }
        size_t c0len = utf8SeqLen(phrase[0]);
        std::string c0 = phrase.substr(0, c0len);
        std::string c1 = phrase.substr(c0len);
        auto setSel = [this](int pos, const std::string &ch) {
            const auto &cands = positions[pos];
            for (size_t j = 0; j < cands.size(); j++) {
                if (cands[j] == ch) {
                    segSel[pos] = static_cast<int>(j);
                    return;
                }
            }
        };
        setSel(i, c0);
        setSel(i + 1, c1);
        userFixed.insert(i);
        userFixed.insert(i + 1);
        rescore();
        segFocus = i;
        candListOpen = false;
        phraseHl = -1;
        chCursor = segSel[segFocus];
    }

    // ←→ with the window open: walk 詞 chips + chars as ONE wrapping loop
    // (deliberate deviation from chewing's ←→=move-focus, documented in the
    // parity waivers). dir = +1 (→) or -1 (←).
    void moveHighlight(int dir) {
        const int nph =
            static_cast<int>(std::min<size_t>(ensurePhrases().size(), 9));
        const int nch = static_cast<int>(positions[segFocus].size());
        const int total = nph + nch;
        if (total <= 0) {
            return;
        }
        int ci = phraseHl >= 0 ? phraseHl : nph + chCursor;
        ci = (ci + dir + total) % total;
        if (ci < nph) {
            phraseHl = ci;
        } else {
            phraseHl = -1;
            chCursor = ci - nph;
        }
    }

    // ←→ with the window closed: move the focused segment to the next/prev
    // ambiguous one (chewing).
    void moveFocus(int dir) {
        const int nseg = static_cast<int>(positions.size());
        for (int i = segFocus + dir; i >= 0 && i < nseg; i += dir) {
            if (positions[i].size() > 1) {
                segFocus = i;
                phraseHl = -1;
                chCursor = segSel[i];
                break;
            }
        }
    }

    // ↓ with the window closed: reopen at the focused char.
    void reopen() {
        candListOpen = true;
        phraseHl = -1;
        chCursor = segSel[segFocus];
    }

    // Enter with the window open: confirm the highlight (詞 or 字).
    void confirmHighlight() {
        const auto &ph = ensurePhrases();
        if (phraseHl >= 0 && phraseHl < static_cast<int>(ph.size())) {
            auto pick = ph[phraseHl];
            pickPhrase(pick.first, pick.second);
        } else {
            pickSegment(chCursor);
        }
    }

    // Learn payload for the user's corrections (changed zh segments +
    // adjacent changed pairs as phrases), built at commit time.
    json learnPayload() const {
        json chars = json::array(), phrases = json::array();
        for (int i : userFixed) {
            if (i < 0 || i >= static_cast<int>(segSel.size()) ||
                i >= static_cast<int>(syllables.size()) ||
                syllables[i].empty()) {
                continue;
            }
            chars.push_back({syllables[i], positions[i][segSel[i]]});
            if (userFixed.count(i + 1) &&
                i + 1 < static_cast<int>(segSel.size()) &&
                i + 1 < static_cast<int>(syllables.size()) &&
                !syllables[i + 1].empty()) {
                phrases.push_back({syllables[i] + " " + syllables[i + 1],
                                   positions[i][segSel[i]] +
                                       positions[i + 1][segSel[i + 1]]});
            }
        }
        return json{{"chars", chars}, {"phrases", phrases}};
    }
};

} // namespace slothing

#endif // _SLOTHING_COMMON_CORE_H_
