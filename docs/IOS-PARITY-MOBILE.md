<!-- Generated 2026-07-14 via the gpt-ime-design workflow (7-agent: inventory web/android/core -> spec).
     Companion: docs/GPT-IME-DESIGN.md (the LLM redesign that unblocks the NEEDS-GPT rows). -->

# Sloth IME Mobile — iOS-Parity Feature Specification

Deliverable #1. Target frontends: web-touch (`space-static/`, the `iosmode`/`iosKb` path) and Android native (`android/app/…`). Reference: iPhone 注音 mobile behavior (13 observed behaviors). Every file:line below is cross-referenced from the current inventories.

---

## Goal

Make Sloth IME on **web-touch and Android** reproduce the iPhone 注音 (iOS Zhuyin) mobile interaction end to end: the editor field shows the **raw space-separated bopomofo you typed** (verify-your-phonetics, not eager conversion), while a **single horizontal candidate bar sitting directly on the keyboard leads with the full-sentence 免選字 conversion** and then flows into **one unified, relevance-ranked stream that intermixes single chars, multi-character words, phrases, emoji, and next-word predictions on a single probability scale** — no separate 字/詞 group labels. The bar tolerates **toneless / mixed-tone input** (音標皆可), offers an **expand chevron** for more candidates, a **context-aware space** (一聲 ↔ 下一個), **tap-to-fix any syllable**, inline **zh/EN code-switch**, and **prediction/association folded into the same bar** both mid-sentence and post-commit. Android always uses this touch model; web uses it whenever the interaction model is `ios` (auto-detected by last input, or forced). Parity is judged against iOS, then re-measured in `eval/ui-parity`.

---

## Parity matrix

Legend for "Where to change": **W**=web (`space-static/ime.js`, `index.html`), **A**=android (Kotlin `…/ime/*.kt`, native `android/app/cpp/session.h`, `jni_sloth.cpp`), **C**=shared core (`engine/common/*.h`, `engine/slothd/slothd_e.py`), **M**=needs the new GPT/decoder model.

| # | Feature | iOS behavior | Web status | Android status | Gap | Where to change |
|---|---|---|---|---|---|---|
| 1 | **Raw bopomofo field** | Field shows raw space-separated syllables you typed, underlined; conversion is NOT eagerly in the field | DONE — preedit paints `tok.v` with `.bopo` class, ime.js:294-296; `.iosmode .pre .bopo` index.html:163 | DONE — `paintComposingRaw()`→`core.getComposingBopo()` SlothImeService.kt:638-641; native builds it session.h:369-382 | Web missing underline styling to match iOS; otherwise parity | W (CSS underline), else none |
| 2 | **Candidate bar on top of keyboard** | One horizontal scroll row directly on the keys | PARTIAL — mounted on `#kb` only for `iosKb` layout, ime.js:676-680; on non-`iosKb` touch (tablet >600px) strips stay in `.ime` panel index.html:217-218 | DONE — `CandidateBar` single-row `HorizontalScrollView` CandidateBar.kt:25 | Web: bar-on-keys is layout-gated, not model-gated | W (mount by `uiMode()==='ios'`, not `iosKb`) |
| 3 | **Full-sentence-first candidate** | Bar LEADS with whole-run conversion, boxed/highlighted | DONE — headline pill `mk(full,…,' hlp')` ime.js:324-330; `commitPrefix`/`commitSentence` :485-516; `.cand.ph.hlp` index.html:95,173 | DONE — `getLiveSuggestions()[0]` rendered selected CandidateBar.kt:108-117; native n-best session.h:307-315 | Parity | — |
| 4 | **Progressive shorter fallbacks** | Full sentence → shorter prefixes → single chars → alternatives | PARTIAL — prefix pills at each zh boundary, hard-capped at 3 (`stops.slice(0,3)`) ime.js:331-334,502-516 | PARTIAL — sentences OR last-word chars (either/or) SlothImeService.kt:590-592 | Web cap of 3; both: fallbacks aren't part of ONE ranked stream | W, A + §3 |
| 5 | **UNIFIED relevance-ranked stream** | One ordered list intermixing 字 + 詞 + phrase + emoji on one probability scale; NO 字/詞 group labels | MISSING — `#cands` ('字') and `#phrases` ('詞') are separate labeled groups ime.js:336-343,366-386,405-408 | MISSING — 詞 row, divider, 字 row segregated CandidateBar.kt:78-98 | **The biggest gap** — see §3 | W, A (render); C optional accessor; M for true joint scoring |
| 6 | **Emoji in stream** | Emoji suggestions inline on the same scale (🌲 📖 after 樹/書) | MISSING — `SYMBOLS` is punctuation/math/shape only ime.js:46-54; emoji only decorative labels :655 | MISSING — `SYMBOLS` CJK punctuation only SlothImeService.kt:59-62 | No emoji candidates anywhere | W, A + data (§4); M for model-native emoji |
| 7 | **Expand chevron ˅** | Chevron reveals more candidates (expand row → grid) | MISSING — overflow is horizontal-scroll only; only ‹/› paging in fix mode ime.js:389,398 | MISSING — one-row scroller, only ‹ › focus chips CandidateBar.kt:69-76 | No expand-to-grid affordance | W, A frontend-only |
| 8 | **Space = 一聲 / 注 watermark** | Space labeled 一聲 with faint 注 watermark | DONE — `iosKb` space labeled 一聲 + 注 watermark ime.js:730; `.sp .spw` index.html:182-184 | DONE — 「空白（一聲）」 strings.xml:19; long-press→choosing KeyboardView.kt:166-172 | Parity on label | — |
| 9 | **Space becomes 下一個 when choosing** | When a conversion/choosing is active, space cycles candidates (下一個); → commits | MISSING — touch space only feeds tone-1, inert in fix mode ime.js:730 | MISSING — `onKey` swallows non-digits while CHOOSING SlothImeService.kt:250-253 | Context-aware space not wired | W, A frontend-only (native `moveHighlight` session.h:620 exists) — §5 |
| 10 | **Toneless tolerance (音標皆可)** | Mixed marked/unmarked syllables both resolve (ㄋㄧㄏㄠㄇ→你好嗎) | MISSING — unmarked = tone-1 row only, no union ime.js:546-547; hint `聲調可省略` :410 overstates | MISSING — `toneOrSpace` demands a tone/space session.h:179-193 | Toneless collapse (呢嚆); tone-1 vs toneless indistinguishable | C (decoder/segmenter + tone-union) + W mirror; M ideal — §6 |
| 11 | **Prediction / association (mid + post-commit)** | Next-word suggestions interleaved into the SAME bar, both after commit and mid-sentence | PARTIAL — `assoc.js` shows 聯 chips on empty buffer after commit ime.js:346-359; NOT mid-sentence, separate label | PARTIAL — `showPredictions` post-commit SlothImeService.kt:657-679; not mid-sentence, own 聯 label | Post-commit only; not interleaved mid-sentence; not merged into unified stream | W, A (merge into §3); M for true outside-span generation — §7 |
| 12 | **Fix a character** | Tap a syllable/candidate → rich single-char list for it | DONE — every preedit char `onclick`→`openFix(i)` ime.js:297; `.pchar.fixsel` index.html:79 | PARTIAL — auto in-strip only for LAST word (`getLastWordCands` session.h:396-401); earlier chars need ↓ window | Android: mid-sentence fix needs the ↓ window, not a tap | A + C (per-position accessor) |
| 13 | **zh-EN code-switch inline** | Inline English preserved (world cup, louisa) mixed with 注音 | DONE — DP re-segmentation `segment(rawKeys)` segment.js:33-148; ime.js:72-73; Shift toggles English :281 | DONE — segmenter re-decides per key session.h:163-171; dual-label keycaps KeyboardView.kt:91-93 | Parity | — |
| — | **Bopomofo composing tail (live segment)** | (implied by #1) | DONE — per-token `.v` ime.js:294-296 | DONE — live-segmented raw tail session.h:369-382 | Parity | — |
| — | **Char (字) candidates for a syllable** | (backing #5/#12) | DONE — top-6 last-word strip, model-ranked ime.js:366-386,620-621; paged fix strip :387-402 | DONE — last-word chips CandidateBar.kt:127-144; numbered choosing chips :88-98 (`kLastCands=8`) | Data exists; ranking/merge is the gap (#5) | — |
| — | **Word/phrase (詞) candidates** | (backing #5) | PARTIAL — encoder 2-char phrases `buildPhrases` ime.js:171-205; shown ONLY on syllable-tap, not while composing :336-343 | DONE (in choosing) — `getPhrases()` chips CandidateBar.kt:78-86; session.h:775-783 | Web: 詞 not proactive during composing; both: not co-ranked with 字 (#5) | W (proactive), then §3 |
| — | **Backspace edits committed output** | (implied by tap-to-fix / editing) | DONE — `doBackspace()`→`backspaceOutput()` edits textarea caret ime.js:134-145; ⌫ wired :685-692 | MISSING — empty-buffer ⌫ only forwards `KEYCODE_DEL` SlothImeService.kt:292-298; core returns Ignored session.h:197-201 | Android can't reclaim a committed word into composing | A + C ("reclaim last commit" surface in session.h) |
| — | **Symbol strip (符)** | (adjacent to emoji, #6) | PARTIAL — `SYMBOLS` punctuation menu with categories ime.js:46-54 | PARTIAL — flat `SYMBOLS` row; native `symbolCats()` exists but Kotlin renders flat SlothImeService.kt:318-323; session.h:145-148 | Android: no category menu; both: no emoji category (#6) | A (categories), W/A emoji (§4) |
| — | **Adaptive model + indicator** | (web-only; iOS is always touch) | DONE — `modeind` AUTO/桌面/行動, `uiMode()` gates branches ime.js:650-651,661-672 | N/A — Android always touch | Web-only extra; keep | — |
| — | **Hardware/BT keyboard path** | (no iOS mobile equivalent) | (desktop model) | DONE — full parallel `handleHardwareKey` SlothImeService.kt:401-561 | Android-only extra; keep | — |

---

## The unified candidate bar (the biggest gap)

**Target (iOS #5).** One horizontal, relevance-ranked stream. For `ㄕㄨˋ`: `是 說 上 ⬆️ 聲 收 剩 樹`; after picking 樹: `樹 🌲 屬 書 📖 上午 食物`. Single chars, multi-char words, phrases, and emoji share **one score axis**. No `字`/`詞`/`聯` section labels in the primary row (labels may survive only as tiny type-badges if desired, but ordering is global).

### 3.1 What replaces the current groups

Today both frontends keep **two-to-three physically separate rendered lists**: web `#phrases` (sentence + prefix pills + 詞) and `#cands` (字), reordered but never co-ranked (ime.js:324-343,366-408); Android renders 詞 row → divider → 字 row (CandidateBar.kt:78-98) or sentences-vs-chars either/or (SlothImeService.kt:590-592). Replace both with a **single ordered item list** `UnifiedCand[]` that the renderer walks in order.

```
UnifiedCand {
  text        // "樹", "上午", "這是樹懶輸入法", "🌲"
  kind        // SENTENCE | PREFIX | WORD | CHAR | EMOJI | PREDICT | SYMBOL
  span        // how many input syllables this consumes if committed (SENTENCE/PREFIX/WORD/CHAR)
  score       // single comparable float (log-prob domain)
  commit()    // kind-specific commit action (see below)
}
```

### 3.2 Ranking formula (sketch)

Work in **log-probability** so heterogeneous sources are additive and comparable. For a candidate covering input syllables `s_i … s_j`:

```
score(c) = Σ_{k=i..j} logP(char_k | context)          // sequence log-prob over its span
         + α · lengthBonus(span)                        // reward covering more input (免選字 bias)
         + β · typeBias(kind)                           // small per-kind nudge, see below
         + γ · learnBonus(c)                            // personalization (slothd_e.py:139-173)
         + δ · assocBonus(c | committedTail)            // bigram/context boost (assoc.h)
```

- **Encoder-era approximation (UNBLOCKED today):** `logP(char_k)` = the masked-logit log-softmax the decoder already computes per position (`slothd_e.py:241-256`). A WORD's score = `Σ` of its positions' log-probs (the same `P(c0)·P(c1)` join as `buildPhrases` ime.js:171-205 / `phrases()` slothd_e.py:258-289, just in log space). A SENTENCE = `Σ` over all positions (the argmax path). This gives a **usable single axis immediately** without the GPT model — its known weakness is that the encoder's per-position marginals are independent, so long spans are over-confident; `α·lengthBonus` is hand-tuned to compensate.
- **GPT-era (NEEDS new model):** `logP` becomes a **true joint** autoregressive sequence probability from a constrained beam (the REPLACE items: joint search over the phonetic mask, real k-best). Same formula, honest scores, `α` shrinks toward 0. This is the clean long-term home.
- **`lengthBonus(span)`** — monotonic in span so the full sentence and long words out-rank their own sub-prefixes when the phonetics are unambiguous, matching iOS leading with the sentence. Tune so a shaky long span still yields to a confident shorter one.
- **`typeBias(kind)`** — near-zero. Only breaks near-ties (e.g. nudge EMOJI slightly **down** so it rides just behind its anchor word — 樹 then 🌲 — never leads; nudge SENTENCE up a hair to guarantee the headline).
- **`assocBonus`** folds prediction into the SAME axis (see §7).

### 3.3 Ordering, headline, and dedup rules

1. **Headline:** the highest-scoring SENTENCE covering the whole current run is pinned to slot 0 and rendered boxed/highlighted (keep `.hlp` pill ime.js:324-330 / selected-style CandidateBar.kt:108-117). This is a **presentation pin**, not a score override.
2. **Body:** everything else sorted by `score` descending — SENTENCE alternates (n-best), PREFIXes, WORDs, CHARs, EMOJI, PREDICTs interleaved purely by score. Remove the hard prefix cap of 3 (ime.js:334); let score + the expand chevron govern length.
3. **Dedup by surface text**, keeping the highest-scoring instance and its richest `kind`/`span` (a CHAR 樹 and the WORD-initial 樹 collapse; prefer the one whose commit consumes more input). Dedup is global across kinds.
4. **Emoji anchoring:** an EMOJI is inserted immediately after the first surviving candidate whose text is its anchor word (post-dedup), overriding pure score by at most one slot, so 🌲 sits next to 樹 (§4).
5. **Stable-ish:** avoid reshuffling already-visible leading items on minor rescore to prevent the bar "jumping" under the thumb; only the tail past the fold re-sorts freely.

### 3.4 Expand chevron (#7)

Add a trailing `˅` affordance. Collapsed = current single scroll row (first N by score). Tap → expand the SAME ordered list into a multi-row grid (wrap instead of horizontal-scroll), chevron flips to `˄`. No new data — both frontends already expose the full lists (`getCandidates`/`getPhrases` on Android; the full arrays in ime.js). Pure frontend: a CSS class toggling `.candbar` from `overflow-x:auto;flex-nowrap` (index.html:150-160) to `flex-wrap:wrap` on web; swap `HorizontalScrollView` for a wrapping `FlexboxLayout`/grid height on Android (CandidateBar.kt:25).

### 3.5 Mapping to each frontend's candidate API

- **Web:** build `UnifiedCand[]` in ime.js by merging the four existing producers — sentence/prefix pills (:324-335), last-word 字 (:366-386, model-ranked :620-621), `buildPhrases` 詞 (:171-205), and `assoc.predictions()` (:346-359) — into one array, score each (§3.2), sort, dedup, then render one list into a single container. Retire the `#phrases`/`#cands` split and the fix-mode reorder (:405-408) in favor of one ranked render; `commit()` dispatches to the existing `commitPrefix`/`commitSentence`/char-pick/`pickPredict` by `kind`.
- **Android:** two clean options. (a) **Frontend-only merge** in `CandidateBar.kt`: pull sentences (`getLiveSuggestions`), last-word chars (`getLastWordCands`), 詞 (`getPhrases`), predictions (`getPredictions`) up into Kotlin, score+sort+dedup, render one list — replacing the segregated rows (CandidateBar.kt:78-98). (b) **Cleaner:** add ONE native accessor `getUnifiedCands()` to `session.h` (beside :387-410) that emits scored `UnifiedCand`s so web-parity logic lives in shared core and both mobile frontends stay thin. Recommend (a) first (fast parity), migrate to (b) when the GPT decoder lands so joint scores are computed once in C++.

---

## Emoji suggestions

**Behavior (iOS #6/#12):** emoji ride inline in the unified stream, anchored to a just-produced or candidate word (樹→🌲, 書→📖), on the same score axis, insertable like any candidate.

**Two implementation tiers:**

1. **Word→emoji map (UNBLOCKED, ship first).** A static bilingual table `model/emoji_map.tsv`: `anchor_word<TAB>emoji[ emoji…]` (e.g. `樹\t🌲`, `書\t📖 📚`, `貓\t🐱`, `聲\t🔊`, plus the arrow glyphs iOS shows like ⬆️ for 上). Build from an open CLDR/emoji-annotation zh-Hant list filtered to high-frequency anchors. At render time, for each surviving CHAR/WORD candidate `w`, look up `emoji_map[w]`; emit an `EMOJI` `UnifiedCand` with `score = score(w) + β_emoji` (β slightly negative) so it lands right after its anchor (§3.3 rule 4). `span=0` (emoji consumes no input; it's an insertion, not a conversion of the syllables). Commit = insert the emoji glyph at the output caret, then continue composing the still-pending run (like a symbol insert, session.h:461-474 / `insertSymbol` path).
   - **Data needed:** `emoji_map.tsv` (shared, referenced by both frontends — web fetches it like `assoc_tc.tsv`; Android bundles it as an asset). Keep it small (top few hundred anchors) for latency.
2. **Model-native emoji (NEEDS GPT model).** The `gen_data.py` JSONL already **preserves emoji verbatim in `out`** (unlike the encoder's `<en>` collapse). A GPT decoder generating outside/inside the span can surface emoji as genuine candidates with real probabilities — folding EMOJI into the same beam so no side table is needed. This is the long-term replacement for the static map; keep the map as a fallback/bias even then.

**Frontend work:** add an `EMOJI` kind to the unified renderer (§3), an emoji category to the symbol menu for manual browse (web `SYMBOLS` ime.js:46-54; Android `CandidateBar`/`SlothImeService.kt:59-62), and ensure emoji glyphs render at candidate size. No core change for tier 1; tier 2 is the model.

---

## Contextual space (一聲 ↔ 下一個) and commit / gesture model

**iOS #7/#13:** space reads **一聲** (tone-1, with 注 watermark) while typing phonetics; the moment a conversion/choosing is active it becomes **下一個** (advance to next candidate); the blue **→** commits/sends. Tapping a candidate commits it. Space cycles.

### 5.1 State machine

States (per active run):

- **COMPOSING** — user is entering syllables; field shows raw bopomofo (#1); bar shows the unified stream headed by the sentence.
- **CHOOSING** — a highlight/selection cursor is live in the candidate bar (entered by first space-tap when a run is pending, or by tapping the ↓/選字 affordance, or by tapping a syllable to fix).
- **IDLE/PREDICT** — buffer empty; bar shows post-commit predictions (#11/§7).

Space key behavior and label:

| State | Space label | Space action |
|---|---|---|
| COMPOSING, run pending | **一聲** (+注 watermark) | First tap: commit any bare/last syllable as **tone-1** AND enter CHOOSING with highlight on candidate 0 (the headline). (Matches iOS: space both closes tone-1 and starts choosing.) |
| CHOOSING | **下一個** | Advance highlight to next candidate in the unified stream (`moveHighlight(+1)`), wrapping; field/preview follows. |
| IDLE (empty buffer) | **一聲** / space | Insert literal space (or tone-1 no-op); no choosing. |

Commit / gesture:

- **Tap a candidate** → commit that candidate immediately (its `commit()` by kind; §3.1), whatever the state. (iOS #13.)
- **→ (blue)** → commit the currently highlighted candidate (in CHOOSING) or the headline sentence (in COMPOSING), then send/confirm to the field. This is the "commits/sends" affordance (#7).
- **Backspace** → edit composing tail first, else reclaim/delete committed output (web already does this, ime.js:134-145; Android needs the reclaim surface — see matrix).
- Picking a candidate that consumes part of the run (`span < remaining`) commits that prefix and **returns to COMPOSING** on the rest (existing `commitPrefix` semantics ime.js:502-516).

### 5.2 Where to wire

- **Web (frontend-only):** the touch space button is currently static 一聲 and inert in fix mode (ime.js:730). Add state to `uiMode()==='ios'`: relabel to 下一個 and route to a highlight-advance when a run/choosing is active; keep tone-1 feed for the first COMPOSING tap (`if(hasRun()) feedKey(' ')` today). Reuse the physical-space paging logic that already exists for fix mode (ime.js:807-810) but drive the on-screen label from state.
- **Android (frontend-only):** `onKey` swallows non-digits in CHOOSING (SlothImeService.kt:250-253) — wire soft-space → `core.moveHighlight(1)` (native `moveHighlight` session.h:620-623 already exists), and relabel the space keycap (`strings.xml:19`) reactively based on session state. Highlight cycling exists for hardware keys (`moveHighlight` SlothImeService.kt:460-505) — reuse it for the soft path. → key maps to `commitOrConfirm`/`confirmChoosing` (SlothImeService.kt:608-625; session.h:656-670).

No core change needed — the highlight/confirm primitives are already in `session.h`/ChoosingCore. This is UNBLOCKED by the current encoder.

---

## Toneless / 音標皆可

**Requirement (iOS #8), precise:** the decoder must accept a syllable stream where **each syllable independently may carry a tone mark or not**, and mixtures within one sentence must resolve correctly. Examples to pass: `ㄋㄧㄏㄠㄇ→你好嗎` (all toneless), `ㄨㄛㄕˋㄇㄠ→我是貓` (mixed marked/unmarked), and a long mixed string → the 有音標或沒有音標皆可 sentence. Crucially, an **unmarked syllable must mean "any tone allowed" (tone-union), NOT silently "tone-1"**, and tone-1 (explicit) vs toneless must both be honored without collapsing to the same wrong glyph (the 呢嚆 failure in MEMORY).

**Current gap:**
- **Web:** an unmarked syllable resolves to its **tone-1 row only**, no union — `candChars` comment "tone-optional removed: unmarked = tone-1" (ime.js:546-547). Space commits tone-1 (:115). The iOS-mode hint `聲調可省略` (ime.js:410) is therefore **false** and must be fixed to match behavior or the behavior fixed to match it (the latter).
- **Android/core:** `toneOrSpace` **demands** a tone mark or space (session.h:179-193); decode consumes exact toned syllables from `comp_.toks` (session.h:298-333). No tone-agnostic path exists in the segmenter or decoder.
- **Reference decoder** already has the right primitive: `slothd_e.py:83-90` builds `self.toneless` = union of all rows sharing a tone-stripped base, selected in `cands()` when the syllable has no tone (:117-119). Same union in `eval_arch.py:19-37` and `gate_g2pw.py:40-48`. Training data already emits toned+toneless+partial variants (`prepare_data_e_g2pw.py:85-96`, `augment_toneless.py`, `gen_data.py:76`).

**The disambiguation subtlety:** because an unmarked syllable is treated as the full tone-union, explicit **tone-1 (space) and toneless become indistinguishable** to the decoder (recorded collapse). The spec: **treat both space-tone-1 and no-mark as "prefer tone-1 row but fall back to tone-union"** — i.e. the mask is the union, but the per-position ranking gives a small prior to the tone-1 row so `你好` still beats `呢嚆` when unmarked. This keeps 音標皆可 while not regressing toned accuracy.

**Fix owner:**
- **SHARED-CORE (primary):** the segmenter must stop requiring a tone mark to close a syllable (`toneOrSpace` session.h:179-193; segment.h `zhAt` hard/soft costs :102-106) and the decode path must mask against the **tone-union** for unmarked syllables (mirror `cands()` slothd_e.py:117-119 into the C++/live path session.h:298-333). This is the same change the MEMORY toneless-gap entry names.
- **WEB (mirror):** replace the tone-1-only `candChars` logic (ime.js:546-547) with the tone-union mask, and make the segmenter accept toneless bases (today pure-digit bases stay "soft", ime.js:50-53). Then the `聲調可省略` hint (ime.js:410) becomes truthful.
- **MODEL (ideal):** the warm-start RoBERTa-base encoder is already the **best toneless model measured** — 84.9 toneless vs baseline 81 (bake-off: `roberta-base + --warmstart`), +3pt from warm-start. The GPT decoder trained on the same toned/toneless/partial augmentation should match or beat it while giving joint scores. So: ship the tone-union masking fix now (works with the current encoder), and let the improved model raise the ceiling. **UNBLOCKED for the masking/segmenter fix; model swap is the accuracy lever, not a blocker.**

---

## Prediction / association

**iOS #9/#11:** next-word suggestions appear **in the same candidate bar**, both **after a commit** (continue the sentence) and **mid-sentence** (continuation of what you're still writing) — not a separate strip, not only when the buffer is empty.

**Current state:**
- Both frontends have a working post-commit path: web `assoc.js` shows 聯 chips on the **empty buffer** after commit (ime.js:346-359; `pickPredict` :519-528, chains ≤5); Android `showPredictions`→`getPredictions` (SlothImeService.kt:657-679; session.h:798-802; `AssocEngine` assoc.h). Both use `assoc_tc.tsv` + personal localStorage bigrams.
- **Gaps:** (1) predictions are a **separate labeled 聯 group**, not merged into the unified stream; (2) they only fire on **empty buffer / post-commit**, never **mid-sentence** while a run is composing.

**Spec:**
1. **Fold predictions into the unified stream (§3).** Emit `PREDICT` `UnifiedCand`s with `span=0` (insertion, not conversion) and `score = assocBaseScore + δ·assocBonus` on the same axis. Post-commit (empty buffer) they naturally dominate because there are no conversion candidates competing. Drop the standalone 聯 section; keep an optional tiny type-badge only.
2. **Mid-sentence prediction.** While COMPOSING, after the sentence headline and high-confidence conversions, append next-word `PREDICT` items conditioned on the **committed tail + current best conversion** (the context the user is mid-writing). `assocBonus` (§3.2 δ) also **re-weights conversion candidates** that agree with the running context. This is the "interleaved into the same bar" behavior.
3. **Encoder era (UNBLOCKED):** use the existing `AssocEngine.predictions()` (assoc.h:59-80) for the `PREDICT` items and as the `assocBonus` source (personal bigrams first, then dict completions). It's purely statistical and orthogonal to conversion, so it slots into the unified scorer with no model change.
4. **GPT era (cleaner):** the user's decode-position framing makes this native — **outside the input span** (past the last typed syllable) the GPT does **unconstrained free generation** = association / next-word / whole-continuation / emoji, while **inside the span** it does phonetic-constrained conversion. One autoregressive model produces both conversion and prediction as one stream, and mid-sentence prediction is just "keep generating past the cursor." At that point `AssocEngine` becomes a bias/fallback rather than the source. This is the eventual home for #9/#11 — **NEEDS the GPT model**, but the merged-bar UX above ships first on the bigram engine.

**Where to change:** web ime.js:346-359,519-528 (emit into unified list instead of own strip; add mid-sentence trigger); Android CandidateBar.kt:151-162 + SlothImeService.kt:657-679 (same). Shared core `assoc.h`/`session.h:798-802` unchanged for tier 1; GPT decoder behind the `Decoder` seam (`decoder.h`) for tier 2.

---

## Implementation plan

Ordered. Each item marked **[W]**/**[A]**/**[C]** owner, **[FE-only]** vs **[core]**, and **[UNBLOCKED]** (works with today's 25M encoder / warm-start RoBERTa) vs **[NEEDS-GPT]** (wants the new decoder).

### Phase 0 — Cheap parity wins (frontend-only, UNBLOCKED)

1. **Context-aware space (§5).** [W][A][FE-only][UNBLOCKED] — relabel 一聲↔下一個 by state; wire soft-space→`moveHighlight(1)` (native exists session.h:620). Web ime.js:730,807-810; Android SlothImeService.kt:250-253.
2. **Expand chevron (§3.4).** [W][A][FE-only][UNBLOCKED] — add `˅` toggling scroll-row↔wrapped grid. Web index.html:150-160; Android CandidateBar.kt:25.
3. **Mount candidate bar by model, not layout (matrix #2).** [W][FE-only][UNBLOCKED] — mount on `#kb` whenever `uiMode()==='ios'`, not only `iosKb` (ime.js:676-680), so tablets get the on-keys bar.
4. **Remove prefix cap of 3 (matrix #4).** [W][FE-only][UNBLOCKED] — ime.js:334; let score + chevron govern length.
5. **Fix the toneless hint or behavior mismatch (§6 stopgap).** [W][FE-only] — until §6 lands, don't show `聲調可省略` (ime.js:410) while behavior is tone-1-only.

### Phase 1 — Unified candidate bar (the headline gap, §3)

6. **Web unified stream.** [W][FE-only][UNBLOCKED] — merge sentence/prefix/字/詞/predict producers (ime.js:171-205,324-343,366-386,346-359) into one scored+deduped list; retire `#phrases`/`#cands` split and fix-mode reorder (:405-408). Encoder log-prob scoring (§3.2 tier 1).
7. **Web: proactive 詞 during composing.** [W][FE-only][UNBLOCKED] — currently 詞 only on syllable-tap (ime.js:336-343 gated on `fix>=0`); surface word candidates continuously via the unified list.
8. **Android unified stream.** [A][FE-only][UNBLOCKED] — merge `getLiveSuggestions`/`getLastWordCands`/`getPhrases`/`getPredictions` in `CandidateBar.kt` (replace :78-98 segregation and the sentences-vs-chars either/or SlothImeService.kt:590-592).
9. **(Optional) native `getUnifiedCands()`.** [A][core][UNBLOCKED] — add to session.h beside :387-410 so scoring lives in C++ shared by both mobile frontends; recommended once Phase 4 lands so joint scores compute once.

### Phase 2 — Emoji + prediction into the stream (§4, §7)

10. **Emoji map + EMOJI kind.** [W][A][FE-only + data][UNBLOCKED] — ship `model/emoji_map.tsv`; anchor emoji after their word in the unified list (§4 tier 1). Web fetch like `assoc_tc.tsv`; Android bundle asset. Add emoji category to symbol menus (ime.js:46-54; SlothImeService.kt:59-62).
11. **Predictions folded into the bar + mid-sentence.** [W][A][FE-only][UNBLOCKED] — emit `PREDICT` items into the unified list (§7 tier 1) using `AssocEngine` (assoc.h:59-80); add the mid-sentence trigger conditioned on committed tail + current conversion. Web ime.js:346-359; Android SlothImeService.kt:657-679.

### Phase 3 — Toneless / 音標皆可 (§6) and Android fix-a-char parity

12. **Tone-union masking in shared core.** [C][core] — segmenter accepts toneless bases (drop `toneOrSpace` tone requirement session.h:179-193; segment.h costs :102-106) and decode masks against the tone-union (mirror `cands()` slothd_e.py:117-119 into session.h:298-333), with a small tone-1 prior to avoid 呢嚆 collapse. **UNBLOCKED** by the current encoder (warm-start RoBERTa already best-in-class toneless).
13. **Web tone-union mirror.** [W][FE-only] — replace tone-1-only `candChars` (ime.js:546-547) with the union; accept toneless bases in the JS segmenter (ime.js:50-53); make the `聲調可省略` hint truthful.
14. **Android fix-a-char for earlier words.** [A][core-assisted] — expose per-position candidates while composing (new session.h accessor beside `getLastWordCands` :396-401) so tapping any char in `getComposingBopo` opens its list without the ↓ window (matrix #12).
15. **Android backspace-reclaim.** [A][core] — add a "reclaim last commit" surface to session.h (session.h:197-201 currently Ignored) so empty-buffer ⌫ pulls committed text back into composing (web already does this ime.js:134-145).

### Phase 4 — GPT decoder swap (accuracy + true unified scoring)

16. **Constrained-beam GPT behind the `Decoder` seam.** [C][core][NEEDS-GPT] — drop the GPT decoder in behind `decoder.h` (`decode`/`decodeWithHints`/`phrasesScored`/`learn`) with **zero UI change**. It REPLACES: per-position argmax (slothd_e.py:245-256) → joint constrained beam; single-flip n-best (:252-255) → real k-best; `<en>` passthrough → real code-switch + emoji generation; inside-span = phonetic-masked conversion, outside-span = free generation = prediction/continuation/emoji (§7 tier 2). It REUSES: phonetic mask/table (`phonetic_table.tsv` + tone-union), segmenter DP (upgradeable to model-scored lattice), corpus/g2pW alignment, learn store + hint channel.
17. **Honest unified scores.** [W][A][core] — once (16) lands, `score` in §3.2 becomes a true joint sequence log-prob; shrink `α·lengthBonus`; emoji/prediction can become model-native (drop reliance on `emoji_map.tsv` and standalone `AssocEngine`, keep as bias/fallback).

### Cross-cutting

- **Re-measure in `eval/ui-parity` after every phase** (MEMORY: parity is measured, 23/27 baseline — don't guess). Add cases for unified ranking, toneless mixtures, emoji-in-stream, and contextual-space.
- **Keep the web adaptive model** (auto/桌面固定/行動固定, ime.js:650-672) and the **Android-always-touch / Linux-always-keyboard** split; none of the above touches the desktop keyboard model.
- **What's blocked vs not, summarized:** Phases 0–3 (contextual space, expand chevron, unified bar, proactive 詞, emoji via map, predictions-in-bar + mid-sentence, toneless tone-union, Android fix/backspace parity) are **all UNBLOCKED by the current encoder** — pure frontend + a bounded shared-core toneless/segmenter change. Only the **honest joint scoring, model-native emoji, and outside-span generative prediction** genuinely **NEED the new GPT model** (Phase 4), and even those slot in behind the existing `Decoder` seam with no UI rework.