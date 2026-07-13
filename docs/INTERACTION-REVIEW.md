# Interaction review: Slothing vs chewing / 微軟新注音 / 自然 / 華碩智慧

A differential review of Slothing's **interaction contracts** against the four
reference 注音 IMEs, grounded in the actual code (`engine/common/core.h`,
`core_test.cpp`, `android/app/cpp/session.h`, `engine/fcitx5-chewing/src/eim.cpp`)
and in the cross-frontend parity harness (`eval/ui-parity/`). This is about
*behaviour* — what a key does and what the UI shows — not decode accuracy (that
is `eval/`, `slothing-real-benchmark`).

Verdicts: **=** matches the references · **~** intended divergence (waiver) ·
**△** real gap / recommendation.

## The reference IMEs

| IME | Input model in one line |
|---|---|
| **libchewing** | Classic bopomofo: tone finalizes a syllable, modal ↓ candidate window, Shift = English passthrough. Open source — the behavioural floor. |
| **微軟新注音** | Modeless intelligent conversion (免選字): type a whole reading, it converts; ↓ / space to correct; fullwidth Chinese punctuation; Shift+Space fullwidth toggle; Shift = English passthrough. |
| **自然輸入法** | Intelligent conversion in the 微軟 mould, historically the strongest 免選字; fullwidth-Chinese / halfwidth-English punctuation convention. |
| **華碩智慧輸入法** | 注音/英文**混合模式 — 免按 Shift 中英混打** (auto zh/en switching), autocomplete, 刪除贅字, Shift+Space fullwidth toggle. The closest sibling to Slothing's model. |

The single most important observation: **Slothing's headline behaviour — auto
zh/en code-switching without a mode key — is not a divergence from the field, it
is exactly 華碩智慧's model.** Slothing sits in the 微軟/自然/華碩 "intelligent,
modeless" lineage, using chewing only as the low-level keying/candidate floor.

## Dimension-by-dimension

### Composing & segmentation
- Bopomofo run accumulates raw keys; the segmenter re-decides zh/en live. **=** chewing keying.
- **Auto zh/en mixing** (invalid-syllable runs → English; `world cup` stays English). **=** 華碩; **~** superset vs chewing/微軟/自然 (which need a Shift mode). Documented waiver in the parity suite.

### Tone & space
- Tone keys `3/4/6/7` finalize the syllable; **space = tone 1** (bare base). **=** chewing/微軟 (space as tone-1/convert is conventional).
- Space between two English words is kept literal (faithful, no CJK-Latin autospace). **=** 自然/華碩 halfwidth-English convention. See `docs/ZH-EN-MIXING.md`.

### Modeless conversion (免選字)
- Type a whole reading; the model converts without opening a window; ↓ only to correct. **=** 微軟/自然/華碩 (the whole point). **△** vs chewing, which is more per-syllable — an intended *improvement*, and the real accuracy bar (see `slothing-real-benchmark`).

### Candidate window (↓)
- ↓ opens a modal window on the segment at the cursor; a pick **closes** the window (no auto-advance); Esc closes; typing/Enter ignored while modal. **=** chewing semantics (implemented from `chewingio.c`, not guessed — `core_test.cpp`).
- **One deliberate deviation:** ←→ move the **highlight** and Enter confirms it (微軟 convention). Our candidate list is horizontal, so arrows must match the visual axis; chewing 0.5 paged with ←→ and had no movable highlight. **~** waiver.
- Numbers select, space pages. **=** chewing.

### Backspace granularity
- Pops one bopomofo symbol of the pending run first; only once the run is empty does it remove a finalized token. **=** chewing (verified `core_test.cpp` + parity `backspace pending` / `backspace syllable`).

### Cursor / arrows
- Arrows ignored while a syllable is pending (chewing); Home/End; mid-buffer editing re-decodes with a no-flicker stale display. **=** chewing/微軟. Parity: `cursor home/end`, `mid-cursor typing`, `cursor left+cand` all pass.

### Esc
- Two-level: in the candidate window, Esc closes the window; in composing, Esc clears only the pending bopomofo, never converted text. **=** chewing. Parity: `esc clears`, `cand esc closes`.

### English mode (lone Shift)
- **The 3 C++ frontends (fcitx5, IBus, Android) PASSTHROUGH** each key to the app — no preedit, no Enter. **=** 微軟/chewing/自然/華碩 (all passthrough in their explicit English mode).
- **The web demo BUFFERS** English into the preedit (editable, commit on Enter). **~** intended, **web-demo-only** divergence — the demo teaches the composing model, so showing English in the buffer is pedagogically clearer. Tracked as a waiver in `compare_core_web.py` (`<S>hi<E>` etc.).
- Note this is *forced* English (the Shift mode). The everyday path is the auto-mixing above, which needs no mode key at all — matching 華碩.

### Punctuation width
- Clause-aware: fullwidth after Chinese, halfwidth in a pure-English clause; a comma after English but in a Chinese-led clause is fullwidth (我推薦 Python，因為…). **=** 自然/華碩/微軟 convention (fullwidth-Chinese / halfwidth-English), and finer than chewing's flat mapping. Verified in `core_test.cpp` (`punctMark`).
- **Shift+Space toggles 全形/半形.** **=** 微軟/華碩 (explicitly the same shortcut).

### Symbol menu
- `` ` `` opens a categorized symbol menu (1–9 select, ←→ category, Esc close). **=** 微軟/chewing symbol tables.

### 聯想 (next-word prediction) & learning
- After a commit the strip flips to predictions (personal bigrams first, then a dictionary); picks chain and are learned; corrections in the ↓ window are learned as char/phrase fixes. **=** 微軟/自然/華碩 (autocomplete + 自動選字 learning). Verified: `core_test.cpp` AssocEngine block.

## Findings

**No behavioural gaps against the reference IMEs surfaced.** Every dimension
either matches the references or is a documented, intentional divergence. The
parity harness (`15/15`) shows the four frontends already agree with each other
on all of the above except the web-only English buffering.

Intended divergences (all waived/tracked):
1. Auto zh/en mixing as the *default* (superset of chewing/微軟/自然; native to 華碩).
2. ←→ move the candidate highlight (微軟 axis convention, vs chewing paging).
3. Web-demo-only English buffering (the 3 shipping frontends passthrough like the references).

Recommendations (optional, not gaps):
- **△ 華碩-style 刪除贅字 / 框選查詢** — 華碩 advertises "delete redundant chars" and select-to-query. Slothing has learning + the ↓ window but no explicit "select a committed span and re-query" gesture. If a future desktop gesture is wanted, that is the reference to match. Low priority; the modeless path already covers most of it.
- **△ Decide whether English buffering should ever reach the shipping frontends.** Today it is web-only by choice. If users ask for editable English on desktop/mobile, `core.h`'s enMode passthrough (`eim.cpp:1171`, `session.h:129`, `main.cpp`) is the single place to change, and the parity waiver would flip to a strict contract. Currently: **keep passthrough** (matches all four references).
