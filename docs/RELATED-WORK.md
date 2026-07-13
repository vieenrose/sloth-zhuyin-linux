# Related Work & Baselines

Prior art closest to Sloth IME, and what we take from it.

## Closest prior system — cross-multi-IME integration (TAAI 2024)

李偉安, 葉展維, 張嘉惠 (National Central University), **《基於深度學習的跨多輸入法
編輯器整合系統》**, TAAI 2024 (計畫 113-2813-C-008-024-E).

A deep-learning system that merges **注音 / 英文 / 倉頡 / 漢語拼音** into one
keystream — the same problem space as Sloth IME, and it uses the **same dachen
keystroke encoding** (大城市 = `284t/6g4`), so the numbers are directly
comparable. Three phases:

1. **IME segmentation** — four independent DNN classifiers (one-hot → 4-layer
   DNN 3000→2, ReLU, 50% dropout), 480K samples (25%/IME) + 10% injected error.
   Split points = where the fore/aft runs classify as different IMEs. IME-ID
   **F1 0.989**; segmentation recall **0.985** (clean) → **0.951** (10% error).
2. **Candidate conversion** — Trie dictionary + an **improved Levenshtein** in
   which an *adjacent-key swap* costs 1, not 2. Recall **>0.74** at 10% error.
3. **Candidate ranking** — homophone word-groups ordered by corpus frequency.

End-to-End (strict whole-sequence) accuracy **0.613**. Datasets CC100 +
Multi-News (~71 MB) with a custom keystroke converter.

Its load-bearing idea is the **「鍵盤輸入八鄰接 (8-adjacency) 錯誤假定」**: model
mis-keys as slips to the 8 physically-neighbouring QWERTY keys (key `h` →
`t y u / g h j / b n m`), used both to *simulate* training noise and to *correct*
at decode time.

## How Sloth IME differs

| | TAAI 2024 | Sloth IME |
|---|---|---|
| **Disambiguation** | candidate selection (Trie + LD + frequency) → still **選字** | legality-masked sentence LLM → **免選字** |
| **en/zh split** | 4 learned DNN classifiers | DP segmenter (`engine/common/segment.h`, [[zhuyin-wins]]) |
| **Error model** | 8-adjacency (noise + correction) | 8-adjacency typo-noise (adopted, below) + phonetic-legality mask |
| **Mixed-script typography** | not addressed | no CJK-Latin autospace, clause-based punct width ([`ZH-EN-MIXING.md`](ZH-EN-MIXING.md)) |
| **Whole-sequence metric** | E2E **0.613** (mixed multi-IME, strict) | **免選字 72 %** held-out (zh-only, whole-sentence) |

The core advance: Sloth IME **replaces the candidate-selection stage with a
legality-masked sentence model** — the thing the TAAI system still leaves to the
user (Phase 2/3 選字) is exactly what Sloth IME removes. (Metrics aren't
apples-to-apples — different tasks, sets, and strictness — but they bracket the
same regime.)

## Adopted from it — the 8-adjacency error model

Sloth IME's simulated typo-noise (`model/train_slothe_ternary.py`,
`build_typo_neighbors`) previously treated **any** edit-distance-1 syllable as an
equally likely slip — physically wrong (`ㄉ→ㄙ` was as likely as `ㄉ→ㄊ`). It now
constrains substitutions to the **QWERTY 8-neighbourhood** (`_dachen_key_adjacency`):
`ㄉ` (key `2`) can slip only to `ㄅㄆㄊㄍ` (keys `1/q/w/e`), never to the far-key
`ㄙ`. This cuts the simulated slip set from **21.8 → 8.9 candidates/syllable**,
all physically reachable — the model now trains against realistic finger-slips.
The same prior is what the [senior grouped keypad](SENIOR-KEYBOARD.md) needs for
tremor / neighbour-slip forgiveness (aging-hand error mode, §1.2).

## Other references worth tracking

- **Si et al. 2023, ReaDIN** — a Chinese benchmark with *realistic* input noise
  (typos, homophones); sub-character tokenisation. The right robustness yardstick.
- **Belinkov & Bisk 2017** — synthetic + natural noise both break NMT; motivates
  training against a faithful noise model rather than a random one.
