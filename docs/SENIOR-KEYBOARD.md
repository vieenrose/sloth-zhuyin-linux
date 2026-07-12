# 銀髮族鍵盤 — Design-Research Report

> Deep research for the queued feature: an alternative keyboard layout for
> seniors / aged users, exploiting Slothing's sentence-level LLM to absorb the
> ambiguity of a T9-style grouped bopomofo keypad. **Target device: an
> iPhone-17-class smartphone (iOS custom keyboard extension)** — NOT a tablet or
> e-ink device (updated 2026-07-12; the e-ink/BOOX profile, §1.5 & §5.5, is now
> out of scope). Every design choice is traced to a cited finding.
>
> **Unit note (iOS):** iPhone 17 / 17 Pro are **402 × 874 pt** (6.3", 460 ppi,
> 3×), Pro Max 440 × 956, Air 420 × 912. At 460 ppi/3× a point is **0.166 mm**
> (≈6.03 pt/mm), so the 66.6 mm-wide screen = 402 pt. iOS HIG's 44 pt min is a
> general-population floor, not a senior target (same caveat as Android's 48 dp).

## 0. Executive summary — the six load-bearing findings

1. **Older adults need much bigger keys than the platform minimum.** Research
   consensus: accuracy drops sharply below ~14 mm (~88 dp); reduced-dexterity
   elders need ~19 mm (~120 dp); the practical floor is ~11.4 mm (~72 dp).
   Material's 48 dp (~9 mm) and WCAG's 24 px are *floors for the general
   population*, not targets for seniors. (§1)
2. **A full 37/41-key zhuyin keyboard is geometrically impossible to make
   senior-sized on a phone.** On the 402 pt-wide iPhone 17, 10 columns → ~40 pt
   (~6.7 mm) keys, far below the 11.4 mm floor. A **3×4 grouped grid** yields
   402/3 ≈ **134 pt (~22 mm)** columns — *above* the elder-optimal 19 mm (Pro Max
   → ~24 mm). Grouping is *forced by the touch-target math*, before we even
   invoke the LLM. Height is the constrained axis on a phone, not width. (§1, §5)
3. **The thing seniors hate most about zhuyin is exactly what Slothing
   removes.** Taiwanese sources repeatedly name 同音字/選字 (homophone
   candidate-picking) as zhuyin's fatal flaw on phones; sentence-level 免選字
   conversion is the real UX bar. (§3, §4)
4. **Every Taiwanese senior already knows bopomofo and the feature-phone 3×4
   grid.** Zhuyin is universally taught first; the 九宮格 grid is muscle memory
   from Nokia-era phones and is still the *default* zhuyin keyboard on
   Galaxy/iPhone in Taiwan. A retention asset, not a new thing to learn. (§3)
5. **Remove, don't add.** Hidden gestures, long-press, mode toggles,
   double-taps, and multi-touch are the documented failure points for elders.
   The LLM lets us delete tone-selection, candidate menus, and symbol modes.
   (§1, §2, §5)
6. **LLM disambiguation of a grouped keypad is a real bet, not a slam-dunk
   (revised 2026-07-12 after measurement).** The English "9.5% CER at 3 keys"
   study does NOT transfer: Chinese stacks a homophone layer on top of the
   keystroke ambiguity. Measured on the 500-case held-out set, grouping +
   tone-drop inflates per-position candidate characters from 44 (today) to
   **722 for Candidate A** and **1,411 for B** (×16 / ×32) — see §6. That is
   still only ~9% of the 8,342-char vocab and analogous to *toneless pinyin*
   (a solved problem at full sentence context), so it is plausibly workable —
   but it requires a decoder **retrained on group-key input**, and the real CER
   is not yet measured. A is decisively the floor; B is last-resort. (§4, §6)

## 1. Aging HCI fundamentals for touch keyboards

### 1.1 Minimum touch-target size (the core number)

| Source | Population | Recommended target | In dp (≈6.3 dp/mm) |
|---|---|---|---|
| Jin, Plocher & Kiff 2007 | Older adults | Min **11.43 mm**; fewest errors at **19.05 mm** | ~72 / ~120 dp |
| Jin et al. 2007 (spacing) | Older adults | Inter-key spacing **3.17–12.7 mm** | ~20–80 dp |
| Motti/Caprani review (2016) | Older users | Optimal **≥20 mm square**; min 11.43 mm | ~126 dp |
| Duff et al. 2010 | General | **≥20 mm** (little gain beyond) | ~126 dp |
| Chen et al. 2013 | **Motor-disabled** | Accuracy keeps improving **past 20 mm** | >126 dp |
| Accuracy threshold (multiple) | Older adults | Accuracy falls below **14 mm** | ~88 dp |
| Apple HIG | General | 44 pt | — |
| Material Design | General (incl. motor) | **48 dp (~9 mm)**, spacing ≥8 dp | 48 dp |
| WCAG 2.2 SC 2.5.8 (AA) | General | 24×24 CSS px (floor) | — |
| WCAG 2.5.5 (AAA) | Enhanced | 44×44 CSS px | — |

**Takeaway:** platform minima are *floors*. For seniors, design to
**~19 mm / ~120 dp** primary keys, never below **~14 mm / ~88 dp**.
Motor-impaired users keep benefiting past 20 mm.

Sources: [Jin et al. (CORE PDF)](https://fileserver-az.core.ac.uk/download/pdf/297018872.pdf) ·
[Leitão PLoP 2012](https://plopcon.org/proceedings/plop/2012/papers/05-leitao.pdf) ·
[W3C Mobile A11y research summary](https://www.w3.org/WAI/GL/mobile-a11y-tf/wiki/Summary_of_Research_on_Touch/Pointer_Target_Size) ·
[Chen et al. 2013 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3572909/) ·
[Material accessibility](https://m2.material.io/design/usability/accessibility.html) ·
[Android target-size help](https://support.google.com/accessibility/android/answer/7101858?hl=en) ·
[WCAG 2.5.8](https://wcag22aa.org/new-criteria/target-size/) ·
[WCAG 2.5.5 Understanding](https://www.w3.org/WAI/WCAG22/Understanding/target-size-enhanced.html)

### 1.2 Error patterns specific to aging hands

- **Omissions dominate, and they're cognitive.** In Nicolau & Jorge's elderly
  touch-typing study (ASSETS 2012), *omission* is the most common error type
  among elders and is primarily cognitively caused; tremor widens the spatial
  distance between successive touches. → the keyboard must tolerate *missing*
  keystrokes, which a sentence-LLM does naturally.
  [Nicolau & Jorge, ASSETS 2012 (PDF)](http://web.tecnico.ulisboa.pt/hugo.nicolau/publications/2012/Nicolau-ASSETS-2012.pdf)
- **Tremor → accidental touches, unintended swipes, finger oscillation.**
  Continuous *sliding* motions suppress tremor oscillation better than
  discrete taps — "swabbing" (slide-to-key) reduced error rate vs tapping at
  equal satisfaction.
  [Wacharamanotham et al., *Work* 2012 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/22317077/)
- **Shaky-hand error rate is size-sensitive:** reported ~40% error at 48 dp
  buttons for shaky users vs 13% for steady, falling to 10% at 64 dp.
  **[SECONDARY — directional only, don't quote exact numbers]**
- **Long-press, double-tap, multi-touch, and time-critical gestures are
  failure points** for older adults.
  [Design for Older People, OzCHI 2024](https://dl.acm.org/doi/10.1145/3726986.3727000) ·
  [Gesture usability by age (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0747563217303254)

### 1.3 Dwell / timing tolerance

- iOS/Android **Touch Accommodations → Hold Duration** (0.25–5 s) is the
  accessibility-standard mechanism for tremor: brief/glancing contacts are
  treated as accidental.
  [SRALab iPhone tremor strategies (PDF)](https://www.sralab.org/sites/default/files/downloads/2021-01/iphone%20adaptation_education_3.pdf)
- **Design implication:** register on **touch-down with a short debounce** (or
  lift-off-within-key); never require long-press; expose an adjustable
  ignore-threshold.

### 1.4 Visual needs

- **Font size (older adults, mobile):** systematic review (Frontiers 2022):
  Yeh 2015: 14 pt optimal (tablet, 40 cm); Yeh 2020: 22 pt reading;
  Chatrangsan & Petrie 2019: 18 pt preferred; Hou et al. 2020 (Chinese):
  ~14 px search / ~17 px intensive / 17–20 px long-form. Size must scale with
  viewing distance (visual angle), and key glyphs should exceed reading-text
  minima. [Frontiers review 2022](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.931646/full)
- **Contrast:** WCAG AA 4.5:1, AAA 7:1; for low-vision elders aim **≥7:1**,
  sans-serif, line-height ~1.5.

### 1.5 E-ink (BOOX) constraints — ⚠️ OUT OF SCOPE (target is iPhone)

*Retained for reference only; the target device is now an iPhone-17-class OLED
smartphone, so e-ink latency/ghosting/contrast constraints no longer drive the
design. iPhone realities (OLED, Taptic Engine, ProMotion) replace these — see
§5.5.* Original e-ink notes: fewer/larger keys reduce redraw regions; no
animation; static high-contrast black-on-white.

### 1.6 Cognitive load

- Replace unclear gestures with explicit, text-labeled buttons; hide
  non-essential controls; one clear path.
  [Design for Older People 2024](https://dl.acm.org/doi/10.1145/3726986.3727000)
- **Mode switching is a principal difficulty factor** → minimize keyboard
  modes. [UI-ability evaluation (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0003687020301654)

## 2. Existing senior-focused keyboards & products

| Product | What it changes | Reported gaps |
|---|---|---|
| **BIG Launcher** | Huge high-contrast buttons, SOS, caregiver lock — but **no own keyboard**; suggests speech-to-text | Text entry unsolved by launchers. [AFB review](https://afb.org/aw/14/11/15740) |
| **Big Keys / large-keyboard apps** | Super-sized glyphs, fewer keys, high contrast | English-centric; no sentence intelligence |
| **iOS accessibility settings** | Display Zoom, landscape bigger keys, Show Lowercase off, Hold Duration | Coarse; nothing zhuyin-specific. [Apple](https://support.apple.com/guide/iphone/adjust-keyboard-settings-ipha7c3927eb/ios) |
| **Samsung zh-TW default** | **九宮格 3×4 zhuyin is the default** on Galaxy in Taiwan | Confirms the grid is the familiar baseline. [Samsung TW](https://www.samsung.com/tw/support/mobile-devices/why-is-my-galaxy-s23-keyboard-nine-grids/) |

**Pattern:** senior products fix launchers/contacts/SOS and **punt text entry
to the stock keyboard or voice**. No widely-used product pairs an elder-sized
grouped zhuyin grid with sentence-level disambiguation. That is Slothing's
opening.

## 3. Zhuyin input for seniors in Taiwan

### 3.1 What older Taiwanese actually use

- **Zhuyin is universal prior knowledge** — taught before characters; "if you
  can speak, you know 注音."
  [巴哈姆特 thread](https://forum.gamer.com.tw/C.php?bsn=60030&snA=663763)
- **Handwriting is strongly preferred by elders** because full-screen 手寫
  sidesteps candidate-picking. **[SECONDARY — no large quantitative TW-elder
  survey found; qualitative only]**
- **Voice is rising**; effort-expectancy (typing difficulty) pushes elders
  toward voice. [Frontiers UTAUT 2025](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1618689/full)
- **The stated pain point is candidate selection** (同音字/選字) — the single
  most important finding: Slothing's 免選字 conversion attacks elders' #1
  complaint.

### 3.2 The feature-phone 注音 grid (muscle memory)

- 九宮格 (3×4) zhuyin is the out-of-box zhuyin keyboard on Galaxy in Taiwan
  and a first-class option on iPhone.
- **Historical grouping convention:** symbols laid out in **fixed zhuyin table
  order (聲母→介母→韻母)** chunked onto number keys (1 = ㄅㄆㄇㄈ,
  2 = ㄉㄊㄋㄌ, …). Reconstructed standard 10-group split:
  `ㄅㄆㄇㄈ | ㄉㄊㄋㄌ | ㄍㄎㄏ | ㄐㄑㄒ | ㄓㄔㄕㄖ | ㄗㄘㄙ | ㄧㄨㄩ | ㄚㄛㄜㄝ | ㄞㄟㄠㄡ | ㄢㄣㄤㄥ(+ㄦ)`
  **[UNVERIFIED at per-model level — pattern verified, exact table is
  vendor-variable; validate with real ex-feature-phone elders]**
  [注音輸入法 (Wikipedia zh)](https://zh.wikipedia.org/wiki/%E6%B3%A8%E9%9F%B3%E8%BC%B8%E5%85%A5%E6%B3%95) ·
  [vChewing layouts](https://vchewing.github.io/manual/arranges.html) ·
  [Mobile01 nine-grid thread](https://www.mobile01.com/topicdetail.php?f=423&t=5117218)

## 4. Grouped-key (T9-style) zhuyin + LLM disambiguation

### 4.1 Prior art

- **T9** used a frequency-sorted dictionary, **not** an LM.
- **LMs cut ambiguous-keyboard error substantially** (Goodman et al.: 1.67–1.87×).
  [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/502716.502753)
- **The decisive datapoint — 3-Key-Input (arXiv 2606.11642, 2026):** 2–5
  physical keys, LLM decoders, English CER:

  | Keys | CER | WER |
  |---|---|---|
  | 2 | 23.3% | 33.7% |
  | **3** | **9.5%** | **12.2%** |
  | 5 | 5.4% | 6.8% |

  Biggest jump is 2→3 keys; mapping choice barely matters (<0.5 pp).
  [arXiv](https://arxiv.org/html/2606.11642)

### 4.2 Why bopomofo grouping is *easier* than English

- Syllable = strict slotted template **[initial][medial][final][tone]**
  (21+3+13+4 = 37 symbols): **positional disambiguation is free** — the
  grammar tells the decoder which group-member is legal before any LM.
- gcin's 21-key layout deliberately co-locates an initial + a final on one key
  (they rarely collide) rather than two initials.
- **Tone can be dropped entirely** (already Slothing's default), the sentence
  decoder absorbs it.

### 4.3 Existing grouped zhuyin IMEs (competitive baseline)

超注音 (statistical whole-sentence, proximity correction) · IQQI 快注音
(nine-key, tone-free) · gcin (21-key) · Gboard 注音九宮格. **Gap:** all use
statistical LMs and still expose candidate strips; none applies constraint-
decoding neural LLM over the whole sentence to make picking genuinely optional
*and* absorb grouped-key ambiguity + dropped tones + omitted keystrokes.

## 5. Design brief

**Principle: the LLM is the error-forgiveness budget.** Every ambiguity added
at the key level (grouping, dropped tones, missed keys, neighbor slips) is
spent against the sentence decoder — freeing us to make keys huge and few.

### 5.1 Candidate A — 大注音九宮格 (Big Nine-Grid) — recommended default

| | | |
|---|---|---|
| ㄅㄆㄇㄈ | ㄉㄊㄋㄌ | ㄍㄎㄏ |
| ㄐㄑㄒ | ㄓㄔㄕㄖ | ㄗㄘㄙ |
| ㄧㄨㄩ | ㄚㄛㄜㄝ | ㄞㄟㄠㄡ |
| ␣ 空白/確認 | ㄢㄣㄤㄥㄦ | ⌫ 刪除 |

- **9 phonetic group-keys + space + backspace + one utility = 12 cells**, the
  familiar 3×4, chunked in traditional table order (muscle memory, §3.2).
- **No tone keys, no required candidate menu, no symbol mode, no long-press.**
  Top result auto-commits on space.
- **Key size (iPhone 17, 402 pt wide):** column ≈**134 pt (~22 mm)** — above the
  19 mm elder-optimum; row height ≈**80–88 pt (~13–15 mm)** (height is the
  constrained axis on a phone; a 4-row grid + prediction strip at ~88 pt rows ≈
  a 420 pt keyboard, ~48% of the 874 pt screen — acceptable for an accessibility
  keyboard). **Spacing:** 8–12 pt visible gutters (tremor slips land in a gap).
- **Glyphs:** 3–4 bopomofo per key at **≥28–32 pt**, honoring Dynamic Type;
  black-on-white ≥7:1.
- **Prediction strip:** single tall (≥56–64 dp) row — top whole-sentence
  result as large text + **at most 2–3 alternatives** as big chips; selection
  optional.
- **Tremor forgiveness:** touch-down + debounce; optional slide-to-key
  ("swabbing") toggle; adjustable ignore/hold-duration (0.25–1 s).
- **Feedback defaults:** haptic ON (~100–200 ms pulse), audio optional/off,
  static press-highlight.

### 5.2 Candidate B — 四鍵大格 (Four-Zone) — accessibility profile

2×3 of enormous zones grouped by articulatory region, ~6 phonetic zones +
space + delete; on the 402 pt-wide iPhone a 2-column layout gives ≈**200 pt
(~33 mm)** zones. More per-syllable ambiguity → leans much harder on the LLM.
**⚠️ The measured ambiguity now argues against B:** grouped-keypad candidate
fan-out is ~722 chars/position for A but **~1,411 for B** (§6) — 2× worse — so B
should be a *last-resort* accessibility profile, not a primary. A is the
default. (The 9.5% CER cited for 3-Key-Input is *English* and does not transfer;
see §6.)

### 5.3 What to REMOVE (each removal cited)

| Removed | Why |
|---|---|
| Tone keys / tone selection | LLM infers tone; fewer keys → bigger keys (§1.1, §4.2) |
| Required candidate-selection | 選字 is elders' #1 zhuyin complaint (§3.1) |
| Long-press paths | documented elder failure (§1.2) |
| Mode toggles on the main path | principal difficulty factor (§1.6) |
| Swipe-typing / hidden gestures | unclear gestures fail for elders (§1.6) |
| Dense multi-row candidate scroller | cognitive overload; ≤3 large chips (§5.1) |
| Full 37/41-key board as default | forces sub-11 mm keys on phones (§1.1) |

### 5.4 Onboarding

- One-screen, large-type primer: 「整句打完,按大空白 — 不用選字」; a
  caregiver-shareable video; numbered large-text steps.
  [Onboarding elders](https://digitalscientists.com/blog/onboarding-elderly-users-5-quick-tips/)
- Training pays off in the first days (differences wash out by day ~4–5) —
  invest up front to prevent early rejection.
  [Learnability (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5945986/)
- **Voice as first-class fallback:** large mic button beside space.

### 5.5 iOS platform realities (custom keyboard extension)

The target is an **iOS custom keyboard extension** (App Extension) — a frontend
Slothing does not yet have (current: fcitx5, IBus, Android, web). The extension
sandbox drives several hard constraints:

- **Memory limit (~60–70 MB, jetsam-enforced).** A keyboard extension is killed
  if it exceeds it. **This is where Slothing's tiny-model bet pays off:** the
  12 M int8 (~12 MB) and the ternary (~8 MB) load in-process with room to spare,
  where a multi-hundred-MB LLM would be killed outright. On-device decode via
  Core ML or ONNX-Runtime-iOS, in-process (no daemon) — mirrors the Android
  Route-B design.
- **Haptics & network require "Allow Full Access."** Without it a keyboard
  extension gets no Taptic feedback and no network. So the §5.1 "haptic ON"
  default must either be gated behind a Full-Access request in onboarding, or
  ship default-off with a prompt. Never make core typing depend on Full Access.
- **Feedback:** Taptic Engine (`UIImpactFeedbackGenerator`, light/medium) with
  Full Access; static press-highlight (invert) as the always-available fallback.
- **Text sizing:** honor **Dynamic Type** for glyphs and the prediction line so
  the board scales with the user's system text-size / accessibility settings.
- **Height:** the extension owns its height — budget ~420 pt for the 4-row grid
  + prediction strip (§5.1); avoid the tiny default 216 pt keyboard.
- **No e-ink**, ProMotion 120 Hz, OLED ≥1M:1 contrast — the §1.5/§5.5-original
  ghosting/repaint constraints do not apply.

## 6. Open questions (validate before implementation)

1. **No hard quantitative TW-elder survey** of 注音 vs 手寫 vs 語音 — run a
   small in-house survey of the target cohort. **[SECONDARY]**
2. **Exact historical per-key grouping is vendor-variable** — validate the
   §5.1 grid against 2–3 real ex-feature-phone elders. **[UNVERIFIED]**
3. **3-Key-Input CERs are English — grouped-zhuyin ambiguity MEASURED
   (2026-07-12).** On the 500-case held-out set, per-position candidate
   characters (tone dropped): exact-today 44 → **Candidate A grouped 722 (×16)**
   → **Candidate B grouped 1,411 (×32)**; 99% of syllable bases lose their unique
   key-sequence (worst A key maps to 23 syllables). Decisions this fixes: **A is
   the floor, B is last-resort** (2× worse).
   **CER MEASURED (2026-07-12) — grouped-A as specified is NOT viable.** A
   decoder retrained on Candidate-A group-keys (52 classes, tone dropped, fp
   pure-CE, dim352, on the 500-case held-out set): char-accuracy peaked **50.9%**
   (ep8) then overfit to 46.7%; whole-sentence **免選字 ~4%**. Training loss
   collapsed 2.46→0.097 = memorization (a sentence's full group-class sequence is
   nearly unique). vs exact input 免選字 72% / char ~95%. The ×16 ambiguity +
   tone-drop exceeds what within-sentence context resolves. **Levers being
   tested / recommended, in order:** (a) **cross-input distillation** (teacher
   reads exact syllables, student reads groups — regularizes the memorization and
   hands over a generalizing signal; *running now*); (b) **keep one partial
   disambiguator key** (a coarse tone, or a medial split) — attacks the ambiguity
   at source (tone-drop alone was ×2.7 of the blow-up), likely *necessary* if (a)
   under-delivers. Adopt the [8-adjacency slip model](RELATED-WORK.md) for the
   tremor-error half. Repro: `train_grouped.bin`, `gate_grouped.py`,
   `grouped_syl_vocab.json` (workstation).
4. The 48 dp shaky-hand error percentages are **[SECONDARY]** — don't quote in
   product spec.

---

**Bottom line:** ship **Candidate A (大注音九宮格)** as default — familiar 3×4,
~120 dp/19 mm keys, table-order chunking, tone-free, no long-press/no mode
toggles, one skippable ≤3-chip large prediction line, haptic-on, Slothing's
sentence-LLM as the error-forgiveness engine that makes 免選字 real. Provide
**Candidate B** + swabbing/hold-duration as accessibility profiles and an
e-ink profile. Before locking specs, run the three §6 validations.
