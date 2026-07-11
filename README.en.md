# Slothing(懶音)— an LLM-powered Zhuyin IME

**Type bopomofo; a model converts the whole sentence.** **Slothing** (Chinese
name **懶音**, from 樹懶 "sloth" + 注音 zhuyin; full zh name 樹懶注音輸入法):
an 11.6M-parameter
language model trained from scratch that decodes zhuyin to Traditional Chinese
locally — libchewing-free, with every character guaranteed to be a legal
reading of what you typed. Four frontends — desktop (fcitx5, IBus), Android,
and the browser — share one core and one model.

**中文說明(預設): [README.md](README.md)** ·
**Try it now (no install): [huggingface.co/spaces/Luigi/slothing-web](https://huggingface.co/spaces/Luigi/slothing-web)**

<p align="center"><img src="docs/demo-web-v12.gif" width="470" alt="Web demo (recorded on a BOOX tablet, 11.6M model): typing 我用claude寫注音輸入法 — correct whole-sentence 免選字, auto zh/en, 字/句 candidates, 聯想 prediction chaining"></p>
<p align="center"><img src="docs/android-boox-demo-v9.gif" width="460" alt="Native Android IME (BOOX e-ink, 11.6M model): typing 我用claude寫注音輸入法 — correct whole-sentence 免選字, 字/句 candidates, 聯想 prediction chaining"></p>

## Highlights

| | |
|---|---|
| **免選字 sentence conversion** | 微軟新注音-style live conversion; **84%** on the 230-case no-selection benchmark (83% on-device) |
| **Auto zh/en** | No mode key: type `我用python寫程式` straight through — a DP segmenter decides |
| **Tone-optional** | Skip tone keys (~35% fewer keystrokes); context disambiguates |
| **Picks re-score the sentence** | Correct one char and the rest re-decodes around it (char-hint channel), with **persistent learning** |
| **Typo repair** | Impossible syllables fixed by the model (edit distance 1) |
| **聯想 prediction** | Next-word suggestions after commit (dictionary + personal habits): tap-to-chain on mobile, ⇧1-9 on desktop |
| **Fully offline** | 13 MB int8 ONNX runs locally — no cloud, no telemetry |

Sourced comparison vs Gboard 注音 and the Boox built-in IME:
**[docs/COMPARISON.md](docs/COMPARISON.md)** (zh-TW); 4-frontend UI logic matrix:
**[docs/UI-MATRIX.md](docs/UI-MATRIX.md)**.

## Install

Desktop platforms need the decode daemon (one-time setup, shared by fcitx5 + IBus):

```sh
pip install onnxruntime numpy
packaging/fetch-model.sh                  # 4.9 MB model
packaging/install-slothingd-service.sh    # auto-start at login
```

| Platform | Install |
|---|---|
| **fcitx5** (KDE, …) | `.deb` from Releases, or `cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing -DCMAKE_INSTALL_PREFIX=/usr && cmake --build engine/fcitx5-chewing/build -j$(nproc) && sudo make -C engine/fcitx5-chewing/build install` |
| **IBus** (GNOME, …) | `.deb` from Releases, or the one-shot `engine/ibus-slothing/install.sh`; see `engine/ibus-slothing/README.md` |
| **Android** | `.apk` from Releases (**no daemon** on the phone — decoding runs on-device), or `cd android && ./gradlew :app:assembleDebug` (needs SDK/NDK; fetch the model first) |
| **Browser** | nothing to install: [HF Space](https://huggingface.co/spaces/Luigi/slothing-web) |

## How it works

Zhuyin→Chinese is *aligned sequence labeling* (N syllables → N characters, each
constrained to its homophone set), so Slothing uses a **bidirectional encoder**
(non-autoregressive, one forward pass) instead of a causal LM: an 11.6M model
scaled up from the sub-5M Hyperband-NAS-winning recipe, trained on g2pW context-aware
readings, with a **char-hint channel** (weight-tied to the output head, ~0
params) carrying pick feedback, document context, and typo repair. A
dependency-free DP segmenter parses the keystream (auto zh/en), and decoding is
masked per position to legal readings.

All four frontends are thin adapters over the shared core in `engine/common` —
one state machine, one segmenter, one prediction engine — held together by
offline contract tests (core_test), a headless IBus end-to-end test, and the
`eval/ui-parity` differential suite.

- Model + full reproduction pipeline (data → labels → NAS → training → ONNX):
  [Luigi/slothlm-e-12m-zhuyin](https://huggingface.co/Luigi/slothlm-e-12m-zhuyin)
- Architecture & design: `ARCHITECTURE.md`, `model/DESIGN-E.md`, `MODEL_BENCHMARKS.md`

## Numbers

| Benchmark | Score |
|---|---|
| 230-case 免選字 set (whole sentence exact) | **84%** (193/230; 83% Android on-device) |
| Tonal per-char accuracy | **83%** (libchewing 71%) |
| Tone-free | 70% |

Ceiling = 微軟新注音/自然輸入法 (the test set is their no-selection answer set);
floor = libchewing. Methodology in `docs/COMPARISON.md`.

## Roadmap

- [x] ~10M model: 免選字 74→**84** (11.6M, shipped to all four frontends; int8 beat int4/QAT/ternary — see experiment ledger)
- [ ] BIO word-boundary + model-based 聯想 head (needs a fine-tune); word-list filtering
- [ ] Android hardware-keyboard polish; regular desktop package releases
- [ ] **Senior-friendly keyboard layout** (Android): big 3×4 grouped grid + LLM disambiguation; design research in [docs/SENIOR-KEYBOARD.md](docs/SENIOR-KEYBOARD.md)

<details><summary>Done (expand)</summary>

libchewing-free engine (keyboard FSM + LLM decode) · web demo · tone-free /
auto zh/en code-switch · SlothLM-E 11.6M (NAS-derived + g2pW) · char-hint channel
(pick re-scoring / document context / typo repair) · 新注音-style live
conversion + chewing-grade candidate window · differential UI-parity suite vs
real libchewing · full reproducibility bundle on HF · IBus engine · native
Android IME (validated on BOOX e-ink) · 聯想 on all four frontends · touch
candidate strip · learn-bonus calibration (2/3) · `.deb` / `.apk` packaging
</details>

**Non-goals:** any cloud inference or telemetry — everything runs locally.
