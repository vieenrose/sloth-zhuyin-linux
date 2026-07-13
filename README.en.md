# Slothing (懶音輸入法)

**Type bopomofo; a small on-device model turns the whole sentence into correct
Chinese — no candidate-picking.**

Slothing decodes Zhuyin to Traditional Chinese with a from-scratch **25M ternary
language model** that runs on your device — no libchewing, no cloud, and every
character guaranteed to be a legal reading of what you typed. Four frontends —
desktop (fcitx5, IBus), Android, and the browser — share one model.

**▶ [Try it now (no install)](https://huggingface.co/spaces/Luigi/slothing-web)** ·
[中文說明](README.md) ·
[Model](https://huggingface.co/Luigi/slothe-t-25m-zhuyin)

<p align="center"><img src="docs/demo-web-v13.gif" width="470" alt="Web demo: typing 晚上熬夜看world cup,白天在louisa key-by-key — whole sentence correct, auto zh/en, each keypress highlighted"></p>
<p align="center"><img src="docs/android-boox-demo-v11.gif" width="470" alt="Native Android IME (BOOX e-ink): typing 晚上熬夜看world cup,白天在louisa key-by-key — whole sentence correct, auto zh/en switch"></p>

## What you get

| | |
|---|---|
| **Whole-sentence conversion** | 微軟新注音-style live conversion; no picking word-by-word |
| **Auto zh/en** | No mode key: type `我用python寫程式` straight through — a segmenter decides |
| **Tone-optional** | Skip tone keys (~35% fewer keystrokes); context disambiguates |
| **Typo repair** | Impossible syllables get fixed by the model |
| **Next-word suggestions** | After you commit (tap-to-chain on mobile, ⇧1-9 on desktop) |
| **Fully offline** | 18 MB model runs on-device — no cloud, no telemetry |

## Install

**Desktop (fcitx5 or IBus) — one command:**

```sh
git clone https://github.com/vieenrose/sloth-zhuyin-linux.git
cd sloth-zhuyin-linux
./install.sh        # auto-detects fcitx5 or IBus: builds, downloads the model, sets up autostart
```

Then add **"Slothing"** in your input-method settings and switch to it with
**Ctrl+Space**. (Needs `git`, `cmake`, a C++ compiler; the engine-install step
asks for `sudo`.)

| Other platforms | How |
|---|---|
| **Android** | Grab the `.apk` from Releases — offline, model built in, no daemon |
| **Browser** | Nothing to install: [live demo](https://huggingface.co/spaces/Luigi/slothing-web) |

## Accuracy

Honest held-out (500 c4-zh-TW sentences, excluded from training):

| Benchmark | Score |
|---|---|
| Whole-sentence exact (免選字) | **76%** |
| Tonal per-char (homophone-hard) | **86%** (libchewing 71%) |
| Tone-free | **77%** |

Ceiling = 微軟新注音 / 自然輸入法; floor = libchewing. Method and sourcing in
[docs/COMPARISON.md](docs/COMPARISON.md), [docs/EVAL.md](docs/EVAL.md).

## Why ternary

<p align="center"><img src="docs/score_vs_latency.png" width="600" alt="Quality vs on-device latency (BOOX SD662): the 25M ternary hits 76% at ~9ms — both faster and better than the 12M int8"></p>

A from-scratch **SlothE-T 25M ternary (W1.58A8) bidirectional encoder**. At
on-device latency (BOOX SD662) it is **Pareto-optimal**: **76%** whole-sentence at
~**9 ms / 6-syllable decode** — the same speed as a tiny 4M int8 but far more
accurate — and both faster and better than the 12M int8 (13 ms, 72%). **At this
scale, ternary beats int8.** The TQ2_0 kernel is ~2.3× int8 on x86 (mainline ggml),
so no bitnet.cpp is needed.

## How it works

Zhuyin→Chinese is *aligned sequence labeling* (N syllables → N characters, each
constrained to its homophone set), so Slothing uses a **bidirectional encoder**
(non-autoregressive, one forward pass) rather than a causal LM. A dependency-free
segmenter parses the keystream (auto zh/en) and decoding is masked per position to
legal readings.

All four frontends are thin adapters over the shared core in `engine/common` and
share one **`libslothe`** ggml forward pass: a native daemon on desktop, NDK arm64
on Android, WASM in the browser. Behavior is held together by offline contract
tests, a headless end-to-end test, and per-layer / per-character golden checks
against PyTorch.

- Model, GGUF + full reproduction pipeline: [Luigi/slothe-t-25m-zhuyin](https://huggingface.co/Luigi/slothe-t-25m-zhuyin)
- Architecture & design: [`ARCHITECTURE.md`](ARCHITECTURE.md), `model/DESIGN-E.md`
- 4-frontend UI logic matrix: [docs/UI-MATRIX.md](docs/UI-MATRIX.md)

## Roadmap

- [x] **25M ternary shipped to all four frontends**: 76 / 86 / 77, all sharing `libslothe` (ggml/TQ2_0), replacing ONNX Runtime
- [ ] **Char-hints v2**: retrain a hinted ternary model to restore pick / document-context re-scoring (recipe still tuning; not yet beating the no-hints model)
- [ ] Model-based next-word head; word-list filtering
- [ ] Android hardware-keyboard polish; regular desktop packages
- [ ] [Senior-friendly keyboard layout](docs/SENIOR-KEYBOARD.md): standard layout + key-error-tolerant decoding

**Non-goals:** any cloud inference or telemetry — everything runs locally.

<details><summary>Milestones done</summary>

libchewing-free engine · web demo · tone-free / auto zh/en · SlothLM-E 11.6M ·
char-hint channel · 新注音-style live conversion + candidate window · libchewing
UI-parity suite · full reproducibility bundle on HF · IBus engine · native Android
IME (BOOX e-ink) · 4-frontend next-word · `.deb` / `.apk` packaging ·
**25M ternary model + libslothe deployed to all four frontends**
</details>
