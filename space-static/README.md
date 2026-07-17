---
title: 樹懶智慧輸入法 Sloth IME
emoji: 🦥
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: In-browser zhuyin IME, 12M ternary model, no server
models:
  - Luigi/sloth-ime-models
---

# Sloth IME — a web Zhuyin (Bopomofo) input method

Type Zhuyin on the on-screen Dàqiān keyboard (or your physical keyboard); a
**12M-parameter ternary (W1.58A8)** from-scratch bidirectional encoder
([SlothE-T 12M 256×12](https://huggingface.co/Luigi/sloth-ime-models)) decodes it
to Traditional Chinese **entirely in your browser** via **ggml compiled to
WebAssembly** (mainline ggml's TQ2_0 ternary kernels; ~18 MB GGUF, cached after
first load; single-thread WASM, so it also works on iOS Safari). One forward
pass decodes the whole sentence — non-autoregressive, N syllables → N characters.

Every character is constrained to a real Taiwan-standard reading of what you
typed — no hallucinations — using a per-position legal-char mask that
reproduces the desktop IME's grammar. Click any character to pick a different
homophone. Mixed zh/en typing is handled by a DP re-segmenter over the raw
keystream.

Trained teacher-free on g2pW context-aware readings (neural Taiwan polyphone
disambiguation): honest held-out **86% homophone-hard / 76% 免選字 sentence /
77% tone-free** — above libchewing's 71% floor. The ternary weights run on the
same `libslothe` (ggml) core across all four frontends (fcitx5, IBus, Android,
web), ~2.3× the throughput of int8.

Demo of [Sloth IME](https://github.com/vieenrose/sloth-zhuyin-linux), a
libchewing-free IME. **No server, no cloud** — a static Space; all inference
runs client-side.


## Models & reproducibility

This demo runs the **12M ternary conversion encoder** fully in your browser (ggml/TQ2_0
WASM). The full Sloth IME system also includes a **60M next-word decoder (v2.1,
TW-chat register fine-tuned)** used by the desktop/Android frontends. All weights
(GGUF + fp32 masters) and training/eval scripts: 
[Luigi/sloth-ime-models](https://huggingface.co/Luigi/sloth-ime-models) · 
[reproducibility guide](https://github.com/vieenrose/sloth-zhuyin-linux#reproducibility)
