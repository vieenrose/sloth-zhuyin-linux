---
title: 懶音輸入法 Slothing
emoji: 🦥
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: In-browser zhuyin IME decoded by a 3.9M model (no server)
models:
  - Luigi/slothlm-e-4m-zhuyin
---

# Slothing — a web Zhuyin (Bopomofo) input method

Type Zhuyin on the on-screen Dàqiān keyboard (or your physical keyboard); a
**3.9M-parameter** from-scratch bidirectional encoder
([SlothLM-E 4M](https://huggingface.co/Luigi/slothlm-e-4m-zhuyin)) decodes it
to Traditional Chinese **entirely in your browser** via onnxruntime-web
(per-channel int8 ONNX, **~5 MB**, cached after first load; single-thread WASM,
so it also works on iOS Safari). One forward pass decodes the whole sentence —
non-autoregressive, N syllables → N characters.

Every character is constrained to a real Taiwan-standard reading of what you
typed — no hallucinations — using a per-position legal-char mask that
reproduces the desktop IME's grammar. Click any character to pick a different
homophone. Mixed zh/en typing is handled by a DP re-segmenter over the raw
keystream.

The model was found by a sub-5M neural architecture search and trained on
g2pW context-aware readings (neural Taiwan polyphone disambiguation):
83% homophone-hard / 76% 免選字 sentence accuracy — above libchewing's 71%
floor at 1/8 the size of the previous 32M model.

Demo of [Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), a
libchewing-free fcitx5 IME. **No server, no cloud** — a static Space; all
inference runs client-side.
