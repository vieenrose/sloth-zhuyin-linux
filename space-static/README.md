---
title: Slothing — Web Zhuyin IME
emoji: 🦥
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: In-browser zhuyin IME decoded by a 34M model (no server)
models:
  - Luigi/slothlm-34m-zhuyin-ONNX
---

# Slothing — a web Zhuyin (Bopomofo) input method

Type Zhuyin on the on-screen Dàqiān keyboard (or your physical keyboard); a
**34M** from-scratch model
([SlothLM](https://huggingface.co/Luigi/slothlm-34m-zhuyin)) decodes it to
Traditional Chinese **entirely in your browser** via Transformers.js (ONNX q8,
~33 MB, cached after first load). Every character is constrained to a real
reading of what you typed — no hallucinations — using a per-position legal-char
mask that reproduces the desktop IME's grammar. Click any character to pick a
different homophone.

Demo of [Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), a
libchewing-free fcitx5 IME. **No server, no cloud** — a static Space; all
inference runs client-side.
