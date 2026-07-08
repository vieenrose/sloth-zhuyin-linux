---
title: Slothing — Web Zhuyin IME (Gradio)
emoji: 🦥
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: LLM zhuyin (bopomofo) input method, decoded by a 34M model
models:
  - Luigi/slothlm-34m-zhuyin
---

# Slothing — a web Zhuyin (Bopomofo) input method (Gradio build)

Type Zhuyin on the on-screen Dàqiān keyboard; a **34M** from-scratch model
([SlothLM](https://huggingface.co/Luigi/slothlm-34m-zhuyin)) decodes it to
Traditional Chinese under a phonetic-legality grammar — so every character is
a real reading of what you typed, never a hallucination. This is the demo of
[Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), a libchewing-free
fcitx5 IME.

This build uses plain Gradio components (button keyboard + radio-group
segment editing) on HF's managed Gradio runtime, trading a little UI polish
for a much more reliable path to staying live vs. a scheduled Docker
container. See `../space/` for the full custom-frontend Docker build.
