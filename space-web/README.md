---
title: Slothing — Web Zhuyin IME
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

# Slothing — a web Zhuyin (Bopomofo) input method

Type Zhuyin on the on-screen Dàqiān keyboard (or your physical keyboard); a
**34M** from-scratch model ([SlothLM](https://huggingface.co/Luigi/slothlm-34m-zhuyin))
decodes it to Traditional Chinese under a phonetic-legality grammar — every
character is a real reading of what you typed, never a hallucination. Then
click any character to pick a different homophone.

Demo of [Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), a
libchewing-free fcitx5 IME. The custom IME frontend runs inside the Gradio
runtime (reliable scheduling), bridged to the SlothLM decode. CPU-only.
