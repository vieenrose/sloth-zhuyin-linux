---
title: Slothing — Web Zhuyin IME
emoji: 🦥
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: LLM zhuyin (bopomofo) input method, decoded by a 34M model
models:
  - Luigi/slothlm-34m-zhuyin
---

# Slothing — a web Zhuyin (Bopomofo) input method

Type Zhuyin; a **34M** from-scratch model
([SlothLM](https://huggingface.co/Luigi/slothlm-34m-zhuyin)) decodes it to
Traditional Chinese under a phonetic-legality grammar — so every character is a
real reading of what you typed, never a hallucination. This is the demo of
[Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), a libchewing-free
fcitx5 IME.

- **Type** bopomofo on the default (Dàqiān) layout — the composing area shows
  your syllables.
- **Convert** (Space / the button) → the model decodes the whole sentence.
- **Edit** each phrase: click a character to pick a different homophone; the
  candidate list is that syllable's phonetically-legal characters.
- **Tone-free** toggle: drop the tone keys and let the model disambiguate.

Runs entirely on CPU in the Space (Q4_0 GGUF, ~100 ms/sentence). No libchewing,
no cloud.
