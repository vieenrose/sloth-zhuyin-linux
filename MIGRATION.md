# Migration: libchewing → LLM decoder

Goal: remove libchewing as a dependency. The LLM (SlothLM) becomes the decoder;
the engine owns the zhuyin keyboard directly. Decided after epoch-1 SlothLM
passed the rerank gates (73% top-1) and decode mode reached 50% — with the final
model expected to clear libchewing's 61%.

## What libchewing does today, and its replacement

| libchewing role | replacement | stage |
|---|---|---|
| keystroke → bopomofo syllable (layout FSM) | `zhuyin.h` ZhuyinBuffer (this repo) | **A (done/testable)** |
| composing buffer + preedit | engine holds `ZhuyinBuffer`, renders syllables | B |
| sentence decode (zhuyin → Han) | slothingd decode mode (`{"syllables":[...]}`) | done |
| candidate window (Down per-phrase) | LLM per-phrase rerank (SELECT1) | later |
| symbols / punctuation | direct passthrough table | B |
| userphrase learning | drop (LLM personalization via LoRA later) | — |

## Stages

- **A. Zhuyin keyboard FSM** — `engine/fcitx5-chewing/src/zhuyin.h`, a pure,
  dependency-free Dachen-layout state machine (key → syllable slots → committed
  bopomofo syllables), with a standalone test (`zhuyin_test.cpp`). No engine
  wiring yet; fully unit-tested offline. **← current**
- **B. Engine on the FSM** — replace `chewing_handle_*`, the buffer, and preedit
  with `ZhuyinBuffer`; convert key sends the committed syllables to decode mode.
  libchewing calls deleted from `eim.cpp`; dependency dropped from CMake.
- **C. Editing + symbols** — backspace within/across syllables, cursor, punctua-
  tion, Chinese/English toggle, full-width symbols.
- **D. Layouts** — ETen/Hsu/etc. beyond Dachen (default) if wanted.

Ship B behind the existing DecodeMode-style path; keep libchewing importable
until B is proven, then delete. Each stage builds and is verified before the
next.
