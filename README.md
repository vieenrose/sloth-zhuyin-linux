# sloth-zhuyin-linux · 懶 Slothing

**A libchewing-free, LLM-powered Zhuyin (Bopomofo) input method** — a fcitx5
IME for Linux, plus a fully in-browser web demo. Type bopomofo; a tiny
from-scratch model decodes it to Traditional Chinese under a phonetic-legality
constraint, so every character is a real reading of what you typed — never a
hallucination.

**中文說明: [README.zh-TW.md](README.zh-TW.md)**

> **Live demo (free, runs entirely in your browser):**
> **https://huggingface.co/spaces/Luigi/slothing-web**
> Type on the on-screen Dàqiān keyboard or your physical keyboard — Chinese,
> English, and mixed input auto-detected with no mode key.

## What it is

Slothing replaces the statistical decoder of a traditional zhuyin IME (like
chewing) with a small language model, while keeping the guarantee that output
is always phonetically legal. Two things make it different from every other
open-source zhuyin IME (McBopomofo, vChewing, libchewing are all purely
statistical):

- **The model decodes, not just reranks.** A bopomofo→Chinese model resolves
  the homophones a dictionary IME gets wrong (它→他, 在/再, 覺/決) using
  sentence context.
- **No libchewing.** A dependency-free keyboard FSM parses keystrokes; the
  model decodes; a per-position legal-character grammar guarantees valid
  readings. Local, private, no cloud.

## The models

| | SlothLM (v1) | **SlothLM-E** (v2) |
|---|---|---|
| type | causal decoder-LM (Llama) | **bidirectional encoder** |
| params | ~34M | **3.9M (NAS)** |
| decode | autoregressive | **non-autoregressive, 1 pass** |
| tonal accuracy | ~beats chewing | **83% (chewing 71%)** |
| tone-free accuracy | weak | **70%** — usable tone-free typing |
| on HF | (removed) | [Luigi/slothlm-e-4m-zhuyin](https://huggingface.co/Luigi/slothlm-e-4m-zhuyin) |

Zhuyin decode is *aligned sequence labeling* (N syllables → N characters, 1:1,
each constrained), so a **bidirectional encoder** fits the task far better than
a causal decoder: it sees the whole sentence (right-context disambiguation:
行走/銀行), decodes in one pass, and is half the size. Both are trained
from scratch on Traditional-Chinese text + synthetic tasks (zhuyin↔text,
tone-free, code-switch), with a **custom byte-level tokenizer** (one token per
bopomofo symbol / Han character). See `model/DESIGN.md` and `model/DESIGN-E.md`.

## Features

- **Grammar-constrained decode** — output is masked to each syllable's
  phonetically-legal characters, so it can never hallucinate an invalid reading.
- **Tone-free typing** — drop the tone keys (~35% fewer keystrokes); the model
  disambiguates from context.
- **Auto Chinese/English** — no mode toggle: valid zhuyin adds one bopomofo
  symbol per keystroke, so an impossible-zhuyin keystroke run is detected as
  English (ASUS-IME style). English passes through verbatim; code-switch
  (`我用 Python 寫 code`) just works.
- **Chewing-shaped editing** — inline conversion, one-Enter commit, preedit
  cursor + mid-sentence editing, paged candidates with number-key selection,
  punctuation, per-position + LLM-ranked phrase candidates, session learning.
- **Validated against libchewing** — every model/decode change must pass
  `eval/chewing_parity.py` (SlothLM ≥ chewing) before shipping.

## Repository layout

- `engine/fcitx5-chewing/` — **Slothing**, the fcitx5 addon. Now libchewing-
  free: keystrokes are parsed by `src/zhuyin.h` (a dependency-free Dàqiān FSM);
  the convert key sends the typed syllables to the daemon's decode path.
- `engine/slothingd/` — **slothingd**, a small Unix-socket daemon (llama.cpp C
  API) that decodes syllables under the phonetic-legality GBNF grammar, tonal
  or toneless, with English passthrough for code-switch.
- `model/` — the models: tokenizer, data prep, training + eval for SlothLM
  (decoder) and **SlothLM-E** (encoder), `phonetic_table.tsv` (syllable → legal
  chars), and `chewing_parity.py` (the validation gate).
- `space-static/` — the web demo (`sdk: static`): the full IME UI running the
  model in-browser via ONNX / Transformers.js — free, never sleeps, no server.
- `eval/` — scored zhuyin→sentence test set and harnesses (rerank, decode,
  chewing-parity).
- `ARCHITECTURE.md`, `RESEARCH-LLM-IME.md`, `MODEL_BENCHMARKS.md`, `MIGRATION.md`.

## Build & install the fcitx5 addon

```sh
cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build engine/fcitx5-chewing/build -j"$(nproc)"
sudo make -C engine/fcitx5-chewing/build install
fcitx5 -r -d
```

Add **Slothing** (🦥) via `fcitx5-configtool`. Then set up the local model +
daemon:

```sh
sh scripts/setup-llm.sh          # llama.cpp + model + slothingd
packaging/run-slothingd.sh       # start the decoder (manual by design)
```

## Roadmap (highlights)

- [x] libchewing-free engine (keyboard FSM + LLM decode)
- [x] SlothLM (34M decoder) trained, on HF, with Q4/Q8 GGUF
- [x] Web demo — in-browser, free, chewing-shaped UX
- [x] Tone-free mode, auto zh/en, code-switch, session learning
- [x] SlothLM-E (16M bidirectional encoder) — smaller/faster/better; finalizing
- [ ] Swap the demo + desktop daemon to the SlothLM-E ONNX model
- [ ] Per-phrase Down-rank; typo tolerance; packaging (.deb)
- [ ] (Future, long-document context) hybrid Transformer + SSM decoder

**Non-goals:** cloud inference, telemetry. Everything runs locally.
