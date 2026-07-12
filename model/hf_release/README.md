---
license: apache-2.0
language:
  - zh
tags:
  - input-method
  - zhuyin
  - bopomofo
  - traditional-chinese
  - ternary
  - bitnet
  - gguf
library_name: gguf
pipeline_tag: token-classification
---

# SlothE-T 25M — Ternary Zhuyin → Traditional Chinese IME model

The conversion model behind **Slothing (懶音輸入法)**: a libchewing-free, on-device
LLM-powered 注音 (Bopomofo) input method that does **免選字** (candidate-free)
whole-sentence conversion — you type the phonetic keystream, it emits the sentence,
with no candidate list to scroll.

- **25M parameters**, ternary weights (W1.58A8), bidirectional encoder.
- **~7 MB** ternary-packed GGUF (TQ2_0) · 99 MB fp32 master.
- Runs on-device across four frontends: **fcitx5, IBus, Android, web**.

## What it does

Given a sequence of Zhuyin **syllables** (the input keystream, e.g. `ㄋㄧˇ ㄏㄠˇ`),
the model emits the **Traditional Chinese characters** for each position as an
aligned sequence-labeling task. The output head is **phonetic-legality-masked**:
at each position only the ~1–50 characters that are legal readings of that
syllable are scored, out of the full 8342-char vocabulary. This is what replaces
the traditional IME 選字 (candidate-selection) step.

## Architecture

| | |
|---|---|
| encoder | bidirectional (BERT-like), 16 layers |
| dim / ffn | 352 / 960 |
| heads | 8 query, 2 KV (GQA), head_dim 44, QK-norm |
| norm | RMSNorm, embed-norm, **SubLN** pre-norm before each ternary linear |
| quantization | ternary weights {−1, 0, +1} × per-output-channel **absmedian** scale; **int8** activations; QAT via STE |
| fp islands | boundary blocks (0 and 15) kept fp16 for stability (`fp_boundary=1`) |
| vocab | 1539 syllables (in) · 8342 characters (out) |

## Evaluation — honest held-out

Measured on **500 fresh zh-TW sentences** (C4-zh-TW, offset far past the training
window, filtered to be **absent from the training corpus**), then g2pW-labeled.

| metric | **this model** | 12M int8 (previous ship) | 32M fp teacher |
|---|---|---|---|
| **免選字** (whole-sentence exact) | **76 %** | 72 % | — |
| **homophone-hard** | **86 %** | 82 % | 83 % |
| **toneless** | **77 %** | 79 % | 81 % |

> **Note on 免選字.** Earlier project numbers (~84 %) were inflated by a benchmark
> leak — the reference set had been sampled from the training corpus, rewarding
> memorization. **76 %** is the honest whole-sentence held-out figure.
> *homophone-hard* and *toneless* are leak-clean throughout. On honest data this
> 25M ternary model **beats the previously-shipped 12M int8 on 免選字 and
> homophone-hard** while being smaller and faster.

### Quality vs. latency

![Held-out 免選字 vs. on-device latency](score_vs_latency.png)

Latency is per-6-syllable decode on a **BOOX (Snapdragon 662, no-dotprod ARM)**.
4M and 12M are measured (ORT int8); the **25M ternary** point is *projected* from
measured-shape TQ2_0 / I2_S kernel benchmarks (the ternary win is diluted by the
int8 8342-way head, which ternary can't accelerate). The takeaway: the 25M ternary
is **Pareto-dominant** — roughly the 4M's latency (~9 ms) but **+6 免選字**, and
both faster *and* more accurate than the 12M.

| model | params | 免選字 | homophone | toneless | latency (BOOX) |
|---|---|---|---|---|---|
| 4M int8 | 4M | 70 % | 83 % | 74 % | 9.1 ms |
| 12M int8 | 12M | 72 % | 82 % | 79 % | 13.3 ms |
| **25M ternary** | **25M** | **76 %** | **86 %** | **77 %** | **~9 ms\*** |

<sub>\* projected, not full end-to-end measured. See the [experiment notes](REPRODUCE.md).</sub>

## Training recipe (teacher-free)

- **Direct cross-entropy** on a g2pW-labeled zh-TW corpus, with **label smoothing 0.1**.
- **Long OneCycleLR schedule (32 epochs), early-stopped at the peak (epoch 24)** —
  the model overfits after (73 % 免選字 by ep32).
- **8-adjacency keyboard-error noise** (TAAI-2024 error model): simulated mis-keys
  are constrained to the QWERTY physical 8-neighbourhood rather than any
  edit-distance-1 syllable.
- DDP on 2× RTX 5090.

Knowledge distillation from a 32M teacher was tried and only **matched** (never
beat) this teacher-free recipe, so the teacher was dropped — simpler and ~2.4×
faster to train. The two levers that carry the quality are (a) label smoothing
(a teacher-free regularizer; +5 免選字 over plain CE) and (b) enough epochs to
exploit the no-overfit headroom.

## Deployment / inference

The ternary weights are shipped as a **GGUF** using ggml's mainline **TQ2_0**
ternary type, so the model runs on stock llama.cpp CPU kernels (ARM NEON, x86
AVX2, and the generic/WASM path). Measured `vec_dot` throughput on x86 AVX2:
**TQ2_0 ≈ 2.3× int8 (Q8_0)** — the fastest quant type in the table — and ~2× on
a no-dotprod ARM (Snapdragon 662). The ternary win is partly memory-bandwidth
(2 bits/weight = 4× fewer weight bytes), so it holds on both weak and
accelerator-equipped CPUs.

> **256-alignment note.** TQ2_0 uses 256-element blocks, so the model's
> in-features (352, 960) are zero-padded up to the next multiple of 256 in the
> GGUF (352 → 512, 960 → 1024). The padding is exact ternary-zero (loss-free) but
> costs some MACs; a future model with 256-aligned dims removes the tax.

## Files

| file | what |
|---|---|
| `slothe-t-25m.gguf` | ternary GGUF — TQ2_0 blocks (layers 1–14) + fp16 islands/embed/head |
| `model.safetensors` | fp32 master weights (HF-native, non-pickle) |
| `slothe.pt` | fp32 master checkpoint with embedded `config` |
| `syl_vocab.json` | syllable tokenizer (input vocab) |
| `syl2legal.npz` | phonetic legality mask, 1539 syllables × 8342 chars (bool) |
| `train_slothe_ternary.py` | training script |
| `gate_slothe_ternary.py` | evaluation / gating script |
| `REPRODUCE.md` | end-to-end reproduction recipe |
| `NAMES.md` | GGUF tensor-name ↔ checkpoint-name map |

## Intended use & limits

- **Intended:** on-device Traditional-Chinese Bopomofo input (Taiwan readings).
- **Out of scope:** Simplified Chinese, mainland pinyin readings, free-form
  generation. The model only scores per-position legal characters — it is a
  converter, not a chat model.
- The whole-sentence (免選字) metric is honest-held-out and modest (76 %) by
  design: it is measured on unseen sentences, not the training distribution.

## Citation / related work

The keyboard-error model and the 免選字-vs-選字 framing are discussed in the
project's `docs/RELATED-WORK.md`, which positions this work against the TAAI-2024
cross-multi-IME system (李偉安, 葉展維, 張嘉惠, National Central University).
