# SlothLM-E: a smaller, better-suited architecture

## Why change architecture

SlothLM v1 is a causal decoder-LM (Llama, 34M). But zhuyin decode is
**monotonic sequence labeling**, not free generation:

- input  = N bopomofo syllables (one token each)
- output = N Han characters (one token each), position-aligned to the input
- each output char is constrained to that syllable's phonetic candidate set

For this task shape a causal decoder is a poor fit:

1. **Causal attention hurts.** To pick char *i* the decoder cannot see
   syllables *i+1…N*, yet disambiguation often needs the right context
   (行: 行走 xíng vs 銀行 háng; 重: 重新 vs 重心 depends on the next char).
   A **bidirectional encoder** conditions each char on the whole sentence.
2. **Autoregression is wasted.** Output length is fixed and aligned to the
   input, so we needn't decode left-to-right N times. A **non-autoregressive**
   encoder predicts all N chars in **one forward pass** — far faster and no
   per-token loop.

So: replace the decoder-LM with a **bidirectional encoder that does
grammar-constrained per-position character classification**. Better accuracy
(bidirectional), faster (single pass), smaller (the task is simpler than
general LM).

## Architecture

| | v1 (SlothLM) | v2 (SlothLM-E) |
|---|---|---|
| type | causal decoder-LM | **bidirectional encoder + per-pos head** |
| attention | causal | **full (bidirectional)** |
| decode | autoregressive, N passes | **non-autoregressive, 1 pass** |
| layers | 10 | 8 |
| hidden | 512 | 384 |
| heads | 8 (4 KV, GQA) | 6 (2 KV, GQA) |
| FFN (SwiGLU) | 1408 | 1024 |
| norm | RMSNorm | RMSNorm + **QK-norm** |
| pos | RoPE θ=10k (causal) | RoPE (bidirectional) |
| vocab | 8342, tied | 8342, tied (option: 5k common → ~2M less) |
| **params** | **~34M** | **~16–18M** |

Extras that stabilise a small/deep model at ~no cost: **QK-norm** (query/key
RMSNorm) lets us use a higher LR and trains more stably than v1; kept SwiGLU,
RMSNorm, tied embeddings, GQA.

### The head
Shared with the input embedding (tied). At each syllable position the logits
over the vocab are **masked to that syllable's candidate token-ids** (the same
phonetic-legality grammar as today, applied once per position instead of per
autoregressive step). `argmax` (or top-k for the candidate strip) per position.

## Input / output format

Input is the raw bopomofo-token sequence (plus optional context prefix and
English passthrough tokens):

```
[BOS] ㄨㄛˇ ㄗㄞˋ ㄔㄨㄥˊ ㄒㄧㄣ ㄎㄠˇ ㄌㄩˋ [EOS]
        ↓     ↓     ↓     ↓     ↓     ↓     (per-position classify, masked)
        我    在    重    新    考    慮
```

- **Code-switch / English**: an English token passes straight through (its
  "candidate set" is itself), and the encoder sees it as bidirectional context
  for the surrounding Chinese — strictly better than the causal version.
- **Tone-free**: identical, the candidate set for a toneless syllable is the
  tone-union (as today).
- **Rerank mode**: same model — feed the syllables, read the per-position
  candidate scores; the desktop "reorder this phrase's candidates" feature is
  just the head's softmax over that position's candidate set.

## Training

Simpler objective than v1: for each (syllables → sentence) pair, cross-entropy
per position over the vocab (optionally restricted to the candidate set for a
sharper signal). All positions supervised, bidirectional attention, one pass —
so **each training step covers the whole sentence at once**, and there's no
ChatML prompt overhead in the token budget → more effective supervision per
token, faster convergence, likely fewer epochs.

Reuse the existing corpus + `prepare_data.py` targets (plain z2t / toneless /
code-switch), reformatted as aligned (bopomofo-seq, char-seq) pairs instead of
ChatML. Tokenizer is unchanged (one token per bopomofo symbol / Han char).

## Serving implications (the one real cost)

llama.cpp/GGUF is decoder-oriented; a bidirectional encoder is served via
**onnxruntime** instead. This is *already* the demo's runtime (Transformers.js
/ ONNX), so the web demo is unaffected and gets faster (1 pass). The desktop
`slothingd` would swap its llama.cpp backend for onnxruntime's C++ API — a
contained change, and arguably cleaner (one runtime, no GGUF conversion). The
phonetic-legality masking stays identical.

## Expected outcome

- **~half the params** (~16M) at equal-or-better accuracy, because
  bidirectional context is more informative per parameter for disambiguation.
- **Faster inference** (single forward pass vs N autoregressive steps) — big
  for the browser demo and the desktop hot path.
- **Faster training** (whole-sentence supervision, no prompt tokens, simpler
  objective) — likely 2 epochs suffice.
- Validated the same way: **`eval/chewing_parity.py` must stay ≥ chewing**,
  plus the decode/rerank evals.

## Risks / open questions

- Non-AR assumes each output char depends only on the syllables + the other
  chars' *distributions*, not their sampled values. For homophone decode this
  holds well (the constraint set is per-position); if joint dependencies matter
  (rare), one iterative-refinement pass (mask-predict) recovers most of it.
- Vocab-reduction to 5k trades rare-char coverage for ~2M fewer params —
  optional, decide by frequency-coverage on the corpus.
- Encoder can't "generate" free English, but English is passthrough anyway.

## Next step
Prototype SlothLM-E at ~16M, train on the existing data, run it through the
chewing-parity gate against v1 (34M) and libchewing. Ship only if it matches/
beats v1 while being smaller + faster.
