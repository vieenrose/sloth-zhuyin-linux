# SlothLM: a from-scratch tiny LM purpose-built for zhuyin conversion

Design synthesized from three research passes (July 2026): the Supra-50M
recipe, the from-scratch tiny-Chinese-LM landscape, and single-RTX-3060
data/compute feasibility. Goal: replace the generic LFM2.5-230M in
`slothd` with a much smaller model that is *better at our one task*.

## Why from-scratch beats fine-tuning a generic model here

- **The task is narrow.** Conversion is homophone ranking over libchewing's
  real candidates (grammar-constrained), plus zhuyin→text mapping. It needs
  Chinese fluency and phonetic alignment — not world knowledge, chat, or
  reasoning. MiniMind proves 26M params buys fluent Chinese (and nothing
  more) in ~1 GPU-hour of pretraining; PERT proves a task-specialized
  5–30M model trained *directly on phonetic→text* hits **96.6% char
  precision**, +11.5 points over the n-gram baselines chewing represents.
- **Generic small models waste their capacity.** Measured on our 80-case
  eval, un-fine-tuned LFM2.5-230M ranks *below* chewing at top-1 (58% vs
  62%) despite 230M params, because none of that capacity is aimed at
  phonetics (the PY-GEC misalignment finding). A model whose entire
  pretraining is our task distribution aims all of it.
- **Supra-50M's real lessons transfer** (primary sources: their blog +
  research Space): llama architecture chosen explicitly because at tiny
  scale "architecture matters less than vocab, data quality, and training
  setup"; a custom tokenizer matched to the target distribution; heavy
  over-training (~400 tok/param). Their published ablations further
  support this design: a vocab-size study showing embedding params
  dominate tiny models (Pareto ceiling ~4k vocab under 7.5M params — our
  8k at 35M is proportionate); an epochs-vs-fresh-data study showing 1
  epoch of fresh data beats repetition (we plan 1 epoch); and a data-mix
  study where task-matched synthetic data took the largest share (ours:
  ~40% synthetic zhuyin tasks). But their 20B general-English tokens
  (~2 weeks of 3060) is exactly what a narrow task avoids: we need
  ~1.5–2B.
- **Latency/footprint win:** ~30M at Q8_0 is a ~30 MB GGUF; expect roughly
  5–8× faster generation than the 230M — conversion latency drops from
  ~1s toward ~150–300ms, and the tone-free/looser-input roadmap items
  (which need *generation*, not just reranking) become affordable.

## Architecture (llama.cpp-compatible by construction)

`LlamaForCausalLM` instantiated from scratch — converts to GGUF with zero
arch work:

| | |
|---|---|
| layers | 10 |
| hidden | 512 |
| heads | 8 (4 KV, GQA) |
| FFN (SwiGLU) | 1408 |
| context | 1024 (our prompts: context ≤240B + candidates + sentence) |
| pos. encoding | RoPE θ=10k |
| vocab | 8192, embeddings tied |
| **params** | **~35M** |

## Tokenizer: the load-bearing decision

Byte-level BPE (HF `tokenizers`), vocab 8192, with the **pre-tokenizer kept
byte-identical to GPT-2's** so `convert_hf_to_gguf.py`'s pre-tokenizer hash
check passes without patching (the known GGUF gotcha):

- every bopomofo symbol (U+3100–312F) + 4 tone marks: **one token each** →
  one token per zhuyin symbol
- top ~6,500 Traditional-Chinese chars: **one token each** → 1 token/char,
  which makes GBNF grammar constraints align *exactly* with candidate
  boundaries (no multi-token chars, no mid-char truncation class of bugs)
- ASCII bytes + ~1k BPE merges for English fragments (code-switching
  survival, not English quality)
- MiniMind's lesson respected: at 35M params an 8k vocab keeps the
  embedding table (~4M tied) from starving the transformer.

## Data mix (~1.5–2B tokens; 1 char ≈ 1 token by construction)

Sources verified on HF: `erhwenkuo/wikipedia-zhtw` (0.5B chars, cc-by-sa),
`erhwenkuo/c4-chinese-zhtw` (~3.5B chars) / `jed351/Traditional-Chinese-
Common-Crawl-Filtered` (native TC, needs dedup), `HuggingFaceFW/fineweb-edu`
(English slice).

| share | content | purpose |
|---|---|---|
| ~40% | plain TC text LM (c4/CC colloquial-weighted + wikipedia) | fluency, register |
| ~40% | **synthetic task data** from the same sentences: zhuyin→text, text→zhuyin, the daemon's exact SELECT format with real harvested libchewing candidates, and **tone-free zhuyin→text** (future-proofs the roadmap's tone-free mode) | the actual job |
| ~12% | English (fineweb-edu sample-10BT slice) | zh/en code-switching survival |
| ~8% | natural mixed-script TC sentences (keep latin-containing lines the pure-Han filter currently drops) | embedded English, product names |

Synthetic generation reuses `finetune/gen_dataset.py` + `eval/harvest`
mechanics at scale (harvest is ms/sentence; millions are feasible).
`eval/testset.tsv` sentences excluded everywhere, as today.

## Training plan (RTX 3060 12GB)

HF `Trainer` + `LlamaConfig` (least-friction path to GGUF), bf16,
`torch.compile`, AdamW (0.9/0.95) LR 6e-4 cosine 2% warmup — the
Supra/MiniMind-consensus recipe. Pre-tokenize to a uint16 memmap so the GPU
stays fed. Effective batch ~128 × 1024 ctx (VRAM is a non-issue: model +
optimizer ≈ 0.8 GB).

Wall-clock (6·N·D at 25–40% MFU): 35M × 1.5B tokens ≈ **13–25 h** — one
long run or two nights. Over-training beyond Chinchilla (~43×) is
intentional and cheap at this scale.

## Gates & risks (honest)

**Ship gates** (same harness, `eval/run_eval.py`):
1. top-1 sentence >= chewing's 61% (159-case baseline; LFM2.5 base is 53%, a net regression)
2. recall >= 73% (LFM2.5 base value on the 159-case set)
3. latency: median conversion < 500ms end-to-end
Else iterate (more task data, +params to ~50M) or stay on LFM2.5.

**Risks:** (a) a 35M model has no world knowledge → rare named entities
will miss; mitigated by chewing's candidates + user learning, and it's the
same weakness PERT had at 96.6%. (b) English output quality will be poor —
acceptable; EN is for survival not generation. (c) GGUF tokenizer-hash
gotcha → **step 0 below de-risks before any training**. (d) Corpus licences:
wikipedia/wikinews cc-by-sa; CC-derived corpora carry the usual Common-Crawl
ambiguity — fine for weights we train, note in the model card.

## Execution phases

0. **GGUF dry-run first** (hours, no training): build the tokenizer, init a
   random 35M checkpoint, run `convert_hf_to_gguf.py` + load in `slothd`.
   Flushes out the entire compatibility risk before a single GPU-hour.
1. Data pipeline at scale on the training box (corpus pull → filter → mix →
   synthetic task data → memmap).
2. Train ~35M × ~1.5B tokens (~1–2 days on the 3060), loss-curve sanity
   checks at intervals.
3. Convert → Q8_0 → A/B on the eval harness vs chewing and LFM2.5-230M.
4. If gates pass: ship as slothd default; upload weights + card to the
   user's HF account; keep LFM2.5 as a documented alternative.

## Status

- **Phase 0 DONE (2026-07-08):** tokenizer built (byte-level BPE, vocab 8342,
  one token per bopomofo symbol / common Han char); random ~34M
  LlamaForCausalLM converts to GGUF via `model/register_tokenizer.py` (maps
  our pre-tokenizer hash → `gpt-2`, the one llama.cpp gotcha the research
  flagged); the GGUF loads in **unmodified** slothd and produces
  grammar-constrained candidate output. The full custom-model → daemon path is
  proven before spending any GPU time. Training box: RTX 3060 12GB, torch
  2.12+cu130, 400k-sentence TC corpus pulled.
