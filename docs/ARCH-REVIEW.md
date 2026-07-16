# Architectural review — latency & quality for the Sloth IME LLMs

*Deep-research synthesis (2026-07-16, 106-agent verified pass) applied to our actual stack:
25M ternary bidirectional encoder (conversion) + ~33M autoregressive predictor (next-word).*

## The single most important finding

**Latency is not the bottleneck — there is ~100× headroom.** A 3B ternary model runs at
11 tok/s (~90ms/tok) on a Raspberry Pi 5 (T-MAC, arXiv 2407.00088); our models are 25–33M,
30–120× smaller. The measured ~9ms/forward is already near the floor. **Optimize for quality,
not speed** — the ≤20ms budget can absorb a bigger/better model.

Corollary: the current app's 40ms encoder decode is an *implementation* gap (unoptimized
release path), not an architectural ceiling.

## Encoder (conversion) — keep the shape, improve the training

| Lever | Verdict | Why |
|---|---|---|
| bitnet.cpp / T-MAC kernel swap | **Skip** | At 125M the gain over fp16 is 1.37–1.6× on ARM (arXiv 2410.16144); we are *already* TQ2_0 ternary, so realizable gain <1.4×. Tuning, not a step change. |
| Non-autoregressive one-pass encoder | **Keep — validated** | One forward + per-position constrained classification is the ideal CPU shape: no KV cache, no per-token loop. |
| MiniPLM offline distillation | **The quality lever** | Difference-sampling KD (top-K log p_teacher/p_ref) matches online KD at ~2.2× lower compute, one teacher pass reusable across students (arXiv 2410.17215). This is how to push toneless past 84.9% without going autoregressive. |

## Predictor (next-word) — three evidence-backed upgrades

| Lever | Evidence | Action |
|---|---|---|
| **Sparse full-attention interleave** | 1 full-attn per 3–6 linear layers restores Transformer-level recall for <1% loss shift; Gated-DeltaNet peaks at 3:1 (arXiv 2507.06457, 2412.06464). BUT that recall advantage is a *long-context* effect (RULER/S-NIAH at 8K tokens). | **MEASURED, tied at our scale:** the hybrid (dense-Qwen3.5, attn every 4th) scored 41.3/71.5 next-word vs pure-GDN 45.6/77.2 on the same held-out set (n=351) — statistically tied / marginally behind. At our ≤10-token IME context the recall benefit does not materialize. **Keep the hybrid anyway** — it is the only one that deploys (official qwen35 GGUF), at no accuracy cost. |
| **Gated DeltaNet-2** | Channel-wise decoupled erase/write gates → 15.90 ppl vs GDN 16.40 at 1.3B, matched state (arXiv 2605.22791). | Drop-in successor when `fla` ships it — free quality at equal latency. |
| **Embedding+head compression** | Embedding and head are ~26% *each* of a 0.1B model (arXiv 2412.10856); our 16k word-piece predictor is directly hit. | SVD low-rank the embed/head → 3.4–5× smaller, negligible loss. This is where the params actually are. |

## Architecture choices now settled by evidence

- **Encoder for conversion** (non-AR, bidirectional, constrained) — right for CPU + homophones.
- **Hybrid (not pure) linear-attention for prediction** — the pure-GDN vs hybrid debate is
  resolved: interleave a few full-attention layers. We already convert this exact arch to GGUF.
- **KV-cache-free recurrent block** (Gated DeltaNet / RWKV-7) for streaming — constant per-token
  time/memory, ideal for incremental typing.

## Don't bother

- ❌ Kernel swap (bitnet.cpp/T-MAC) — <1.4× over existing ternary, not worth integration risk.
- ❌ Speculative decoding — pointless when a single forward is already ~9ms.
- ❌ Chasing latency generally — 100× headroom; spend the effort on quality.

## Prioritized roadmap

1. ~~Verify the hybrid predictor beats pure-GDN~~ — **DONE, resolved:** tied at our short
   context (41.3/71.5 vs 45.6/77.2, n=351). Keep the hybrid because it *deploys*, not because
   it's more accurate. The recall win is a long-context effect our IME task doesn't stress.
2. **MiniPLM distillation** into the encoder — the toneless-quality lever (highest value now).
3. **SVD-compress the predictor embedding/head** — where the params are (16k word-piece vocab).
4. **Then** the mechanical BOOX-ARM latency measurement (Android NDK build).

## What our own measurement adds to the research

The hybrid-vs-pure result is a genuine refinement: the literature's "interleave attention to
restore recall" is validated *at long context* but **does not help our ≤10-token IME task** —
consistent with our earlier 50M bake-off (parallel/hybrid only edged pure-attention slightly,
and at short context hybrids give quality not latency). Practical implication: **at IME context
length, the linear-attention backbone choice is roughly quality-neutral** — pick the one that
deploys and quantizes best (dense-Qwen3.5 hybrid), and spend the real quality budget on the
encoder (conversion is 86.8/84.9 and matters most) via distillation, plus head/embedding
compression on the predictor.

## Sources (all high-confidence, verified verbatim)

- BitNet / bitnet.cpp — arXiv 2410.16144, github.com/microsoft/BitNet
- T-MAC LUT kernels — arXiv 2407.00088 (EuroSys 2025)
- Gated DeltaNet — arXiv 2412.06464 (ICLR 2025); Gated DeltaNet-2 — arXiv 2605.22791
- RWKV-7 "Goose" — arXiv 2503.14456; RWKV-Lite — arXiv 2412.10856
- Hybrid ratio ablations — arXiv 2507.06457
- MiniPLM distillation — arXiv 2410.17215
- flash-linear-attention — github.com/fla-org/flash-linear-attention (no CPU benchmarks; measure your own)

---

# Empirical results (overnight run, 2026-07-16)

Every roadmap lever from the review, tested on our stack. Held-out evals; honest negatives kept.

## Predictor (next-word) — scale-up wins, ternary/KD/SVD don't

| Change | top-1 / top-5 | Verdict |
|---|---|---|
| 33M baseline (Qwen3.5 GDN hybrid) | 41.3 / 71.5 | — |
| **60M (dim640/d8/ffn2560), 3ep** | **47.3 / 75.8** | **WIN** — spend the latency headroom on size |
| 99M, 3ep | 39.9 / 74.4 | undertrained |
| 60M, 6ep | 34.2 / 63.0 | **overfits** — 3ep optimal, data-limited |
| corpus augmentation | — | no new data (conversion gold = corpus_e3 dups) |
| Q4-QAT | 44.2 / 74.4 | near-lossless — **deploy floor** |
| **Ternary-QAT** | **18.5 / 48.4** | **catastrophic** — generation over 16k vocab can't take ternary |

**Deployable predictor: 60M @ 3ep, Q4, 46 MB, 8.5 ms/word on BOOX.** Two-model e2e ≈17.5 ms.
Ternary is only for the encoder (constrained classification tolerates it; open generation doesn't).

## Encoder (conversion) — more epochs is a free win; data-cleaning hurts

| Change | toned / toneless | Verdict |
|---|---|---|
| baseline (RoBERTa warm-start, 2ep, .bin) | 86.8 / 84.9 | — |
| **.bin, 3ep** | **90.6 / 86.2** | **WIN** — +1 epoch, both axes up |
| all-zh jsonl, 3ep | 88.1 / 83.0 | worse (jsonl loses .bin's toneless augmentation) |
| toneless-upweighted all-zh, 3ep | 88.1 / 84.9 | trades toned for toneless |
| MiniPLM diff-sampling (drop noisy 15%) | 89.3 / 81.8 vs 90.6/83.0 | **HURTS** — low-confidence = useful hard examples |

**The encoder — the highest-value component — gains ~+4 toned / +1.3 toneless for one extra epoch.**
Retrain the shipping ternary encoder at 3ep. Ceiling sweep (4/5ep) in progress.

## Cross-cutting confirmations
- **Latency was never the bottleneck** — measured on BOOX: encoder ~9 ms, 60M-Q4 predictor 8.5 ms, e2e ~17.5 ms under the 20 ms budget. Every quality lever was affordable.
- **Task shape decides quantization tolerance** — ternary fine for the classifying encoder, fatal for the generating predictor.
- **T5 / enc-dec dropped** — 1:1 conversion + open generation are opposite shapes with no shared cross-attention need.
