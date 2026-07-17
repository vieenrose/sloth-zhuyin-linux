# Architectural review — latency & quality for the Sloth IME LLMs

*Deep-research synthesis (2026-07-16, 106-agent verified pass) applied to our actual stack:
25M ternary bidirectional encoder (conversion) + ~33M autoregressive predictor (next-word).*

## The single most important finding

**LATENCY CORRECTION (2026-07-17, measured on real BOOX):** the "~9ms encoder" figure used
below was a *projection* (the HF model card marks it `projected, not measured`); the review's
"100× headroom" premise inherited it. Real numbers, single 6-syl forward on BOOX SD662:
- 25M ternary @ old 8-thread default: **40.6ms** (the "40ms release gap" was thread
  oversubscription onto the A53 little cores, not an implementation gap)
- 25M ternary @ 4 threads: **18.5ms** (fixed default, commit fdaadc8)
- 25M ternary @ 2 threads: **31.9ms**
Latency is real but manageable: headroom is ~2×, not 100×. Quality levers below remain valid;
the speculative-decoding/kernel-swap dismissals hold for the opposite reason (no headroom to
need them at current size).

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

## Encoder scale-up — pretraining, not size (added)

| int4 SlothE-T | toned / toneless | Verdict |
|---|---|---|
| 25M (shipping) | ~82 / ~80 | — |
| 50M | 82.4 / 78.6 | flat |
| 75M | 84.9 / 78.6 | +2 toned, −toneless |
| *RoBERTa float 102M (ref)* | *90.6 / 86.2* | too big for budget |

Unlike the predictor, **encoder scale-up barely helps** — the gap to the RoBERTa is **pretraining +
float precision, not size**. From-scratch int4 is capped ~84/79; the RoBERTa reaches 90.6/86.2 but at
102M float it would be ~30–40 ms on BOOX (over budget).

**RoBERTa→SlothE-T soft-label distillation — TESTED, WIN on toned (added 2026-07-17).** Clean
same-harness ablation (eval_arch.py, n=159), int4 SlothE-T 16.4M, 2ep, teacher = RoBERTa 3ep (90.6/86.2):

| int4 SlothE-T (16.4M, 2ep) | toned / toneless |
|---|---|
| baseline (CE only) | 84.3 / 79.2 |
| **+ RoBERTa soft-label KD (kd=1.0)** | **86.8 / 79.2** |

KD adds `args.kd·KL(student ‖ teacher-softmax)` over the 5433/8342 shared single-Han-char columns at
non-ignore positions (`train_slothlm_e_kd.py`). Result: **+2.5 toned at zero inference cost** (identical
int4 model/latency) — the first lever to lift the from-scratch int4 toned ceiling toward the teacher's
90.6. **Toneless did not move** (79.2): the teacher's toneless edge (86.2) doesn't transfer through
char-space KD — likely because the teacher's own toneless strength comes from its `.bin` toned/toneless
augmentation, not from per-char logit shape the student can absorb. Net: distillation is a real,
deployable encoder lever for the toned axis; toneless still needs the `.bin` augmentation path (3ep
`.bin` teacher itself = 90.6/86.2). **Deploy: SlothE-T int4 + RoBERTa-KD for +2.5 toned free.**

## Overnight sweep — the complete map

**Wins:** predictor 60M/3ep (47.3/75.8, 8.5 ms Q4 on BOOX, e2e ~17.5 ms); encoder-float 3ep (90.6/86.2).
**Negatives, dropped with reasons:** ternary-predictor (generation can't take ternary), MiniPLM
diff-sampling (hard examples are useful), SVD (embedding not low-rank), encoder-scale-up (pretraining
> size), >3 epochs (overfit), T5/enc-dec (wrong shape), kernel-swaps (<1.4× over ternary).
**Laws that held:** latency was never the bottleneck; 3 epochs optimal for both models; ternary suits
classifiers not generators; scale-up helps generation-with-headroom (predictor) but not
pretraining-limited classification (encoder).
**Encoder distillation — DONE, WIN:** RoBERTa→SlothE-T soft-label KD = 84.3→86.8 toned (+2.5, free),
toneless flat 79.2. The deployable-encoder-accuracy lever confirmed for the toned axis; toneless still
needs `.bin` augmentation. Sweep complete.

---

# BOOX 2-thread latency sweep (2026-07-17, measured on device 800D1C1B)

Goal: best IME at ≤20ms e2e on BOOX with ONLY 2 threads (leave cores for system/app).
Method: synthetic random-weight models of candidate shapes (latency is shape-dependent,
value-independent), timed via the cached slothe_logits path, T=6, on the real device.

| Encoder shape (ternary) | params | median @2t | fits 20ms? |
|---|---|---|---|
| 352×16 ffn960 (current 25M) | 23M | 31.9ms | NO |
| 352×8 ffn960 | 17M | 21.1ms | borderline |
| 256×14 ffn1024 | 16M | 19.1ms | no headroom |
| **256×12 ffn768** | **11.6M** | **15.4ms** | **YES** |
| 256×8 ffn768 | 8.6M | 12.7ms | YES |
| 192×8 ffn512 | 5.6M | 9.0ms | YES |

| Predictor (Q4, llama-bench) | @2t ms/word |
|---|---|
| 60M | 13.1 |
| **33M** | **7.7** |

Findings:
- **dim=256 is free speed**: exactly TQ2_0 256-aligned → zero pad tax. The current dim=352
  pads to 512 = 31% wasted attention MACs (the model card's "256-alignment note" confirmed).
- Latency ~linear in T: 256×12 = 15.4ms @T6 but 29.7ms @T12. Budgets are per-6-syllable.
- Thread default fixed to 4 (was 8; little cores dragged 2.7×). SLOTHE_THREADS overrides.

**RESOLVED — the 2-thread stack (trained + evaluated 2026-07-17):**

| Encoder (KD, 2ep) | params | @2t | toned | toneless |
|---|---|---|---|---|
| 25M (352×16) reference | 23M | 31.9ms | 86.8 | 79.2 |
| **256×12 ffn768 — WINNER** | **11.6M** | **15.4ms** | **86.2** | **79.2** |
| 256×8 ffn768 | 8.6M | 12.7ms | 84.9 | 73.6 |

The 256×12 matches the 25M within noise (−0.6 toned, toneless identical) at half the params
and half the latency — dim-256 zero-pad-tax + RoBERTa-KD fully paid for the shrink. 256×8
marks the floor (toneless collapses). **SHIP: SlothE-T 256×12 KD (15.4ms) + 60M-Q4 predictor
(13.1ms/word, separate event; 33M @7.7ms for strict-sum). Both events ≤20ms @2 threads on
BOOX, measured.**

**DEVICE-VALIDATED (2026-07-17, real GGUF slothe-t-12m-256x12.gguf 9.65MB on BOOX):
15.8ms @2t (projection said 15.4) / 9.3ms @4t — the once-projected "9ms encoder" is now real,
one size down.** Gate accuracy 84%/84% (230-sent). Remaining: bundle into app assets + selfTest.

ARM references if BOOX unavailable: rpi4 `ssh raspberrypi` (Cortex-A72, no-dotprod — closest
proxy to the BOOX big cores), Jetson `ssh picard@picard-desktop` (CPU-only mode).
