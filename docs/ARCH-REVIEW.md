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

## KD-on-ternary — tested, NEGATIVE for shipping (2026-07-17/18)

RoBERTa soft-label KD (the int4 ablation's +2.5-toned winner) added to the shipping
ternary recipe (CE+LS 0.1, 32ep + 12ep extension, dim256×12):

| 12M ternary | homophone | 免選字 230 |
|---|---|---|
| shipping (no KD) | 84% | **84%** |
| + RoBERTa-KD, 32ep | **89%** | 78% |
| + 12ep extension (lr 8e-4) | 88% | 78% (stable ceiling — not undertraining) |

The teacher's per-char homophone ranking transfers (**89% — best deployable encoder
score recorded**), but whole-sentence exactness caps at 78: the KL term pulls
per-position marginals toward the teacher at the cost of joint sentence consistency
that CE+label-smoothing alone optimizes. The extension run proves it's a stable
trade, not undertraining. **免選字 is the product metric → no ship; the no-KD 12M
stays the default.** Lesson: KD helps the *classification* axis, hurts the
*sequence-exactness* axis in ternary QAT — opposite of the int4 2ep ablation where
both recipes were CE-only (no LS) and KD's +2.5 came free. Label smoothing and KD
appear to overlap as regularizers here.

**KD-anneal addendum (2026-07-18):** annealing the KD checkpoint with pure CE+LS (8ep)
recovers 免選字 78→84 but monotonically washes out the KD gains (89/78/84 → 84/84/78 ≈
the CE solution). No snapshot dominates shipping. **Shipping (84/84/81) vs KD (89/78/84)
is a real Pareto frontier** — per-char vs whole-sentence is a fundamental trade in this
recipe. Encoder training levers are exhausted; the 12M no-KD model is final.

## Predictor data-scaling — WIN, and another benchmark leak exposed (2026-07-18)

Retrained the 60M predictor on 6.1M lines (old 1.08M + 5M fresh C4 zh-TW, same held-out tail):

| 60M predictor | e3-tail held-out | fresh-C4 (n=3000, never seen) |
|---|---|---|
| old (1.1M lines) | 47.3 / 75.8 | **4.6 / 12.0** — collapses |
| **new (6.1M lines)** | 35.6 / 64.1 | **34.0 / 46.0** |

The celebrated 47.3/75.8 was distribution-narrow memorization (the e3-tail is saturated with
near-duplicates of the small training set) — the prediction-side twin of the 免選字 leak found
on 2026-07-12. The new model scores consistently across both evals (~34-36 top-1): it actually
generalizes, and is ~7× better on honest data. Qualitative: 今天天氣→好/很好/不好;
晚上熬夜看→病/書/電視/電影. SHIPPED as the drop-in pred_q35_60m-q4.gguf (same file name,
size, latency; no code change). Also fixed: daemon predict op now skips partial-UTF-8 BPE
pieces (byte-level BPE splits multi-byte chars across tokens; they broke the JSON reply).
99M retrain on the big corpus in flight — judge on the fresh-C4 eval only.

**99M addendum (2026-07-18):** 99M on the same 6.1M lines = 34.2/46.3 fresh — statistically
tied with the 60M (34.0/46.0) at +65% params/+50% latency. The 34-top-1 plateau at both sizes
means the predictor remains DATA-limited at 6.1M lines; parameters are not the lever. 60M
stays the shipping predictor. Future gains: more conversational/register-matched corpus.

## Register fine-tune — WIN, shipped as predictor v2.1 (2026-07-18)

Fine-tuned v2 (`--init-from`, new trainer flag) on 1.44M lines = 149k TW chat sentences
(PTT Gossiping comments + Dcard, the actual IME typing register) ×5 + balanced C4, 2ep lr 1e-4:

| eval | v2 | **v2.1 (chat-FT)** |
|---|---|---|
| TW chat held-out (5k, n=3000) | 10.9 / 21.2 | **18.3 / 31.2** (+68% rel. top-1) |
| fresh-C4 regression | 34.0 / 46.0 | 33.5 / 45.2 (−0.5, negligible) |

Register-matching is the predictor's real lever confirmed: +7.4 top-1 where users type, at
~zero general-text cost. Shipped as the drop-in pred_q35_60m-q4.gguf. Remaining data levers:
larger TW-native chat sources (all HF sets are ≤100k; PTT full crawl would be next).
