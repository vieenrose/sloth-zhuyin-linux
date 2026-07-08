# Evaluation harness

Scores the full Slothing pipeline (libchewing harvest → slothingd → top-1)
against a homophone-trap test set, replicating the engine's exact harvest
logic so the numbers reflect what users actually get.

```sh
gcc eval/harvest.c -o eval/harvest $(pkg-config --cflags --libs chewing)
python3 eval/run_eval.py            # daemon must be running
```

- `harvest.c` — types a Dachen key sequence into libchewing and emits the
  same interval/candidate JSON the engine sends (cand-list walk, span
  filter, dry-interval fallback).
- `testset.tsv` — bopomofo + expected sentence (+ optional context), 30
  cases weighted toward 在/再, 的/得, 他/她-style traps.
- `run_eval.py` — converts bopomofo→keys, harvests, queries the daemon,
  reports chewing-baseline vs LLM-top-1 sentence/char accuracy and latency.

## Baseline (2026-07-08, LFM2.5-230M Q4_0, n=4, no fine-tuning)

Measured on the **80-case** set. (An earlier 30-case sample showed the LLM at
83% sentence vs chewing 67% — that set was unrepresentatively favourable;
the larger set tells the real story.)

| metric | chewing alone | LLM (base model) |
|---|---|---|
| top-1 sentence | 50/80 (62%) | 46/80 (58%) |
| top-1 char | 93.6% | 90.4% |
| **recall (right answer in candidate list)** | 62% (single answer) | **76%** |

Median latency ~1.0s, p95 ~3.4s (CPU, 22-core desktop).

**Reading this honestly:** with the *un-fine-tuned* base model, the LLM's
**top-1 pick is a slight regression** vs chewing's own 1-best. But the
pick-from-list UX doesn't rely on top-1 — it relies on the right answer being
*reachable* in the candidate list, and there the LLM (76%) clearly beats
chewing's single guess (62%). So the base model already widens reachability;
the remaining job — **the whole point of `finetune/`** — is to convert that
recall headroom into a reliable top-1 ranking. Recall is itself capped by
chewing's candidate coverage (the grammar can only reorder real candidates,
by design), so ~76% is roughly the ceiling reranking alone can hit on this set
without looser candidate generation.

Known LLM top-1 misses (fine-tuning targets): 音樂→陰月 (semantic),
個→个 (simplified variant), 嗎→嘛 (final particle).
