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

| | sentence | char |
|---|---|---|
| chewing alone | 20/30 (67%) | 93.4% |
| LLM top-1 | 25/30 (83%) | 95.8% |

Median latency 882ms, p95 2980ms (CPU, 22-core Xeon-class desktop).

Known LLM regressions at baseline (targets for fine-tuning):
- 音樂 → 陰月 (semantic miss)
- 個 → 个 (simplified variant preferred from candidate set)
- 嗎 → 嘛 (final-particle confusion, also missed by chewing)
