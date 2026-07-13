# Evaluation harness

Scores the full Sloth IME pipeline (libchewing harvest → slothd → top-1)
against a homophone-trap test set, replicating the engine's exact harvest
logic so the numbers reflect what users actually get.

```sh
gcc eval/harvest.c -o eval/harvest $(pkg-config --cflags --libs chewing)
python3 eval/run_eval.py            # daemon must be running
```

- `harvest.c` — types a Dachen key sequence into libchewing and emits the
  same interval/candidate JSON the engine sends (cand-list walk, span
  filter, dry-interval fallback).
- `testset.tsv` — bopomofo + expected sentence (+ optional context), 159
  cases weighted toward homophone traps (在/再, 的/得/地, 他/她/它, 做/坐 ...); grow with make_testset.py.
- `run_eval.py` — converts bopomofo→keys, harvests, queries the daemon,
  reports chewing-baseline vs LLM-top-1 sentence/char accuracy and latency.

## Baseline (2026-07-08, LFM2.5-230M Q4_0, n=4, no fine-tuning)

Measured on the **159-case** set (grown via `make_testset.py`; earlier 30- and
80-case samples were smaller and read more favourably).

| metric | chewing alone | LLM (base model) |
|---|---|---|
| top-1 sentence | 97/159 (61%) | 84/159 (53%) |
| top-1 char | 92.8% | 89.6% |
| **recall (right answer in candidate list)** | 61% (single answer) | **73%** |

Per-char vs chewing: fixed 31, broke 69 (**net -38**). chewing's dominant
confusion is 它→他 (20x); the base LLM breaks via rare-char picks
(尼→你, 蘱→累, 詪→很). Median latency ~1.3s, p95 ~3.2s.

**Reading this honestly:** the un-fine-tuned base model's top-1 is a net
regression vs chewing; it wins only on recall (right answer reachable in the
pick-from-list UX: 73% vs 62%). Converting that recall headroom into a
reliable top-1 is the entire job of the SlothLM from-scratch model
(`model/DESIGN.md`) / fine-tuning (`finetune/`). Recall is capped by chewing's
candidate coverage (the grammar only reorders real candidates), so ~73% is
near the reranking-only ceiling on this set.

Known LLM top-1 misses (targets): 音樂→陰月 (semantic), 個→个 (variant),
嗎→嘛 (final particle).

