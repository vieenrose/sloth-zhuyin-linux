# Evaluation — honest 免選字, and a leakage fix (2026-07-12)

## The leak we found

The 230-case `eval/reference_mspy.tsv` was built by **sampling sentences from the
training corpus** (`model/corpus_e3.txt`) and g2pW-labeling them
(`eval/build_reference_mspy.py`). Verification: **174/174** of the *sampled*
reference sentences are verbatim in training (only the ~56 hand-written seed
lines are not). So the 免選字 metric **rewarded memorization** and inflated every
whole-sentence number by ~12–18 points.

## Honest numbers (held-out)

`eval/reference_heldout.tsv` — **500 fresh sentences from c4-chinese-zhtw
(offset 600k), excluded against the training corpus**, g2pW-labeled the same way.
Whole-sentence exact 免選字:

| model | 免選字 (leaked, old) | **免選字 (held-out, honest)** | homophone (clean) | toneless (clean) |
|---|---|---|---|---|
| **12M int8 (shipped)** | 84% | **72%** | 82% | 79% |
| distilled ternary 25M (research) | 78% | **76%** | 83% | 81% |
| pure-CE ternary 25M (research) | 96% | **71%** | 84% | 75% |

Notes:
- The **shipped model's real 免選字 is ~72%**, not 84%. The homophone-hard (159)
  and toneless (159) sets were already clean (6/160 training overlap) and are
  unchanged — those are the trustworthy metrics all along.
- The pure-CE ternary's leaked 96% was **pure memorization** — it collapses to
  71% held-out. Distillation generalizes better (76% held-out), reversing the
  mid-experiment worry that distillation was hurting.
- A distilled **ternary** student (76 / 83 / 81) **beats the shipped 12M int8**
  (72 / 82 / 79) on all three honest metrics, at ~half the size — see the model
  experiment ledger. (Speed advantage is still a projection pending an on-device
  port of the ternary kernels.)

## Don't leak again

`eval/build_reference_mspy.py` now takes `--exclude <training_corpus>` and drops
any sampled sentence found in it. Always build the reference from data the model
never saw:

```bash
G2PW_CUDA=1 python3 eval/build_reference_mspy.py \
    --corpus <held-out corpus> --exclude model/corpus_e3.txt \
    --n 500 --out eval/reference_heldout.tsv
```
