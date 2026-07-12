# Senior IME — R&D log (paused 2026-07-12)

Unfinished research toward a Slothing keyboard for seniors / aged users.
**Paused** to return to mainline ternary work; this branch preserves everything
needed to resume. Design report: [`docs/SENIOR-KEYBOARD.md`](../../docs/SENIOR-KEYBOARD.md).
Target device: **iPhone-17-class smartphone** (iOS custom keyboard extension).

## TL;DR — the load-bearing insight

**The senior IME is a *key-error-tolerating* IME, not a grouped keypad.** There is
one axis — how the decoder handles a finger on the wrong key:

- **Grouped keypad** (hard tolerance): pre-merge physically-adjacent keys into one.
  Loses information (which exact key) and forces relearning (new layout).
- **Standard board + 8-adjacency + tone-soft decode** (soft tolerance): keep every
  key, train the decoder to forgive slips. No info loss, **zero relearning**, and it
  **generalizes to every user** (fat-finger / tremor / walking-and-typing).

Soft tolerance dominates. The mechanism is the TAAI-2024 8-adjacency error model
(see [`docs/RELATED-WORK.md`](../../docs/RELATED-WORK.md)) + tone-soft masking.

## Measured results (held-out 500, honest, no leakage)

Anchor: exact syllable + tone input → 免選字 **72%**, char ~95% (the working baseline).

| experiment | per-pos candidates | char acc | 免選字 | verdict |
|---|---|---|---|---|
| **Candidate A** (10 grouped keys, tone DROPPED) | 722 | **51%** (ceiling) | ~4% | ❌ NO-GO |
| — pure-CE / cross-input-distill / (4× model, killed) | | all ~51% | | **information-limited**, not capacity |
| G10 + tone (~15 keys) | 241 | *(running when paused)* | | promising |
| **G21 + tone** (~25 keys, split columns) | **84.5 (≈2× exact)** | *(running when paused)* | | **most promising** |

**Key findings:**
1. Grouping (tone-dropped) is ×16–32 more ambiguous per position than exact input;
   the English "9.5% CER at 3 keys" study does NOT transfer (Chinese adds a
   homophone layer).
2. **Candidate A is a hard NO-GO at char 51%.** Pure-CE, cross-input distillation,
   and a 4× model all ceiling at ~51% → the tone-dropped grouping *discards the
   information irrecoverably*; no training method or model size recovers it.
3. **Tone is the lever, not key count.** Tone-drop alone was ×2.7 of the ambiguity
   blow-up. Keeping tone: G10+tone 241, G21+tone 84.5 — the latter is right next to
   the working exact input (44), so it should decode at near-exact accuracy.
4. Seniors won't change behavior to drop tones; keep tone as input but make the
   decoder **tolerate tone-typing errors** (read the tone, but mask over all tones
   so a tone slip never *excludes* the right char + tone-noise in training).

## Scripts (`scripts/`)

- `make_grouped_data.py` — Candidate A data: remap `train_e_g2pw.bin` syllables →
  10-group classes, tone dropped → `train_grouped.bin` + `grouped_syl_vocab.json`.
- `make_grouped2.py` — tone-KEPT data: builds `g10t` (10 col-groups + tone) and
  `g21t` (21 split-col groups + tone) → `train_g10t/g21t.bin`, `*_vocab.json`, `*_cfg.json`.
- `gate_grouped.py` — gate for Candidate A (tone-dropped, expanded legality mask).
- `gate_grouped2.py` — gate for tone-kept models. `--soft-tone` uses the tolerant
  mask (union over tones); default is the hard-tone ceiling.
- `inject_key_errors.py` — inject 8-adjacency key slips into a held-out set at rate
  `--p` → the noisy eval sets for the robustness curve.
- `run_grouped.sh` (Cand A pure-CE), `run_crossinput.sh` (Cand A cross-input
  distill), `run_g10t.sh` / `run_g21t.sh` (tone-kept, cross-input distill),
  `run_grouped_large.sh` (4× capacity test).

## Trainer changes (`trainer-patches/`)

These `patch_*.py` were applied to the mainline `model/train_slothe_ternary.py`
(untracked, workstation-only). Kept here to document and re-apply if needed:
- `patch_distill.py` — restored `load_teacher` + `distill_loss` (they had been lost;
  distillation was silently broken → the feat run was pure-CE, not distilled).
- `patch_snap.py` — `--save-every N` (per-epoch snapshots for accuracy-vs-epoch curves).
- `patch_crossinput.py` — `--teacher-data` (cross-input distillation: teacher reads
  exact syllables, student reads the grouped/coarse input; same records).
- `patch_8adj.py` — 8-adjacency typo model (slips constrained to QWERTY 3×3
  neighbours instead of arbitrary edit-distance-1).

## Regenerate data / checkpoints

Data `.bin` are 96 MB each and NOT committed. Regenerate on the workstation
(`~/sloth-zhuyin-linux/model/`) from `train_e_g2pw.bin` + `syl_vocab.json`:
```
python3 make_grouped_data.py     # Candidate A (train_grouped.bin)
python3 make_grouped2.py         # tone-kept (train_g10t.bin, train_g21t.bin)
python3 inject_key_errors.py --in ../eval/reference_heldout.tsv --p 0.1   # noisy eval
```
Checkpoints (workstation): `slothe_grouped_a` (Cand A pure-CE), `slothe_grouped_xin`
(cross-input distill), `slothe_g10t` / `slothe_g21t` (tone-kept, may be partial —
killed at pause). Teacher: `slothe_32m_g2pw`.

## Resume plan

1. **Read g10t/g21t** (re-run if killed mid-way): `gate_grouped2.py --cfg
   {g10t,g21t}_cfg.json` with and without `--soft-tone`. Expectation: g21t leaps
   toward the exact input's char ~95% / 免選字 ~72% → validates the tone-kept design.
2. **Robustness curve** (the real senior-IME spec): train the STANDARD 37-key
   decoder with heavy 8-adjacency + tone-noise, gate on `eval/heldout_e{10,20,30}.tsv`
   → plot accuracy vs slip-rate. Because it's the standard board, it ships to everyone.
3. **Hybrid product**: moderate key-reduction (bigger targets → fewer slips) +
   neighbour-tolerant decode (forgive the rest) + tone-soft. All three are now
   measurable levers.

Open item not yet built: **tone-noise** training augmentation (adjacent tone-key
slips) — needed to make the soft-tone tolerance real, not just mask-level.
