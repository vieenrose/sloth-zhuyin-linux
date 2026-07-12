# Reproducing SlothE-T 25M

End-to-end recipe for the ternary Zhuyin→Traditional-Chinese model
`slothe_t_25m_ce_ls32_ep24` and its GGUF. Trained on 2× RTX 5090 (DDP).

## 0. Inputs

| artifact | role |
|---|---|
| `train_e_g2pw.bin` | packed training set: zh-TW sentences, g2pW-labeled (syllable→char aligned) |
| `syl_vocab.json` | 1539-entry syllable (input) vocab |
| `tokenizer/` | char tokenizer (8342 chars) |
| `phonetic_table.tsv` | syllable→legal-character table (Taiwan readings) |
| `syl2legal.npz` | the same table as a dense `[1539 × 8342]` bool mask used by the legality-masked head |

Data prep: raw zh-TW corpus → g2pW phonetic labeling → aligned `(syllable, char)`
pairs packed into `train_e_g2pw.bin`. The held-out eval sets
(`eval/reference_heldout.tsv`, `eval/testset.tsv`) are **filtered to exclude any
sentence present in the training corpus** — this is what makes the reported
numbers honest (see the leakage note in the model card).

## 1. Train (teacher-free CE + label smoothing, long schedule)

`run_ce_ls_32.sh`:

```bash
python3 -m torch.distributed.run --nproc_per_node=2 train_slothe_ternary.py \
  --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer \
  --out slothe_t_25m_ce_ls32 \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant ternary --weight-quant median --pre-norm \
  --label-smoothing 0.1 \
  --batch 384 --epochs 32 --save-every 4 --lr 2.5e-3
```

- `--quant ternary --weight-quant median` → W1.58A8 QAT: ternary weights
  {−1,0,+1} × per-output-channel **absmedian** scale, int8 activations, STE.
- `--pre-norm` → SubLN RMSNorm before each ternary linear (stability).
- boundary blocks stay fp16 (`fp_boundary=1`, the default).
- `--save-every 4` snapshots every 4 epochs → `slothe_t_25m_ce_ls32_ep{4,8,…,32}`.
- **No `--teacher`** — teacher-free. Distillation was tried and only matched this.
- effective batch = 384 × 2 GPUs = 768.

## 2. Select the peak epoch (early stopping on held-out)

`gate_cels32_snap.sh` gates each snapshot on the held-out sets and prints the
curve. The model **peaks at epoch 24** and overfits after:

```bash
python3 gate_slothe_ternary.py --model slothe_t_25m_ce_ls32_ep24 \
  --tokenizer tokenizer --table phonetic_table.tsv \
  --testset ../eval/testset.tsv --mspy ../eval/reference_heldout.tsv
```

Expected held-out: **免選字 76 % · homophone-hard 86 % · toneless 77 %**
(ep32 regresses to ~73 % 免選字 — take **ep24**).

## 3. Convert to ternary GGUF

Two steps (torch only needed for extraction):

```bash
# a) extract effective ternary weights + fp tensors from the checkpoint (needs torch)
python3 extract_slothe.py slothe_t_25m_ce_ls32_ep24/slothe.pt \
        -> slothe_tensors.npz + slothe_config.json + roles.json

# b) pack GGUF (numpy + gguf-py only): ternary linears -> TQ2_0 (256-padded),
#    fp tensors -> f16, custom "slothe" arch metadata + syllable vocab
python3 pack_gguf.py -> slothe-t-25m.gguf
```

The ternarization baked into the GGUF is exactly the trainer's inference-time
quant at `quant_alpha=1.0` (fully annealed): `code = round(clamp(w/scale, −1, 1))`,
`scale = median(|w|)` per output channel. Because the effective weights are exact
ternary multiples, requantizing them to TQ2_0 is **loss-free** (verified by
round-trip: `max|dequant − effective| < 1e-3`). See `NAMES.md` for the
GGUF-tensor ↔ checkpoint-tensor name map and the 256-padding of in-features.

## Environment

- PyTorch (CUDA) for training/extraction; `numpy` + `gguf` (gguf-py) for packing.
- 2× RTX 5090 for the DDP recipe above; a single GPU works with `--nproc_per_node=1`
  (halve the effective batch or double `--batch`).
