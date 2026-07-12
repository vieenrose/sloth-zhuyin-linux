#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== g10t (tone-kept, cross-input distill, fp, 16ep, GPU0, snap/4) START ==="; date
CUDA_VISIBLE_DEVICES=0 python3 train_slothe_ternary.py \
  --data train_g10t.bin --vocab g10t_vocab.json --tokenizer tokenizer --out slothe_g10t \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant fp --weight-quant median --pre-norm \
  --teacher slothe_32m_g2pw --teacher-data train_e_g2pw.bin \
  --distill-alpha 0.7 --distill-temp 2.0 --anneal-frac 0.15 \
  --batch 384 --epochs 16 --save-every 4 --lr 2.5e-3
echo "=== GATE g10t (hard then soft) ==="
for sflag in "" "--soft-tone"; do CUDA_VISIBLE_DEVICES='' python3 gate_grouped2.py --model slothe_g10t --cfg g10t_cfg.json --tokenizer tokenizer --table phonetic_table.tsv --mspy ../eval/reference_heldout.tsv $sflag 2>/dev/null; done
echo "g10t_DONE"; date
