#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== HONEST ternary-distilled 25M (working distillation), 16ep, GPU0, snap/4 START ==="; date
CUDA_VISIBLE_DEVICES=0 python3 train_slothe_ternary.py \
  --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer --out slothe_t_25m_honest \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant ternary --weight-quant median --pre-norm \
  --teacher slothe_32m_g2pw --distill-alpha 0.7 --distill-temp 2.0 --anneal-frac 0.15 \
  --batch 384 --epochs 16 --save-every 4 --lr 2.5e-3
echo "=== GATE slothe_t_25m_honest (held-out) ==="
CUDA_VISIBLE_DEVICES='' python3 gate_slothe_ternary.py --model slothe_t_25m_honest --tokenizer tokenizer --table phonetic_table.tsv --testset ../eval/testset.tsv --mspy ../eval/reference_heldout.tsv
echo "TERN_HONEST_DONE"; date
