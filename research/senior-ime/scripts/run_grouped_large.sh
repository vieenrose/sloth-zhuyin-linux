#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== GROUPED-A LARGE: dim512 depth24 ffn2048 (~95M), fp, cross-input distill, 16ep, DDP, snap/4 START ==="; date
python3 -m torch.distributed.run --nproc_per_node=2 train_slothe_ternary.py \
  --data train_grouped.bin --vocab grouped_syl_vocab.json --tokenizer tokenizer \
  --out slothe_grouped_large \
  --dim 512 --depth 24 --heads 8 --kv-heads 2 --ffn 2048 --embed-norm \
  --quant fp --weight-quant median --pre-norm \
  --teacher slothe_32m_g2pw --teacher-data train_e_g2pw.bin \
  --distill-alpha 0.7 --distill-temp 2.0 --anneal-frac 0.15 \
  --batch 256 --epochs 16 --save-every 4 --lr 2.0e-3
echo "LARGE_DONE"; date
