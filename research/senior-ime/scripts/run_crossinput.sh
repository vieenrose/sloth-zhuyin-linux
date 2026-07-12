#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== GROUPED-A cross-input distill: student=grouped, teacher=exact-32M, fp, 16ep, DDP, snap/4 START ==="; date
python3 -m torch.distributed.run --nproc_per_node=2 train_slothe_ternary.py \
  --data train_grouped.bin --vocab grouped_syl_vocab.json --tokenizer tokenizer \
  --out slothe_grouped_xin \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant fp --weight-quant median --pre-norm \
  --teacher slothe_32m_g2pw --teacher-data train_e_g2pw.bin \
  --distill-alpha 0.7 --distill-temp 2.0 --anneal-frac 0.15 \
  --batch 384 --epochs 16 --save-every 4 --lr 2.5e-3
echo "XIN_DONE"; date
