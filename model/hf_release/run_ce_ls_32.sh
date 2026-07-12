#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== DIRECT CE ternary 25M, label-smoothing 0.1, teacher-free, 32ep (train-longer), DDP, snap/4 START ==="; date
python3 -m torch.distributed.run --nproc_per_node=2 train_slothe_ternary.py \
  --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer --out slothe_t_25m_ce_ls32 \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant ternary --weight-quant median --pre-norm \
  --label-smoothing 0.1 \
  --batch 384 --epochs 32 --save-every 4 --lr 2.5e-3
echo "CE_LS32_DONE"; date
