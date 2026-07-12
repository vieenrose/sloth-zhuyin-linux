#!/bin/bash
cd ~/sloth-zhuyin-linux/model
echo "=== GROUPED-A feasibility: fp pure-CE, no features, 16ep, DDP 2-GPU, snapshot/4 START ==="; date
python3 -m torch.distributed.run --nproc_per_node=2 train_slothe_ternary.py \
  --data train_grouped.bin --vocab grouped_syl_vocab.json --tokenizer tokenizer \
  --out slothe_grouped_a \
  --dim 352 --depth 16 --heads 8 --kv-heads 2 --ffn 960 --embed-norm \
  --quant fp --weight-quant median --pre-norm \
  --batch 384 --epochs 16 --save-every 4 --lr 2.5e-3
echo "GROUPED_TRAIN_DONE"; date
