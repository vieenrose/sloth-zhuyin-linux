#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python3 distill_student.py --teacher x --no-teacher --ssm-type gdn --data gen_pairs_mix.jsonl --vocab student_vocab.json \
  --out slothe_50m_gdn --dim 512 --depth 11 --heads 8 --kv 2 --ffn 1280 --hybrid ssm \
  --epochs 3 --batch 48 --lr 3e-4 --maxlen 32 --max-pairs 2000000
echo TRAIN_DONE
{ echo "=== 50M pure-gdn vs pure-attn 80.5/69.8, pure-Mamba1 ==="; python3 allin1_eval.py slothe_50m_gdn 2>/dev/null; echo "=== DONE ==="; } > gdn50_eval.txt 2>&1
