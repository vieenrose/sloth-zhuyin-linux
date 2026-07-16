#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
python3 distill_student.py --teacher x --no-teacher --qat --data gen_pairs_mix.jsonl --vocab student_vocab.json \
  --out slothe_50m_hybrid --dim 512 --depth 12 --heads 8 --kv 2 --ffn 1280 \
  --hybrid ssm,ssm,par,ssm,ssm \
  --epochs 3 --batch 48 --lr 3e-4 --maxlen 32 --max-pairs 2000000
echo "TRAIN_DONE"
{ echo "=== 50M HYBRID (attn+mamba parallel, 49.3M) Q4-QAT vs pure-attn 50M ==="
  python3 allin1_eval.py slothe_50m_hybrid 2>/dev/null; echo "=== DONE ==="; } > hybrid50_eval.txt 2>&1
