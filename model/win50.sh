#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python3 distill_student.py --teacher x --no-teacher --qat --data gen_pairs_mix.jsonl --vocab student_vocab.json \
  --out slothe_50m_qat --dim 640 --depth 10 --heads 10 --kv 2 --ffn 1792 \
  --epochs 3 --batch 48 --lr 3e-4 --maxlen 32 --max-pairs 2000000
echo "TRAIN_DONE"
{ echo "=== 50M wide-shallow (640x10) + Q4 QAT (deployed ~13MB) vs 79M 83.6/72.3 ==="
  python3 allin1_eval.py slothe_50m_qat 2>/dev/null; echo "=== DONE ==="; } > win50_eval.txt 2>&1
