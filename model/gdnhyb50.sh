#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 distill_student.py --teacher x --no-teacher --ssm-type gdn --data gen_pairs_mix.jsonl --vocab student_vocab.json \
  --out slothe_50m_gdnhyb --dim 512 --depth 10 --heads 8 --kv 2 --ffn 1280 --hybrid ssm,ssm,par,ssm,ssm \
  --epochs 3 --batch 32 --lr 3e-4 --maxlen 32 --max-pairs 2000000
echo TRAIN_DONE
{ echo "=== 50M GDN-HYBRID (attn+GatedDeltaNet par) vs pure-GDN, pure-attn 80.5/69.8 ==="; python3 allin1_eval.py slothe_50m_gdnhyb 2>/dev/null; echo "=== DONE ==="; } > gdnhyb50_eval.txt 2>&1
