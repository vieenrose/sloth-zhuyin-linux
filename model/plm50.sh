#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
# PREFIX-LM: identical 49M config to the pure-attention baseline (80.5/69.8),
# ONLY the attention mask differs (bidirectional over input span).
python3 distill_student.py --teacher x --no-teacher --prefix-lm --qat \
  --data gen_pairs_mix.jsonl --vocab student_vocab.json --out slothe_50m_plm \
  --dim 640 --depth 10 --heads 10 --kv 2 --ffn 1792 \
  --epochs 3 --batch 48 --lr 3e-4 --maxlen 32 --max-pairs 2000000
echo TRAIN_DONE
{ echo "=== 50M PREFIX-LM (bidir input + causal output) vs pure-attn causal 80.5/69.8, encoder 86.8/84.9 ==="
  python3 allin1_eval.py slothe_50m_plm 2>/dev/null; echo "=== DONE ==="; } > plm50_eval.txt 2>&1
