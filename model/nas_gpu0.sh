#!/bin/bash
cd ~/sloth-zhuyin-linux/model
source ~/slothc_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
# name dim depth heads kv ffn
CFGS=("wide-shallow 640 10 10 2 1792" "deep-narrow 512 16 8 2 1408" "ffn-heavy 544 10 8 2 2176")
for cfg in "${CFGS[@]}"; do
  read name dim depth heads kv ffn <<< "$cfg"
  out="nas_$name"
  python3 distill_student.py --teacher x --no-teacher --data gen_pairs_mix.jsonl --vocab student_vocab.json \
    --out $out --dim $dim --depth $depth --heads $heads --kv $kv --ffn $ffn \
    --epochs 1 --batch 48 --lr 3e-4 --maxlen 32 --max-pairs 500000 > nas_${name}.log 2>&1
  { echo "### $name (dim=$dim depth=$depth ffn=$ffn)"; python3 allin1_eval.py $out 2>/dev/null | grep INSIDE; } >> nas_results_gpu0.txt
done
echo "GPU0_NAS_DONE" >> nas_results_gpu0.txt
