#!/bin/bash
# usage: boox_measure.sh <hf_model_dir_on_workstation>  e.g. pred_q35_60m
# Assumes: model on ai-workstation, llcvt converter there, arm binaries local, BOOX connected.
set -e
M="$1"; ADB=~/Android/Sdk/platform-tools/adb
echo "=== convert $M -> GGUF on workstation ==="
ssh ai-workstation ". ~/slothc_venv/bin/activate && cd ~/sloth-zhuyin-linux/model && \
  python3 ~/llcvt/convert_hf_to_gguf.py $M --outfile ${M}.gguf --outtype f16 2>&1 | grep -aiE 'successfully|error' | tail -1"
scp -q ai-workstation:~/sloth-zhuyin-linux/model/${M}.gguf /tmp/${M}.gguf
echo "fp16 size: $(ls -la /tmp/${M}.gguf | awk '{print $5}')"
$ADB push /tmp/${M}.gguf /data/local/tmp/sloth/${M}.gguf >/dev/null 2>&1
$ADB shell "cd /data/local/tmp/sloth && ./llama-quantize ${M}.gguf ${M}-q4.gguf Q4_K_M >/dev/null 2>&1"
echo "=== BOOX latency: $M (fp16 + Q4) ==="
$ADB shell "cd /data/local/tmp/sloth && ./llama-bench -m ${M}.gguf    -p 8 -n 16 -t 4 -r 3 2>/dev/null" | grep -aE "qwen35|tg16"
$ADB shell "cd /data/local/tmp/sloth && ./llama-bench -m ${M}-q4.gguf -p 8 -n 16 -t 4 -r 3 2>/dev/null" | grep -aE "qwen35|tg16"
$ADB shell "ls -la /data/local/tmp/sloth/${M}-q4.gguf" | awk '{print "Q4 size:", $5}'
