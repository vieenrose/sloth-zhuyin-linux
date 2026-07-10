#!/bin/sh
# Fetch the SlothLM-E 4M ONNX model from Hugging Face into model/slothe_4m_onnx
# (what slothingd_e.py serves). Vocab/char tables come from the repo itself.
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_DIR/model/slothe_4m_onnx"
mkdir -p "$DEST"
URL="https://huggingface.co/Luigi/slothlm-e-4m-zhuyin/resolve/main"
echo "downloading model_quantized.onnx (5 MB) ..."
if command -v curl >/dev/null; then
    curl -sL "$URL/onnx/model_quantized.onnx" -o "$DEST/model_quantized.onnx"
else
    wget -q "$URL/onnx/model_quantized.onnx" -O "$DEST/model_quantized.onnx"
fi
cp "$REPO_DIR/space-static/enc/syl_vocab.json" "$DEST/"
cp "$REPO_DIR/space-static/enc/char2id.json" "$DEST/"
ls -la "$DEST"
echo "model ready at $DEST"
