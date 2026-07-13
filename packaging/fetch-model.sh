#!/bin/sh
# Fetch the 25M ternary GGUF from Hugging Face into model/slothe_t_25m
# (what slothd_slothe serves). Vocab/char maps come from the repo itself.
#
# The previous 11.6M int8 ONNX (slothe_10m_onnx, served by slothd_e.py) is
# no longer the default; recover it from git history if you need the ONNX path.
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_DIR/model/slothe_t_25m"
mkdir -p "$DEST"
URL="https://huggingface.co/Luigi/slothe-t-25m-zhuyin/resolve/main"
echo "downloading slothe-t-25m.gguf (18 MB) ..."
if command -v curl >/dev/null; then
    curl -sL "$URL/slothe-t-25m.gguf" -o "$DEST/slothe-t-25m.gguf"
else
    wget -q "$URL/slothe-t-25m.gguf" -O "$DEST/slothe-t-25m.gguf"
fi
cp "$REPO_DIR/space-static/enc/syl_vocab.json" "$DEST/"
cp "$REPO_DIR/space-static/enc/char2id.json" "$DEST/"
ls -la "$DEST"
echo "model ready at $DEST (phonetic table: model/phonetic_table.tsv)"
