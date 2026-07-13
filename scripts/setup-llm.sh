#!/bin/sh
# One-command setup for the Sloth IME LLM runtime: clone + build llama.cpp,
# download the model, and build slothd. Idempotent -- safe to re-run; each
# step is skipped if already done. Does NOT install the fcitx5 addon (that
# needs sudo; see README "Build & install the fcitx5 addon").
#
# Requires: git, cmake, a C++ toolchain, pkg-config, libchewing dev headers,
# fcitx5 dev headers, and `hf` (pip install huggingface_hub) on PATH.
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LLAMA_DIR="llm/llama.cpp"
MODEL_DIR="llm/models/lfm2.5-230m-q4"
MODEL_FILE="$MODEL_DIR/LFM2.5-230M-Q4_0.gguf"
JOBS="$(nproc 2>/dev/null || echo 4)"

say() { printf '\033[1;32m==>\033[0m %s\n' "$1"; }

need() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'error: missing required tool: %s\n' "$1" >&2
        exit 1
    }
}
need git; need cmake; need pkg-config
pkg-config --exists chewing || {
    echo "error: libchewing dev headers not found (install libchewing3-dev)" >&2
    exit 1
}

# 1. llama.cpp (headers + libs that both slothd and the engine link against)
if [ ! -d "$LLAMA_DIR/.git" ]; then
    say "cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
    say "llama.cpp already present, skipping clone"
fi
if [ ! -e "$LLAMA_DIR/build/bin/libllama.so" ] && \
   [ ! -e "$LLAMA_DIR/build/src/libllama.so" ]; then
    say "building llama.cpp (this takes a few minutes)"
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build" -j"$JOBS" --target llama
else
    say "llama.cpp already built, skipping"
fi

# 2. model
if [ ! -f "$MODEL_FILE" ]; then
    need hf
    say "downloading LFM2.5-230M Q4_0 (~140 MB)"
    hf download LiquidAI/LFM2.5-230M-GGUF LFM2.5-230M-Q4_0.gguf \
        --local-dir "$MODEL_DIR"
else
    say "model already downloaded, skipping"
fi

# 3. slothd
if [ ! -x engine/slothd/build/slothd ]; then
    say "building slothd"
    cmake -B engine/slothd/build -S engine/slothd -DCMAKE_BUILD_TYPE=Release
    cmake --build engine/slothd/build -j"$JOBS"
else
    say "slothd already built, skipping"
fi

say "done. Start the reranker with: packaging/run-slothd.sh"
say "To build/install the fcitx5 addon (needs sudo), see the README."
