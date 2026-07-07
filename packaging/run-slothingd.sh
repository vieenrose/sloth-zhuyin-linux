#!/bin/sh
# Manual launcher for the Slothing LLM reranker (slothingd + LFM2.5-230M).
# Run this by hand whenever you want the reranker active; Ctrl+C to stop.
# eim.cpp talks to it over the Unix socket at $SOCKET below.

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOCKET="${SLOTHINGD_SOCKET:-/tmp/slothingd.sock}"

exec "$REPO_DIR/engine/slothingd/build/slothingd" \
    -m "$REPO_DIR/llm/models/lfm2.5-230m/LFM2.5-230M-Q8_0.gguf" \
    -s "$SOCKET"
