#!/bin/sh
# Manual launcher for the Slothing LLM reranker (slothingd + LFM2.5-230M).
# Run this by hand whenever you want the reranker active; Ctrl+C to stop.
# eim.cpp talks to it over the Unix socket at $SOCKET below.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOCKET="${SLOTHINGD_SOCKET:-/tmp/slothingd.sock}"

# Restart loop: the daemon vanished once under heavy system load (likely
# OOM-killed; nothing in its own log), which silently disables reranking
# until someone notices. Ctrl+C still stops everything: SIGINT goes to the
# whole foreground process group, killing this loop along with the child.
while :; do
    echo "$(date '+%F %T') starting slothingd" >&2
    "$REPO_DIR/engine/slothingd/build/slothingd" \
        -m "$REPO_DIR/llm/models/lfm2.5-230m-q4/LFM2.5-230M-Q4_0.gguf" \
        -s "$SOCKET"
    echo "$(date '+%F %T') slothingd exited ($?), restarting in 2s" >&2
    sleep 2
done
