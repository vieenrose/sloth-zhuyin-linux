#!/bin/sh
# Manual launcher for the Slothing decode daemon (slothingd_e.py + SlothLM-E
# 4M ONNX). Run this by hand whenever you want LLM decode active; Ctrl+C to
# stop. The fcitx5 engine talks to it over the socket derived below.
# (The old llama.cpp/GGUF slothingd remains at engine/slothingd/build/.)

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Socket path: keep this derivation identical to slothingdSocketPath() in
# engine/fcitx5-chewing/src/eim.cpp and default_socket_path() in slothingd.cpp
# so both ends always agree. Precedence: $SLOTHINGD_SOCKET, then a per-user
# private path under $XDG_RUNTIME_DIR, then /tmp as a last resort.
if [ -n "$SLOTHINGD_SOCKET" ]; then
    SOCKET="$SLOTHINGD_SOCKET"
elif [ -n "$XDG_RUNTIME_DIR" ]; then
    SOCKET="$XDG_RUNTIME_DIR/slothingd.sock"
else
    SOCKET="/tmp/slothingd.sock"
fi

MODEL="$REPO_DIR/model/slothe_4m_onnx"
TABLE="$REPO_DIR/model/phonetic_table.tsv"
DAEMON="$REPO_DIR/engine/slothingd/slothingd_e.py"

# Restart loop: the daemon vanished once under heavy system load (likely
# OOM-killed; nothing in its own log), which silently disables reranking until
# someone notices. Ctrl+C still stops everything: SIGINT reaches the whole
# foreground process group, killing this loop along with the child.
#
# Backoff/cap: a config error (missing model/binary, unbindable socket) makes
# slothingd exit instantly; without a cap this loop would hot-restart a ~200MB
# model reload every 2s forever, scrolling the real error away. Stop after a
# run of rapid consecutive failures instead.
fails=0
while :; do
    start=$(date +%s)
    echo "$(date '+%F %T') starting slothingd on $SOCKET" >&2
    python3 "$DAEMON" --model "$MODEL" --table "$TABLE" -s "$SOCKET"
    code=$?
    end=$(date +%s)

    if [ $((end - start)) -lt 5 ]; then
        fails=$((fails + 1))
    else
        fails=0
    fi
    if [ "$fails" -ge 5 ]; then
        echo "$(date '+%F %T') slothingd exited ($code) in <5s, 5 times running -- giving up. Check the model path ($MODEL) and socket ($SOCKET)." >&2
        exit 1
    fi
    echo "$(date '+%F %T') slothingd exited ($code), restarting in 2s" >&2
    sleep 2
done
