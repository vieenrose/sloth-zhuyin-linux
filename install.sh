#!/usr/bin/env bash
# ============================================================================
# Slothing desktop installer — one command, fcitx5 or IBus.
#
#   ./install.sh              # auto-detect fcitx5 or IBus
#   ./install.sh fcitx5       # force fcitx5
#   ./install.sh ibus         # force IBus
#
# Idempotent: safe to re-run; each already-done step is skipped. Building the
# engine needs sudo (you'll be prompted); everything else is per-user. The very
# last step — making the IME active — is manual and printed at the end, because
# input-method env vars only take effect at your next login.
# ============================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO_DIR"
JOBS="$(nproc 2>/dev/null || echo 4)"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 0. pick the frontend --------------------------------------------------
FRONTEND="${1:-auto}"
if [ "$FRONTEND" = auto ]; then
    if   command -v fcitx5 >/dev/null 2>&1; then FRONTEND=fcitx5
    elif command -v ibus   >/dev/null 2>&1; then FRONTEND=ibus
    else die "neither fcitx5 nor ibus found. Install one, then re-run — or pass it explicitly: ./install.sh fcitx5|ibus"
    fi
fi
[ "$FRONTEND" = fcitx5 ] || [ "$FRONTEND" = ibus ] || die "unknown frontend '$FRONTEND' (use fcitx5 or ibus)"
say "installing for: $FRONTEND"

for t in git cmake g++; do command -v "$t" >/dev/null 2>&1 || die "missing build tool: $t (install build-essential, cmake, git)"; done

# ---- 1. ggml runtime (only ggml, NOT the full llama.cpp) -------------------
LLAMA_DIR="llm/llama.cpp"
if [ ! -e "$LLAMA_DIR/build/bin/libggml.so" ] && [ ! -e "$LLAMA_DIR/build/src/libggml.so" ]; then
    if [ ! -d "$LLAMA_DIR/.git" ]; then
        say "fetching the ggml runtime (llama.cpp checkout)"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    say "building ggml (a few minutes, one time)"
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$LLAMA_DIR/build" -j"$JOBS" --target ggml ggml-cpu ggml-base
else
    say "ggml runtime already built — skipping"
fi

# ---- 2. decode daemon (slothingd_slothe, links ggml only) ------------------
if [ ! -x engine/slothingd/build_slothe/slothingd_slothe ]; then
    say "building the decode daemon"
    cmake -S engine/slothingd -B engine/slothingd/build_slothe -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build engine/slothingd/build_slothe --target slothingd_slothe -j"$JOBS"
else
    say "decode daemon already built — skipping"
fi

# ---- 3. model (18 MB ternary GGUF) -----------------------------------------
if [ ! -f model/slothe_t_25m/slothe-t-25m.gguf ]; then
    say "downloading the 25M ternary model (18 MB)"
    packaging/fetch-model.sh
else
    say "model already present — skipping"
fi

# ---- 4. auto-start the daemon at login (systemd user service) --------------
say "installing the decode daemon as a login service"
packaging/install-slothingd-service.sh

# ---- 5. the frontend engine ------------------------------------------------
if [ "$FRONTEND" = fcitx5 ]; then
    say "building the fcitx5 addon"
    cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing -DCMAKE_INSTALL_PREFIX=/usr >/dev/null
    cmake --build engine/fcitx5-chewing/build -j"$JOBS"
    say "installing the fcitx5 addon (sudo)"
    sudo cmake --install engine/fcitx5-chewing/build
    command -v fcitx5-remote >/dev/null 2>&1 && fcitx5-remote -r 2>/dev/null || true
    cat <<EOF

============================================================================
Done. One manual step: add "Slothing" as an input method.

  • Open  Fcitx5 Configuration  (or run: fcitx5-configtool)
  • Add  "Slothing"  to your input-method list
  • Switch to it with  Ctrl+Space

Then type   5k4  ->  ㄓㄜˋ  ->  這   (↓ opens candidates, Enter commits).
A lone Shift tap toggles 中/英.
============================================================================
EOF
else
    say "installing the IBus engine (delegates to its own installer)"
    engine/ibus-slothing/install.sh
fi
