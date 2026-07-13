#!/usr/bin/env bash
# Build the fcitx5-slothing .deb (the fcitx5 addon). The decode daemon + model
# are NOT in the package — they come from ./install.sh (or packaging/
# fetch-model.sh + install-slothingd-service.sh); the addon behaves like stock
# chewing until slothingd is running.
#
# Needs: cmake, g++, pkg-config, fcitx5 dev headers, libchewing dev headers.
# Missing build deps are installed with apt (sudo) unless SLOTHING_NO_APT=1.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- build deps ----
need=""
pkg-config --exists Fcitx5Core 2>/dev/null || need="$need fcitx5 libfcitx5core-dev libfcitx5utils-dev libfcitx5config-dev"
pkg-config --exists chewing   2>/dev/null || need="$need libchewing3-dev"
command -v cmake >/dev/null 2>&1 || need="$need cmake"
command -v g++   >/dev/null 2>&1 || need="$need g++"
command -v cpack >/dev/null 2>&1 || need="$need cmake"
if [ -n "$need" ]; then
    if [ "${SLOTHING_NO_APT:-0}" = 1 ]; then
        die "missing build deps:$need (install them, or run without SLOTHING_NO_APT)"
    fi
    say "installing build deps (sudo):$need"
    sudo apt update && sudo apt install -y $need
fi

BUILD=engine/fcitx5-chewing/build_deb
say "configuring + building the fcitx5 addon"
cmake -S engine/fcitx5-chewing -B "$BUILD" -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

say "packaging (cpack -G DEB)"
( cd "$BUILD" && cpack -G DEB )
DEB="$(ls -t "$BUILD"/*.deb 2>/dev/null | head -1)"
[ -n "$DEB" ] || die "cpack produced no .deb"

say "built: $DEB"
cat <<EOF

Install the addon:   sudo apt install "$DEB"
Then the decoder:    ./install.sh          # builds the daemon, fetches the 18 MB model, sets up autostart
                     #  (or: packaging/fetch-model.sh && packaging/install-slothingd-service.sh)

The addon behaves like stock chewing until the slothingd daemon is running.
EOF
