#!/usr/bin/env bash
# Proper end-to-end install of the Slothing IBus engine on Debian/Ubuntu/
# Kubuntu. Idempotent — safe to re-run. Build+install needs sudo; IBus
# enrollment is per-user (no sudo). It does NOT switch your input framework
# or log you out — those are the last manual steps, printed at the end,
# because env vars only take effect at the next login.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!  %s\033[0m\n' "$*"; }

# 1. build dependencies -----------------------------------------------------
say "checking build dependencies"
missing=""
for p in cmake g++ pkg-config ibus; do
    command -v "$p" >/dev/null 2>&1 || missing="$missing $p"
done
dpkg -s libibus-1.0-dev >/dev/null 2>&1 || missing="$missing libibus-1.0-dev"
if [ -n "$missing" ]; then
    say "installing (sudo):$missing"
    sudo apt update && sudo apt install -y $missing
fi

# 2. build + install the engine --------------------------------------------
say "building the IBus engine"
cmake -S engine/ibus-slothing -B engine/ibus-slothing/build \
      -DCMAKE_INSTALL_PREFIX=/usr >/dev/null
cmake --build engine/ibus-slothing/build -j"$(nproc)"
say "installing (sudo): /usr/libexec + component XML + phonetic table"
sudo cmake --install engine/ibus-slothing/build

# 3. register the component with IBus ---------------------------------------
say "registering the component with IBus"
ibus write-cache 2>/dev/null || true
ibus restart     2>/dev/null || warn "ibus not active in this session yet (ok)"

# 4. ensure the decode daemon (slothingd) -----------------------------------
say "ensuring slothingd (the decode daemon — both frontends use it)"
SOCK="${XDG_RUNTIME_DIR:-/tmp}/slothingd.sock"
if [ -S "$SOCK" ] || systemctl --user is-active --quiet slothingd 2>/dev/null; then
    echo "   slothingd already running."
else
    if [ ! -f model/slothe_4m_onnx/model_quantized.onnx ]; then
        say "fetching model (~5 MB ONNX)"
        packaging/fetch-model.sh
    fi
    say "installing slothingd as a systemd user service (auto-start at login)"
    packaging/install-slothingd-service.sh
fi

# 5. enroll the engine for your user ----------------------------------------
say "enrolling 'slothing' in IBus preload-engines"
if command -v gsettings >/dev/null 2>&1; then
    cur="$(gsettings get org.freedesktop.ibus.general preload-engines 2>/dev/null || echo '@as []')"
    case "$cur" in
        *"'slothing'"*) echo "   already enrolled." ;;
        *) gsettings set org.freedesktop.ibus.general \
               preload-engines "['slothing']" ;;
    esac
else
    warn "gsettings not found; enroll via ibus-setup -> Input Method -> Add"
fi

# 6. verify the pieces are in place -----------------------------------------
say "verifying"
ok=1
[ -x /usr/libexec/ibus-engine-slothing ] || { warn "engine binary missing"; ok=0; }
[ -f /usr/share/ibus/component/slothing.xml ] || { warn "component XML missing"; ok=0; }
[ -f /usr/share/slothing/phonetic_table.tsv ] || { warn "phonetic table missing"; ok=0; }
[ -S "$SOCK" ] || warn "slothingd socket not up yet (starts at next login)"
[ "$ok" = 1 ] && echo "   engine files OK."

# 7. final manual step ------------------------------------------------------
cat <<'EOF'

============================================================================
Installed and enrolled. One manual step remains — make IBus your input
framework (env vars only apply at login):

  im-config -n ibus
  #  then LOG OUT fully and log back in,
  #  choosing "Plasma (X11)" at the SDDM login screen (NOT Wayland)

After relogin, verify and use:

  echo "$GTK_IM_MODULE $QT_IM_MODULE $XMODIFIERS"   # must read: ibus ibus @im=ibus
  ibus engine slothing            # or press Super+Space to switch to it

Then type   5k4  ->  ㄓㄜˋ  ->  這   (↓ opens candidates, Enter commits).

Note: a lone Shift tap toggles 中/英 (微軟-style English passthrough). If it
types English only, tap Shift once to switch back to Chinese.

To revert to fcitx5 later:  im-config -n fcitx5   (then log out/in)
============================================================================
EOF
