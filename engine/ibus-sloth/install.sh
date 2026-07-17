#!/usr/bin/env bash
# Proper end-to-end install of the Sloth IME IBus engine on Debian/Ubuntu/
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
cmake -S engine/ibus-sloth -B engine/ibus-sloth/build \
      -DCMAKE_INSTALL_PREFIX=/usr >/dev/null
cmake --build engine/ibus-sloth/build -j"$(nproc)"
say "installing (sudo): /usr/libexec + component XML + phonetic table"
sudo cmake --install engine/ibus-sloth/build

# 3. register the component with IBus ---------------------------------------
say "registering the component with IBus"
ibus write-cache 2>/dev/null || true
ibus restart     2>/dev/null || warn "ibus not active in this session yet (ok)"

# 4. ensure the decode daemon (slothd) -----------------------------------
say "ensuring slothd (the decode daemon — both frontends use it)"
SOCK="${XDG_RUNTIME_DIR:-/tmp}/slothd.sock"
if [ -S "$SOCK" ] || systemctl --user is-active --quiet slothd 2>/dev/null; then
    echo "   slothd already running."
else
    if [ ! -f model/slothe_t_12m/slothe-t-12m-256x12.gguf ]; then
        say "fetching model (18 MB ternary GGUF)"
        packaging/fetch-model.sh
    fi
    say "installing slothd as a systemd user service (auto-start at login)"
    packaging/install-slothd-service.sh
fi

# 5. enroll the engine for your user ----------------------------------------
say "enrolling 'sloth' in IBus preload-engines"
if command -v gsettings >/dev/null 2>&1; then
    cur="$(gsettings get org.freedesktop.ibus.general preload-engines 2>/dev/null || echo '@as []')"
    case "$cur" in
        *"'sloth'"*) echo "   already enrolled." ;;
        *) gsettings set org.freedesktop.ibus.general \
               preload-engines "['sloth']" ;;
    esac
else
    warn "gsettings not found; enroll via ibus-setup -> Input Method -> Add"
fi

# 5b. icon into the user theme (KDE has no ibus autostart AND resolves the
#     component's <icon>ibus-sloth</icon> from the icon theme; the system
#     copy went in via `sudo cmake --install`, but drop a user copy too so the
#     sloth shows even without root and refresh the cache). -----------------
say "installing the sloth icon (ibus-sloth) into the user icon theme"
ICONSRC="engine/ibus-sloth/data/icons"
if [ -d "$ICONSRC" ]; then
    for s in 16x16 22x22 24x24 48x48; do
        install -Dm644 "$ICONSRC/$s/apps/ibus-sloth.png" \
            "$HOME/.local/share/icons/hicolor/$s/apps/ibus-sloth.png" 2>/dev/null || true
    done
    install -Dm644 "$ICONSRC/scalable/apps/ibus-sloth.svg" \
        "$HOME/.local/share/icons/hicolor/scalable/apps/ibus-sloth.svg" 2>/dev/null || true
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

# 5c. autostart: KDE ships no ibus autostart (the ibus package's file is
#     OnlyShowIn=GNOME), so ibus-daemon never launches at login on Plasma.
#     Drop a plain XDG autostart entry so it comes up every session. --------
say "adding ibus-daemon XDG autostart (KDE lacks one)"
mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/ibus-daemon.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=IBus daemon
Comment=Start the IBus input-method daemon (KDE ships no ibus autostart)
Exec=ibus-daemon --daemonize --xim --replace
Terminal=false
X-GNOME-Autostart-enabled=true
NoDisplay=true
DESK

# 6. verify the pieces are in place -----------------------------------------
say "verifying"
ok=1
[ -x /usr/libexec/ibus-engine-sloth ] || { warn "engine binary missing"; ok=0; }
[ -f /usr/share/ibus/component/sloth.xml ] || { warn "component XML missing"; ok=0; }
[ -f /usr/share/sloth/phonetic_table.tsv ] || { warn "phonetic table missing"; ok=0; }
[ -S "$SOCK" ] || warn "slothd socket not up yet (starts at next login)"
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
  ibus engine sloth            # or press Super+Space to switch to it

Then type   5k4  ->  ㄓㄜˋ  ->  這   (↓ opens candidates, Enter commits).

Note: a lone Shift tap toggles 中/英 (微軟-style English passthrough). If it
types English only, tap Shift once to switch back to Chinese.

To revert to fcitx5 later:  im-config -n fcitx5   (then log out/in)
============================================================================
EOF
