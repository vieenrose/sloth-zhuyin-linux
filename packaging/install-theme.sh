#!/bin/sh
# Install the Sloth fcitx5 themes (light + dark) into the user's fcitx5
# theme directory, and optionally select one. No sudo needed (per-user).
#
#   packaging/install-theme.sh          # install both themes
#   packaging/install-theme.sh --set    # ...and set light as the active theme
set -eu

SRC="$(cd "$(dirname "$0")/themes" && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/fcitx5/themes"
mkdir -p "$DEST"
cp -r "$SRC/Sloth" "$SRC/Sloth-Dark" "$DEST/"
echo "installed themes to $DEST (Sloth, Sloth-Dark)"

if [ "${1:-}" = "--set" ]; then
    CONF="${XDG_CONFIG_HOME:-$HOME/.config}/fcitx5/conf/classicui.conf"
    mkdir -p "$(dirname "$CONF")"
    if grep -q '^Theme=' "$CONF" 2>/dev/null; then
        sed -i 's/^Theme=.*/Theme=Sloth/' "$CONF"
    else
        printf 'Theme=Sloth\n' >> "$CONF"
    fi
    echo "set active theme to Sloth in $CONF"
    echo "restart fcitx5 (fcitx5 -r -d) to apply."
else
    echo "select it in fcitx5-configtool → Global Options → Theme,"
    echo "or re-run with --set."
fi
