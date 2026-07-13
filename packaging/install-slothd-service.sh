#!/bin/sh
# Install slothd as a systemd user service (auto-start at login,
# auto-restart on failure). Run as your normal user, no sudo.
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
sed "s|@REPO_DIR@|$REPO_DIR|g" "$REPO_DIR/packaging/slothd.service" \
    > "$UNIT_DIR/slothd.service"
systemctl --user daemon-reload
systemctl --user enable --now slothd.service
systemctl --user status slothd.service --no-pager | head -5
echo "slothd installed: starts at login, restarts on failure."
echo "  logs:   journalctl --user -u slothd -f"
echo "  stop:   systemctl --user disable --now slothd"
