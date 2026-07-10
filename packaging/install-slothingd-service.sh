#!/bin/sh
# Install slothingd as a systemd user service (auto-start at login,
# auto-restart on failure). Run as your normal user, no sudo.
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
sed "s|@REPO_DIR@|$REPO_DIR|g" "$REPO_DIR/packaging/slothingd.service" \
    > "$UNIT_DIR/slothingd.service"
systemctl --user daemon-reload
systemctl --user enable --now slothingd.service
systemctl --user status slothingd.service --no-pager | head -5
echo "slothingd installed: starts at login, restarts on failure."
echo "  logs:   journalctl --user -u slothingd -f"
echo "  stop:   systemctl --user disable --now slothingd"
