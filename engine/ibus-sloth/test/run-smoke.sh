#!/usr/bin/env bash
# Headless end-to-end test: private dbus session + ibus-daemon + the sloth
# engine + the smoke client. Needs a running slothd (real decode) and the
# built binaries:
#   ENGINE=/path/to/ibus-engine-sloth SMOKE=/path/to/smoke ./run-smoke.sh
set -e
ENGINE="${ENGINE:-./ibus-engine-sloth}"
SMOKE="${SMOKE:-./smoke}"
export SLOTHING_PHONETIC_TABLE="${SLOTHING_PHONETIC_TABLE:-$(dirname "$0")/../../../model/phonetic_table.tsv}"
export SLOTHING_ASSOC_TABLE="${SLOTHING_ASSOC_TABLE:-$(dirname "$0")/../../../model/assoc_tc.tsv}"
# hermetic personal stores (assoc_user.tsv) so 聯想 assertions are deterministic
export XDG_DATA_HOME="$(mktemp -d)"

exec dbus-run-session -- bash -c '
  ibus-daemon --panel disable --daemonize --verbose 2>/dev/null
  sleep 1
  "'"$ENGINE"'" &
  ENGINE_PID=$!
  sleep 1
  "'"$SMOKE"'"; RC=$?
  kill $ENGINE_PID 2>/dev/null
  exit $RC
'
