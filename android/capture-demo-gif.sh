#!/usr/bin/env bash
# Capture the Android demo GIF on a connected BOOX (or any adb device) by
# INJECTING the dachen keystream as hardware keyevents — the same method that
# produced docs/android-boox-demo-v10.gif. Keyevents route through
# SlothingImeService.onKeyDown -> core.feedKey, so they are immune to the BOOX
# rotated-touch-panel bug (`input tap` coords are broken; keyevents are not).
#
# Sentence: 晚上熬夜看 world cup，白天在 louisa
#   - zh syllables via dachen letter+tone keys
#   - world / cup / louisa auto-detected as English by the DP segmenter
#     (validated offline: cup and louisa both resolve to en, not zhuyin)
#   - the space in "world cup" is a literal space inside an English run
#   - the fullwidth ，comes from Shift+Comma ('<' -> punctMap -> ，); it renders
#     fullwidth because 看 sits in the clause before it (punctInEnglishClause=false)
#
# Prereqs: Slothing IME selected as the active keyboard, a text field focused
# (e.g. a note/search box), device unlocked. Run:  bash android/capture-demo-gif.sh
set -euo pipefail

OUT=${OUT:-/tmp/android_demo_frames}
GIF=${GIF:-docs/android-boox-demo-v11.gif}
FLASH_MS=${FLASH_MS:-900}     # keyboard.flashKey hold time so each screencap catches the flash
SETTLE=${SETTLE:-0.85}        # wait after each key before screencap (e-ink refresh + flash)
CROP=${CROP:-}                # optional ffmpeg crop "w:h:x:y" for the IME region; empty = full screen
WIDTH=${WIDTH:-380}           # final gif width (matches README width)

adb wait-for-device
echo "== device: $(adb get-serialno)"

# --- keep the e-ink panel awake & responsive -------------------------------
adb shell svc power stayon true
adb shell dumpsys deviceidle disable >/dev/null || true
adb shell settings put global slothing_flash_ms "$FLASH_MS"
echo "== flash hold = ${FLASH_MS}ms"

rm -rf "$OUT"; mkdir -p "$OUT"
n=0
shot(){ adb exec-out screencap -p > "$OUT/f$(printf '%03d' "$n").png"; n=$((n+1)); }
key(){ adb shell input keyevent "KEYCODE_$1"; sleep "$SETTLE"; shot; }
punct_comma(){ adb shell input keycombination KEYCODE_SHIFT_LEFT KEYCODE_COMMA; sleep "$SETTLE"; shot; }

shot   # frame 0: empty field

# 晚 j03  上 g;4  熬 l6  夜 u,4  看 d04
for k in J 0 3  G SEMICOLON 4  L 6  U COMMA 4  D 0 4; do key "$k"; done
# world  (space)  cup
for k in W O R L D  SPACE  C U P; do key "$k"; done
# ，(fullwidth, Shift+Comma)
punct_comma
# 白 196  天 wu0 (toneless)  在 y94
for k in 1 9 6  W U 0  Y 9 4; do key "$k"; done
# louisa
for k in L O U I S A; do key "$k"; done

# commit (上字 = Enter)
adb shell input keyevent KEYCODE_ENTER; sleep "$SETTLE"; shot

echo "== captured $n frames in $OUT"

# --- assemble gif (6fps, snappy; palette for size) -------------------------
VF="fps=6,scale=${WIDTH}:-1:flags=lanczos"
[ -n "$CROP" ] && VF="crop=${CROP},${VF}"
ffmpeg -y -v error -framerate 6 -pattern_type glob -i "$OUT/*.png" \
  -vf "${VF},palettegen=stats_mode=diff" /tmp/adpal.png
ffmpeg -y -v error -framerate 6 -pattern_type glob -i "$OUT/*.png" -i /tmp/adpal.png \
  -lavfi "${VF}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" -loop 0 "$GIF"
echo "== wrote $GIF ($(du -h "$GIF" | cut -f1))"

# restore
adb shell settings delete global slothing_flash_ms || true
adb shell svc power stayon false
