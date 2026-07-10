# UI-parity suite: libchewing vs Slothing

Differential testing so behavior parity with chewing is **measured, not
discovered case-by-case**: identical key sequences go to real libchewing and
to the Slothing web demo; the observable UI state after every keystroke is
compared structurally (converted-vs-bopomofo shape, candidate window
open/closed, cursor, commit events — never the chosen characters, which the
model is *supposed* to improve on).

## Setup (once)

```sh
git clone --depth 1 --branch v0.5.1 https://github.com/chewing/libchewing ~/libchewing
cmake -S ~/libchewing -B ~/libchewing/build -DCMAKE_INSTALL_PREFIX=$HOME/.local \
      -DBUILD_TESTING=OFF -DWITH_SQLITE3=OFF && cmake --build ~/libchewing/build -j && \
      cmake --install ~/libchewing/build
gcc eval/ui-parity/chewing_trace.c -o /tmp/chewing_trace \
    -I ~/.local/include/chewing -L ~/.local/lib -lchewing -Wl,-rpath,$HOME/.local/lib
npm i playwright   # in eval/ui-parity or reachable via NODE_PATH
```

## Run

```sh
(cd space-static && python3 -m http.server 8777 &)
python3 eval/ui-parity/compare_traces.py --random 15
```

Deterministic scenarios cover one interaction contract each (typing, tones,
backspace granularity, Esc semantics, candidate open/pick/close, cursor
movement, mid-cursor editing). `--random N` adds seeded traces built from
valid syllable chunks + control keys.

## Current score: 23/27 (12/12 deterministic, 11/15 random)

Contracts this suite has already caught and fixed:
- Esc must clear only the pending bopomofo, never converted text
- ↓ targets the char AT the cursor (last char at end of buffer), moves the
  cursor onto it, restores it on close
- a pick CLOSES the candidate window (no focus auto-advance)
- the candidate window is modal: typing/Enter ignored, ←→ page candidates
- arrows are ignored while a syllable is pending
- Home/End cursor movement

Known INTENDED divergences (Slothing superset features, excluded or expected
to fail in random traces):
- auto zh/en: invalid-syllable key runs become English; chewing treats them
  as a pending-bopomofo error state (its second-initial keystroke *replaces*
  the pending initial — ours flows into the keystream DP instead)
- Enter with an incomplete syllable commits the auto-detected run; chewing
  ignores Enter in the error state
- remaining random failures trace to chewing 0.5's inconsistent
  type-while-window-open edge semantics

The fcitx engine shares behavior with the web demo by construction (same
segmenter with lock-step tests, same daemon); contracts fixed here are
mirrored in eim.cpp in the same commits.
