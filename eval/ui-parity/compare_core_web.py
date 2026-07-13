#!/usr/bin/env python3
"""Internal 4-way UI-parity: the Sloth IME web demo vs the SHARED C++ core.

fcitx5, IBus and Android all drive engine/common/core.h (ComposingCore +
ChoosingCore) *by construction*, so they behave identically to one another. The
only independent reimplementation of the interaction logic is the web demo
(space-static/ime.js). This harness pins that reimplementation to the shared
core keystroke-for-keystroke, using the same structural schema and scenarios as
compare_traces.py — except the external reference (libchewing) is replaced by
our own C++ core via eval/ui-parity/core_trace.cpp.

  g++ -std=c++17 -I android/app/cpp -I engine/common \
      eval/ui-parity/core_trace.cpp -o /tmp/core_trace
  (cd space-static && python3 -m http.server 8777 &)
  python3 eval/ui-parity/compare_core_web.py

Requires: /tmp/core_trace, model/phonetic_table.tsv, a web server on :8777,
node + playwright. Structural fields only (zh, bopo, cand, cursor, commit) —
never the chosen characters (the model is supposed to differ).
"""
import argparse, json, os, subprocess, sys

from compare_traces import CORPUS, demo_trace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CORE_BIN = os.environ.get("CORE_TRACE", "/tmp/core_trace")
TABLE = os.path.join(ROOT, "model", "phonetic_table.tsv")

# Forced-English (lone-Shift <S>) scenarios. The web demo BUFFERS English into
# the preedit (editable, commit on Enter); the three C++ frontends PASSTHROUGH
# each key to the app immediately, matching 微軟新注音 / chewing / 自然 / 華碩.
# This is an intended, web-demo-only divergence (docs/INTERACTION-REVIEW.md):
# these scenarios are still run, but their mismatches are reported WAIVED, so a
# regression that made them *accidentally* agree or that broke a non-English
# field would still surface in the diff.
ENGLISH_WAIVED = [
    ("english word",      "<S>hi<E>"),
    ("english + space",   "<S>hi go<E>"),
    ("english then zhuyin", "<S>ok<S>su3<E>"),
]


def core_trace(keys):
    out = subprocess.run([CORE_BIN, keys, TABLE],
                         capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def diff_fields(a, b):
    d = []
    for f in ("zh", "bopo", "cand", "cursor"):
        if a[f] != b[f]:
            d.append((f, a[f], b[f]))
    # commit: 0/nonzero only (byte length differs by en spacing)
    if (a["commit"] > 0) != (b["commit"] > 0):
        d.append(("commit", a["commit"], b["commit"]))
    return d


def compare(name, keys, base, waived, verbose):
    core, web = core_trace(keys), demo_trace(keys, base)
    if len(core) != len(web):
        tag = "WAIVE" if waived else "FAIL"
        print(f"{tag} {name}: trace length core={len(core)} web={len(web)}")
        return waived
    diffs = []
    converting = False  # a conversion (↓ window / post-pick converted display)
    for i, (a, b) in enumerate(zip(core, web)):
        # The `cursor` field only has a shared meaning in plain composing. Once
        # a conversion is active — the modal candidate window is open, OR it has
        # been picked/closed but the buffer is still in the converted "choosing"
        # display — the two UIs track different things (web keeps the insertion
        # cursor at the buffer end; the C++ core reports the focused segment),
        # a documented UI-specific deviation. It resets when the sentence
        # commits. cursor is still compared everywhere else (see cursor
        # home/end, mid-cursor typing, and the pre-↓ arrow in cursor left+cand).
        active = converting or a["cand"] or b["cand"]
        for f, av, bv in diff_fields(a, b):
            if f == "cursor" and active:
                continue
            diffs.append((i, a["key"], f, av, bv))
        converting = active
        if a["zh"] == 0 and b["zh"] == 0 and (a["commit"] or b["commit"]):
            converting = False  # sentence committed → back to plain composing
    if diffs:
        tag = "WAIVE" if waived else "FAIL"
        note = "  [web-demo-only English buffering]" if waived else ""
        print(f"{tag} {name} ({keys}): {len(diffs)} state mismatches{note}")
        for i, k, f, av, bv in diffs[: 99 if verbose else 6]:
            print(f"    step {i} key '{k}': {f} core={av} web={bv}")
        return waived
    print(f"PASS {name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8777")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ok = total = 0
    for n, k in CORPUS:
        total += 1
        ok += compare(n, k, args.base, False, args.verbose)
    for n, k in ENGLISH_WAIVED:
        total += 1
        ok += compare(n, k, args.base, True, args.verbose)

    print(f"\n{ok}/{total} scenarios at UI parity "
          f"({len(CORPUS)} strict + {len(ENGLISH_WAIVED)} English-waived)")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
