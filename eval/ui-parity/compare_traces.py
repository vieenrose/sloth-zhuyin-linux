#!/usr/bin/env python3
"""Differential UI-parity suite: libchewing vs the Sloth IME web demo.

Feeds identical key sequences to both, compares the OBSERVABLE UI STATE after
every keystroke (converted-vs-bopomofo shape, candidate window, cursor,
commits) — never the chosen characters (the model is supposed to differ).
Parity becomes a number instead of case-by-case bug reports.

  python3 eval/ui-parity/compare_traces.py            # run the corpus
  python3 eval/ui-parity/compare_traces.py --random 20 --seed 7

Requires: /tmp/chewing_trace (see chewing_trace.c), a local web server on
:8777 serving space-static/, node+playwright.
"""
import argparse, json, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Deterministic scenarios: each exercises one interaction contract.
CORPUS = [
    ("type+commit",        "su3cl3<E>"),
    ("tone1 via space",    "w8 su3<E>"),
    ("backspace pending",  "su<B>u3<E>"),
    ("backspace syllable", "su3cl3<B><E>"),
    ("esc clears",         "su3cl3<ESC>su3<E>"),
    ("candidates open",    "su3cl3<D>"),
    ("candidate pick",     "su3cl3<D>1<E>"),
    ("cand esc closes",    "su3cl3<D><ESC><E>"),
    ("cursor left+cand",   "su3cl3<L><D>1<E>"),
    ("cursor home/end",    "su3cl3<H><N><E>"),
    ("mid-cursor typing",  "su3cl3<L>w8 <E>"),
    ("long sentence",      "ji3y94tj/6vup dl3xm4<E>"),
]

# random traces are built from VALID syllable chunks + controls: junk key
# runs diverge by design (Sloth IME auto-detects them as English — a superset
# feature chewing doesn't have).
ALPHA = ["su3", "cl3", "w8 ", "ji3", "y94", "dl3", "xm4", "vup ", "tj/6",
         "5k4", "rm,6", "2k7", "cp3", "1;4",
         "<D>", "<D>1", "<L>", "<R>", "<B>", "<ESC>", "<H>", "<N>", "<E>"]

# Documented INTENDED divergences (superset features), excluded from random
# generation rather than waived per-field:
#  - auto zh/en: invalid-syllable key runs become English (chewing: pending
#    bopomofo error state)
#  - Enter with an incomplete syllable commits the auto-detected run
#    (chewing ignores Enter in the bopomofo-pending error state)

# Documented, INTENDED differences (field -> reason). A mismatch on these
# fields alone counts as a waiver, not a failure.
WAIVERS = {
    # (none currently)
}


def chewing_trace(keys):
    env = {**os.environ, "CHEWING_PATH":
           os.path.expanduser("~/.local/share/libchewing")}
    out = subprocess.run(["/tmp/chewing_trace", keys], env=env,
                         capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def demo_trace(keys, base):
    out = subprocess.run(
        ["node", os.path.join(HERE, "demo_trace.mjs"), keys, base],
        capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def compare(name, keys, base, verbose):
    ct, dt = chewing_trace(keys), demo_trace(keys, base)
    if len(ct) != len(dt):
        print(f"FAIL {name}: trace length {len(ct)} vs {len(dt)}")
        return False
    diffs = []
    for i, (a, b) in enumerate(zip(ct, dt)):
        for f in ("zh", "bopo", "cand", "cursor"):
            if a[f] != b[f]:
                diffs.append((i, a["key"], f, a[f], b[f]))
        # commit: compare 0/nonzero (length can differ by en spacing)
        if (a["commit"] > 0) != (b["commit"] > 0):
            diffs.append((i, a["key"], "commit", a["commit"], b["commit"]))
    if diffs:
        print(f"FAIL {name} ({keys}): {len(diffs)} state mismatches")
        for i, k, f, av, bv in diffs[: 6 if not verbose else 99]:
            print(f"    step {i} key '{k}': {f} chewing={av} sloth={bv}")
        return False
    print(f"PASS {name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8777")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cases = list(CORPUS)
    rng = random.Random(args.seed)
    for r in range(args.random):
        keys = "".join(rng.choice(ALPHA) for _ in range(rng.randint(4, 16)))
        cases.append((f"random-{r}", keys + "<E>"))

    passed = sum(compare(n, k, args.base, args.verbose) for n, k in cases)
    print(f"\n{passed}/{len(cases)} scenarios at UI parity")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
