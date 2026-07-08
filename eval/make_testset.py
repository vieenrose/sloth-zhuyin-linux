#!/usr/bin/env python3
"""Turn a plain list of Traditional-Chinese sentences into validated
testset.tsv rows (bopomofo \t expected [\t context]).

Bopomofo is generated with pypinyin, then every row is validated by running
the real harvest tool: we type the derived key sequence into libchewing and
require chewing's buffer to have the SAME LENGTH as the sentence (content may
differ -- that difference is exactly the signal the eval measures). Rows that
don't validate are dropped with a reason, so the test set never contains a
case the pipeline can't even attempt.

Usage:
  python3 eval/make_testset.py --sentences corpus.txt >> eval/testset.tsv
"""
import argparse
import json
import os
import subprocess
import sys

from pypinyin import pinyin, Style

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST = os.path.join(HERE, "harvest")

KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s",
    "ㄌ": "x", "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m", "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".", "ㄢ": "0", "ㄣ": "p", "ㄤ": ";",
    "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}
TONES = "ˊˇˋ˙"


def sentence_bopomofo(s):
    out = []
    for group in pinyin(s, style=Style.BOPOMOFO, errors="ignore"):
        syl = group[0]
        if not syl or any(c not in KEYMAP for c in syl):
            return None
        out.append(syl)
    return out if len(out) == len(s) else None


def keys(syllables):
    k = []
    for syl in syllables:
        k += [KEYMAP[c] for c in syl]
        if syl[-1] not in TONES:
            k.append(" ")
    return "".join(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    args = ap.parse_args()
    if not os.path.exists(HARVEST):
        sys.exit("build eval/harvest first (see eval/README.md)")

    kept = dropped = 0
    seen = set()
    for line in open(args.sentences, encoding="utf-8"):
        s = line.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        if not (2 <= len(s) <= 18):
            print(f"# drop (length) {s}", file=sys.stderr)
            dropped += 1
            continue
        syls = sentence_bopomofo(s)
        if syls is None:
            print(f"# drop (bopomofo) {s}", file=sys.stderr)
            dropped += 1
            continue
        r = subprocess.run([HARVEST, keys(syls)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"# drop (harvest) {s}", file=sys.stderr)
            dropped += 1
            continue
        buf = json.loads(r.stdout)["buffer"]
        if len(buf) != len(s):
            print(f"# drop (len mismatch buf={buf}) {s}", file=sys.stderr)
            dropped += 1
            continue
        print(f"{' '.join(syls)}\t{s}")
        kept += 1
    print(f"# kept {kept}, dropped {dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
