#!/usr/bin/env python3
"""Build the phonetic table: bopomofo syllable -> legal Traditional characters.

This is the libchewing-free legality constraint: slothingd's decode mode maps
each typed syllable to this table's character set and builds a GBNF grammar
from it, so the model can only ever emit a character actually read that way.

Readings come from pypinyin (heteronym=True) -- deliberately the SAME source
prepare_data.py used to render the z2t/toneless training documents, so the
table's notion of "how is this character pronounced" matches what SlothLM was
trained on. Characters are ordered by corpus frequency (most frequent first,
unseen-in-corpus last) so candidate lists surface common chars first.

Output TSV, one syllable per line:  ㄨㄛˇ<TAB>我沃倭...

Usage:
  python3 model/build_phonetic_table.py \
      [--corpus model/corpus_big.txt] [--out model/phonetic_table.tsv]
"""
import argparse
import collections
import os
import sys

from pypinyin import Style, pinyin

HERE = os.path.dirname(os.path.abspath(__file__))
# Keep in sync with prepare_data.py: a syllable is only usable if every symbol
# is typeable on the default layout.
KEYMAP_CHARS = set(
    "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
    "ㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦˊˇˋ˙"
)

CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # Extension A (rare, but pypinyin knows some)
]


def corpus_freq(path):
    freq = collections.Counter()
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                freq.update(line.strip())
        print(f"corpus frequency from {path}: {len(freq)} distinct chars",
              file=sys.stderr)
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_big.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "phonetic_table.tsv"))
    ap.add_argument("--charset", default="big5",
                    help="only keep chars encodable in this charset "
                         "(big5 ~= Traditional/TW coverage, like libchewing; "
                         "'none' disables and admits simplified chars too)")
    args = ap.parse_args()

    def in_charset(ch):
        if args.charset == "none":
            return True
        try:
            ch.encode(args.charset)
            return True
        except UnicodeEncodeError:
            return False

    freq = corpus_freq(args.corpus)
    table = collections.defaultdict(list)  # syllable -> [chars]
    n_chars = 0
    for lo, hi in CJK_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            if not in_charset(ch):
                continue
            try:
                readings = pinyin(ch, style=Style.BOPOMOFO, heteronym=True,
                                  errors=lambda x: None)
            except Exception:
                continue
            if not readings or not readings[0]:
                continue
            seen = set()
            for syl in readings[0]:
                if (not syl or syl in seen
                        or any(c not in KEYMAP_CHARS for c in syl)):
                    continue
                seen.add(syl)
                table[syl].append(ch)
            if seen:
                n_chars += 1

    # frequency order within each syllable's list
    for syl in table:
        table[syl].sort(key=lambda c: (-freq.get(c, 0), c))

    with open(args.out, "w", encoding="utf-8") as f:
        for syl in sorted(table):
            f.write(syl + "\t" + "".join(table[syl]) + "\n")
    print(f"wrote {len(table)} syllables covering {n_chars} chars to {args.out}")


if __name__ == "__main__":
    main()
