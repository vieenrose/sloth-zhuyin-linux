#!/usr/bin/env python3
"""Build a much larger Traditional-Chinese SENTENCE corpus for IME training.

We only sampled 1M short sentences from c4-chinese-zhtw before; C4 is far bigger.
Stream it, split documents into sentences on CJK punctuation, keep short
conversational-length lines (IME-relevant), dedup, target N unique sentences.
Output: one sentence per line (Traditional Chinese, mostly Han + light ASCII).

  python3 model/build_corpus_big.py --out model/corpus_huge.txt --target 6000000
"""
import argparse
import re
import sys

from datasets import load_dataset

SPLIT = re.compile(r"[。！？!?；;\n\r]+")
# a line is IME-relevant if it's mostly Han, 4..28 chars, no URLs/markup noise
HAN = re.compile(r"[一-鿿]")
BAD = re.compile(r"[<>{}\[\]|\\@#\^`~=_*]|https?://|www\.")
KEEP = re.compile(r"^[一-鿿，、A-Za-z0-9 ，、]+$")


LEAD = re.compile(r"^[^一-鿿A-Za-z]+")   # strip leading digits/punctuation


def sentences(text):
    for part in SPLIT.split(text):
        s = LEAD.sub("", part.strip()).strip("，、 ")
        if not (4 <= len(s) <= 28):
            continue
        if s[0] in "，、":
            continue
        if BAD.search(s):
            continue
        han = len(HAN.findall(s))
        if han < max(3, int(0.6 * len(s))):     # >=60% Han
            continue
        if not KEEP.match(s):
            continue
        yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="erhwenkuo/c4-chinese-zhtw")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="model/corpus_huge.txt")
    ap.add_argument("--target", type=int, default=6_000_000)
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    seen = set()
    docs = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            docs += 1
            for s in sentences(row.get(args.text_field, "")):
                if s in seen:
                    continue
                seen.add(s)
                f.write(s + "\n")
                if len(seen) >= args.target:
                    print(f"reached {len(seen)} unique sentences from {docs} docs")
                    return
            if docs % 100000 == 0:
                print(f"  {docs} docs -> {len(seen)} sentences", file=sys.stderr)
    print(f"exhausted stream: {len(seen)} unique sentences from {docs} docs")


if __name__ == "__main__":
    main()
