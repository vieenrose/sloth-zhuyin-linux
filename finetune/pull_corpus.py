#!/usr/bin/env python3
"""Stream a native Traditional-Chinese dataset and emit short, clean,
all-Han sentences suitable for training-data generation.

Defaults to yentinglin/zh_TW_c4 (Taiwan-LLM project; native Traditional
Chinese web text). Streamed, so nothing is fully downloaded. Sentences are
split on 。！？!?\\n, kept only if 2-18 chars and *entirely* CJK ideographs
(so bopomofo conversion downstream is clean), and deduped.

Usage:
  python3 finetune/pull_corpus.py --n 8000 > corpus.txt
"""
import argparse
import re
import sys

from datasets import load_dataset

SPLIT = re.compile(r"[。！？!?\n]+")
# CJK Unified Ideographs (+ Ext A); excludes punctuation, latin, digits, kana.
HAN = re.compile(r"^[一-鿿㐀-䶿]+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--dataset", default="yentinglin/zh_TW_c4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--max", type=int, default=18)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    seen = set()
    emitted = 0
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        for seg in SPLIT.split(text):
            seg = seg.strip()
            if (args.min <= len(seg) <= args.max and HAN.match(seg)
                    and seg not in seen):
                seen.add(seg)
                print(seg)
                emitted += 1
                if emitted >= args.n:
                    print(f"emitted {emitted} sentences", file=sys.stderr)
                    return
    print(f"emitted {emitted} (dataset exhausted)", file=sys.stderr)


if __name__ == "__main__":
    main()
