#!/usr/bin/env python3
"""Extract common Traditional-Chinese words from the corpus and emit them as
upweighted training lines, so SlothLM-E learns strong defaults for common
words (the web-crawl corpus under-represents everyday words -> misses like
重新→崇心).

Segments the corpus with jieba, counts 2–4 char words, keeps frequent ones,
and writes each word repeated ~by its log-frequency (capped). Appended to the
training corpus, this gives the aligned (syllable, char) model many clean
in-isolation examples of common words.

  python3 model/common_words.py --corpus model/corpus_big.txt \
      --out model/common_words.txt --min-count 40 --max-repeat 30
"""
import argparse
import collections
import math

import jieba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=40)
    ap.add_argument("--max-repeat", type=int, default=30)
    ap.add_argument("--max-lines", type=int, default=400000,
                    help="segment at most this many corpus lines (speed)")
    args = ap.parse_args()

    freq = collections.Counter()
    n = 0
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s:
            continue
        for w in jieba.cut(s):
            if 2 <= len(w) <= 4 and all("一" <= c <= "鿿" for c in w):
                freq[w] += 1
        n += 1
        if n >= args.max_lines:
            break

    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for w, c in freq.most_common():
            if c < args.min_count:
                break
            # repeat ~ log-frequency so very common words get more examples,
            # capped so nothing dominates
            r = min(args.max_repeat, 1 + int(math.log2(c)))
            for _ in range(r):
                f.write(w + "\n")
            kept += 1
    print(f"segmented {n} lines; wrote {kept} common words (>= {args.min_count}) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
