#!/usr/bin/env python3
"""Build the 聯想 (next-word association) table from libchewing's tsi.src.

Classic dictionary 聯想 (微軟新注音-style): after the user commits text ending
in character X, suggest the completions of the highest-frequency words that
START with X — e.g. after 電, suggest 腦/影/話 (from 電腦/電影/電話).

Input:  tsi.src lines `word freq ㄅㄆ...` (LGPL-2.1, from the libchewing
        project — fetch: https://raw.githubusercontent.com/chewing/libchewing/v0.9.1/data/tsi.src)
Output: assoc_tc.tsv lines `firstChar<TAB>comp1 comp2 ...` (freq-ranked
        completions, deduped, top N per head char).

Usage: python3 model/build_assoc.py /tmp/tsi.src model/assoc_tc.tsv
"""
import sys
from collections import defaultdict

TOP_N = 8          # completions kept per head char
MAX_COMP = 3       # completion length cap (chars) — keeps the strip tidy

def main(src: str, dst: str) -> None:
    heads: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for line in open(src, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 2:
            continue
        word, freq = parts[0], parts[1]
        try:
            f = int(freq)
        except ValueError:
            continue
        chars = list(word)
        if len(chars) < 2 or len(chars) > 1 + MAX_COMP:
            continue
        if not all("一" <= c <= "鿿" for c in chars):
            continue
        heads[chars[0]].append((f, "".join(chars[1:])))

    kept = 0
    with open(dst, "w", encoding="utf-8") as out:
        for head in sorted(heads):
            comps: list[str] = []
            for _, comp in sorted(heads[head], key=lambda t: -t[0]):
                if comp not in comps:
                    comps.append(comp)
                if len(comps) >= TOP_N:
                    break
            out.write(head + "\t" + " ".join(comps) + "\n")
            kept += 1
    print(f"{kept} head chars -> {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
