#!/usr/bin/env python3
"""Rebuild phonetic_table.tsv with TAIWAN-standard readings from libchewing.

The original table came from pypinyin, which uses mainland readings -- so
Taiwan-standard readings were unreachable (微 filed under ㄨㄟ instead of ㄨㄟˊ;
also 期ㄑㄧˊ, 息ㄒㄧˊ, 檔ㄉㄤˇ, ~10/18 sampled). This queries libchewing itself
(eval/harvest candidate enumeration) for every syllable x tone: the candidate
list IS the authoritative Taiwan mapping.

Keeps pypinyin entries as a fallback union (rare chars chewing may not list),
so coverage never shrinks -- readings are only ever ADDED.

  python3 model/build_phonetic_table_tw.py --old model/phonetic_table.tsv \
      --harvest eval/harvest --out model/phonetic_table.tsv
"""
import argparse
import json
import subprocess
import sys

TONES = ["", "ˊ", "ˇ", "ˋ", "˙"]
KEYMAP = {"ㄅ":"1","ㄆ":"q","ㄇ":"a","ㄈ":"z","ㄉ":"2","ㄊ":"w","ㄋ":"s","ㄌ":"x",
 "ㄍ":"e","ㄎ":"d","ㄏ":"c","ㄐ":"r","ㄑ":"f","ㄒ":"v","ㄓ":"5","ㄔ":"t","ㄕ":"g",
 "ㄖ":"b","ㄗ":"y","ㄘ":"h","ㄙ":"n","ㄧ":"u","ㄨ":"j","ㄩ":"m","ㄚ":"8","ㄛ":"i",
 "ㄜ":"k","ㄝ":",","ㄞ":"9","ㄟ":"o","ㄠ":"l","ㄡ":".","ㄢ":"0","ㄣ":"p","ㄤ":";",
 "ㄥ":"/","ㄦ":"-","ˊ":"6","ˇ":"3","ˋ":"4","˙":"7"}


def keys_of(syl):
    k = [KEYMAP[c] for c in syl]
    if syl[-1] not in "ˊˇˋ˙":
        k.append(" ")          # tone 1 commits with space
    return "".join(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="model/phonetic_table.tsv")
    ap.add_argument("--harvest", default="eval/harvest")
    ap.add_argument("--out", default="model/phonetic_table.tsv")
    args = ap.parse_args()

    # bases = every toneless syllable skeleton in the old table
    old = {}
    bases = set()
    for line in open(args.old, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if not rest: continue
        old[s] = list(dict.fromkeys(rest))
        bases.add("".join(c for c in s if c not in "ˊˇˋ˙"))

    table = {}
    probed = found = 0
    for base in sorted(bases):
        for tone in TONES:
            syl = base + tone
            probed += 1
            try:
                r = subprocess.run([args.harvest, keys_of(syl)],
                                   capture_output=True, text=True, timeout=10)
                data = json.loads(r.stdout)
            except Exception:
                continue
            chars = []
            for pos in data.get("positions", []):
                for c in pos:
                    if len(c) == 1 and c not in chars:
                        chars.append(c)
            if chars:
                table[syl] = chars
                found += 1
        if probed % 500 < len(TONES):
            print(f"  {probed} probed, {found} syllables found", file=sys.stderr)

    # union with the old table so coverage never shrinks
    added_syls = 0
    for s, chars in old.items():
        if s not in table:
            table[s] = chars; added_syls += 1
        else:
            for c in chars:
                if c not in table[s]:
                    table[s].append(c)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in sorted(table):
            f.write(s + "\t" + "".join(table[s]) + "\n")
    n_chars = len({c for v in table.values() for c in v})
    print(f"wrote {len(table)} syllables ({found} from libchewing, "
          f"{added_syls} pypinyin-only kept), {n_chars} distinct chars -> {args.out}")


if __name__ == "__main__":
    main()
