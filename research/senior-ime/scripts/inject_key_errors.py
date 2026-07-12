#!/usr/bin/env python3
"""Inject 8-adjacency key slips into a held-out eval set to simulate shaky-hand
typing: with prob p per syllable, one symbol slips to a physically-adjacent
key's symbol. The GOLD sentence is unchanged (what the user MEANT) — so the
error-tolerant decoder must recover the intent from the slipped input.
  python3 inject_key_errors.py --in ../eval/reference_heldout.tsv --p 0.1 --seed 0
"""
import argparse, random
TONES = set("ˊˇˋ˙")
rows = ["1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"]
pos = {k: (r, c) for r, row in enumerate(rows) for c, k in enumerate(row)}
key_nb = {k: [k2 for k2, (r2, c2) in pos.items() if abs(r2-p[0]) <= 1 and abs(c2-p[1]) <= 1 and k2 != k]
          for k, p in pos.items()}
DACHEN = {'1':'ㄅ','q':'ㄆ','a':'ㄇ','z':'ㄈ','2':'ㄉ','w':'ㄊ','s':'ㄋ','x':'ㄌ','e':'ㄍ','d':'ㄎ',
          'c':'ㄏ','r':'ㄐ','f':'ㄑ','v':'ㄒ','5':'ㄓ','t':'ㄔ','g':'ㄕ','b':'ㄖ','y':'ㄗ','h':'ㄘ',
          'n':'ㄙ','u':'ㄧ','j':'ㄨ','m':'ㄩ','8':'ㄚ','i':'ㄛ','k':'ㄜ',',':'ㄝ','9':'ㄞ','o':'ㄟ',
          'l':'ㄠ','.':'ㄡ','0':'ㄢ','p':'ㄣ',';':'ㄤ','/':'ㄥ','-':'ㄦ'}
sym_key = {v: k for k, v in DACHEN.items()}
sym_nb = {s: [DACHEN[k2] for k2 in key_nb.get(k, []) if k2 in DACHEN] for s, k in sym_key.items()}

def slip(syl, rng):
    chars = list(syl); idx = [i for i, c in enumerate(chars) if c in sym_nb and sym_nb[c]]
    if not idx: return syl
    i = rng.choice(idx); chars[i] = rng.choice(sym_nb[chars[i]])
    return "".join(chars)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", default="")
    ap.add_argument("--p", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    out = a.out or a.inp.replace(".tsv", f"_e{int(a.p*100)}.tsv")
    n = slips = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(a.inp, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                f.write(line + "\n"); continue
            bopo, sent = line.split("\t", 1); syls = bopo.split()
            new = []
            for s in syls:
                n += 1
                if rng.random() < a.p:
                    s2 = slip(s, rng)
                    if s2 != s: slips += 1
                    s = s2
                new.append(s)
            f.write(" ".join(new) + "\t" + sent + "\n")
    print(f"{out}: injected {slips}/{n} syllable slips ({100*slips/n:.1f}%) at p={a.p}")

if __name__ == "__main__": main()
