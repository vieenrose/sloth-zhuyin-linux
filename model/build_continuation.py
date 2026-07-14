#!/usr/bin/env python3
"""Build OUTSIDE-SPAN continuation training pairs (design §5): input = syllables
of a PREFIX only; target = the FULL sentence. Teaches the GPT to convert what is
typed (inside span) then keep predicting the rest (outside span) — the prediction/
association job a bidirectional encoder structurally cannot do.

  python3 build_continuation.py --in gen_pairs_full.jsonl --out gen_pairs_cont.jsonl
Then mix with conversion pairs for a GPT that does BOTH inside conversion and
outside prediction. Empirically, gemma trained on exact-length pairs ONLY cannot
continue past the span (repeats a char / stops) — these pairs fix that.
"""
import argparse, json, random
TONES = "ˊˇˋ˙"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="gen_pairs_full.jsonl")
    ap.add_argument("--out", default="gen_pairs_cont.jsonl")
    ap.add_argument("--max", type=int, default=800000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.inp, encoding="utf-8"):
            d = json.loads(line)
            sy = d["in"].split()
            zh = [c for c in d["out"] if "一" <= c <= "鿿"]
            if len(sy) != len(zh) or len(sy) < 3:
                continue
            k = random.randint(1, len(sy) - 1)            # typed prefix length
            out.write(json.dumps({"in": " ".join(sy[:k]), "out": d["out"]},
                                 ensure_ascii=False) + "\n")
            n += 1
            if n >= args.max: break
    print(f"wrote {n} continuation pairs to {args.out}")

if __name__ == "__main__":
    main()
