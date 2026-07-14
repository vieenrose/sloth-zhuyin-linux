#!/usr/bin/env python3
"""Eval the distilled 28M custom-decoder student. Its vocab is char-level (1 token
= 1 char), so constrained decode is CLEAN — no BPE-merge issue. At each position,
mask to the syllable's tone-union legal char ids, argmax, advance. Toned + toneless.

  python3 student_eval.py --model slothe_student30m [--toneless]
"""
import argparse, json, torch
from distill_student import TinyGPT

TONES = "ˊˇˋ˙"
VAR = str.maketrans({"臺": "台", "箇": "個"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="slothe_student30m")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--toneless", action="store_true")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"{args.model}/student.pt", map_location=dev)
    c = ckpt["config"]
    vocab = json.load(open(f"{args.model}/student_vocab.json", encoding="utf-8"))
    m = TinyGPT(c["vocab"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"]).to(dev)
    m.load_state_dict(ckpt["model"]); m.eval()
    BOS, SEP, UNK = vocab["<bos>"], vocab["<sep>"], vocab.get("<unk>", 1)

    tonal = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if r: tonal[s] = set(r)
    def legal_ids(syl):
        has = any(ch in TONES for ch in syl)
        chars = tonal.get(syl) if has else None
        if chars is None:
            base = "".join(x for x in syl if x not in TONES); chars = set()
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base: chars |= v
        return [vocab[x] for x in (chars or {syl}) if x in vocab]

    id2tok = {v: k for k, v in vocab.items()}

    @torch.no_grad()
    def decode(syls):
        inp = [BOS] + [vocab.get(s, UNK) for s in syls] + [SEP]
        gen = []
        for i, syl in enumerate(syls):
            ids = torch.tensor([inp + gen], device=dev)
            logits = m(ids)[0, -1]
            lids = legal_ids(syl)
            if not lids: gen.append(UNK); continue
            masked = torch.full_like(logits, float("-inf"))
            lt = torch.tensor(lids, device=dev)
            masked[lt] = logits[lt]
            gen.append(int(masked.argmax()))
        return "".join(id2tok.get(g, "?") for g in gen)

    n = ok = okv = 0; fails = []
    for line in open(args.testset, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        p = line.split("\t"); syls = p[0].split(); exp = p[1]
        if args.toneless:
            syls = ["".join(ch for ch in y if ch not in TONES) for y in syls]
        got = decode(syls)
        n += 1; ok += (got == exp); okv += (got.translate(VAR) == exp.translate(VAR))
        if got.translate(VAR) != exp.translate(VAR) and len(fails) < args.show:
            fails.append("    " + exp + " | " + got)
    tag = "toneless" if args.toneless else "toned   "
    print(f"student28M {tag}: raw {ok}/{n} ({100*ok/n:.1f}%)  variant-norm {okv}/{n} ({100*okv/n:.1f}%)")
    import sys
    for f in fails: print(f, file=sys.stderr)


if __name__ == "__main__":
    main()
