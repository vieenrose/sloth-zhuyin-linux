#!/usr/bin/env python3
"""OUTSIDE-span prediction accuracy — a real metric (not eyeballed examples).

The GPT's second job is: convert the typed prefix (inside, constrained) then KEEP
PREDICTING the rest (outside, free). This measures that quantitatively.

Held-out sentences are split at a word boundary: the model is given only the
PREFIX syllables and must (a) convert them, then (b) continue. We score:
  * inside-exact   : did it convert the typed prefix correctly?
  * next-char@1    : is the first predicted (outside) char the true next char?
  * next-char@5    : is the true next char in the model's top-5 at that position?
  * continuation-2 : are the next TWO chars both right (a word-ish unit)?

  python3 outside_eval.py --model slothe_50m_qat --n 300
"""
import argparse, json, random, sys
import torch
import distill_student as D
from distill_student import TinyGPT, QAT

TONES = "ˊˇˋ˙"
BOPO = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + TONES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="gen_pairs_full.jsonl")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--n", type=int, default=300, help="held-out samples")
    ap.add_argument("--skip", type=int, default=1900000, help="skip to unseen tail of corpus")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{args.model}/student.pt", map_location=dev)
    c = ck["config"]
    vocab = json.load(open(f"{args.model}/student_vocab.json", encoding="utf-8"))
    QAT["on"] = c.get("qat", False)
    D.SSM_TYPE["t"] = c.get("ssm_type", "mamba1")
    m = TinyGPT(c["vocab"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"],
                c.get("pattern")).to(dev)
    sd = ck["model"]
    try:
        m.load_state_dict(sd)
    except Exception:
        import re
        m.load_state_dict({re.sub(r'\.attn\.(q|k|v|o|qn|kn)\.', r'.mix.\1.', k): v
                           for k, v in sd.items()})
    m.eval()
    BOS, SEP, UNK = vocab["<bos>"], vocab["<sep>"], vocab.get("<unk>", 1)
    id2 = {v: k for k, v in vocab.items()}

    tonal = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if r: tonal[s] = set(r)
    def legal_ids(syl):
        has = any(x in TONES for x in syl)
        ch = tonal.get(syl) if has else None
        if ch is None:
            base = "".join(x for x in syl if x not in TONES); ch = set()
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base: ch |= v
        return [vocab[x] for x in (ch or {syl}) if x in vocab]

    # held-out: take sentences from the far tail of the corpus (unseen: training
    # used --max-pairs 2M and the file is 2.12M)
    samples = []
    for i, line in enumerate(open(args.data, encoding="utf-8")):
        if i < args.skip: continue
        d = json.loads(line)
        sy = [s for s in d["in"].split() if s and s[0] in BOPO]
        gold = [x for x in d["out"] if "一" <= x <= "鿿"]
        if len(sy) == len(gold) and 4 <= len(sy) <= 12:
            samples.append((sy, gold))
        if len(samples) >= args.n: break

    n = ins_ok = n1 = n5 = cont2 = 0
    with torch.no_grad():
        for sy, gold in samples:
            k = max(1, len(sy) // 2)                    # typed prefix = half
            psy, pgold = sy[:k], gold[:k]
            inp = [BOS] + [vocab.get(s, UNK) for s in psy] + [SEP]
            gen = []
            # inside: constrained convert of the typed prefix
            for syl in psy:
                lg = m(torch.tensor([inp + gen], device=dev))[0, -1]
                lids = legal_ids(syl)
                if not lids: gen.append(UNK); continue
                mask = torch.full_like(lg, float("-inf"))
                lt = torch.tensor(lids, device=dev); mask[lt] = lg[lt]
                gen.append(int(mask.argmax()))
            n += 1
            ins_ok += ("".join(id2.get(g, "?") for g in gen) == "".join(pgold))
            # outside: free prediction of the next char(s)
            lg = m(torch.tensor([inp + gen], device=dev))[0, -1]
            top5 = torch.topk(lg, 5).indices.tolist()
            true_next = gold[k]
            n1 += (id2.get(top5[0]) == true_next)
            n5 += (true_next in [id2.get(t) for t in top5])
            if len(gold) > k + 1:
                nxt = int(lg.argmax()); gen2 = gen + [nxt]
                lg2 = m(torch.tensor([inp + gen2], device=dev))[0, -1]
                cont2 += (id2.get(nxt) == gold[k] and id2.get(int(lg2.argmax())) == gold[k + 1])

    print(f"{args.model}  (held-out n={n})")
    print(f"  inside-exact (typed prefix converted): {100*ins_ok/n:.1f}%")
    print(f"  OUTSIDE next-char@1 : {100*n1/n:.1f}%")
    print(f"  OUTSIDE next-char@5 : {100*n5/n:.1f}%")
    print(f"  OUTSIDE continuation-2 (both chars):  {100*cont2/n:.1f}%")


if __name__ == "__main__":
    main()
