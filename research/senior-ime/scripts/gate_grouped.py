#!/usr/bin/env python3
"""Grouped-keypad accuracy gate. The model sees Candidate-A group-classes
(tone dropped, 52 classes) and must recover characters via context + the
EXPANDED legality mask (union of legal chars over every syllable in the class).
Reports whole-sentence 免選字 and per-char accuracy on a held-out set."""
import argparse, json, os, sys
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothe_ternary import SlothE_T

TONES = "ˊˇˋ˙"
strip = lambda s: "".join(x for x in s if x not in TONES)
A = ["ㄅㄆㄇㄈ","ㄉㄊㄋㄌ","ㄍㄎㄏ","ㄐㄑㄒ","ㄓㄔㄕㄖ","ㄗㄘㄙ",
     "ㄧㄨㄩ","ㄚㄛㄜㄝ","ㄞㄟㄠㄡ","ㄢㄣㄤㄥㄦ"]
sym2grp = {c: i for i, g in enumerate(A) for c in g}


def build(ck):
    c = ck["config"]
    m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"],
                 c["ffn"], embed_norm=c.get("embed_norm", False),
                 weight_bits=c.get("weight_bits", "ternary"),
                 weight_quant=c.get("weight_quant", "median"),
                 fp_boundary=c.get("fp_boundary", 1), pre_norm=c.get("pre_norm", True),
                 act_bits=c.get("act_bits", 8), char_hints=c.get("char_hints", False),
                 tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"]); m.set_quant_alpha(1.0); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--mspy", default="../eval/reference_heldout.tsv")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu")
    m = build(ck)
    gv = json.load(open(os.path.join(args.model, "syl_vocab.json"), encoding="utf-8"))

    def class_id(syl):
        try: k = tuple(sym2grp[c] for c in strip(syl))
        except KeyError: return 1
        return gv.get("G_" + "_".join(map(str, k)), 1)

    class2chars = defaultdict(set)
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if not r: continue
        cls = class_id(s)
        for x in r:
            t = tok.convert_tokens_to_ids(x)
            if t is not None: class2chars[cls].add(t)
    class2chars = {k: list(v) for k, v in class2chars.items()}

    def dec(syls):
        cids = [class_id(y) for y in syls]
        sid = torch.tensor([cids]); am = torch.ones_like(sid, dtype=torch.bool)
        with torch.no_grad():
            lg = m(sid, am)[0].numpy()
        out = []
        for i, c in enumerate(cids):
            legal = class2chars.get(c)
            out.append(tok.convert_ids_to_tokens(max(legal, key=lambda t: lg[i, t]))
                       if legal else "?")
        return out

    nsent = oksent = nch = okch = 0
    for line in open(args.mspy, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        bopo, exp = line.split("\t")[:2]
        pred = dec(bopo.split()); gold = list(exp)
        if len(pred) == len(gold):
            okch += sum(p == g for p, g in zip(pred, gold)); nch += len(gold)
        nsent += 1; oksent += ("".join(pred) == exp)
    print(f"=== {os.path.basename(args.model)} GROUPED-A (tone-dropped 3x4) ===")
    print(f"  免選字 whole-sentence: {oksent}/{nsent} ({100*oksent/nsent:.1f}%)")
    print(f"  char accuracy:         {okch}/{nch} ({100*okch/nch:.1f}%)")


if __name__ == "__main__":
    main()
