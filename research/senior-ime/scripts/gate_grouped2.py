#!/usr/bin/env python3
"""Tone-kept grouped gate. Model input = (group-keys + tone) signature. Two mask
modes: hard-tone (ceiling, correct tone required) and --soft-tone (tolerant:
legal chars = union over all tones, so a tone slip never excludes the answer)."""
import argparse, json, os, sys
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothe_ternary import SlothE_T
TONES = "ˊˇˋ˙"
strip = lambda s: "".join(x for x in s if x not in TONES)
tone_of = lambda s: "".join(x for x in s if x in TONES) or "1"

def build(ck):
    c = ck["config"]
    m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"],
                 embed_norm=c.get("embed_norm", False), weight_bits=c.get("weight_bits", "ternary"),
                 weight_quant=c.get("weight_quant", "median"), fp_boundary=c.get("fp_boundary", 1),
                 pre_norm=c.get("pre_norm", True), act_bits=c.get("act_bits", 8),
                 char_hints=c.get("char_hints", False), tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"]); m.set_quant_alpha(1.0); m.eval(); return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cfg", required=True)
    ap.add_argument("--tokenizer", default="tokenizer"); ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--mspy", default="../eval/reference_heldout.tsv"); ap.add_argument("--soft-tone", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.cfg)); groups = cfg["groups"]; tone_kept = cfg["tone_kept"]
    gm = {c: i for i, g in enumerate(groups) for c in g}
    def sig(s):
        try: g = tuple(gm[c] for c in strip(s))
        except KeyError: return None
        return "S_" + "_".join(map(str, g)) + ("|" + tone_of(s) if tone_kept else "")
    def notone(s):
        try: g = tuple(gm[c] for c in strip(s))
        except KeyError: return None
        return "N_" + "_".join(map(str, g))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu"); m = build(ck)
    gv = json.load(open(os.path.join(args.model, "syl_vocab.json"), encoding="utf-8"))
    hard = defaultdict(set); soft = defaultdict(set)
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if not r: continue
        k, kn = sig(s), notone(s)
        for x in r:
            t = tok.convert_tokens_to_ids(x)
            if t is None: continue
            if k: hard[k].add(t)
            if kn: soft[kn].add(t)
    def dec(syls):
        cids = [gv.get(sig(y), 1) for y in syls]
        sid = torch.tensor([cids]); am = torch.ones_like(sid, dtype=torch.bool)
        with torch.no_grad(): lg = m(sid, am)[0].numpy()
        out = []
        for i, y in enumerate(syls):
            legal = (soft.get(notone(y)) if args.soft_tone else hard.get(sig(y)))
            out.append(tok.convert_ids_to_tokens(max(legal, key=lambda t: lg[i, t])) if legal else "?")
        return out
    ns = oks = nc = okc = 0
    for line in open(args.mspy, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        bopo, exp = line.split("\t")[:2]
        pred = dec(bopo.split()); gold = list(exp)
        if len(pred) == len(gold): okc += sum(p == g for p, g in zip(pred, gold)); nc += len(gold)
        ns += 1; oks += ("".join(pred) == exp)
    print(f"=== {os.path.basename(args.model)} {'SOFT' if args.soft_tone else 'HARD'}-tone ===")
    print(f"  免選字 whole-sentence: {oks}/{ns} ({100*oks/ns:.1f}%)")
    print(f"  char accuracy:         {okc}/{nc} ({100*okc/nc:.1f}%)")
if __name__ == "__main__": main()
