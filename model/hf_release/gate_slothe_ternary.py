#!/usr/bin/env python3
"""
Accuracy gate for the ternary SlothLM-E — mirrors reproduce/gate_g2pw.py but
rebuilds the model from the ternary `config` and evaluates at full ternary
(quant_alpha=1). Reports the same three numbers so you can compare directly to
the 12M int8 baseline (免選字 84% / homophone ~84% / toneless 70%).

    python3 gate_slothe_ternary.py --model slothe_t_20m \
        --tokenizer tokenizer --table phonetic_table.tsv
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothe_ternary import SlothE_T

TONES = "ˊˇˋ˙"


def build(ck):
    c = ck["config"]
    m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"],
                 c["ffn"], embed_norm=c.get("embed_norm", False),
                 weight_bits=c.get("weight_bits", "ternary"),
                 weight_quant=c.get("weight_quant", "median"),
                 fp_boundary=c.get("fp_boundary", 1),
                 pre_norm=c.get("pre_norm", True), act_bits=c.get("act_bits", 8), char_hints=c.get("char_hints", False), tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"])
    m.set_quant_alpha(1.0)   # evaluate the deployed (fully ternary) model
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="eval/testset.tsv")
    ap.add_argument("--mspy", default="eval/reference_mspy.tsv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu")
    m = build(ck)
    sv = json.load(open(os.path.join(args.model, "syl_vocab.json"), encoding="utf-8"))

    tonal = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if r:
            tonal[s] = list(r)

    def cid(sy):
        ht = any(x in TONES for x in sy)
        ch = tonal.get(sy) if ht else None
        if ch is None:
            b = "".join(x for x in sy if x not in TONES); ch = []
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == b:
                    for x in v:
                        if x not in ch:
                            ch.append(x)
        return [tok.convert_tokens_to_ids(x) for x in (ch or [sy])
                if tok.convert_tokens_to_ids(x) is not None]

    def dec(sy, toneless=False):
        if toneless:
            sy = ["".join(x for x in y if x not in TONES) for y in sy]
        sid = torch.tensor([[sv.get(y, 1) for y in sy]])
        am = torch.ones_like(sid, dtype=torch.bool)
        with torch.no_grad():
            lg = m(sid, am)[0].numpy()
        out = []
        for i, s in enumerate(sy):
            ids = cid(s)
            out.append(tok.convert_ids_to_tokens(max(ids, key=lambda t: lg[i, t]))
                       if ids else "?")
        return "".join(out)

    def run(path, name, toneless=False):
        n = ok = 0
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            bopo, exp = line.split("\t")[:2]
            n += 1; ok += dec(bopo.split(), toneless) == exp
        print(f"  {name}: {ok}/{n} ({100*ok/n:.0f}%)")

    npar = sum(p.numel() for p in m.parameters())
    print(f"=== {os.path.basename(args.model)} ({npar/1e6:.0f}M, ternary) ===")
    run(args.testset, "homophone-hard (12M int8: 84%)")
    run(args.mspy, "免選字 230      (12M int8: 84%)")


if __name__ == "__main__":
    main()
