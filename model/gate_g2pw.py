#!/usr/bin/env python3
"""Self-contained SlothLM-E accuracy gate (no libchewing needed).

Decodes eval/testset.tsv (homophone-hard) and eval/reference_mspy.tsv (免選字)
with per-position argmax over each syllable's legal chars, reports sentence
accuracy. Compare to the E3 baseline (84% / 73%).

  python3 model/gate_g2pw.py --model model/slothe_10m_g2pw
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import SlothE

TONES = "ˊˇˋ˙"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu")
    c = ck["config"]
    m = SlothE(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"],
               embed_norm=c.get("embed_norm", False),
               char_hints=c.get("char_hints", False),
               tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"]); m.eval()
    sv = json.load(open(os.path.join(args.model, "syl_vocab.json"), encoding="utf-8"))
    tonal = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if r: tonal[s] = list(r)

    def cid(sy):
        ht = any(x in TONES for x in sy); ch = tonal.get(sy) if ht else None
        if ch is None:
            b = "".join(x for x in sy if x not in TONES); ch = []
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == b:
                    for x in v:
                        if x not in ch: ch.append(x)
        return [tok.convert_tokens_to_ids(x) for x in (ch or [sy]) if tok.convert_tokens_to_ids(x) is not None]

    def dec(sy, toneless=False):
        if toneless: sy = ["".join(x for x in y if x not in TONES) for y in sy]
        sid = torch.tensor([[sv.get(y, 1) for y in sy]]); am = torch.ones_like(sid, dtype=torch.bool)
        with torch.no_grad(): lg = m(sid, am)[0].numpy()
        out = []
        for i, s in enumerate(sy):
            ids = cid(s)
            out.append(tok.convert_ids_to_tokens(max(ids, key=lambda t: lg[i, t])) if ids else "?")
        return "".join(out)

    def run(path, name, toneless=False):
        n = ok = 0
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"): continue
            bopo, exp = line.split("\t")[:2]
            n += 1; ok += dec(bopo.split(), toneless) == exp
        print(f"  {name}: {ok}/{n} ({100*ok/n:.0f}%)")

    print(f"=== {os.path.basename(args.model)} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M) ===")
    run("eval/testset.tsv", "homophone-hard (E3: 84%)")
    run("eval/reference_mspy.tsv", "免選字 230  (chewing floor not yet re-measured on this set)")
    run("eval/testset.tsv", "toneless    (E3: 82%)", toneless=True)


if __name__ == "__main__":
    main()
