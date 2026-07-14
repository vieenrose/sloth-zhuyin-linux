#!/usr/bin/env python3
"""HYBRID: GPT proposes constrained-beam n-best (recall@8 ~97%), the warm-start
ENCODER reranks (it's the stronger toneless whole-sentence scorer). Tests whether
the hybrid gets the GPT's toned strength AND pushes toneless toward the recall
ceiling — resolving the toneless gap without touching either model.

  python3 gpt_ime_rerank.py --gpt slothe_gpt_gemma --enc slothe_roberta_warm [--toneless]
"""
import argparse, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

TONES = "ˊˇˋ˙"
VAR = str.maketrans({"臺": "台", "箇": "個"})


def tone_union(table_path):
    tonal = {}
    for line in open(table_path, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if rest: tonal[s] = set(rest)
    def legal(syl):
        has = any(c in TONES for c in syl)
        chars = tonal.get(syl) if has else None
        if chars is None:
            base = "".join(x for x in syl if x not in TONES)
            chars = set()
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base: chars |= v
        return chars or {syl}
    return legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--vocab", default="syl_vocab.json")
    ap.add_argument("--toneless", action="store_true")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--scan", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on encoder score (gpt=1)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    legal = tone_union(args.table)

    # --- GPT (proposer) ---
    gtok = AutoTokenizer.from_pretrained(args.gpt)
    gpt = AutoModelForCausalLM.from_pretrained(args.gpt, dtype=torch.bfloat16).to(dev).eval()
    gdc = {}
    def dec1(t):
        if t not in gdc: gdc[t] = gtok.decode([t])
        return gdc[t]

    # --- Encoder (reranker) ---
    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    adapt = json.load(open(os.path.join(args.enc, "adapt.json")))
    btok = AutoTokenizer.from_pretrained(adapt["base"])
    bvocab = btok.get_vocab()
    enc = AutoModelForMaskedLM.from_pretrained(args.enc).to(dev).eval()
    hid = enc.config.hidden_size
    syl_embed = nn.Embedding(adapt["n_syl"], hid)
    syl_embed.load_state_dict(torch.load(os.path.join(args.enc, "syl_embed.pt"), map_location="cpu"))
    syl_embed.to(dev).eval()

    @torch.no_grad()
    def gpt_nbest(syls):
        N = len(syls); ls = [legal(s) for s in syls]
        prompt = gtok(" ".join(syls) + " →", add_special_tokens=False,
                      return_tensors="pt").input_ids.to(dev)
        def la(gids):
            ids = torch.cat([prompt, torch.tensor([gids], device=dev, dtype=prompt.dtype)], 1) if gids else prompt
            return F.log_softmax(gpt(ids).logits[0, -1], -1)
        beams = [(0.0, [], "", 0)]
        for _ in range(N + 2):
            if all(p >= N for _, _, _, p in beams): break
            cand = []
            for score, gids, dec, p in beams:
                if p >= N: cand.append((score, gids, dec, p)); continue
                lp = la(gids)
                for t in torch.argsort(lp, descending=True)[: args.scan].tolist():
                    txt = dec1(t)
                    if not txt: continue
                    nd = dec + txt; ch = list(nd)
                    if len(ch) <= p: continue
                    if all(ch[i] in ls[i] for i in range(min(len(ch), N))):
                        cand.append((score + lp[t].item(), gids + [t], nd, min(len(ch), N)))
            cand.sort(key=lambda x: x[0], reverse=True); beams = cand[: args.beam]
        seen = {}
        for s, g, dec, p in beams:
            if p >= N:
                seen[dec[:N]] = max(seen.get(dec[:N], -1e9), s)
        return sorted(seen.items(), key=lambda x: -x[1])   # [(txt, gpt_score)]

    @torch.no_grad()
    def enc_score(syls, cand):
        sids = torch.tensor([[syl_vocab.get(y, 1) for y in syls]], device=dev)
        am = torch.ones_like(sids)
        logits = enc(inputs_embeds=syl_embed(sids), attention_mask=am).logits[0]  # [T, bvocab]
        lp = F.log_softmax(logits, -1)
        s = 0.0
        for i, ch in enumerate(cand):
            bid = bvocab.get(ch)
            if bid is None: return -1e9
            s += lp[i, bid].item()
        return s

    n = gpt1 = enc1 = hyb1 = 0
    for line in open(args.testset, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        p = line.split("\t"); syls = p[0].split(); exp = p[1]
        if args.toneless:
            syls = ["".join(c for c in y if c not in TONES) for y in syls]
        nb = gpt_nbest(syls)
        if not nb: n += 1; continue
        gpt_top = nb[0][0]
        # hybrid: rerank the GPT n-best by gpt_score + alpha*enc_score
        resc = [(txt, gs + args.alpha * enc_score(syls, list(txt))) for txt, gs in nb]
        hyb_top = max(resc, key=lambda x: x[1])[0]
        # encoder-alone top (rescore all beam candidates by encoder only)
        enc_top = max(nb, key=lambda x: enc_score(syls, list(x[0])))[0]
        n += 1
        eq = lambda a: a.translate(VAR) == exp.translate(VAR)
        gpt1 += eq(gpt_top); enc1 += eq(enc_top); hyb1 += eq(hyb_top)
    tag = "toneless" if args.toneless else "toned   "
    print(f"{tag}: GPT-alone {gpt1}/{n} ({100*gpt1/n:.1f}%)  "
          f"ENC-rerank {enc1}/{n} ({100*enc1/n:.1f}%)  "
          f"HYBRID(gpt+{args.alpha}*enc) {hyb1}/{n} ({100*hyb1/n:.1f}%)")


if __name__ == "__main__":
    main()
