#!/usr/bin/env python3
"""Investigate the toneless gap: constrained BEAM eval (vs the greedy cdecode).
Reports top-1 accuracy AND recall@B (is the gold answer anywhere in the beam?).

If beam top-1 >> greedy top-1  -> whole-sentence log-prob ranking closes the gap
   (greedy's left-only myopia was the problem, not model capacity).
If recall@B is high but top-1 low -> the answer is reachable; a better reranker
   (or bidirectional rescoring) recovers it — NOT an inherent limitation.
If recall@B is also low -> the model genuinely can't produce the gold under the
   phonetic constraint (capacity/training limit).

  python3 gpt_ime_beameval.py --model slothe_gpt_gemma --beam 8 [--toneless]
"""
import argparse, sys
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    ap.add_argument("--model", required=True)
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--toneless", action="store_true")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--scan", type=int, default=60)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    legal = tone_union(args.table)
    dcache = {}
    def dec1(t):
        if t not in dcache: dcache[t] = tok.decode([t])
        return dcache[t]

    @torch.no_grad()
    def beam_nbest(syls):
        N = len(syls); ls = [legal(s) for s in syls]
        prompt = tok(" ".join(syls) + " →", add_special_tokens=False,
                     return_tensors="pt").input_ids.to(dev)
        def logits_after(gids):
            ids = torch.cat([prompt, torch.tensor([gids], device=dev, dtype=prompt.dtype)], 1) if gids else prompt
            return F.log_softmax(model(ids).logits[0, -1], -1)
        beams = [(0.0, [], "", 0)]
        for _ in range(N + 2):
            if all(p >= N for _, _, _, p in beams): break
            cand = []
            for score, gids, dec, p in beams:
                if p >= N: cand.append((score, gids, dec, p)); continue
                lp = logits_after(gids)
                for t in torch.argsort(lp, descending=True)[: args.scan].tolist():
                    txt = dec1(t)
                    if not txt: continue
                    nd = dec + txt; ch = list(nd)
                    if len(ch) <= p: continue
                    if all(ch[i] in ls[i] for i in range(min(len(ch), N))):
                        cand.append((score + lp[t].item(), gids + [t], nd, min(len(ch), N)))
            cand.sort(key=lambda x: x[0], reverse=True)
            beams = cand[: args.beam]
        seen = {}
        for s, g, dec, p in beams:
            if p >= N:
                txt = dec[:N]
                if txt not in seen or s > seen[txt]: seen[txt] = s
        return [t for t, _ in sorted(seen.items(), key=lambda x: -x[1])]

    n = top1 = topv = rec = 0; fails = []
    for line in open(args.testset, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        p = line.split("\t"); syls = p[0].split(); exp = p[1]
        if args.toneless:
            syls = ["".join(c for c in y if c not in TONES) for y in syls]
        nb = beam_nbest(syls)
        n += 1
        top1 += (nb and nb[0] == exp)
        topv += (nb and nb[0].translate(VAR) == exp.translate(VAR))
        inbeam = any(t.translate(VAR) == exp.translate(VAR) for t in nb)
        rec += inbeam
        if not inbeam and len(fails) < args.show:
            fails.append("    " + exp + " | top:" + (nb[0] if nb else "?") + "  beam:" + " ".join(nb[:5]))
    tag = "toneless" if args.toneless else "toned   "
    print(f"BEAM(B={args.beam}) {tag}: top1 {top1}/{n} ({100*top1/n:.1f}%)  "
          f"top1-var {topv}/{n} ({100*topv/n:.1f}%)  recall@{args.beam} {rec}/{n} ({100*rec/n:.1f}%)")
    print("  gold NOT in beam (genuine misses):", file=sys.stderr)
    for f in fails: print(f, file=sys.stderr)


if __name__ == "__main__":
    main()
