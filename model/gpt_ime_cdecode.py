#!/usr/bin/env python3
"""Char-level CONSTRAINED-decode eval for the GPT-IME (gemma/qwen fine-tune).

The earlier per-token single-token mask (gpt_ime_eval.py) was broken for BPE-merge
tokenizers (你好 = 1 token). The correct constraint is on the DECODED TEXT: at each
step pick the highest-logit token whose decoded chars all land in their syllable's
phonetic-legal set (tone-union for unmarked => toneless grounding). A token may add
1 char (advance p by 1), a merged word like 你好 (advance p by 2, both checked), and
English runs are force-copied. KV-cached, one committed token per step.

This is the design's "inside-span phonetic-constrained decode" — it grounds the
autoregressive model to the typed syllables, the mechanism that should close the
toneless gap (gemma unconstrained: toned 93.1 / toneless 81.8).

  python3 gpt_ime_cdecode.py --model slothe_gpt_gemma [--toneless] [--scan 50]
"""
import argparse, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TONES = "ˊˇˋ˙"
VAR = str.maketrans({"臺": "台", "箇": "個"})   # accept common Traditional variants


def tone_union(table_path):
    tonal = {}
    for line in open(table_path, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if rest:
            tonal[s] = set(rest)
    def legal(syl):
        has = any(c in TONES for c in syl)
        chars = tonal.get(syl) if has else None
        if chars is None:
            base = "".join(x for x in syl if x not in TONES)
            chars = set()
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base:
                    chars |= v
        return chars or {syl}
    return legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--toneless", action="store_true")
    ap.add_argument("--scan", type=int, default=50, help="top-N logits scanned per step")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    legal = tone_union(args.table)
    dcache = {}
    def dec1(t):                                   # cached single-token decode
        if t not in dcache: dcache[t] = tok.decode([t])
        return dcache[t]

    @torch.no_grad()
    def cdecode(syls):
        legalsets = [legal(s) for s in syls]
        N = len(syls)
        ids = tok(" ".join(syls) + " →", add_special_tokens=False,
                  return_tensors="pt").input_ids.to(dev)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        decoded, p = "", 0
        while p < N:
            order = torch.argsort(logits, descending=True)[: args.scan].tolist()
            picked = None
            for t in order:
                txt = dec1(t)
                if not txt:
                    continue
                nd = decoded + txt
                chars = list(nd)
                if len(chars) <= p:               # must advance the pointer
                    continue
                # every char up to min(len,N) must be legal at its position
                good = all(chars[i] in legalsets[i] for i in range(min(len(chars), N)))
                if good:
                    picked = t; decoded = nd; p = min(len(chars), N); break
            if picked is None:                     # stuck: no legal advancing token in top-scan
                break
            step = model(torch.tensor([[picked]], device=dev),
                         past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[0, -1]
        return decoded[:N]

    n = ok = okv = 0; fails = []
    t0 = time.time()
    for line in open(args.testset, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t"); syls = p[0].split(); exp = p[1]
        if args.toneless:
            syls = ["".join(c for c in y if c not in TONES) for y in syls]
        got = cdecode(syls)
        n += 1
        ok += (got == exp)
        okv += (got.translate(VAR) == exp.translate(VAR))
        if got.translate(VAR) != exp.translate(VAR) and len(fails) < args.show:
            fails.append("    " + exp + " | " + got)
    dt = time.time() - t0
    tag = "toneless" if args.toneless else "toned   "
    print(f"gemma CONSTRAINED {tag}: raw {ok}/{n} ({100*ok/n:.1f}%)  "
          f"variant-norm {okv}/{n} ({100*okv/n:.1f}%)  [{1000*dt/n:.0f} ms/sent]")
    for f in fails:
        print(f, file=sys.stderr)


if __name__ == "__main__":
    main()
