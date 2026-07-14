#!/usr/bin/env python3
"""Constrained-decode eval for a GPT-IME (any HF causal LM fine-tuned on
bopomofo→text via gen_convert.py). Tests the thesis: does an autoregressive LM
with INSIDE-span phonetic-constrained decoding beat the SlothE encoder on
conversion accuracy (toned + toneless)?

Decoding = the design's inside-constrained greedy: at output position p, mask the
next-token logits to the SINGLE-TOKEN legal chars of syllable p (tone-union for
unmarked => toneless), argmax, advance p. KV-cached. (Single-token masking is the
MVP; a trie over multi-token chars is the production upgrade — on gemma-3-270m /
Qwen the testset is ~100% single-token so this loses ~nothing.)

  python3 gpt_ime_eval.py --model slothe_gpt_qwen --base Qwen/Qwen2.5-0.5B
  python3 gpt_ime_eval.py --model slothe_gpt_qwen --base Qwen/Qwen2.5-0.5B --toneless
"""
import argparse, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TONES = "ˊˇˋ˙"


def tone_union(table_path):
    tonal = {}
    for line in open(table_path, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if rest:
            tonal[s] = list(rest)
    def legal(syl):
        has = any(c in TONES for c in syl)
        chars = tonal.get(syl) if has else None
        if chars is None:
            base = "".join(x for x in syl if x not in TONES)
            chars = []
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base:
                    for ch in v:
                        if ch not in chars:
                            chars.append(ch)
        return chars or [syl]
    return legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--toneless", action="store_true")
    ap.add_argument("--sep", default=" →")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print N failing cases")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model if args.model else args.base)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    legal = tone_union(args.table)

    # precompute: syllable -> tensor of single-token legal char ids (+ id->char)
    cache = {}
    def legal_ids(syl):
        if syl in cache:
            return cache[syl]
        id2ch = {}
        for ch in legal(syl):
            ids = tok(ch, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:                 # MVP: single-token legal chars only
                id2ch.setdefault(ids[0], ch)
        idt = torch.tensor(list(id2ch.keys()), device=dev) if id2ch else None
        cache[syl] = (idt, id2ch)
        return cache[syl]

    @torch.no_grad()
    def decode(syls):
        prompt = " ".join(syls) + args.sep
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        res = []
        for p, syl in enumerate(syls):
            idt, id2ch = legal_ids(syl)
            if idt is None:                   # no single-token legal char -> give up this pos
                res.append("?"); nxt = tok("?", add_special_tokens=False)["input_ids"][0]
            else:
                masked = torch.full_like(logits, float("-inf"))
                masked[idt] = logits[idt]
                nxt = int(masked.argmax())
                res.append(id2ch.get(nxt, "?"))
            step = model(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[0, -1]
        return "".join(res)

    n = ok = 0; fails = []
    t0 = time.time()
    for line in open(args.testset, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t"); syls = p[0].split(); exp = p[1]
        if args.toneless:
            syls = ["".join(c for c in y if c not in TONES) for y in syls]
        got = decode(syls)
        n += 1; ok += (got == exp)
        if got != exp and len(fails) < args.show:
            fails.append(f"  {exp}  got:{got}  [{p[0]}]")
        if args.limit and n >= args.limit:
            break
    dt = time.time() - t0
    tag = "toneless" if args.toneless else "toned   "
    print(f"GPT-IME {args.model}  {tag}: {ok}/{n} ({100*ok/n:.1f}%)  "
          f"[{1000*dt/n:.0f} ms/sent]")
    for f in fails:
        print(f, file=sys.stderr)


if __name__ == "__main__":
    main()
