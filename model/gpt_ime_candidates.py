#!/usr/bin/env python3
"""Candidate extraction for the GPT-IME — the inside-span constrained search that
feeds the unified iOS candidate bar. From one input it produces, on ONE log-prob
scale (the design's unified ranking):

  * n-best SENTENCES        — constrained beam (width B), top-k full conversions
  * per-syllable CHAR cands — masked top-k at a position, conditioned on the prefix
  * WORD/PHRASE cands       — local constrained beam of depth K from a position

All grounded to each syllable's phonetic-legal set (tone-union for unmarked). This
is what turns the 93%-toned converter into an IME. Demo:

  python3 gpt_ime_candidates.py --model slothe_gpt_gemma --input "su3 cl3"
  python3 gpt_ime_candidates.py --model slothe_gpt_gemma --input "ㄋㄧ ㄏㄠ" --nbest 5
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

TONES = "ˊˇˋ˙"
DACHEN = {'1':'ㄅ','q':'ㄆ','a':'ㄇ','z':'ㄈ','2':'ㄉ','w':'ㄊ','s':'ㄋ','x':'ㄌ','e':'ㄍ','d':'ㄎ',
 'c':'ㄏ','r':'ㄐ','f':'ㄑ','v':'ㄒ','5':'ㄓ','t':'ㄔ','g':'ㄕ','b':'ㄖ','y':'ㄗ','h':'ㄘ','n':'ㄙ',
 'u':'ㄧ','j':'ㄨ','m':'ㄩ','8':'ㄚ','i':'ㄛ','k':'ㄜ',',':'ㄝ','9':'ㄞ','o':'ㄟ','l':'ㄠ','.':'ㄡ',
 '0':'ㄢ','p':'ㄣ',';':'ㄤ','/':'ㄥ','-':'ㄦ'}
TONEK = {'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'}


def to_bopo(tok):
    """Accept either bopomofo ('ㄋㄧˇ') or a dachen key-run ('su3') per syllable."""
    if any(c in DACHEN.values() or c in TONES for c in tok):
        return tok
    out = ""
    for c in tok:
        if c in DACHEN: out += DACHEN[c]
        elif c in TONEK: out += TONEK[c]
    return out


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
    ap.add_argument("--input", required=True, help="space-separated syllables (bopomofo or dachen)")
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--scan", type=int, default=60)
    ap.add_argument("--chark", type=int, default=6)
    ap.add_argument("--wordK", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    legal = tone_union(args.table)
    dcache = {}
    def dec1(t):
        if t not in dcache: dcache[t] = tok.decode([t])
        return dcache[t]

    syls = [to_bopo(s) for s in args.input.split()]
    N = len(syls)
    legalsets = [legal(s) for s in syls]
    prompt = tok(" ".join(syls) + " →", add_special_tokens=False,
                 return_tensors="pt").input_ids.to(dev)

    @torch.no_grad()
    def logits_after(gen_ids):
        ids = torch.cat([prompt, torch.tensor([gen_ids], device=dev, dtype=prompt.dtype)], 1) \
              if gen_ids else prompt
        return F.log_softmax(model(ids).logits[0, -1], -1)

    # ---- constrained beam over the whole span -> n-best sentences ----
    @torch.no_grad()
    def beam(depth_limit=N):
        beams = [(0.0, [], "", 0)]          # score, gen_ids, decoded, p
        for _ in range(depth_limit + 2):
            if all(p >= depth_limit for _, _, _, p in beams): break
            cand = []
            for score, gids, dec, p in beams:
                if p >= depth_limit:
                    cand.append((score, gids, dec, p)); continue
                lp = logits_after(gids)
                for t in torch.argsort(lp, descending=True)[: args.scan].tolist():
                    txt = dec1(t)
                    if not txt: continue
                    nd = dec + txt; ch = list(nd)
                    if len(ch) <= p: continue
                    if all(ch[i] in legalsets[i] for i in range(min(len(ch), depth_limit))):
                        cand.append((score + lp[t].item(), gids + [t], nd, min(len(ch), depth_limit)))
            cand.sort(key=lambda x: x[0], reverse=True)
            beams = cand[: args.beam]
        done = [(s, dec[:depth_limit]) for s, g, dec, p in beams if p >= depth_limit]
        # dedup by surface, keep best
        seen = {}
        for s, txt in done:
            if txt not in seen or s > seen[txt]: seen[txt] = s
        return sorted(seen.items(), key=lambda x: -x[1])

    # ---- per-syllable char candidates (fix-a-char), conditioned on greedy prefix ----
    @torch.no_grad()
    def char_cands(pos):
        # greedy-fill positions < pos, then top-k legal at pos
        gids, dec, p = [], "", 0
        while p < pos:
            lp = logits_after(gids)
            for t in torch.argsort(lp, descending=True)[: args.scan].tolist():
                txt = dec1(t); nd = dec + txt; ch = list(nd)
                if len(ch) > p and all(ch[i] in legalsets[i] for i in range(min(len(ch), pos + 1))):
                    gids.append(t); dec = nd; p = len(ch); break
            else: break
        lp = logits_after(gids)
        scored = []
        for t in torch.argsort(lp, descending=True)[: args.scan * 2].tolist():
            txt = dec1(t)
            if len(txt) == 1 and txt in legalsets[pos]:
                scored.append((txt, lp[t].item()))
            if len(scored) >= args.chark: break
        return scored

    print(f"input: {' '.join(syls)}  ({N} syllables)")
    print(f"\n=== n-best SENTENCES (constrained beam B={args.beam}) ===")
    for txt, s in beam()[: args.nbest]:
        print(f"  {txt}    ({s:.2f})")
    print(f"\n=== CHAR candidates for syllable 0 ({syls[0]}) ===")
    print("  " + "  ".join(f"{c}" for c, _ in char_cands(0)))
    if N >= 2:
        print(f"=== CHAR candidates for syllable {N-1} ({syls[-1]}) ===")
        print("  " + "  ".join(f"{c}" for c, _ in char_cands(N - 1)))
    print(f"\n=== WORD/PHRASE from syllable 0 (constrained beam depth<={min(args.wordK,N)}) ===")
    for txt, s in beam(depth_limit=min(args.wordK, N))[:6]:
        print(f"  {txt}    ({s:.2f})")


if __name__ == "__main__":
    main()
