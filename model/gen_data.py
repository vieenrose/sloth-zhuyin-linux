#!/usr/bin/env python3
"""Generation-style data for the ONE-LLM prototype: `phonetic → text` pairs that
PRESERVE English (zh-EN code-switch) and emoji, with tone-dropout augmentation.

Unlike prepare_data_e_g2pw.py (which packs token-ids and collapses English to an
<en> placeholder), this keeps the literal text so a causal LM learns:
  input : bopomofo syllables (space-separated) with English/number runs inline
  output: the original sentence (Chinese + English + emoji), verbatim

Each sentence yields a toned, a toneless, and a random partial-tone variant so
the LLM tolerates 音標皆可 input.

  G2PW_CUDA=1 python3 gen_data.py --corpus corpus_e3.txt --out gen_pairs.jsonl
"""
import argparse, json, os, random, sys
if os.environ.get("G2PW_CUDA"):
    import onnxruntime as _ort
    _o = _ort.InferenceSession
    _ort.InferenceSession = lambda *a, **k: _o(*a, **{**k, "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]})
from g2pw import G2PWConverter

TONES = "ˊˇˋ˙"
TONE_NUM = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}
KEEP = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + TONES)


def to_bopo(reading):
    if not reading: return None
    base, tone = reading[:-1], reading[-1]
    if tone not in TONE_NUM: return None
    syl = base + TONE_NUM[tone]
    return syl if all(c in KEEP for c in syl) else None


def strip_tones(syl): return "".join(c for c in syl if c not in TONES)


def build_input(sent, readings, drop):
    """drop: prob a syllable loses its tone (0=toned, 1=toneless)."""
    parts, i = [], 0
    while i < len(sent):
        ch = sent[i]
        if ("一" <= ch <= "鿿") and readings[i]:
            syl = to_bopo(readings[i])
            if syl is None: return None
            if random.random() < drop: syl = strip_tones(syl)
            parts.append(syl); i += 1
        elif ch.isascii() and (ch.isalnum()):
            run = ""
            while i < len(sent) and sent[i].isascii() and sent[i].isalnum():
                run += sent[i]; i += 1
            parts.append(run)                    # English/number literal
        else:
            i += 1                               # skip spaces/punct in the INPUT
    return " ".join(parts) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="gen_pairs.jsonl")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-sentences", type=int, default=0)
    args = ap.parse_args()
    random.seed(0)
    g2p = G2PWConverter(style="bopomofo", batch_size=args.batch)

    fout = open(args.out, "w", encoding="utf-8")
    buf, n, npair = [], 0, 0

    def flush():
        nonlocal npair
        sents = [s for s in buf if s]
        if not sents: return
        for s, readings in zip(sents, g2p(sents)):
            for drop in (0.0, 1.0, random.uniform(0.3, 0.8)):   # toned / toneless / partial
                inp = build_input(s, readings, drop)
                if inp and any("一" <= c <= "鿿" for c in s):
                    fout.write(json.dumps({"in": inp, "out": s}, ensure_ascii=False) + "\n")
                    npair += 1

    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (1 <= len(s) <= 40): continue
        n += 1
        if args.max_sentences and n > args.max_sentences: break
        buf.append(s)
        if len(buf) >= args.batch:
            flush(); buf = []
            if n % 51200 == 0: print(f"  {n} sents -> {npair} pairs", file=sys.stderr)
    flush(); fout.close()
    print(f"wrote {npair} pairs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
