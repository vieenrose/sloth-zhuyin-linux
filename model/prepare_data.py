#!/usr/bin/env python3
"""Build SlothLM's pretraining token stream as a uint16 .bin memmap.

Mixes, per the design:
  - plain Traditional-Chinese sentences (language fluency)
  - synthetic TASK examples rendered in the serving ChatML format:
      * SELECT  : the daemon's exact "第N字選(候選...)" prompt -> sentence,
                  candidates harvested from real libchewing (eval/harvest)
      * Z2T     : zhuyin syllables -> sentence
      * T2Z     : sentence -> zhuyin syllables
      * TONELESS: toneless zhuyin -> sentence (future tone-free mode)
Each document ends with <|im_end|>+<|endoftext|> so the model learns to stop.
eval/testset.tsv sentences are excluded (no eval leakage).

Usage:
  python3 model/prepare_data.py --corpus model/corpus_big.txt \
      --tokenizer model/tokenizer --out model/train.bin
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from pypinyin import Style, pinyin
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HARVEST = os.path.join(REPO, "eval", "harvest")
KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s",
    "ㄌ": "x", "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m", "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".", "ㄢ": "0", "ㄣ": "p", "ㄤ": ";",
    "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}
TONES = "ˊˇˋ˙"


def bopomofo(s):
    out = []
    for g in pinyin(s, style=Style.BOPOMOFO, errors="ignore"):
        syl = g[0]
        if not syl or any(c not in KEYMAP for c in syl):
            return None
        out.append(syl)
    return out if len(out) == len(s) else None


def keys(syls):
    k = []
    for syl in syls:
        k += [KEYMAP[c] for c in syl]
        if syl[-1] not in TONES:
            k.append(" ")
    return "".join(k)


def toneless(syls):
    return " ".join("".join(c for c in syl if c not in TONES) for syl in syls)


def harvest_positions(k):
    r = subprocess.run([HARVEST, k], capture_output=True, text=True)
    if r.returncode != 0:
        return None, None
    d = json.loads(r.stdout)
    return d["buffer"], d["positions"]


def load_holdout():
    hold = set()
    ts = os.path.join(REPO, "eval", "testset.tsv")
    if os.path.exists(ts):
        for line in open(ts, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                c = line.split("\t")
                if len(c) > 1:
                    hold.add(c[1])
    return hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--out", default="model/train.bin")
    ap.add_argument("--max-sentences", type=int, default=0, help="0 = all")
    ap.add_argument("--select-limit", type=int, default=200000,
                    help="cap SELECT examples (each needs a slow harvest call)")
    args = ap.parse_args()
    if not os.path.exists(HARVEST):
        sys.exit("build eval/harvest first")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    hold = load_holdout()

    def chatml(system, user, assistant):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)

    ids = []
    stats = {"plain": 0, "select": 0, "z2t": 0, "t2z": 0, "toneless": 0}

    def emit(text):
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        ids.append(eot)

    n = 0
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or s in hold or not (2 <= len(s) <= 18):
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        # plain LM
        emit(s)
        stats["plain"] += 1
        syls = bopomofo(s)
        if syls is None:
            continue
        emit(chatml("注音轉繁體中文。", " ".join(syls), s)); stats["z2t"] += 1
        emit(chatml("繁體中文轉注音。", s, " ".join(syls))); stats["t2z"] += 1
        emit(chatml("注音轉繁體中文。", toneless(syls), s)); stats["toneless"] += 1
        if stats["select"] < args.select_limit:
            buf, positions = harvest_positions(keys(syls))
            if buf == s and positions and any(len(p) > 1 for p in positions):
                user = " ".join(f"第{i+1}字選(" + "/".join(p) + ")"
                                for i, p in enumerate(positions))
                emit(chatml("選字。", user, s)); stats["select"] += 1
        if n % 20000 == 0:
            print(f"  {n} sentences, {len(ids)/1e6:.1f}M tokens", file=sys.stderr)

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {len(arr)} tokens ({len(arr)/1e6:.1f}M) to {args.out}")
    print(f"  docs by type: {stats}")


if __name__ == "__main__":
    main()
