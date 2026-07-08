#!/usr/bin/env python3
"""Build SlothLM's pretraining token stream as a uint16 .bin memmap.

Mixes, per the design:
  - plain Traditional-Chinese sentences (language fluency)
  - synthetic TASK examples rendered in the serving ChatML format:
      * SELECT  : the daemon's exact "第N字選(候選...)" prompt -> sentence,
                  candidates harvested from real libchewing (eval/harvest)
      * SELECT1 : single-slot SELECT -- every position but one is fixed to
                  the right char (the per-phrase Down-rank query shape)
      * Z2T     : zhuyin syllables -> sentence
      * T2Z     : sentence -> zhuyin syllables
      * TONELESS: toneless zhuyin -> sentence (future tone-free mode)
A fraction of SELECT/Z2T examples carry the serving-time context prefix
("前文＿＿＿\n" + user msg) built from the preceding corpus sentence, so
context-bearing queries are in-distribution (the daemon prepends exactly
this shape when the app reports surrounding text).
Each document ends with <|im_end|>+<|endoftext|> so the model learns to stop.
eval/testset.tsv sentences are excluded (no eval leakage).

Usage:
  python3 model/prepare_data.py --corpus model/corpus_big.txt \
      --tokenizer model/tokenizer --out model/train.bin
"""
import argparse
import hashlib
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
    stats = {"plain": 0, "select": 0, "select1": 0, "z2t": 0, "t2z": 0,
             "toneless": 0, "with_ctx": 0}

    def emit(text):
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        ids.append(eot)

    def bucket(s, salt):
        # deterministic pseudo-random in [0,100) so builds are reproducible
        h = hashlib.sha1((salt + s).encode()).digest()
        return h[0] * 100 // 256

    n = 0
    prev = ""  # preceding corpus sentence (adjacent lines are same-doc)
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or s in hold or not (2 <= len(s) <= 18):
            prev = ""
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        # Serving-time context prefix (see build_user_message in slothingd):
        # committed text before the cursor, phrased as fill-in-the-blank.
        # ~30% of task docs carry it, when a preceding sentence exists.
        ctx = prev + "＿＿＿\n" if prev and bucket(s, "ctx") < 30 else ""
        if ctx:
            stats["with_ctx"] += 1
        # plain LM
        emit(s)
        stats["plain"] += 1
        syls = bopomofo(s)
        if syls is None:
            prev = s
            continue
        emit(chatml("注音轉繁體中文。", ctx + " ".join(syls), s))
        stats["z2t"] += 1
        emit(chatml("繁體中文轉注音。", s, " ".join(syls))); stats["t2z"] += 1
        emit(chatml("注音轉繁體中文。", ctx + toneless(syls), s))
        stats["toneless"] += 1
        if stats["select"] < args.select_limit:
            buf, positions = harvest_positions(keys(syls))
            if buf == s and positions and any(len(p) > 1 for p in positions):
                user = " ".join(f"第{i+1}字選(" + "/".join(p) + ")"
                                for i, p in enumerate(positions))
                emit(chatml("選字。", ctx + user, s)); stats["select"] += 1
                # Single-slot variant: the per-phrase Down-rank query shape.
                # One interval keeps its full candidate list; every other is
                # pinned to the right answer. Slot picked deterministically
                # among multi-candidate intervals.
                multi = [i for i, p in enumerate(positions) if len(p) > 1]
                slot = multi[bucket(s, "slot") % len(multi)]
                spans, off = [], 0
                for p in positions:
                    w = len(p[0])
                    spans.append(s[off:off + w])
                    off += w
                if off == len(s):
                    user1 = " ".join(
                        f"第{i+1}字選(" + "/".join(
                            positions[i] if i == slot else [spans[i]]) + ")"
                        for i in range(len(positions)))
                    emit(chatml("選字。", ctx + user1, s))
                    stats["select1"] += 1
        prev = s
        if n % 20000 == 0:
            print(f"  {n} sentences, {len(ids)/1e6:.1f}M tokens", file=sys.stderr)

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {len(arr)} tokens ({len(arr)/1e6:.1f}M) to {args.out}")
    print(f"  docs by type: {stats}")


if __name__ == "__main__":
    main()
