#!/usr/bin/env python3
"""Build aligned (syllable-id, char-id) pairs for SlothLM-E (encoder).

Unlike the decoder pipeline (symbol-level tokens, ChatML), SlothLM-E is a
sequence-labeling model: input is one **syllable** token per position, output
is one **char** token per position, 1:1 aligned. This writes a packed uint16
stream of records:  [N] [syl_0..syl_{N-1}] [char_0..char_{N-1}]

Input syllable vocab (syl_vocab.json): built from phonetic_table.tsv (tonal
syllables) + their toneless bases + PAD/UNK/EN. Output char ids come from the
existing HF tokenizer (so the phonetic-legality candidate mask is unchanged).

Usage:
  python3 model/prepare_data_e.py --corpus model/corpus_cs.txt \
      --tokenizer model/tokenizer --table model/phonetic_table.tsv \
      --out model/train_e.bin --vocab model/syl_vocab.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from pypinyin import Style, pinyin
from transformers import AutoTokenizer

TONES = "ˊˇˋ˙"
KEYMAP_OK = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + TONES)
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+.#_-]*")

# reserved input tokens
PAD, UNK, EN = 0, 1, 2


def bopomofo_syls(s):
    out = []
    for g in pinyin(s, style=Style.BOPOMOFO, errors="ignore"):
        syl = g[0]
        if not syl or any(c not in KEYMAP_OK for c in syl):
            return None
        out.append(syl)
    return out if len(out) == len(s) else None


def build_syl_vocab(table_path):
    v = {"<pad>": PAD, "<unk>": UNK, "<en>": EN}
    tonal = set()
    for line in open(table_path, encoding="utf-8"):
        syl = line.split("\t")[0]
        if syl:
            tonal.add(syl)
    for syl in sorted(tonal):
        v.setdefault(syl, len(v))
    for syl in sorted(tonal):                       # toneless bases too
        base = "".join(c for c in syl if c not in TONES)
        v.setdefault(base, len(v))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    ap.add_argument("--out", default="model/train_e.bin")
    ap.add_argument("--vocab", default="model/syl_vocab.json")
    ap.add_argument("--max-sentences", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    syl_vocab = build_syl_vocab(args.table)
    json.dump(syl_vocab, open(args.vocab, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"syllable vocab: {len(syl_vocab)}; char vocab: {tok.vocab_size}",
          file=sys.stderr)

    def char_id(c):
        i = tok.convert_tokens_to_ids(c)
        return i if i is not None else tok.unk_token_id

    rec = []
    n = kept = 0
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (2 <= len(s) <= 30):
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        # Build aligned syllable/char sequences; English runs = one <en> input
        # position mapping to... skip code-switch here (encoder passthrough is
        # handled at serving); train on pure-Han sentences for clean alignment.
        if LATIN.search(s):
            continue
        syls = bopomofo_syls(s)
        if syls is None:
            continue
        syl_ids = [syl_vocab.get(y, UNK) for y in syls]
        char_ids = [char_id(c) for c in s]
        if len(syl_ids) != len(char_ids) or not syl_ids:
            continue
        rec.append(len(syl_ids))
        rec.extend(syl_ids)
        rec.extend(char_ids)
        kept += 1
        if kept % 100000 == 0:
            print(f"  {kept} aligned pairs", file=sys.stderr)

    arr = np.array(rec, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {kept} aligned pairs ({len(arr)} uint16) to {args.out}")


if __name__ == "__main__":
    main()
