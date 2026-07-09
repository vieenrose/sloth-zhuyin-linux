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

    IGNORE = 65535  # char label at English positions -> loss ignored (passthrough)

    def char_id(c):
        i = tok.convert_tokens_to_ids(c)
        return i if i is not None else tok.unk_token_id

    def aligned(s, toneless):
        """Return (syl_ids, char_ids) or None. Han runs -> (syllable, char);
        English runs -> one <en> input position with IGNORE label (the encoder
        sees English as bidirectional context; English is passthrough at
        serving). toneless=True strips tone marks from the input syllable."""
        syl_ids, char_ids, i = [], [], 0
        while i < len(s):
            m = LATIN.match(s, i)
            if m:
                syl_ids.append(EN)
                char_ids.append(IGNORE)
                i = m.end()
                continue
            if s[i].isspace():
                i += 1
                continue
            syl = bopomofo_syls(s[i])
            if syl is None:
                return None
            y = syl[0]
            if toneless:
                y = "".join(c for c in y if c not in TONES)
            syl_ids.append(syl_vocab.get(y, UNK))
            char_ids.append(char_id(s[i]))
            i += 1
        # need at least one decoded (Han) position
        if not any(c != IGNORE for c in char_ids):
            return None
        return syl_ids, char_ids

    rec = []
    stats = {"tonal": 0, "toneless": 0, "codeswitch": 0}
    n = 0
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (2 <= len(s) <= 30):
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        mixed = bool(LATIN.search(s))
        for toneless in (False, True):          # every objective: tonal + toneless
            a = aligned(s, toneless)
            if a is None:
                continue
            syl_ids, char_ids = a
            rec.append(len(syl_ids))
            rec.extend(syl_ids)
            rec.extend(char_ids)
            k = "toneless" if toneless else ("codeswitch" if mixed else "tonal")
            stats[k] += 1
        if n % 100000 == 0:
            print(f"  {n} sentences", file=sys.stderr)

    arr = np.array(rec, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {sum(stats.values())} aligned records "
          f"({len(arr)} uint16) to {args.out}; by type: {stats}")


if __name__ == "__main__":
    main()
