#!/usr/bin/env python3
"""SlothLM-K data prep: raw keystream -> per-key labels (solution 2).

The model that OWNS zh/en segmentation: input is one token per raw keystroke
(Dachen keys, letters, digits, tones); output is one label per keystroke:
  - the key that COMPLETES a Chinese syllable is labeled with the Han char
    (tone key if tonal; last symbol key if tone-1 committed by the next
    boundary); all other keys of that syllable are labeled BLANK.
  - an English keystroke is labeled with ITSELF (passthrough, learned).
So decoding stays non-autoregressive: one forward pass, read non-blank labels.
Segmentation (zh vs en) is no longer a hand-tuned DP -- the model learns it
from context, which is exactly what the DP cannot do (model/world/banana).

Records (uint16): [N] [key_id * N] [label_id * N]
  key vocab: model/key_vocab.json (PAD/UNK + printable keystrokes)
  label vocab: tokenizer ids for Han chars; ASCII keys label as themselves via
  a small offset table; BLANK=1.

Usage:
  python3 model/prepare_data_k.py --corpus model/corpus_cs.txt \
      --tokenizer model/tokenizer --out model/train_k.bin --vocab model/key_vocab.json
"""
import argparse
import json
import re
import sys

import numpy as np
from pypinyin import Style, pinyin
from transformers import AutoTokenizer

TONES = "ˊˇˋ˙"
KEYMAP = {"ㄅ":"1","ㄆ":"q","ㄇ":"a","ㄈ":"z","ㄉ":"2","ㄊ":"w","ㄋ":"s","ㄌ":"x",
 "ㄍ":"e","ㄎ":"d","ㄏ":"c","ㄐ":"r","ㄑ":"f","ㄒ":"v","ㄓ":"5","ㄔ":"t","ㄕ":"g",
 "ㄖ":"b","ㄗ":"y","ㄘ":"h","ㄙ":"n","ㄧ":"u","ㄨ":"j","ㄩ":"m","ㄚ":"8","ㄛ":"i",
 "ㄜ":"k","ㄝ":",","ㄞ":"9","ㄟ":"o","ㄠ":"l","ㄡ":".","ㄢ":"0","ㄣ":"p","ㄤ":";",
 "ㄥ":"/","ㄦ":"-","ˊ":"6","ˇ":"3","ˋ":"4","˙":"7"}
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+.#_-]*")

PAD, UNK, BLANK = 0, 1, 1   # key PAD/UNK; label BLANK=1 (labels use tokenizer ids >= 2 anyway)
# label ids for ASCII passthrough: 2..129 = chr(0)..chr(127) shifted
ASCII_BASE = 2


TW_RANK = {}       # char -> {syllable: rank in that syllable's candidate list}


def load_tw_table(path):
    for line in open(path, encoding="utf-8"):
        syl, _, rest = line.rstrip("\n").partition("\t")
        if not rest:
            continue
        for rank, ch in enumerate(rest):
            TW_RANK.setdefault(ch, {})[syl] = rank


def tw_correct(ch, syl):
    """pypinyin gives MAINLAND readings. The table is frequency-ordered per
    syllable (libchewing), with old pypinyin-only entries appended at the tail.
    Substitute pypinyin's reading when the char ranks FAR better under another
    reading (微: rank ~2 under ㄨㄟˊ vs tail under ㄨㄟ) -- but keep genuine
    context-dependent heteronyms (了 ㄌㄜ˙/ㄌㄧㄠˇ both rank well)."""
    ranks = TW_RANK.get(ch)
    if not ranks:
        return syl
    best_syl = min(ranks, key=ranks.get)
    cur = ranks.get(syl)
    if cur is None or cur > ranks[best_syl] + 15:
        return best_syl
    return syl


def bopomofo_syls(s):
    out = []
    for ch, g in zip(s, pinyin(s, style=Style.BOPOMOFO, errors="ignore")):
        syl = tw_correct(ch, g[0]) if g[0] else g[0]
        if not syl or any(c not in KEYMAP for c in syl):
            return None
        out.append(syl)
    return out if len(out) == len(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv",
                    help="TW phonetic table; corrects pypinyin's mainland readings")
    ap.add_argument("--out", default="model/train_k.bin")
    ap.add_argument("--vocab", default="model/key_vocab.json")
    ap.add_argument("--label-base", type=int, default=130,
                    help="tokenizer char ids are stored as id+label_base")
    ap.add_argument("--max-sentences", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    load_tw_table(args.table)
    print(f"TW table: {len(TW_READINGS)} chars", file=sys.stderr)
    # key vocab: every printable ASCII keystroke gets an input id
    key_vocab = {"<pad>": PAD, "<unk>": 1}
    for i in range(32, 127):
        key_vocab.setdefault(chr(i), len(key_vocab))
    json.dump(key_vocab, open(args.vocab, "w", encoding="utf-8"), ensure_ascii=False)

    def char_label(c):
        i = tok.convert_tokens_to_ids(c)
        return (i + args.label_base) if i is not None else BLANK

    def ascii_label(c):
        o = ord(c)
        return ASCII_BASE + o if o < 128 else BLANK

    rec = []
    n = kept = 0
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (1 <= len(s) <= 40):
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        keys, labels = [], []
        ok = True
        i = 0
        while i < len(s):
            m = LATIN.match(s, i)
            if m:                                   # English run: passthrough
                w = m.group(0)
                for c in w:
                    keys.append(c); labels.append(ascii_label(c))
                # boundary space after an English word (as typed in the IME)
                if m.end() < len(s):
                    keys.append(" "); labels.append(BLANK)
                i = m.end()
                continue
            if s[i].isspace():
                i += 1
                continue
            syl = bopomofo_syls(s[i])
            if syl is None:
                ok = False; break
            y = syl[0]
            ks = [KEYMAP[c] for c in y]
            tonal = y[-1] in TONES
            for j, k in enumerate(ks):
                keys.append(k)
                last = (j == len(ks) - 1)
                labels.append(char_label(s[i]) if last else BLANK)
            if not tonal:                           # tone-1: committed by space
                keys.append(" "); labels.append(BLANK)
            i += 1
        if not ok or not keys or len(keys) > 120:
            continue
        rec.append(len(keys))
        rec.extend(key_vocab.get(k, 1) for k in keys)
        rec.extend(labels)
        kept += 1
        if kept % 200000 == 0:
            print(f"  {kept}", file=sys.stderr)

    arr = np.array(rec, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {kept} keystream records ({len(arr)} uint16) to {args.out}; "
          f"key vocab {len(key_vocab)}, label = tokenizer id + {args.label_base}")


if __name__ == "__main__":
    main()
