#!/usr/bin/env python3
"""Augment a SlothLM-E .bin with RANDOM PARTIAL-tone dropout.

The data prep (prepare_data_e_g2pw.py) already emits a fully-toned and a
fully-toneless record per sentence, but the model still disambiguates toneless
input poorly (~40% vs ~83% toned) because real "音標皆可" input is a *mix* of
marked and unmarked syllables and the all-or-nothing training under-covers that
combinatorial space. This adds K variants per toned record where each syllable
independently keeps its tone with prob (1-drop); drop is sampled per-variant
from a range so the model sees every marked/unmarked mixture.

Record layout (uint16 stream, repeated): [len, syl_id*len, char_id*len].
A syllable token is detoned by mapping it to its unmarked base token (strip tone
marks, look up in syl_vocab); tokens with no base stay as-is. char_ids unchanged.

  python3 augment_toneless.py --in train_e_g2pw.bin --vocab syl_vocab.json \
      --out train_e_g2pw_tl.bin --variants 2 --seed 0
"""
import argparse, json
import numpy as np

TONES = "ˊˇˋ˙"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=2,
                    help="partial-dropout variants to add per toned record")
    ap.add_argument("--drop-lo", type=float, default=0.3)
    ap.add_argument("--drop-hi", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab = json.load(open(args.vocab, encoding="utf-8"))
    id2syl = {v: k for k, v in vocab.items()}
    # toned token id -> unmarked base token id (if the base exists in vocab)
    detone = {}
    for syl, tid in vocab.items():
        if any(t in syl for t in TONES):
            base = "".join(c for c in syl if c not in TONES)
            if base in vocab:
                detone[tid] = vocab[base]
    print(f"detone map: {len(detone)} toned->base token pairs")

    data = np.fromfile(args.inp, dtype=np.uint16)
    rng = np.random.default_rng(args.seed)
    out = []  # list of np arrays (whole records) to concatenate
    n_rec = n_toned = n_added = 0
    i = 0
    while i < len(data):
        L = int(data[i])
        rec = data[i:i + 1 + 2 * L]
        if len(rec) < 1 + 2 * L:
            break
        out.append(rec)                     # keep original
        n_rec += 1
        syl = rec[1:1 + L]
        is_toned = any(int(s) in detone for s in syl)
        if is_toned:
            n_toned += 1
            for _ in range(args.variants):
                drop = rng.uniform(args.drop_lo, args.drop_hi)
                new = rec.copy()
                for j in range(L):
                    tid = int(new[1 + j])
                    if tid in detone and rng.random() < drop:
                        new[1 + j] = detone[tid]
                out.append(new)
                n_added += 1
        i += 1 + 2 * L

    arr = np.concatenate(out).astype(np.uint16)
    arr.tofile(args.out)
    print(f"records: {n_rec} in ({n_toned} toned) -> +{n_added} partial-dropout "
          f"variants; wrote {len(arr)} uint16 to {args.out}")


if __name__ == "__main__":
    main()
