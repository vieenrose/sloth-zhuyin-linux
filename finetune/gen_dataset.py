#!/usr/bin/env python3
"""Generate the Sloth IME fine-tuning dataset.

For each training sentence we build three kinds of example, all in the chat
format the daemon uses at inference so training matches serving:

  1. SELECT  — the real task: given the per-position candidate lists that
     libchewing actually produces for the sentence's pronunciation (harvested
     via eval/harvest, same code path as the engine), output the correct
     sentence. This is what the grammar-constrained decoder is scored on.
  2. Z2T     — zhuyin sequence -> sentence (no candidate constraint). Teaches
     the phonetic->text mapping directly (PY-GEC: off-the-shelf LLMs align
     these poorly until explicitly trained; cosine 0.26 -> 0.82).
  3. T2Z     — sentence -> zhuyin sequence. The reverse task, which PY-GEC
     found further strengthens alignment.

Sentences whose harvested candidate sets don't reconstruct the sentence, or
that appear in the eval test set, are skipped (no train/test leakage).

Usage:
  python3 finetune/gen_dataset.py --sentences corpus.txt --out train.jsonl
corpus.txt: one Traditional-Chinese sentence per line.
"""
import argparse
import json
import os
import subprocess
import sys

from pypinyin import pinyin, Style

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HARVEST = os.path.join(REPO, "eval", "harvest")

# Dachen layout: same table as eval/run_eval.py (keep in sync).
KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s",
    "ㄌ": "x", "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m", "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".", "ㄢ": "0", "ㄣ": "p", "ㄤ": ";",
    "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}
TONES = "ˊˇˋ˙"
SYSTEM = "選字。"


def sentence_bopomofo(s):
    """List of per-character bopomofo syllables, or None if any char is
    non-Han (pypinyin echoes those back unchanged)."""
    out = []
    for group in pinyin(s, style=Style.BOPOMOFO, errors="ignore"):
        syl = group[0]
        if not syl or syl[0] not in KEYMAP or any(c not in KEYMAP for c in syl):
            return None
        out.append(syl)
    if len(out) != len(s):
        return None
    return out


def bopomofo_to_keys(syllables):
    keys = []
    for syl in syllables:
        for ch in syl:
            keys.append(KEYMAP[ch])
        if syl[-1] not in TONES:
            keys.append(" ")
    return "".join(keys)


def harvest(keys):
    r = subprocess.run([HARVEST, keys], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def build_user_select(positions):
    """Mirror slothd.cpp build_user_message (no context)."""
    parts = []
    for i, cands in enumerate(positions):
        parts.append(f"第{i+1}字選(" + "/".join(cands) + ")")
    return " ".join(parts)


def chat(system, user, assistant):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def load_eval_expected():
    expected = set()
    ts = os.path.join(REPO, "eval", "testset.tsv")
    if os.path.exists(ts):
        for line in open(ts, encoding="utf-8"):
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                cols = line.split("\t")
                if len(cols) > 1:
                    expected.add(cols[1])
    return expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=18)
    args = ap.parse_args()

    if not os.path.exists(HARVEST):
        sys.exit("build eval/harvest first (see eval/README.md)")

    holdout = load_eval_expected()
    n_in = n_skip_eval = n_skip_harvest = n_written = 0
    stats = {"select": 0, "z2t": 0, "t2z": 0}

    with open(args.sentences, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n_in += 1
            if s in holdout:
                n_skip_eval += 1
                continue
            if len(s) < 2 or len(s) > args.max_len:
                continue
            syls = sentence_bopomofo(s)
            if syls is None:
                continue
            keys = bopomofo_to_keys(syls)
            h = harvest(keys)
            if not h or h.get("buffer") != s:
                n_skip_harvest += 1
                continue
            positions = h["positions"]
            # SELECT must be genuinely ambiguous to be useful signal.
            if any(len(c) > 1 for c in positions):
                out.write(json.dumps(
                    chat(SYSTEM, build_user_select(positions), s),
                    ensure_ascii=False) + "\n")
                stats["select"] += 1
                n_written += 1
            zseq = " ".join(syls)
            out.write(json.dumps(
                chat("注音轉繁體中文。", zseq, s), ensure_ascii=False) + "\n")
            stats["z2t"] += 1
            out.write(json.dumps(
                chat("繁體中文轉注音。", s, zseq), ensure_ascii=False) + "\n")
            stats["t2z"] += 1
            n_written += 2

    print(f"in={n_in} skipped(eval-holdout={n_skip_eval}, "
          f"harvest-mismatch={n_skip_harvest}) examples written={n_written}")
    print(f"  by task: {stats}")


if __name__ == "__main__":
    main()
