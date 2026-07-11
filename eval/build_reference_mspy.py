#!/usr/bin/env python3
"""Build the 免選字 eval set by sampling everyday sentences and generating gold
bopomofo with g2pW (the same context-aware labeler used for training, so the
readings are consistent).

⚠️  LEAKAGE WARNING (2026-07-12): --corpus MUST be a HELD-OUT corpus that is NOT
in training. An earlier version sampled from the TRAINING corpus (corpus_e3.txt);
174/174 of those sampled sentences were verbatim in training, so the metric
rewarded memorization and inflated every 免選字 number by ~15-18 pts (shipped
model measured 84% on the leaked set vs ~66% honest held-out). ALWAYS pass
--exclude <training_corpus> so any sampled sentence that appears in training is
dropped; and prefer a --corpus drawn from data the model never saw.

  # honest held-out build (fresh c4 rows, excluding training):
  G2PW_CUDA=1 python3 eval/build_reference_mspy.py --corpus heldout_corpus.txt \
      --exclude model/corpus_e3.txt --n 500 --out eval/reference_mspy.tsv
"""
import argparse, os, sys
if os.environ.get("G2PW_CUDA"):
    import onnxruntime as _ort
    _o = _ort.InferenceSession
    _ort.InferenceSession = lambda *a, **k: _o(
        *a, **{**k, "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]})
from g2pw import G2PWConverter

TONE_NUM = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}
KEYOK = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦˊˇˋ˙")


def bopo(r):
    if not r or r[-1] not in TONE_NUM:
        return None
    s = r[:-1] + TONE_NUM[r[-1]]
    return s if all(c in KEYOK for c in s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="eval/reference_mspy.tsv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min", type=int, default=6)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--exclude", default="",
                    help="training corpus; drop any sampled sentence found in it "
                         "(anti-leakage — see the module docstring)")
    args = ap.parse_args()

    # anti-leakage: never sample a sentence the model was trained on
    train = set()
    if args.exclude:
        train = set(l.strip() for l in open(args.exclude, encoding="utf-8"))
        print(f"excluding {len(train)} training sentences from the eval",
              file=sys.stderr)

    # keep existing seed lines (header + hand-written); collect their sentences to dedup
    seed, have = [], set()
    if os.path.exists(args.out):
        for ln in open(args.out, encoding="utf-8"):
            seed.append(ln.rstrip("\n"))
            if ln.strip() and not ln.startswith("#"):
                have.add(ln.split("\t")[1].strip() if "\t" in ln else "")

    # sample pure-Han everyday sentences, spread across the corpus
    lines = [l.strip() for l in open(args.corpus, encoding="utf-8")]
    cand, step = [], max(1, len(lines) // (args.n * 6))
    for i in range(0, len(lines), step):
        s = lines[i]
        if args.min <= len(s) <= args.max and all("一" <= c <= "鿿" for c in s) \
                and s not in have and s not in train:
            cand.append(s); have.add(s)
        if len(cand) >= args.n * 3:
            break

    g2p = G2PWConverter(style="bopomofo", batch_size=256)
    allr = g2p(cand)
    new = []
    for s, rd in zip(cand, allr):
        syl = [bopo(r) for r in rd]
        if all(syl) and len(syl) == len(s):
            new.append(" ".join(syl) + "\t" + s)
        if len(new) >= args.n:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(seed) + "\n")
        f.write(f"# --- +{len(new)} sampled from {os.path.basename(args.corpus)} "
                f"(g2pW gold, {args.min}-{args.max} chars) ---\n")
        f.write("\n".join(new) + "\n")
    print(f"kept {sum(1 for l in seed if l.strip() and not l.startswith('#'))} seed "
          f"+ {len(new)} sampled = larger 免選字 set -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
