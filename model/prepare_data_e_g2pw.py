#!/usr/bin/env python3
"""SlothLM-E data prep using g2pW context-aware readings (better DATA, lever #1).

The residual ~16% errors were reading/heteronym mistakes from pypinyin's
rule-based g2p (重新->ㄓㄨㄥˋ, 息->ㄒㄧ). g2pW is a neural TW polyphone
disambiguator that picks the reading FROM SENTENCE CONTEXT and gives 重 two
readings by context (重新 ㄔㄨㄥˊ vs 重要 ㄓㄨㄥˋ). It returns per-char readings
(None for non-Han), so code-switch aligns for free. Output records identical to
prepare_data_e (aligned syllable-id / char-id, tonal + toneless, English=<en>).

  python3 model/prepare_data_e_g2pw.py --corpus model/corpus_e3.txt \
      --tokenizer model/tokenizer --table model/phonetic_table.tsv \
      --out model/train_e_g2pw.bin --vocab model/syl_vocab.json
"""
import argparse, json, os, sys
import numpy as np
# GPU: g2pw's onnx session defaults to CPU-only; force the CUDA provider when
# G2PW_CUDA=1 (also set LD_LIBRARY_PATH to torch's nvidia libs). ~15x faster
# (batched BERT on GPU ~3000 sent/s vs ~200 on CPU).
if os.environ.get("G2PW_CUDA"):
    import onnxruntime as _ort
    _orig_sess = _ort.InferenceSession
    _ort.InferenceSession = lambda *a, **k: _orig_sess(
        *a, **{**k, "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]})
from transformers import AutoTokenizer
from g2pw import G2PWConverter

TONES = "ˊˇˋ˙"
TONE_NUM = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}
KEYMAP_OK = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + TONES)
PAD, UNK, EN = 0, 1, 2
IGNORE = 65535


def to_bopomofo(reading):
    """g2pW 'ㄔㄨㄥ2' -> 'ㄔㄨㄥˊ'."""
    if not reading:
        return None
    base, tone = reading[:-1], reading[-1]
    if tone not in TONE_NUM:
        return None
    syl = base + TONE_NUM[tone]
    return syl if all(c in KEYMAP_OK for c in syl) else None


def build_syl_vocab(table_path):
    v = {"<pad>": PAD, "<unk>": UNK, "<en>": EN}
    tonal = set()
    for line in open(table_path, encoding="utf-8"):
        s = line.split("\t")[0]
        if s:
            tonal.add(s)
    for s in sorted(tonal):
        v.setdefault(s, len(v))
    for s in sorted(tonal):
        v.setdefault("".join(c for c in s if c not in TONES), len(v))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    ap.add_argument("--out", default="model/train_e_g2pw.bin")
    ap.add_argument("--vocab", default="model/syl_vocab.json")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-sentences", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    syl_vocab = build_syl_vocab(args.table)
    json.dump(syl_vocab, open(args.vocab, "w", encoding="utf-8"), ensure_ascii=False)
    g2p = G2PWConverter(style="bopomofo", batch_size=args.batch)

    def char_id(c):
        i = tok.convert_tokens_to_ids(c)
        return i if i is not None else tok.unk_token_id

    def records_for(sent, readings):
        """Build tonal+toneless aligned records for one sentence; readings is
        the per-char g2pW list (None for non-Han). Returns list of (syl_ids,
        char_ids)."""
        out = []
        for toneless in (False, True):
            syl_ids, char_ids, ok = [], [], True
            i = 0
            while i < len(sent):
                ch = sent[i]
                if ("一" <= ch <= "鿿") and readings[i]:
                    syl = to_bopomofo(readings[i])
                    if syl is None:
                        ok = False; break
                    if toneless:
                        syl = "".join(c for c in syl if c not in TONES)
                    syl_ids.append(syl_vocab.get(syl, UNK)); char_ids.append(char_id(ch))
                    i += 1
                elif ("a" <= ch.lower() <= "z") or ch.isdigit():
                    # English/number run -> one <en> position (passthrough)
                    syl_ids.append(EN); char_ids.append(IGNORE)
                    while i < len(sent) and (sent[i].isascii() and (sent[i].isalnum())):
                        i += 1
                else:
                    i += 1                       # skip spaces/punctuation
            if ok and syl_ids and any(c != IGNORE for c in char_ids):
                out.append((syl_ids, char_ids))
        return out

    rec = []
    stats = {"records": 0}
    n = 0
    buf = []

    def flush(buf):
        nonlocal rec
        sents = [b for b in buf if b]
        if not sents:
            return
        allr = g2p(sents)
        for s, readings in zip(sents, allr):
            for syl_ids, char_ids in records_for(s, readings):
                rec.append(len(syl_ids)); rec.extend(syl_ids); rec.extend(char_ids)
                stats["records"] += 1

    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (1 <= len(s) <= 40):
            continue
        n += 1
        if args.max_sentences and n > args.max_sentences:
            break
        buf.append(s)
        if len(buf) >= args.batch:
            flush(buf); buf = []
            if n % 51200 == 0:
                print(f"  {n} sentences -> {stats['records']} records", file=sys.stderr)
    flush(buf)

    arr = np.array(rec, dtype=np.uint16)
    arr.tofile(args.out)
    print(f"wrote {stats['records']} aligned records ({len(arr)} uint16) to "
          f"{args.out}; g2pW context-aware readings")


if __name__ == "__main__":
    main()
