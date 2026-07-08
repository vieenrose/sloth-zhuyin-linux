#!/usr/bin/env python3
"""Build SlothLM's tokenizer: a byte-level BPE (vocab ~8192) whose
pre-tokenizer is byte-identical to GPT-2's, so llama.cpp's
convert_hf_to_gguf.py recognises it (pre-tokenizer hash check) without a
patch. Every bopomofo symbol and every common Traditional-Chinese character
is guaranteed an atomic token (added as special/initial tokens), so
grammar-constrained decoding aligns exactly to candidate boundaries.

Usage:
  python3 model/build_tokenizer.py --corpus corpus.txt --out model/tokenizer \
      --vocab-size 8192 --top-chars 6500
"""
import argparse
import os
from collections import Counter

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

BOPOMOFO = [chr(c) for c in range(0x3105, 0x312A)]  # ㄅ..ㄪ
TONES = ["ˊ", "ˇ", "ˋ", "˙", "ˉ"]
SPECIALS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def han_freq(corpus_path, top):
    counter = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            for ch in line:
                if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
                    counter[ch] += 1
    return [c for c, _ in counter.most_common(top)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="model/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--top-chars", type=int, default=6500)
    args = ap.parse_args()

    chars = han_freq(args.corpus, args.top_chars)
    print(f"selected {len(chars)} Han chars from corpus")

    # Tokens we require to exist atomically (added after training so they are
    # never split), on top of the byte-level base alphabet.
    forced = BOPOMOFO + TONES + chars

    tok = Tokenizer(BPE(unk_token=None))
    # GPT-2-identical pre-tokenizer + decoder -> recognised hash on convert.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([args.corpus], trainer)
    tok.add_tokens(forced)  # guarantee atomic bopomofo + common chars
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "tokenizer.json")
    tok.save(raw)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=raw,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )
    fast.save_pretrained(args.out)
    print(f"tokenizer vocab={tok.get_vocab_size()} saved to {args.out}")

    # sanity: bopomofo + a char must each be a single token
    for probe in ["ㄋㄧˇ", "你好", "iPhone"]:
        ids = fast(probe)["input_ids"]
        print(f"  {probe!r} -> {len(ids)} tokens {ids}")


if __name__ == "__main__":
    main()
