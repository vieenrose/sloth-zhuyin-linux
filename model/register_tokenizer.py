#!/usr/bin/env python3
"""Register SlothLM's tokenizer hash with a llama.cpp checkout so
convert_hf_to_gguf.py accepts it.

Any model with a custom BPE vocab produces a novel pre-tokenizer hash that
llama.cpp's get_vocab_base_pre() doesn't know, and refuses to convert. Our
tokenizer is byte-level GPT-2-style, so it is safe to map its hash to the
"gpt-2" pre-tokenizer identifier (that only selects the byte-level
pre-tokenization behaviour at inference, which is exactly ours). This inserts
that mapping idempotently; the resulting GGUF embeds "gpt-2" as its
pre-tokenizer, so the serving daemon needs no patch.

Usage:
  python3 model/register_tokenizer.py --llama-cpp ~/llama.cpp
"""
import argparse
import os
import sys

# Hash emitted by convert_hf_to_gguf for model/tokenizer (build_tokenizer.py,
# byte-level BPE). Re-run the converter and read "chkhsh:" if the tokenizer
# recipe ever changes.
DEFAULT_HASH = "8d15facc5436f57461561fb75ffcaee67634fddf17bc438387f922c1407b50f2"
MARKER = "# slothlm-tokenizer"


def find_base_py(root):
    for rel in ("conversion/base.py", "convert_hf_to_gguf.py"):
        p = os.path.join(root, rel)
        if os.path.exists(p) and "def get_vocab_base_pre" in open(p, encoding="utf-8").read():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--hash", default=DEFAULT_HASH)
    args = ap.parse_args()

    path = find_base_py(args.llama_cpp)
    if not path:
        sys.exit("could not find get_vocab_base_pre() in the llama.cpp checkout")

    src = open(path, encoding="utf-8").read()
    if MARKER in src or args.hash in src:
        print(f"already registered in {path}")
        return

    anchor = "        res = None\n"
    if anchor not in src:
        sys.exit("anchor 'res = None' not found; llama.cpp layout changed")
    inject = (anchor
              + f'        if chkhsh == "{args.hash}":\n'
              + f'            res = "gpt-2"  {MARKER}\n')
    src = src.replace(anchor, inject, 1)
    open(path, "w", encoding="utf-8").write(src)
    print(f"registered {args.hash[:12]}... -> gpt-2 in {path}")


if __name__ == "__main__":
    main()
