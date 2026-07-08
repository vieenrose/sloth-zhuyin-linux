#!/usr/bin/env python3
"""Upload a trained SlothLM (HF checkpoint + GGUF) to the Hugging Face Hub.

Token comes from the HF_TOKEN env var (never hard-code it). Only run this
after the model clears the eval gates (see model/DESIGN.md).

  HF_TOKEN=hf_xxx python3 model/upload_to_hf.py \
      --repo <user>/slothlm-230m-zhuyin \
      --checkpoint model/slothlm --gguf model/slothlm-Q8_0.gguf \
      --eval eval-result.txt
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
language:
- zh
library_name: gguf
tags:
- zhuyin
- bopomofo
- input-method
- traditional-chinese
- llama.cpp
- tiny
---

# SlothLM — a tiny from-scratch LM for Zhuyin input-method conversion

SlothLM is a ~34M-parameter LlamaForCausalLM trained from scratch to do one
job well: rank/convert Traditional-Chinese homophone candidates for a Zhuyin
(Bopomofo) input method. It is the conversion model for
[Slothing](https://github.com/vieenrose/sloth-zhuyin-linux), an fcitx5 IME
that reranks libchewing candidates with grammar-constrained decoding.

Unlike general chat models, its entire pretraining distribution is the task:
plain Traditional-Chinese text plus synthetic zhuyin↔text, tone-free zhuyin,
and candidate-selection examples built from real libchewing harvests.

- **Params:** ~34M (10L, hidden 512, GQA), llama arch → llama.cpp/GGUF native
- **Tokenizer:** custom byte-level BPE, vocab ~8.3k; one token per bopomofo
  symbol and per common Han char (so GBNF candidate constraints align to
  token boundaries)
- **Context:** 1024
- **Intended use:** grammar-constrained candidate selection inside Slothing;
  not a general-purpose chat/knowledge model

{eval_block}

## Limitations

A 34M model has no world knowledge — rare named entities may be missed;
mitigated by libchewing's candidate set and per-user learning. English output
quality is poor by design (English is included only for code-switching
survival). Use via grammar-constrained decoding, not free generation.

Trained on Traditional-Chinese corpora (Wikinews/Wikipedia cc-by-sa,
Common-Crawl-derived text). See the repo for the full recipe.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", default="model/slothlm")
    ap.add_argument("--gguf")
    ap.add_argument("--eval", help="eval-result.txt to embed in the card")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN in the environment (do not hard-code it)")

    eval_block = ""
    if args.eval and os.path.exists(args.eval):
        eval_block = "## Evaluation\n\n```\n" + \
            open(args.eval, encoding="utf-8").read().strip() + "\n```"

    api = HfApi(token=token)
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    with open(os.path.join(args.checkpoint, "README.md"), "w",
              encoding="utf-8") as f:
        f.write(CARD.format(eval_block=eval_block))
    api.upload_folder(folder_path=args.checkpoint, repo_id=args.repo)
    if args.gguf and os.path.exists(args.gguf):
        api.upload_file(path_or_fileobj=args.gguf,
                        path_in_repo=os.path.basename(args.gguf),
                        repo_id=args.repo)
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
