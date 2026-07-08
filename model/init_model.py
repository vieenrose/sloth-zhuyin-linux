#!/usr/bin/env python3
"""Instantiate a randomly-initialised SlothLM (LlamaForCausalLM, ~35M) with
the custom tokenizer's vocab and save it as an HF checkpoint. Used both for
the Phase-0 GGUF dry-run (random weights) and as the training starting point.

Usage:
  python3 model/init_model.py --tokenizer model/tokenizer --out model/slothlm-init
"""
import argparse

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--out", default="model/slothlm-init")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--intermediate", type=int, default=1408)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.ctx,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"SlothLM: {n/1e6:.1f}M params (vocab {len(tok)})")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
