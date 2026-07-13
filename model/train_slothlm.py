#!/usr/bin/env python3
"""Pretrain SlothLM from scratch on the uint16 token memmap.

Full pretraining (not LoRA) over model/train.bin, packing the stream into
fixed-length blocks. Saves an HF LlamaForCausalLM checkpoint that
convert_hf_to_gguf.py + model/register_tokenizer.py turn into a slothd
GGUF.

  python3 model/train_slothlm.py --data model/train.bin \
      --tokenizer model/tokenizer --out model/slothlm --block 1024
"""
import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                          Trainer, TrainingArguments)


class PackedBin(Dataset):
    """Contiguous blocks over the token stream; labels = inputs (causal LM,
    Trainer shifts internally)."""

    def __init__(self, path, block):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block = block
        self.n = (len(self.data) - 1) // block

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.block
        x = torch.from_numpy(
            self.data[s:s + self.block].astype(np.int64))
        return {"input_ids": x, "labels": x.clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--out", default="model/slothlm")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--intermediate", type=int, default=1408)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=6)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = PackedBin(args.data, args.block)
    print(f"{len(ds)} blocks x {args.block} = {len(ds)*args.block/1e6:.1f}M "
          f"train tokens; vocab {len(tok)}")

    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=args.hidden,
        intermediate_size=args.intermediate, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.block, rope_theta=10000.0,
        tie_word_embeddings=True, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(cfg)
    print(f"SlothLM {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.02,
        weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.0,
        bf16=True, tf32=True, logging_steps=50, save_strategy="epoch",
        # torch.compile fuses the tiny model's kernels + uses CUDA graphs; with
        # fixed-length blocks the shapes are static so it compiles cleanly and
        # cuts step time on a small (kernel-launch-bound) model.
        torch_compile=True,
        dataloader_num_workers=8, dataloader_pin_memory=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    try:
        ppl = math.exp(trainer.state.log_history[-1].get("train_loss", 0))
        print(f"final train ppl ~ {ppl:.1f}")
    except Exception:
        pass
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
