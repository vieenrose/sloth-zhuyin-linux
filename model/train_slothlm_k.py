#!/usr/bin/env python3
"""SlothLM-K: raw-keystream -> per-key labels (solution 2; see prepare_data_k).

Same bidirectional encoder as SlothLM-E (RoPE/GQA/QK-norm/SwiGLU) but the model
OWNS zh/en segmentation: input = one token per raw keystroke, output = per-key
label (Han char at syllable-final keys, ASCII for English passthrough, BLANK
elsewhere). BLANK is a *trained* class (not ignored) -- predicting where
syllables end IS the segmentation task.

  python3 model/train_slothlm_k.py --data model/train_k.bin \
      --vocab model/key_vocab.json --tokenizer model/tokenizer \
      --out model/slothk --depth 12 --epochs 3
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import AlignedBin, SlothE, collate

LABEL_BASE = 130   # tokenizer id + LABEL_BASE; 0=pad(unused) 1=BLANK 2..129=ASCII


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="model/key_vocab.json")
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--out", default="model/slothk")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    key_vocab = json.load(open(args.vocab, encoding="utf-8"))
    n_label = len(tok) + LABEL_BASE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = AlignedBin(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    collate_fn=collate, pin_memory=True, drop_last=True)
    model = SlothE(len(key_vocab), n_label, args.dim, args.depth,
                   args.heads, args.kv_heads, args.ffn).to(dev)
    print(f"SlothLM-K {sum(p.numel() for p in model.parameters())/1e6:.1f}M params "
          f"(keys {len(key_vocab)} -> labels {n_label})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    total = int(len(dl) * args.epochs) if not args.steps else args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.03)
    model.train()
    step = 0
    for _ in range(math.ceil(args.epochs)):
        for keys, labels, mask in dl:
            keys, labels, mask = keys.to(dev), labels.to(dev), mask.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(keys, mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       labels.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total} loss {loss.item():.3f}", flush=True)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"n_syl": len(key_vocab), "n_char": n_label,
                           "dim": args.dim, "depth": args.depth,
                           "heads": args.heads, "kv": args.kv_heads,
                           "ffn": args.ffn, "label_base": LABEL_BASE}},
               os.path.join(args.out, "slothe.pt"))
    json.dump(key_vocab, open(os.path.join(args.out, "key_vocab.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
