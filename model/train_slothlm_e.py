#!/usr/bin/env python3
"""SlothLM-E: a bidirectional encoder for zhuyin decode (see model/DESIGN-E.md).

Input: one syllable token per position. Output: one Han-char token per
position, predicted non-autoregressively with full (bidirectional) attention.
Architecture: RoPE + GQA + QK-norm + RMSNorm + SwiGLU, per-position char head.

  python3 model/train_slothlm_e.py --data model/train_e.bin \
      --vocab model/syl_vocab.json --tokenizer model/tokenizer \
      --out model/slothe --epochs 2
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---- data ----
class AlignedBin(Dataset):
    """Records [N][syl*N][char*N] in a uint16 stream."""

    def __init__(self, path):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.idx = []
        i, d = 0, self.data
        while i < len(d):
            n = int(d[i])
            self.idx.append((i + 1, n))
            i += 1 + 2 * n
        print(f"{len(self.idx)} aligned pairs")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        s, n = self.idx[k]
        syl = self.data[s:s + n].astype(np.int64)
        chr_ = self.data[s + n:s + 2 * n].astype(np.int64)
        return torch.from_numpy(syl), torch.from_numpy(chr_)


def collate(batch, pad=0):
    n = max(len(s) for s, _ in batch)
    B = len(batch)
    syl = torch.zeros(B, n, dtype=torch.long)
    chr_ = torch.full((B, n), -100, dtype=torch.long)   # ignore_index
    mask = torch.zeros(B, n, dtype=torch.bool)
    for i, (s, c) in enumerate(batch):
        syl[i, :len(s)] = s
        chr_[i, :len(c)] = c
        mask[i, :len(s)] = True
    return syl, chr_, mask


# ---- model ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


def rope(x, pos, dim):
    # x: [B,H,T,Dh]; standard rotary
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    ang = pos[:, None].float() * freq[None, :]        # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class Attn(nn.Module):
    def __init__(self, dim, heads, kv):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv, dim // heads
        self.q = nn.Linear(dim, heads * self.dh, bias=False)
        self.k = nn.Linear(dim, kv * self.dh, bias=False)
        self.v = nn.Linear(dim, kv * self.dh, bias=False)
        self.o = nn.Linear(heads * self.dh, dim, bias=False)
        self.qn = RMSNorm(self.dh)                    # QK-norm
        self.kn = RMSNorm(self.dh)

    def forward(self, x, pos, amask):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = rope(q, pos, self.dh), rope(k, pos, self.dh)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        # bidirectional: only padding mask (no causal)
        attn_mask = amask[:, None, None, :]           # [B,1,1,T]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.o(o)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, kv, ffn):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attn(dim, heads, kv)
        self.n2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn)

    def forward(self, x, pos, amask):
        x = x + self.attn(self.n1(x), pos, amask)
        x = x + self.ffn(self.n2(x))
        return x


class SlothE(nn.Module):
    def __init__(self, n_syl, n_char, dim=384, depth=8, heads=6, kv=2, ffn=1024):
        super().__init__()
        self.embed = nn.Embedding(n_syl, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, kv, ffn) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, n_char, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, syl, amask):
        pos = torch.arange(syl.shape[1], device=syl.device)
        x = self.embed(syl)
        for b in self.blocks:
            x = b(x, pos, amask)
        return self.head(self.norm(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="model/syl_vocab.json")
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--out", default="model/slothe")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=0, help="cap steps (smoke test)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = AlignedBin(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    collate_fn=collate, pin_memory=True, drop_last=True)
    model = SlothE(len(syl_vocab), len(tok), args.dim, args.depth,
                   args.heads, args.kv_heads, args.ffn).to(dev)
    print(f"SlothLM-E {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    total = int(len(dl) * args.epochs) if not args.steps else args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.03)
    model.train()
    step = 0
    for ep in range(math.ceil(args.epochs)):
        for syl, chr_, mask in dl:
            syl, chr_, mask = syl.to(dev), chr_.to(dev), mask.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(syl, mask)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       chr_.reshape(-1), ignore_index=-100)
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
                "config": {"n_syl": len(syl_vocab), "n_char": len(tok),
                           "dim": args.dim, "depth": args.depth,
                           "heads": args.heads, "kv": args.kv_heads,
                           "ffn": args.ffn}},
               os.path.join(args.out, "slothe.pt"))
    json.dump(syl_vocab, open(os.path.join(args.out, "syl_vocab.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
