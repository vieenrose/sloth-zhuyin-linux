#!/usr/bin/env python3
"""Stage-1 Chinese char masked-LM pretraining of the SlothE encoder (Exp C).

Injects language-model knowledge (homophone/tone disambiguation lives in the
LM prior) into the exact same transformer stack the conversion model uses, so
Stage-2 (train_slothlm_e.py --init) can transfer blocks + norm + head and only
learn a fresh syllable input embedding. Reuses Block/RMSNorm from
train_slothlm_e so state_dict keys line up 1:1.

  python3 mlm_pretrain.py --corpus corpus_e3.txt --tokenizer tokenizer \
      --out slothe_mlm --dim 768 --depth 20 --heads 12 --kv-heads 2 --ffn 2048 \
      --embed-norm --epochs 3 --batch 256 --lr 8e-4
"""
import argparse, json, math, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from train_slothlm_e import Block, RMSNorm  # exact same modules → transferable


class MLMEncoder(nn.Module):
    """char-in / char-out; embed_norm/blocks/norm/head named exactly as SlothE."""
    def __init__(self, n_char, dim, depth, heads, kv, ffn, embed_norm=True):
        super().__init__()
        self.embed = nn.Embedding(n_char + 1, dim)        # last id = [MASK]
        self.embed_norm = RMSNorm(dim) if embed_norm else None
        self.blocks = nn.ModuleList([Block(dim, heads, kv, ffn) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, n_char, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, amask):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids)
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        for b in self.blocks:
            x = b(x, pos, amask)
        return self.head(self.norm(x))


class CharDS(Dataset):
    def __init__(self, seqs, maxlen):
        self.seqs = seqs; self.maxlen = maxlen
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]


def collate(batch, mask_id, n_char, pad=0, mlm_p=0.15):
    T = max(len(s) for s in batch)
    ids = np.full((len(batch), T), pad, dtype=np.int64)
    lab = np.full((len(batch), T), -100, dtype=np.int64)
    am = np.zeros((len(batch), T), dtype=bool)
    for i, s in enumerate(batch):
        L = len(s); ids[i, :L] = s; am[i, :L] = True
        m = np.random.rand(L) < mlm_p
        for j in np.where(m)[0]:
            lab[i, j] = s[j]
            r = np.random.rand()
            if r < 0.8: ids[i, j] = mask_id
            elif r < 0.9: ids[i, j] = np.random.randint(0, n_char)
            # else keep
    return (torch.from_numpy(ids), torch.from_numpy(am), torch.from_numpy(lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="slothe_mlm")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--embed-norm", action="store_true")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--maxlen", type=int, default=40)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = tok.get_vocab()                     # token(str) -> id
    n_char = tok.vocab_size
    mask_id = n_char                            # extra input row
    unk = tok.unk_token_id or 1

    # tokenize corpus to char-id sequences (Han chars only; skip others)
    seqs = []
    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not (2 <= len(s) <= args.maxlen):
            continue
        ids = [vocab.get(c, unk) for c in s if ("一" <= c <= "鿿")]
        if len(ids) >= 2:
            seqs.append(np.array(ids, dtype=np.int64))
    print(f"MLM corpus: {len(seqs)} char sequences, n_char={n_char}", file=sys.stderr)

    dev = "cuda"
    model = MLMEncoder(n_char, args.dim, args.depth, args.heads, args.kv_heads,
                       args.ffn, args.embed_norm).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"MLM encoder {nparam/1e6:.1f}M params", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    dl = DataLoader(CharDS(seqs, args.maxlen), batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True,
                    collate_fn=lambda b: collate(b, mask_id, n_char))
    total = int(len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.03)
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    model.train()
    for ep in range(math.ceil(args.epochs)):
        for ids, am, lab in dl:
            if step >= total: break
            ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, am)
                loss = F.cross_entropy(logits.view(-1, n_char), lab.view(-1),
                                       ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.3f}", flush=True)
            step += 1
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"dim": args.dim, "depth": args.depth, "heads": args.heads,
                           "kv": args.kv_heads, "ffn": args.ffn, "n_char": n_char}},
               os.path.join(args.out, "mlm.pt"))
    print(f"saved {args.out}/mlm.pt", file=sys.stderr)


if __name__ == "__main__":
    main()
