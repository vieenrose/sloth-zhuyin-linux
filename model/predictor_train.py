#!/usr/bin/env python3
"""Apple-style on-device next-word predictor: small causal GPT (dim512, 6 layers,
~33M) over a word-piece BPE vocab (words/emojis = single tokens => one forward per
prediction). Plain LM on the TW corpus. This is the OUTSIDE-prediction model that
pairs with the ternary BERT encoder (inside conversion) under the <=20ms BOOX budget.

  python3 predictor_train.py --epochs 3
"""
import argparse, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import distill_student as D   # reuse TinyGPT (causal, pattern=None)


class LMDS(Dataset):
    def __init__(self, ids, seqlen, bos):
        self.seqlen = seqlen
        # pack the token stream into fixed windows
        self.n = len(ids) // seqlen
        self.ids = np.array(ids[: self.n * seqlen], dtype=np.int32).reshape(self.n, seqlen)
        self.bos = bos
    def __len__(self): return self.n
    def __getitem__(self, i):
        x = self.ids[i].astype(np.int64)
        return torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="predictor_tok.json")
    ap.add_argument("--corpus", default="corpus_e3.txt")
    ap.add_argument("--out", default="slothe_predictor")
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=2048); ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--qat", action="store_true")
    ap.add_argument("--ssm-type", default="mamba1"); ap.add_argument("--hybrid", default="")
    args = ap.parse_args()
    D.QAT["on"] = args.qat; D.SSM_TYPE["t"] = args.ssm_type
    dev = "cuda"
    tok = Tokenizer.from_file(args.tok)
    V = tok.get_vocab_size(); BOS = tok.token_to_id("<bos>")
    # tokenize corpus into one stream, BOS between lines
    print("tokenizing corpus...", file=sys.stderr)
    stream = []
    for enc in tok.encode_batch([l.rstrip("\n") for l in open(args.corpus, encoding="utf-8")]):
        stream.append(BOS); stream.extend(enc.ids)
    print(f"vocab={V} stream={len(stream)} tokens", file=sys.stderr)
    ds = LMDS(stream, args.seqlen, BOS)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    pat = args.hybrid.split(",") if args.hybrid else None
    m = D.TinyGPT(V, args.dim, args.depth, args.heads, args.kv, args.ffn, pat).to(dev)
    print(f"predictor {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, vocab {V}", file=sys.stderr)
    total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.02)
    step = 0; m.train()
    for ep in range(math.ceil(args.epochs)):
        for x in dl:
            if step >= total: break
            x = x.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)                                    # causal
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
            if step % 200 == 0: print(f"step {step}/{total} loss {loss.item():.3f}", flush=True)
            step += 1
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": m.state_dict(),
                "config": {"vocab": V, "dim": args.dim, "depth": args.depth,
                           "heads": args.heads, "kv": args.kv, "ffn": args.ffn, "qat": args.qat,
                           "ssm_type": args.ssm_type, "pattern": (args.hybrid.split(",") if args.hybrid else None)}},
               os.path.join(args.out, "predictor.pt"))
    import shutil; shutil.copy(args.tok, os.path.join(args.out, "predictor_tok.json"))
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__": main()
