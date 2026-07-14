#!/usr/bin/env python3
"""Exp B: generative (causal-LM) baseline for the conversion task — the
architecturally-disfavoured control (autoregressive, no enforced 1:1 alignment)
to confirm the bidirectional encoder is the right family.

Reconstructs (syllables -> chars) text pairs from the augmented .bin and
causal-fine-tunes a small pretrained Chinese LM. Prompt tokens are loss-masked.

  python3 gen_convert.py --base Mxode/SmolLM-Chinese-180M \
      --data train_e_g2pw_tl.bin --vocab syl_vocab.json --our-tokenizer tokenizer \
      --out slothe_gen --epochs 1 --batch 32 --lr 2e-4 --max-pairs 800000
"""
import argparse, json, math, os, sys, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

IGNORE = 65535


def reconstruct(path, id2syl, id2char, max_pairs):
    # JSONL of {"in": bopomofo(+english), "out": text} — code-switch preserved.
    import json as _j
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        d = _j.loads(line)
        if d.get("in") and d.get("out"):
            pairs.append((d["in"], d["out"]))
            if len(pairs) >= max_pairs: break
    return pairs


class GenDS(Dataset):
    def __init__(self, pairs, tok, maxlen=96):
        self.ex = []
        for ss, cs in pairs:
            p = tok(ss + " →", add_special_tokens=False)["input_ids"]
            c = tok(cs, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = p + c
            if len(ids) > maxlen: continue
            lab = [-100] * len(p) + c[:]
            self.ex.append((ids, lab))
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]


def collate(batch, pad):
    T = max(len(i) for i, _ in batch)
    ids = np.full((len(batch), T), pad, dtype=np.int64)
    lab = np.full((len(batch), T), -100, dtype=np.int64)
    am = np.zeros((len(batch), T), dtype=np.int64)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = i; lab[k, :len(l)] = l; am[k, :len(i)] = 1
    return torch.from_numpy(ids), torch.from_numpy(am), torch.from_numpy(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Mxode/SmolLM-Chinese-180M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="syl_vocab.json")
    ap.add_argument("--our-tokenizer", default="tokenizer")
    ap.add_argument("--out", default="slothe_gen")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-pairs", type=int, default=800000)
    args = ap.parse_args()

    # DDP (torchrun --nproc_per_node=N): each rank owns one GPU.
    lrank = int(os.environ.get("LOCAL_RANK", -1))
    ddp = lrank >= 0
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        dist.init_process_group("nccl")
        torch.cuda.set_device(lrank); dev = f"cuda:{lrank}"
        rank0 = dist.get_rank() == 0
    else:
        dev = "cuda"; rank0 = True

    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    id2syl = {v: k for k, v in syl_vocab.items()}
    our = AutoTokenizer.from_pretrained(args.our_tokenizer)
    id2char = {v: k for k, v in our.get_vocab().items() if len(k) == 1}
    pairs = reconstruct(args.data, id2syl, id2char, args.max_pairs)
    random.seed(0); random.shuffle(pairs)
    if rank0: print(f"pairs: {len(pairs)} e.g. {pairs[0]}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev)
    if rank0: print(f"{args.base}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", file=sys.stderr)
    ds = GenDS(pairs, tok)
    if rank0: print(f"train examples (<=96 tok): {len(ds)}", file=sys.stderr)
    sampler = DistributedSampler(ds) if ddp else None
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=4, drop_last=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    train = DDP(model, device_ids=[lrank]) if ddp else model
    total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(train.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.03)
    step = 0; train.train()
    for ep in range(math.ceil(args.epochs)):
        if sampler: sampler.set_epoch(ep)
        for ids, am, lab in dl:
            if step >= total: break
            ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev)
            out = train(input_ids=ids, attention_mask=am, labels=lab)
            out.loss.backward()
            nn.utils.clip_grad_norm_(train.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 100 == 0 and rank0:
                print(f"step {step}/{total} loss {out.loss.item():.3f}", flush=True)
            step += 1
    if rank0:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"saved {args.out}", file=sys.stderr)
    if ddp:
        import torch.distributed as dist; dist.destroy_process_group()


if __name__ == "__main__":
    main()
