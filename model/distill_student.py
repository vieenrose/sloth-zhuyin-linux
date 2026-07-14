#!/usr/bin/env python3
"""Distill Breeze-7B (teacher) into a tiny (~30M) CUSTOM causal decoder with a
100%-coverage vocab (student_vocab.json: syllable input tokens + Han char output
tokens). The student learns hard labels (gold conversions) PLUS char-level logit
KD from the teacher's soft distribution over each syllable's legal chars — that
soft distribution is the homophone/tone knowledge a from-scratch model can't get.

  torchrun --nproc_per_node=2 distill_student.py --teacher slothe_breeze_teacher \
      --data gen_pairs_full.jsonl --vocab student_vocab.json \
      --dim 512 --depth 8 --heads 8 --kv 2 --ffn 1536 --kd 1.0 --epochs 2

Single-GPU also works (drop torchrun). Teacher is frozen (eval, no grad).
"""
import argparse, json, math, os, random, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

TONES = "ˊˇˋ˙"
BOPO = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + TONES)


# ======================= custom tiny causal decoder =======================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, pos, dim):
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    ang = pos[:, None].float() * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Attn(nn.Module):
    def __init__(self, dim, heads, kv):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv, dim // heads
        self.q = nn.Linear(dim, heads * self.dh, bias=False)
        self.k = nn.Linear(dim, kv * self.dh, bias=False)
        self.v = nn.Linear(dim, kv * self.dh, bias=False)
        self.o = nn.Linear(heads * self.dh, dim, bias=False)
        self.qn = RMSNorm(self.dh); self.kn = RMSNorm(self.dh)
    def forward(self, x, pos):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.dh).transpose(1, 2))
        k = self.kn(self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2))
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = rope(q, pos, self.dh), rope(k, pos, self.dh)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # CAUSAL
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, dim, heads, kv, ffn):
        super().__init__()
        self.n1 = RMSNorm(dim); self.attn = Attn(dim, heads, kv)
        self.n2 = RMSNorm(dim)
        self.w1 = nn.Linear(dim, ffn, bias=False); self.w3 = nn.Linear(dim, ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)
    def forward(self, x, pos):
        x = x + self.attn(self.n1(x), pos)
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyGPT(nn.Module):
    def __init__(self, vocab, dim, depth, heads, kv, ffn):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, kv, ffn) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight            # tied
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)
    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids)
        for b in self.blocks: x = b(x, pos)
        return self.head(self.norm(x))


# ======================= data =======================
class DS(Dataset):
    """Each item: (syl_toks[list], gold_chars[str]) — the student builds its own
    sequence; the teacher builds its own. Kept as strings for cross-tokenizer KD."""
    def __init__(self, path, vocab, maxlen, max_pairs):
        self.ex = []
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            sy = [s for s in d["in"].split() if s and s[0] in BOPO]
            gold = [c for c in d["out"] if "一" <= c <= "鿿"]
            if not sy or len(sy) != len(gold):        # keep clean 1:1 all-zh pairs
                continue
            if all(s in vocab for s in sy) and all(c in vocab for c in gold) and len(sy) <= maxlen:
                self.ex.append((sy, gold))
            if len(self.ex) >= max_pairs: break
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", default="gen_pairs_full.jsonl")
    ap.add_argument("--vocab", default="student_vocab.json")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--out", default="slothe_student30m")
    ap.add_argument("--dim", type=int, default=512); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--maxlen", type=int, default=24); ap.add_argument("--max-pairs", type=int, default=1500000)
    args = ap.parse_args()

    lrank = int(os.environ.get("LOCAL_RANK", -1)); ddp = lrank >= 0
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        dist.init_process_group("nccl"); torch.cuda.set_device(lrank)
        dev = f"cuda:{lrank}"; rank0 = dist.get_rank() == 0
    else:
        dev = "cuda"; rank0 = True

    vocab = json.load(open(args.vocab, encoding="utf-8"))
    PAD, BOS, SEP, EOS = vocab["<pad>"], vocab["<bos>"], vocab["<sep>"], vocab["<eos>"]
    # legal char ids (student vocab) per syllable, tone-union for unmarked
    tonal = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, r = line.rstrip("\n").partition("\t")
        if r: tonal[s] = set(r)
    def legal_chars(syl):
        has = any(c in TONES for c in syl)
        ch = tonal.get(syl) if has else None
        if ch is None:
            base = "".join(x for x in syl if x not in TONES); ch = set()
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base: ch |= v
        return [c for c in (ch or {syl}) if c in vocab]
    legal_cache = {}
    def legal(syl):
        if syl not in legal_cache: legal_cache[syl] = legal_chars(syl)
        return legal_cache[syl]

    # teacher (frozen)
    ttok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    tvocab = ttok.get_vocab()
    def tid(c):                                  # teacher single-token id for a char (or None)
        ids = ttok(c, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    student = TinyGPT(len(vocab), args.dim, args.depth, args.heads, args.kv, args.ffn).to(dev)
    if rank0: print(f"student {sum(p.numel() for p in student.parameters())/1e6:.1f}M params, vocab {len(vocab)}", file=sys.stderr)
    train = DDP(student, device_ids=[lrank]) if ddp else student

    ds = DS(args.data, vocab, args.maxlen, args.max_pairs)
    if rank0: print(f"train pairs: {len(ds)}", file=sys.stderr)
    sampler = DistributedSampler(ds) if ddp else None

    def collate(b):
        return b   # keep as list; we build per-example sequences (variable teacher align)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=2, drop_last=True, collate_fn=collate)
    total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(train.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.03)
    scaler = torch.amp.GradScaler("cuda")

    step = 0; train.train()
    for ep in range(math.ceil(args.epochs)):
        if sampler: sampler.set_epoch(ep)
        for batch in dl:
            if step >= total: break
            # ---- build padded STUDENT sequences: [BOS syls SEP chars EOS] ----
            seqs, labels = [], []
            tprompts, tfulls, tmaps = [], [], []   # teacher prompt ids, full ids, per-char (pos,legal_tids,legal_sidx)
            for sy, gold in batch:
                sids = [vocab[s] for s in sy]; cids = [vocab[c] for c in gold]
                seq = [BOS] + sids + [SEP] + cids + [EOS]
                lab = [-100] * (len(sids) + 2) + cids + [EOS]   # predict char region + EOS
                seqs.append(seq); labels.append(lab)
                # teacher: "sy → gold", per-char legal target
                tp = ttok(" ".join(sy) + " →", add_special_tokens=False)["input_ids"]
                per = []; cur = list(tp)
                ok = True
                for i, c in enumerate(gold):
                    ci = ttok(c, add_special_tokens=False)["input_ids"]
                    if len(ci) == 1:
                        lc = legal(sy[i]); ltids = [tid(x) for x in lc]
                        keep = [(x, t) for x, t in zip(lc, ltids) if t is not None and x in vocab]
                        if keep:
                            per.append((len(cur) - 1, [t for _, t in keep], [vocab[x] for x, _ in keep]))
                    cur += ci
                tprompts.append(tp); tfulls.append(cur); tmaps.append(per)

            T = max(len(s) for s in seqs)
            sarr = np.full((len(seqs), T), PAD, np.int64); larr = np.full((len(seqs), T), -100, np.int64)
            for i, (s, l) in enumerate(zip(seqs, labels)):
                sarr[i, :len(s)] = s; larr[i, :len(l)] = l
            sids_t = torch.from_numpy(sarr).to(dev); lab_t = torch.from_numpy(larr).to(dev)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = train(sids_t)
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)),
                                     lab_t[:, 1:].reshape(-1), ignore_index=-100)
            # ---- teacher forward (padded) + char-level KD ----
            kd = torch.zeros((), device=dev); ncmp = 0
            TT = max(len(f) for f in tfulls)
            tarr = np.full((len(tfulls), TT), ttok.pad_token_id or 0, np.int64)
            tam = np.zeros((len(tfulls), TT), np.int64)
            for i, f in enumerate(tfulls):
                tarr[i, :len(f)] = f; tam[i, :len(f)] = 1
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tlog = teacher(input_ids=torch.from_numpy(tarr).to(dev),
                               attention_mask=torch.from_numpy(tam).to(dev)).logits
            # student char-position logits: recompute in fp for KD stability
            slog = logits.float()
            for i, (sy, gold) in enumerate(batch):
                base = len(sy) + 2 - 1                 # student pos predicting char_0 = BOS+syls+SEP index-1
                for j, (tpos, ltids, lsidx) in enumerate(tmaps[i]):
                    tdist = F.log_softmax(tlog[i, tpos].float()[ltids], -1).exp()
                    spos = base + j                     # student pos predicting char_j
                    sdist = F.log_softmax(slog[i, spos][lsidx], -1)
                    kd = kd + F.kl_div(sdist, tdist, reduction="sum"); ncmp += 1
            kd = kd / max(ncmp, 1)
            loss = ce + args.kd * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(train.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 100 == 0 and rank0:
                print(f"step {step}/{total} loss {loss.item():.3f} (ce {ce.item():.3f} kd {kd.item():.3f})", flush=True)
            step += 1

    if rank0:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": student.state_dict(),
                    "config": {"vocab": len(vocab), "dim": args.dim, "depth": args.depth,
                               "heads": args.heads, "kv": args.kv, "ffn": args.ffn}},
                   os.path.join(args.out, "student.pt"))
        import shutil; shutil.copy(args.vocab, os.path.join(args.out, "student_vocab.json"))
        print(f"saved {args.out}", file=sys.stderr)
    if ddp:
        import torch.distributed as dist; dist.destroy_process_group()


if __name__ == "__main__":
    main()
