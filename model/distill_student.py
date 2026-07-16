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
# Q4 QAT: per-output-channel symmetric int4 fake-quant with straight-through
# estimator, so the model trains robust to 4-bit deployment (~20MB for 79M).
QAT = {"on": False, "mode": "q4"}
def q4(w):
    if QAT["on"] and QAT.get("mode") == "ternary":
        s = w.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        wq = (w / s).round().clamp_(-1, 1) * s
        return w + (wq - w).detach()
    if not QAT["on"]:
        return w
    s = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5) / 7.0   # int4: [-7,7]
    wq = (w / s).round().clamp_(-7, 7) * s
    return w + (wq - w).detach()                                    # STE


class QLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, q4(self.weight), self.bias)


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
        self.q = QLinear(dim, heads * self.dh, bias=False)
        self.k = QLinear(dim, kv * self.dh, bias=False)
        self.v = QLinear(dim, kv * self.dh, bias=False)
        self.o = QLinear(heads * self.dh, dim, bias=False)
        self.qn = RMSNorm(self.dh); self.kn = RMSNorm(self.dh)
    def forward(self, x, pos, attn_mask=None):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.h, self.dh).transpose(1, 2))
        k = self.kn(self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2))
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = rope(q, pos, self.dh), rope(k, pos, self.dh)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # CAUSAL
        else:   # PREFIX-LM: bidirectional over the input span, causal after
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


# Pure-PyTorch Mamba (selective SSM) block — no custom CUDA kernels. Sequential
# scan is cheap for our short sequences (<=34). Faithful Mamba-1 core.
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.di = expand * dim; self.ds = d_state
        self.dt_rank = dt_rank or max(16, dim // 16)
        self.in_proj = QLinear(dim, 2 * self.di, bias=False)
        self.conv1d = nn.Conv1d(self.di, self.di, d_conv, groups=self.di,
                                padding=d_conv - 1, bias=True)
        self.x_proj = QLinear(self.di, self.dt_rank + 2 * self.ds, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.di, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.ds + 1).float()
                                             .repeat(self.di, 1)))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = QLinear(self.di, dim, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xi = F.silu(xi)
        dbl = self.x_proj(xi)
        dt, Bm, Cm = dbl.split([self.dt_rank, self.ds, self.ds], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B,T,di)
        A = -torch.exp(self.A_log.float())                # (di,ds)
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (B,T,di,ds)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)  # (B,T,di,ds)
        h = torch.zeros(B, self.di, self.ds, device=x.device, dtype=dA.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + self.D * xi
        return self.out_proj(y * F.silu(z))


# SSM factory: mamba1 (pure-torch, no kernels), or fla's mamba2 / gdn (Gated
# DeltaNet — Triton, sm_120-ok, recall-fixing). fla layers return a tuple; wrap.
SSM_TYPE = {"t": "mamba1"}
class FLAWrap(nn.Module):
    def __init__(self, layer): super().__init__(); self.layer = layer
    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out
def make_ssm(dim):
    t = SSM_TYPE["t"]
    if t == "mamba1":
        return MambaBlock(dim)
    if t == "mamba2":
        from fla.layers import Mamba2
        # constraint: num_heads*head_dim == expand*hidden_size (=2*dim)
        return FLAWrap(Mamba2(hidden_size=dim, num_heads=(2 * dim) // 64, head_dim=64,
                              state_size=128, expand=2, conv_kernel=4))
    if t == "gdn":
        from fla.layers import GatedDeltaNet
        # expand_v=1 => head_v_dim == head_k_dim (square heads) so the weights map
        # onto llama.cpp's qwen3next DeltaNet (which requires head_k==head_v==d_state).
        return FLAWrap(GatedDeltaNet(hidden_size=dim, head_dim=128, expand_v=1.0,
                                     num_heads=max(1, dim // 128), mode="chunk"))
    raise ValueError(t)


class Block(nn.Module):
    # kind: "attn" (transformer), "ssm" (SSM per SSM_TYPE), or "par" (PARALLEL
    # Hymba-style: attention + SSM same input, per-branch RMSNorm + learnable-beta).
    def __init__(self, dim, heads, kv, ffn, kind="attn"):
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(dim)
        if kind == "attn":
            self.mix = Attn(dim, heads, kv)
        elif kind == "ssm":
            self.mix = make_ssm(dim)
        else:  # par
            self.attn = Attn(dim, heads, kv); self.ssm = make_ssm(dim)
            self.na = RMSNorm(dim); self.nm = RMSNorm(dim)
            self.beta = nn.Parameter(torch.ones(2))
        self.n2 = RMSNorm(dim)
        self.w1 = QLinear(dim, ffn, bias=False); self.w3 = QLinear(dim, ffn, bias=False)
        self.w2 = QLinear(ffn, dim, bias=False)
    def forward(self, x, pos, attn_mask=None):
        h1 = self.n1(x)
        if self.kind == "attn":
            mixed = self.mix(h1, pos, attn_mask)
        elif self.kind == "ssm":
            mixed = self.mix(h1)
        else:  # parallel: normalize each branch, learnable-weighted sum
            b = torch.softmax(self.beta, 0)
            a = self.na(self.attn(h1, pos, attn_mask)).to(h1.dtype)  # cast branches
            s = self.nm(self.ssm(h1)).to(h1.dtype)         # common dtype (fla vs Attn)
            mixed = b[0] * a + b[1] * s
        x = x + mixed
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyGPT(nn.Module):
    # hybrid via `pattern` (repeating cycle of "attn"/"ssm"/"par"). Default all-attn.
    # Research: parallel(intra) > sequential(inter); ~1:5 attn:ssm; never attn-first;
    # distribute evenly / middle. Untied embed/head; depth>width for hybrids.
    def __init__(self, vocab, dim, depth, heads, kv, ffn, pattern=None, untie=False):
        super().__init__()
        if pattern is None: pattern = ["attn"] * depth
        pattern = [pattern[i % len(pattern)] for i in range(depth)]
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, kv, ffn, k) for k in pattern])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if not untie:
            self.head.weight = self.embed.weight        # tied (fp; embeds high-prec in Q4)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)
    def forward(self, ids, prefix_len=None):
        # prefix_len: (B,) length of the INPUT span per row. Positions < prefix_len
        # attend BIDIRECTIONALLY (encoder-quality syllable reps); positions >=
        # prefix_len are CAUSAL (generation). None => fully causal (decoder-only).
        pos = torch.arange(ids.shape[1], device=ids.device)
        mask = None
        if prefix_len is not None:
            T = ids.shape[1]
            causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
            j = torch.arange(T, device=ids.device)
            # key j is in the prefix -> visible to every query
            inpref = (j[None, :] < prefix_len[:, None])                  # (B,T)
            mask = causal[None, None] | inpref[:, None, None, :]         # (B,1,T,T)
        x = self.embed(ids)
        for b in self.blocks: x = b(x, pos, mask)
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
            # allow continuation pairs: len(gold) >= len(sy) (prefix syllables ->
            # full sentence). KD (1:1 alignment) only used when --kd/teacher on.
            if not sy or len(gold) < len(sy):
                continue
            if all(s in vocab for s in sy) and all(c in vocab for c in gold) \
                    and len(sy) <= maxlen and len(gold) <= maxlen + 8:
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
    ap.add_argument("--prefix-lm", action="store_true", help="bidirectional over input span (encoder-quality reps), causal after")
    ap.add_argument("--ssm-type", default="mamba1", choices=["mamba1","mamba2","gdn"])
    ap.add_argument("--untie", action="store_true", help="untie embed/head (research: untied even at 0.5B)")
    ap.add_argument("--hybrid", default="", help="repeating block pattern e.g. ssm,ssm,attn (empty=all attn)")
    ap.add_argument("--qat", action="store_true", help="4-bit QAT (deploy-ready Q4)")
    ap.add_argument("--no-teacher", action="store_true", help="pure CE, skip 7B teacher (fast size-curve test)")
    ap.add_argument("--maxlen", type=int, default=24); ap.add_argument("--max-pairs", type=int, default=1500000)
    args = ap.parse_args()
    QAT["on"] = args.qat
    SSM_TYPE["t"] = args.ssm_type

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

    # teacher (frozen). --no-teacher / --kd 0 => pure-CE fast path (no 7B forward)
    # for size-curve tests (isolates architecture vs size).
    use_teacher = (not args.no_teacher) and args.kd > 0
    teacher = None; ttok = None
    def tid(c): return None
    if use_teacher:
        ttok = AutoTokenizer.from_pretrained(args.teacher)
        if os.path.exists(os.path.join(args.teacher, "adapter_config.json")):
            from peft import PeftModel
            base_name = json.load(open(os.path.join(args.teacher, "adapter_config.json")))["base_model_name_or_path"]
            base_m = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
            teacher = PeftModel.from_pretrained(base_m, args.teacher).merge_and_unload().to(dev).eval()
        else:
            teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        # context-aware char -> teacher-token-id (SentencePiece prefixes ▁ on
        # isolated chars, so tokenize IN CONTEXT and take the last token). Cached.
        _tid = {}
        _prevlen = len(ttok("的", add_special_tokens=False)["input_ids"])
        def tid(c):
            if c not in _tid:
                ids = ttok("的" + c, add_special_tokens=False)["input_ids"]
                _tid[c] = ids[-1] if len(ids) == _prevlen + 1 else None
            return _tid[c]

    pattern = args.hybrid.split(",") if args.hybrid else None
    student = TinyGPT(len(vocab), args.dim, args.depth, args.heads, args.kv, args.ffn, pattern, args.untie).to(dev)
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
            tfulls, tmaps = [], []   # teacher full-seq ids, per-char (pos, legal_tids, legal_sidx)
            for sy, gold in batch:
                sids = [vocab[s] for s in sy]; cids = [vocab[c] for c in gold]
                seq = [BOS] + sids + [SEP] + cids + [EOS]
                lab = [-100] * (len(sids) + 2) + cids + [EOS]   # predict char region + EOS
                seqs.append(seq); labels.append(lab)
                if not use_teacher:
                    continue
                # teacher: "sy → gold" tokenized consistently; per-char legal target
                # via incremental prefix tokenization (position predicting char i).
                prompt_str = " ".join(sy) + " →"
                full = ttok(prompt_str + "".join(gold), add_special_tokens=False)["input_ids"]
                prev_len = len(ttok(prompt_str, add_special_tokens=False)["input_ids"])
                acc = prompt_str; per = []
                for i, c in enumerate(gold):
                    acc += c
                    nlen = len(ttok(acc, add_special_tokens=False)["input_ids"])
                    # KD only on the INSIDE span (i < len(sy), aligned to a
                    # syllable's legal set); outside chars get CE only.
                    if i < len(sy) and nlen == prev_len + 1:
                        lc = [x for x in legal(sy[i]) if x in vocab and tid(x) is not None]
                        if lc:
                            per.append((i, prev_len - 1, [tid(x) for x in lc], [vocab[x] for x in lc]))
                    prev_len = nlen
                tfulls.append(full); tmaps.append(per)

            T = max(len(s) for s in seqs)
            sarr = np.full((len(seqs), T), PAD, np.int64); larr = np.full((len(seqs), T), -100, np.int64)
            for i, (s, l) in enumerate(zip(seqs, labels)):
                sarr[i, :len(s)] = s; larr[i, :len(l)] = l
            sids_t = torch.from_numpy(sarr).to(dev); lab_t = torch.from_numpy(larr).to(dev)
            # prefix span = [BOS syls SEP] -> len(sy)+2 per row
            plen = torch.tensor([len(sy) + 2 for sy, _ in batch], device=dev) \
                   if args.prefix_lm else None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = train(sids_t, plen)
                ce = F.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)),
                                     lab_t[:, 1:].reshape(-1), ignore_index=-100)
            # ---- teacher forward (padded) + char-level KD ----
            kd = torch.zeros((), device=dev); ncmp = 0
            if use_teacher:
                TT = max(len(f) for f in tfulls)
                tarr = np.full((len(tfulls), TT), ttok.pad_token_id or 0, np.int64)
                tam = np.zeros((len(tfulls), TT), np.int64)
                for i, f in enumerate(tfulls):
                    tarr[i, :len(f)] = f; tam[i, :len(f)] = 1
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    tlog = teacher(input_ids=torch.from_numpy(tarr).to(dev),
                                   attention_mask=torch.from_numpy(tam).to(dev)).logits
                slog = logits.float()   # student char-position logits, fp for KD stability
                for i, (sy, gold) in enumerate(batch):
                    base = len(sy) + 2 - 1             # student pos predicting char_0
                    for (ci, tpos, ltids, lsidx) in tmaps[i]:
                        tdist = F.log_softmax(tlog[i, tpos].float()[ltids], -1).exp()
                        spos = base + ci               # student pos predicting char ci
                        sdist = F.log_softmax(slog[i, spos][lsidx], -1)
                        kd = kd + F.kl_div(sdist, tdist, reduction="sum"); ncmp += 1
                kd = kd / max(ncmp, 1)
            loss = ce + (args.kd * kd if use_teacher else 0.0)
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
                               "heads": args.heads, "kv": args.kv, "ffn": args.ffn, "qat": args.qat, "untie": args.untie, "prefix_lm": args.prefix_lm, "ssm_type": args.ssm_type, "pattern": (args.hybrid.split(",") if args.hybrid else None)}},
                   os.path.join(args.out, "student.pt"))
        import shutil; shutil.copy(args.vocab, os.path.join(args.out, "student_vocab.json"))
        print(f"saved {args.out}", file=sys.stderr)
    if ddp:
        import torch.distributed as dist; dist.destroy_process_group()


if __name__ == "__main__":
    main()
