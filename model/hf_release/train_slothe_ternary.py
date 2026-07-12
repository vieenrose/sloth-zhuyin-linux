#!/usr/bin/env python3
"""
Ternary (1.58-bit) SlothLM-E trainer — a drop-in replacement for
`reproduce/train_slothlm_e.py` that adds:

  * BitLinear (ternary {-1,0,+1} weights, absMEDIAN scale per output channel,
    int8 activations, straight-through estimator) in place of nn.Linear inside
    every transformer block.
  * Mixed-precision islands: the syllable embedding, the char head, and the
    first/last `--fp-boundary` blocks stay full precision (int8 at deploy).
  * Progressive fp->ternary annealing (`--anneal-frac`).
  * "Extra RMSNorm before every linear" (SubLN) for ternary stability.
  * Knowledge distillation from a full-precision SlothLM-E teacher (`--teacher`).

Module names MIRROR reproduce/train_slothlm_e.py exactly, so:
  - a full-precision teacher checkpoint saved by the original script loads into
    an all-fp SlothE_T for distillation, and
  - the saved ternary checkpoint carries a `config` dict that the companion
    gate/export scripts use to rebuild the model.

Design rationale + citations: see docs/GUIDE_TERNARY_SLOTHLM_E.md in this repo.

Example (on the RTX 5090):
    python3 train_slothe_ternary.py \
        --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer \
        --out slothe_t_20m \
        --dim 320 --depth 16 --heads 8 --kv-heads 2 --ffn 880 --embed-norm \
        --teacher slothe_32m --distill-alpha 0.7 --distill-temp 2.0 \
        --anneal-frac 0.15 --weight-quant median --pre-norm \
        --batch 384 --epochs 8 --lr 2.5e-3
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ======================================================================
# Quantization primitives (ternary weights + int8 activations, STE)
# ======================================================================

def activation_quant(x, eps=1e-5):
    """Per-token int8 (absmax) fake-quant; returns a dequantized tensor."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=eps)
    return (x * scale).round().clamp_(-128, 127) / scale


def weight_quant_ternary(w, mode="median", eps=1e-5):
    """Ternarize to {-1,0,+1} * scale, per OUTPUT channel.

    mode="median" uses the absmedian scale (BitNet b1.58 Reloaded, arXiv:2407.09527
    — more robust than absmean for SMALL models). mode="mean" is vanilla b1.58.
    """
    if mode == "median":
        scale = w.abs().median(dim=1, keepdim=True).values.clamp_(min=eps)
    else:
        scale = w.abs().mean(dim=1, keepdim=True).clamp_(min=eps)
    return (w / scale).round().clamp_(-1, 1) * scale


def _ste(real, quant, alpha):
    """Straight-through estimator with anneal blend (alpha in [0,1])."""
    blended = real + alpha * (quant - real)
    return real + (blended - real).detach()


class RMSNorm(nn.Module):
    # param name `.w` matches reproduce/train_slothlm_e.py so teacher weights load
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


class BitLinear(nn.Linear):
    """nn.Linear drop-in. weight_bits='ternary' -> W1.58A8 QAT; 'fp' -> plain fp.

    pre_norm=True inserts an RMSNorm before the projection (the "extra RMSNorm"
    stabilizer, arXiv:2505.08823). For 'fp' islands we keep it off so the state
    dict matches the original SlothE exactly (teacher loading).
    """

    def __init__(self, in_f, out_f, bias=False, weight_bits="ternary",
                 act_bits=8, weight_quant="median", pre_norm=True):
        super().__init__(in_f, out_f, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.wq_mode = weight_quant
        self.pre = RMSNorm(in_f) if (pre_norm and weight_bits == "ternary") else None
        self.register_buffer("quant_alpha", torch.tensor(1.0), persistent=True)

    def set_quant_alpha(self, a):
        self.quant_alpha.fill_(float(a))

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        if self.weight_bits == "fp":
            return F.linear(x, self.weight, self.bias)
        a = self.quant_alpha  # tensor keeps torch.compile stable across anneal
        if self.act_bits:
            x = _ste(x, activation_quant(x), a)
        w = _ste(self.weight, weight_quant_ternary(self.weight, self.wq_mode), a)
        return F.linear(x, w, self.bias)


# ======================================================================
# Model — mirrors reproduce/train_slothlm_e.py, Linear -> BitLinear
# ======================================================================

def rope(x, pos, dim):
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    ang = pos[:, None].float() * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def _lin(in_f, out_f, fp, wq, pre_norm):
    return BitLinear(in_f, out_f, bias=False,
                     weight_bits="fp" if fp else "ternary",
                     weight_quant=wq, pre_norm=pre_norm)


class Attn(nn.Module):
    def __init__(self, dim, heads, kv, fp, wq, pre_norm):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv, dim // heads
        self.q = _lin(dim, heads * self.dh, fp, wq, pre_norm)
        self.k = _lin(dim, kv * self.dh, fp, wq, pre_norm)
        self.v = _lin(dim, kv * self.dh, fp, wq, pre_norm)
        self.o = _lin(heads * self.dh, dim, fp, wq, pre_norm)
        self.qn = RMSNorm(self.dh)
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
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=amask[:, None, None, :])
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.o(o)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, fp, wq, pre_norm):
        super().__init__()
        self.w1 = _lin(dim, hidden, fp, wq, pre_norm)
        self.w3 = _lin(dim, hidden, fp, wq, pre_norm)
        self.w2 = _lin(hidden, dim, fp, wq, pre_norm)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, kv, ffn, fp, wq, pre_norm):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attn(dim, heads, kv, fp, wq, pre_norm)
        self.n2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn, fp, wq, pre_norm)

    def forward(self, x, pos, amask):
        x = x + self.attn(self.n1(x), pos, amask)
        x = x + self.ffn(self.n2(x))
        return x


class SlothE_T(nn.Module):
    """Ternary SlothLM-E. embed + head + first/last `fp_boundary` blocks stay fp."""

    def __init__(self, n_syl, n_char, dim=320, depth=16, heads=8, kv=2, ffn=880,
                 embed_norm=True, weight_bits="ternary", weight_quant="median",
                 fp_boundary=1, pre_norm=True, act_bits=8, char_hints=False, tie_hints=False):
        super().__init__()
        self.embed = nn.Embedding(n_syl, dim)                 # fp island
        self.hint_tied = char_hints and tie_hints
        self.hint_embed = nn.Embedding(n_char + 1, dim, padding_idx=0) if (char_hints and not tie_hints) else None
        self.hint_none = nn.Parameter(torch.zeros(dim)) if self.hint_tied else None
        self.embed_norm = RMSNorm(dim) if embed_norm else None
        blocks = []
        for i in range(depth):
            is_fp = (weight_bits == "fp") or i < fp_boundary or i >= depth - fp_boundary
            blocks.append(Block(dim, heads, kv, ffn, is_fp, weight_quant, pre_norm))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, n_char, bias=False)        # fp island (int8 at deploy)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_quant_alpha(self, a):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.set_quant_alpha(a)

    def forward(self, syl, amask, hints=None):
        pos = torch.arange(syl.shape[1], device=syl.device)
        x = self.embed(syl)
        if hints is not None and self.hint_tied:
            has = (hints > 0).unsqueeze(-1).to(x.dtype)
            he = F.embedding((hints - 1).clamp(min=0), self.head.weight)
            x = x + he * has + self.hint_none * (1.0 - has)
        elif self.hint_embed is not None and hints is not None:
            x = x + self.hint_embed(hints)
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        for b in self.blocks:
            x = b(x, pos, amask)
        return self.head(self.norm(x))


# ======================================================================
# Data (reuses the AlignedBin format from the original pipeline)
# ======================================================================

class AlignedBin(Dataset):
    CTX_MAX = 12
    def __init__(self, path, ctx_p=0.0, typo=None, typo_p=0.0, teacher_path=None):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.tdata = np.fromfile(teacher_path, dtype=np.uint16) if teacher_path else None
        self.ctx_p, self.typo, self.typo_p = ctx_p, typo, typo_p
        self.idx = []
        i, d = 0, self.data
        while i < len(d):
            n = int(d[i]); self.idx.append((i + 1, n)); i += 1 + 2 * n
        print(f"{len(self.idx)} aligned pairs")
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        s, n = self.idx[k]
        syl = self.data[s:s + n].astype(np.int64)
        tsyl = (self.tdata[s:s + n].astype(np.int64)
                if self.tdata is not None else syl.copy())
        chr_ = self.data[s + n:s + 2 * n].astype(np.int64)
        chr_[chr_ == 65535] = -100
        if self.typo is not None and self.typo_p > 0 and np.random.rand() < self.typo_p:
            zh = np.nonzero(chr_ >= 0)[0]
            if len(zh):
                i = int(np.random.choice(zh)); nb = self.typo.get(int(syl[i]))
                if nb: syl[i] = int(nb[np.random.randint(len(nb))])
        forced = np.zeros(n, dtype=np.int64)
        if self.ctx_p > 0 and k >= 2 and np.random.rand() < self.ctx_p:
            ps, pn = self.idx[k - 2]
            pch = self.data[ps + pn:ps + 2 * pn].astype(np.int64)
            pch = pch[pch != 65535][-self.CTX_MAX:]
            if len(pch):
                L = len(pch)
                syl = np.concatenate([np.zeros(L, dtype=np.int64), syl])
                tsyl = np.concatenate([np.zeros(L, dtype=np.int64), tsyl])
                chr_ = np.concatenate([np.full(L, -100, dtype=np.int64), chr_])
                forced = np.concatenate([pch + 1, forced])
        return (torch.from_numpy(syl), torch.from_numpy(tsyl),
                torch.from_numpy(chr_), torch.from_numpy(forced))


def collate(batch, pad=0):
    n = max(len(s) for s, _, _, _ in batch); B = len(batch)
    syl = torch.zeros(B, n, dtype=torch.long)
    tsyl = torch.zeros(B, n, dtype=torch.long)
    chr_ = torch.full((B, n), -100, dtype=torch.long)
    mask = torch.zeros(B, n, dtype=torch.bool)
    forced = torch.zeros(B, n, dtype=torch.long)
    for i, (sq, tq, c, f) in enumerate(batch):
        syl[i, :len(sq)] = sq; tsyl[i, :len(tq)] = tq
        chr_[i, :len(c)] = c; mask[i, :len(sq)] = True; forced[i, :len(f)] = f
    return syl, tsyl, chr_, mask, forced


def masked_ce(logits, target, syl, legal_mask, eps):
    """CE + label-smoothing restricted to the per-position legal char set (mirrors
    serve-time masking): the ternary net optimizes only the ~1-50 legal chars eval
    scores on, not all 8342. Positions whose gold char is not legal (train/serve
    skew) are dropped so the loss never goes -inf. logits (B,T,V); target (B,T,-100
    pad); syl (B,T) input syllable ids; legal_mask (n_syl,V) bool."""
    lm = legal_mask[syl]
    logits = logits.masked_fill(~lm, float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    tgt = target.clamp(min=0)
    gold_legal = lm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    valid = (target != -100) & gold_legal
    ce_hard = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nlegal = lm.sum(-1).clamp(min=1)
    ls_term = -(logp.masked_fill(~lm, 0.0)).sum(-1) / nlegal
    loss = (1.0 - eps) * ce_hard + eps * ls_term
    return loss[valid].mean()


def load_teacher(path, dev):
    """Load a frozen all-fp SlothE_T teacher (KD). Mirrors gate build()."""
    ck = torch.load(os.path.join(path, "slothe.pt"), map_location="cpu")
    c = ck["config"]
    m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"],
                 c["kv"], c["ffn"], embed_norm=c.get("embed_norm", False),
                 weight_bits=c.get("weight_bits") or "fp",
                 weight_quant=c.get("weight_quant", "median"),
                 fp_boundary=c.get("fp_boundary", 1),
                 pre_norm=c.get("pre_norm", True), act_bits=c.get("act_bits", 8),
                 char_hints=c.get("char_hints", False),
                 tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"], strict=False)  # teacher predates quant_alpha buffers (unused for fp)
    m.set_quant_alpha(1.0)
    m.eval().to(dev)
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m


def distill_loss(logits, t_logits, target, alpha, temp):
    """(1-alpha)*CE + alpha*KL(student||teacher) at temperature `temp`."""
    V = logits.shape[-1]
    flat = target.reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), flat, ignore_index=-100)
    keep = flat != -100
    s = logits.reshape(-1, V)[keep]
    t = t_logits.reshape(-1, V)[keep]
    kl = F.kl_div(F.log_softmax(s / temp, -1), F.softmax(t / temp, -1),
                  reduction="batchmean") * (temp * temp)
    return (1.0 - alpha) * ce + alpha * kl, ce.detach(), kl.detach()



def _dachen_key_adjacency():
    """Bopomofo symbol -> set of symbols on physically 8-adjacent keys, per the
    TAAI-2024 keyboard 8-adjacency error model (Li/Yeh/Chang, 大城市 = 284t/6g4):
    a real mis-key lands on one of the 8 neighbours in the QWERTY 3x3 block
    ('h' -> t/y/u/g/j/b/n/m). Used to make typo-noise reflect finger slips
    instead of arbitrary edit-distance-1 substitutions."""
    rows = ["1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"]
    pos = {k: (r, c) for r, row in enumerate(rows) for c, k in enumerate(row)}
    key_nb = {}
    for k, (r, c) in pos.items():
        key_nb[k] = {k2 for k2, (r2, c2) in pos.items()
                     if abs(r2 - r) <= 1 and abs(c2 - c) <= 1 and k2 != k}
    DACHEN = {"1": "\u3105", "q": "\u3106", "a": "\u3107", "z": "\u3108",
              "2": "\u3109", "w": "\u310a", "s": "\u310b", "x": "\u310c",
              "e": "\u310d", "d": "\u310e", "c": "\u310f", "r": "\u3110",
              "f": "\u3111", "v": "\u3112", "5": "\u3113", "t": "\u3114",
              "g": "\u3115", "b": "\u3116", "y": "\u3117", "h": "\u3118",
              "n": "\u3119", "u": "\u3127", "j": "\u3128", "m": "\u3129",
              "8": "\u311a", "i": "\u311b", "k": "\u311c", ",": "\u311d",
              "9": "\u311e", "o": "\u311f", "l": "\u3120", ".": "\u3121",
              "0": "\u3122", "p": "\u3123", ";": "\u3124", "/": "\u3125",
              "-": "\u3126"}
    sym_key = {v: k for k, v in DACHEN.items()}
    return {sym: {DACHEN[k2] for k2 in key_nb.get(k, ()) if k2 in DACHEN}
            for sym, k in sym_key.items()}


def build_typo_neighbors(syl_vocab):
    TONES = set("ˊˇˋ˙"); by_key = {}
    sym_adj = _dachen_key_adjacency()  # TAAI 8-adjacency slip model
    for sy, i in syl_vocab.items():
        if sy.startswith("<"): continue
        tone = "".join(c for c in sy if c in TONES); base = tuple(c for c in sy if c not in TONES)
        by_key.setdefault((tone, len(base)), []).append((base, i))
    out = {}
    for sy, i in syl_vocab.items():
        if sy.startswith("<"): continue
        tone = "".join(c for c in sy if c in TONES); base = tuple(c for c in sy if c not in TONES); nb = []
        for dl in (-1, 0, 1):
            for other, j in by_key.get((tone, len(base) + dl), []):
                if j == i: continue
                a, b = base, other
                if len(a) == len(b):
                    diffs = [(x, y) for x, y in zip(a, b) if x != y]
                    if len(diffs) == 1 and diffs[0][1] in sym_adj.get(diffs[0][0], ()):
                        nb.append(j)  # only physically-adjacent-key slips
                else:
                    s1, s2 = (a, b) if len(a) < len(b) else (b, a)
                    for pp in range(len(s2)):
                        if s2[:pp] + s2[pp+1:] == s1: nb.append(j); break
        if nb: out[i] = nb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="syl_vocab.json")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="slothe_t")
    # architecture (recommended ~20M ternary: dim320 depth16 heads8 kv2 ffn880)
    ap.add_argument("--dim", type=int, default=320)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=880)
    ap.add_argument("--embed-norm", action="store_true")
    # quantization
    ap.add_argument("--quant", choices=["ternary", "fp"], default="ternary")
    ap.add_argument("--weight-quant", choices=["median", "mean"], default="median")
    ap.add_argument("--fp-boundary", type=int, default=1)
    ap.add_argument("--pre-norm", action="store_true", default=True)
    ap.add_argument("--no-pre-norm", dest="pre_norm", action="store_false")
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--anneal-frac", type=float, default=0.15,
                    help="ramp quant_alpha 0->1 over this fraction of steps")
    # distillation
    ap.add_argument("--teacher", default="", help="dir with a fp teacher slothe.pt")
    ap.add_argument("--teacher-data", default="", help="exact-syllable .bin for cross-input distillation (teacher input)")
    ap.add_argument("--distill-alpha", type=float, default=0.7)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    # optimization (retuned for small ternary: higher LR ok, see 2407.09527)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--char-hints", action="store_true")
    ap.add_argument("--tie-hints", action="store_true")
    ap.add_argument("--typo-noise", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (compile BEFORE DDP wrap)")
    ap.add_argument("--compile-mode", default="reduce-overhead", help="reduce-overhead | cudagraphs | default")
    ap.add_argument("--legal-mask", default="", help="npz (n_syl,V) legal-char mask -> mask CE + LS to the legal set")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="teacher-free soft targets for direct (pure-CE) training")
    ap.add_argument("--context", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=0, help="snapshot {out}_ep{N} every N epochs (for accuracy-vs-epoch curve)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"]); rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(local_rank); dev = f"cuda:{local_rank}"
    else:
        rank, local_rank, world = 0, 0, 1
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    is_main = (rank == 0)
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    typo_map = build_typo_neighbors(syl_vocab) if args.typo_noise > 0 else None
    ds = AlignedBin(args.data, ctx_p=args.context, typo=typo_map, typo_p=args.typo_noise,
                    teacher_path=(args.teacher_data or None))
    sampler = torch.utils.data.DistributedSampler(ds, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                    num_workers=8, collate_fn=collate, pin_memory=True, drop_last=True)
    model = SlothE_T(len(syl_vocab), len(tok), args.dim, args.depth, args.heads,
                     args.kv_heads, args.ffn, embed_norm=args.embed_norm,
                     weight_bits=args.quant, weight_quant=args.weight_quant,
                     fp_boundary=args.fp_boundary, pre_norm=args.pre_norm,
                     act_bits=args.act_bits, char_hints=args.char_hints, tie_hints=args.tie_hints).to(dev)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=dev)["model"])
        if int(os.environ.get("RANK","0"))==0: print("resumed from "+args.resume, flush=True)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"SlothLM-E-T {np_/1e6:.1f}M params | quant={args.quant} "
          f"wq={args.weight_quant} fp_boundary={args.fp_boundary} pre_norm={args.pre_norm}")

    teacher = load_teacher(args.teacher, dev) if args.teacher else None
    legal_mask = (torch.from_numpy(np.load(args.legal_mask)["mask"]).to(dev)
                  if args.legal_mask else None)
    raw_model = model
    if args.compile:
        model = (torch.compile(model, backend="cudagraphs")
                 if args.compile_mode == "cudagraphs"
                 else torch.compile(model, mode=args.compile_mode))
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    total = args.steps or int(len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.03)
    anneal_steps = max(1, int(args.anneal_frac * total))

    model.train()
    step = 0
    for ep in range(math.ceil(args.epochs)):
        if ddp: sampler.set_epoch(ep)
        for syl, tsyl, chr_, mask, forced in dl:
            syl, tsyl, chr_, mask, forced = (syl.to(dev), tsyl.to(dev),
                chr_.to(dev), mask.to(dev), forced.to(dev))
            hints = None
            if args.char_hints:
                reveal_p = torch.rand(syl.shape[0], 1, device=dev) * 0.3
                reveal = (torch.rand(syl.shape, device=dev) < reveal_p) & (chr_ >= 0)
                hints = torch.where(reveal, chr_ + 1, torch.zeros_like(chr_))
                hints = torch.where(forced > 0, forced, hints)
            raw_model.set_quant_alpha(1.0 if args.resume else min(1.0, step / anneal_steps))
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(syl, mask, hints)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(tsyl, mask, hints)  # exact syllables
                    loss, ce, kl = distill_loss(logits, t_logits, chr_,
                                                args.distill_alpha, args.distill_temp)
                elif legal_mask is not None:
                    loss = masked_ce(logits, chr_, syl, legal_mask, args.label_smoothing)
                    ce = kl = loss.detach()
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           chr_.reshape(-1), ignore_index=-100,
                                           label_smoothing=args.label_smoothing)
                    ce = kl = loss.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0 and is_main:
                a = min(1.0, step / anneal_steps)
                print(f"step {step}/{total} loss {loss.item():.3f} "
                      f"ce {ce.item():.3f} kl {kl.item():.3f} alpha {a:.2f}", flush=True)
            if step >= total:
                break
        if args.save_every and (ep + 1) % args.save_every == 0:
            if is_main:
                _snap = f"{args.out}_ep{ep + 1}"
                os.makedirs(_snap, exist_ok=True)
                torch.save({"model": raw_model.state_dict(),
                            "config": {"n_syl": len(syl_vocab), "n_char": len(tok),
                                       "dim": args.dim, "depth": args.depth,
                                       "heads": args.heads, "kv": args.kv_heads,
                                       "ffn": args.ffn, "embed_norm": args.embed_norm,
                                       "weight_bits": args.quant,
                                       "weight_quant": args.weight_quant,
                                       "fp_boundary": args.fp_boundary,
                                       "pre_norm": args.pre_norm,
                                       "act_bits": args.act_bits,
                                       "char_hints": args.char_hints,
                                       "tie_hints": args.tie_hints}},
                           os.path.join(_snap, "slothe.pt"))
                json.dump(syl_vocab, open(os.path.join(_snap, "syl_vocab.json"),
                          "w", encoding="utf-8"), ensure_ascii=False)
                print(f"snapshot saved to {_snap}", flush=True)
            if ddp: dist.barrier()
        if step >= total:
            break

    raw_model.set_quant_alpha(1.0)
    if ddp: dist.barrier()
    if not is_main:
        if ddp: dist.destroy_process_group()
        return
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": raw_model.state_dict(),
                "config": {"n_syl": len(syl_vocab), "n_char": len(tok),
                           "dim": args.dim, "depth": args.depth, "heads": args.heads,
                           "kv": args.kv_heads, "ffn": args.ffn,
                           "embed_norm": args.embed_norm, "weight_bits": args.quant,
                           "weight_quant": args.weight_quant,
                           "fp_boundary": args.fp_boundary, "pre_norm": args.pre_norm,
                           "act_bits": args.act_bits, "char_hints": args.char_hints,
                           "tie_hints": args.tie_hints}},
               os.path.join(args.out, "slothe.pt"))
    _sd = raw_model.state_dict()
    json.dump(syl_vocab, open(os.path.join(args.out, "syl_vocab.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
