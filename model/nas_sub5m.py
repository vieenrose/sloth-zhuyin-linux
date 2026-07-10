#!/usr/bin/env python3
"""Neural architecture search for the best sub-5M SlothLM-E (zhuyin/en-mix IME).

Successive-halving (Hyperband) search: sample many sub-5M architectures, train
each briefly, keep the top fraction, retrain the survivors longer, repeat. The
model owns zh decode + <en> passthrough; zh/en segmentation is the DP's job
(segment.js, arch-independent), so fitness = decode accuracy weighted toward the
north-star 免選字 (230-case) + homophone-hard + toneless (159 each).

  CUDA_VISIBLE_DEVICES=0 python3 model/nas_sub5m.py --n0 18 --gpu 0
"""
import argparse, json, os, random, re, subprocess, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import SlothE

NS, NC = 1539, 8342          # syllable / char vocab
BUDGET = (2.8e6, 5.0e6)      # sub-5M window
DATA = "model/train_e_g2pw.bin"
RES = "model/nas_sub5m_results.tsv"


def params(dim, depth, heads, kv, ffn, en):
    m = SlothE(NS, NC, dim, depth, heads, kv, ffn, embed_norm=en)
    return sum(p.numel() for p in m.parameters())


def sample_space(rng, n):
    """Draw n distinct sub-5M configs."""
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 200:
        tries += 1
        dim = rng.choice([128, 144, 160, 176, 192, 208, 224])
        heads = rng.choice([h for h in (4, 6, 8) if dim % h == 0])
        kv = rng.choice([k for k in (1, 2) if heads % k == 0])
        ffn = int(round(dim * rng.choice([2.0, 2.5, 3.0, 3.5]) / 16) * 16)
        depth = rng.randint(4, 12)
        en = rng.choice([0, 1])
        key = (dim, depth, heads, kv, ffn, en)
        if key in seen:
            continue
        p = params(dim, depth, heads, kv, ffn, en)
        if not (BUDGET[0] <= p <= BUDGET[1]):
            continue
        seen.add(key)
        out.append({"dim": dim, "depth": depth, "heads": heads, "kv": kv,
                    "ffn": ffn, "en": en, "params": p})
    return out


def tag(c):
    return f"d{c['dim']}L{c['depth']}h{c['heads']}k{c['kv']}f{c['ffn']}e{c['en']}"


def train_and_gate(c, epochs, gpu):
    t = tag(c)
    out = f"model/nas/{t}_ep{epochs}"
    cmd = ["python3", "model/train_slothlm_e.py", "--data", DATA,
           "--vocab", "model/syl_vocab.json", "--tokenizer", "model/tokenizer",
           "--out", out, "--dim", str(c["dim"]), "--depth", str(c["depth"]),
           "--heads", str(c["heads"]), "--kv-heads", str(c["kv"]),
           "--ffn", str(c["ffn"]), "--batch", "512", "--epochs", str(epochs),
           "--lr", "1.5e-3"]
    if c["en"]:
        cmd.append("--embed-norm")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"/tmp/nas_{t}.log", "w")
    r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return None
    g = subprocess.run(["python3", "model/gate_g2pw.py", "--model", out],
                       env=env, capture_output=True, text=True).stdout
    # match "N/M (P%)" so label text like "(E3: 84%)" can never be mistaken
    # for the score (that bug made every config report h84/t82)
    pct = lambda k: (int(re.search(rf"{k}.*?\d+/\d+ \((\d+)%\)", g).group(1))
                     if re.search(rf"{k}.*?\d+/\d+ \((\d+)%\)", g) else 0)
    h, m, tl = pct("homophone"), pct("免選字"), pct("toneless")
    fit = 0.5 * m + 0.3 * h + 0.2 * tl        # 免選字 is the north star
    res = {**c, "epochs": epochs, "h": h, "m": m, "tl": tl, "fit": round(fit, 2),
           "out": out}
    with open(RES, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"  {tag(c)} {c['params']/1e6:.1f}M ep{epochs}: "
          f"h{h}/m{m}/t{tl} fit{fit:.1f}", file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=18)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rungs", default="1.5,4,8")   # epochs per rung
    ap.add_argument("--keep", default="6,3")         # survivors after each rung
    args = ap.parse_args()
    os.makedirs("model/nas", exist_ok=True)
    open(RES, "w").close()
    rng = random.Random(args.seed)
    rungs = [float(x) for x in args.rungs.split(",")]
    keep = [int(x) for x in args.keep.split(",")]

    pool = sample_space(rng, args.n0)
    print(f"=== NAS: {len(pool)} sub-5M configs, rungs {rungs}, keep {keep} ===",
          file=sys.stderr, flush=True)
    for i, ep in enumerate(rungs):
        print(f"--- rung {i}: {len(pool)} configs @ {ep} epochs ---",
              file=sys.stderr, flush=True)
        scored = [r for c in pool if (r := train_and_gate(c, ep, args.gpu))]
        scored.sort(key=lambda r: r["fit"], reverse=True)
        for r in scored[:5]:
            print(f"    {tag(r)} {r['params']/1e6:.1f}M -> "
                  f"h{r['h']}/m{r['m']}/t{r['tl']} fit{r['fit']}",
                  file=sys.stderr, flush=True)
        if i < len(keep):
            pool = scored[:keep[i]]
    if scored:
        w = scored[0]
        print(f"\n=== WINNER: {tag(w)} {w['params']/1e6:.1f}M "
              f"h{w['h']}/m{w['m']}/t{w['tl']} fit{w['fit']} @ {w['out']} ===",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
