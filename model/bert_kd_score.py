#!/usr/bin/env python3
"""MiniPLM-style offline Difference Sampling for the conversion corpus.

Score each (syllables -> gold-text) pair by the Breeze-7B teacher's average
per-character log-probability of the GOLD conversion (teacher-forced). Low score =
the 7B teacher is uncertain about the gold => likely noisy / mis-segmented /
rare-char crawl junk. We drop the low tail and keep the confident, learnable
middle+top — the "data quality > scale" lever (arXiv 2410.17215).

One offline teacher pass, reusable. Batched on 1 GPU.

  python3 bert_kd_score.py --teacher slothe_breeze_teacher --in gen_pairs_full.jsonl \
      --out gen_pairs_scored.jsonl
"""
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_teacher(path, dev):
    tok = AutoTokenizer.from_pretrained(path)
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        base = json.load(open(os.path.join(path, "adapter_config.json")))["base_model_name_or_path"]
        bm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(bm, path).merge_and_unload().to(dev).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    return tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="slothe_breeze_teacher")
    ap.add_argument("--in", dest="inp", default="gen_pairs_full.jsonl")
    ap.add_argument("--out", default="gen_pairs_scored.jsonl")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--sep", default=" →")
    args = ap.parse_args()
    dev = "cuda"
    tok, m = load_teacher(args.teacher, dev)

    pairs = []
    for line in open(args.inp, encoding="utf-8"):
        d = json.loads(line)
        if d.get("in") and d.get("out"):
            pairs.append((d["in"], d["out"]))
        if args.max and len(pairs) >= args.max:
            break
    print(f"scoring {len(pairs)} pairs", file=sys.stderr)

    fout = open(args.out, "w", encoding="utf-8")
    n = 0
    with torch.no_grad():
        for b0 in range(0, len(pairs), args.batch):
            batch = pairs[b0:b0 + args.batch]
            enc, plens, glens = [], [], []
            for syl, gold in batch:
                prompt = syl + args.sep
                pid = tok(prompt, add_special_tokens=False)["input_ids"]
                gid = tok(gold, add_special_tokens=False)["input_ids"]
                enc.append(pid + gid); plens.append(len(pid)); glens.append(len(gid))
            T = max(len(e) for e in enc)
            pad = tok.pad_token_id or 0
            ids = torch.full((len(enc), T), pad, dtype=torch.long, device=dev)
            am = torch.zeros((len(enc), T), dtype=torch.long, device=dev)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e, device=dev); am[i, :len(e)] = 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(input_ids=ids, attention_mask=am).logits
            logp = torch.log_softmax(logits.float(), -1)
            for i, (syl, gold) in enumerate(batch):
                pl, gl = plens[i], glens[i]
                s = 0.0
                for j in range(gl):
                    pos = pl + j - 1                      # predicts token at pl+j
                    tgt = enc[i][pl + j]
                    s += logp[i, pos, tgt].item()
                avg = s / max(gl, 1)                      # avg per-char gold logprob
                fout.write(json.dumps({"in": syl, "out": gold, "score": round(avg, 4)},
                                      ensure_ascii=False) + "\n")
            n += len(batch)
            if b0 // args.batch % 200 == 0:
                print(f"  {n}/{len(pairs)}", file=sys.stderr)
    fout.close()
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
