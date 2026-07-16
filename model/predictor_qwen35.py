#!/usr/bin/env python3
"""B2: train a TINY DENSE Qwen3.5 (~33M) as the next-word predictor, so the
OFFICIAL convert_hf_to_gguf.py produces a correct llama.cpp qwen35 GGUF (no
reverse-engineering, no fork). Dense hybrid: linear-attention (Gated DeltaNet)
layers + full-attention every 4th (layer_types), word-piece vocab + emojis.

  python3 predictor_qwen35.py --epochs 3
"""
import argparse, math, os, sys, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM


class LMDS(Dataset):
    def __init__(self, ids, seqlen):
        self.n = len(ids) // seqlen
        self.ids = np.array(ids[: self.n * seqlen], dtype=np.int64).reshape(self.n, seqlen)
    def __len__(self): return self.n
    def __getitem__(self, i): return torch.from_numpy(self.ids[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="predictor_tok.json")
    ap.add_argument("--corpus", default="corpus_e3.txt")
    ap.add_argument("--out", default="pred_qwen35")
    ap.add_argument("--seqlen", type=int, default=64); ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--ffn", type=int, default=2048)
    args = ap.parse_args()
    dev = "cuda"
    tok = Tokenizer.from_file(args.tok)
    V = tok.get_vocab_size(); BOS = tok.token_to_id("<bos>")
    cfg = Qwen3_5TextConfig(
        vocab_size=V, hidden_size=args.dim, num_hidden_layers=args.depth, intermediate_size=args.ffn,
        num_attention_heads=8, num_key_value_heads=2, head_dim=64,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        tie_word_embeddings=True, max_position_embeddings=2048,
        bos_token_id=BOS, eos_token_id=tok.token_to_id("<eos>"))
    m = Qwen3_5ForCausalLM(cfg).to(dev).to(torch.bfloat16)
    print(f"tiny dense Qwen3.5: {sum(p.numel() for p in m.parameters())/1e6:.1f}M, layer_types={cfg.layer_types[:6]}", file=sys.stderr)

    lines = [l.rstrip("\n") for l in open(args.corpus, encoding="utf-8")]
    if args.holdout: lines = lines[:-args.holdout]
    stream = []
    for enc in tok.encode_batch(lines):
        stream.append(BOS); stream.extend(enc.ids)
    print(f"vocab={V} stream={len(stream)}", file=sys.stderr)
    dl = DataLoader(LMDS(stream, args.seqlen), batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True)
    total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.02)
    step = 0; m.train()
    for ep in range(math.ceil(args.epochs)):
        for x in dl:
            if step >= total: break
            x = x.to(dev)
            out = m(input_ids=x, labels=x)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0: print(f"step {step}/{total} loss {out.loss.item():.3f}", flush=True)
            step += 1
    m.save_pretrained(args.out)
    # save HF tokenizer files so convert_hf_to_gguf can read them
    import shutil; shutil.copy(args.tok, os.path.join(args.out, "tokenizer.json"))
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__": main()
