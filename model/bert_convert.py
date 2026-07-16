#!/usr/bin/env python3
"""Exp C': adapt a PRETRAINED Chinese RoBERTa/BERT (massive LM knowledge) to the
syllable->char conversion task.

The task is 1:1 position-aligned bidirectional char classification, which is
exactly what a char-MLM head does — so we KEEP the pretrained encoder + MLM head
(they already predict Chinese chars with real LM priors) and only REPLACE the
word-embedding with a fresh syllable embedding. One non-autoregressive pass.

Targets in the .bin are char ids in OUR tokenizer's vocab; we remap them to the
BERT vocab at load time. <en>/IGNORE positions -> label -100.

  python3 bert_convert.py --base hfl/chinese-roberta-wwm-ext-large \
      --data train_e_g2pw_tl.bin --vocab syl_vocab.json --our-tokenizer tokenizer \
      --out slothe_bert --epochs 2 --batch 96 --lr 3e-5
"""
import argparse, json, math, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

IGNORE = 65535


class BinDS(Dataset):
    """Yields (syl_ids[np], char_bert_ids[np]) records from the packed uint16 .bin."""
    def __init__(self, path, char_remap, maxlen=48):
        data = np.fromfile(path, dtype=np.uint16)
        self.recs = []
        i = 0
        while i < len(data):
            L = int(data[i])
            if i + 1 + 2 * L > len(data): break
            syl = data[i + 1:i + 1 + L].astype(np.int64)
            ch = data[i + 1 + L:i + 1 + 2 * L].astype(np.int64)
            lab = np.array([char_remap.get(int(c), -100) if c != IGNORE else -100 for c in ch],
                           dtype=np.int64)
            if L <= maxlen and (lab != -100).any():
                self.recs.append((syl, lab))
            i += 1 + 2 * L
    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]



class JsonlDS(Dataset):
    """Same (syl_ids, char_bert_labels) records but from a scored jsonl, with an
    optional teacher-confidence floor (MiniPLM difference-sampling)."""
    _BOPO = set("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ" + "ˊˇˋ˙")
    def __init__(self, path, syl_vocab, bvocab, maxlen=48, min_score=None):
        self.recs = []
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if min_score is not None and d.get("score", 0.0) < min_score:
                continue
            sy = [x for x in d["in"].split() if x and x[0] in self._BOPO]
            gold = [c for c in d["out"] if "一" <= c <= "鿿"]
            if not sy or len(sy) != len(gold) or len(sy) > maxlen:
                continue
            if not all(x in syl_vocab for x in sy):
                continue
            syl = np.array([syl_vocab[x] for x in sy], dtype=np.int64)
            lab = np.array([bvocab.get(c, -100) for c in gold], dtype=np.int64)
            if (lab != -100).any():
                self.recs.append((syl, lab))
    def __len__(self): return len(self.recs)
    def __getitem__(self, i): return self.recs[i]

def collate(batch, pad_syl=0):
    T = max(len(s) for s, _ in batch)
    B = len(batch)
    syl = np.full((B, T), pad_syl, dtype=np.int64)
    lab = np.full((B, T), -100, dtype=np.int64)
    am = np.zeros((B, T), dtype=np.int64)
    for i, (s, l) in enumerate(batch):
        syl[i, :len(s)] = s; lab[i, :len(l)] = l; am[i, :len(s)] = 1
    return (torch.from_numpy(syl), torch.from_numpy(am), torch.from_numpy(lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="syl_vocab.json")
    ap.add_argument("--our-tokenizer", default="tokenizer")
    ap.add_argument("--out", default="slothe_bert")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmstart", action="store_true",
                    help="init each syllable embed = mean of its candidate chars' "
                         "PRETRAINED char embeddings (keeps the encoder's synergy)")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args()

    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    n_syl = len(syl_vocab)
    our_tok = AutoTokenizer.from_pretrained(args.our_tokenizer)
    bert_tok = AutoTokenizer.from_pretrained(args.base)
    # our char-id -> char string -> BERT vocab id (skip unmapped/multi-piece)
    our_vocab = our_tok.get_vocab()
    id2char = {v: k for k, v in our_vocab.items()}
    bvocab = bert_tok.get_vocab()
    char_remap = {}
    for cid, ch in id2char.items():
        if len(ch) == 1 and ch in bvocab:
            char_remap[cid] = bvocab[ch]
    print(f"char remap: {len(char_remap)}/{len(id2char)} our-chars -> BERT vocab", file=sys.stderr)

    if args.data.endswith(".jsonl"):
        ds = JsonlDS(args.data, syl_vocab, bvocab, min_score=args.min_score)
    else:
        ds = BinDS(args.data, char_remap)
    print(f"records: {len(ds)}; n_syl={n_syl}", file=sys.stderr)

    dev = "cuda"
    model = AutoModelForMaskedLM.from_pretrained(args.base)
    hid = model.config.hidden_size
    # swap the word-embedding for a fresh syllable embedding (kept as a plain
    # Embedding we feed via inputs_embeds so position/type embeddings still apply)
    syl_embed = nn.Embedding(n_syl, hid)
    nn.init.normal_(syl_embed.weight, std=0.02)
    if args.warmstart:
        from eval_arch import tone_union_chars
        cands = tone_union_chars(args.table)
        id2syl = {v: k for k, v in syl_vocab.items()}
        W = model.get_input_embeddings().weight.data  # pretrained char embeddings
        nwarm = 0
        with torch.no_grad():
            for sid, syl in id2syl.items():
                if syl in ("<pad>", "<unk>", "<en>"):
                    continue
                ids = [bvocab[c] for c in cands(syl) if c in bvocab]
                if ids:
                    syl_embed.weight[sid] = W[ids].mean(0); nwarm += 1
        print(f"warm-started {nwarm} syllable embeddings from char embeddings", file=sys.stderr)
    model.to(dev); syl_embed.to(dev)
    nparam = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in syl_embed.parameters())
    print(f"{args.base}: {nparam/1e6:.0f}M params (encoder+head+syl-embed)", file=sys.stderr)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, collate_fn=collate)
    total = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(list(model.parameters()) + list(syl_embed.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    step = 0; model.train()
    for ep in range(math.ceil(args.epochs)):
        for syl, am, lab in dl:
            if step >= total: break
            syl, am, lab = syl.to(dev), am.to(dev), lab.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = syl_embed(syl)
                out = model(inputs_embeds=emb, attention_mask=am)
                loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)),
                                       lab.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(syl_embed.parameters()), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.3f}", flush=True)
            step += 1
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    torch.save(syl_embed.state_dict(), os.path.join(args.out, "syl_embed.pt"))
    json.dump({"base": args.base, "n_syl": n_syl,
               "char_remap": {str(k): v for k, v in char_remap.items()}},
              open(os.path.join(args.out, "adapt.json"), "w"))
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
