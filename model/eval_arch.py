#!/usr/bin/env python3
"""Unified constrained-decode eval across architectures (bake-off scorer).

Scores toned AND toneless sentence accuracy on eval/testset.tsv with the same
tone-union constraint the decoder uses. Handles:
  --arch slothe : SlothE encoder (train_slothlm_e.py), our-tokenizer char space
  --arch bert   : pretrained-BERT adapted (bert_convert.py), BERT vocab space

  python3 eval_arch.py --arch slothe --model slothe_tl130m
  python3 eval_arch.py --arch bert --model slothe_bert_large
"""
import argparse, json, os, sys
import torch, torch.nn as nn
from transformers import AutoTokenizer

TONES = "ˊˇˋ˙"


def tone_union_chars(table_path):
    tonal = {}
    for line in open(table_path, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if rest:
            tonal[s] = list(rest)
    def cands(syl):
        has = any(c in TONES for c in syl)
        chars = tonal.get(syl) if has else None
        if chars is None:
            base = "".join(x for x in syl if x not in TONES)
            chars = []
            for k, v in tonal.items():
                if "".join(x for x in k if x not in TONES) == base:
                    for ch in v:
                        if ch not in chars:
                            chars.append(ch)
        return chars or [syl]
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["slothe", "bert"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--table", default="phonetic_table.tsv")
    ap.add_argument("--testset", default="../eval/testset.tsv")
    ap.add_argument("--vocab", default="syl_vocab.json")
    args = ap.parse_args()

    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    cands = tone_union_chars(args.table)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == "slothe":
        from train_slothlm_e import SlothE
        our = AutoTokenizer.from_pretrained(args.tokenizer)
        ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu")
        c = ck["config"]
        model = SlothE(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"],
                       c["kv"], c["ffn"], embed_norm=c.get("embed_norm", False),
                       char_hints=c.get("char_hints", False), tie_hints=c.get("tie_hints", False))
        model.load_state_dict(ck["model"]); model.to(dev).eval()
        def char2id(ch): return our.convert_tokens_to_ids(ch)
        def logits_of(sids):
            am = torch.ones_like(sids, dtype=torch.bool)
            with torch.no_grad():
                return model(sids, am)[0]
        def id2tok(i): return our.convert_ids_to_tokens(i)
    else:
        from transformers import AutoModelForMaskedLM
        adapt = json.load(open(os.path.join(args.model, "adapt.json")))
        bert_tok = AutoTokenizer.from_pretrained(adapt["base"])
        bvocab = bert_tok.get_vocab()
        model = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()
        hid = model.config.hidden_size
        syl_embed = nn.Embedding(adapt["n_syl"], hid)
        syl_embed.load_state_dict(torch.load(os.path.join(args.model, "syl_embed.pt"), map_location="cpu"))
        syl_embed.to(dev).eval()
        def char2id(ch): return bvocab.get(ch)
        def logits_of(sids):
            am = torch.ones_like(sids)
            with torch.no_grad():
                return model(inputs_embeds=syl_embed(sids), attention_mask=am).logits[0]
        def id2tok(i): return bert_tok.convert_ids_to_tokens(i)

    def decode(syls):
        sids = torch.tensor([[syl_vocab.get(y, 1) for y in syls]], device=dev)
        lg = logits_of(sids)
        out = []
        for i, syl in enumerate(syls):
            ids = [j for j in (char2id(ch) for ch in cands(syl)) if j is not None]
            if not ids:
                out.append("?"); continue
            best = max(ids, key=lambda t: lg[i, t].item())
            out.append(id2tok(best))
        return "".join(out)

    for toneless in (False, True):
        n = ok = 0
        for line in open(args.testset, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"): continue
            p = line.split("\t"); syls = p[0].split(); exp = p[1]
            if toneless:
                syls = ["".join(c for c in y if c not in TONES) for y in syls]
            n += 1; ok += decode(syls) == exp
        tag = "toneless" if toneless else "toned   "
        print(f"{args.arch} {os.path.basename(args.model)}  {tag}: {ok}/{n} ({100*ok/n:.1f}%)")


if __name__ == "__main__":
    main()
