#!/usr/bin/env python3
"""Evaluate SlothLM-E (encoder) against libchewing + the expected sentences.

Non-autoregressive: one forward pass over the syllable sequence, per-position
argmax restricted to each syllable's phonetic-legal char set (the same
constraint as the decoder). Reports sentence/char accuracy and the
chewing-parity gate (LLM must be >= chewing).

  python3 model/eval_slothe.py --model model/slothe --tokenizer model/tokenizer \
      --table model/phonetic_table.tsv --testset eval/testset.tsv --harvest eval/harvest
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import SlothE   # reuse the arch

TONES = "ˊˇˋ˙"
KEYMAP = {"ㄅ":"1","ㄆ":"q","ㄇ":"a","ㄈ":"z","ㄉ":"2","ㄊ":"w","ㄋ":"s","ㄌ":"x",
 "ㄍ":"e","ㄎ":"d","ㄏ":"c","ㄐ":"r","ㄑ":"f","ㄒ":"v","ㄓ":"5","ㄔ":"t","ㄕ":"g",
 "ㄖ":"b","ㄗ":"y","ㄘ":"h","ㄙ":"n","ㄧ":"u","ㄨ":"j","ㄩ":"m","ㄚ":"8","ㄛ":"i",
 "ㄜ":"k","ㄝ":",","ㄞ":"9","ㄟ":"o","ㄠ":"l","ㄡ":".","ㄢ":"0","ㄣ":"p","ㄤ":";",
 "ㄥ":"/","ㄦ":"-","ˊ":"6","ˇ":"3","ˋ":"4","˙":"7"}


def keys(bopomofo):
    k=[]
    for syl in bopomofo.split():
        k+=[KEYMAP[c] for c in syl]
        if syl[-1] not in TONES: k.append(" ")
    return "".join(k)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="model/slothe")
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    ap.add_argument("--testset", default="eval/testset.tsv")
    ap.add_argument("--harvest", default="eval/harvest")
    ap.add_argument("--toneless", action="store_true")
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.tokenizer)
    ckpt=torch.load(os.path.join(args.model,"slothe.pt"),map_location="cpu")
    syl_vocab=json.load(open(os.path.join(args.model,"syl_vocab.json"),encoding="utf-8"))
    c=ckpt["config"]
    model=SlothE(c["n_syl"],c["n_char"],c["dim"],c["depth"],c["heads"],c["kv"],c["ffn"])
    model.load_state_dict(ckpt["model"]); model.eval()

    tonal={}
    for line in open(args.table,encoding="utf-8"):
        s,_,rest=line.rstrip("\n").partition("\t")
        if rest: tonal[s]=list(rest)
    def cand_ids(syl):
        # a toneless syllable (no tone mark) must use the tone-union, not the
        # tone-1 entry that tonal.get would return for it.
        has_tone=any(c in TONES for c in syl)
        chars=tonal.get(syl) if has_tone else None
        if chars is None:
            base="".join(x for x in syl if x not in TONES)
            chars=[]
            for k,v in tonal.items():
                if "".join(x for x in k if x not in TONES)==base:
                    for ch in v:
                        if ch not in chars: chars.append(ch)
        ids=[]
        for ch in (chars or [syl]):
            i=tok.convert_tokens_to_ids(ch)
            if i is not None and i not in ids: ids.append(i)
        return ids

    def decode(syls):
        sids=torch.tensor([[syl_vocab.get(y,1) for y in syls]])
        amask=torch.ones_like(sids,dtype=torch.bool)
        with torch.no_grad():
            logits=model(sids,amask)[0]        # [T, vocab]
        out=[]
        for i,syl in enumerate(syls):
            ids=cand_ids(syl)
            if not ids: out.append("?"); continue
            best=max(ids,key=lambda t:logits[i,t].item())
            out.append(tok.convert_ids_to_tokens(best))
        return "".join(out)

    n=ch_ok=lm_ok=agree=0; lm_wins=ch_wins=both_wrong=0
    for line in open(args.testset,encoding="utf-8"):
        line=line.rstrip("\n")
        if not line or line.startswith("#"): continue
        p=line.split("\t"); bopomofo,expected=p[0],p[1]
        syls=bopomofo.split()
        if args.toneless:
            syls=["".join(c for c in y if c not in TONES) for y in syls]
        chew=json.loads(subprocess.run([args.harvest,keys(bopomofo)],
              capture_output=True,text=True).stdout)["buffer"]
        lm=decode(syls)
        n+=1; ch_ok+=chew==expected; lm_ok+=lm==expected; agree+=chew==lm
        if chew!=lm:
            if lm==expected: lm_wins+=1
            elif chew==expected: ch_wins+=1
            else: both_wrong+=1

    print(f"=== SlothLM-E{' (toneless)' if args.toneless else ''}: {n} cases ===")
    print(f"libchewing sentence: {ch_ok}/{n} ({100*ch_ok/n:.0f}%)")
    print(f"SlothLM-E  sentence: {lm_ok}/{n} ({100*lm_ok/n:.0f}%)  "
          f"[{'PASS' if lm_ok>=ch_ok else 'REGRESSION'}]")
    print(f"agree {agree}/{n}; of divergences LLM-right {lm_wins} chewing-right "
          f"{ch_wins} both-wrong {both_wrong}")
    sys.exit(0 if lm_ok>=ch_ok else 1)


if __name__=="__main__":
    main()
