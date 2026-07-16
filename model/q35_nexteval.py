import sys, torch
from tokenizers import Tokenizer
from transformers import Qwen3_5ForCausalLM
mdir = sys.argv[1]
tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
m = Qwen3_5ForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).cuda().eval()
BOS = tok.token_to_id("<bos>")
# SAME held-out split as pure-GDN eval: last 20k lines (reserved from training)
lines = [l.rstrip("\n") for l in open("corpus_e3.txt", encoding="utf-8")][-20000:]
n = t1 = t5 = 0
with torch.no_grad():
    for line in lines:
        ids = tok.encode(line).ids
        if len(ids) < 3: continue
        if n >= 3000: break
        k = max(1, len(ids)//2); ctx=[BOS]+ids[:k]; true=ids[k]
        lg = m(torch.tensor([ctx],device="cuda")).logits[0,-1]
        top5 = torch.topk(lg,5).indices.tolist()
        n+=1; t1+=(top5[0]==true); t5+=(true in top5)
print("%s  hybrid next-word (n=%d): top-1 %.1f%%  top-5 %.1f%%" % (mdir, n, 100*t1/n, 100*t5/n))
