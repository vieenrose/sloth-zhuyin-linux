import sys
from tokenizers import Tokenizer
from llama_cpp import Llama
gguf=sys.argv[1]
tok=Tokenizer.from_file("pred_q35_60m/tokenizer.json")
m=Llama(model_path=gguf, n_ctx=96, n_gpu_layers=0, verbose=False)
lines=[l.rstrip("\n") for l in open("corpus_e3.txt",encoding="utf-8")][-20000:]
n=t1=t5=0
for line in lines:
    ids=tok.encode(line).ids
    if len(ids)<3: continue
    if n>=351: break
    k=max(1,len(ids)//2)
    prefix=tok.decode(ids[:k]); true=tok.decode([ids[k]])
    if not prefix.strip(): continue
    out=m(prefix, max_tokens=1, temperature=0.0, top_k=1)
    gen=out["choices"][0]["text"]
    n+=1; t1+=(gen.strip()==true.strip())
print("%s next-word (n=%d): top-1 %.1f%%" % (gguf.split("/")[-1], n, 100*t1/n))
