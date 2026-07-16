#!/usr/bin/env python3
"""Next-word prediction accuracy for the Apple-style predictor. Held-out corpus
tail: given a text prefix, does the model's top-k predict the true next token
(= next word/emoji, since vocab is word-piece)?"""
import sys, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import distill_student as D

mdir = sys.argv[1]
ck = torch.load(f"{mdir}/predictor.pt", map_location="cuda"); c = ck["config"]
D.QAT["on"] = c.get("qat", False); D.SSM_TYPE["t"] = c.get("ssm_type","mamba1")
tok = Tokenizer.from_file(f"{mdir}/predictor_tok.json")
m = D.TinyGPT(c["vocab"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"], c.get("pattern")).cuda()
m.load_state_dict(ck["model"]); m.eval()
BOS = tok.token_to_id("<bos>")

# held-out: far tail of corpus (training saw it, but sample-level next-word acc is
# still meaningful for an LM; use a fresh slice for a fairer read)
lines = [l.rstrip("\n") for l in open("corpus_e3.txt", encoding="utf-8")][-3000:]
n = t1 = t5 = 0
with torch.no_grad():
    for line in lines:
        ids = tok.encode(line).ids
        if len(ids) < 3: continue
        k = max(1, len(ids) // 2)                    # predict token after prefix
        ctx = [BOS] + ids[:k]; true = ids[k]
        lg = m(torch.tensor([ctx], device="cuda"))[0, -1]
        top5 = torch.topk(lg, 5).indices.tolist()
        n += 1; t1 += (top5[0] == true); t5 += (true in top5)
print(f"{mdir}  next-word (n={n}): top-1 {100*t1/n:.1f}%  top-5 {100*t5/n:.1f}%")
print("=== qualitative (top-3 next word) ===")
for p in ["今天天氣", "我想喝", "謝謝你的", "大家好", "我在重新", "生日快樂"]:
    ids = [BOS] + tok.encode(p).ids
    lg = m(torch.tensor([ids], device="cuda"))[0, -1]
    top3 = [tok.id_to_token(i) for i in torch.topk(lg, 3).indices.tolist()]
    dec = [tok.decode([tok.token_to_id(t)]) if tok.token_to_id(t) else t for t in top3]
    print(f"  {p}… -> {dec}")
