#!/usr/bin/env python3
"""Golden reference for the ggml port: PyTorch intermediates + final logits at
quant_alpha=1 (deployed ternary), on held-out sentences. The ggml forward must
match these layer-by-layer, then argmax-exact per position."""
import torch, json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothe_ternary import SlothE_T

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "slothe_t_25m_ce_ls32_ep24"
REF = sys.argv[2] if len(sys.argv) > 2 else "../eval/reference_heldout.tsv"
N = 6

ck = torch.load(os.path.join(MODEL_DIR, "slothe.pt"), map_location="cpu")
c = ck["config"]
m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"],
             c["ffn"], embed_norm=c.get("embed_norm", False),
             weight_bits=c.get("weight_bits", "ternary"),
             weight_quant=c.get("weight_quant", "median"),
             fp_boundary=c.get("fp_boundary", 1),
             pre_norm=c.get("pre_norm", True), act_bits=c.get("act_bits", 8),
             char_hints=c.get("char_hints", False), tie_hints=c.get("tie_hints", False))
m.load_state_dict(ck["model"]); m.set_quant_alpha(1.0); m.eval()
sv = json.load(open(os.path.join(MODEL_DIR, "syl_vocab.json"), encoding="utf-8"))

# hooks to capture intermediates for sentence 0
acts = {}
def hook(name):
    def f(mod, inp, out): acts[name] = out.detach().float().numpy().copy()
    return f
m.embed_norm.register_forward_hook(hook("embed_norm"))
for i, b in enumerate(m.blocks): b.register_forward_hook(hook(f"blk{i}"))
m.norm.register_forward_hook(hook("norm"))

lines = [l.rstrip("\n") for l in open(REF, encoding="utf-8")
         if l.strip() and not l.startswith("#")][:N]

out = {"config": c}
sents = []
for idx, line in enumerate(lines):
    bopo, exp = line.split("\t")[:2]
    sy = bopo.split()
    sid = torch.tensor([[sv.get(y, 1) for y in sy]])
    am = torch.ones_like(sid, dtype=torch.bool)
    acts.clear()
    with torch.no_grad():
        lg = m(sid, am)[0].numpy()   # [T, n_char]
    rec = {"idx": idx, "bopo": bopo, "exp": exp,
           "syl_ids": sid[0].tolist(),
           "argmax": lg.argmax(-1).tolist(),
           "T": int(sid.shape[1])}
    sents.append(rec)
    np.savez(f"golden_sent{idx}.npz", syl_ids=sid[0].numpy().astype(np.int32),
             logits=lg.astype(np.float32),
             **({k: v[0] for k, v in acts.items()} if idx == 0 else {}))
    print(f"sent{idx} T={rec['T']} exp={exp!r} argmax0={rec['argmax'][:6]}")

json.dump({"sents": sents}, open("golden.json", "w"), ensure_ascii=False, indent=1)
# also dump embed + head weights already in GGUF; here just report shapes for sent0
d = np.load("golden_sent0.npz")
print("sent0 arrays:", {k: d[k].shape for k in d.files})
print("OK — golden_sent{0..%d}.npz + golden.json written" % (len(lines) - 1))
