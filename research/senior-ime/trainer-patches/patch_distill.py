import re, io, sys
p = "train_slothe_ternary.py"
s = open(p, encoding="utf-8").read()
if "def load_teacher" in s:
    print("already patched"); sys.exit(0)
funcs = '''
def load_teacher(path, dev):
    """Load a frozen all-fp SlothE_T teacher (KD). Mirrors gate build()."""
    ck = torch.load(os.path.join(path, "slothe.pt"), map_location="cpu")
    c = ck["config"]
    m = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"],
                 c["kv"], c["ffn"], embed_norm=c.get("embed_norm", False),
                 weight_bits=c.get("weight_bits") or "fp",
                 weight_quant=c.get("weight_quant", "median"),
                 fp_boundary=c.get("fp_boundary", 1),
                 pre_norm=c.get("pre_norm", True), act_bits=c.get("act_bits", 8),
                 char_hints=c.get("char_hints", False),
                 tie_hints=c.get("tie_hints", False))
    m.load_state_dict(ck["model"])
    m.set_quant_alpha(1.0)
    m.eval().to(dev)
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m


def distill_loss(logits, t_logits, target, alpha, temp):
    """(1-alpha)*CE + alpha*KL(student||teacher) at temperature `temp`."""
    V = logits.shape[-1]
    flat = target.reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), flat, ignore_index=-100)
    keep = flat != -100
    s = logits.reshape(-1, V)[keep]
    t = t_logits.reshape(-1, V)[keep]
    kl = F.kl_div(F.log_softmax(s / temp, -1), F.softmax(t / temp, -1),
                  reduction="batchmean") * (temp * temp)
    return (1.0 - alpha) * ce + alpha * kl, ce.detach(), kl.detach()


'''
anchor = "def build_typo_neighbors(syl_vocab):"
assert anchor in s, "anchor not found"
s = s.replace(anchor, funcs.lstrip("\n") + "\n" + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched: added load_teacher + distill_loss before build_typo_neighbors")
