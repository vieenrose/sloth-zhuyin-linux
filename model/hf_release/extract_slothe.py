#!/usr/bin/env python3
"""Step 1 - extraction (run ON ai-workstation, needs torch).

Loads the ternary IME checkpoint and emits plain-numpy artifacts:
  - slothe_tensors.npz : per-tensor fp32 arrays.
        * ternary linears (blocks 1..14, q/o/k/v/w1/w2/w3): store `eff` (effective fp32 = code*scale)
        * everything else: raw fp32
        keys = original checkpoint tensor names, minus the *.quant_alpha scalars (dropped)
  - slothe_config.json : the config dict
  - roles.json         : tensor name -> "ternary" | "fp"
  - model.safetensors  : RAW fp32 state_dict (all tensors as-is), minus *.quant_alpha (HF-native master)
"""
import json, sys
import numpy as np
import torch
from safetensors.torch import save_file

CKPT = "/home/luigi/sloth-zhuyin-linux/model/slothe_t_25m_ce_ls32_ep24/slothe.pt"
OUT_NPZ = "/home/luigi/slothe_tensors.npz"
OUT_CFG = "/home/luigi/slothe_config.json"
OUT_ROLES = "/home/luigi/roles.json"
OUT_ST = "/home/luigi/model.safetensors"

FP_BOUNDARY = 1
DEPTH = 16
# the 7 linear leaf names inside a block that carry weights
LINEAR_SUFFIXES = ["attn.q", "attn.o", "attn.k", "attn.v", "ffn.w1", "ffn.w2", "ffn.w3"]


def is_ternary_linear(name):
    """True iff `name` is a `blocks.i.<suffix>.weight` on a ternary block (1..DEPTH-2)."""
    if not name.startswith("blocks.") or not name.endswith(".weight"):
        return False
    parts = name.split(".")
    try:
        i = int(parts[1])
    except ValueError:
        return False
    if i < FP_BOUNDARY or i >= DEPTH - FP_BOUNDARY:
        return False  # fp island block (0 and 15)
    core = ".".join(parts[2:-1])  # strip 'blocks.i.' prefix and '.weight' suffix
    return core in LINEAR_SUFFIXES


def ternarize_eff(w):
    """Effective fp32 weight per verified trainer math (quant_alpha == 1.0)."""
    scale = w.abs().median(dim=1, keepdim=True).values.clamp(min=1e-5)  # [out,1] per output channel
    code = (w / scale).round().clamp(-1, 1)  # {-1,0,+1}
    eff = code * scale
    return eff, code, scale


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    config = ck["config"]

    npz = {}
    roles = {}
    st_tensors = {}  # raw fp32 master
    sanity_done = False

    for name, t in sd.items():
        if name.endswith(".quant_alpha"):
            continue  # drop the (all-1.0) quant scalars
        t = t.detach().to(torch.float32)
        # raw master (safetensors) keeps every non-quant_alpha tensor as-is
        st_tensors[name] = t.contiguous().clone()

        if is_ternary_linear(name):
            eff, code, scale = ternarize_eff(t)
            npz[name] = eff.numpy().astype(np.float32)
            roles[name] = "ternary"
            if not sanity_done:
                uniq = set(int(x) for x in torch.unique(code).tolist())
                assert uniq.issubset({-1, 0, 1}), f"code has bad values: {uniq}"
                zero_frac = float((code == 0).float().mean())
                print(f"[sanity] {name}: code.unique()={sorted(uniq)} subset of {{-1,0,1}} OK; "
                      f"zero-fraction={zero_frac:.4f} (expected ~0.27)")
                sanity_done = True
        else:
            npz[name] = t.numpy().astype(np.float32)
            roles[name] = "fp"

    np.savez(OUT_NPZ, **npz)
    with open(OUT_CFG, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(OUT_ROLES, "w") as f:
        json.dump(roles, f, indent=2, ensure_ascii=False)
    save_file(st_tensors, OUT_ST)

    n_tern = sum(1 for v in roles.values() if v == "ternary")
    n_fp = sum(1 for v in roles.values() if v == "fp")
    print(f"[done] stored {len(npz)} tensors: {n_tern} ternary, {n_fp} fp")
    print(f"[done] npz={OUT_NPZ} cfg={OUT_CFG} roles={OUT_ROLES} safetensors={OUT_ST}")


if __name__ == "__main__":
    main()
