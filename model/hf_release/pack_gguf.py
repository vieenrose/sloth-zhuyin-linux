#!/usr/bin/env python3
"""Step 2 - GGUF packing (run LOCALLY, numpy + gguf-py only).

Reads slothe_tensors.npz / roles.json / slothe_config.json / syl_vocab.json and
writes slothe-t-25m.gguf:
  - fp tensors  -> F16
  - ternary linears -> in-features padded to next multiple of 256, quantized to TQ2_0 (lossless on exact-ternary)
"""
import json, os, sys

GGUF_PY = "/tmp/sloth.cpp/gguf-py"
sys.path.insert(0, GGUF_PY)

import numpy as np
import gguf
from gguf import GGUFWriter, GGUFReader, quants
from gguf.constants import GGMLQuantizationType

HF = "/home/luigi/sloth-zhuyin-linux/model/hf_release"
NPZ = os.path.join(HF, "slothe_tensors.npz")
ROLES = os.path.join(HF, "roles.json")
CFG = os.path.join(HF, "slothe_config.json")
SYL = os.path.join(HF, "syl_vocab.json")
OUT = os.path.join(HF, "slothe-t-25m.gguf")
NAMES_MD = os.path.join(HF, "NAMES.md")

PAD_TO = 256
ARCH = "slothe"


def next_mult(n, m):
    return ((n + m - 1) // m) * m


def ggml_name(name):
    """Map original checkpoint tensor name -> ggml tensor name."""
    top = {
        "embed.weight": "token_embd.weight",
        "embed_norm.w": "token_embd_norm.weight",
        "norm.w": "output_norm.weight",
        "head.weight": "output.weight",
    }
    if name in top:
        return top[name]
    assert name.startswith("blocks."), name
    parts = name.split(".")
    i = int(parts[1])
    rest = ".".join(parts[2:])
    per_block = {
        "n1.w": "attn_norm.weight",
        "n2.w": "ffn_norm.weight",
        "attn.qn.w": "attn_q_rmsnorm.weight",
        "attn.kn.w": "attn_k_rmsnorm.weight",
        "attn.q.weight": "attn_q.weight",
        "attn.q.pre.w": "attn_q_subln.weight",
        "attn.o.weight": "attn_output.weight",
        "attn.o.pre.w": "attn_output_subln.weight",
        "attn.k.weight": "attn_k.weight",
        "attn.k.pre.w": "attn_k_subln.weight",
        "attn.v.weight": "attn_v.weight",
        "attn.v.pre.w": "attn_v_subln.weight",
        "ffn.w1.weight": "ffn_gate.weight",
        "ffn.w1.pre.w": "ffn_gate_subln.weight",
        "ffn.w3.weight": "ffn_up.weight",
        "ffn.w3.pre.w": "ffn_up_subln.weight",
        "ffn.w2.weight": "ffn_down.weight",
        "ffn.w2.pre.w": "ffn_down_subln.weight",
    }
    assert rest in per_block, f"unmapped: {name}"
    return f"blk.{i}.{per_block[rest]}"


def main():
    tensors = dict(np.load(NPZ))
    roles = json.load(open(ROLES))
    cfg = json.load(open(CFG))
    syl = json.load(open(SYL))

    # syllable vocab in index order (handle dict token->id or list)
    if isinstance(syl, dict):
        syl_tokens = [None] * len(syl)
        for tok, idx in syl.items():
            syl_tokens[int(idx)] = tok
        assert all(t is not None for t in syl_tokens), "gap in syl vocab ids"
    else:
        syl_tokens = list(syl)
    assert len(syl_tokens) == cfg["n_syl"], (len(syl_tokens), cfg["n_syl"])

    writer = GGUFWriter(OUT, ARCH)
    writer.add_architecture()
    writer.add_name("slothe-t-25m-zhuyin")
    writer.add_uint32("slothe.context_length", 512)
    writer.add_uint32("slothe.embedding_length", cfg["dim"])           # 352
    writer.add_uint32("slothe.block_count", cfg["depth"])              # 16
    writer.add_uint32("slothe.feed_forward_length", cfg["ffn"])        # 960
    writer.add_uint32("slothe.attention.head_count", cfg["heads"])     # 8
    writer.add_uint32("slothe.attention.head_count_kv", cfg["kv"])     # 2
    writer.add_uint32("slothe.attention.key_length", 44)
    writer.add_uint32("slothe.attention.value_length", 44)
    writer.add_float32("slothe.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_uint32("slothe.fp_boundary", cfg["fp_boundary"])        # 1
    writer.add_uint32("slothe.vocab_syl", cfg["n_syl"])                # 1539
    writer.add_uint32("slothe.vocab_char", cfg["n_char"])              # 8342
    writer.add_uint32("slothe.ternary_pad_to", PAD_TO)                 # 256
    writer.add_array("slothe.syl_tokens", syl_tokens)

    # deterministic order: embed/norm/head first, then blocks
    order = sorted(tensors.keys(), key=lambda n: (not n.startswith("embed"),
                                                  not n.endswith("head.weight") and n in ("norm.w",),
                                                  n))
    # simpler explicit order for readability of the reader table
    def sort_key(n):
        pri = {"embed.weight": 0, "embed_norm.w": 1, "norm.w": 2, "head.weight": 3}
        if n in pri:
            return (0, pri[n], "")
        parts = n.split(".")
        return (1, int(parts[1]), n)
    order = sorted(tensors.keys(), key=sort_key)

    names_rows = []          # (ggml_name, orig_name, role, orig_shape, stored_dtype, padded_in)
    packed_cache = {}        # ggml_name -> (packed, eff_padded, orig_shape) for validation
    for name in order:
        arr = np.asarray(tensors[name], dtype=np.float32)
        gname = ggml_name(name)
        role = roles[name]
        orig_shape = tuple(int(x) for x in arr.shape)
        if role == "fp":
            writer.add_tensor(gname, arr.astype(np.float16))
            names_rows.append((gname, name, role, orig_shape, "F16", ""))
        else:  # ternary linear -> pad in-features -> TQ2_0
            out_f, in_f = arr.shape
            pad_in = next_mult(in_f, PAD_TO)
            padded = np.zeros((out_f, pad_in), dtype=np.float32)
            padded[:, :in_f] = arr
            packed = quants.quantize(padded, GGMLQuantizationType.TQ2_0)
            writer.add_tensor(gname, packed, raw_dtype=GGMLQuantizationType.TQ2_0)
            names_rows.append((gname, name, role, orig_shape, "TQ2_0", pad_in))
            packed_cache[gname] = (packed, padded, orig_shape, in_f)

    # ---- VALIDATION 1: round-trip on 2 sample ternary tensors (pre-write) ----
    print("== VALIDATION 1: TQ2_0 round-trip on exact-ternary weights ==")
    samples = [g for g in ("blk.1.attn_q.weight", "blk.7.ffn_down.weight") if g in packed_cache]
    for g in samples:
        packed, padded, orig_shape, in_f = packed_cache[g]
        deq = quants.dequantize(packed, GGMLQuantizationType.TQ2_0).reshape(padded.shape)
        deq_stripped = deq[:, :in_f]
        eff = padded[:, :in_f]
        maxerr = float(np.abs(deq_stripped - eff).max())
        print(f"  {g}: orig={orig_shape} padded={padded.shape} maxerr={maxerr:.3e}")
        if maxerr >= 1e-3:
            print(f"  FATAL: {g} round-trip maxerr {maxerr} >= 1e-3 -- lossy TQ2_0, aborting")
            sys.exit(1)
    print("  round-trip OK (lossless).")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ---- NAMES.md ----
    with open(NAMES_MD, "w") as f:
        f.write("# slothe GGUF tensor name map\n\n")
        f.write(f"Architecture: `{ARCH}`  |  ternary in-features padded to multiple of {PAD_TO} "
                f"(recorded as `slothe.ternary_pad_to`).\n\n")
        f.write("Ternary linears (blocks 1..14 q/o/k/v/w1/w2/w3) are stored TQ2_0 with the "
                "in-features zero-padded to the next multiple of 256 (352->512, 960->1024). "
                "Strip padding using the original in-features listed below. All other tensors are F16.\n\n")
        f.write("| ggml name | original name | role | original shape | stored dtype | padded in-features |\n")
        f.write("|---|---|---|---|---|---|\n")
        for gname, oname, role, oshape, dt, pin in names_rows:
            f.write(f"| `{gname}` | `{oname}` | {role} | {oshape} | {dt} | {pin} |\n")

    # ---- VALIDATION 2: reload GGUF and print reader table ----
    print("\n== VALIDATION 2: GGUFReader reload ==")
    r = GGUFReader(OUT)
    arch_field = r.fields.get("general.architecture")
    arch_val = None
    if arch_field is not None:
        arch_val = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    print(f"  architecture: {arch_val}")
    print(f"  tensor count: {len(r.tensors)}")
    print(f"  {'name':40s} {'shape':20s} {'ggml_type'}")
    for t in r.tensors[:12]:
        shp = "x".join(str(int(d)) for d in t.shape)
        print(f"  {t.name:40s} {shp:20s} {t.tensor_type.name}")

    # ---- VALIDATION 3: file size ----
    sz = os.path.getsize(OUT)
    print(f"\n== VALIDATION 3: file size ==")
    print(f"  {OUT}: {sz} bytes ({sz/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    main()
