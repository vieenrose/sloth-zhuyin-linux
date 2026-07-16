#!/usr/bin/env python3
"""Convert an attention-only TinyGPT predictor (predictor.pt + predictor_tok.json)
to a llama.cpp GGUF using the qwen3 architecture (RoPE + RMSNorm + SwiGLU + GQA +
per-head-dim q/k norm — all of which TinyGPT's attention block already has). Then
llama.cpp can load, quantize (Q4_K), and run it on-device.

  PYTHONPATH=/tmp/llcpp/gguf-py python3 tinygpt_to_gguf.py --model pred2_gpt2 --out predictor.gguf
"""
import argparse, json, torch, numpy as np
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="predictor.gguf")
    args = ap.parse_args()

    ck = torch.load(f"{args.model}/predictor.pt", map_location="cpu")
    c = ck["config"]; sd = ck["model"]
    dim, depth, heads, kv, ffn = c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"]
    hd = dim // heads
    tok = json.load(open(f"{args.model}/predictor_tok.json", encoding="utf-8"))
    vocab = tok["model"]["vocab"]                       # token -> id
    merges = tok["model"]["merges"]
    id2tok = {v: k for k, v in vocab.items()}
    V = len(vocab)

    w = gguf.GGUFWriter(args.out, "qwen3")
    w.add_context_length(2048); w.add_embedding_length(dim)
    w.add_block_count(depth); w.add_feed_forward_length(ffn)
    w.add_head_count(heads); w.add_head_count_kv(kv)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0); w.add_rope_dimension_count(hd)
    w.add_key_length(hd); w.add_value_length(hd)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # --- tokenizer (gpt2 byte-level BPE) ---
    toks = [id2tok[i] for i in range(V)]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types([gguf.TokenType.NORMAL] * V)
    w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
    w.add_bos_token_id(vocab.get("<bos>", 1)); w.add_eos_token_id(vocab.get("<eos>", 2))
    w.add_unk_token_id(vocab.get("<unk>", 3)); w.add_pad_token_id(vocab.get("<pad>", 0))

    def t(name, arr): w.add_tensor(name, arr.float().numpy().astype(np.float32))
    t("token_embd.weight", sd["embed.weight"])
    t("output_norm.weight", sd["norm.w"])
    t("output.weight", sd["head.weight"])              # tied but written explicitly
    for i in range(depth):
        p = f"blocks.{i}."
        t(f"blk.{i}.attn_norm.weight",   sd[p + "n1.w"])
        t(f"blk.{i}.attn_q.weight",      sd[p + "mix.q.weight"])
        t(f"blk.{i}.attn_k.weight",      sd[p + "mix.k.weight"])
        t(f"blk.{i}.attn_v.weight",      sd[p + "mix.v.weight"])
        t(f"blk.{i}.attn_output.weight", sd[p + "mix.o.weight"])
        t(f"blk.{i}.attn_q_norm.weight", sd[p + "mix.qn.w"])
        t(f"blk.{i}.attn_k_norm.weight", sd[p + "mix.kn.w"])
        t(f"blk.{i}.ffn_norm.weight",    sd[p + "n2.w"])
        t(f"blk.{i}.ffn_gate.weight",    sd[p + "w1.weight"])
        t(f"blk.{i}.ffn_up.weight",      sd[p + "w3.weight"])
        t(f"blk.{i}.ffn_down.weight",    sd[p + "w2.weight"])

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}: qwen3 arch, {depth}L dim{dim} heads{heads}/{kv} ffn{ffn} vocab{V}")


if __name__ == "__main__":
    main()
