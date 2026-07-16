#!/usr/bin/env python3
"""Convert a pure-GDN TinyGPT predictor (fla GatedDeltaNet, expand_v=1 => square
heads) to a llama.cpp qwen3next GGUF. The hard part is FUSING fla's separate
projections into qwen3next's fused, HEAD-INTERLEAVED tensors:

  SSM_IN         = per-head concat [q(hd) | k(hd) | v(hd) | z/gate(hd)]  (from q/k/v/g_proj)
  SSM_BETA_ALPHA = per-head concat [b(1) | a(1)]                        (from b/a_proj)
  SSM_CONV       = concat conv weights [q_conv | k_conv | v_conv]
  SSM_DT         = dt_bias ;  SSM_A_NOSCAN = A_log
  SSM_NORM       = o_norm  ;  SSM_OUT = o_proj
hparams: ssm_d_state = head_dim ; ssm_n_group = num_k_heads ;
         ssm_dt_rank = num_v_heads ; ssm_conv_kernel = d_conv.
The MLP/attn layers (this is pure-GDN so all layers are GDN + FFN) map like qwen3.

  PYTHONPATH=/tmp/llcpp/gguf-py python3 gdn_to_gguf.py --model pred_gdn_sq --out predictor_gdn.gguf
"""
import argparse, json, numpy as np, torch
import gguf


def interleave_heads(mats, nh, hd_list):
    """mats[i]: (nh*hd_i, dim). Return (sum(hd)*nh, dim) laid out per-head:
    head h -> rows [m0_h | m1_h | ...]."""
    dim = mats[0].shape[1]
    out = []
    for h in range(nh):
        for m, hd in zip(mats, hd_list):
            out.append(m[h * hd:(h + 1) * hd])
    return torch.cat(out, 0)                       # (sum(hd)*nh, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="predictor_gdn.gguf")
    args = ap.parse_args()
    ck = torch.load(f"{args.model}/predictor.pt", map_location="cpu")
    c = ck["config"]; sd = ck["model"]
    dim, depth, ffn = c["dim"], c["depth"], c["ffn"]
    tokj = json.load(open(f"{args.model}/predictor_tok.json", encoding="utf-8"))
    vmap = tokj["model"]["vocab"]; merges = tokj["model"]["merges"]
    id2 = {v: k for k, v in vmap.items()}; V = len(vmap)

    # infer GDN head geometry from tensor shapes (expand_v=1 => square)
    q = sd["blocks.0.mix.layer.q_proj.weight"]; a = sd["blocks.0.mix.layer.a_proj.weight"]
    nh = a.shape[0]                                 # num heads (= a_proj out)
    hd = q.shape[0] // nh                           # head_dim (square: k=v=hd)
    dconv = sd["blocks.0.mix.layer.q_conv1d.weight"].shape[-1]

    w = gguf.GGUFWriter(args.out, "qwen35")
    w.add_context_length(2048); w.add_embedding_length(dim); w.add_block_count(depth)
    w.add_feed_forward_length(ffn); w.add_layer_norm_rms_eps(1e-6)
    w.add_ssm_state_size(hd); w.add_ssm_conv_kernel(dconv)
    w.add_ssm_group_count(nh); w.add_ssm_time_step_rank(nh)   # num_k=num_v=nh (expand_v=1)
    w.add_ssm_inner_size(hd * nh)
    hdim = dim // (dim // 128 if dim >= 128 else 1)   # attn head_dim not used (pure GDN)
    w.add_rope_dimension_count(128); w.add_rope_freq_base(10000.0)
    try: w.add_rope_dimension_sections([0, 0, 0, 0])   # standard RoPE (no MRoPE)
    except Exception: w.add_array("qwen35.rope.dimension_sections", [0, 0, 0, 0])
    w.add_head_count(nh); w.add_head_count_kv(nh)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    # tokenizer
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default")
    w.add_token_list([id2[i] for i in range(V)])
    w.add_token_types([gguf.TokenType.NORMAL] * V)
    w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in merges])
    w.add_bos_token_id(vmap.get("<bos>", 1)); w.add_eos_token_id(vmap.get("<eos>", 2))

    def T(name, arr): w.add_tensor(name, arr.float().numpy().astype(np.float32))
    T("token_embd.weight", sd["embed.weight"]); T("output_norm.weight", sd["norm.w"])
    T("output.weight", sd["head.weight"])
    for i in range(depth):
        p = f"blocks.{i}."; m = p + "mix.layer."
        T(f"blk.{i}.attn_norm.weight", sd[p + "n1.w"])
        # fused, head-interleaved SSM_IN = [q|k|v|z] per head
        T(f"blk.{i}.ssm_in.weight",
          interleave_heads([sd[m + "q_proj.weight"], sd[m + "k_proj.weight"],
                            sd[m + "v_proj.weight"], sd[m + "g_proj.weight"]], nh, [hd, hd, hd, hd]))
        # SSM_BETA_ALPHA = [b|a] per head (a/b_proj out = nh, so hd=1)
        T(f"blk.{i}.ssm_beta_alpha.weight",
          interleave_heads([sd[m + "b_proj.weight"], sd[m + "a_proj.weight"]], nh, [1, 1]))
        # conv weights concat [q|k|v] (each (dim,1,dconv) -> (dim,dconv))
        conv = torch.cat([sd[m + "q_conv1d.weight"][:, 0], sd[m + "k_conv1d.weight"][:, 0],
                          sd[m + "v_conv1d.weight"][:, 0]], 0)
        T(f"blk.{i}.ssm_conv1d.weight", conv)
        T(f"blk.{i}.ssm_dt.weight", sd[m + "dt_bias"])
        T(f"blk.{i}.ssm_a.weight", sd[m + "A_log"])
        T(f"blk.{i}.ssm_norm.weight", sd[m + "o_norm.weight"])
        T(f"blk.{i}.ssm_out.weight", sd[m + "o_proj.weight"])
        T(f"blk.{i}.ffn_norm.weight", sd[p + "n2.w"])
        T(f"blk.{i}.ffn_gate.weight", sd[p + "w1.weight"])
        T(f"blk.{i}.ffn_up.weight", sd[p + "w3.weight"])
        T(f"blk.{i}.ffn_down.weight", sd[p + "w2.weight"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}: qwen3next {depth}L dim{dim} heads{nh} hd{hd} conv{dconv} vocab{V}")


if __name__ == "__main__":
    main()
