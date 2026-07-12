# slothe GGUF tensor name map

Architecture: `slothe`  |  ternary in-features padded to multiple of 256 (recorded as `slothe.ternary_pad_to`).

Ternary linears (blocks 1..14 q/o/k/v/w1/w2/w3) are stored TQ2_0 with the in-features zero-padded to the next multiple of 256 (352->512, 960->1024). Strip padding using the original in-features listed below. All other tensors are F16.

| ggml name | original name | role | original shape | stored dtype | padded in-features |
|---|---|---|---|---|---|
| `token_embd.weight` | `embed.weight` | fp | (1539, 352) | F16 |  |
| `token_embd_norm.weight` | `embed_norm.w` | fp | (352,) | F16 |  |
| `output_norm.weight` | `norm.w` | fp | (352,) | F16 |  |
| `output.weight` | `head.weight` | fp | (8342, 352) | F16 |  |
| `blk.0.attn_k.weight` | `blocks.0.attn.k.weight` | fp | (88, 352) | F16 |  |
| `blk.0.attn_k_rmsnorm.weight` | `blocks.0.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.0.attn_output.weight` | `blocks.0.attn.o.weight` | fp | (352, 352) | F16 |  |
| `blk.0.attn_q.weight` | `blocks.0.attn.q.weight` | fp | (352, 352) | F16 |  |
| `blk.0.attn_q_rmsnorm.weight` | `blocks.0.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.0.attn_v.weight` | `blocks.0.attn.v.weight` | fp | (88, 352) | F16 |  |
| `blk.0.ffn_gate.weight` | `blocks.0.ffn.w1.weight` | fp | (960, 352) | F16 |  |
| `blk.0.ffn_down.weight` | `blocks.0.ffn.w2.weight` | fp | (352, 960) | F16 |  |
| `blk.0.ffn_up.weight` | `blocks.0.ffn.w3.weight` | fp | (960, 352) | F16 |  |
| `blk.0.attn_norm.weight` | `blocks.0.n1.w` | fp | (352,) | F16 |  |
| `blk.0.ffn_norm.weight` | `blocks.0.n2.w` | fp | (352,) | F16 |  |
| `blk.1.attn_k_subln.weight` | `blocks.1.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.1.attn_k.weight` | `blocks.1.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.1.attn_k_rmsnorm.weight` | `blocks.1.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.1.attn_output_subln.weight` | `blocks.1.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.1.attn_output.weight` | `blocks.1.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.1.attn_q_subln.weight` | `blocks.1.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.1.attn_q.weight` | `blocks.1.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.1.attn_q_rmsnorm.weight` | `blocks.1.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.1.attn_v_subln.weight` | `blocks.1.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.1.attn_v.weight` | `blocks.1.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.1.ffn_gate_subln.weight` | `blocks.1.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.1.ffn_gate.weight` | `blocks.1.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.1.ffn_down_subln.weight` | `blocks.1.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.1.ffn_down.weight` | `blocks.1.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.1.ffn_up_subln.weight` | `blocks.1.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.1.ffn_up.weight` | `blocks.1.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.1.attn_norm.weight` | `blocks.1.n1.w` | fp | (352,) | F16 |  |
| `blk.1.ffn_norm.weight` | `blocks.1.n2.w` | fp | (352,) | F16 |  |
| `blk.2.attn_k_subln.weight` | `blocks.2.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.2.attn_k.weight` | `blocks.2.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.2.attn_k_rmsnorm.weight` | `blocks.2.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.2.attn_output_subln.weight` | `blocks.2.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.2.attn_output.weight` | `blocks.2.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.2.attn_q_subln.weight` | `blocks.2.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.2.attn_q.weight` | `blocks.2.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.2.attn_q_rmsnorm.weight` | `blocks.2.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.2.attn_v_subln.weight` | `blocks.2.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.2.attn_v.weight` | `blocks.2.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.2.ffn_gate_subln.weight` | `blocks.2.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.2.ffn_gate.weight` | `blocks.2.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.2.ffn_down_subln.weight` | `blocks.2.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.2.ffn_down.weight` | `blocks.2.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.2.ffn_up_subln.weight` | `blocks.2.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.2.ffn_up.weight` | `blocks.2.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.2.attn_norm.weight` | `blocks.2.n1.w` | fp | (352,) | F16 |  |
| `blk.2.ffn_norm.weight` | `blocks.2.n2.w` | fp | (352,) | F16 |  |
| `blk.3.attn_k_subln.weight` | `blocks.3.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.3.attn_k.weight` | `blocks.3.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.3.attn_k_rmsnorm.weight` | `blocks.3.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.3.attn_output_subln.weight` | `blocks.3.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.3.attn_output.weight` | `blocks.3.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.3.attn_q_subln.weight` | `blocks.3.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.3.attn_q.weight` | `blocks.3.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.3.attn_q_rmsnorm.weight` | `blocks.3.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.3.attn_v_subln.weight` | `blocks.3.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.3.attn_v.weight` | `blocks.3.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.3.ffn_gate_subln.weight` | `blocks.3.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.3.ffn_gate.weight` | `blocks.3.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.3.ffn_down_subln.weight` | `blocks.3.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.3.ffn_down.weight` | `blocks.3.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.3.ffn_up_subln.weight` | `blocks.3.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.3.ffn_up.weight` | `blocks.3.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.3.attn_norm.weight` | `blocks.3.n1.w` | fp | (352,) | F16 |  |
| `blk.3.ffn_norm.weight` | `blocks.3.n2.w` | fp | (352,) | F16 |  |
| `blk.4.attn_k_subln.weight` | `blocks.4.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.4.attn_k.weight` | `blocks.4.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.4.attn_k_rmsnorm.weight` | `blocks.4.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.4.attn_output_subln.weight` | `blocks.4.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.4.attn_output.weight` | `blocks.4.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.4.attn_q_subln.weight` | `blocks.4.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.4.attn_q.weight` | `blocks.4.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.4.attn_q_rmsnorm.weight` | `blocks.4.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.4.attn_v_subln.weight` | `blocks.4.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.4.attn_v.weight` | `blocks.4.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.4.ffn_gate_subln.weight` | `blocks.4.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.4.ffn_gate.weight` | `blocks.4.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.4.ffn_down_subln.weight` | `blocks.4.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.4.ffn_down.weight` | `blocks.4.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.4.ffn_up_subln.weight` | `blocks.4.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.4.ffn_up.weight` | `blocks.4.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.4.attn_norm.weight` | `blocks.4.n1.w` | fp | (352,) | F16 |  |
| `blk.4.ffn_norm.weight` | `blocks.4.n2.w` | fp | (352,) | F16 |  |
| `blk.5.attn_k_subln.weight` | `blocks.5.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.5.attn_k.weight` | `blocks.5.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.5.attn_k_rmsnorm.weight` | `blocks.5.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.5.attn_output_subln.weight` | `blocks.5.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.5.attn_output.weight` | `blocks.5.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.5.attn_q_subln.weight` | `blocks.5.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.5.attn_q.weight` | `blocks.5.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.5.attn_q_rmsnorm.weight` | `blocks.5.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.5.attn_v_subln.weight` | `blocks.5.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.5.attn_v.weight` | `blocks.5.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.5.ffn_gate_subln.weight` | `blocks.5.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.5.ffn_gate.weight` | `blocks.5.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.5.ffn_down_subln.weight` | `blocks.5.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.5.ffn_down.weight` | `blocks.5.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.5.ffn_up_subln.weight` | `blocks.5.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.5.ffn_up.weight` | `blocks.5.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.5.attn_norm.weight` | `blocks.5.n1.w` | fp | (352,) | F16 |  |
| `blk.5.ffn_norm.weight` | `blocks.5.n2.w` | fp | (352,) | F16 |  |
| `blk.6.attn_k_subln.weight` | `blocks.6.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.6.attn_k.weight` | `blocks.6.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.6.attn_k_rmsnorm.weight` | `blocks.6.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.6.attn_output_subln.weight` | `blocks.6.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.6.attn_output.weight` | `blocks.6.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.6.attn_q_subln.weight` | `blocks.6.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.6.attn_q.weight` | `blocks.6.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.6.attn_q_rmsnorm.weight` | `blocks.6.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.6.attn_v_subln.weight` | `blocks.6.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.6.attn_v.weight` | `blocks.6.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.6.ffn_gate_subln.weight` | `blocks.6.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.6.ffn_gate.weight` | `blocks.6.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.6.ffn_down_subln.weight` | `blocks.6.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.6.ffn_down.weight` | `blocks.6.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.6.ffn_up_subln.weight` | `blocks.6.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.6.ffn_up.weight` | `blocks.6.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.6.attn_norm.weight` | `blocks.6.n1.w` | fp | (352,) | F16 |  |
| `blk.6.ffn_norm.weight` | `blocks.6.n2.w` | fp | (352,) | F16 |  |
| `blk.7.attn_k_subln.weight` | `blocks.7.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.7.attn_k.weight` | `blocks.7.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.7.attn_k_rmsnorm.weight` | `blocks.7.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.7.attn_output_subln.weight` | `blocks.7.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.7.attn_output.weight` | `blocks.7.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.7.attn_q_subln.weight` | `blocks.7.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.7.attn_q.weight` | `blocks.7.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.7.attn_q_rmsnorm.weight` | `blocks.7.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.7.attn_v_subln.weight` | `blocks.7.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.7.attn_v.weight` | `blocks.7.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.7.ffn_gate_subln.weight` | `blocks.7.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.7.ffn_gate.weight` | `blocks.7.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.7.ffn_down_subln.weight` | `blocks.7.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.7.ffn_down.weight` | `blocks.7.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.7.ffn_up_subln.weight` | `blocks.7.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.7.ffn_up.weight` | `blocks.7.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.7.attn_norm.weight` | `blocks.7.n1.w` | fp | (352,) | F16 |  |
| `blk.7.ffn_norm.weight` | `blocks.7.n2.w` | fp | (352,) | F16 |  |
| `blk.8.attn_k_subln.weight` | `blocks.8.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.8.attn_k.weight` | `blocks.8.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.8.attn_k_rmsnorm.weight` | `blocks.8.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.8.attn_output_subln.weight` | `blocks.8.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.8.attn_output.weight` | `blocks.8.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.8.attn_q_subln.weight` | `blocks.8.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.8.attn_q.weight` | `blocks.8.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.8.attn_q_rmsnorm.weight` | `blocks.8.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.8.attn_v_subln.weight` | `blocks.8.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.8.attn_v.weight` | `blocks.8.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.8.ffn_gate_subln.weight` | `blocks.8.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.8.ffn_gate.weight` | `blocks.8.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.8.ffn_down_subln.weight` | `blocks.8.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.8.ffn_down.weight` | `blocks.8.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.8.ffn_up_subln.weight` | `blocks.8.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.8.ffn_up.weight` | `blocks.8.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.8.attn_norm.weight` | `blocks.8.n1.w` | fp | (352,) | F16 |  |
| `blk.8.ffn_norm.weight` | `blocks.8.n2.w` | fp | (352,) | F16 |  |
| `blk.9.attn_k_subln.weight` | `blocks.9.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.9.attn_k.weight` | `blocks.9.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.9.attn_k_rmsnorm.weight` | `blocks.9.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.9.attn_output_subln.weight` | `blocks.9.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.9.attn_output.weight` | `blocks.9.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.9.attn_q_subln.weight` | `blocks.9.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.9.attn_q.weight` | `blocks.9.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.9.attn_q_rmsnorm.weight` | `blocks.9.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.9.attn_v_subln.weight` | `blocks.9.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.9.attn_v.weight` | `blocks.9.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.9.ffn_gate_subln.weight` | `blocks.9.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.9.ffn_gate.weight` | `blocks.9.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.9.ffn_down_subln.weight` | `blocks.9.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.9.ffn_down.weight` | `blocks.9.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.9.ffn_up_subln.weight` | `blocks.9.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.9.ffn_up.weight` | `blocks.9.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.9.attn_norm.weight` | `blocks.9.n1.w` | fp | (352,) | F16 |  |
| `blk.9.ffn_norm.weight` | `blocks.9.n2.w` | fp | (352,) | F16 |  |
| `blk.10.attn_k_subln.weight` | `blocks.10.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.10.attn_k.weight` | `blocks.10.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.10.attn_k_rmsnorm.weight` | `blocks.10.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.10.attn_output_subln.weight` | `blocks.10.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.10.attn_output.weight` | `blocks.10.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.10.attn_q_subln.weight` | `blocks.10.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.10.attn_q.weight` | `blocks.10.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.10.attn_q_rmsnorm.weight` | `blocks.10.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.10.attn_v_subln.weight` | `blocks.10.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.10.attn_v.weight` | `blocks.10.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.10.ffn_gate_subln.weight` | `blocks.10.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.10.ffn_gate.weight` | `blocks.10.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.10.ffn_down_subln.weight` | `blocks.10.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.10.ffn_down.weight` | `blocks.10.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.10.ffn_up_subln.weight` | `blocks.10.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.10.ffn_up.weight` | `blocks.10.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.10.attn_norm.weight` | `blocks.10.n1.w` | fp | (352,) | F16 |  |
| `blk.10.ffn_norm.weight` | `blocks.10.n2.w` | fp | (352,) | F16 |  |
| `blk.11.attn_k_subln.weight` | `blocks.11.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.11.attn_k.weight` | `blocks.11.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.11.attn_k_rmsnorm.weight` | `blocks.11.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.11.attn_output_subln.weight` | `blocks.11.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.11.attn_output.weight` | `blocks.11.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.11.attn_q_subln.weight` | `blocks.11.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.11.attn_q.weight` | `blocks.11.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.11.attn_q_rmsnorm.weight` | `blocks.11.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.11.attn_v_subln.weight` | `blocks.11.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.11.attn_v.weight` | `blocks.11.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.11.ffn_gate_subln.weight` | `blocks.11.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.11.ffn_gate.weight` | `blocks.11.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.11.ffn_down_subln.weight` | `blocks.11.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.11.ffn_down.weight` | `blocks.11.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.11.ffn_up_subln.weight` | `blocks.11.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.11.ffn_up.weight` | `blocks.11.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.11.attn_norm.weight` | `blocks.11.n1.w` | fp | (352,) | F16 |  |
| `blk.11.ffn_norm.weight` | `blocks.11.n2.w` | fp | (352,) | F16 |  |
| `blk.12.attn_k_subln.weight` | `blocks.12.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.12.attn_k.weight` | `blocks.12.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.12.attn_k_rmsnorm.weight` | `blocks.12.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.12.attn_output_subln.weight` | `blocks.12.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.12.attn_output.weight` | `blocks.12.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.12.attn_q_subln.weight` | `blocks.12.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.12.attn_q.weight` | `blocks.12.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.12.attn_q_rmsnorm.weight` | `blocks.12.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.12.attn_v_subln.weight` | `blocks.12.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.12.attn_v.weight` | `blocks.12.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.12.ffn_gate_subln.weight` | `blocks.12.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.12.ffn_gate.weight` | `blocks.12.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.12.ffn_down_subln.weight` | `blocks.12.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.12.ffn_down.weight` | `blocks.12.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.12.ffn_up_subln.weight` | `blocks.12.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.12.ffn_up.weight` | `blocks.12.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.12.attn_norm.weight` | `blocks.12.n1.w` | fp | (352,) | F16 |  |
| `blk.12.ffn_norm.weight` | `blocks.12.n2.w` | fp | (352,) | F16 |  |
| `blk.13.attn_k_subln.weight` | `blocks.13.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.13.attn_k.weight` | `blocks.13.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.13.attn_k_rmsnorm.weight` | `blocks.13.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.13.attn_output_subln.weight` | `blocks.13.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.13.attn_output.weight` | `blocks.13.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.13.attn_q_subln.weight` | `blocks.13.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.13.attn_q.weight` | `blocks.13.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.13.attn_q_rmsnorm.weight` | `blocks.13.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.13.attn_v_subln.weight` | `blocks.13.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.13.attn_v.weight` | `blocks.13.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.13.ffn_gate_subln.weight` | `blocks.13.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.13.ffn_gate.weight` | `blocks.13.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.13.ffn_down_subln.weight` | `blocks.13.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.13.ffn_down.weight` | `blocks.13.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.13.ffn_up_subln.weight` | `blocks.13.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.13.ffn_up.weight` | `blocks.13.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.13.attn_norm.weight` | `blocks.13.n1.w` | fp | (352,) | F16 |  |
| `blk.13.ffn_norm.weight` | `blocks.13.n2.w` | fp | (352,) | F16 |  |
| `blk.14.attn_k_subln.weight` | `blocks.14.attn.k.pre.w` | fp | (352,) | F16 |  |
| `blk.14.attn_k.weight` | `blocks.14.attn.k.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.14.attn_k_rmsnorm.weight` | `blocks.14.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.14.attn_output_subln.weight` | `blocks.14.attn.o.pre.w` | fp | (352,) | F16 |  |
| `blk.14.attn_output.weight` | `blocks.14.attn.o.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.14.attn_q_subln.weight` | `blocks.14.attn.q.pre.w` | fp | (352,) | F16 |  |
| `blk.14.attn_q.weight` | `blocks.14.attn.q.weight` | ternary | (352, 352) | TQ2_0 | 512 |
| `blk.14.attn_q_rmsnorm.weight` | `blocks.14.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.14.attn_v_subln.weight` | `blocks.14.attn.v.pre.w` | fp | (352,) | F16 |  |
| `blk.14.attn_v.weight` | `blocks.14.attn.v.weight` | ternary | (88, 352) | TQ2_0 | 512 |
| `blk.14.ffn_gate_subln.weight` | `blocks.14.ffn.w1.pre.w` | fp | (352,) | F16 |  |
| `blk.14.ffn_gate.weight` | `blocks.14.ffn.w1.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.14.ffn_down_subln.weight` | `blocks.14.ffn.w2.pre.w` | fp | (960,) | F16 |  |
| `blk.14.ffn_down.weight` | `blocks.14.ffn.w2.weight` | ternary | (352, 960) | TQ2_0 | 1024 |
| `blk.14.ffn_up_subln.weight` | `blocks.14.ffn.w3.pre.w` | fp | (352,) | F16 |  |
| `blk.14.ffn_up.weight` | `blocks.14.ffn.w3.weight` | ternary | (960, 352) | TQ2_0 | 512 |
| `blk.14.attn_norm.weight` | `blocks.14.n1.w` | fp | (352,) | F16 |  |
| `blk.14.ffn_norm.weight` | `blocks.14.n2.w` | fp | (352,) | F16 |  |
| `blk.15.attn_k.weight` | `blocks.15.attn.k.weight` | fp | (88, 352) | F16 |  |
| `blk.15.attn_k_rmsnorm.weight` | `blocks.15.attn.kn.w` | fp | (44,) | F16 |  |
| `blk.15.attn_output.weight` | `blocks.15.attn.o.weight` | fp | (352, 352) | F16 |  |
| `blk.15.attn_q.weight` | `blocks.15.attn.q.weight` | fp | (352, 352) | F16 |  |
| `blk.15.attn_q_rmsnorm.weight` | `blocks.15.attn.qn.w` | fp | (44,) | F16 |  |
| `blk.15.attn_v.weight` | `blocks.15.attn.v.weight` | fp | (88, 352) | F16 |  |
| `blk.15.ffn_gate.weight` | `blocks.15.ffn.w1.weight` | fp | (960, 352) | F16 |  |
| `blk.15.ffn_down.weight` | `blocks.15.ffn.w2.weight` | fp | (352, 960) | F16 |  |
| `blk.15.ffn_up.weight` | `blocks.15.ffn.w3.weight` | fp | (960, 352) | F16 |  |
| `blk.15.attn_norm.weight` | `blocks.15.n1.w` | fp | (352,) | F16 |  |
| `blk.15.ffn_norm.weight` | `blocks.15.n2.w` | fp | (352,) | F16 |  |
