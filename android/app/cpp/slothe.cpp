// slothe.cpp — standalone ggml forward pass for the "slothe" bidirectional IME encoder.
//
// Core API (can later become libslothe):
//   slothe_model * slothe_load(const char * gguf_path);
//   void           slothe_logits(slothe_model*, const int32_t* syl_ids, int T, float* out_logits); // [T*n_char]
//   void           slothe_free(slothe_model*);
//
// Plus a validation `main` that checks against the PyTorch golden reference.

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>

// ---------------- hyperparameters ----------------
struct slothe_hparams {
    int dim        = 352;
    int n_layer    = 16;
    int n_head     = 8;
    int n_head_kv  = 2;
    int head_dim   = 44;
    int ffn        = 960;
    int n_syl      = 1539;
    int n_char     = 8342;
    int fp_boundary= 1;   // blocks 0 and (n_layer-1) are fp
    float rms_eps  = 1e-6f;
    float rope_base= 10000.0f;
    int pad_to     = 256; // ternary in-features padded to multiple of this
};

struct slothe_model {
    slothe_hparams hp;
    ggml_context * ctx_w = nullptr;    // holds weight tensor metadata
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // compute-graph cache — reused across slothe_logits() calls of the same T so the
    // hot conversion path avoids a fresh 64MB ggml_init + graph rebuild + allocator
    // alloc/free every keystroke. Rebuilt only when the sequence length changes.
    ggml_context * ctx_c   = nullptr;
    ggml_gallocr_t galloc  = nullptr;
    ggml_cgraph  * gf      = nullptr;
    ggml_tensor  * c_syl   = nullptr;
    ggml_tensor  * c_pos   = nullptr;
    ggml_tensor  * c_logits= nullptr;
    int            cached_T = -1;

    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "FATAL: tensor not found: %s\n", name.c_str());
            exit(1);
        }
        return it->second;
    }
    bool has(const std::string & name) const { return tensors.count(name) > 0; }
};

// ---------------- load ----------------
slothe_model * slothe_load(const char * path) {
    slothe_model * m = new slothe_model();
    m->backend = ggml_backend_cpu_init();
    // parallelize the decode across the device's cores (capped); matches the
    // engine/slothd copy. Android/NDK has real pthreads.
    {
        // Default cap 4, not 8: on big.LITTLE SoCs (BOOX SD662 = 4xA73+4xA53) the
        // little cores DRAG the matmul — measured on-device 6-syl decode: 8 threads
        // 49.6ms / 6t 40.2ms / 4t 18.4ms / 3t 22.5ms. 4 big-core threads is 2.7x
        // faster than the old min(cores,8) default and matches the predictor bench.
        unsigned hc = std::thread::hardware_concurrency();
        int nt = hc ? (int) (hc < 4u ? hc : 4u) : 4;
        if (const char * e = getenv("SLOTHE_THREADS")) { int v = atoi(e); if (v > 0 && v <= 16) nt = v; }
        ggml_backend_cpu_set_n_threads(m->backend, nt);
    }

    struct gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&m->ctx_w };
    struct gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) { fprintf(stderr, "failed to open gguf %s\n", path); exit(1); }

    // register all tensors by name
    for (ggml_tensor * t = ggml_get_first_tensor(m->ctx_w); t != nullptr; t = ggml_get_next_tensor(m->ctx_w, t)) {
        m->tensors[ggml_get_name(t)] = t;
    }

    // allocate a backend buffer for all weights and fill it from the file
    m->buf_w = ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend);

    FILE * f = fopen(path, "rb");
    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t nt = gguf_get_n_tensors(gguf);
    std::vector<char> tmp;
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = m->tensors[name];
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t nb = ggml_nbytes(t);
        tmp.resize(nb);
        if (fseek(f, (long)off, SEEK_SET) != 0) { fprintf(stderr, "seek fail\n"); exit(1); }
        if (fread(tmp.data(), 1, nb, f) != nb)   { fprintf(stderr, "read fail %s\n", name); exit(1); }
        ggml_backend_tensor_set(t, tmp.data(), 0, nb);
    }
    fclose(f);
    gguf_free(gguf);
    return m;
}

void slothe_free(slothe_model * m) {
    if (!m) return;
    if (m->galloc) ggml_gallocr_free(m->galloc);
    if (m->ctx_c)  ggml_free(m->ctx_c);
    if (m->buf_w) ggml_backend_buffer_free(m->buf_w);
    if (m->ctx_w) ggml_free(m->ctx_w);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

int slothe_n_char(const slothe_model * m) { return m->hp.n_char; }

// ---------------- graph building ----------------
// multiply activation by a (possibly f16) weight vector, broadcasting over ne0.
static ggml_tensor * mul_w(ggml_context * ctx, ggml_tensor * a, ggml_tensor * w) {
    if (w->type != GGML_TYPE_F32) w = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_mul(ctx, a, w);
}
// LIN: one linear projection. fp blocks use the raw f16 weight; ternary blocks
// apply SubLN then zero-pad the feature dim to the padded in-features.
static ggml_tensor * slothe_lin(ggml_context * ctx, const slothe_model & m, int il,
                                const char * name, ggml_tensor * y, bool is_fp,
                                int in_features, int pad_to) {
    char buf[128];
    if (is_fp) {
        snprintf(buf, sizeof(buf), "blk.%d.%s.weight", il, name);
        return ggml_mul_mat(ctx, m.get(buf), y);
    }
    // ternary: subln, pad, matmul
    snprintf(buf, sizeof(buf), "blk.%d.%s_subln.weight", il, name);
    ggml_tensor * subln = m.get(buf);
    y = ggml_rms_norm(ctx, y, m.hp.rms_eps);
    y = mul_w(ctx, y, subln);
    if (pad_to > in_features) {
        y = ggml_pad(ctx, y, pad_to - in_features, 0, 0, 0);
    }
    snprintf(buf, sizeof(buf), "blk.%d.%s.weight", il, name);
    return ggml_mul_mat(ctx, m.get(buf), y);
}

// One transformer block. x: [dim, T]. pos: i32 [T]. Returns [dim, T].
static ggml_tensor * slothe_block(ggml_context * ctx, const slothe_model & m, int il,
                                  ggml_tensor * x, ggml_tensor * pos, int T) {
    const auto & hp = m.hp;
    const bool is_fp = (il == 0) || (il == hp.n_layer - 1);
    const int hd = hp.head_dim, nh = hp.n_head, nkv = hp.n_head_kv;
    const float eps = hp.rms_eps;
    char buf[128];

    // ---- attention (pre-norm) ----
    snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", il);
    ggml_tensor * h = mul_w(ctx, ggml_rms_norm(ctx, x, eps), m.get(buf));

    const int padd = ((hp.dim + hp.pad_to - 1)/hp.pad_to)*hp.pad_to;
    const int padf = ((hp.ffn + hp.pad_to - 1)/hp.pad_to)*hp.pad_to;
    ggml_tensor * q = slothe_lin(ctx, m, il, "attn_q", h, is_fp, hp.dim, padd); // [352,T]
    ggml_tensor * k = slothe_lin(ctx, m, il, "attn_k", h, is_fp, hp.dim, padd); // [88,T]
    ggml_tensor * v = slothe_lin(ctx, m, il, "attn_v", h, is_fp, hp.dim, padd); // [88,T]

    q = ggml_reshape_3d(ctx, q, hd, nh,  T);
    k = ggml_reshape_3d(ctx, k, hd, nkv, T);
    v = ggml_reshape_3d(ctx, v, hd, nkv, T);

    // QK-norm over head_dim
    snprintf(buf, sizeof(buf), "blk.%d.attn_q_rmsnorm.weight", il);
    q = mul_w(ctx, ggml_rms_norm(ctx, q, eps), m.get(buf));
    snprintf(buf, sizeof(buf), "blk.%d.attn_k_rmsnorm.weight", il);
    k = mul_w(ctx, ggml_rms_norm(ctx, k, eps), m.get(buf));

    // RoPE (NEOX) after QK-norm
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      hp.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // attention, no mask, GQA via native mul_mat broadcast (kv head = qhead / (nh/nkv))
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [hd, T, nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [hd, T, nkv]
    v = ggml_permute(ctx, v, 0, 2, 1, 3); // [hd, T, nkv]

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q); // [T, T, nh]
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf((float)hd), 0.0f);

    v = ggml_cont(ctx, ggml_transpose(ctx, v)); // [T, hd, nkv]
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq); // [hd, T, nh]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);     // [hd, nh, T]
    ggml_tensor * o = ggml_cont_2d(ctx, kqv, hd * nh, T); // [352, T]

    o = slothe_lin(ctx, m, il, "attn_output", o, is_fp, hp.dim, padd);
    x = ggml_add(ctx, x, o);

    // ---- ffn (pre-norm, SwiGLU) ----
    snprintf(buf, sizeof(buf), "blk.%d.ffn_norm.weight", il);
    ggml_tensor * h2 = mul_w(ctx, ggml_rms_norm(ctx, x, eps), m.get(buf));
    ggml_tensor * g = slothe_lin(ctx, m, il, "ffn_gate", h2, is_fp, hp.dim, padd);  // [960,T]
    ggml_tensor * u = slothe_lin(ctx, m, il, "ffn_up",   h2, is_fp, hp.dim, padd);  // [960,T]
    ggml_tensor * ff = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ff = slothe_lin(ctx, m, il, "ffn_down", ff, is_fp, hp.ffn, padf);              // [352,T]
    x = ggml_add(ctx, x, ff);
    return x;
}

// Full-forward graph. Captures embed_norm, each block output, norm, logits as outputs.
struct fwd_nodes {
    ggml_tensor * embed_norm = nullptr;
    ggml_tensor * blk[64]    = {nullptr};
    ggml_tensor * norm       = nullptr;
    ggml_tensor * logits     = nullptr;
    ggml_tensor * syl_in     = nullptr;
    ggml_tensor * pos_in     = nullptr;
};

static ggml_cgraph * build_forward_graph(ggml_context * ctx, const slothe_model & m, int T, fwd_nodes & out) {
    const auto & hp = m.hp;
    ggml_tensor * syl = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(syl); ggml_set_name(syl, "syl_ids");
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos); ggml_set_name(pos, "pos");
    out.syl_in = syl; out.pos_in = pos;

    ggml_tensor * x = ggml_get_rows(ctx, m.get("token_embd.weight"), syl); // [352,T]
    x = mul_w(ctx, ggml_rms_norm(ctx, x, hp.rms_eps), m.get("token_embd_norm.weight"));
    out.embed_norm = x; ggml_set_output(x);

    for (int il = 0; il < hp.n_layer; ++il) {
        x = slothe_block(ctx, m, il, x, pos, T);
        out.blk[il] = x; ggml_set_output(x);
    }
    x = mul_w(ctx, ggml_rms_norm(ctx, x, hp.rms_eps), m.get("output_norm.weight"));
    out.norm = x; ggml_set_output(x);

    ggml_tensor * logits = ggml_mul_mat(ctx, m.get("output.weight"), x); // [n_char, T]
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    out.logits = logits; ggml_set_output(logits);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// Single-block isolation graph: input x [dim,T], run one block.
static ggml_cgraph * build_block_graph(ggml_context * ctx, const slothe_model & m, int il, int T,
                                       ggml_tensor ** x_in, ggml_tensor ** pos_in, ggml_tensor ** out) {
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.hp.dim, T);
    ggml_set_input(x); ggml_set_name(x, "x_in");
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos); ggml_set_name(pos, "pos");
    ggml_tensor * y = slothe_block(ctx, m, il, x, pos, T);
    ggml_set_output(y);
    *x_in = x; *pos_in = pos; *out = y;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    return gf;
}

// ---------------- public logits API ----------------
void slothe_logits(slothe_model * m, const int32_t * syl_ids, int T, float * out_logits) {
    // (Re)build the graph only when T changes. Metadata-only context (no_alloc=true),
    // so this is graph structs, not the compute buffer — the galloc owns that and is
    // reused across calls of the same length.
    if (m->cached_T != T || !m->gf) {
        if (m->galloc) { ggml_gallocr_free(m->galloc); m->galloc = nullptr; }
        if (m->ctx_c)  { ggml_free(m->ctx_c); m->ctx_c = nullptr; }
        ggml_init_params ip = { (size_t)16*1024*1024, nullptr, /*no_alloc=*/true };
        m->ctx_c = ggml_init(ip);
        fwd_nodes nd;
        m->gf = build_forward_graph(m->ctx_c, *m, T, nd);
        m->c_syl = nd.syl_in; m->c_pos = nd.pos_in; m->c_logits = nd.logits;
        m->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
        ggml_gallocr_alloc_graph(m->galloc, m->gf);
        m->cached_T = T;
    }

    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = i;
    ggml_backend_tensor_set(m->c_syl, syl_ids, 0, T*sizeof(int32_t));
    ggml_backend_tensor_set(m->c_pos, pos.data(), 0, T*sizeof(int32_t));

    ggml_backend_graph_compute(m->backend, m->gf);
    ggml_backend_tensor_get(m->c_logits, out_logits, 0, (size_t)T*m->hp.n_char*sizeof(float));
}

// ================= validation harness =================
static std::string BASE; // slothe_port dir

static std::vector<float> read_f32(const std::string & path, size_t * n = nullptr) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz/4);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr,"read fail %s\n",path.c_str()); exit(1);}
    fclose(f);
    if (n) *n = v.size();
    return v;
}
static std::vector<int32_t> read_i32(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz/4);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr,"read fail\n"); exit(1);}
    fclose(f);
    return v;
}
static std::string gbin(int s, const char * key) {
    return BASE + "/golden/bin/sent" + std::to_string(s) + "__" + key + ".bin";
}
static double maxabs_diff(const float * a, const float * b, size_t n, double * meanabs=nullptr) {
    double mx = 0, sum = 0;
    for (size_t i = 0; i < n; ++i) { double d = fabs((double)a[i]-(double)b[i]); mx = std::max(mx,d); sum += d; }
    if (meanabs) *meanabs = sum/n;
    return mx;
}

// run a single isolated block given input x (host [dim*T]), return output host [dim*T]
static std::vector<float> run_block(slothe_model & m, int il, int T, const float * x_host) {
    size_t mem = (size_t)128*1024*1024;
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor *x_in, *pos_in, *out;
    ggml_cgraph * gf = build_block_graph(ctx, m, il, T, &x_in, &pos_in, &out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    std::vector<int32_t> pos(T); for (int i=0;i<T;++i) pos[i]=i;
    ggml_backend_tensor_set(x_in, x_host, 0, (size_t)T*m.hp.dim*sizeof(float));
    ggml_backend_tensor_set(pos_in, pos.data(), 0, T*sizeof(int32_t));
    ggml_backend_graph_compute(m.backend, gf);
    std::vector<float> res((size_t)T*m.hp.dim);
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return res;
}

// compute embed_norm only (host) for a sentence — via full graph but read embed_norm
static void run_full(slothe_model & m, int T, const int32_t * syl,
                     fwd_nodes & nd_out, std::vector<float> & emb,
                     std::vector<std::vector<float>> & blks,
                     std::vector<float> & norm, std::vector<float> & logits) {
    size_t mem = (size_t)128*1024*1024 + (size_t)T*m.hp.n_char*8;
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    fwd_nodes nd;
    ggml_cgraph * gf = build_forward_graph(ctx, m, T, nd);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    std::vector<int32_t> pos(T); for (int i=0;i<T;++i) pos[i]=i;
    ggml_backend_tensor_set(nd.syl_in, syl, 0, T*sizeof(int32_t));
    ggml_backend_tensor_set(nd.pos_in, pos.data(), 0, T*sizeof(int32_t));
    ggml_backend_graph_compute(m.backend, gf);

    emb.resize((size_t)T*m.hp.dim);
    ggml_backend_tensor_get(nd.embed_norm, emb.data(), 0, emb.size()*sizeof(float));
    blks.resize(m.hp.n_layer);
    for (int il=0; il<m.hp.n_layer; ++il){
        blks[il].resize((size_t)T*m.hp.dim);
        ggml_backend_tensor_get(nd.blk[il], blks[il].data(), 0, blks[il].size()*sizeof(float));
    }
    norm.resize((size_t)T*m.hp.dim);
    ggml_backend_tensor_get(nd.norm, norm.data(), 0, norm.size()*sizeof(float));
    logits.resize((size_t)T*m.hp.n_char);
    ggml_backend_tensor_get(nd.logits, logits.data(), 0, logits.size()*sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}

#ifndef SLOTHE_NO_MAIN
int main(int argc, char ** argv) {
    BASE = (argc > 1) ? argv[1] : "/tmp/sloth.cpp/slothe_port";
    std::string gguf = BASE + "/slothe-t-25m.gguf";
    slothe_model * m = slothe_load(gguf.c_str());
    ggml_backend_cpu_set_n_threads(m->backend, 4);
    const int dim = m->hp.dim, n_char = m->hp.n_char;

    printf("=== slothe ggml forward validation ===\n");
    printf("loaded %zu tensors from %s\n\n", m->tensors.size(), gguf.c_str());

    // ---- sent0 structural / chain validation ----
    const int s = 0;
    std::vector<int32_t> syl0 = read_i32(gbin(s, "syl_ids"));
    int T = (int)syl0.size();
    std::vector<float> g_emb = read_f32(gbin(s,"embed_norm"));
    std::vector<std::vector<float>> g_blk(16);
    for (int i=0;i<16;++i) g_blk[i] = read_f32(gbin(s, ("blk"+std::to_string(i)).c_str()));
    std::vector<float> g_norm = read_f32(gbin(s,"norm"));
    std::vector<float> g_logits = read_f32(gbin(s,"logits"));

    // full forward
    fwd_nodes nd; std::vector<float> emb; std::vector<std::vector<float>> blks;
    std::vector<float> norm, logits;
    run_full(*m, T, syl0.data(), nd, emb, blks, norm, logits);

    printf("--- sent0 stage diffs (T=%d) ---\n", T);
    printf("%-14s %14s %14s %14s\n", "stage", "chain_maxabs", "isol_maxabs", "isol_meanabs");
    double mn;
    double d = maxabs_diff(emb.data(), g_emb.data(), emb.size(), &mn);
    printf("%-14s %14.3e %14s %14.3e\n", "embed_norm", d, "-", mn);

    // isolation: feed golden blk_{i-1} (or golden embed_norm for i=0) into block i
    for (int i=0;i<16;++i){
        const float * xin = (i==0) ? g_emb.data() : g_blk[i-1].data();
        std::vector<float> yo = run_block(*m, i, T, xin);
        double mni, mnc;
        double di = maxabs_diff(yo.data(), g_blk[i].data(), yo.size(), &mni);
        double dc = maxabs_diff(blks[i].data(), g_blk[i].data(), blks[i].size(), &mnc);
        printf("%-14s %14.3e %14.3e %14.3e\n", ("blk"+std::to_string(i)).c_str(), dc, di, mni);
    }
    d = maxabs_diff(norm.data(), g_norm.data(), norm.size(), &mn);
    printf("%-14s %14.3e %14s %14.3e\n", "norm", d, "-", mn);
    d = maxabs_diff(logits.data(), g_logits.data(), logits.size(), &mn);
    printf("%-14s %14.3e %14s %14.3e\n", "logits", d, "-", mn);

    // ---- 6-sentence argmax ship gate ----
    printf("\n--- ship gate: per-sentence argmax agreement ---\n");
    printf("%-6s %4s %10s %10s %s\n","sent","T","match","total","flips(pos: mine/gold logit_mine logit_gold)");
    int gtot=0, gmatch=0;
    for (int si=0; si<6; ++si){
        std::vector<int32_t> syl = read_i32(gbin(si,"syl_ids"));
        int Ti = (int)syl.size();
        std::vector<float> gl = read_f32(gbin(si,"logits"));
        std::vector<float> ml((size_t)Ti*n_char);
        slothe_logits(m, syl.data(), Ti, ml.data());
        int match=0;
        std::string flips;
        for (int t=0;t<Ti;++t){
            const float * mr = &ml[(size_t)t*n_char];
            const float * gr = &gl[(size_t)t*n_char];
            int ma=0, ga=0;
            for (int c=1;c<n_char;++c){ if(mr[c]>mr[ma]) ma=c; if(gr[c]>gr[ga]) ga=c; }
            if (ma==ga) match++;
            else {
                char b[160];
                snprintf(b,sizeof(b)," [t%d: %d/%d  m(%.3f vs %.3f) g(%.3f vs %.3f)]",
                         t, ma, ga, mr[ma], mr[ga], gr[ma], gr[ga]);
                flips += b;
            }
        }
        gtot += Ti; gmatch += match;
        printf("%-6d %4d %10d %10d %s\n", si, Ti, match, Ti, flips.c_str());
    }
    printf("\nTOTAL argmax agreement: %d/%d = %.2f%%\n", gmatch, gtot, 100.0*gmatch/gtot);
    printf("SHIP GATE: %s\n", (gmatch==gtot) ? "PASS (100%)" : "SEE FLIPS ABOVE");

    slothe_free(m);
    return 0;
}
#endif // SLOTHE_NO_MAIN
