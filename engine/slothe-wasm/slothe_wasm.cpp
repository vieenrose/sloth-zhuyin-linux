// Emscripten wrapper around the libslothe ggml forward pass (engine/slothingd/
// slothe.{h,cpp}). Exposes 4 C entry points to ime.js: load the GGUF from a
// byte buffer (written to MEMFS, then slothe_load), query n_char, run logits,
// free. Built by build.sh -> space-static/enc/slothe.{js,wasm}.
#include "slothe.h"
#include <emscripten.h>
#include <cstdio>
#include <cstdint>

static slothe_model * g_m = nullptr;

extern "C" {

EMSCRIPTEN_KEEPALIVE int slothe_wasm_load(const uint8_t * data, int len) {
    FILE * f = fopen("/model.gguf", "wb");
    if (!f) return 0;
    fwrite(data, 1, (size_t) len, f);
    fclose(f);
    if (g_m) { slothe_free(g_m); g_m = nullptr; }
    g_m = slothe_load("/model.gguf");
    return g_m ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int slothe_wasm_n_char() {
    return g_m ? slothe_n_char(g_m) : 0;
}

EMSCRIPTEN_KEEPALIVE void slothe_wasm_logits(const int32_t * syl, int T, float * out) {
    if (g_m) slothe_logits(g_m, syl, T, out);
}

EMSCRIPTEN_KEEPALIVE void slothe_wasm_free() {
    if (g_m) { slothe_free(g_m); g_m = nullptr; }
}

}
