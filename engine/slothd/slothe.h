// slothe.h — reusable core API for the slothe ggml forward pass.
#pragma once
#include <cstdint>

struct slothe_model;

slothe_model * slothe_load(const char * gguf_path);
void           slothe_free(slothe_model * m);
int            slothe_n_char(const slothe_model * m);
// out_logits: caller-owned buffer of T * n_char floats, row-major [T][n_char].
void           slothe_logits(slothe_model * m, const int32_t * syl_ids, int T, float * out_logits);
