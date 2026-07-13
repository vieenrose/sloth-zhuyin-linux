#!/usr/bin/env bash
# Build libslothe for the browser: ggml CPU forward (engine/slothingd/slothe.cpp)
# + the emscripten wrapper -> space-static/enc/slothe.{js,wasm}.
#   SIMD=1 (default) -> -msimd128 wasm SIMD
#   THREADS=1        -> -pthread multi-thread (needs COOP/COEP via coi-serviceworker)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
GGML="$REPO/llm/llama.cpp/ggml"
EMSDK="${EMSDK:-$HOME/emsdk}"
# shellcheck disable=SC1091
source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1 || true
EMCC="$EMSDK/upstream/emscripten/emcc"

SIMD="${SIMD:-1}"; THREADS="${THREADS:-0}"
CFLAGS="-O3 -DNDEBUG -DGGML_USE_CPU"
LINK=""
[ "$SIMD" = 1 ] && CFLAGS="$CFLAGS -msimd128"
[ "$THREADS" = 1 ] && { CFLAGS="$CFLAGS -pthread"; LINK="$LINK -pthread -sPTHREAD_POOL_SIZE=8"; }

SRC=(
  "$GGML/src/ggml.c" "$GGML/src/ggml-alloc.c" "$GGML/src/ggml-backend.cpp"
  "$GGML/src/ggml-backend-reg.cpp" "$GGML/src/ggml-backend-meta.cpp"
  "$GGML/src/ggml-threading.cpp"
  "$GGML/src/ggml-quants.c" "$GGML/src/gguf.cpp"
  "$GGML/src/ggml-cpu/ggml-cpu.c" "$GGML/src/ggml-cpu/ggml-cpu.cpp"
  "$GGML/src/ggml-cpu/ops.cpp" "$GGML/src/ggml-cpu/vec.cpp"
  "$GGML/src/ggml-cpu/binary-ops.cpp" "$GGML/src/ggml-cpu/unary-ops.cpp"
  "$GGML/src/ggml-cpu/repack.cpp" "$GGML/src/ggml-cpu/traits.cpp"
  "$GGML/src/ggml-cpu/hbm.cpp"
  "$GGML/src/ggml-cpu/quants.c" "$GGML/src/ggml-cpu/arch/wasm/quants.c"
  "$REPO/engine/slothingd/slothe.cpp" "$REPO/engine/slothe-wasm/slothe_wasm.cpp"
)
INC="-I$GGML/include -I$GGML/src -I$GGML/src/ggml-cpu -I$REPO/engine/slothingd"

set -x
"$EMCC" $CFLAGS -DGGML_VERSION='"0.0.0"' -DGGML_COMMIT='"wasm"' $INC "${SRC[@]}" $LINK \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSlotheModule -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS='["_slothe_wasm_load","_slothe_wasm_n_char","_slothe_wasm_logits","_slothe_wasm_free","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS","HEAPU8","HEAP32","HEAPF32"]' \
  -o "$REPO/space-static/enc/slothe.js"
