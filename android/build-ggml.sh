#!/bin/sh
# Cross-compile the ggml CPU backend for Android arm64-v8a and stage the static
# libs into android/app/ggml/lib/arm64-v8a/ (gitignored — regenerate with this).
# libslothe (android/app/cpp/slothe.cpp) links these; headers are in
# android/app/ggml/include/. Requires $ANDROID_NDK (e.g. ~/Android/Sdk/ndk/<ver>).
#
# Baseline armv8-a (NDK auto-disables dotprod/i8mm) → runs on no-dotprod SoCs
# like the BOOX (Snapdragon 662).
set -e
: "${ANDROID_NDK:?set ANDROID_NDK to your NDK path, e.g. ~/Android/Sdk/ndk/27.2.12479018}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_DIR/llm/llama.cpp"      # ggml source (the local llama.cpp runtime)
BUILD="${TMPDIR:-/tmp}/slothe_android_ggml"
DEST="$REPO_DIR/android/app/ggml/lib/arm64-v8a"

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_shared \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_CPU=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target ggml ggml-base ggml-cpu -j"$(nproc)"

mkdir -p "$DEST"
find "$BUILD" -name 'libggml*.a' -exec cp -v {} "$DEST/" \;
echo "staged arm64 ggml libs -> $DEST"
