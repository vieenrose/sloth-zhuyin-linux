# externalNativeBuild for libsloth.so — the on-device Sloth IME IME core.
# Compiles the header-only shared core (engine/common: core.h, segment.h,
# zhuyin.h, display.h) + the decode port + JNI glue + libslothe (the ggml
# ternary-encoder forward pass) into one .so that com.sloth.ime.Core loads
# with System.loadLibrary("sloth"). ggml is linked STATICALLY (vendored
# under app/ggml/) so there is no companion .so to ship — unlike the old ORT
# path which needed libonnxruntime.so in jniLibs.
#
# Wire it up from app/build.gradle:
#   android {
#     compileSdk 34
#     defaultConfig {
#       minSdk 30                              # API30
#       ndk { abiFilters "arm64-v8a" }         # phone target
#     }
#     externalNativeBuild { cmake { path "CMakeLists.txt"; version "3.22.1" } }
#   }
# Gradle drives this once per ABI with ANDROID_ABI / ANDROID_PLATFORM already
# set by the NDK toolchain, so we don't hardcode them here.
cmake_minimum_required(VERSION 3.22.1)
project(sloth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Hide everything except the JNI_* / Java_* entry points (JNIEXPORT marks them).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# The shared frontend-free core lives two dirs up, in engine/common.
# (app/ -> android/ -> <repo>/  =>  ../../engine/common)
set(SLOTHING_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../../engine/common")
if(NOT EXISTS "${SLOTHING_COMMON}/core.h")
  message(FATAL_ERROR "engine/common not found at ${SLOTHING_COMMON}")
endif()

# --- libslothe / ggml (ternary GGUF encoder, cross-compiled for arm64) -------
# Replaces ONNX Runtime. Provides the ggml headers + three CPU static libs
# (libggml{,-cpu,-base}.a) built via the NDK toolchain for arm64-v8a (armv8-a
# baseline, NO dotprod/i8mm — safe on the BOOX SD662). Vendored under app/ggml/.
# Build recipe (from /tmp): cmake -S llm/llama.cpp -B <build>
#   -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
#   -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
#   -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF
#   -DGGML_CPU=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF ; then
#   cmake --build <build> --target ggml ggml-base ggml-cpu
set(GGML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "vendored ggml root")
set(GGML_INCLUDE_DIR "${GGML_ROOT}/include")
set(GGML_LIBDIR "${GGML_ROOT}/lib/${ANDROID_ABI}")
foreach(gl ggml ggml-cpu ggml-base)
  if(NOT EXISTS "${GGML_LIBDIR}/lib${gl}.a")
    message(FATAL_ERROR "lib${gl}.a not found at ${GGML_LIBDIR} "
                        "(cross-build ggml for ${ANDROID_ABI} — see comment above)")
  endif()
  add_library(${gl} STATIC IMPORTED)
  set_target_properties(${gl} PROPERTIES IMPORTED_LOCATION "${GGML_LIBDIR}/lib${gl}.a")
endforeach()

# --- libsloth.so ---------------------------------------------------------
add_library(sloth SHARED
  cpp/jni_sloth.cpp     # JNI boundary + one SlothSession per handle
  cpp/onnx_decoder.cpp     # the decode port's Decoder impl (now ggml-backed)
  cpp/slothe.cpp)          # libslothe: the ggml forward pass (validated port)
# Header-only pieces (session.h, decoder.h, onnx_decoder.h, and all of
# engine/common) come in via the include dirs below.

# slothe.cpp carries its own validation main() behind SLOTHE_NO_MAIN — suppress it.
set_source_files_properties(cpp/slothe.cpp PROPERTIES COMPILE_DEFINITIONS SLOTHE_NO_MAIN)

target_include_directories(sloth PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp"   # session.h, decoder.h, onnx_decoder.h, slothe.h
  "${SLOTHING_COMMON}"                # core.h, segment.h, zhuyin.h, display.h, nlohmann/
  "${GGML_INCLUDE_DIR}")             # ggml.h, gguf.h, ggml-cpu.h, ...

target_compile_options(sloth PRIVATE -Wall)
# nlohmann/json needs exceptions + RTTI; both are ON by NDK default
# (don't add -fno-exceptions / -fno-rtti here).

find_library(log-lib log)
# Link order matters for the static ggml libs: ggml -> ggml-cpu -> ggml-base.
target_link_libraries(sloth PRIVATE ggml ggml-cpu ggml-base android ${log-lib})

# No external .so to bundle: ggml is linked statically into libsloth.so.
