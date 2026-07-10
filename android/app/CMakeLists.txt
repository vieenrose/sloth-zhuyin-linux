# externalNativeBuild for libslothing.so — the on-device Slothing IME core.
# Compiles the header-only shared core (engine/common: core.h, segment.h,
# zhuyin.h, display.h) + the decode port + JNI glue into one .so that
# com.slothing.ime.Core loads with System.loadLibrary("slothing").
#
# Wire it up from app/build.gradle:
#   android {
#     compileSdk 34
#     defaultConfig {
#       minSdk 30                              # API30
#       ndk { abiFilters "arm64-v8a" }         # phone target
#       externalNativeBuild { cmake { arguments "-DORT_ROOT=${ortRoot}" } }
#     }
#     externalNativeBuild { cmake { path "CMakeLists.txt"; version "3.22.1" } }
#   }
# Gradle drives this once per ABI with ANDROID_ABI / ANDROID_PLATFORM already
# set by the NDK toolchain, so we don't hardcode them here.
cmake_minimum_required(VERSION 3.22.1)
project(slothing LANGUAGES CXX)

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

# --- ONNX Runtime (from the ort-android work) -------------------------------
# Provides libonnxruntime.so + onnxruntime_cxx_api.h for arm64-v8a. Override
# with -DORT_ROOT=/abs/path from build.gradle; defaults to a vendored copy.
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ort" CACHE PATH "ONNX Runtime prebuilt root")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB "${ORT_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")
if(NOT EXISTS "${ORT_LIB}")
  message(FATAL_ERROR "libonnxruntime.so not found at ${ORT_LIB} "
                      "(set -DORT_ROOT to the ort-android prebuilt)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")

# --- libslothing.so ---------------------------------------------------------
add_library(slothing SHARED
  cpp/jni_slothing.cpp     # JNI boundary + one SlothingSession per handle
  cpp/onnx_decoder.cpp)    # the decode port's Decoder impl (ORT-backed)
# Header-only pieces (session.h, decoder.h, onnx_decoder.h, and all of
# engine/common) come in via the include dirs below.

target_include_directories(slothing PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp"   # session.h, decoder.h, onnx_decoder.h
  "${SLOTHING_COMMON}"                # core.h, segment.h, zhuyin.h, display.h, nlohmann/
  "${ORT_INCLUDE_DIR}")

target_compile_options(slothing PRIVATE -Wall)
# nlohmann/json + the ORT C++ API need exceptions + RTTI; both are ON by NDK
# default (don't add -fno-exceptions / -fno-rtti here).

find_library(log-lib log)
target_link_libraries(slothing PRIVATE onnxruntime android ${log-lib})

# libonnxruntime.so must ship in the APK next to libslothing.so. Easiest:
# copy the prebuilt into src/main/jniLibs/${ABI}/ (Gradle bundles jniLibs
# automatically), OR add to build.gradle:
#   android.sourceSets.main.jniLibs.srcDirs += "${ORT_ROOT}/lib"
