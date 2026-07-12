cmake_minimum_required(VERSION 3.12)
project(slothingd)

# Path to the already-built llama.cpp tree (see README.md "Set up the local
# LLM runtime"). Not vendored/installed system-wide, so we point directly at
# its build output rather than using find_package(Llama).
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llm/llama.cpp" CACHE PATH
    "Path to the llama.cpp checkout used to build slothingd")

add_executable(slothingd slothingd.cpp)
target_compile_features(slothingd PRIVATE cxx_std_17)
target_include_directories(slothingd PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/vendor"
)
target_link_directories(slothingd PRIVATE "${LLAMA_CPP_DIR}/build/bin")
target_link_libraries(slothingd PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(slothingd PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin"
)

# slothingd_slothe: same socket protocol, but served by the validated ternary
# IME encoder (slothe.{h,cpp} ggml forward) instead of a generative llama.cpp
# model. Links only ggml (NOT llama). slothe.cpp carries its own validation
# main(), compiled out here with -DSLOTHE_NO_MAIN.
add_executable(slothingd_slothe slothingd_slothe.cpp slothe.cpp)
target_compile_features(slothingd_slothe PRIVATE cxx_std_17)
target_compile_definitions(slothingd_slothe PRIVATE SLOTHE_NO_MAIN)
target_include_directories(slothingd_slothe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/vendor"
)
target_link_directories(slothingd_slothe PRIVATE "${LLAMA_CPP_DIR}/build/bin")
target_link_libraries(slothingd_slothe PRIVATE
    ggml ggml-cpu ggml-base ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(slothingd_slothe PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin"
)
