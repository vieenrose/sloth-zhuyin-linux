cmake_minimum_required(VERSION 3.12)
project(slothd)

# Path to the already-built llama.cpp tree (see README.md "Set up the local
# LLM runtime"). Not vendored/installed system-wide, so we point directly at
# its build output rather than using find_package(Llama).
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llm/llama.cpp" CACHE PATH
    "Path to the llama.cpp checkout used to build slothd")

add_executable(slothd slothd.cpp)
target_compile_features(slothd PRIVATE cxx_std_17)
target_include_directories(slothd PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/vendor"
)
target_link_directories(slothd PRIVATE "${LLAMA_CPP_DIR}/build/bin")
target_link_libraries(slothd PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(slothd PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin"
)

# slothd_slothe: same socket protocol, but served by the validated ternary
# IME encoder (slothe.{h,cpp} ggml forward) instead of a generative llama.cpp
# model. Links ggml + llama (llama for the optional -p next-word predictor).
# main(), compiled out here with -DSLOTHE_NO_MAIN.
add_executable(slothd_slothe slothd_slothe.cpp slothe.cpp)
target_compile_features(slothd_slothe PRIVATE cxx_std_17)
target_compile_definitions(slothd_slothe PRIVATE SLOTHE_NO_MAIN)
target_include_directories(slothd_slothe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/vendor"
)
target_link_directories(slothd_slothe PRIVATE "${LLAMA_CPP_DIR}/build/bin")
target_link_libraries(slothd_slothe PRIVATE
    llama ggml ggml-cpu ggml-base ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(slothd_slothe PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin"
)
