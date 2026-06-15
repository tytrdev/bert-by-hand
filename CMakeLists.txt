# Min required cmake version
cmake_minimum_required(VERSION 3.24)

# Emit compile_commands.json for clangd / LSP.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Need to use older gcc on cachyos
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)

# Project Specs
project(bert_cuda LANGUAGES CXX CUDA)

# Cmake cuda/c++ specs
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


enable_testing()

add_library(bert_kernels STATIC
  src/kernels/embedding.cu
  src/kernels/bias.cu
  src/kernels/matmul.cu
  src/kernels/layernorm.cu
  src/kernels/gelu.cu
  src/kernels/softmax.cu
  src/kernels/attention.cu
  src/kernels/residual.cu
  src/kernels/pooling.cu
  src/kernels/normalize.cu
  src/model/encoder.cu
  src/model/forward.cu
)

target_include_directories(bert_kernels PUBLIC src)
set_target_properties(bert_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bench bench/bench.cu)
target_include_directories(bench PRIVATE src)
target_link_libraries(bench PRIVATE bert_kernels)

function(add_cuda_test name source)
  add_executable(test_${name} ${source})
  target_include_directories(test_${name} PRIVATE src)
  target_link_libraries(test_${name} PRIVATE bert_kernels)
  add_test(
    NAME ${name}
    COMMAND test_${name}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endfunction()

add_cuda_test(sanity tests/sanity.cu)
add_cuda_test(model_config tests/test_config.cu)
add_cuda_test(parity tests/test_parity.cu)
add_cuda_test(matmul tests/test_matmul.cu)
add_cuda_test(layernorm tests/test_layernorm.cu)
add_cuda_test(gelu tests/test_gelu.cu)
add_cuda_test(embedding tests/test_embedding.cu)
add_cuda_test(softmax tests/test_softmax.cu)
add_cuda_test(attention tests/test_attention.cu)
add_cuda_test(layer tests/test_layer.cu)
add_cuda_test(embed tests/test_embed.cu)
