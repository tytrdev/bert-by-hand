# Min required cmake version
cmake_minimum_required(VERSION 3.24)

# Emit compile_commands.json for clangd / LSP.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project Specs
project(bert_cuda LANGUAGES CXX CUDA)

# Cmake cuda/c++ specs
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Need to use older gcc on cachyos
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)

enable_testing()

# Initial executable
# add_executable(hello src/main.cu)
# target_include_directories(hello PRIVATE src)

function(add_cuda_test name source)
  add_executable(test_${name} ${source})
  target_include_directories(test_${name} PRIVATE src)
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

add_cuda_test(sanity tests/sanity.cu)
add_cuda_test(model_config tests/test_config.cu)
