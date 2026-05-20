# Min required cmake version
cmake_minimum_required(VERSION 3.24)

# Emit compile_commands.json for clangd / LSP.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project Specs
project(bert_cuda LANGUAGES CXX CUDA)

# Cmake cuda/c++ specs
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

# Need to use older gcc on cachyos
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)

# Initial executable
add_executable(hello src/main.cu)
target_include_directories(hello PRIVATE src)
