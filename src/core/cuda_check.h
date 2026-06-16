#pragma once

#include <cstdio>  // IWYU pragma: keep
#include <cstdlib> // IWYU pragma: keep
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                   cudaGetErrorName(_err), __FILE__, __LINE__,                 \
                   cudaGetErrorString(_err));                                  \
                                                                               \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Only checks the launch itself. No device sync, so async errors surface at the
// next sync point (memcpy, event). Keeps the forward from stalling per kernel.
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
  } while (0)
