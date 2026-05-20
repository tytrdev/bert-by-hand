#include "core/cuda_check.h"
#include <cstdio>
#include <cuda_runtime.h>

// Start by sanity checking GPU
int main() {
  printf("Running main\n");

  int n;
  CUDA_CHECK(cudaGetDeviceCount(&n));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Check that CUDA_CHECK throws errors cleanly
  // printf("Testing failing CUDA_CHECK\n");
  // CUDA_CHECK(cudaErrorInvalidValue);

  printf("Device 0: %s, sm_%d%d\n", prop.name, prop.major, prop.minor);
  return 0;
}
