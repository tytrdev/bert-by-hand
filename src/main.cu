#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>

// Start by sanity checking GPU
int main() {
  int n;
  cudaGetDeviceCount(&n);
  assert(n > 0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device 0: %s, sm_%d%d\n", prop.name, prop.major, prop.minor);
  return 0;
}
