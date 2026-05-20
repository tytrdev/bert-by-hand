// #include "core/cuda_check.h"
#include "core/device_buffer.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

// Start by sanity checking GPU
int main() {
  constexpr size_t N = 1024;

  // Fill host buffer with some values
  std::vector<__half> host_in(N);
  for (size_t i = 0; i < N; i++) {
    host_in[i] = __float2half(static_cast<float>(i));
  }

  // Allocate device buffer, copy up, copy back to host
  DeviceBuffer dbuf(N * sizeof(__half));
  dbuf.from_host(host_in.data(), N * sizeof(__half));

  std::vector<__half> host_out(N);
  dbuf.to_host(host_out.data(), N * sizeof(__half));

  // Verify every element matches
  for (size_t i = 0; i < N; i++) {
    float a = __half2float(host_in[i]);
    float b = __half2float(host_out[i]);

    if (a != b) {
      std::fprintf(stderr, "Mistmatch at %zu: %f != %f\n", i, a, b);
      return 1;
    }
  }

  printf("Round trip OK: %zu elements\n", N);

  // Check that CUDA_CHECK throws errors cleanly
  // printf("Testing failing CUDA_CHECK\n");
  // CUDA_CHECK(cudaErrorInvalidValue);
  return 0;
}
