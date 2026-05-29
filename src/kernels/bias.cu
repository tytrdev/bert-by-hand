#include "core/cuda_check.h"
#include "kernels/bias.h"

namespace {

constexpr int BIAS_BLOCK = 256;

__global__ void add_bias_kernel(__half *__restrict__ x,
                                const __half *__restrict__ bias, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total)
    return;

  int n = idx % N;
  float v = __half2float(x[idx]) + __half2float(bias[n]);
  x[idx] = __float2half(v);
}

} // namespace

void launch_add_bias(__half *x, const __half *bias, int M, int N) {
  int total = M * N;
  dim3 block(BIAS_BLOCK);
  dim3 grid((total + BIAS_BLOCK - 1) / BIAS_BLOCK);
  add_bias_kernel<<<grid, block>>>(x, bias, M, N);
  CUDA_CHECK_KERNEL();
}
