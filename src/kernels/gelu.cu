#include "core/cuda_check.h"
#include "kernels/gelu.h"

namespace {

constexpr int GELU_BLOCK = 256;
constexpr float INV_SQRT2 = 0.70710677f; // 1 / sqrt(2)

__global__ void gelu_kernel(__half *__restrict__ x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = __half2float(x[i]);
  float y = v * 0.5f * (1.0f + erff(v * INV_SQRT2));
  x[i] = __float2half(y);
}

} // namespace

void launch_gelu(__half *x, int n) {
  dim3 block(GELU_BLOCK);
  dim3 grid((n + GELU_BLOCK - 1) / GELU_BLOCK);
  gelu_kernel<<<grid, block>>>(x, n);
  CUDA_CHECK_KERNEL();
}
