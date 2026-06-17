#include "core/cuda_check.h"
#include "kernels/layernorm.h"

namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 8;

__inline__ __device__ float warp_sum(float v) {
  for (int o = WARP / 2; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// One warp per row. When residual != nullptr the layernorm input is
// (x + residual), folding the residual add into this kernel.
__global__ void layernorm_kernel(const __half *__restrict__ x,
                                 const __half *__restrict__ residual,
                                 const __half *__restrict__ gamma,
                                 const __half *__restrict__ beta,
                                 __half *__restrict__ y, int M, int D,
                                 float eps) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
  if (row >= M)
    return;
  int lane = threadIdx.x % WARP;

  const __half *x_row = x + size_t(row) * D;
  const __half *res_row = residual ? residual + size_t(row) * D : nullptr;
  __half *y_row = y + size_t(row) * D;

  float sum = 0.0f;
  for (int i = lane; i < D; i += WARP) {
    float v = __half2float(x_row[i]);
    if (res_row)
      v += __half2float(res_row[i]);
    sum += v;
  }
  float mean = warp_sum(sum) / float(D);

  float sq = 0.0f;
  for (int i = lane; i < D; i += WARP) {
    float v = __half2float(x_row[i]);
    if (res_row)
      v += __half2float(res_row[i]);
    float d = v - mean;
    sq += d * d;
  }
  float rstd = rsqrtf(warp_sum(sq) / float(D) + eps);

  for (int i = lane; i < D; i += WARP) {
    float v = __half2float(x_row[i]);
    if (res_row)
      v += __half2float(res_row[i]);
    float xn = (v - mean) * rstd;
    y_row[i] =
        __float2half(xn * __half2float(gamma[i]) + __half2float(beta[i]));
  }
}

} // namespace

void launch_layernorm(const __half *x, const __half *gamma, const __half *beta,
                      __half *y, int M, int D, float eps,
                      const __half *residual) {
  int threads = WARPS_PER_BLOCK * WARP;
  int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  layernorm_kernel<<<blocks, threads>>>(x, residual, gamma, beta, y, M, D, eps);
  CUDA_CHECK_KERNEL();
}
