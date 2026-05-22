#include "core/cuda_check.h"
#include "kernels/layernorm.h"

namespace {

constexpr int LN_BLOCK = 256;

__global__ void layernorm_naive_kernel(const __half *__restrict__ x,
                                       const __half *__restrict__ gamma,
                                       const __half *__restrict__ beta,
                                       __half *__restrict__ y, int M, int D,
                                       float eps) {
  int row = blockIdx.x;
  if (row >= M)
    return;
  int tid = threadIdx.x;

  const __half *x_row = x + size_t(row) * D;
  __half *y_row = y + size_t(row) * D;

  __shared__ float sdata[LN_BLOCK];

  // sum for mean
  float sum = 0.0f;
  for (int i = tid; i < D; i += LN_BLOCK)
    sum += __half2float(x_row[i]);
  sdata[tid] = sum;
  __syncthreads();

  for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float mean = sdata[0] / float(D);
  __syncthreads();

  // sum of squared deviations for variance
  float sqsum = 0.0f;
  for (int i = tid; i < D; i += LN_BLOCK) {
    float d = __half2float(x_row[i]) - mean;
    sqsum += d * d;
  }
  sdata[tid] = sqsum;
  __syncthreads();

  for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float var = sdata[0] / float(D);
  float rstd = rsqrtf(var + eps);

  // normalize
  for (int i = tid; i < D; i += LN_BLOCK) {
    float xn = (__half2float(x_row[i]) - mean) * rstd;
    float g = __half2float(gamma[i]);
    float b = __half2float(beta[i]);
    y_row[i] = __float2half(xn * g + b);
  }
}

} // namespace

void launch_layernorm(const __half *x, const __half *gamma, const __half *beta,
                      __half *y, int M, int D, float eps) {
  dim3 grid(M);
  dim3 block(LN_BLOCK);
  layernorm_naive_kernel<<<grid, block>>>(x, gamma, beta, y, M, D, eps);
  CUDA_CHECK_KERNEL();
}
