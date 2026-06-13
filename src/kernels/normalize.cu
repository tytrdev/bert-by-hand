#include "core/cuda_check.h"
#include "kernels/normalize.h"

namespace {

constexpr int NORM_BLOCK = 256;
constexpr float NORM_FLOOR = 1e-12f;

// Single block normalizes the whole vector.
__global__ void l2_normalize_kernel(__half *__restrict__ x, int n) {
  int tid = threadIdx.x;
  __shared__ float sdata[NORM_BLOCK];

  float sq = 0.0f;
  for (int i = tid; i < n; i += NORM_BLOCK) {
    float v = __half2float(x[i]);
    sq += v * v;
  }
  sdata[tid] = sq;
  __syncthreads();

  for (int s = NORM_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  float norm = sqrtf(sdata[0]);
  float inv = 1.0f / fmaxf(norm, NORM_FLOOR);
  for (int i = tid; i < n; i += NORM_BLOCK)
    x[i] = __float2half(__half2float(x[i]) * inv);
}

} // namespace

void launch_l2_normalize(__half *x, int n) {
  l2_normalize_kernel<<<1, NORM_BLOCK>>>(x, n);
  CUDA_CHECK_KERNEL();
}
