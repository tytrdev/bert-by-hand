#include "core/cuda_check.h"
#include "kernels/softmax.h"

namespace {

constexpr int SM_BLOCK = 256;

__global__ void softmax_kernel(__half *__restrict__ x, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;
  int tid = threadIdx.x;

  __half *x_row = x + size_t(row) * N;
  __shared__ float sdata[SM_BLOCK];

  // row max for stability
  float m = -1e30f;
  for (int i = tid; i < N; i += SM_BLOCK)
    m = fmaxf(m, __half2float(x_row[i]));
  sdata[tid] = m;
  __syncthreads();

  for (int s = SM_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  float row_max = sdata[0];
  __syncthreads();

  // sum of exp
  float sum = 0.0f;
  for (int i = tid; i < N; i += SM_BLOCK)
    sum += expf(__half2float(x_row[i]) - row_max);
  sdata[tid] = sum;
  __syncthreads();

  for (int s = SM_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float inv = 1.0f / sdata[0];

  for (int i = tid; i < N; i += SM_BLOCK) {
    float e = expf(__half2float(x_row[i]) - row_max);
    x_row[i] = __float2half(e * inv);
  }
}

} // namespace

void launch_softmax(__half *x, int M, int N) {
  dim3 grid(M);
  dim3 block(SM_BLOCK);
  softmax_kernel<<<grid, block>>>(x, M, N);
  CUDA_CHECK_KERNEL();
}
