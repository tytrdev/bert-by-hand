#include "core/cuda_check.h"
#include "kernels/softmax.h"
#include <cstdint>

namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 8;

__inline__ __device__ float warp_max(float v) {
  for (int o = WARP / 2; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

__inline__ __device__ float warp_sum(float v) {
  for (int o = WARP / 2; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// One warp per row. Rows here are at most a few hundred wide, so a warp with
// shuffle reductions beats a full block with shared memory and __syncthreads.
// When mask != nullptr, column j with mask[j] == 0 is dropped (padding key),
// folding the attention mask into the same pass.
__global__ void softmax_kernel(__half *__restrict__ x, int M, int N,
                               const int32_t *__restrict__ mask,
                               int mask_stride) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
  if (row >= M)
    return;
  int lane = threadIdx.x % WARP;
  __half *r = x + size_t(row) * N;

  // Pick this row's mask: rows are grouped by batch, mask_stride rows per
  // batch.
  const int32_t *row_mask = nullptr;
  if (mask)
    row_mask = mask + (mask_stride > 0 ? (row / mask_stride) * N : 0);

  float m = -1e30f;
  for (int i = lane; i < N; i += WARP) {
    if (row_mask && row_mask[i] == 0)
      continue;
    m = fmaxf(m, __half2float(r[i]));
  }
  m = warp_max(m);

  float sum = 0.0f;
  for (int i = lane; i < N; i += WARP) {
    if (row_mask && row_mask[i] == 0)
      continue;
    sum += expf(__half2float(r[i]) - m);
  }
  float inv = 1.0f / warp_sum(sum);

  for (int i = lane; i < N; i += WARP) {
    if (row_mask && row_mask[i] == 0)
      r[i] = __float2half(0.0f);
    else
      r[i] = __float2half(expf(__half2float(r[i]) - m) * inv);
  }
}

} // namespace

void launch_softmax(__half *x, int M, int N, const int32_t *mask,
                    int mask_stride) {
  int threads = WARPS_PER_BLOCK * WARP;
  int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  softmax_kernel<<<blocks, threads>>>(x, M, N, mask, mask_stride);
  CUDA_CHECK_KERNEL();
}
