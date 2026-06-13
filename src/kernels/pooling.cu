#include "core/cuda_check.h"
#include "kernels/pooling.h"

namespace {

constexpr int POOL_BLOCK = 256;

__global__ void mean_pool_kernel(const __half *__restrict__ hidden,
                                 const int32_t *__restrict__ mask,
                                 __half *__restrict__ out, int seq, int dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim)
    return;

  float acc = 0.0f;
  int count = 0;
  for (int s = 0; s < seq; s++) {
    int m = mask[s];
    count += m;
    if (m)
      acc += __half2float(hidden[s * dim + d]);
  }

  float denom = count > 0 ? float(count) : 1.0f;
  out[d] = __float2half(acc / denom);
}

} // namespace

void launch_mean_pool(const __half *hidden, const int32_t *mask, __half *out,
                      int seq, int dim) {
  dim3 block(POOL_BLOCK);
  dim3 grid((dim + POOL_BLOCK - 1) / POOL_BLOCK);
  mean_pool_kernel<<<grid, block>>>(hidden, mask, out, seq, dim);
  CUDA_CHECK_KERNEL();
}
