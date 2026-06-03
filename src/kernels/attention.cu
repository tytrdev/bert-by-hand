#include "core/cuda_check.h"
#include "kernels/attention.h"

namespace {

constexpr int ATTN_BLOCK = 256;

// (seq, heads * head_dim) -> (heads, seq, head_dim)
__global__ void split_heads_kernel(const __half *__restrict__ x,
                                   __half *__restrict__ out, int seq, int heads,
                                   int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq * heads * head_dim;
  if (idx >= total)
    return;

  int d = idx % head_dim;
  int h = (idx / head_dim) % heads;
  int s = idx / (head_dim * heads);

  int src = s * (heads * head_dim) + h * head_dim + d;
  int dst = (h * seq + s) * head_dim + d;
  out[dst] = x[src];
}

} // namespace

void launch_split_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim) {
  int total = seq * heads * head_dim;
  dim3 block(ATTN_BLOCK);
  dim3 grid((total + ATTN_BLOCK - 1) / ATTN_BLOCK);
  split_heads_kernel<<<grid, block>>>(x, out, seq, heads, head_dim);
  CUDA_CHECK_KERNEL();
}
