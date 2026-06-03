#include "core/cuda_check.h"
#include "kernels/attention.h"
#include <cstdint>

namespace {

constexpr int ATTN_BLOCK = 256;
constexpr float MASK_NEG = -1e30f;

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

// scores[h, i, j] = scale * sum_d q[h, i, d] * k[h, j, d]
__global__ void attention_scores_kernel(const __half *__restrict__ q,
                                        const __half *__restrict__ k,
                                        __half *__restrict__ scores, int heads,
                                        int seq, int head_dim, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = heads * seq * seq;
  if (idx >= total)
    return;

  int j = idx % seq;
  int i = (idx / seq) % seq;
  int h = idx / (seq * seq);

  const __half *q_row = q + (size_t(h) * seq + i) * head_dim;
  const __half *k_row = k + (size_t(h) * seq + j) * head_dim;

  float acc = 0.0f;
  for (int d = 0; d < head_dim; d++)
    acc += __half2float(q_row[d]) * __half2float(k_row[d]);
  scores[idx] = __float2half(acc * scale);
}

// scores[h, i, j] += -inf where key j is padding
__global__ void mask_scores_kernel(__half *__restrict__ scores,
                                   const int32_t *__restrict__ mask, int heads,
                                   int seq) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = heads * seq * seq;
  if (idx >= total)
    return;

  int j = idx % seq;
  if (mask[j] == 0)
    scores[idx] = __float2half(__half2float(scores[idx]) + MASK_NEG);
}

// ctx[h, i, d] = sum_j probs[h, i, j] * v[h, j, d]
__global__ void attention_context_kernel(const __half *__restrict__ probs,
                                         const __half *__restrict__ v,
                                         __half *__restrict__ ctx, int heads,
                                         int seq, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = heads * seq * head_dim;
  if (idx >= total)
    return;

  int d = idx % head_dim;
  int i = (idx / head_dim) % seq;
  int h = idx / (head_dim * seq);

  const __half *p_row = probs + (size_t(h) * seq + i) * seq;
  const __half *v_head = v + size_t(h) * seq * head_dim;

  float acc = 0.0f;
  for (int j = 0; j < seq; j++)
    acc += __half2float(p_row[j]) * __half2float(v_head[j * head_dim + d]);
  ctx[idx] = __float2half(acc);
}

// (heads, seq, head_dim) -> (seq, heads * head_dim)
__global__ void merge_heads_kernel(const __half *__restrict__ x,
                                   __half *__restrict__ out, int seq, int heads,
                                   int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq * heads * head_dim;
  if (idx >= total)
    return;

  int d = idx % head_dim;
  int h = (idx / head_dim) % heads;
  int s = idx / (head_dim * heads);

  int src = (h * seq + s) * head_dim + d;
  int dst = s * (heads * head_dim) + h * head_dim + d;
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

void launch_attention_scores(const __half *q, const __half *k, __half *scores,
                             int heads, int seq, int head_dim, float scale) {
  int total = heads * seq * seq;
  dim3 block(ATTN_BLOCK);
  dim3 grid((total + ATTN_BLOCK - 1) / ATTN_BLOCK);
  attention_scores_kernel<<<grid, block>>>(q, k, scores, heads, seq, head_dim,
                                           scale);
  CUDA_CHECK_KERNEL();
}

void launch_mask_scores(__half *scores, const int32_t *mask, int heads,
                        int seq) {
  int total = heads * seq * seq;
  dim3 block(ATTN_BLOCK);
  dim3 grid((total + ATTN_BLOCK - 1) / ATTN_BLOCK);
  mask_scores_kernel<<<grid, block>>>(scores, mask, heads, seq);
  CUDA_CHECK_KERNEL();
}

void launch_attention_context(const __half *probs, const __half *v, __half *ctx,
                              int heads, int seq, int head_dim) {
  int total = heads * seq * head_dim;
  dim3 block(ATTN_BLOCK);
  dim3 grid((total + ATTN_BLOCK - 1) / ATTN_BLOCK);
  attention_context_kernel<<<grid, block>>>(probs, v, ctx, heads, seq,
                                            head_dim);
  CUDA_CHECK_KERNEL();
}

void launch_merge_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim) {
  int total = seq * heads * head_dim;
  dim3 block(ATTN_BLOCK);
  dim3 grid((total + ATTN_BLOCK - 1) / ATTN_BLOCK);
  merge_heads_kernel<<<grid, block>>>(x, out, seq, heads, head_dim);
  CUDA_CHECK_KERNEL();
}
