#include "core/cuda_check.h"
#include "kernels/attention.h"
#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int WM = 16, WN = 16, WK = 16;

// Scores read Q and K straight out of the fused QKV buffer with a strided wmma
// load (ld = qkv_stride, column q_off/k_off), so no separate split-heads pass
// is needed. blockIdx.z is the global head b * heads + h; scores is
// (batch * heads, seq, seq).
__global__ void attention_scores_kernel(const __half *__restrict__ qkv,
                                        __half *__restrict__ scores, int heads,
                                        int seq, int head_dim, int qkv_stride,
                                        int q_off, int k_off, float scale) {
  int z = blockIdx.z;
  int b = z / heads, hl = z % heads;
  int row = blockIdx.y * WM, col = blockIdx.x * WN;
  const __half *q =
      qkv + (size_t(b) * seq) * qkv_stride + q_off + hl * head_dim;
  const __half *k =
      qkv + (size_t(b) * seq) * qkv_stride + k_off + hl * head_dim;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
  wmma::fill_fragment(cf, 0.0f);
  for (int d0 = 0; d0 < head_dim; d0 += WK) {
    wmma::load_matrix_sync(af, q + row * qkv_stride + d0, qkv_stride);
    wmma::load_matrix_sync(bf, k + col * qkv_stride + d0, qkv_stride);
    wmma::mma_sync(cf, af, bf, cf);
  }

  __shared__ float tile[WM * WN];
  wmma::store_matrix_sync(tile, cf, WN, wmma::mem_row_major);
  __half *scores_h = scores + size_t(z) * seq * seq;
  for (int i = threadIdx.x; i < WM * WN; i += warpSize)
    scores_h[(row + i / WN) * seq + col + i % WN] =
        __float2half(tile[i] * scale);
}

// Context = probs @ V. V is read from the fused QKV buffer (column v_off); the
// result is written into the merged (batch * seq, heads * head_dim) layout, so
// no separate merge-heads pass is needed.
__global__ void attention_context_kernel(const __half *__restrict__ probs,
                                         const __half *__restrict__ qkv,
                                         __half *__restrict__ merged, int heads,
                                         int seq, int head_dim, int qkv_stride,
                                         int v_off) {
  int z = blockIdx.z;
  int b = z / heads, hl = z % heads;
  int row = blockIdx.y * WM, col = blockIdx.x * WN;
  const __half *probs_h = probs + size_t(z) * seq * seq;
  const __half *v =
      qkv + (size_t(b) * seq) * qkv_stride + v_off + hl * head_dim;

  wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
  wmma::fill_fragment(cf, 0.0f);
  for (int k0 = 0; k0 < seq; k0 += WK) {
    wmma::load_matrix_sync(af, probs_h + row * seq + k0, seq);
    wmma::load_matrix_sync(bf, v + k0 * qkv_stride + col, qkv_stride);
    wmma::mma_sync(cf, af, bf, cf);
  }

  int out_stride = heads * head_dim;
  __half *out = merged + (size_t(b) * seq) * out_stride + hl * head_dim;
  __shared__ float tile[WM * WN];
  wmma::store_matrix_sync(tile, cf, WN, wmma::mem_row_major);
  for (int i = threadIdx.x; i < WM * WN; i += warpSize)
    out[(row + i / WN) * out_stride + col + i % WN] = __float2half(tile[i]);
}

} // namespace

void launch_attention_scores(const __half *qkv, __half *scores, int batch,
                             int heads, int seq, int head_dim, int qkv_stride,
                             int q_off, int k_off, float scale) {
  dim3 grid(seq / WN, seq / WM, batch * heads);
  attention_scores_kernel<<<grid, 32>>>(qkv, scores, heads, seq, head_dim,
                                        qkv_stride, q_off, k_off, scale);
  CUDA_CHECK_KERNEL();
}

void launch_attention_context(const __half *probs, const __half *qkv,
                              __half *merged, int batch, int heads, int seq,
                              int head_dim, int qkv_stride, int v_off) {
  dim3 grid(head_dim / WN, seq / WM, batch * heads);
  attention_context_kernel<<<grid, 32>>>(probs, qkv, merged, heads, seq,
                                         head_dim, qkv_stride, v_off);
  CUDA_CHECK_KERNEL();
}
