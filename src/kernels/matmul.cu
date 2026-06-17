#include "core/cuda_check.h"
#include "kernels/matmul.h"
#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int TILE = 16;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// C = A @ B^T, A is (M, K), B is (N, K). Shared-memory tiled: each block stages
// a TILE x TILE block of A and B once and reuses it across the inner product.
__global__ void matmul_tiled_kernel(const __half *A, const __half *B, __half *C,
                                    int M, int N, int K, const __half *bias) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int ty = threadIdx.y, tx = threadIdx.x;
  int m = blockIdx.y * TILE + ty;
  int n = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    int ak = k0 + tx;
    int bk = k0 + ty;
    As[ty][tx] = (m < M && ak < K) ? __half2float(A[m * K + ak]) : 0.0f;
    Bs[ty][tx] = (n < N && bk < K) ? __half2float(B[n * K + bk]) : 0.0f;
    __syncthreads();

    for (int kk = 0; kk < TILE; kk++)
      acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();
  }

  if (m < M && n < N) {
    if (bias)
      acc += __half2float(bias[n]);
    C[m * N + n] = __float2half(acc);
  }
}

// Tensor core path for 16-aligned shapes. C = A @ B^T: A (M, K) row major feeds
// matrix_a, B (N, K) row major is exactly B^T as a col major (K, N) so it feeds
// matrix_b directly. One warp per 16x16 output tile. fp32 accumulate.
__global__ void matmul_wmma_kernel(const __half *A, const __half *B, __half *C,
                                   int M, int N, int K, const __half *bias) {
  int tile_row = blockIdx.y * WMMA_M;
  int tile_col = blockIdx.x * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    wmma::load_matrix_sync(a_frag, A + tile_row * K + k0, K);
    wmma::load_matrix_sync(b_frag, B + tile_col * K + k0, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ float tile[WMMA_M * WMMA_N];
  wmma::store_matrix_sync(tile, c_frag, WMMA_N, wmma::mem_row_major);
  for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += warpSize) {
    int r = i / WMMA_N, c = i % WMMA_N;
    float v = tile[i];
    if (bias)
      v += __half2float(bias[tile_col + c]);
    C[(tile_row + r) * N + tile_col + c] = __float2half(v);
  }
}

} // namespace

void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K, const __half *bias) {
  if (M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0) {
    dim3 grid(N / WMMA_N, M / WMMA_M);
    matmul_wmma_kernel<<<grid, 32>>>(A, B, C, M, N, K, bias);
    CUDA_CHECK_KERNEL();
    return;
  }

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K, bias);
  CUDA_CHECK_KERNEL();
}
