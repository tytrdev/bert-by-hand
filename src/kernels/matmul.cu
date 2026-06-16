#include "core/cuda_check.h"
#include "kernels/matmul.h"

namespace {

constexpr int TILE = 16;

// C = A @ B^T, A is (M, K), B is (N, K). Shared-memory tiled: each block stages
// a TILE x TILE block of A and B once and reuses it across the inner product.
__global__ void matmul_tiled_kernel(const __half *A, const __half *B, __half *C,
                                    int M, int N, int K) {
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

  if (m < M && n < N)
    C[m * N + n] = __float2half(acc);
}

} // namespace

void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_KERNEL();
}
