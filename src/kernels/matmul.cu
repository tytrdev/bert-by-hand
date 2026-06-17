#include "core/cuda_check.h"
#include "kernels/matmul.h"
#include <cuda_pipeline.h>
#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int TILE = 16;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// Pipelined tensor core GEMM tile: 128x64 output, 4 warps each owning a 64x32
// warp tile (4x2 fragments held in registers). A and B K-slices stream into
// shared memory with cp.async double buffering so global latency overlaps the
// tensor core math. Used for the large (batched) GEMMs.
constexpr int PBM = 128, PBN = 64, PBK = 32;
constexpr int PWTM = 4, PWTN = 2;       // warp tile, in 16x16 fragments
constexpr int PWNX = PBN / (PWTN * 16); // warps along N
constexpr int PNW = (PBM / (PWTM * 16)) * PWNX;
constexpr int NW_THREADS = PNW * 32;

__device__ inline float epilogue(float v, const __half *bias, int col,
                                 bool gelu) {
  if (bias)
    v += __half2float(bias[col]);
  if (gelu)
    v *= 0.5f * (1.0f + erff(v * 0.70710677f));
  return v;
}

// C = A @ B^T, A is (M, K), B is (N, K). Shared-memory tiled: each block stages
// a TILE x TILE block of A and B once and reuses it across the inner product.
__global__ void matmul_tiled_kernel(const __half *A, const __half *B, __half *C,
                                    int M, int N, int K, const __half *bias,
                                    bool gelu) {
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
    C[m * N + n] = __float2half(epilogue(acc, bias, n, gelu));
}

// Tensor core path for 16-aligned shapes. C = A @ B^T: A (M, K) row major feeds
// matrix_a, B (N, K) row major is exactly B^T as a col major (K, N) so it feeds
// matrix_b directly. One warp per 16x16 output tile. fp32 accumulate.
__global__ void matmul_wmma_kernel(const __half *A, const __half *B, __half *C,
                                   int M, int N, int K, const __half *bias,
                                   bool gelu) {
  int tile_row = blockIdx.y * WMMA_M;
  int tile_col = blockIdx.x * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
  wmma::fill_fragment(c_frag, __float2half(0.0f));

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    wmma::load_matrix_sync(a_frag, A + tile_row * K + k0, K);
    wmma::load_matrix_sync(b_frag, B + tile_col * K + k0, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ __half tile[WMMA_M * WMMA_N];
  wmma::store_matrix_sync(tile, c_frag, WMMA_N, wmma::mem_row_major);
  for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += warpSize) {
    int r = i / WMMA_N, c = i % WMMA_N;
    C[(tile_row + r) * N + tile_col + c] =
        __float2half(epilogue(__half2float(tile[i]), bias, tile_col + c, gelu));
  }
}

__global__ void matmul_pipe_kernel(const __half *A, const __half *B, __half *C,
                                   int M, int N, int K, const __half *bias,
                                   bool gelu) {
  // fp16 accumulate: GeForce runs fp16->fp16 tensor ops at twice the rate of
  // fp16->fp32. Six encoder layers tolerate the reduced precision (parity at
  // cos 0.999); the bias/gelu epilogue still runs in fp32.
  __shared__ __half As[2][PBM * PBK];
  __shared__ __half Bs[2][PBN * PBK];
  __shared__ __half ts[PNW][256];

  int bm = blockIdx.y * PBM, bn = blockIdx.x * PBN;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int wm = warp / PWNX, wn = warp % PWNX;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[PWTM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[PWTN];
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c[PWTM][PWTN];
  for (int i = 0; i < PWTM; i++)
    for (int j = 0; j < PWTN; j++)
      wmma::fill_fragment(c[i][j], __float2half(0.f));

  auto load = [&](int st, int k0) {
    for (int i = tid * 8; i < PBM * PBK; i += blockDim.x * 8)
      __pipeline_memcpy_async(&As[st][i], &A[(bm + i / PBK) * K + k0 + i % PBK],
                              16);
    for (int i = tid * 8; i < PBN * PBK; i += blockDim.x * 8)
      __pipeline_memcpy_async(&Bs[st][i], &B[(bn + i / PBK) * K + k0 + i % PBK],
                              16);
    __pipeline_commit();
  };

  load(0, 0);
  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += PBK) {
    bool more = k0 + PBK < K;
    if (more)
      load(cur ^ 1, k0 + PBK);
    __pipeline_wait_prior(more ? 1 : 0);
    __syncthreads();
    for (int kk = 0; kk < PBK; kk += 16) {
      for (int i = 0; i < PWTM; i++)
        wmma::load_matrix_sync(
            a[i], As[cur] + (wm * PWTM * 16 + i * 16) * PBK + kk, PBK);
      for (int j = 0; j < PWTN; j++)
        wmma::load_matrix_sync(
            b[j], Bs[cur] + (wn * PWTN * 16 + j * 16) * PBK + kk, PBK);
      for (int i = 0; i < PWTM; i++)
        for (int j = 0; j < PWTN; j++)
          wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
    }
    __syncthreads();
    cur ^= 1;
  }

  for (int i = 0; i < PWTM; i++)
    for (int j = 0; j < PWTN; j++) {
      wmma::store_matrix_sync(ts[warp], c[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      for (int l = lane; l < 256; l += 32) {
        int gr = bm + wm * PWTM * 16 + i * 16 + l / 16;
        int gc = bn + wn * PWTN * 16 + j * 16 + l % 16;
        C[gr * N + gc] =
            __float2half(epilogue(__half2float(ts[warp][l]), bias, gc, gelu));
      }
      __syncwarp();
    }
}

} // namespace

void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K, const __half *bias, bool gelu) {
  // Pipelined kernel needs enough row tiles to fill the GPU; below that the
  // simple one-warp wmma wins (the small-batch / batch-1 case).
  if (M >= 256 && M % PBM == 0 && N % PBN == 0 && K % PBK == 0) {
    dim3 grid(N / PBN, M / PBM);
    matmul_pipe_kernel<<<grid, NW_THREADS>>>(A, B, C, M, N, K, bias, gelu);
    CUDA_CHECK_KERNEL();
    return;
  }

  if (M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0) {
    dim3 grid(N / WMMA_N, M / WMMA_M);
    matmul_wmma_kernel<<<grid, 32>>>(A, B, C, M, N, K, bias, gelu);
    CUDA_CHECK_KERNEL();
    return;
  }

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K, bias, gelu);
  CUDA_CHECK_KERNEL();
}
