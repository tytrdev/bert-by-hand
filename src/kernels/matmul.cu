#include "core/cuda_check.h"
#include "kernels/matmul.h"

namespace {

__global__ void matmul_naive_kernel(const __half *A, const __half *B, __half *C,
                                    int M, int N, int K) {
  // Is there really no builtin/stdlib for this?
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; k++) {
    float a = __half2float(A[m * K + k]);
    float b = __half2float(B[n * K + k]);
    acc += a * b;
  }
  C[m * N + n] = __float2half(acc);
}

} // namespace

void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK_KERNEL(); // Not bad!
}
