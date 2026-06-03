#include "core/cuda_check.h"
#include "kernels/residual.h"

namespace {

constexpr int RES_BLOCK = 256;

__global__ void residual_add_kernel(__half *__restrict__ x,
                                    const __half *__restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

} // namespace

void launch_residual_add(__half *x, const __half *y, int n) {
  dim3 block(RES_BLOCK);
  dim3 grid((n + RES_BLOCK - 1) / RES_BLOCK);
  residual_add_kernel<<<grid, block>>>(x, y, n);
  CUDA_CHECK_KERNEL();
}
