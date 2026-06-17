#pragma once

#include <cuda_fp16.h>

// Row-wise layernorm of (x + residual) when residual != nullptr, else of x.
void launch_layernorm(const __half *x, const __half *gamma, const __half *beta,
                      __half *y, int M, int D, float eps,
                      const __half *residual = nullptr);
