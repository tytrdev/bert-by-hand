// Basic matmul
// torch nn.Linear stores weights as (out_features, in_features)
// output = input @ weight.T + bias
// input (M, K) w/ weight (N, K) => (M, N)

#pragma once

#include <cuda_fp16.h>

// C = A @ B^T (+ bias broadcast over rows, when bias != nullptr). When gelu is
// true, the exact gelu activation is applied in the epilogue.
void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K, const __half *bias = nullptr, bool gelu = false);
