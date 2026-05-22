// Basic matmul
// torch nn.Linear stores weights as (out_features, in_features)
// output = input @ weight.T + bias
// input (M, K) w/ weight (N, K) => (M, N)

#pragma once

#include <cuda_fp16.h>

void launch_matmul(const __half *A, const __half *B, __half *C, int M, int N,
                   int K);
