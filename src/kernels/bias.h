#pragma once

#include <cuda_fp16.h>

// Add a per-column bias to a (M, N) row-major matrix, in place.
// x[m, n] += bias[n]
void launch_add_bias(__half *x, const __half *bias, int M, int N);
