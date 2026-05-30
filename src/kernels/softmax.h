#pragma once

#include <cuda_fp16.h>

// Row-wise softmax over a (M, N) row-major matrix, in place.
// Numerically stable: subtract the row max before exp.
void launch_softmax(__half *x, int M, int N);
