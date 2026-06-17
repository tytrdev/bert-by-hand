#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Row-wise softmax over a (M, N) row-major matrix, in place.
// Numerically stable: subtract the row max before exp. When mask != nullptr,
// columns with mask[j] == 0 are treated as padding and zeroed out.
void launch_softmax(__half *x, int M, int N, const int32_t *mask = nullptr);
