#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Mean pool hidden states over the sequence, counting only unmasked tokens.
// hidden is (seq, dim); out is (dim,). out[d] = sum_s mask[s] * h[s,d]
// / sum_s mask[s], matching sentence-transformers mean pooling.
void launch_mean_pool(const __half *hidden, const int32_t *mask, __half *out,
                      int seq, int dim);
