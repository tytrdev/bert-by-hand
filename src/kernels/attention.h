#pragma once

#include <cuda_fp16.h>

// Reshape a projected (seq, heads * head_dim) tensor into per-head layout
// (heads, seq, head_dim) so each head is contiguous for the score matmul.
void launch_split_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim);
