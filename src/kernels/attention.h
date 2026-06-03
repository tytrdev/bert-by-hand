#pragma once

#include <cuda_fp16.h>

// Reshape a projected (seq, heads * head_dim) tensor into per-head layout
// (heads, seq, head_dim) so each head is contiguous for the score matmul.
void launch_split_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim);

// Per-head scaled dot product: scores[h, i, j] = scale * q[h,i,.] . k[h,j,.]
// q, k are (heads, seq, head_dim); scores is (heads, seq, seq).
void launch_attention_scores(const __half *q, const __half *k, __half *scores,
                             int heads, int seq, int head_dim, float scale);
