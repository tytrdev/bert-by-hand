#pragma once

#include <cuda_fp16.h>

// Reshape a projected (batch * seq, heads * head_dim) tensor into per-head
// layout (batch * heads, seq, head_dim) so each head is contiguous for the
// score matmul.
void launch_split_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim, int batch = 1);

// Per-head scaled dot product: scores[h, i, j] = scale * q[h,i,.] . k[h,j,.]
// q, k are (heads, seq, head_dim); scores is (heads, seq, seq).
void launch_attention_scores(const __half *q, const __half *k, __half *scores,
                             int heads, int seq, int head_dim, float scale);

// Push padded key columns (mask[j] == 0) to -inf before softmax so they drop
// out. mask is the (seq,) attention mask. scores is (heads, seq, seq).
void launch_mask_scores(__half *scores, const int32_t *mask, int heads,
                        int seq);

// Weighted sum of values: ctx[h, i, .] = sum_j probs[h, i, j] * v[h, j, .]
// probs is (heads, seq, seq), v is (heads, seq, head_dim).
void launch_attention_context(const __half *probs, const __half *v, __half *ctx,
                              int heads, int seq, int head_dim);

// Inverse of split_heads: (batch * heads, seq, head_dim) ->
// (batch * seq, heads * head_dim).
void launch_merge_heads(const __half *x, __half *out, int seq, int heads,
                        int head_dim, int batch = 1);
