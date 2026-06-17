#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Per-head scaled dot product Q @ K^T, reading Q and K out of the fused QKV
// buffer (columns q_off/k_off, row stride qkv_stride) so no split-heads pass is
// needed. scores is (batch * heads, seq, seq).
void launch_attention_scores(const __half *qkv, __half *scores, int batch,
                             int heads, int seq, int head_dim, int qkv_stride,
                             int q_off, int k_off, float scale);

// Weighted sum of values probs @ V, reading V out of the fused QKV buffer
// (column v_off, row stride qkv_stride) and writing straight into merged
// (batch * seq, heads * head_dim). probs is (batch * heads, seq, seq).
void launch_attention_context(const __half *probs, const __half *qkv,
                              __half *merged, int batch, int heads, int seq,
                              int head_dim, int qkv_stride, int v_off);
