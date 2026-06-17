#pragma once

#include "core/model_config.h"
#include <cstdint>
#include <cuda_fp16.h>

struct Workspace;

// Raw weight pointers for one self-attention sublayer, in device memory.
// q/k/v are fused into one (3 * HIDDEN, HIDDEN) projection.
struct AttnWeights {
  const __half *qkv_w, *qkv_b; // fused query/key/value projection
  const __half *o_w, *o_b;     // attention output dense
  const __half *ln_w, *ln_b;   // attention output LayerNorm
};

// Raw weight pointers for one feed forward sublayer, in device memory.
struct FfnWeights {
  const __half *inter_w, *inter_b; // intermediate dense (HIDDEN -> FFN_DIM)
  const __half *out_w, *out_b;     // output dense (FFN_DIM -> HIDDEN)
  const __half *ln_w, *ln_b;       // output LayerNorm
};

// One BERT self-attention block: projections, scaled dot product attention with
// the padding mask, output projection, residual and LayerNorm. hidden and out
// are (batch * SEQ_LEN, HIDDEN); mask is the (batch * SEQ_LEN,) attention mask.
void attention_block(Workspace &ws, const __half *hidden, const AttnWeights &w,
                     const int32_t *mask, __half *out, int batch = 1,
                     int seq = model::SEQ_LEN);

// One BERT feed forward block: intermediate dense, gelu, output dense, residual
// and LayerNorm. hidden and out are (batch * SEQ_LEN, HIDDEN).
void ffn_block(Workspace &ws, const __half *hidden, const FfnWeights &w,
               __half *out, int batch = 1, int seq = model::SEQ_LEN);

// All weights for one encoder layer.
struct LayerWeights {
  AttnWeights attn;
  FfnWeights ffn;
};

// One full BERT encoder layer: self-attention block then feed forward block.
// hidden and out are (batch * SEQ_LEN, HIDDEN); mask is (batch * SEQ_LEN,).
void encoder_layer(Workspace &ws, const __half *hidden, const LayerWeights &w,
                   const int32_t *mask, __half *out, int batch = 1,
                   int seq = model::SEQ_LEN);
