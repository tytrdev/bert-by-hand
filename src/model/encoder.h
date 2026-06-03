#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// Raw weight pointers for one self-attention sublayer, in device memory.
struct AttnWeights {
  const __half *q_w, *q_b;
  const __half *k_w, *k_b;
  const __half *v_w, *v_b;
  const __half *o_w, *o_b;   // attention output dense
  const __half *ln_w, *ln_b; // attention output LayerNorm
};

// One BERT self-attention block: projections, scaled dot product attention with
// the padding mask, output projection, residual and LayerNorm.
// hidden and out are (SEQ_LEN, HIDDEN); mask is the (SEQ_LEN,) attention mask.
void attention_block(const __half *hidden, const AttnWeights &w,
                     const int32_t *mask, __half *out);
