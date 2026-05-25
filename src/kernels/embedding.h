#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// BERT input embeddings: word + position + token_type, summed per token.
// position id is just the row index (no padding offset).
// out is (seq, hidden), layernorm gets applied separately.
void launch_embedding(const int32_t *input_ids, const int32_t *token_type_ids,
                      const __half *word_emb, const __half *pos_emb,
                      const __half *type_emb, __half *out, int seq_len,
                      int hidden);
