#pragma once

#include "model/weights.h"
#include <cstdint>
#include <cuda_fp16.h>

// Embeddings followed by all encoder layers. out holds the final hidden states,
// shape (SEQ_LEN, HIDDEN). input_ids, token_type_ids and mask are (SEQ_LEN,).
void bert_encode(const ModelWeights &w, const int32_t *input_ids,
                 const int32_t *token_type_ids, const int32_t *mask,
                 __half *out);

// Full sentence embedding: encode, masked mean pool, L2 normalize.
// embedding is (HIDDEN,).
void bert_embed(const ModelWeights &w, const int32_t *input_ids,
                const int32_t *token_type_ids, const int32_t *mask,
                __half *embedding);
