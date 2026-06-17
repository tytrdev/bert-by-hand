#pragma once

#include "model/weights.h"
#include "model/workspace.h"
#include <cstdint>
#include <cuda_fp16.h>

// Embeddings followed by all encoder layers. out holds the final hidden states,
// shape (batch * SEQ_LEN, HIDDEN). input_ids, token_type_ids and mask are
// (batch * SEQ_LEN,).
void bert_encode(Workspace &ws, const ModelWeights &w, const int32_t *input_ids,
                 const int32_t *token_type_ids, const int32_t *mask,
                 __half *out, int batch = 1);

// Full sentence embedding: encode, masked mean pool, L2 normalize.
// embedding is (batch, HIDDEN).
void bert_embed(Workspace &ws, const ModelWeights &w, const int32_t *input_ids,
                const int32_t *token_type_ids, const int32_t *mask,
                __half *embedding, int batch = 1);
