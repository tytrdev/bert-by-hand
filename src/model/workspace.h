#pragma once

#include "core/device_buffer.h"
#include "core/model_config.h"
#include <cuda_fp16.h>

// Scratch for one forward pass, allocated once and reused across every layer
// and every call. Sizes are fixed by the model config; the same Workspace
// serves the whole encoder.
struct Workspace {
  DeviceBuffer qkv;                // fused q/k/v projection (rows, 3 * HIDDEN)
  DeviceBuffer scores;             // (batch * heads, seq, seq)
  DeviceBuffer merged;             // attention context, merged heads
  DeviceBuffer attn_proj;          // attention output dense
  DeviceBuffer inter, ffn_proj;    // ffn intermediate and output dense
  DeviceBuffer attn_out;           // layer attention result / final hidden
  DeviceBuffer summed, ping, pong; // embeddings and layer ping-pong
};

inline Workspace make_workspace(int batch = 1) {
  using namespace model;
  const size_t mat = size_t(batch) * SEQ_LEN * HIDDEN * sizeof(__half);
  const size_t scores =
      size_t(batch) * NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(__half);
  const size_t inter = size_t(batch) * SEQ_LEN * FFN_DIM * sizeof(__half);
  return {
      DeviceBuffer(3 * mat),                    // qkv
      DeviceBuffer(scores),                     // scores
      DeviceBuffer(mat),                        // merged
      DeviceBuffer(mat),                        // attn_proj
      DeviceBuffer(inter),   DeviceBuffer(mat), // inter ffn_proj
      DeviceBuffer(mat),                        // attn_out
      DeviceBuffer(mat),     DeviceBuffer(mat),
      DeviceBuffer(mat), // summed ping pong
  };
}
