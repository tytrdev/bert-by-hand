#pragma once

#include "core/device_buffer.h"
#include "core/model_config.h"
#include <cuda_fp16.h>

// Scratch for one forward pass, allocated once and reused across every layer
// and every call. Sizes are fixed by the model config (BATCH=1), so the same
// Workspace serves the whole encoder. Buffers are disjoint where they need to
// be live at once; the per-layer attention and ffn scratch is overwritten each
// layer.
struct Workspace {
  DeviceBuffer q, k, v;            // qkv projections
  DeviceBuffer qh, kh, vh;         // split into heads
  DeviceBuffer scores;             // (heads, seq, seq)
  DeviceBuffer ctx, merged;        // attention output, pre projection
  DeviceBuffer attn_proj;          // attention output dense
  DeviceBuffer inter, ffn_proj;    // ffn intermediate and output dense
  DeviceBuffer attn_out;           // layer attention result / final hidden
  DeviceBuffer summed, ping, pong; // embeddings and layer ping-pong
};

inline Workspace make_workspace() {
  using namespace model;
  const size_t mat = size_t(SEQ_LEN) * HIDDEN * sizeof(__half);
  const size_t scores = size_t(NUM_HEADS) * SEQ_LEN * SEQ_LEN * sizeof(__half);
  const size_t inter = size_t(SEQ_LEN) * FFN_DIM * sizeof(__half);
  return {
      DeviceBuffer(mat),    DeviceBuffer(mat),   DeviceBuffer(mat),
      DeviceBuffer(mat),    DeviceBuffer(mat),   DeviceBuffer(mat),
      DeviceBuffer(scores), DeviceBuffer(mat),   DeviceBuffer(mat),
      DeviceBuffer(mat),    DeviceBuffer(inter), DeviceBuffer(mat),
      DeviceBuffer(mat),    DeviceBuffer(mat),   DeviceBuffer(mat),
      DeviceBuffer(mat),
  };
}
