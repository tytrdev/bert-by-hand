#include "core/device_buffer.h"
#include "core/model_config.h"
#include "kernels/attention.h"
#include "kernels/layernorm.h"
#include "kernels/matmul.h"
#include "kernels/softmax.h"
#include "model/encoder.h"
#include "model/workspace.h"
#include <cmath>

namespace {

inline __half *as_half(DeviceBuffer &b) {
  return static_cast<__half *>(b.data());
}

// out (m, n) = x (m, k) @ weight^T (n, k) + bias, bias fused into the matmul.
void linear(const __half *x, const __half *w, const __half *b, __half *out,
            int m, int n, int k) {
  launch_matmul(x, w, out, m, n, k, b);
}

} // namespace

void attention_block(Workspace &ws, const __half *hidden, const AttnWeights &w,
                     const int32_t *mask, __half *out, int batch, int seq) {
  using namespace model;
  const float scale = 1.0f / std::sqrt(float(HEAD_DIM));
  const int rows = batch * seq;
  const int bheads = batch * NUM_HEADS;

  // One fused projection produces Q | K | V stacked along the column dim. The
  // attention kernels then read Q/K/V straight out of this buffer (strided wmma
  // loads) and write context into the merged layout, so there is no separate
  // split-heads or merge-heads pass.
  const int qkv = 3 * HIDDEN;
  linear(hidden, w.qkv_w, w.qkv_b, as_half(ws.qkv), rows, qkv, HIDDEN);

  launch_attention_scores(as_half(ws.qkv), as_half(ws.scores), batch, NUM_HEADS,
                          seq, HEAD_DIM, qkv, 0 * HIDDEN, 1 * HIDDEN, scale);
  launch_softmax(as_half(ws.scores), bheads * seq, seq, mask, NUM_HEADS * seq);

  launch_attention_context(as_half(ws.scores), as_half(ws.qkv),
                           as_half(ws.merged), batch, NUM_HEADS, seq, HEAD_DIM,
                           qkv, 2 * HIDDEN);

  linear(as_half(ws.merged), w.o_w, w.o_b, as_half(ws.attn_proj), rows, HIDDEN,
         HIDDEN);
  launch_layernorm(as_half(ws.attn_proj), w.ln_w, w.ln_b, out, rows, HIDDEN,
                   LAYER_NORM_EPS, hidden);
}

void ffn_block(Workspace &ws, const __half *hidden, const FfnWeights &w,
               __half *out, int batch, int seq) {
  using namespace model;
  const int rows = batch * seq;

  launch_matmul(hidden, w.inter_w, as_half(ws.inter), rows, FFN_DIM, HIDDEN,
                w.inter_b, /*gelu=*/true);

  linear(as_half(ws.inter), w.out_w, w.out_b, as_half(ws.ffn_proj), rows,
         HIDDEN, FFN_DIM);
  launch_layernorm(as_half(ws.ffn_proj), w.ln_w, w.ln_b, out, rows, HIDDEN,
                   LAYER_NORM_EPS, hidden);
}

void encoder_layer(Workspace &ws, const __half *hidden, const LayerWeights &w,
                   const int32_t *mask, __half *out, int batch, int seq) {
  attention_block(ws, hidden, w.attn, mask, as_half(ws.attn_out), batch, seq);
  ffn_block(ws, as_half(ws.attn_out), w.ffn, out, batch, seq);
}
