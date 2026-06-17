#include "core/device_buffer.h"
#include "core/model_config.h"
#include "kernels/attention.h"
#include "kernels/gelu.h"
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
                     const int32_t *mask, __half *out) {
  using namespace model;
  const float scale = 1.0f / std::sqrt(float(HEAD_DIM));

  linear(hidden, w.q_w, w.q_b, as_half(ws.q), SEQ_LEN, HIDDEN, HIDDEN);
  linear(hidden, w.k_w, w.k_b, as_half(ws.k), SEQ_LEN, HIDDEN, HIDDEN);
  linear(hidden, w.v_w, w.v_b, as_half(ws.v), SEQ_LEN, HIDDEN, HIDDEN);

  launch_split_heads(as_half(ws.q), as_half(ws.qh), SEQ_LEN, NUM_HEADS,
                     HEAD_DIM);
  launch_split_heads(as_half(ws.k), as_half(ws.kh), SEQ_LEN, NUM_HEADS,
                     HEAD_DIM);
  launch_split_heads(as_half(ws.v), as_half(ws.vh), SEQ_LEN, NUM_HEADS,
                     HEAD_DIM);

  launch_attention_scores(as_half(ws.qh), as_half(ws.kh), as_half(ws.scores),
                          NUM_HEADS, SEQ_LEN, HEAD_DIM, scale);
  launch_softmax(as_half(ws.scores), NUM_HEADS * SEQ_LEN, SEQ_LEN, mask);

  launch_attention_context(as_half(ws.scores), as_half(ws.vh), as_half(ws.ctx),
                           NUM_HEADS, SEQ_LEN, HEAD_DIM);
  launch_merge_heads(as_half(ws.ctx), as_half(ws.merged), SEQ_LEN, NUM_HEADS,
                     HEAD_DIM);

  linear(as_half(ws.merged), w.o_w, w.o_b, as_half(ws.attn_proj), SEQ_LEN,
         HIDDEN, HIDDEN);
  launch_layernorm(as_half(ws.attn_proj), w.ln_w, w.ln_b, out, SEQ_LEN, HIDDEN,
                   LAYER_NORM_EPS, hidden);
}

void ffn_block(Workspace &ws, const __half *hidden, const FfnWeights &w,
               __half *out) {
  using namespace model;

  linear(hidden, w.inter_w, w.inter_b, as_half(ws.inter), SEQ_LEN, FFN_DIM,
         HIDDEN);
  launch_gelu(as_half(ws.inter), SEQ_LEN * FFN_DIM);

  linear(as_half(ws.inter), w.out_w, w.out_b, as_half(ws.ffn_proj), SEQ_LEN,
         HIDDEN, FFN_DIM);
  launch_layernorm(as_half(ws.ffn_proj), w.ln_w, w.ln_b, out, SEQ_LEN, HIDDEN,
                   LAYER_NORM_EPS, hidden);
}

void encoder_layer(Workspace &ws, const __half *hidden, const LayerWeights &w,
                   const int32_t *mask, __half *out) {
  attention_block(ws, hidden, w.attn, mask, as_half(ws.attn_out));
  ffn_block(ws, as_half(ws.attn_out), w.ffn, out);
}
