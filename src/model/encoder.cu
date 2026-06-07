#include "core/device_buffer.h"
#include "core/model_config.h"
#include "kernels/attention.h"
#include "kernels/bias.h"
#include "kernels/gelu.h"
#include "kernels/layernorm.h"
#include "kernels/matmul.h"
#include "kernels/residual.h"
#include "kernels/softmax.h"
#include "model/encoder.h"
#include <cmath>

namespace {

inline __half *as_half(DeviceBuffer &b) {
  return static_cast<__half *>(b.data());
}

// hidden (M, HIDDEN) @ weight^T (HIDDEN, HIDDEN) + bias -> out (M, HIDDEN)
void linear(const __half *x, const __half *w, const __half *b, __half *out,
            int m, int n, int k) {
  launch_matmul(x, w, out, m, n, k);
  launch_add_bias(out, b, m, n);
}

} // namespace

void attention_block(const __half *hidden, const AttnWeights &w,
                     const int32_t *mask, __half *out) {
  using namespace model;
  const int rows = SEQ_LEN * HIDDEN;
  const float scale = 1.0f / std::sqrt(float(HEAD_DIM));

  DeviceBuffer q(rows * sizeof(__half));
  DeviceBuffer k(rows * sizeof(__half));
  DeviceBuffer v(rows * sizeof(__half));
  DeviceBuffer qh(rows * sizeof(__half));
  DeviceBuffer kh(rows * sizeof(__half));
  DeviceBuffer vh(rows * sizeof(__half));
  DeviceBuffer scores(size_t(NUM_HEADS) * SEQ_LEN * SEQ_LEN * sizeof(__half));
  DeviceBuffer ctx(rows * sizeof(__half));
  DeviceBuffer merged(rows * sizeof(__half));
  DeviceBuffer attn(rows * sizeof(__half));

  linear(hidden, w.q_w, w.q_b, as_half(q), SEQ_LEN, HIDDEN, HIDDEN);
  linear(hidden, w.k_w, w.k_b, as_half(k), SEQ_LEN, HIDDEN, HIDDEN);
  linear(hidden, w.v_w, w.v_b, as_half(v), SEQ_LEN, HIDDEN, HIDDEN);

  launch_split_heads(as_half(q), as_half(qh), SEQ_LEN, NUM_HEADS, HEAD_DIM);
  launch_split_heads(as_half(k), as_half(kh), SEQ_LEN, NUM_HEADS, HEAD_DIM);
  launch_split_heads(as_half(v), as_half(vh), SEQ_LEN, NUM_HEADS, HEAD_DIM);

  launch_attention_scores(as_half(qh), as_half(kh), as_half(scores), NUM_HEADS,
                          SEQ_LEN, HEAD_DIM, scale);
  launch_mask_scores(as_half(scores), mask, NUM_HEADS, SEQ_LEN);
  launch_softmax(as_half(scores), NUM_HEADS * SEQ_LEN, SEQ_LEN);

  launch_attention_context(as_half(scores), as_half(vh), as_half(ctx),
                           NUM_HEADS, SEQ_LEN, HEAD_DIM);
  launch_merge_heads(as_half(ctx), as_half(merged), SEQ_LEN, NUM_HEADS,
                     HEAD_DIM);

  linear(as_half(merged), w.o_w, w.o_b, as_half(attn), SEQ_LEN, HIDDEN, HIDDEN);
  launch_residual_add(as_half(attn), hidden, rows);
  launch_layernorm(as_half(attn), w.ln_w, w.ln_b, out, SEQ_LEN, HIDDEN,
                   LAYER_NORM_EPS);
}

void ffn_block(const __half *hidden, const FfnWeights &w, __half *out) {
  using namespace model;
  const int rows = SEQ_LEN * HIDDEN;

  DeviceBuffer inter(size_t(SEQ_LEN) * FFN_DIM * sizeof(__half));
  DeviceBuffer ffn(rows * sizeof(__half));

  linear(hidden, w.inter_w, w.inter_b, as_half(inter), SEQ_LEN, FFN_DIM,
         HIDDEN);
  launch_gelu(as_half(inter), SEQ_LEN * FFN_DIM);

  linear(as_half(inter), w.out_w, w.out_b, as_half(ffn), SEQ_LEN, HIDDEN,
         FFN_DIM);
  launch_residual_add(as_half(ffn), hidden, rows);
  launch_layernorm(as_half(ffn), w.ln_w, w.ln_b, out, SEQ_LEN, HIDDEN,
                   LAYER_NORM_EPS);
}

void encoder_layer(const __half *hidden, const LayerWeights &w,
                   const int32_t *mask, __half *out) {
  using namespace model;
  DeviceBuffer attn(size_t(SEQ_LEN) * HIDDEN * sizeof(__half));
  attention_block(hidden, w.attn, mask, as_half(attn));
  ffn_block(as_half(attn), w.ffn, out);
}
