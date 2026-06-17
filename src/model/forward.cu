#include "core/cuda_check.h"
#include "core/device_buffer.h"
#include "core/model_config.h"
#include "kernels/embedding.h"
#include "kernels/layernorm.h"
#include "kernels/normalize.h"
#include "kernels/pooling.h"
#include "model/encoder.h"
#include "model/forward.h"
#include <utility>

namespace {

inline __half *as_half(DeviceBuffer &b) {
  return static_cast<__half *>(b.data());
}

} // namespace

void bert_encode(Workspace &ws, const ModelWeights &w, const int32_t *input_ids,
                 const int32_t *token_type_ids, const int32_t *mask,
                 __half *out, int batch, int seq) {
  using namespace model;
  using detail::h;
  const int rows = batch * seq;
  const size_t mat = size_t(rows) * HIDDEN;

  launch_embedding(input_ids, token_type_ids, h(w.word_emb), h(w.pos_emb),
                   h(w.type_emb), as_half(ws.summed), seq, HIDDEN, batch);
  launch_layernorm(as_half(ws.summed), h(w.emb_ln_w), h(w.emb_ln_b),
                   as_half(ws.ping), rows, HIDDEN, LAYER_NORM_EPS);

  __half *cur = as_half(ws.ping);
  __half *nxt = as_half(ws.pong);
  for (int i = 0; i < NUM_LAYERS; i++) {
    encoder_layer(ws, cur, w.layer(i), mask, nxt, batch, seq);
    std::swap(cur, nxt);
  }

  CUDA_CHECK(
      cudaMemcpy(out, cur, mat * sizeof(__half), cudaMemcpyDeviceToDevice));
}

void bert_embed(Workspace &ws, const ModelWeights &w, const int32_t *input_ids,
                const int32_t *token_type_ids, const int32_t *mask,
                __half *embedding, int batch, int seq) {
  using namespace model;

  bert_encode(ws, w, input_ids, token_type_ids, mask, as_half(ws.attn_out),
              batch, seq);
  launch_mean_pool(as_half(ws.attn_out), mask, embedding, seq, HIDDEN, batch);
  launch_l2_normalize(embedding, HIDDEN, batch);
}
