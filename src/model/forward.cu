#include "core/cuda_check.h"
#include "core/device_buffer.h"
#include "core/model_config.h"
#include "kernels/embedding.h"
#include "kernels/layernorm.h"
#include "model/encoder.h"
#include "model/forward.h"
#include <utility>

namespace {

inline __half *as_half(DeviceBuffer &b) {
  return static_cast<__half *>(b.data());
}

} // namespace

void bert_encode(const ModelWeights &w, const int32_t *input_ids,
                 const int32_t *token_type_ids, const int32_t *mask,
                 __half *out) {
  using namespace model;
  using detail::h;
  const size_t mat = size_t(SEQ_LEN) * HIDDEN;

  DeviceBuffer summed(mat * sizeof(__half));
  DeviceBuffer a(mat * sizeof(__half));
  DeviceBuffer b(mat * sizeof(__half));

  launch_embedding(input_ids, token_type_ids, h(w.word_emb), h(w.pos_emb),
                   h(w.type_emb), as_half(summed), SEQ_LEN, HIDDEN);
  launch_layernorm(as_half(summed), h(w.emb_ln_w), h(w.emb_ln_b), as_half(a),
                   SEQ_LEN, HIDDEN, LAYER_NORM_EPS);

  __half *cur = as_half(a);
  __half *nxt = as_half(b);
  for (int i = 0; i < NUM_LAYERS; i++) {
    encoder_layer(cur, w.layer(i), mask, nxt);
    std::swap(cur, nxt);
  }

  CUDA_CHECK(
      cudaMemcpy(out, cur, mat * sizeof(__half), cudaMemcpyDeviceToDevice));
}
