#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "kernels/embedding.h"
#include "kernels/layernorm.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

int main() {
  using namespace model;
  const size_t mat = size_t(SEQ_LEN) * HIDDEN;

  auto ids = load_ref_i32("input_ids", SEQ_LEN);
  auto types = load_ref_i32("token_type_ids", SEQ_LEN);
  auto word = load_weight_fp16("embeddings.word_embeddings.weight",
                               size_t(VOCAB_SIZE) * HIDDEN);
  auto pos = load_weight_fp16("embeddings.position_embeddings.weight",
                              size_t(MAX_POSITION) * HIDDEN);
  auto type = load_weight_fp16("embeddings.token_type_embeddings.weight",
                               size_t(TYPE_VOCAB_SIZE) * HIDDEN);
  auto gamma = load_weight_fp16("embeddings.LayerNorm.weight", HIDDEN);
  auto beta = load_weight_fp16("embeddings.LayerNorm.bias", HIDDEN);
  auto ref = load_ref_fp32("emb_out", mat);

  DeviceBuffer summed(mat * sizeof(__half));
  DeviceBuffer out(mat * sizeof(__half));

  launch_embedding(static_cast<const int32_t *>(ids.data()),
                   static_cast<const int32_t *>(types.data()),
                   static_cast<const __half *>(word.data()),
                   static_cast<const __half *>(pos.data()),
                   static_cast<const __half *>(type.data()),
                   static_cast<__half *>(summed.data()), SEQ_LEN, HIDDEN);

  launch_layernorm(static_cast<const __half *>(summed.data()),
                   static_cast<const __half *>(gamma.data()),
                   static_cast<const __half *>(beta.data()),
                   static_cast<__half *>(out.data()), SEQ_LEN, HIDDEN,
                   LAYER_NORM_EPS);

  std::vector<__half> out_h(mat);
  out.to_host(out_h.data(), mat * sizeof(__half));
  std::vector<float> actual(mat);
  for (size_t i = 0; i < mat; ++i)
    actual[i] = __half2float(out_h[i]);

  auto r = parity_fp32(ref, actual);
  return report("embedding", r, 0.9995) ? 0 : 1;
}
