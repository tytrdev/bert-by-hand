#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "model/forward.h"
#include "model/weights.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

int main() {
  using namespace model;

  auto w = load_model_weights();
  auto ids = load_ref_i32("input_ids", SEQ_LEN);
  auto types = load_ref_i32("token_type_ids", SEQ_LEN);
  auto mask = load_ref_i32("attention_mask", SEQ_LEN);
  auto expected = load_ref_fp32("expected_embedding", HIDDEN);

  Workspace ws = make_workspace();
  DeviceBuffer emb(HIDDEN * sizeof(__half));
  bert_embed(ws, w, static_cast<const int32_t *>(ids.data()),
             static_cast<const int32_t *>(types.data()),
             static_cast<const int32_t *>(mask.data()),
             static_cast<__half *>(emb.data()));

  std::vector<__half> emb_h(HIDDEN);
  emb.to_host(emb_h.data(), HIDDEN * sizeof(__half));
  std::vector<float> actual(HIDDEN);
  for (int i = 0; i < HIDDEN; ++i)
    actual[i] = __half2float(emb_h[i]);

  auto r = parity_fp32(expected, actual);
  return report("embed", r, 0.999) ? 0 : 1;
}
