#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "model/encoder.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

static const __half *hp(const DeviceBuffer &b) {
  return static_cast<const __half *>(b.data());
}

int main() {
  using namespace model;
  const size_t mat = size_t(SEQ_LEN) * HIDDEN;
  const std::string p = "encoder.layer.0.";

  // Feed the reference embedding output as the layer input.
  auto emb = load_ref_fp32("emb_out", mat);
  std::vector<__half> emb_h(mat);
  for (size_t i = 0; i < mat; ++i)
    emb_h[i] = __float2half(emb[i]);
  DeviceBuffer hidden(mat * sizeof(__half));
  hidden.from_host(emb_h.data(), mat * sizeof(__half));

  auto mask = load_ref_i32("attention_mask", SEQ_LEN);

  auto qw =
      load_weight_fp16(p + "attention.self.query.weight", HIDDEN * HIDDEN);
  auto qb = load_weight_fp16(p + "attention.self.query.bias", HIDDEN);
  auto kw = load_weight_fp16(p + "attention.self.key.weight", HIDDEN * HIDDEN);
  auto kb = load_weight_fp16(p + "attention.self.key.bias", HIDDEN);
  auto vw =
      load_weight_fp16(p + "attention.self.value.weight", HIDDEN * HIDDEN);
  auto vb = load_weight_fp16(p + "attention.self.value.bias", HIDDEN);
  auto ow =
      load_weight_fp16(p + "attention.output.dense.weight", HIDDEN * HIDDEN);
  auto ob = load_weight_fp16(p + "attention.output.dense.bias", HIDDEN);
  auto lw = load_weight_fp16(p + "attention.output.LayerNorm.weight", HIDDEN);
  auto lb = load_weight_fp16(p + "attention.output.LayerNorm.bias", HIDDEN);

  AttnWeights w{hp(qw), hp(qb), hp(kw), hp(kb), hp(vw),
                hp(vb), hp(ow), hp(ob), hp(lw), hp(lb)};

  DeviceBuffer out(mat * sizeof(__half));
  attention_block(hp(hidden), w, static_cast<const int32_t *>(mask.data()),
                  static_cast<__half *>(out.data()));

  std::vector<__half> out_h(mat);
  out.to_host(out_h.data(), mat * sizeof(__half));
  std::vector<float> actual(mat);
  for (size_t i = 0; i < mat; ++i)
    actual[i] = __half2float(out_h[i]);

  auto ref = load_ref_fp32("attn0_out", mat);
  auto r = parity_fp32(ref, actual);
  return report("attention", r, 0.999) ? 0 : 1;
}
