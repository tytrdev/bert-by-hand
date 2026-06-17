#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "model/encoder.h"
#include "model/weights.h"
#include "model/workspace.h"
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

  auto emb = load_ref_fp32("emb_out", mat);
  std::vector<__half> emb_h(mat);
  for (size_t i = 0; i < mat; ++i)
    emb_h[i] = __float2half(emb[i]);
  DeviceBuffer hidden(mat * sizeof(__half));
  hidden.from_host(emb_h.data(), mat * sizeof(__half));

  auto mask = load_ref_i32("attention_mask", SEQ_LEN);

  auto qkv_w = detail::load_qkv(p, "weight", size_t(HIDDEN) * HIDDEN);
  auto qkv_b = detail::load_qkv(p, "bias", HIDDEN);
  auto ow =
      load_weight_fp16(p + "attention.output.dense.weight", HIDDEN * HIDDEN);
  auto ob = load_weight_fp16(p + "attention.output.dense.bias", HIDDEN);
  auto aln_w =
      load_weight_fp16(p + "attention.output.LayerNorm.weight", HIDDEN);
  auto aln_b = load_weight_fp16(p + "attention.output.LayerNorm.bias", HIDDEN);

  auto iw = load_weight_fp16(p + "intermediate.dense.weight", FFN_DIM * HIDDEN);
  auto ib = load_weight_fp16(p + "intermediate.dense.bias", FFN_DIM);
  auto fw = load_weight_fp16(p + "output.dense.weight", HIDDEN * FFN_DIM);
  auto fb = load_weight_fp16(p + "output.dense.bias", HIDDEN);
  auto fln_w = load_weight_fp16(p + "output.LayerNorm.weight", HIDDEN);
  auto fln_b = load_weight_fp16(p + "output.LayerNorm.bias", HIDDEN);

  LayerWeights w{
      {hp(qkv_w), hp(qkv_b), hp(ow), hp(ob), hp(aln_w), hp(aln_b)},
      {hp(iw), hp(ib), hp(fw), hp(fb), hp(fln_w), hp(fln_b)},
  };

  Workspace ws = make_workspace();
  DeviceBuffer out(mat * sizeof(__half));
  encoder_layer(ws, hp(hidden), w, static_cast<const int32_t *>(mask.data()),
                static_cast<__half *>(out.data()));

  std::vector<__half> out_h(mat);
  out.to_host(out_h.data(), mat * sizeof(__half));
  std::vector<float> actual(mat);
  for (size_t i = 0; i < mat; ++i)
    actual[i] = __half2float(out_h[i]);

  auto ref = load_ref_fp32("layer0_out", mat);
  auto r = parity_fp32(ref, actual);
  return report("layer", r, 0.999) ? 0 : 1;
}
