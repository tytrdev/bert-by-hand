#pragma once

#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "model/encoder.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

// All BERT weights resident on the device. One ModelWeights owns every buffer
// for the whole forward pass; layer(i) hands out the pointer views the encoder
// blocks expect.
struct ModelWeights {
  DeviceBuffer word_emb;
  DeviceBuffer pos_emb;
  DeviceBuffer type_emb;
  DeviceBuffer emb_ln_w;
  DeviceBuffer emb_ln_b;

  struct Layer {
    DeviceBuffer q_w, q_b, k_w, k_b, v_w, v_b;
    DeviceBuffer o_w, o_b, attn_ln_w, attn_ln_b;
    DeviceBuffer inter_w, inter_b, out_w, out_b, ffn_ln_w, ffn_ln_b;
  };
  std::vector<Layer> layers;

  LayerWeights layer(int i) const;
};

namespace detail {

inline const __half *h(const DeviceBuffer &b) {
  return static_cast<const __half *>(b.data());
}

inline ModelWeights::Layer load_layer(int i) {
  using namespace model;
  const std::string p = "encoder.layer." + std::to_string(i) + ".";
  return {
      load_weight_fp16(p + "attention.self.query.weight", HIDDEN * HIDDEN),
      load_weight_fp16(p + "attention.self.query.bias", HIDDEN),
      load_weight_fp16(p + "attention.self.key.weight", HIDDEN * HIDDEN),
      load_weight_fp16(p + "attention.self.key.bias", HIDDEN),
      load_weight_fp16(p + "attention.self.value.weight", HIDDEN * HIDDEN),
      load_weight_fp16(p + "attention.self.value.bias", HIDDEN),
      load_weight_fp16(p + "attention.output.dense.weight", HIDDEN * HIDDEN),
      load_weight_fp16(p + "attention.output.dense.bias", HIDDEN),
      load_weight_fp16(p + "attention.output.LayerNorm.weight", HIDDEN),
      load_weight_fp16(p + "attention.output.LayerNorm.bias", HIDDEN),
      load_weight_fp16(p + "intermediate.dense.weight", FFN_DIM * HIDDEN),
      load_weight_fp16(p + "intermediate.dense.bias", FFN_DIM),
      load_weight_fp16(p + "output.dense.weight", HIDDEN * FFN_DIM),
      load_weight_fp16(p + "output.dense.bias", HIDDEN),
      load_weight_fp16(p + "output.LayerNorm.weight", HIDDEN),
      load_weight_fp16(p + "output.LayerNorm.bias", HIDDEN),
  };
}

} // namespace detail

inline LayerWeights ModelWeights::layer(int i) const {
  using detail::h;
  const Layer &l = layers[i];
  return {
      {h(l.q_w), h(l.q_b), h(l.k_w), h(l.k_b), h(l.v_w), h(l.v_b), h(l.o_w),
       h(l.o_b), h(l.attn_ln_w), h(l.attn_ln_b)},
      {h(l.inter_w), h(l.inter_b), h(l.out_w), h(l.out_b), h(l.ffn_ln_w),
       h(l.ffn_ln_b)},
  };
}

inline ModelWeights load_model_weights() {
  using namespace model;
  ModelWeights w{
      load_weight_fp16("embeddings.word_embeddings.weight",
                       VOCAB_SIZE * HIDDEN),
      load_weight_fp16("embeddings.position_embeddings.weight",
                       MAX_POSITION * HIDDEN),
      load_weight_fp16("embeddings.token_type_embeddings.weight",
                       TYPE_VOCAB_SIZE * HIDDEN),
      load_weight_fp16("embeddings.LayerNorm.weight", HIDDEN),
      load_weight_fp16("embeddings.LayerNorm.bias", HIDDEN),
      {},
  };

  w.layers.reserve(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++)
    w.layers.push_back(detail::load_layer(i));

  return w;
}
