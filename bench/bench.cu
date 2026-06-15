#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/timer.h"
#include "model/forward.h"
#include "model/weights.h"
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

// Single sentence latency for the full embedding forward (BATCH=1, SEQ_LEN).
int main() {
  using namespace model;
  constexpr int WARMUP = 20;
  constexpr int ITERS = 1000;

  auto w = load_model_weights();
  auto ids = load_ref_i32("input_ids", SEQ_LEN);
  auto types = load_ref_i32("token_type_ids", SEQ_LEN);
  auto mask = load_ref_i32("attention_mask", SEQ_LEN);

  const auto *ids_p = static_cast<const int32_t *>(ids.data());
  const auto *types_p = static_cast<const int32_t *>(types.data());
  const auto *mask_p = static_cast<const int32_t *>(mask.data());

  DeviceBuffer emb(HIDDEN * sizeof(__half));
  auto *emb_p = static_cast<__half *>(emb.data());

  for (int i = 0; i < WARMUP; i++)
    bert_embed(w, ids_p, types_p, mask_p, emb_p);

  Timer timer;
  timer.start();
  for (int i = 0; i < ITERS; i++)
    bert_embed(w, ids_p, types_p, mask_p, emb_p);
  timer.stop();

  float ms = timer.elapsed_ms();
  float per = ms / ITERS;
  printf("cuda forward: %d iters, %.2f ms total\n", ITERS, ms);
  printf("  %.4f ms / embedding\n", per);
  printf("  %.1f embeddings / sec\n", 1000.0f / per);
  return 0;
}
