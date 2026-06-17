#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/timer.h"
#include "model/forward.h"
#include "model/weights.h"
#include "model/workspace.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <string>
#include <vector>

using namespace model;

static DeviceBuffer tile_i32(const std::vector<int32_t> &one, int batch) {
  std::vector<int32_t> all(size_t(batch) * SEQ_LEN);
  for (int b = 0; b < batch; b++)
    std::copy(one.begin(), one.end(), all.begin() + size_t(b) * SEQ_LEN);
  DeviceBuffer buf(all.size() * sizeof(int32_t));
  buf.from_host(all.data(), all.size() * sizeof(int32_t));
  return buf;
}

static void run(const ModelWeights &w, const std::vector<int32_t> &ids1,
                const std::vector<int32_t> &types1,
                const std::vector<int32_t> &mask1, int batch) {
  constexpr int WARMUP = 20, ITERS = 300;

  auto ids = tile_i32(ids1, batch);
  auto types = tile_i32(types1, batch);
  auto mask = tile_i32(mask1, batch);
  const auto *ids_p = static_cast<const int32_t *>(ids.data());
  const auto *types_p = static_cast<const int32_t *>(types.data());
  const auto *mask_p = static_cast<const int32_t *>(mask.data());

  Workspace ws = make_workspace(batch);
  DeviceBuffer emb(size_t(batch) * HIDDEN * sizeof(__half));
  auto *emb_p = static_cast<__half *>(emb.data());

  for (int i = 0; i < WARMUP; i++)
    bert_embed(ws, w, ids_p, types_p, mask_p, emb_p, batch);

  Timer timer;
  timer.start();
  for (int i = 0; i < ITERS; i++)
    bert_embed(ws, w, ids_p, types_p, mask_p, emb_p, batch);
  timer.stop();

  float ms = timer.elapsed_ms();
  float per_sent = ms / (ITERS * batch);
  float sps = 1000.0f / per_sent;
  printf("  batch %-3d  %8.4f ms/sent  %9.1f sent/s\n", batch, per_sent, sps);
}

int main() {
  auto w = load_model_weights();
  auto ids = read_bin<int32_t>("ref/input_ids.bin", SEQ_LEN);
  auto types = read_bin<int32_t>("ref/token_type_ids.bin", SEQ_LEN);
  auto mask = read_bin<int32_t>("ref/attention_mask.bin", SEQ_LEN);

  printf("full embedding forward (seq %d):\n", SEQ_LEN);
  for (int b : {1, 8, 32, 64})
    run(w, ids, types, mask, b);
  return 0;
}
