#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "core/timer.h"
#include "model/forward.h"
#include "model/weights.h"
#include "model/workspace.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

using namespace model;

// Tile the first `seq` tokens of one sequence across the batch.
static DeviceBuffer tile_i32(const std::vector<int32_t> &one, int batch,
                             int seq) {
  std::vector<int32_t> all(size_t(batch) * seq);
  for (int b = 0; b < batch; b++)
    std::copy(one.begin(), one.begin() + seq, all.begin() + size_t(b) * seq);
  DeviceBuffer buf(all.size() * sizeof(int32_t));
  buf.from_host(all.data(), all.size() * sizeof(int32_t));
  return buf;
}

static double run(const ModelWeights &w, const std::vector<int32_t> &ids1,
                  const std::vector<int32_t> &types1,
                  const std::vector<int32_t> &mask1, int batch, int seq) {
  constexpr int WARMUP = 20, ITERS = 300;
  auto ids = tile_i32(ids1, batch, seq);
  auto types = tile_i32(types1, batch, seq);
  auto mask = tile_i32(mask1, batch, seq);
  const auto *ip = static_cast<const int32_t *>(ids.data());
  const auto *tp = static_cast<const int32_t *>(types.data());
  const auto *mp = static_cast<const int32_t *>(mask.data());

  Workspace ws = make_workspace(batch);
  DeviceBuffer emb(size_t(batch) * HIDDEN * sizeof(__half));
  auto *ep = static_cast<__half *>(emb.data());

  for (int i = 0; i < WARMUP; i++)
    bert_embed(ws, w, ip, tp, mp, ep, batch, seq);
  Timer t;
  t.start();
  for (int i = 0; i < ITERS; i++)
    bert_embed(ws, w, ip, tp, mp, ep, batch, seq);
  t.stop();
  return t.elapsed_ms() / (ITERS * batch);
}

int main() {
  auto w = load_model_weights();
  auto ids = read_bin<int32_t>("ref/input_ids.bin", SEQ_LEN);
  auto types = read_bin<int32_t>("ref/token_type_ids.bin", SEQ_LEN);
  auto mask = read_bin<int32_t>("ref/attention_mask.bin", SEQ_LEN);

  int n_real = 0;
  for (int m : mask)
    n_real += m;
  int eff = ((n_real + 15) / 16) * 16; // round up to wmma alignment

  // Correctness: the effective-length embedding must match the reference.
  {
    auto expected = load_ref_fp32("expected_embedding", HIDDEN);
    auto eids = tile_i32(ids, 1, eff);
    auto etypes = tile_i32(types, 1, eff);
    auto emask = tile_i32(mask, 1, eff);
    Workspace ws = make_workspace(1);
    DeviceBuffer emb(HIDDEN * sizeof(__half));
    bert_embed(ws, w, static_cast<const int32_t *>(eids.data()),
               static_cast<const int32_t *>(etypes.data()),
               static_cast<const int32_t *>(emask.data()),
               static_cast<__half *>(emb.data()), 1, eff);
    std::vector<__half> eh(HIDDEN);
    emb.to_host(eh.data(), HIDDEN * sizeof(__half));
    std::vector<float> got(HIDDEN);
    for (int i = 0; i < HIDDEN; i++)
      got[i] = __half2float(eh[i]);
    report("effective-length embedding parity", parity_fp32(expected, got),
           0.999);
  }

  printf("\nfull seq=%d (padded):\n", SEQ_LEN);
  for (int b : {1, 8, 32, 64}) {
    double ms = run(w, ids, types, mask, b, SEQ_LEN);
    printf("  batch %-3d  %8.4f ms/sent  %9.1f sent/s\n", b, ms, 1000.0 / ms);
  }
  printf("\neffective seq=%d (%d real tokens, no padding waste):\n", eff,
         n_real);
  for (int b : {1, 8, 32, 64}) {
    double ms = run(w, ids, types, mask, b, eff);
    printf("  batch %-3d  %8.4f ms/sent  %9.1f sent/s\n", b, ms, 1000.0 / ms);
  }
  return 0;
}
