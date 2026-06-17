#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "model/forward.h"
#include "model/weights.h"
#include "model/workspace.h"
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <string>
#include <vector>

// Run the reference sentence replicated across a batch. Every row of the output
// must match the single-sentence reference embedding.
static DeviceBuffer tile_i32(const char *name, int batch) {
  auto one =
      read_bin<int32_t>(std::string("ref/") + name + ".bin", model::SEQ_LEN);
  std::vector<int32_t> all(size_t(batch) * model::SEQ_LEN);
  for (int b = 0; b < batch; b++)
    std::copy(one.begin(), one.end(), all.begin() + size_t(b) * model::SEQ_LEN);

  DeviceBuffer buf(all.size() * sizeof(int32_t));
  buf.from_host(all.data(), all.size() * sizeof(int32_t));
  return buf;
}

int main() {
  using namespace model;
  constexpr int B = 8;

  auto w = load_model_weights();
  auto ids = tile_i32("input_ids", B);
  auto types = tile_i32("token_type_ids", B);
  auto mask = tile_i32("attention_mask", B);
  auto expected = load_ref_fp32("expected_embedding", HIDDEN);

  Workspace ws = make_workspace(B);
  DeviceBuffer emb(size_t(B) * HIDDEN * sizeof(__half));
  bert_embed(ws, w, static_cast<const int32_t *>(ids.data()),
             static_cast<const int32_t *>(types.data()),
             static_cast<const int32_t *>(mask.data()),
             static_cast<__half *>(emb.data()), B);

  std::vector<__half> emb_h(size_t(B) * HIDDEN);
  emb.to_host(emb_h.data(), emb_h.size() * sizeof(__half));

  bool ok = true;
  for (int b = 0; b < B; b++) {
    std::vector<float> actual(HIDDEN);
    for (int i = 0; i < HIDDEN; i++)
      actual[i] = __half2float(emb_h[size_t(b) * HIDDEN + i]);
    auto r = parity_fp32(expected, actual);
    ok &= report(("batch row " + std::to_string(b)).c_str(), r, 0.999);
  }
  return ok ? 0 : 1;
}
