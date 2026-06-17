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
#include <fstream>
#include <numeric>
#include <vector>

using namespace model;

static int corpus_size() {
  std::ifstream f("ref/corpus_ids.bin", std::ios::binary | std::ios::ate);
  return int(size_t(f.tellg()) / (SEQ_LEN * sizeof(int32_t)));
}

static double cosine(const float *a, const float *b, int n) {
  double d = 0, na = 0, nb = 0;
  for (int i = 0; i < n; i++) {
    d += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  return d / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

int main() {
  constexpr int B = 64, ITERS = 20;
  auto w = load_model_weights();
  int n = corpus_size();

  auto ids = read_bin<int32_t>("ref/corpus_ids.bin", size_t(n) * SEQ_LEN);
  auto types = read_bin<int32_t>("ref/corpus_types.bin", size_t(n) * SEQ_LEN);
  auto mask = read_bin<int32_t>("ref/corpus_mask.bin", size_t(n) * SEQ_LEN);
  auto ref = read_bin<float>("ref/corpus_emb.bin", size_t(n) * HIDDEN);

  std::vector<int> len(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < SEQ_LEN; j++)
      len[i] += mask[size_t(i) * SEQ_LEN + j];

  Workspace ws = make_workspace(B);
  DeviceBuffer emb(size_t(B) * HIDDEN * sizeof(__half));
  auto *ep = static_cast<__half *>(emb.data());

  // ---- padded: every sentence at the full SEQ_LEN ----
  DeviceBuffer dids(size_t(n) * SEQ_LEN * sizeof(int32_t));
  DeviceBuffer dtypes(size_t(n) * SEQ_LEN * sizeof(int32_t));
  DeviceBuffer dmask(size_t(n) * SEQ_LEN * sizeof(int32_t));
  dids.from_host(ids.data(), ids.size() * sizeof(int32_t));
  dtypes.from_host(types.data(), types.size() * sizeof(int32_t));
  dmask.from_host(mask.data(), mask.size() * sizeof(int32_t));
  auto *ip = static_cast<const int32_t *>(dids.data());
  auto *tp = static_cast<const int32_t *>(dtypes.data());
  auto *mp = static_cast<const int32_t *>(dmask.data());

  auto run_padded = [&]() {
    for (int s = 0; s < n; s += B) {
      int bs = std::min(B, n - s);
      bert_embed(ws, w, ip + size_t(s) * SEQ_LEN, tp + size_t(s) * SEQ_LEN,
                 mp + size_t(s) * SEQ_LEN, ep, bs, SEQ_LEN);
    }
  };
  run_padded();
  Timer t;
  t.start();
  for (int it = 0; it < ITERS; it++)
    run_padded();
  t.stop();
  double padded_sps = double(n) * ITERS / (t.elapsed_ms() / 1000.0);

  // ---- dynamic: sort by length, pad each batch to its longest (rounded) ----
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return len[a] < len[b]; });

  struct Batch {
    DeviceBuffer ids, types, mask;
    int size, eff, start;
  };
  std::vector<Batch> batches;
  size_t eff_sum = 0;
  for (int s = 0; s < n; s += B) {
    int bs = std::min(B, n - s);
    int mx = 0;
    for (int k = 0; k < bs; k++)
      mx = std::max(mx, len[order[s + k]]);
    int eff = std::max(16, (mx + 15) / 16 * 16);
    eff_sum += size_t(eff) * bs;
    std::vector<int32_t> hi(size_t(bs) * eff), ht(size_t(bs) * eff),
        hm(size_t(bs) * eff);
    for (int k = 0; k < bs; k++) {
      const int si = order[s + k];
      for (int j = 0; j < eff; j++) {
        hi[size_t(k) * eff + j] = ids[size_t(si) * SEQ_LEN + j];
        ht[size_t(k) * eff + j] = types[size_t(si) * SEQ_LEN + j];
        hm[size_t(k) * eff + j] = mask[size_t(si) * SEQ_LEN + j];
      }
    }
    Batch b{DeviceBuffer(hi.size() * sizeof(int32_t)),
            DeviceBuffer(ht.size() * sizeof(int32_t)),
            DeviceBuffer(hm.size() * sizeof(int32_t)),
            bs,
            eff,
            s};
    b.ids.from_host(hi.data(), hi.size() * sizeof(int32_t));
    b.types.from_host(ht.data(), ht.size() * sizeof(int32_t));
    b.mask.from_host(hm.data(), hm.size() * sizeof(int32_t));
    batches.push_back(std::move(b));
  }

  auto run_dynamic = [&]() {
    for (auto &b : batches)
      bert_embed(ws, w, static_cast<const int32_t *>(b.ids.data()),
                 static_cast<const int32_t *>(b.types.data()),
                 static_cast<const int32_t *>(b.mask.data()), ep, b.size,
                 b.eff);
  };
  run_dynamic();
  t.start();
  for (int it = 0; it < ITERS; it++)
    run_dynamic();
  t.stop();
  double dyn_sps = double(n) * ITERS / (t.elapsed_ms() / 1000.0);

  // correctness: gather dynamic embeddings back to original order vs pytorch
  std::vector<float> got(size_t(n) * HIDDEN);
  for (auto &b : batches) {
    bert_embed(ws, w, static_cast<const int32_t *>(b.ids.data()),
               static_cast<const int32_t *>(b.types.data()),
               static_cast<const int32_t *>(b.mask.data()), ep, b.size, b.eff);
    std::vector<__half> h(size_t(b.size) * HIDDEN);
    emb.to_host(h.data(), h.size() * sizeof(__half));
    for (int k = 0; k < b.size; k++) {
      int si = order[b.start + k];
      for (int d = 0; d < HIDDEN; d++)
        got[size_t(si) * HIDDEN + d] = __half2float(h[size_t(k) * HIDDEN + d]);
    }
  }
  double cos = 0;
  for (int i = 0; i < n; i++)
    cos += cosine(&got[size_t(i) * HIDDEN], &ref[size_t(i) * HIDDEN], HIDDEN);
  cos /= n;

  printf("corpus: %d sentences, mean real length %.1f\n", n,
         std::accumulate(len.begin(), len.end(), 0.0) / n);
  printf("  mean cosine vs pytorch: %.5f\n", cos);
  printf("  padded   (seq 128)        %9.0f sent/s\n", padded_sps);
  printf("  dynamic  (eff %.0f avg)      %9.0f sent/s\n", double(eff_sum) / n,
         dyn_sps);
  return 0;
}
