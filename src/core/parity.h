#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

struct ParityReport {
  double cosine;
  double max_abs_diff;
  double a_norm;
  double b_norm;
};

inline ParityReport parity_fp32(const std::vector<float> &a,
                                const std::vector<float> &b) {
  if (a.size() != b.size()) {
    std::fprintf(stderr, "parity: size mismatch %zu vs %zu\n", a.size(),
                 b.size());
    std::abort();
  }

  double dot = 0.0, na = 0.0, nb = 0.0, mad = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    double x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
    double d = std::fabs(x - y);
    if (d > mad)
      mad = d;
  }

  double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  return {cos, mad, std::sqrt(na), std::sqrt(nb)};
}

inline bool report(const char *tag, const ParityReport &r,
                   double cos_min = 0.9999) {
  bool pass = r.cosine >= cos_min;
  std::printf("[%s cos=%.6f max_abs=%.6e |a|=%.4f |b| = %.4f %s\n", tag,
              r.cosine, r.max_abs_diff, r.a_norm, r.b_norm,
              pass ? "PASS" : "FAIL");
  return pass;
}
