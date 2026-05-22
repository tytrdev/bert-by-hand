#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/parity.h"
#include "kernels/layernorm.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

int main() {
  constexpr int M = 128;
  constexpr int D = 384;
  constexpr float EPS = 1e-12f;
  const size_t mat = size_t(M) * D;

  auto x_h = read_bin<__half>("ref/ln_x.bin", mat);
  auto gamma_h = read_bin<__half>("ref/ln_gamma.bin", D);
  auto beta_h = read_bin<__half>("ref/ln_beta.bin", D);
  auto y_ref = read_bin<float>("ref/ln_y.bin", mat);

  DeviceBuffer x(mat * sizeof(__half));
  DeviceBuffer g(D * sizeof(__half));
  DeviceBuffer b(D * sizeof(__half));
  DeviceBuffer y(mat * sizeof(__half));
  x.from_host(x_h.data(), mat * sizeof(__half));
  g.from_host(gamma_h.data(), D * sizeof(__half));
  b.from_host(beta_h.data(), D * sizeof(__half));

  launch_layernorm(static_cast<const __half *>(x.data()),
                   static_cast<const __half *>(g.data()),
                   static_cast<const __half *>(b.data()),
                   static_cast<__half *>(y.data()), M, D, EPS);

  std::vector<__half> y_h(mat);
  y.to_host(y_h.data(), mat * sizeof(__half));
  std::vector<float> y_actual(mat);
  for (size_t i = 0; i < mat; ++i)
    y_actual[i] = __half2float(y_h[i]);

  auto r = parity_fp32(y_ref, y_actual);
  return report("layernorm", r, 0.9995) ? 0 : 1;
}
