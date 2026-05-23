#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/parity.h"
#include "kernels/gelu.h"
#include <cuda_fp16.h>
#include <vector>

int main() {
  constexpr int N = 128 * 1536;

  auto x_h = read_bin<__half>("ref/gelu_x.bin", N);
  auto y_ref = read_bin<float>("ref/gelu_y.bin", N);

  DeviceBuffer x(N * sizeof(__half));
  x.from_host(x_h.data(), N * sizeof(__half));

  launch_gelu(static_cast<__half *>(x.data()), N);

  std::vector<__half> y_h(N);
  x.to_host(y_h.data(), N * sizeof(__half));
  std::vector<float> y_actual(N);
  for (int i = 0; i < N; ++i)
    y_actual[i] = __half2float(y_h[i]);

  auto r = parity_fp32(y_ref, y_actual);
  return report("gelu", r, 0.9999) ? 0 : 1;
}
