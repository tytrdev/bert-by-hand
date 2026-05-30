#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/parity.h"
#include "kernels/softmax.h"
#include <cuda_fp16.h>
#include <vector>

int main() {
  constexpr int M = 128;
  constexpr int N = 128;
  const size_t mat = size_t(M) * N;

  auto x_h = read_bin<__half>("ref/softmax_x.bin", mat);
  auto y_ref = read_bin<float>("ref/softmax_y.bin", mat);

  DeviceBuffer x(mat * sizeof(__half));
  x.from_host(x_h.data(), mat * sizeof(__half));

  launch_softmax(static_cast<__half *>(x.data()), M, N);

  std::vector<__half> y_h(mat);
  x.to_host(y_h.data(), mat * sizeof(__half));
  std::vector<float> y_actual(mat);
  for (size_t i = 0; i < mat; ++i)
    y_actual[i] = __half2float(y_h[i]);

  auto r = parity_fp32(y_ref, y_actual);
  return report("softmax", r, 0.9999) ? 0 : 1;
}
