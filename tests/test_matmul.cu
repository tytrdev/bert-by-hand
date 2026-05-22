#include "core/cuda_check.h"
#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/parity.h"
#include "kernels/matmul.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

struct Shape {
  const char *tag;
  int M;
  int N;
  int K;
};

static bool run_case(const Shape &s) {
  const size_t a_n = size_t(s.M) * s.K;
  const size_t b_n = size_t(s.N) * s.K;
  const size_t c_n = size_t(s.M) * s.N;

  auto A_h =
      read_bin<__half>("ref/matmul_" + std::string(s.tag) + "_A.bin", a_n);
  auto B_h =
      read_bin<__half>("ref/matmul_" + std::string(s.tag) + "_B.bin", b_n);
  auto C_ref =
      read_bin<float>("ref/matmul_" + std::string(s.tag) + "_C.bin", c_n);

  DeviceBuffer A(a_n * sizeof(__half));
  DeviceBuffer B(b_n * sizeof(__half));
  DeviceBuffer C(c_n * sizeof(__half));
  A.from_host(A_h.data(), a_n * sizeof(__half));
  B.from_host(B_h.data(), b_n * sizeof(__half));

  launch_matmul(static_cast<const __half *>(A.data()),
                static_cast<const __half *>(B.data()),
                static_cast<__half *>(C.data()), s.M, s.N, s.K);

  std::vector<__half> C_h(c_n);
  C.to_host(C_h.data(), c_n * sizeof(__half));
  std::vector<float> C_actual(c_n);
  for (size_t i = 0; i < c_n; i++)
    C_actual[i] = __half2float(C_h[i]);

  auto r = parity_fp32(C_ref, C_actual);
  return report(s.tag, r, 0.9995);
}

int main() {
  bool ok = true;
  ok &= run_case({"small", 4, 16, 8});
  ok &= run_case({"real", 128, 384, 384});
  return ok ? 0 : 1;
}
