#pragma once

#include "core/device_buffer.h"
#include "core/file_io.h"
#include "core/model_config.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

inline DeviceBuffer load_weight_fp16(const std::string &name, size_t numel) {
  auto host = read_bin<__half>(
      std::string(model::WEIGHTS_DIR) + "/" + name + ".bin", numel);
  DeviceBuffer buf(numel * sizeof(__half));
  buf.from_host(host.data(), host.size() * sizeof(__half));
  return buf;
}

inline DeviceBuffer load_ref_i32(const std::string &name, size_t numel) {
  auto host = read_bin<int32_t>("ref/" + name + ".bin", numel);
  DeviceBuffer buf(numel * sizeof(int32_t));
  buf.from_host(host.data(), host.size() * sizeof(int32_t));
  return buf;
}

inline std::vector<float> load_ref_fp32(const std::string &name, size_t numel) {
  return read_bin<float>("ref/" + name + ".bin", numel);
}
