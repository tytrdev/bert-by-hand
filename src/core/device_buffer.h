#pragma once

#include "cuda_check.h"
#include <cstddef>
#include <cuda_runtime.h>

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  void *data() { return ptr_; }
  const void *data() const { return ptr_; }
  size_t bytes() const { return bytes_; }

  void from_host(const void *src, size_t n);
  void to_host(void *dst, size_t n) const;

private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
};

inline DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  CUDA_CHECK(cudaMalloc(&ptr_, bytes));
}

inline DeviceBuffer::~DeviceBuffer() {
  if (ptr_)
    cudaFree(ptr_);
}

inline DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

inline DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    if (ptr_)
      cudaFree(ptr_);
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }

  return *this;
};

inline void DeviceBuffer::from_host(const void *src, size_t n) {
  CUDA_CHECK(cudaMemcpy(ptr_, src, n, cudaMemcpyHostToDevice));
}

inline void DeviceBuffer::to_host(void *dst, size_t n) const {
  CUDA_CHECK(cudaMemcpy(dst, ptr_, n, cudaMemcpyDeviceToHost));
}
