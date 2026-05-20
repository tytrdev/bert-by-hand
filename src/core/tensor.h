#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cuda_fp16.h>
#include <initializer_list>

class Tensor {
public:
  static constexpr int MaxRank = 4;

  Tensor(__half *ptr, std::initializer_list<int> dims);

  const __half *data() const { return ptr_; }
  int rank() const { return rank_; }
  int dim(int i) const {
    assert(i >= 0 && i < rank_);
    return dims_[i];
  }

  size_t numel() const;
  size_t bytes() const { return numel() * sizeof(__half); }

private:
  __half *ptr_;
  std::array<int, MaxRank> dims_{};
  int rank_;
};

inline Tensor::Tensor(__half *ptr, std::initializer_list<int> dims)
    : ptr_(ptr), rank_(static_cast<int>(dims.size())) {
  assert(rank_ <= MaxRank);

  int i = 0;
  for (int d : dims)
    dims_[i++] = d;
}

inline size_t Tensor::numel() const {
  size_t n = 1;

  for (int i = 0; i < rank_; i++) {
    n *= static_cast<size_t>(dims_[i]);
  }

  return n;
}
