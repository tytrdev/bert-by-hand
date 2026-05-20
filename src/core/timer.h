#pragma once

#include "cuda_check.h"
#include <cuda_runtime.h>

class Timer {
public:
  Timer();
  ~Timer();

  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;

  void start();
  void stop();
  float elapsed_ms();
  void print();

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

inline Timer::Timer() {
  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
}

inline Timer::~Timer() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

inline void Timer::start() { CUDA_CHECK(cudaEventRecord(start_)); }

inline void Timer::stop() {
  CUDA_CHECK(cudaEventRecord(stop_));
  CUDA_CHECK(cudaEventSynchronize(stop_));
}

inline float Timer::elapsed_ms() {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
  return ms;
}

inline void Timer::print() {
  printf("Timer elapsed_ms: %.3f ms\n", this->elapsed_ms());
}
