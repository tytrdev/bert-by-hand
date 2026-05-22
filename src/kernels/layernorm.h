#pragma once

#include <cuda_fp16.h>

void launch_layernorm(const __half *x, const __half *gamma, const __half *beta,
                      __half *y, int M, int D, float eps);
