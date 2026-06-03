#pragma once

#include <cuda_fp16.h>

// Elementwise residual add, in place: x[i] += y[i]
void launch_residual_add(__half *x, const __half *y, int n);
