#pragma once

#include <cuda_fp16.h>

// Torch uses exact gelu, not tanh-approx
// Do I need to worry about rounding errors/precision?
// Element wise, in place.
// y[i] = x[i] * 0.5 * (1 + erf(x[i] / sqrt(2)))
void launch_gelu(__half *x, int n);
