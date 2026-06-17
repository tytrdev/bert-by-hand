#pragma once

#include <cuda_fp16.h>

// L2 normalize a single vector of length n, in place. Divides by the norm
// clamped to a small floor, matching the sentence-transformers normalize step.
void launch_l2_normalize(__half *x, int n, int batch = 1);
