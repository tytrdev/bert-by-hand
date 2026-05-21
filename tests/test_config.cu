#include "core/model_config.h"
#include <cstdio>

int main() {
  using namespace model;
  static_assert(HIDDEN == NUM_HEADS * HEAD_DIM);
  static_assert(FFN_DIM == 4 * HIDDEN);
  static_assert(LAYER_NORM_EPS > 0.0f);
  printf("config OK\n");
  return 0;
}
