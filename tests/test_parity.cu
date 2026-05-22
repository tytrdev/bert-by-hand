#include "core/device_buffer.h"
#include "core/loader.h"
#include "core/model_config.h"
#include "core/parity.h"
#include "core/tensor.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

int main() {
  using namespace model;

  // Refs
  auto ids_buf = load_ref_i32("input_ids", BATCH * SEQ_LEN);
  auto mask_buf = load_ref_i32("attention_mask", BATCH * SEQ_LEN);
  auto exp_emb = load_ref_fp32("expected_embedding", BATCH * HIDDEN);

  // Spot check
  auto wemb_buf = load_weight_fp16("embeddings.word_embeddings.weight",
                                   VOCAB_SIZE * HIDDEN);
  Tensor wemb(reinterpret_cast<__half *>(wemb_buf.data()),
              {VOCAB_SIZE, HIDDEN});

  printf("ids buf:     %zu bytes\n", ids_buf.bytes());
  printf("mask buf:    %zu bytes\n", mask_buf.bytes());
  printf("exp emb:     %zu floats, |emb|=%.6f\n", exp_emb.size(),
         parity_fp32(exp_emb, exp_emb).a_norm);
  printf("word emb:    shape=(%d, %d) numel=%zu bytes=%zu\n", wemb.dim(0),
         wemb.dim(1), wemb.numel(), wemb.bytes());

  return 0;
}
