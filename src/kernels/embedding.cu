#include "core/cuda_check.h"
#include "kernels/embedding.h"

namespace {

constexpr int EMB_BLOCK = 256;

__global__ void embedding_kernel(const int32_t *__restrict__ input_ids,
                                 const int32_t *__restrict__ token_type_ids,
                                 const __half *__restrict__ word_emb,
                                 const __half *__restrict__ pos_emb,
                                 const __half *__restrict__ type_emb,
                                 __half *__restrict__ out, int seq_len,
                                 int hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hidden;
  if (idx >= total)
    return;

  int s = idx / hidden;
  int h = idx % hidden;

  int wid = input_ids[s];
  int tid = token_type_ids[s];

  float w = __half2float(word_emb[size_t(wid) * hidden + h]);
  float p = __half2float(pos_emb[size_t(s) * hidden + h]);
  float t = __half2float(type_emb[size_t(tid) * hidden + h]);
  out[idx] = __float2half(w + p + t);
}

} // namespace

void launch_embedding(const int32_t *input_ids, const int32_t *token_type_ids,
                      const __half *word_emb, const __half *pos_emb,
                      const __half *type_emb, __half *out, int seq_len,
                      int hidden) {
  int total = seq_len * hidden;
  dim3 block(EMB_BLOCK);
  dim3 grid((total + EMB_BLOCK - 1) / EMB_BLOCK);
  embedding_kernel<<<grid, block>>>(input_ids, token_type_ids, word_emb,
                                    pos_emb, type_emb, out, seq_len, hidden);
  CUDA_CHECK_KERNEL();
}
