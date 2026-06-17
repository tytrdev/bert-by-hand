# bert-by-hand

Raw CUDA inference for `all-MiniLM-L6-v2` sentence embeddings. No cuBLAS, no
CUTLASS, no libraries — every kernel hand-written. Built on CachyOS against a
3090.

Goal: match or beat PyTorch (which is cuBLAS underneath) on the same machine.
Result: **parity at seq=128, ~1.7x faster at the real sequence length.**

## Build / test / bench

```bash
uv run python scripts/dump_weights.py --dump   # weights + reference tensors
cmake -S . -B build && cmake --build build -j
ctest --test-dir build                         # every kernel parity-checked vs torch
./build/bench                                  # ours, full + effective length
uv run python scripts/benchmark_varlen.py      # pytorch length sweep

uv run python scripts/dump_corpus.py           # tokenize 2000 STS sentences + torch refs
./build/corpus                                 # ours, real corpus throughput + correctness
uv run python scripts/benchmark_corpus.py      # pytorch, same corpus
```

fp16 weights, fp32 reference. Every kernel has a parity test; end-to-end cosine
vs the torch embedding is 0.9999.

## Result (batch 64, same 9-token sentence)

| seq processed        | ours ms/sent | pytorch ms/sent |
| -------------------- | ------------ | --------------- |
| 128 (forced padding) | 0.090        | 0.092           |
| 16 (effective length)| **0.015**    | 0.025           |
| 9 (pytorch tightest) | — (16 floor) | 0.023           |

Parity where compute dominates; a real win where per-forward overhead dominates.

Over 2000 real sentences (STS benchmark, varied length), forward only, each
batch padded to its longest like sentence-transformers does. Mean cosine vs the
torch embedding is 1.00000.

| approach                  | throughput     |
| ------------------------- | -------------- |
| ours, padded to 128       | 11,300 sent/s  |
| ours, dynamic length      | **54,000 sent/s** |
| pytorch, dynamic length   | 39,000 sent/s  |

## How (each step is one commit; batch-64 ms/sent)

Single sentence, working up the kernel stack: naive 7.0 → tiled shared mem 2.6 →
tensor cores (wmma) 0.91 → masked attention parity. Then batched (M = B*seq) and
the real work begins:

| step | ms/sent | idea |
| ---- | ------- | ---- |
| batched baseline (simple wmma) | 0.225 | one warp per 16x16 output tile |
| cp.async pipelined matmul      | 0.140 | overlap global load with mma |
| fuse qkv + read from it        | 0.118 | one projection, no split/merge |
| warp-per-row layernorm         | 0.109 | shuffle reduction, no shared/sync |
| fuse gelu into matmul epilogue | 0.102 | drop a kernel per layer |
| register-block attention scores| 0.101 | 32x32 warp tile |
| **fp16 tensor core accumulate**| 0.091 | 2x rate on GeForce; **parity** |
| **effective sequence length**  | 0.015 | skip padding; **beats pytorch** |

**cp.async pipeline** (the biggest GEMM win): double-buffer the next K-slice in
shared while the tensor cores chew the current one.

```cuda
__pipeline_memcpy_async(&As[st][i], &A[...], 16);   // global -> shared, no stall
__pipeline_commit();
// ... compute on As[cur] while As[cur^1] loads
__pipeline_wait_prior(1);
```

**fp16 accumulate** — GeForce caps `fp16->fp32` tensor ops at half the
`fp16->fp16` rate. Six layers tolerate the precision loss (cos stays 0.9999):

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c;  // was float
```

**fused QKV** — concat the three projection weights, one matmul, then read each
slice straight out with a strided wmma load (no split-heads buffer):

```cuda
wmma::load_matrix_sync(af, qkv + row * qkv_stride + q_off + d0, qkv_stride);
```

**effective length** — the sentence is 9 real tokens; we (and the seq=128
benchmark) burn 87% of every matmul on padding that gets masked then averaged
away. Process the real length (rounded to 16); the embedding is identical.

```cpp
int eff = ((n_real + 15) / 16) * 16;
bert_embed(ws, w, ids, types, mask, emb, batch, eff);   // seq is a runtime arg
```

## What did not work, and why

- **Hand-blocked shared-memory GEMM** (tried twice, 32x32 and 64x64): *slower*
  than the simple per-warp wmma. The simple kernel already gets its reuse from
  L2; explicit shared staging just adds syncs and cuts occupancy.
- **3–5 stage pipeline** (what cuBLAS uses): slower for us. Our K is small
  (384/1536), so a 2-stage buffer already hides the latency; more stages burn
  shared memory and occupancy for nothing. PyTorch's deep pipelines are tuned
  for general/large K.
- **Fused flash attention** (scores+softmax+context in one block, scores in
  shared): *slower*. 32 KB of shared scores → 1 block/SM, and the three
  serialized phases lose more than the saved global traffic (L2 caches it). A
  flash kernel that wins needs key-blocking + online softmax to keep occupancy —
  that is `fmha_cutlassF`.
- **Shared-memory padding / bigger tiles**: net wash. Bank conflicts weren't the
  bottleneck, and big tiles only help large-N GEMMs; ours are small-N.

## Insights

- **PyTorch is ~70% GEMM too** (profiled both sides with nsys). There is no
  magic kernel we missed — the GEMM is the wall for both.
- **We're ahead on fusion.** PyTorch runs gelu, bias, and residual as *separate*
  kernels; we fold them into matmul/layernorm epilogues. That advantage is what
  lets a simpler 2-stage GEMM reach parity with cuBLAS.
- **fp16 accumulate is the lever that closes the GEMM gap** — cuBLAS keeps fp32
  accumulate (the safe default) and eats the GeForce half-rate penalty.
- **The length win is overhead, not compute.** At seq=16 the math is tiny, so
  the forward is launch/dispatch bound, and our fused, fixed-shape, zero-alloc
  kernel set has far less overhead than PyTorch's many kernels + Python. The
  length speedup itself helps both engines equally — it's a "do less work" win,
  not out-engineering cuBLAS.
