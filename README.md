# README

Crazy man attempts to outperform pytorch by writing raw CUDA.

Goal is to get a model generating embeddings against some corpus faster than pytorch on the same machine. Real goal is showing public signal that I have some skill with CUDA.

Writing this on CachyOS against a 3090.

Model: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2> (for now).

## Status

Full embedding forward runs in CUDA and matches the pytorch reference end to end (cosine 0.999999):

- embeddings
- 6 encoder layers (masked attention + FFN)
- masked mean pooling
- L2 normalize

Each kernel has its own parity test. fp16 weights with fp32 accumulation, fixed to BATCH=1, SEQ_LEN=128.

## Build / test

```bash
uv run python scripts/dump_weights.py --dump   # weights + reference tensors
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

## Results

Single sentence latency on the 3090 (batch 1, seq 128):

| impl              | ms / embedding | embeddings / sec |
| ----------------- | -------------- | ---------------- |
| this (tiled cuda) | 2.56           | 390              |
| pytorch gpu fp16  | ~2.1           | ~475             |

A shared-memory tiled matmul got us to roughly naive pytorch parity at batch 1. But batch 1 is pytorch at its weakest (launch bound, gpu mostly idle). The real target is corpus throughput, where pytorch batches and saturates the card (scripts/benchmark.py, batch 32):

| pytorch (batch 32, corpus) | ms / sent | sent / sec |
| -------------------------- | --------- | ---------- |
| gpu fp16                   | 0.092     | 10859      |
| gpu fp16 + torch.compile   | 0.119     | 8410       |
| gpu fp32                   | 0.193     | 5189       |

So we match pytorch where it is weak and lose ~28x where it is strong. Still plenty to do: tensor cores on the matmul, batching, and skipping padded tokens (the ref sentence is 9 real tokens out of 128).
