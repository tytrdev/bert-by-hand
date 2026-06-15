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
| pytorch gpu fp16  | 1.60           | 625              |
| this (naive cuda) | 7.37           | 136              |

Parity is done. The CUDA path is still the naive version (every kernel syncs, scratch reallocated each call, textbook matmul), so pytorch wins for now. Next up is perf.
