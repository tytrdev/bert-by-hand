"""Dump a corpus of real sentences for the CUDA throughput benchmark.

Tokenizes the STS benchmark sentences to ref/corpus_*.bin and writes the
sentence-transformers reference embedding for each, so bench/corpus can measure
throughput over varied-length text and check correctness against pytorch.
"""

import numpy as np
import torch
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_NAME = "mteb/stsbenchmark-sts"
REF_DIR = Path("ref")
SEQ_LEN = 128
MAX_SENTENCES = 2000


def main():
    model = SentenceTransformer(MODEL_NAME).to("cuda").half()
    tok = model.tokenizer

    ds = load_dataset(CORPUS_NAME, split="test")
    seen: dict[str, int] = {}
    for row in ds:
        for s in (row["sentence1"], row["sentence2"]):
            if s not in seen and len(seen) < MAX_SENTENCES:
                seen[s] = len(seen)
    sentences = list(seen)
    n = len(sentences)

    enc = tok(
        sentences, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt"
    )
    ids = enc["input_ids"].numpy().astype(np.int32)
    mask = enc["attention_mask"].numpy().astype(np.int32)
    types = np.zeros_like(ids)

    embs = model.encode(sentences, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    ids.tofile(REF_DIR / "corpus_ids.bin")
    mask.tofile(REF_DIR / "corpus_mask.bin")
    types.tofile(REF_DIR / "corpus_types.bin")
    embs.astype(np.float32).tofile(REF_DIR / "corpus_emb.bin")

    real = mask.sum(axis=1)
    print(f"corpus: {n} sentences, seq={SEQ_LEN}")
    print(f"  real length: min={real.min()} mean={real.mean():.1f} max={real.max()}")
    print(f"  wrote ref/corpus_*.bin ({n} x {SEQ_LEN} ids, {n} x {embs.shape[1]} embeddings)")


if __name__ == "__main__":
    main()
