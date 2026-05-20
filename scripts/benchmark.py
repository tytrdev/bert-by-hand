"""Reference Benchmark. Load model, print config."""

import time
import numpy as np
import torch
from dataclasses import dataclass
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_NAME = "mteb/stsbenchmark-sts"


@dataclass
class BenchResult:
    name: str
    embeddings: np.ndarray()
    elapsed_ms: float


def correlate(result, pairs, gold):
    embs = result.embeddings
    idx = np.array(pairs)
    a = embs[idx[:, 0]]
    b = embs[idx[:, 1]]
    sims = np.einsum("ij,ij->i", a, b)
    rho, _ = spearmanr(sims, gold)
    return float(rho)


def encode_and_time(model, sentences, name, batch_size=32):
    on_cuda = next(model.parameters()).device.type == "cuda"
    _ = model.encode(
        sentences[:batch_size],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if on_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    embs = model.encode(
        sentences, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )

    if on_cuda:
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    return BenchResult(name=name, embeddings=embs, elapsed_ms=(t1 - t0) * 1000)


def load_corpus():
    ds = load_dataset(CORPUS_NAME, split="test")
    seen: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    scores: list[float] = []

    for row in ds:
        s1, s2 = row["sentence1"], row["sentence2"]
        i1 = seen.setdefault(s1, len(seen))
        i2 = seen.setdefault(s2, len(seen))
        pairs.append((i1, i2))
        scores.append(float(row["score"]))

    sentences = list(seen.keys())
    return sentences, pairs, scores


def run_all(sentences):
    results = []
    model = SentenceTransformer(MODEL_NAME)

    model.to("cpu").float()
    results.append(encode_and_time(model, sentences, "cpu_fp32"))

    model.to("cuda").float()
    results.append(encode_and_time(model, sentences, "gpu_fp32"))

    model.to("cuda").half()
    results.append(encode_and_time(model, sentences, "gpu_fp16"))

    return results


def main():
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"loaded {MODEL_NAME}")
    print(model)

    sentences, pairs, scores = load_corpus()
    print(
        f"corpus: {len(sentences)} unique sentences, ",
        f"{len(pairs)} pairs, ",
        f"score range[{min(scores):.2f}, {max(scores):.2f}]",
    )

    results = run_all(sentences)

    print(f"\n{'config':<12} {'total ms':>10} {'ms/sent':>10} {'sent/s':>10}, {'rho':>8}")
    print("-" * 55)

    for r in results:
        n = len(sentences)
        per = r.elapsed_ms / n
        thr = n / (r.elapsed_ms / 1000)
        rho = correlate(r, pairs, scores)
        print(f"{r.name:<12} {r.elapsed_ms:>10.1f} {per:>10.3f} {thr:>10.1f} {rho:>8.4f}")


if __name__ == "__main__":
    main()
