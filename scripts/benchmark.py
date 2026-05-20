"""Reference Benchmark. Load model, print config."""

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_NAME = "mteb/stsbenchmark-sts"


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

    for s in sentences[:3]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
