"""Reference Benchmark. Load model, print config."""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"loaded {MODEL_NAME}")
    print(model)


if __name__ == "__main__":
    main()
