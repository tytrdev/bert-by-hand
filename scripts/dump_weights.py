"""Dump model weights
--inspect: print catalaog, no writes
--dump: write .bin files
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# TODO: Make this an arg
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CONFIG_KEYS = [
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "type_vocab_size",
    "layer_norm_eps",
    "hidden_act",
]

WEIGHTS_DIR = Path("weights")
SKIP_PREFIXES = ("pooler.",)


def get_bert(model):
    # Could be more complex for other models later...
    return model[0].auto_model


def inspect(model):
    bert = get_bert(model)
    cfg = bert.config

    print("=== Model Config ===")

    for k in MODEL_CONFIG_KEYS:
        print(f"  {k}: {getattr(cfg, k)}")

    print("=== state_dict Tensors ===")
    total_params = 0
    total_bytes_fp16 = 0
    for name, t in bert.state_dict().items():
        shape = tuple(t.shape)
        n = t.numel()
        total_params += n
        total_bytes_fp16 += n * 2  # I'm not convinced...
        print(f"  {name:<60} {str(shape):<22} {str(t.dtype):<14} {n:>10}")

    print(f"\ntotal tensors: {len(bert.state_dict())}")
    print(f"total params: {total_params}")
    print(f"fp16 footprint: {total_bytes_fp16 / 1e6:.2f} MB")


def dump(model):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    bert = get_bert(model)
    written, skipped = [], []

    for name, t in bert.state_dict().items():
        if name.startswith(SKIP_PREFIXES):
            skipped.append(name)
            continue

        arr = t.detach().to(torch.float16).cpu().contiguous().numpy()
        path = WEIGHTS_DIR / f"{name}.bin"
        arr.tofile(path)
        written.append((name, arr.shape, path))

    # Sanity check
    name0, shape0, path0 = written[0]
    expected = bert.state_dict()[name0].detach().to(torch.float16).cpu().contiguous().numpy()
    actual = np.fromfile(path0, dtype=np.float16).reshape(shape0)
    assert np.array_equal(expected, actual), f"round-trip mismatch on {name0}"

    total_bytes = sum(p.stat().st_size for _, _, p in written)
    print(f"total:  {total_bytes / 1e6:.2f} MB on disk")
    print(f"wrote   {len(written):>3} tensors to {WEIGHTS_DIR}")
    print(f"skipped {len(skipped):>3}: {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dump", action="store_true")
    # ap.add_argument("--model", action="store_true")
    args = ap.parse_args()

    # Make an arg later
    model = SentenceTransformer(MODEL_NAME)

    if args.inspect:
        inspect(model)
    elif args.dump:
        dump(model)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
