"""Dump model weights
--inspect: print catalaog, no writes
--dump: write .bin files
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
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

REF_DIR = Path("ref")
REF_SENTENCE = "A girl is styling her hair."
SEQ_LEN = 128

REF_MATMUL_SHAPES = [
    ("small", 4, 16, 8),  # for sanity
    ("real", 128, 384, 384),  # qkv linear
]


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


def dump_ref(model):
    tokenizer = model.tokenizer

    enc = tokenizer(
        REF_SENTENCE, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt"
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    model_cpu = model.to("cpu").float()
    bert = get_bert(model_cpu)

    with torch.no_grad():
        out = bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embedding = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    input_ids.numpy().astype(np.int32).tofile(REF_DIR / "input_ids.bin")
    attention_mask.numpy().astype(np.int32).tofile(REF_DIR / "attention_mask.bin")
    token_type_ids.numpy().astype(np.int32).tofile(REF_DIR / "token_type_ids.bin")
    emb_np = embedding.numpy().astype(np.float32)
    emb_np.tofile(REF_DIR / "expected_embedding.bin")

    n_real = int(attention_mask.sum().item())
    print(f"\nref: {REF_SENTENCE!r}")
    print(f"  seq_len={SEQ_LEN}, real tokens={n_real}, padding={SEQ_LEN - n_real}")
    print(f"  embedding[:5] = {emb_np[0, :5]}")
    print(f"  L2 norm = {np.linalg.norm(emb_np)}")


def dump_embedding_ref(model):
    tokenizer = model.tokenizer
    enc = tokenizer(
        REF_SENTENCE, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    bert = get_bert(model.to("cpu").float())
    with torch.no_grad():
        emb = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

    y = emb[0].numpy().astype(np.float32)
    y.tofile(REF_DIR / "emb_out.bin")
    print(f"emb ref: shape={y.shape} |y|={np.linalg.norm(y):.4f}")


def dump_matmul_refs(model):
    print("\nDumping matmul refs")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for tag, M, N, K in REF_MATMUL_SHAPES:
        A = rng.standard_normal((M, K)).astype(np.float32) * 0.1
        B = rng.standard_normal((N, K)).astype(np.float32) * 0.1
        C = A @ B.T

        A.astype(np.float16).tofile(REF_DIR / f"matmul_{tag}_A.bin")
        B.astype(np.float16).tofile(REF_DIR / f"matmul_{tag}_B.bin")
        C.astype(np.float32).tofile(REF_DIR / f"matmul_{tag}_C.bin")
        print(f"matmul ref [{tag}]: M={M} N={N} K={K} |C|={np.linalg.norm(C):.4f}")


def dump_layernorm_ref():
    M, D = 128, 384
    eps = 1e-12
    rng = np.random.default_rng(1)

    x_f32 = (rng.standard_normal((M, D)) * 1.0).astype(np.float16).astype(np.float32)
    gamma_f32 = (rng.standard_normal((D,)) * 0.1 + 1.0).astype(np.float16).astype(np.float32)
    beta_f32 = (rng.standard_normal((D,)) * 0.1).astype(np.float16).astype(np.float32)

    y = F.layer_norm(
        torch.from_numpy(x_f32),
        (D,),
        weight=torch.from_numpy(gamma_f32),
        bias=torch.from_numpy(beta_f32),
        eps=eps,
    ).numpy()

    x_f32.astype(np.float16).tofile(REF_DIR / "ln_x.bin")
    gamma_f32.astype(np.float16).tofile(REF_DIR / "ln_gamma.bin")
    beta_f32.astype(np.float16).tofile(REF_DIR / "ln_beta.bin")
    y.astype(np.float32).tofile(REF_DIR / "ln_y.bin")
    print(f"ln ref: M={M} D={D} |y|={np.linalg.norm(y):.4f}")


def dump_gelu_ref():
    n = 128 * 1536  # B*S * FFN_DIM, the BERT FFN intermediate footprint
    rng = np.random.default_rng(2)
    x_f32 = (rng.standard_normal(n) * 1.5).astype(np.float16).astype(np.float32)
    y = F.gelu(torch.from_numpy(x_f32)).numpy()

    x_f32.astype(np.float16).tofile(REF_DIR / "gelu_x.bin")
    y.astype(np.float32).tofile(REF_DIR / "gelu_y.bin")
    print(f"gelu ref: n={n} |y|={np.linalg.norm(y):.4f}")


def dump_softmax_ref():
    M, N = 128, 128
    rng = np.random.default_rng(3)
    x_f32 = (rng.standard_normal((M, N)) * 2.0).astype(np.float16).astype(np.float32)
    y = F.softmax(torch.from_numpy(x_f32), dim=-1).numpy()

    x_f32.astype(np.float16).tofile(REF_DIR / "softmax_x.bin")
    y.astype(np.float32).tofile(REF_DIR / "softmax_y.bin")
    print(f"softmax ref: M={M} N={N} |y|={np.linalg.norm(y):.4f}")


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
        dump_ref(model)
        dump_embedding_ref(model)
        dump_matmul_refs(model)
        dump_layernorm_ref()
        dump_gelu_ref()
        dump_softmax_ref()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
