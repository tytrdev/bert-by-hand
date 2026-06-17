"""Length-aware benchmark. The reference sentence is 9 real tokens; the seq=128
number is PyTorch forced to pad to max_length. This shows what PyTorch does when
it processes the real length instead (which sentence-transformers does by
default via dynamic padding), so the comparison against our effective-length
path is apples to apples.
"""

import time
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE = "A girl is styling her hair."
BATCH = 64


def time_forward(bert, ids, am, tt, iters=300):
    def fwd():
        with torch.no_grad():
            o = bert(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
            mk = am.unsqueeze(-1).half()
            p = (o * mk).sum(1) / mk.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(p, dim=-1)

    for _ in range(30):
        fwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fwd()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / (iters * BATCH)


def main():
    m = SentenceTransformer(MODEL_NAME).to("cuda").half()
    bert = m[0].auto_model
    tok = m.tokenizer

    n_real = int(tok(SENTENCE, return_tensors="pt")["attention_mask"].sum())
    print(f"sentence: {SENTENCE!r} -> {n_real} real tokens\n")
    print(f"{'seq':>6}  {'ms/sent':>9}  {'sent/s':>10}")
    print("-" * 30)

    for seq in (128, 32, 16, n_real):
        enc = tok(
            [SENTENCE] * BATCH,
            padding="max_length",
            truncation=True,
            max_length=seq,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to("cuda")
        am = enc["attention_mask"].to("cuda")
        tt = torch.zeros_like(ids)
        ms = time_forward(bert, ids, am, tt)
        print(f"{seq:>6}  {ms:>9.4f}  {1000 / ms:>10.0f}")


if __name__ == "__main__":
    main()
