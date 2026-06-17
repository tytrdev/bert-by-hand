"""PyTorch forward-only throughput over the STS corpus, the apples-to-apples
companion to bench/corpus. Pre-tokenizes, sorts by length, pads each batch to
its longest (what sentence-transformers does), and times only the GPU forward
so the comparison excludes tokenization and Python dispatch on both sides.
"""

import time
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_NAME = "mteb/stsbenchmark-sts"
MAX_SENTENCES = 2000
BATCH = 64
ITERS = 20


def main():
    model = SentenceTransformer(MODEL_NAME).to("cuda").half()
    bert = model[0].auto_model
    tok = model.tokenizer

    ds = load_dataset(CORPUS_NAME, split="test")
    seen: dict[str, int] = {}
    for row in ds:
        for s in (row["sentence1"], row["sentence2"]):
            if s not in seen and len(seen) < MAX_SENTENCES:
                seen[s] = len(seen)
    sentences = list(seen)

    lengths = [int(tok(s, return_tensors="pt")["attention_mask"].sum()) for s in sentences]
    order = sorted(range(len(sentences)), key=lambda i: lengths[i])

    batches = []
    eff_sum = 0
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        enc = tok([sentences[i] for i in idx], padding=True, return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        am = enc["attention_mask"].to("cuda")
        tt = torch.zeros_like(ids)
        eff_sum += ids.shape[0] * ids.shape[1]
        batches.append((ids, am, tt))

    def run():
        for ids, am, tt in batches:
            with torch.no_grad():
                o = bert(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state
                mk = am.unsqueeze(-1).half()
                p = (o * mk).sum(1) / mk.sum(1).clamp(min=1e-9)
                torch.nn.functional.normalize(p, dim=-1)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run()
    torch.cuda.synchronize()
    sps = len(sentences) * ITERS / (time.perf_counter() - t0)

    print(f"pytorch corpus forward: {len(sentences)} sentences")
    print(f"  dynamic (eff {eff_sum / len(sentences):.0f} avg)  {sps:>9.0f} sent/s")


if __name__ == "__main__":
    main()
