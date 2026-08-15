#!/usr/bin/env python3
"""Retrieval eval: for each verified fact question, does top-k retrieval surface
the gold passage? Gold match = a retrieved chunk sharing a 200-char run with
the stored gold passage."""
import json
import numpy as np

qs = [json.loads(l) for l in open("data/processed/facts_verified.jsonl", encoding="utf-8")]
passages = [json.loads(l)["text"] for l in open("rag/index/passages.jsonl", encoding="utf-8")]
emb = np.load("rag/index/embeddings.npy").astype(np.float32)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", model_kwargs={"torch_dtype": "float16"})
qv = model.encode([q["user"] for q in qs], batch_size=64,
                  normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

scores = qv @ emb.T                      # (968, 61778)
topk = np.argpartition(-scores, 5, axis=1)[:, :5]

def overlaps(gold, chunk, n=200):
    step = 100
    for i in range(0, max(len(gold) - n, 1), step):
        if gold[i:i+n] in chunk or chunk[:n] in gold:
            return True
    return False

hits1 = hits5 = 0
for i, q in enumerate(qs):
    idx = topk[i][np.argsort(-scores[i][topk[i]])]
    got = [overlaps(q["passage"], passages[j]) for j in idx]
    hits1 += got[0]
    hits5 += any(got)
n = len(qs)
print(f"retrieval hit@1: {hits1}/{n} = {hits1/n:.1%}")
print(f"retrieval hit@5: {hits5}/{n} = {hits5/n:.1%}")
