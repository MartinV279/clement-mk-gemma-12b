#!/usr/bin/env python3
"""RAG prototype step 1 — embed the mkwiki slice of the corpus on the 8GB GPU.

Passages: mkwiki docs from mk_corpus_filtered, chunked to ~1200 chars.
Embedder: BAAI/bge-m3 (multilingual, strong on MK); fp16 on the 4060.
Output:  rag/index/passages.jsonl + embeddings.npy (float16, normalized)

  .venv/bin/python rag/build_index.py [--limit-docs 0]
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def chunks(text: str, size: int = 1200, overlap: int = 150):
    step = size - overlap
    for i in range(0, max(len(text) - overlap, 1), step):
        c = text[i:i + size]
        if len(c) >= 200:
            yield c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out = Path("rag/index")
    out.mkdir(parents=True, exist_ok=True)

    passages = []
    n_docs = 0
    with gzip.open("data/processed/mk_corpus_filtered.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "mkwiki":
                continue
            n_docs += 1
            if args.limit_docs and n_docs > args.limit_docs:
                break
            for c in chunks(r["text"]):
                passages.append(c)
    print(f"{n_docs} mkwiki docs -> {len(passages)} passages", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", model_kwargs={"torch_dtype": "float16"})
    emb = model.encode(passages, batch_size=args.batch, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(out / "embeddings.npy", emb.astype(np.float16))
    with (out / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    print(f"index: {emb.shape} -> rag/index/")


if __name__ == "__main__":
    main()
