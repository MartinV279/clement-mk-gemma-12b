#!/usr/bin/env python3
"""Embedding near-dedup for SFT/synthetic data (CLAUDE.md data pipeline).

Embeds documents with a multilingual sentence-transformer on the local GPU
(approved local workload) and drops near-duplicates above a cosine threshold,
keeping the first occurrence. Intended for SFT-scale data (tens of thousands
of docs); the CPT corpus is already deduplicated upstream.

Usage:
  uv run python data/filters/dedup.py --input in.jsonl --output out.jsonl \
      [--threshold 0.92] [--field text]
"""

import argparse
import gzip
import json
import sys

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--field", default="text")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    lines, texts = [], []
    with open_maybe_gz(args.input) as f:
        for line in f:
            if line.strip():
                lines.append(line)
                texts.append(json.loads(line)[args.field][:2000])

    model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    emb = model.encode(texts, batch_size=args.batch_size, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)

    keep = torch.ones(len(texts), dtype=torch.bool)
    # chunked pairwise sim: doc i is dropped if similar to any KEPT doc j < i
    chunk = 1024
    for start in range(0, len(texts), chunk):
        end = min(start + chunk, len(texts))
        sims = emb[start:end] @ emb[:end].T  # (chunk, end)
        for i in range(start, end):
            if not keep[i]:
                continue
            row = sims[i - start, :i]
            if row.numel() and bool((row[keep[:i]] > args.threshold).any()):
                keep[i] = False

    kept = int(keep.sum())
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, line in enumerate(lines):
            if keep[i]:
                fout.write(line if line.endswith("\n") else line + "\n")
    print(f"kept {kept}/{len(lines)} ({kept/len(lines):.1%}), "
          f"dropped {len(lines)-kept} near-dups @ cos>{args.threshold}", file=sys.stderr)


if __name__ == "__main__":
    main()
