#!/usr/bin/env python3
"""Language-ID filter: keep documents fastText classifies as Macedonian with
confidence > 0.65 (CLAUDE.md data pipeline, first stage after raw).

Usage:
  uv run python data/filters/langid_filter.py --input in.jsonl[.gz] --output out.jsonl \
      [--model data/models/lid.176.bin] [--min-conf 0.65] [--field text]
"""

import argparse
import gzip
import json
import sys


def open_maybe_gz(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def predict_lang(model, text: str, k: int = 1):
    """fasttext<=0.9.3 predict() crashes under numpy>=2 (copy=False); the
    low-level binding avoids numpy entirely. Returns (labels, confs)."""
    preds = model.f.predict(text.replace("\n", " "), k, 0.0, "strict")
    if not preds:
        return [], []
    confs, labels = zip(*preds)
    return list(labels), list(confs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="data/models/lid.176.bin")
    ap.add_argument("--min-conf", type=float, default=0.65)
    ap.add_argument("--lang", default="mk")
    ap.add_argument("--field", default="text")
    args = ap.parse_args()

    import fasttext
    model = fasttext.load_model(args.model)
    label = f"__label__{args.lang}"

    kept = dropped = 0
    with open_maybe_gz(args.input) as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            # first 2000 chars are plenty for LID
            labels, confs = predict_lang(model, obj[args.field][:2000])
            if labels and labels[0] == label and confs[0] > args.min_conf:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            else:
                dropped += 1
    total = kept + dropped
    print(f"kept {kept}/{total} ({kept/total:.1%}), dropped {dropped} "
          f"(not {args.lang}@conf>{args.min_conf})", file=sys.stderr)


if __name__ == "__main__":
    main()
