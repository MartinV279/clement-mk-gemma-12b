#!/usr/bin/env python3
"""KenLM perplexity filter (CLAUDE.md data pipeline).

The KenLM is trained on the clean Wikipedia+books slice of the pinned MK
corpus (sources: mkwiki, MMORE). This script PROPOSES a rejection threshold
from the perplexity distribution; the USER confirms it before any filtering.

Subcommands:
  train   — extract slice, normalize, lmplz 5-gram, build_binary
            uv run python data/filters/kenlm_filter.py train \
                --corpus data/downloads/mk_corpus/macedonian_corpus_cleaned_deduplicated.jsonl.gz
  score   — perplexity distribution of held-out clean text vs a target file;
            prints percentiles and a PROPOSED threshold (needs user sign-off)
  filter  — apply a CONFIRMED threshold to a jsonl file
"""

import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

LMPLZ = ".tools/kenlm/build/bin/lmplz"
BUILD_BINARY = ".tools/kenlm/build/bin/build_binary"
DEFAULT_MODEL = "data/models/kenlm_mk_5gram.bin"
CLEAN_SOURCES = ("mkwiki", "MMORE")
_PUNCT_RE = re.compile(r"([^\w\s]|_)", re.UNICODE)


def normalize(text: str) -> str:
    """NFC, lowercase, punctuation split off as separate tokens, one space."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT_RE.sub(r" \1 ", text)
    return " ".join(text.split())


def open_maybe_gz(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def iter_docs(path: str, sources=None, field: str = "text"):
    with open_maybe_gz(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if sources and obj.get("source") not in sources:
                continue
            yield obj[field]


def doc_perplexity(model, text: str) -> float:
    sent = normalize(text)
    words = sent.count(" ") + 1
    return 10 ** (-model.score(sent) / max(words, 1))


def cmd_train(args) -> None:
    out_dir = Path(args.model).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "kenlm_train.txt"
    holdout_txt = out_dir / "kenlm_holdout.txt"
    n_train = n_hold = 0
    with train_txt.open("w", encoding="utf-8") as ftr, holdout_txt.open("w", encoding="utf-8") as fho:
        for i, text in enumerate(iter_docs(args.corpus, sources=CLEAN_SOURCES)):
            line = normalize(text) + "\n"
            if i % 50 == 0:  # 2% held out for the threshold proposal
                fho.write(line); n_hold += 1
            else:
                ftr.write(line); n_train += 1
    print(f"slice extracted: {n_train} train docs, {n_hold} holdout docs")

    arpa = str(Path(args.model).with_suffix(".arpa"))
    subprocess.run([LMPLZ, "-o", str(args.order), "-S", "40%", "--skip_symbols",
                    "--text", str(train_txt), "--arpa", arpa], check=True)
    subprocess.run([BUILD_BINARY, arpa, args.model], check=True)
    Path(arpa).unlink()  # keep only the compact binary
    train_txt.unlink()
    print(f"model: {args.model} | holdout kept: {holdout_txt}")


def _percentiles(values, ps=(50, 75, 90, 95, 99)):
    values = sorted(values)
    return {p: values[min(len(values) - 1, int(len(values) * p / 100))] for p in ps}


def cmd_score(args) -> None:
    import kenlm
    model = kenlm.Model(args.model)

    holdout = Path(args.model).parent / "kenlm_holdout.txt"
    clean_ppl = []
    with holdout.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words = line.count(" ") + 1
                clean_ppl.append(10 ** (-model.score(line.strip()) / max(words, 1)))

    print(f"clean holdout ({len(clean_ppl)} docs) ppl percentiles:")
    pct = _percentiles(clean_ppl)
    for p, v in pct.items():
        print(f"  p{p}: {v:,.0f}")
    proposed = pct[99]
    print(f"\nPROPOSED threshold: ppl > {proposed:,.0f} (p99 of clean holdout) => reject")
    print("USER must confirm before this threshold is applied (CLAUDE.md).")

    if args.input:
        target_ppl = [doc_perplexity(model, t) for t in iter_docs(args.input, field=args.field)]
        tp = _percentiles(target_ppl)
        rej = sum(1 for v in target_ppl if v > proposed)
        print(f"\ntarget {args.input} ({len(target_ppl)} docs) percentiles: "
              + ", ".join(f"p{p}={v:,.0f}" for p, v in tp.items()))
        print(f"would reject {rej}/{len(target_ppl)} ({rej/max(len(target_ppl),1):.1%}) at proposed threshold")


def cmd_filter(args) -> None:
    import kenlm
    model = kenlm.Model(args.model)
    kept = dropped = 0
    with open_maybe_gz(args.input) as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if doc_perplexity(model, obj[args.field]) <= args.threshold:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"kept {kept}, dropped {dropped} (ppl > {args.threshold:,.0f})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--corpus", required=True)
    tr.add_argument("--model", default=DEFAULT_MODEL)
    tr.add_argument("--order", type=int, default=5)

    sc = sub.add_parser("score")
    sc.add_argument("--model", default=DEFAULT_MODEL)
    sc.add_argument("--input", help="optional target jsonl to preview rejection rate on")
    sc.add_argument("--field", default="text")

    fl = sub.add_parser("filter")
    fl.add_argument("--input", required=True)
    fl.add_argument("--output", required=True)
    fl.add_argument("--model", default=DEFAULT_MODEL)
    fl.add_argument("--threshold", type=float, required=True,
                    help="USER-CONFIRMED perplexity threshold")
    fl.add_argument("--field", default="text")

    args = ap.parse_args()
    {"train": cmd_train, "score": cmd_score, "filter": cmd_filter}[args.cmd](args)


if __name__ == "__main__":
    main()
