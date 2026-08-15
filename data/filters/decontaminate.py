#!/usr/bin/env python3
"""Decontamination: enforce CLAUDE.md hard rule 8 — the frozen vibes set (and
eval benchmarks) must NEVER leak into any training dataset.

Method: normalized word 8-gram overlap. Contamination sources shorter than 8
words are matched as whole normalized substrings instead. Any training doc
that overlaps is dropped and logged.

Usage:
  uv run python data/filters/decontaminate.py --input train.jsonl --output clean.jsonl \
      [--vibes eval/vibes_prompts.jsonl] [--extra bench1.jsonl ...] [--field text] [--n 8]
"""

import argparse
import gzip
import json
import re
import sys
import unicodedata

_norm_re = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_words(text: str) -> list:
    text = unicodedata.normalize("NFC", text).lower()
    return _norm_re.sub(" ", text).split()


def ngrams(words: list, n: int):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def contamination_sets(paths: list, n: int):
    """Collect n-gram set + short-text substring list from all eval sources."""
    grams, shorts = set(), []
    for path in paths:
        with open_maybe_gz(path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # take every string field — prompts, answers, rubrics
                for v in obj.values():
                    if not isinstance(v, str) or not v.strip():
                        continue
                    words = normalize_words(v)
                    if len(words) >= n:
                        grams |= ngrams(words, n)
                    elif len(words) >= 3:
                        shorts.append(" ".join(words))
    return grams, shorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vibes", default="eval/vibes_prompts.jsonl")
    ap.add_argument("--extra", nargs="*", default=[], help="benchmark jsonl files")
    ap.add_argument("--field", default="text")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--log", default=None, help="optional jsonl log of dropped docs")
    args = ap.parse_args()

    grams, shorts = contamination_sets([args.vibes] + args.extra, args.n)
    print(f"contamination index: {len(grams)} {args.n}-grams, {len(shorts)} short texts",
          file=sys.stderr)

    kept = dropped = 0
    log = open(args.log, "w", encoding="utf-8") if args.log else None
    with open_maybe_gz(args.input) as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            words = normalize_words(obj[args.field])
            doc_grams = ngrams(words, args.n)
            joined = " ".join(words)
            hit = bool(doc_grams & grams) or any(s in joined for s in shorts)
            if hit:
                dropped += 1
                if log:
                    log.write(line if line.endswith("\n") else line + "\n")
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    if log:
        log.close()
    print(f"kept {kept}, dropped {dropped} contaminated", file=sys.stderr)


if __name__ == "__main__":
    main()
