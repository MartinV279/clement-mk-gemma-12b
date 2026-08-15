#!/usr/bin/env python3
"""Measure Gemma tokenizer fertility on Macedonian text (Phase 0, playbook 0.4).

tokens/word on ~1MB of MK text. Expected ~2.3-2.7; if wildly worse, STOP and
flag to the user before any training decision. Downloads only tokenizer files
(a few MB), not weights. Can run locally once the HF license is accepted.

Usage:
  uv run python eval/tokenizer_fertility.py --text-file <mk_sample.txt> \
      [--tokenizer google/gemma-4-12B]
"""

import argparse
import os
from pathlib import Path

EXPECTED_RANGE = (2.3, 2.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True, help="~1MB of clean Macedonian text")
    ap.add_argument("--tokenizer", default="google/gemma-4-12B")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    load_dotenv()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=os.environ.get("HF_TOKEN"))

    text = Path(args.text_file).read_text(encoding="utf-8")
    words = text.split()
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    fertility = len(tokens) / len(words)

    print(f"tokenizer : {args.tokenizer}")
    print(f"text      : {len(text)/1e6:.2f} MB, {len(words):,} words, {len(tokens):,} tokens")
    print(f"fertility : {fertility:.3f} tokens/word (expected {EXPECTED_RANGE[0]}-{EXPECTED_RANGE[1]})")
    if fertility > EXPECTED_RANGE[1] * 1.3:
        print("WARNING: fertility is wildly worse than expected — STOP and flag to the user "
              "(playbook Phase 0 step 4).")


if __name__ == "__main__":
    main()
