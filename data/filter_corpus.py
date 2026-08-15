#!/usr/bin/env python3
"""Single-pass filter of the MK CPT corpus with the USER-APPROVED settings
(data/filters/approved.yaml): langID -> bleed screen -> KenLM perplexity ->
decontamination vs the frozen vibes set. Order per CLAUDE.md data pipeline.

Writes filtered gz + per-filter drop stats. Run before mixture build.

Usage:
  uv run python data/filter_corpus.py \
      --input data/downloads/mk_corpus/macedonian_corpus_cleaned_deduplicated.jsonl.gz \
      --output data/processed/mk_corpus_filtered.jsonl.gz
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "filters"))
from bleed_blocklist import find_hits, load_approved  # noqa: E402
from decontaminate import contamination_sets, ngrams, normalize_words  # noqa: E402
from kenlm_filter import doc_perplexity  # noqa: E402
from langid_filter import predict_lang  # noqa: E402

KENLM_THRESHOLD = 3595       # user-approved 2026-07-23
LANGID_MIN_CONF = 0.65
MAX_SCRIPT_CHARS = 2
NGRAM = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lid-model", default="data/models/lid.176.bin")
    ap.add_argument("--kenlm-model", default="data/models/kenlm_mk_5gram.bin")
    ap.add_argument("--vibes", default="eval/vibes_prompts.jsonl")
    args = ap.parse_args()

    import fasttext
    import kenlm
    lid = fasttext.load_model(args.lid_model)
    lm = kenlm.Model(args.kenlm_model)
    lexical = load_approved()
    grams, shorts = contamination_sets([args.vibes], NGRAM)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with gzip.open(args.input, "rt", encoding="utf-8") as fin, \
         gzip.open(args.output, "wt", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            stats["total"] += 1
            obj = json.loads(line)
            text = obj["text"]

            labels, confs = predict_lang(lid, text[:2000])
            if not labels or labels[0] != "__label__mk" or confs[0] <= LANGID_MIN_CONF:
                stats["drop_langid"] += 1
                continue

            script_hits, lex_hits = find_hits(text, lexical)
            if len(script_hits) > MAX_SCRIPT_CHARS or lex_hits:
                stats["drop_bleed"] += 1
                continue

            if doc_perplexity(lm, text) > KENLM_THRESHOLD:
                stats["drop_kenlm"] += 1
                continue

            words = normalize_words(text)
            joined = " ".join(words)
            if (ngrams(words, NGRAM) & grams) or any(s in joined for s in shorts):
                stats["drop_decontaminate"] += 1
                continue

            fout.write(line if line.endswith("\n") else line + "\n")
            stats["kept"] += 1

            if stats["total"] % 200000 == 0:
                print(dict(stats), flush=True)

    print("FINAL:", dict(stats))
    kept, total = stats["kept"], stats["total"]
    print(f"kept {kept}/{total} ({kept/total:.1%})")


if __name__ == "__main__":
    main()
