#!/usr/bin/env python3
"""Exact character-filter census over the FULL datasets (no sampling).

Reports, per dataset: documents rejected by each sub-rule (foreign letters /
Latin homoglyphs / лј digraph / blocklist words), the per-letter document and
occurrence counts, and rule overlap. Output: data/CHAR_CENSUS.md

Usage: uv run python data/char_census.py
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "filters"))
from bleed_blocklist import (FOREIGN_CYRILLIC, HOMOGLYPH_WORD,  # noqa: E402
                             LJ_DIGRAPH, load_approved)

GROUPS = [("Serbian", "ђћ"), ("Bulgarian modern", "ъщюяйь"),
          ("Bulgarian archaic", "ѣѫѭѩ"), ("Russian", "эыё"),
          ("Ukrainian", "іїєґ"), ("Belarusian", "ў")]
LEX = load_approved()


def census(name, iterator):
    st = Counter()
    letter_docs, letter_hits = Counter(), Counter()
    homo_words = Counter()
    for text in iterator:
        st["docs"] += 1
        low = text.lower()
        f = FOREIGN_CYRILLIC.findall(low)
        h = HOMOGLYPH_WORD.findall(low)
        lj = LJ_DIGRAPH.findall(text)
        lex = [t for t in LEX if t in low]
        for c in set(f):
            letter_docs[c] += 1
        for c in f:
            letter_hits[c] += 1
        if f:
            st["docs_with_any_foreign_letter"] += 1
        # sub-rule attribution (independent, so overlaps are visible)
        if len(f) > 2:
            st["reject_letters"] += 1
        if len(h) > 2:
            st["reject_homoglyph"] += 1
            homo_words.update(h)
        if len(lj) > 0:
            st["hit_lj"] += 1
        if lex:
            st["reject_blocklist"] += 1
        if len(f) + len(h) + len(lj) > 2 or lex:
            st["rejected_total"] += 1
        if st["docs"] % 500000 == 0:
            print(f"  [{name}] {st['docs']:,}...", flush=True)
    return st, letter_docs, letter_hits, homo_words


def gz_text(p, field="text"):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)[field]


def parallel_mk():
    with gzip.open("data/downloads/parallel/MaCoCu-mk-en.sent.txt.gz", "rt",
                   encoding="utf-8") as f:
        hdr = f.readline().split("\t")
        i = hdr.index("src_text")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > i and p[i].strip():
                yield p[i]


DATASETS = [
    ("MK web corpus (raw)", lambda: gz_text(
        "data/downloads/mk_corpus/macedonian_corpus_cleaned_deduplicated.jsonl.gz")),
    ("parallel MK side", parallel_mk),
]


def main():
    out = ["# Exact character-filter census (full datasets, no sampling)",
           "", "Rule: >=3 non-Macedonian Cyrillic letters, or >2 combined "
           "script anomalies, or any blocklist word => reject.", ""]
    for name, it in DATASETS:
        print(f"=== {name} ===", flush=True)
        st, ldocs, lhits, homo = census(name, it())
        n = st["docs"]
        out += [f"## {name}", "",
                f"{n:,} documents scanned · **{st['rejected_total']:,} rejected "
                f"({st['rejected_total']/n:.2%})**", "",
                "| sub-rule | documents | share |", "|---|---|---|",
                f"| foreign letters (>=3) | {st['reject_letters']:,} | {st['reject_letters']/n:.3%} |",
                f"| Latin homoglyphs (>2) | {st['reject_homoglyph']:,} | {st['reject_homoglyph']/n:.3%} |",
                f"| blocklist word | {st['reject_blocklist']:,} | {st['reject_blocklist']/n:.3%} |",
                f"| лј digraph present | {st['hit_lj']:,} | {st['hit_lj']/n:.3%} |",
                "",
                f"Documents containing at least one foreign letter (any count): "
                f"**{st['docs_with_any_foreign_letter']:,}** "
                f"({st['docs_with_any_foreign_letter']/n:.2%})", "",
                "| alphabet | docs | occurrences | per-letter (docs) |",
                "|---|---|---|---|"]
        for gname, letters in GROUPS:
            d = sum(ldocs[c] for c in letters)
            h = sum(lhits[c] for c in letters)
            detail = " ".join(f"{c}:{ldocs[c]:,}" for c in letters if ldocs[c])
            out.append(f"| {gname} | {d:,} | {h:,} | {detail or '—'} |")
        out.append("")
        if homo:
            top = ", ".join(f"`{w}`×{c:,}" for w, c in homo.most_common(12))
            out += [f"Top homoglyph words: {top}", ""]
    Path("data/CHAR_CENSUS.md").write_text("\n".join(out), encoding="utf-8")
    print("written: data/CHAR_CENSUS.md")


if __name__ == "__main__":
    main()
