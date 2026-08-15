#!/usr/bin/env python3
"""Serbian/Bulgarian lexical-bleed screen (CLAUDE.md data pipeline).

Two-layer detection:
  1. SCRIPT layer (automatic, safe): any Cyrillic letter outside the Macedonian
     alphabet is a hard signal — ђ ћ (Serbian), ъ щ ю я й ь э ы (Bulgarian/Russian).
  2. LEXICAL layer (user-gated): tell-tale words spelled entirely in MK letters
     (такође, неколико...). Seeded from the user-authored constitution §3;
     `draft` mode proposes NEW candidates with corpus frequencies and contexts —
     the USER approves entries before they enter blocklist_approved.txt.
     Claude Code never finalizes lexical entries alone.

Subcommands:
  draft  — scan a jsonl corpus, emit blocklist_draft.tsv for user review
  filter — apply script layer + blocklist_approved.txt to a jsonl file
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
APPROVED = HERE / "blocklist_approved.txt"
DRAFT = HERE / "blocklist_draft.tsv"

MK_LETTERS = set("абвгдѓежзѕијклљмнњопрстќуфхцчџш")
# ѐ/ѝ appear in normative MK typography (сѐ, нѐ, ѝ)
MK_EXTRA = set("ѐѝ́")
# Cyrillic letters absent from Macedonian: Serbian (ђћ), Bulgarian modern
# (ъщюяйь) + archaic (ѣѫѭѩ), Russian (эыё), Ukrainian (іїєґ), Belarusian (ў).
# >=3 occurrences => text is not Macedonian (user rule, 2026-07-27).
FOREIGN_CYRILLIC = re.compile(r"[ђћъщюяйьэыёѣѫѭѩіїєґў]", re.IGNORECASE)
# Latin homoglyphs inside Cyrillic words („Сe на сe" with Latin e — found by
# user review 2026-07-26) and the лј digraph that should be љ
HOMOGLYPH_WORD = re.compile(r"\b(?=\w*[а-шѓќѕџљњј])(?=\w*[a-z])\w+\b", re.IGNORECASE)
LJ_DIGRAPH = re.compile(r"\b[Лл]ј")  # word-initial only: Билјана/илјада are correct MK

# Seed lexical list from docs/constitution.md §3 (user-authored => pre-approved)
SEED = [
    "такође", "неколико", "врло", "хвала", "тачно", "требало би", "трябва",
    "искам", "обичам", "нещо", "нищо",
]

WORD_RE = re.compile(r"[а-шђћъщюяйьэыёѓќѕџљњјѐѝ]+", re.IGNORECASE)


def open_maybe_gz(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def load_approved() -> list:
    terms = list(SEED)
    if APPROVED.exists():
        terms += [t.strip().lower() for t in APPROVED.read_text(encoding="utf-8").splitlines()
                  if t.strip() and not t.startswith("#")]
    return sorted(set(terms))


def find_hits(text: str, lexical_terms) -> tuple:
    low = text.lower()
    script_hits = FOREIGN_CYRILLIC.findall(low)
    script_hits += HOMOGLYPH_WORD.findall(low)
    script_hits += LJ_DIGRAPH.findall(text)
    lex_hits = [t for t in lexical_terms if t in low]
    return script_hits, lex_hits


def cmd_draft(args) -> None:
    lexical = load_approved()
    foreign_words = Counter()
    seed_hits = Counter()
    contexts = {}
    n = 0
    with open_maybe_gz(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            text = json.loads(line)[args.field]
            low = text.lower()
            for w in WORD_RE.findall(low):
                if FOREIGN_CYRILLIC.search(w):
                    foreign_words[w] += 1
                    if w not in contexts:
                        i = low.find(w)
                        contexts[w] = text[max(0, i - 40):i + 40].replace("\n", " ")
            for t in lexical:
                if t in low:
                    seed_hits[t] += 1

    with DRAFT.open("w", encoding="utf-8") as f:
        f.write("# Bleed blocklist DRAFT — candidates for USER approval.\n")
        f.write("# Move approved WORDS (one per line) into blocklist_approved.txt.\n")
        f.write("# type\tterm\tcount\texample_context\n")
        for w, c in foreign_words.most_common(args.top):
            f.write(f"script\t{w}\t{c}\t{contexts.get(w,'')}\n")
        for t, c in seed_hits.most_common():
            f.write(f"seed\t{t}\t{c}\t(constitution seed — already approved)\n")
    print(f"scanned {n} docs; {len(foreign_words)} distinct foreign-script words, "
          f"draft written to {DRAFT} — awaiting user review")


def cmd_filter(args) -> None:
    lexical = load_approved()
    kept = dropped = 0
    with open_maybe_gz(args.input) as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            script_hits, lex_hits = find_hits(obj[args.field], lexical)
            # tolerate a stray foreign char (quotes/names); density is the signal
            if len(script_hits) > args.max_script_chars or lex_hits:
                dropped += 1
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    print(f"kept {kept}, dropped {dropped}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft")
    d.add_argument("--input", required=True)
    d.add_argument("--field", default="text")
    d.add_argument("--limit", type=int, default=200000, help="docs to scan (0 = all)")
    d.add_argument("--top", type=int, default=300)

    f = sub.add_parser("filter")
    f.add_argument("--input", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--field", default="text")
    f.add_argument("--max-script-chars", type=int, default=2)

    args = ap.parse_args()
    {"draft": cmd_draft, "filter": cmd_filter}[args.cmd](args)


if __name__ == "__main__":
    main()
