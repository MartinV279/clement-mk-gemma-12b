#!/usr/bin/env python3
"""Detect grounding-specific artifacts (failure taxonomy from the user's
2026-07-26 review of the grounded batch):

  1. SOURCE LEAK — assistant references the passage the user never saw
     („изворите", „тој опис", „не пишува во...")
  2. ADOPTED DEIXIS — assistant narrates first-person experience lifted from
     a narrative source („слушнав еднаш...", „заедно читаме...", full
     fabricated autobiographies)
  3. VERBATIM ECHO — the same non-trivial sentence duplicated across turns

Usage:
  uv run python data/filters/grounding_artifacts.py --input pool.jsonl --output clean.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter

SOURCE_LEAK = re.compile(
    r"извор(от|ите)?\b|тој опис|описот|не пишува во|наведено (е )?во|"
    r"според текстот|во текстот (што|кој)|од изворите|ги фатив од", re.IGNORECASE)

# first-person experiential markers in ASSISTANT turns (not banned in user turns)
FIRST_PERSON = re.compile(
    r"\b(слушнав|видов|бев|отидов|ме повлекоа|доживеав|се сеќавам|"
    r"јас лично (бев|видов|слушнав)|заедно (читаме|стоиме|одиме)|"
    r"мојата (баба|мајка|сестра) ми (рече|кажа))\b", re.IGNORECASE)


def sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 40]


def flags_for(conv: list) -> list:
    flags = []
    asst = [t["content"] for t in conv if t["role"] == "assistant"]
    joined = " ".join(asst)
    if SOURCE_LEAK.search(joined):
        flags.append("source_leak")
    if FIRST_PERSON.search(joined):
        flags.append("adopted_deixis")
    sent_counts = Counter(s for a in asst for s in sentences(a))
    if any(c > 1 for c in sent_counts.values()):
        flags.append("verbatim_echo")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    kept = dropped = 0
    reasons = Counter()
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            flags = flags_for(r["conversations"])
            if flags:
                dropped += 1
                reasons.update(flags)
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    print(f"kept {kept}, dropped {dropped} | {dict(reasons)}", file=sys.stderr)


if __name__ == "__main__":
    main()
