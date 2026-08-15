#!/usr/bin/env python3
"""Unseal the sealed key and tally the blind arena.

Run from the repo root:
    python eval/blind/reveal.py

Reproduces the published head-to-head from the raw artifacts in this directory,
so the tally can be recomputed rather than taken on trust.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
MODELS = ("Clement-12B", "LVSTCK/domestic-yak-8B-instruct")

key = {int(k): v for k, v in json.load((HERE / "key_clement-vs-yak.json").open()).items()
       if not k.startswith("_")}
cat = {r["qid"]: r["category"]
       for r in map(json.loads, (HERE / "pairs_clement-vs-yak.jsonl").open(encoding="utf-8"))}
votes = [json.loads(l) for l in (HERE / "votes_clement-vs-yak.jsonl").open(encoding="utf-8")]
assert len(votes) == 50 and len({v["qid"] for v in votes}) == 50

tally = defaultdict(lambda: defaultdict(int))
overall = defaultdict(int)
rows = []
for v in votes:
    qid = v["qid"]
    if v["vote"] == 0:
        winner = "tie"
    else:
        # vote 1 = ANS1 won. The key says which model wrote ANS1 for this item.
        winner = key[qid] if v["vote"] == 1 else [m for m in MODELS if m != key[qid]][0]
    overall[winner] += 1
    tally[cat[qid]][winner] += 1
    rows.append((qid, cat[qid], winner, v["conf"]))

print("=== OVERALL ===")
for m, n in sorted(overall.items(), key=lambda x: -x[1]):
    print(f"  {m:35} {n}")
print("\n=== BY CATEGORY ===")
for c in sorted(tally):
    parts = ", ".join(f"{m.split('/')[-1].replace('domestic-', '')}: {n}"
                      for m, n in sorted(tally[c].items(), key=lambda x: -x[1]))
    print(f"  {c:14} {parts}")
print("\n=== PER MATCHUP ===")
for qid, c, w, conf in rows:
    print(f"  q{qid:02d} {c:14} -> {w:35} ({conf})")
