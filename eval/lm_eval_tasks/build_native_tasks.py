#!/usr/bin/env python3
"""Normalize the three NATIVE MK eval sets into local jsonl for lm-eval.
EVAL-ONLY — never train on these (dataset_census.md rule).

  exams_mk    mhardalov/exams crosslingual_mk (train+validation = 2,075)
  copa_mk     classla/COPA-MK (all splits = 1,000)
  include_mk  CohereLabs/include-base-44 "North Macedonian" (571)
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"
LETTER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def dump(name, rows):
    p = OUT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} -> {p}")


rows = []
for split in ("train", "validation"):
    for d in load_dataset("mhardalov/exams", "crosslingual_mk", split=split):
        q = d["question"]
        if d["answerKey"] not in LETTER or len(q["choices"]["text"]) < 2:
            continue
        gold = LETTER[d["answerKey"]]
        if gold >= len(q["choices"]["text"]):
            continue
        rows.append({"question": q["stem"].strip(),
                     "choices": [c.strip() for c in q["choices"]["text"]],
                     "gold": gold, "subject": d["info"]["subject"]})
dump("exams_test_mk", rows)

rows = []
for split in ("train", "dev", "test"):
    for d in load_dataset("classla/COPA-MK", split=split):
        # standard COPA framing: connector depends on cause/effect
        conn = "затоа што" if d["question"] == "cause" else "па затоа"
        prem = d["premise"].rstrip(".")
        rows.append({"question": f"{prem}, {conn}",
                     "choices": [d["choice1"][0].lower() + d["choice1"][1:],
                                 d["choice2"][0].lower() + d["choice2"][1:]],
                     "gold": d["label"]})
dump("copa_test_mk", rows)

rows = []
for split in ("test", "validation"):
    for d in load_dataset("CohereLabs/include-base-44", "North Macedonian",
                          split=split):
        opts = [d[f"option_{x}"] for x in "abcd" if d.get(f"option_{x}")]
        gold = d["answer"] if isinstance(d["answer"], int) else LETTER.get(str(d["answer"]).strip().upper(), None)
        if gold is None or gold >= len(opts):
            continue
        rows.append({"question": d["question"].strip(),
                     "choices": [o.strip() for o in opts], "gold": gold,
                     "subject": d.get("subject", "")})
dump("include_test_mk", rows)
