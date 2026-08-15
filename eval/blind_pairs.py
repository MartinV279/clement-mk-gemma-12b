#!/usr/bin/env python3
"""Self-blinding for Fable-delegated arena judging (user request 2026-08-08).

Pairs the two generation files, shuffles left/right per matchup (seeded),
writes:
  eval/blind/pairs.jsonl   {qid, category, prompt, check, ans_1, ans_2}
  eval/blind/key.json      {qid: model_of_ans_1}   <- NOT read until voting done
The judge reads only pairs.jsonl; reveal.py joins votes with the key.
"""
import json
import random
from pathlib import Path

FILES = ["eval/generations/skazna-chat.jsonl",
         "eval/generations/LVSTCK__domestic-yak-8B-instruct.jsonl"]
rng = random.Random(20260808)

prompts = {json.loads(l)["id"]: json.loads(l)
           for l in open("eval/vibes_prompts.jsonl", encoding="utf-8")}
gens = {}
for f in FILES:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        gens.setdefault(r["id"], {})[r["model"]] = r["answer"]

Path("eval/blind").mkdir(exist_ok=True)
key = {}
with open("eval/blind/pairs.jsonl", "w", encoding="utf-8") as out:
    for qid in sorted(gens):
        models = sorted(gens[qid])
        assert len(models) == 2, f"{qid}: {models}"
        rng.shuffle(models)
        key[qid] = models[0]  # model behind ans_1
        p = prompts[qid]
        out.write(json.dumps({
            "qid": qid, "category": p["category"], "prompt": p["prompt"],
            "check": p.get("check", ""),
            "ans_1": gens[qid][models[0]], "ans_2": gens[qid][models[1]],
        }, ensure_ascii=False) + "\n")
json.dump(key, open("eval/blind/key.json", "w"))
print(f"blinded {len(key)} pairs -> eval/blind/pairs.jsonl (key sealed)")
