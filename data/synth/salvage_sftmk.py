#!/usr/bin/env python3
"""Strict salvage from LVSTCK/sft-mk for the medium-quality tier.
Hard rejects: AI boilerplate, deixis to absent content, bleeds, mixed-script,
short/bloated rows. Cap 2,000 (medium tier must not dilute our gold)."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved
from data.synth.generate_sft import MIXED_WORD, CJK

BAD = re.compile(r"вештачка интелигенциј|јазичен модел|како модел|немам способност|"
                 r"не можам да пристапам|as an AI|language model|OpenAI|ChatGPT", re.I)
DEIXIS = re.compile(r"наведен\w+|дадениот текст|горенаведен\w+|следнава статија|"
                    r"текстот што|статијата „|врз основа на текстот", re.I)
CYR = re.compile(r"[а-шѓќѕџљњј]")

from datasets import load_dataset
lex = load_approved()
d = load_dataset("LVSTCK/sft-mk", split="train")
kept = []
for r in d:
    convs = r["conversations"]
    if not (2 <= len(convs) <= 8) or convs[0]["role"] != "user":
        continue
    text = " ".join(m["content"] for m in convs)
    a_lens = [len(m["content"]) for m in convs if m["role"] == "assistant"]
    if not a_lens or not (250 <= sum(a_lens) <= 2500):
        continue
    if BAD.search(text) or DEIXIS.search(text):
        continue
    if len(CYR.findall(text)) / max(len(text), 1) < 0.55:
        continue
    h, l = find_hits(text, lex)
    if h or l or CJK.search(text) or MIXED_WORD.search(text):
        continue
    kept.append(convs)
    if len(kept) >= 2000:
        break
out = Path("data/synth/v5_salvage_sftmk.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, c in enumerate(kept):
        f.write(json.dumps({"id": f"v5slv-{i:05d}", "kind": "salvage",
                            "conversations": c}, ensure_ascii=False) + "\n")
print(f"salvaged {len(kept)} -> {out}")
