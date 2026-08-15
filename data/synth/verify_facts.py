#!/usr/bin/env python3
"""Entailment panel for generated fact rows — a small multi-model jury.

Each row's `claims` are checked against its stored `passage`. Voters:
  deepseek   deepseek-chat (reads MK, outputs JSON only — its weak MK *prose*
             never appears in training data)
  groq       llama-4-maverick via Groq free tier
  mistral    mistral-large via free tier
A row KEEPS only if >=2 voters say every claim is supported. Output:
  data/processed/facts_verified.jsonl   (kept rows, panel votes attached)
  data/synth/facts_rejected.jsonl       (audit)

  .venv/bin/python data/synth/verify_facts.py --in data/synth/facts_bulk.jsonl
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are a strict fact-verification judge. Below is a PASSAGE (Macedonian) and a list of CLAIMS.
For each claim decide if it is SUPPORTED by the passage (entailed — the passage states it or it follows directly).
Numbers, dates and names must match exactly. Answer with strict JSON only:
{{"verdicts": [true/false, ...], "all_supported": true/false}}
The verdicts list must have exactly {n} entries, in claim order.

PASSAGE:
{passage}

CLAIMS:
{claims}"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _post(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.5.0", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_voter(name, url, key, model):
    def vote(passage, claims):
        text = PROMPT.format(n=len(claims), passage=passage[:5000],
                             claims="\n".join(f"{i+1}. {c}" for i, c in enumerate(claims)))
        payload = {"model": model, "temperature": 0.0, "max_tokens": 4000,
                   "messages": [{"role": "user", "content": text}]}
        d = _post(url, {"Authorization": f"Bearer {key}"}, payload)
        raw = d["choices"][0]["message"]["content"]
        m = JSON_RE.search(raw.replace("```json", "").replace("```", ""))
        v = json.loads(m.group(0))
        verd = v.get("verdicts", [])
        ok = bool(v.get("all_supported")) and len(verd) == len(claims) and all(verd)
        return ok
    vote.__name__ = name
    return vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--no-reasoner", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    voters = []
    if os.environ.get("TEACHER_API_KEY"):
        voters.append(make_voter("deepseek", "https://api.deepseek.com/chat/completions",
                                 os.environ["TEACHER_API_KEY"], "deepseek-chat"))
        if not args.no_reasoner:
            voters.append(make_voter("ds_reasoner",
                                     "https://api.deepseek.com/chat/completions",
                                     os.environ["TEACHER_API_KEY"], "deepseek-reasoner"))
    for suf in ("_1", "_2"):
        k = os.environ.get(f"GROQ_API_KEY{suf}")
        if k:
            voters.append(make_voter(f"groq{suf}",
                                     "https://api.groq.com/openai/v1/chat/completions",
                                     k, "llama-3.3-70b-versatile"))
            break
    for suf in ("_1", "_2"):
        k = os.environ.get(f"MISTRAL_API_KEY{suf}")
        if k:
            voters.append(make_voter(f"mistral{suf}",
                                     "https://api.mistral.ai/v1/chat/completions",
                                     k, "mistral-large-latest"))
            break
    print("panel:", [v.__name__ for v in voters])
    assert len(voters) >= 2, "need at least 2 voters"

    out_keep = Path("data/processed/facts_verified.jsonl")
    out_rej = Path("data/synth/facts_rejected.jsonl")
    done_ids = set()
    for p in (out_keep, out_rej):
        if p.exists():
            done_ids |= {json.loads(l)["id"] for l in p.open(encoding="utf-8")}
    fk = out_keep.open("a", encoding="utf-8")
    fr = out_rej.open("a", encoding="utf-8")
    kept = rej = 0
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")
            if json.loads(l)["id"] not in done_ids]
    prog = [0]

    def judge(r):
        votes = {}
        for v in voters:
            for attempt in range(4):
                try:
                    votes[v.__name__] = v(r["passage"], r["claims"])
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        time.sleep(10 * (attempt + 1))
                        continue
                    votes[v.__name__] = None
                    break
                except Exception:
                    votes[v.__name__] = None
                    break
        yes = sum(1 for x in votes.values() if x is True)
        valid = sum(1 for x in votes.values() if x is not None)
        r["panel"] = votes
        nonlocal_kept = yes >= args.min_votes or (valid == 1 and yes == 1)
        with lock:
            if nonlocal_kept:
                fk.write(json.dumps(r, ensure_ascii=False) + "\n"); fk.flush()
            else:
                fr.write(json.dumps(r, ensure_ascii=False) + "\n"); fr.flush()
            prog[0] += 1
            if prog[0] % 50 == 0:
                print(f"[{prog[0]}/{len(rows)}]", flush=True)
        return nonlocal_kept

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(judge, rows))
    kept = sum(results); rej = len(results) - kept
    print(f"DONE kept={kept} rej={rej}")


if __name__ == "__main__":
    main()
