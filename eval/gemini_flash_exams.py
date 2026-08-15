#!/usr/bin/env python3
"""Reference point: gemini-flash on exams_mk (native MK, 2,075 items).

Protocol difference vs the battery, stated plainly: our models are scored by
LOGLIKELIHOOD RANKING over the four options (lm-eval); an API model must answer
GENERATIVELY (output the letter). Generative MC usually scores a bit differently
from ranking, so treat the result as a reference, not a same-protocol row.

Usage capped by tokens so a retry storm cannot run the key dry:
  .venv/bin/python eval/gemini_flash_exams.py [--limit N]
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOKEN_CAP = 3_000_000
_usage = {"in": 0, "out": 0, "think": 0}
LETTERS = "АБВГД"


def tokens_used():
    return _usage["in"] + _usage["out"] + _usage["think"]


def ask(key, item):
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(item["choices"]))
    q = item.get("question") or item.get("query")
    prompt = (f"{q}\n{opts}\n\n"
              f"Одговори САМО со буквата на точниот одговор ({'/'.join(LETTERS[:len(item['choices'])])}).")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-3-flash-preview:generateContent?key={key}")
    for attempt in range(4):
        if tokens_used() >= TOKEN_CAP:
            return None
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "skazna-eval"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            u = d.get("usageMetadata", {})
            _usage["in"] += u.get("promptTokenCount", 0)
            _usage["out"] += u.get("candidatesTokenCount", 0)
            _usage["think"] += u.get("thoughtsTokenCount", 0)
            try:
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]
                               if not p.get("thought")).strip()
            except (KeyError, IndexError):
                return None
            m = re.search(rf"[{LETTERS}]", text.upper())
            return LETTERS.index(m.group(0)) if m else None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(10 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--task", default="exams_test_mk.jsonl")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ["GOOGLE_API_KEY_PAID"]

    rows = [json.loads(l) for l in
            open(f"eval/lm_eval_tasks/data/{args.task}", encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} items from {args.task}; token cap {TOKEN_CAP:,}")

    correct = answered = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (item, pred) in enumerate(zip(rows, ex.map(lambda r: ask(key, r), rows)), 1):
            if pred is not None:
                answered += 1
                correct += (pred == item["gold"])
            if i % 200 == 0:
                print(f"  [{i}/{len(rows)}] acc so far {correct}/{answered} = "
                      f"{correct/max(answered,1):.4f}  {tokens_used():,}tok  "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

    acc = correct / max(answered, 1)
    out = {"model": "gemini-3-flash-preview", "task": args.task, "protocol": "generative-letter",
           "n": len(rows), "answered": answered, "correct": correct,
           "acc": acc, "tokens": tokens_used()}
    Path("eval/results/gemini_flash_exams_mk.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nGEMINI-FLASH exams_mk: {correct}/{answered} = {acc:.4f}  "
          f"(unanswered: {len(rows)-answered})  {tokens_used():,} tokens")


if __name__ == "__main__":
    main()
