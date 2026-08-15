#!/usr/bin/env python3
"""Answer-key verification panel for the science-MCQ pack.

Unlike the fact panel (which checks claims against a passage), these items
have no source passage — the question is the model's own invention. So the panel
does the only check that matters: each voter answers the MCQ independently,
cold, and we keep the item only if the voters agree with the stated answer_idx.

That makes a wrong key detectable rather than merely unlikely: if two strong
models pick B and the generator said C, the item is dropped. Teaching a wrong
science fact would feed exactly the confident-fabrication failure we are trying
to kill, and arc_easy_mk would punish it twice.

Voters (2-of-3 must match the key):
  deepseek   deepseek-chat
  groq       llama-3.3-70b-versatile  (free tier)
  mistral    mistral-large-latest     (free tier)

  .venv/bin/python data/synth/verify_scimcq.py --in data/synth/scimcq_bulk.jsonl
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """Ова е прашање по природни науки со четири понудени одговори.
Одговори САМО со строг JSON: {{"answer_idx": 0-3, "confidence": "high"/"low"}}
Не објаснувај. Избери го најточниот одговор.

ПРАШАЊЕ: {q}

ОПЦИИ:
{opts}"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _post(url, headers, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.5.0", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_voter(name, url, key, model):
    def vote(item):
        opts = "\n".join(f"{i}. {c}" for i, c in enumerate(item["choices"]))
        payload = {"model": model, "temperature": 0.0, "max_tokens": 200,
                   "messages": [{"role": "user",
                                 "content": PROMPT.format(q=item["question"], opts=opts)}]}
        for attempt in range(3):
            try:
                d = _post(url, {"Authorization": f"Bearer {key}"}, payload)
                raw = d["choices"][0]["message"]["content"]
                m = JSON_RE.search(raw.replace("```json", "").replace("```", ""))
                idx = json.loads(m.group(0)).get("answer_idx")
                return idx if isinstance(idx, int) and 0 <= idx <= 3 else None
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    import time
                    time.sleep(8 * (attempt + 1))
                    continue
                return None
            except Exception:
                return None
        return None
    vote.__name__ = name
    return vote


def make_gemini_voter(key, model="gemini-flash-latest"):
    """Gemini speaks a different API dialect, so it needs its own caller."""
    def vote(item):
        opts = "\n".join(f"{i}. {c}" for i, c in enumerate(item["choices"]))
        body = {"system_instruction": {"parts": [{"text": "Answer with strict JSON only."}]},
                "contents": [{"role": "user",
                              "parts": [{"text": PROMPT.format(q=item["question"], opts=opts)}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json",
                                                      "User-Agent": "skazna/6.0"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.load(r)
                parts = d["candidates"][0]["content"]["parts"]
                raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                m = JSON_RE.search(raw.replace("```json", "").replace("```", ""))
                idx = json.loads(m.group(0)).get("answer_idx")
                return idx if isinstance(idx, int) and 0 <= idx <= 3 else None
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    import time
                    time.sleep(6 * (attempt + 1))
                    continue
                return None
            except Exception:
                return None
        return None
    vote.__name__ = "gemini"
    return vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/processed/scimcq_verified.jsonl")
    ap.add_argument("--rejected", default="data/synth/scimcq_rejected.jsonl")
    ap.add_argument("--unresolved", default="data/synth/scimcq_unresolved.jsonl",
                    help="items the panel could not judge because voters were "
                         "rate-limited. NOT rejections — they are retried on the "
                         "next run and never silently discarded.")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-voters", default="",
                    help="comma-separated voters to disable, e.g. mistral. A "
                         "hard-rate-limited voter costs ~24s of retry backoff "
                         "per item and contributes nothing but null votes.")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    def firstkey(*names):
        for n in names:
            if os.environ.get(n):
                return os.environ[n]
        return ""

    voters = []
    if firstkey("TEACHER_API_KEY"):
        voters.append(make_voter("deepseek", "https://api.deepseek.com/chat/completions",
                                 firstkey("TEACHER_API_KEY"), "deepseek-chat"))
    if firstkey("GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY"):
        voters.append(make_voter("groq", "https://api.groq.com/openai/v1/chat/completions",
                                 firstkey("GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY"),
                                 "llama-3.3-70b-versatile"))
    if firstkey("GOOGLE_API_KEY_PAID"):
        voters.append(make_gemini_voter(firstkey("GOOGLE_API_KEY_PAID")))
    if firstkey("MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2", "MISTRAL_API_KEY"):
        voters.append(make_voter("mistral", "https://api.mistral.ai/v1/chat/completions",
                                 firstkey("MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2",
                                          "MISTRAL_API_KEY"),
                                 "mistral-large-latest"))
    skip = {x.strip().lower() for x in args.skip_voters.split(",") if x.strip()}
    if skip:
        voters = [v for v in voters if v.__name__ not in skip]
        print(f"disabled voters: {sorted(skip)}")
    if len(voters) < args.min_votes:
        sys.exit(f"only {len(voters)} voters available but --min-votes={args.min_votes}; "
                 "that would reject every item for lack of votes")
    if not voters:
        sys.exit("no voter API keys available")
    print(f"panel: {[v.__name__ for v in voters]} | need {args.min_votes} agreeing votes")

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} items, {len(done)} already verified, {len(todo)} to go")

    fout = open(args.out, "a", encoding="utf-8")
    frej = open(args.rejected, "a", encoding="utf-8")
    funr = open(args.unresolved, "a", encoding="utf-8")
    kept = dropped = unresolved = 0

    def judge(item):
        with ThreadPoolExecutor(max_workers=len(voters)) as ex:
            return list(ex.map(lambda v: v(item), voters))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item, votes in zip(todo, pool.map(judge, todo)):
            agree = sum(1 for v in votes if v is not None and v == item["answer_idx"])
            valid = sum(1 for v in votes if v is not None)
            rec = {**item, "panel_votes": votes, "agree": agree}
            # THREE outcomes, not two. A rate-limited voter is a missing verdict,
            # never a failing one — conflating them filed 448 good items as
            # "rejected" when Groq went silent. Unresolved items are not written
            # to --out, so the next run retries them.
            if valid < args.min_votes:
                funr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                unresolved += 1
            elif agree >= args.min_votes:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                frej.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dropped += 1
            n = kept + dropped + unresolved
            if n % 100 == 0:
                fout.flush(); frej.flush(); funr.flush()
                print(f"  [{n}/{len(todo)}] kept {kept} dropped {dropped} "
                      f"unresolved {unresolved}", flush=True)
                if unresolved > 50 and unresolved > 3 * (kept + dropped):
                    print("  !! voters are mostly dead — aborting rather than "
                          "producing a meaningless verdict. Lower --workers.",
                          flush=True)
                    raise SystemExit(3)

    fout.close(); frej.close(); funr.close()
    total = kept + dropped
    print(f"DONE kept {kept}/{total} ({100 * kept / max(total, 1):.1f}%) -> {args.out}")
    print(f"     dropped {dropped} -> {args.rejected}")
    if unresolved:
        print(f"     UNRESOLVED {unresolved} (voters unavailable) -> {args.unresolved}"
              f"  — rerun to retry these; they are NOT rejections")


if __name__ == "__main__":
    main()
