#!/usr/bin/env python3
"""Exam-MCQ and national-canon data, plus a naturalness edit pass.

Kinds:
  mcq    exam-style MK multiple-choice items (option-ranking calibration).
         HARD-DECONTAMINATED against exams/copa/include/arc test sets (6-gram).
  canon  named national-canon facts (Мисирков, Делчев, сказна...) as 10-way
         paraphrase packs, grounded in wiki passages found per topic.
  edit   naturalness edit-pass over conversational-core rows: rewrite the
         assistant turn per the constitution — SAME content, native rhythm.

Usage is capped by tokens (see TOKEN_CAP):
  .venv/bin/python data/synth/gen_mcq_canon.py --kind mcq --sample 12
  .venv/bin/python data/synth/gen_mcq_canon.py --kind mcq --target 3500
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved  # noqa: E402
from data.synth.generate_sft import MIXED_WORD, CJK  # noqa: E402

SEED = 51
TOKEN_CAP = 12_000_000   # in + out + thinking, summed across workers
LEDGER = Path("data/synth/.gemini_usage_mcq_canon.json")
_usage = json.load(LEDGER.open()) if LEDGER.exists() else {"in": 0, "out": 0, "think": 0}


def tokens_used() -> int:
    return _usage["in"] + _usage["out"] + _usage["think"]


def gemini(system: str, user: str, thinking: bool = False, max_tokens: int = 3072) -> str:
    import time as _t
    if tokens_used() >= TOKEN_CAP:
        raise SystemExit(f"TOKEN CAP {TOKEN_CAP:,} reached ({tokens_used():,} used)")
    gen_cfg = {"temperature": 0.85, "maxOutputTokens": max_tokens}
    if not thinking:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {"system_instruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user}]}],
               "generationConfig": gen_cfg}
    last = None
    for model in ("gemini-3-flash-preview", "gemini-flash-latest"):
        for attempt in range(4):
            key = os.environ["GOOGLE_API_KEY_PAID"]
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "skazna-data"})
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    d = json.load(r)
                u = d.get("usageMetadata", {})
                _usage["in"] += u.get("promptTokenCount", 0)
                _usage["out"] += u.get("candidatesTokenCount", 0)
                _usage["think"] += u.get("thoughtsTokenCount", 0)
                json.dump(_usage, LEDGER.open("w"))
                try:
                    parts = d["candidates"][0]["content"]["parts"]
                except (KeyError, IndexError):
                    return ""
                text = "".join(p.get("text", "") for p in parts
                               if not p.get("thought")).strip()
                if text:
                    return text
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502):
                    _t.sleep(25 * (attempt + 1))
                    continue
                if e.code in (404, 503):
                    break
                if e.code == 400:
                    return ""
                raise
        else:
            continue
    raise last or RuntimeError("no model answered")


SYS_MCQ = """You are generating MULTIPLE-CHOICE calibration data for Скажна, a Macedonian LLM.
Purpose: teach the option-ranking FORMAT of school exams (the model knows facts
but is badly calibrated on MC format). Language constitution applies:
{constitution}

Invent ONE original exam-style question in Macedonian for subject {subject},
difficulty: high-school. RULES:
1. The question must be ORIGINAL — a fact any textbook covers, but phrased by
   you now. Never reproduce a known standardized-test item.
2. Exactly 4 options, exactly one correct. Distractors plausible, same length.
3. The fact must be REAL and verifiable — no invented specifics.
Return strict JSON: {{"question": "...", "choices": ["...","...","...","..."], "answer_idx": 0-3}}"""

MCQ_SUBJECTS = ["историја на Македонија", "географија на Македонија", "биологија",
                "физика", "хемија", "македонски јазик и литература", "светска историја",
                "математика (основна)", "информатика", "граѓанско образование"]

SYS_CANON = """You are creating KNOWLEDGE-ANNEALING data about a core Macedonian canon topic.
Below is a verified PASSAGE about the topic "{topic}".
Write 10 SHORT standalone Macedonian texts (1-3 sentences each), each restating
the key facts in a DIFFERENT form (encyclopedia line, quiz Q+A, casual mention,
textbook sentence, dialogue snippet...). NEVER alter names, numbers, dates.
Include at least 2 variants formulated as a short question with its answer.
Return strict JSON: {{"variants": ["...", ...]}} with exactly 10 strings.

PASSAGE:
{passage}"""

CANON_TOPICS = ["Крсте Петков Мисирков", "Гоце Делчев", "сказна", "Илинден",
                "Крушевска Република", "Охридско Езеро", "Кораб", "Галичка свадба",
                "АСНОМ", "кирилица", "Вардар", "Скопје", "Битола", "Охрид",
                "Даме Груев", "Јане Сандански", "ВМРО", "Тоше Проески",
                "Блаже Конески", "Кочо Рацин", "Марко Цепенков", "Битолски конгрес",
                "Пред дождот", "Медена земја", "ајвар", "тавче гравче",
                "Стоби", "Хераклеа", "Самоил", "Охридска архиепископија"]

SYS_EDIT = """You are the language editor for Скажна, a Macedonian assistant. Below is an
assistant answer produced by a weaker generator. REWRITE it so it reads like a
smart native speaker from Skopje wrote it — per this constitution:
{constitution}

STRICT RULES:
1. PRESERVE every fact, number, name, recommendation and the overall structure.
   You are polishing the LANGUAGE, not the content.
2. Kill calques, translationese word order, AI-isms, bureaucratic register.
   Restore natural Macedonian rhythm, particles, idiom.
3. Keep length within ±25% of the original. Keep any markdown structure.
4. If the original is already natural, return it with minimal touches.
Return strict JSON: {{"edited": "..."}}

ORIGINAL ANSWER:
{answer}"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_bench_grams():
    grams = set()
    for f in Path("eval/lm_eval_tasks/data").glob("*_test_mk.jsonl"):
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            for txt in [r.get("question", "")] + list(r.get("choices", [])):
                w = txt.lower().split()
                grams |= {" ".join(w[i:i+6]) for i in range(max(len(w)-5, 1))}
    return grams


def find_canon_passages():
    """Pull the best wiki passage per canon topic from the corpus."""
    pats = {t: re.compile(re.escape(t.split()[0 if t[0].isupper() else 0]), re.I)
            for t in CANON_TOPICS}
    best = {t: None for t in CANON_TOPICS}
    with gzip.open("data/processed/mk_corpus_filtered.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "mkwiki":
                continue
            text = r["text"]
            first = text[:100].lower()
            for t in CANON_TOPICS:
                tl = t.lower()
                # wiki articles open with their subject: demand the FULL topic
                # string inside the first 100 chars
                if tl in first and len(text) > 600:
                    if best[t] is None or len(text) > len(best[t]):
                        best[t] = text[:2500]
    return {t: p for t, p in best.items() if p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["mcq", "canon", "edit"])
    ap.add_argument("--sample", type=int)
    ap.add_argument("--target", type=int)
    args = ap.parse_args()
    n = args.sample or args.target
    tag = f"sample_{n}" if args.sample else "bulk"

    from dotenv import load_dotenv
    load_dotenv()
    lex = load_approved()
    constitution = Path("docs/constitution.md").read_text(encoding="utf-8")
    rng = random.Random(SEED if args.sample else SEED + 1)
    out_path = Path(f"data/synth/v51_{args.kind}_{tag}.jsonl")
    done = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    f = out_path.open("a", encoding="utf-8")
    stats = {"ok": done, "rej_parse": 0, "rej_bleed": 0, "rej_decontam": 0}

    if args.kind == "mcq":
        bench = load_bench_grams()
        while stats["ok"] < n:
            raw = gemini(SYS_MCQ.format(constitution=constitution,
                                        subject=rng.choice(MCQ_SUBJECTS)),
                         "Generate. JSON only.")
            m = JSON_RE.search(raw.replace("```json", "").replace("```", "", 1))
            try:
                d = json.loads(m.group(0)) if m else None
            except Exception:
                d = None
            if not d or len(d.get("choices", [])) != 4 or \
                    not isinstance(d.get("answer_idx"), int) or not 0 <= d["answer_idx"] <= 3:
                stats["rej_parse"] += 1
                continue
            blob = d["question"] + " " + " ".join(d["choices"])
            w = blob.lower().split()
            g6 = {" ".join(w[i:i+6]) for i in range(max(len(w)-5, 1))}
            if g6 & bench:
                stats["rej_decontam"] += 1
                continue
            h, l = find_hits(blob, lex)
            if len(h) > 2 or l or CJK.search(blob) or MIXED_WORD.search(blob):
                stats["rej_bleed"] += 1
                continue
            f.write(json.dumps({"id": f"v51q-{stats['ok']:05d}", **d},
                               ensure_ascii=False) + "\n")
            f.flush()
            stats["ok"] += 1
            if stats["ok"] % 100 == 0:
                print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)

    elif args.kind == "canon":
        passages = find_canon_passages()
        print(f"canon passages found: {len(passages)}/{len(CANON_TOPICS)}:",
              sorted(passages), flush=True)
        reps = max(1, n // max(len(passages), 1))
        for topic, passage in passages.items():
            for rep in range(reps):
                if stats["ok"] >= n:
                    break
                raw = gemini(SYS_CANON.format(topic=topic, passage=passage),
                             "Generate. JSON only.", max_tokens=2560)
                m = JSON_RE.search(raw.replace("```json", "").replace("```", "", 1))
                try:
                    d = json.loads(m.group(0)) if m else None
                except Exception:
                    d = None
                if not d or len(d.get("variants", [])) < 8:
                    stats["rej_parse"] += 1
                    continue
                blob = " ".join(d["variants"])
                h, l = find_hits(blob, lex)
                if len(h) > 2 or l or CJK.search(blob) or MIXED_WORD.search(blob):
                    stats["rej_bleed"] += 1
                    continue
                f.write(json.dumps({"id": f"v51c-{stats['ok']:05d}", "topic": topic,
                                    "variants": d["variants"][:10]},
                                   ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1
        print(f"canon done at {stats['ok']}", flush=True)

    else:  # edit
        core = []
        for path, kind in (("data/synth/v5_razgovor_pro_bulk.jsonl", "razg_pro"),
                           ("data/synth/v5_razgovor_bulk.jsonl", "razgovor"),
                           ("data/synth/v5_multi_bulk.jsonl", "multi")):
            for l in Path(path).open(encoding="utf-8"):
                r = json.loads(l)
                core.append((kind, r))
        rng.shuffle(core)
        done_ids = set()
        if out_path.exists():
            done_ids = {json.loads(l)["src_id"] for l in out_path.open(encoding="utf-8")}
        for kind, r in core:
            if stats["ok"] >= n:
                break
            rid = r["id"]
            if rid in done_ids:
                continue
            if kind == "multi":
                texts = [t["content"] for t in r["turns"] if t["role"] == "assistant"]
                orig = texts[-1]          # edit the longest-stake final turn
            else:
                orig = r["assistant"]
            raw = gemini(SYS_EDIT.format(constitution=constitution, answer=orig[:2800]),
                         "Rewrite. JSON only.", thinking=True, max_tokens=3584)
            m = JSON_RE.search(raw.replace("```json", "").replace("```", "", 1))
            try:
                d = json.loads(m.group(0)) if m else None
            except Exception:
                d = None
            ed = d.get("edited", "") if d else ""
            if not ed or not (0.6 * len(orig) <= len(ed) <= 1.4 * len(orig)):
                stats["rej_parse"] += 1
                continue
            h, l = find_hits(ed, lex)
            if len(h) > 2 or l or CJK.search(ed) or MIXED_WORD.search(ed):
                stats["rej_bleed"] += 1
                continue
            f.write(json.dumps({"id": f"v51e-{stats['ok']:05d}", "src_id": rid,
                                "kind": kind, "original": orig, "edited": ed},
                               ensure_ascii=False) + "\n")
            f.flush()
            stats["ok"] += 1
            if stats["ok"] % 100 == 0:
                print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)

    f.close()
    print(f"DONE {stats} | {tokens_used():,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
