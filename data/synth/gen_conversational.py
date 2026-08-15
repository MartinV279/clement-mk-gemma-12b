#!/usr/bin/env python3
"""Conversational-core data generation — the research-amended round.

Kinds (each sample-gated at 20 rows before any bulk run):
  kratki    short question -> SHORT correct answer (1-2 sentences, discipline)
  struct    markdown-structured answers where structure genuinely helps
  razgovor  real-user-register short prompts -> deep warm answers. The earlier
            shallowness diagnosis was prompt DISTRIBUTION, not length dials.
  code      practical MK programming QA, thinking OFF, syntax-gated
  refusal   questions whose facts we DON'T cover -> honest useful "не знам"
  para      ~10 paraphrase/format variants per verified fact (ANNEAL data,
            pretraining-style, per Physics-of-LMs 3.1)

Usage is capped by tokens (see TOKEN_CAP), summed across concurrent workers:
  .venv/bin/python data/synth/gen_conversational.py --kind kratki --sample 20
  .venv/bin/python data/synth/gen_conversational.py --kind kratki --target 2000
"""

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved  # noqa: E402
from data.synth.generate_sft import MIXED_WORD, CJK  # noqa: E402

SEED = 55
TOKEN_CAP = 40_000_000   # in + out + thinking, summed across workers
LEDGER = Path("data/synth/.gemini_usage_conversational.json")

_usage = json.load(LEDGER.open()) if LEDGER.exists() else {"in": 0, "out": 0, "think": 0}


def tokens_used() -> int:
    return _usage["in"] + _usage["out"] + _usage["think"]


def gemini(system: str, user: str, thinking: bool = False, max_tokens: int = 3072,
           models: tuple = ("gemini-3-flash-preview", "gemini-flash-latest")) -> str:
    if tokens_used() >= TOKEN_CAP:
        raise SystemExit(f"TOKEN CAP {TOKEN_CAP:,} reached ({tokens_used():,} used)")
    gen_cfg = {"temperature": 0.9, "maxOutputTokens": max_tokens}
    if not thinking:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {"system_instruction": {"parts": [{"text": system}]},
               "contents": [{"role": "user", "parts": [{"text": user}]}],
               "generationConfig": gen_cfg}
    import time as _t
    last = None
    for model in models:
      for _attempt in range(4):
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
                return ""   # safety-filtered / empty candidate: skip row
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                _t.sleep(25 * (_attempt + 1))
                continue        # backoff, retry same model
            if e.code in (404, 503):
                break           # try next model name
            if e.code == 400:
                return ""
            raise
      else:
        continue
    raise last or RuntimeError("no model answered")


# ---------------------------------------------------------------- prompts
BASE = """You are generating training data for Скажна, a Macedonian-language assistant.
The language constitution below governs ALL Macedonian prose (mandatory):
{constitution}

=== GOLD REGISTER (native-written examples — match this voice) ===
{gold}
"""

SYS_KRATKI = BASE + """
Invent ONE exchange teaching LENGTH DISCIPLINE. The user asks a simple factual
or practical question that deserves a SHORT answer ("Колку...", "Кој...",
"Кога...", "Дали...", quick everyday asks). The assistant answers in
MAXIMUM 1-2 sentences — correct, direct, natural — and STOPS. No lists, no
caveats, no "дополнително". If a number/name is needed and commonly known,
state it plainly; if it genuinely varies, one short sentence saying what it
depends on. THE ENTIRE VALUE OF THIS ROW IS THE RESTRAINT.
Topic area: {topic}.
Return strict JSON: {{"user": "...", "assistant": "..."}}"""

SYS_STRUCT = BASE + """
Invent ONE exchange where MARKDOWN STRUCTURE genuinely helps: steps, a
comparison, a plan, an organized overview. The answer USES markdown properly:
**bold** key terms, numbered steps or bullets, ### subheadings when long.
Substance first — every bullet carries a concrete fact or action, never
decoration. 600-1100 characters. Native Macedonian per the constitution.
Topic: {topic}.
Return strict JSON: {{"user": "...", "assistant": "..."}}"""

SYS_RAZGOVOR = BASE + """
Invent ONE exchange in the register of a REAL Macedonian user typing quickly:
the question is SHORT (5-15 words), informal (на „ти"), often emotional or
situational — the kind of thing people actually type ("Скарав се со...",
"Досадно ми е...", "Вреди ли да...", "Како да ѝ кажам..."). The assistant
answers with WARMTH AND REAL DEPTH: concrete, personal, culturally grounded
(Skopje/Macedonia everyday reality), 500-900 characters, no bullet-list
therapy-slop, no AI disclaimers — a smart friend who actually engages.
Scenario area: {topic}.
Return strict JSON: {{"user": "...", "assistant": "..."}}"""

SYS_CODE = BASE + """
Invent ONE realistic programming exchange: a Macedonian user with a concrete
problem (often with a broken snippet), forum tone, informal. The answer:
explanation IN MACEDONIAN of the problem and WHY, then a fenced ```code block
that WORKS (complete, runnable, no "..." placeholders), then the key lines
explained and the most common mistake. Identifiers English, comments Macedonian.
Topic: {topic}. Difficulty: {level}.
Return strict JSON: {{"user": "...", "assistant": "...", "lang": "python|javascript|sql|bash|html|other"}}"""

SYS_REFUSAL = BASE + """
Invent ONE exchange teaching HONEST KNOWLEDGE BOUNDARIES. The user asks a
SPECIFIC factual question that a language model CANNOT reliably know: exact
current prices/schedules/contacts in Macedonia, precise statistics nobody
memorizes, obscure local biography details, recent/changeable rules. The
assistant does NOT invent specifics. It says plainly it does not know the
exact answer, gives whatever reliable CONTEXT it can (what is generally true,
what it depends on), and tells the user exactly WHERE to verify (the right
institution/site/office). Warm and useful — never a curt refusal, never
bureaucratic. 300-600 characters.
Trap area: {topic}.
Return strict JSON: {{"user": "...", "assistant": "..."}}"""

SYS_MULTI = BASE + """
Invent ONE realistic MULTI-TURN conversation (4-6 turns total, alternating,
starting with user). A real Macedonian user with an evolving need: they ask,
get an answer, follow up with a complication/clarification ("а што ако...",
"да, ама..."), maybe change direction. The assistant stays consistent across
turns, refers back naturally, keeps the constitution register. Turn lengths
natural: user turns short, assistant turns substantive but not bloated.
Scenario: {topic}.
Return strict JSON: {{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]}}"""

SYS_STAPICI = BASE + """
Invent ONE exchange teaching FALSE-PREMISE CORRECTION. The user asks a
question containing a WRONG assumption (wrong date/person/place/причина,
a myth, a mixed-up fact — plausible mistakes real people make about
Macedonia or general knowledge). The assistant FIRST kindly corrects the
premise (никогаш снисходливо), THEN answers the underlying need with the
correct information. 300-700 characters, natural tone.
Trap domain: {topic}.
Return strict JSON: {{"user": "...", "assistant": "..."}}"""

SYS_PARA = """You are creating KNOWLEDGE-ANNEALING data for a Macedonian language model.
Below is a verified FACT PASSAGE and the discrete CLAIMS it supports.
Write {n} SHORT standalone Macedonian texts (1-3 sentences each), each
restating the SAME core facts in a DIFFERENT form: encyclopedia sentence,
casual explanation, quiz question WITH the answer, "did you know" style,
textbook sentence, dialogue snippet, comparison framing, etc. Vary word order,
syntax and framing aggressively; NEVER alter names, numbers or dates.
Language: native Macedonian (no calques). Return strict JSON:
{{"variants": ["...", "...", ...]}} with exactly {n} strings.

PASSAGE:
{passage}

CLAIMS:
{claims}"""

TOPICS = {
    "kratki": ["мерки и конверзии", "географија на Македонија", "секојдневни правни/административни",
               "храна и готвење", "спорт", "техника и уреди", "здравје basics", "јазик и правопис",
               "историски датуми", "пари и цени концепти"],
    "struct": ["здравје", "финансии", "патување", "кариера", "домаќинство", "технологија",
               "образование", "храна", "спорт", "право", "автомобили", "градинарство"],
    "razgovor": ["односи и семејство", "работа и шеф", "пари и стрес", "соседи и станови",
                 "излегување и досада", "љубов и раскинувања", "пријателства", "здравствени грижи",
                 "селидба и иселување", "празници и гости", "деца и родители", "спорт и хоби"],
    "code": ["python основи", "python датотеки", "python pandas", "python грешки",
             "javascript основи", "javascript DOM", "SQL прашалници", "bash скрипти",
             "git команди", "HTML/CSS", "python API", "regex", "excel формули", "дебагирање"],
    "refusal": ["точни цени и тарифи во Македонија", "работно време и контакти на институции",
                "прецизни локални статистики", "возни редови и превоз", "малку познати личности",
                "актуелни правила што се менуваат", "спортски резултати и состави",
                "медицински дози и специфики"],
    "multi": ["планирање патување", "решавање технички проблем", "готвење со комплкации",
              "бирократска постапка чекор по чекор", "кариерна дилема", "купување автомобил/стан",
              "учење нова вештина", "организирање настан", "здравствено прашање со follow-up",
              "семеен конфликт што се развива"],
    "stapici": ["македонска историја", "географија", "јазик и зборови", "наука",
                "спорт и рекорди", "култура и уметност", "институции и права", "технологија"],
    "razgovor_pro": ["односи и семејство", "тешки животни одлуки", "иселување и носталгија",
                     "загуба и тага", "работна криза", "родителски дилеми",
                     "пријателства што се менуваат", "осаменост"],
}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
MD = re.compile(r"\*\*[^*]+\*\*|^\d+\.\s|^[-•] |^#{1,3} ", re.M)


def check_code(answer: str, lang: str) -> bool:
    blocks = FENCE.findall(answer)
    if not blocks:
        return False
    for tag, body in blocks:
        l = (tag or lang).lower()
        if l == "python":
            try:
                ast.parse(body)
            except SyntaxError:
                return False
        elif l in ("javascript", "js"):
            try:
                r = subprocess.run(["node", "--check", "/dev/stdin"], input=body.encode(),
                                   capture_output=True, timeout=10)
                if r.returncode != 0:
                    return False
            except FileNotFoundError:
                pass
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["kratki", "struct", "razgovor", "code", "refusal", "para",
                             "multi", "stapici", "razgovor_pro"])
    ap.add_argument("--sample", type=int)
    ap.add_argument("--target", type=int)
    args = ap.parse_args()
    n = args.sample or args.target
    tag = f"sample_{n}" if args.sample else "bulk"

    from dotenv import load_dotenv
    load_dotenv()
    lex = load_approved()
    constitution = Path("docs/constitution.md").read_text(encoding="utf-8")
    hw = [json.loads(l)["conversations"]
          for l in Path("data/handwritten/handwritten.jsonl").open(encoding="utf-8")]
    anchors = [c for c in hw if len(c) == 2 and 600 <= len(c[1]["content"]) <= 1200][:2]
    gold = "\n\n".join(f"ПРАШАЊЕ: {c[0]['content']}\nОДГОВОР: {c[1]['content']}"
                       for c in anchors)
    rng = random.Random(SEED if args.sample else SEED + 1)

    out_path = Path(f"data/synth/v5_{args.kind}_{tag}.jsonl")
    done = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    f = out_path.open("a", encoding="utf-8")
    stats = {"ok": done, "rej_parse": 0, "rej_bleed": 0, "rej_len": 0, "rej_gate": 0}

    # --- para kind iterates over verified facts, not freeform topics ---
    if args.kind == "para":
        facts = [json.loads(l) for l in
                 Path("data/processed/facts_verified.jsonl").open(encoding="utf-8")]
        done_src = set()
        if out_path.exists():
            done_src = {json.loads(l)["src_id"] for l in out_path.open(encoding="utf-8")}
        facts = [x for x in facts if x["id"] not in done_src]
        rng.shuffle(facts)
        for fact in facts:
            if stats["ok"] >= n:
                break
            raw = gemini(SYS_PARA.format(n=10, passage=fact["passage"][:4000],
                                         claims="\n".join(fact["claims"])),
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
            f.write(json.dumps({"id": f"v5p-{stats['ok']:05d}", "src_id": fact["id"],
                                "variants": d["variants"][:10]}, ensure_ascii=False) + "\n")
            f.flush()
            stats["ok"] += 1
            if stats["ok"] % 50 == 0:
                print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)
        f.close()
        print(f"DONE {stats} | {tokens_used():,} tokens -> {out_path}")
        return

    SYS = {"kratki": SYS_KRATKI, "struct": SYS_STRUCT, "razgovor": SYS_RAZGOVOR,
           "code": SYS_CODE, "refusal": SYS_REFUSAL, "multi": SYS_MULTI,
           "stapici": SYS_STAPICI, "razgovor_pro": SYS_RAZGOVOR}[args.kind]
    # pro previews 429-throttled on this key -> flash + thinking ON for gold depth
    MODELS = ("gemini-3-flash-preview", "gemini-flash-latest")
    THINK = args.kind == "razgovor_pro"
    while stats["ok"] < n:
        kw = {"constitution": constitution, "gold": gold,
              "topic": rng.choice(TOPICS[args.kind])}
        if args.kind == "code":
            kw["level"] = rng.choice(["почетник", "средно", "напредно"])
        raw = gemini(SYS.format(**kw), "Generate. JSON only.", thinking=THINK,
                     max_tokens=6144 if args.kind in ("code", "multi") else 3072,
                     models=MODELS)
        m = JSON_RE.search(raw.replace("```json", "").replace("```", "", 1))
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
        if args.kind == "multi":
            t = d.get("turns") if d else None
            ok = (t and 4 <= len(t) <= 6
                  and all(x.get("role") == ("user" if i % 2 == 0 else "assistant")
                          and 5 <= len(x.get("content", "")) <= 1400
                          for i, x in enumerate(t)))
            if not ok:
                stats["rej_parse"] += 1
                continue
            blob = " ".join(x["content"] for x in t)
            h, l = find_hits(blob, lex)
            if len(h) > 2 or l or CJK.search(blob) or MIXED_WORD.search(blob):
                stats["rej_bleed"] += 1
                continue
            f.write(json.dumps({"id": f"v5m-{stats['ok']:05d}", "kind": "multi",
                                "turns": t}, ensure_ascii=False) + "\n")
            f.flush()
            stats["ok"] += 1
            if stats["ok"] % 50 == 0:
                print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)
            continue
        if not d or "user" not in d or "assistant" not in d:
            stats["rej_parse"] += 1
            continue
        u, a = d["user"], d["assistant"]
        # per-kind gates
        if args.kind == "kratki" and (len(a) > 260 or a.count(".") > 3 or MD.search(a)):
            stats["rej_gate"] += 1
            continue
        if args.kind == "struct" and not MD.search(a):
            stats["rej_gate"] += 1
            continue
        if args.kind == "razgovor" and not (400 <= len(a) <= 1100 and len(u) <= 140
                                            and not MD.search(a)):
            stats["rej_gate"] += 1
            continue
        if args.kind == "refusal" and not (250 <= len(a) <= 700):
            stats["rej_gate"] += 1
            continue
        if args.kind == "stapici" and not (250 <= len(a) <= 800):
            stats["rej_gate"] += 1
            continue
        if args.kind == "razgovor_pro" and not (400 <= len(a) <= 1200 and len(u) <= 160
                                                and not MD.search(a)):
            stats["rej_gate"] += 1
            continue
        if args.kind == "code" and not check_code(a, d.get("lang", "")):
            stats["rej_gate"] += 1
            continue
        blob = u + " " + a
        prose = FENCE.sub(" ", blob) if args.kind == "code" else blob
        h, l = find_hits(prose, lex)
        if len(h) > 2 or l or CJK.search(prose) or \
                (args.kind != "code" and MIXED_WORD.search(prose)):
            stats["rej_bleed"] += 1
            continue
        f.write(json.dumps({"id": f"conv-{args.kind[0]}-{stats['ok']:05d}",
                            "kind": args.kind, **d}, ensure_ascii=False) + "\n")
        f.flush()
        stats["ok"] += 1
        if stats["ok"] % 50 == 0:
            print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)
    f.close()
    print(f"DONE {stats} | {tokens_used():,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
