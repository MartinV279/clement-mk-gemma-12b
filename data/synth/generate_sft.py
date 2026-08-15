#!/usr/bin/env python3
"""Synthetic SFT generation — depth-calibrated, multi-teacher.

Written after a blind arena round showed the previous batch produced answers
that were correct but shallow (median 165 chars against yak's 594). Two causes,
both fixed here:
  1. the earlier LENGTH_DIALS asked for "кратка/концизен" in 2 of 3 settings on
     every call -> systemic brevity. Replaced by DEPTH_DIALS, which calibrate
     by question type and never ask for terse.
  2. the previous teacher lost a blind bakeoff 0/10. Replaced by the winners:
     Gemini (5/10) primary, GLM-5.2 (2/10) secondary.

Teachers are rate-limited rather than quota-limited in practice: adaptive
throttle + key rotation + exponential backoff, fully resumable (re-run the same
command after any interruption and it continues).

Usage:
  python data/synth/generate_sft.py --target 6000
  python data/synth/generate_sft.py --target 20 --out data/synth/sft_sample.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved  # noqa: E402

SEED = 1903
OUT_DEFAULT = Path("data/synth/sft_synth.jsonl")
TOPICS = Path("data/synth/seeds/topics_mk.yaml")

# Depth is the whole point here. No dial asks for brevity; the shortest setting is
# "direct but complete", which is what a genuinely simple question deserves.
DEPTH_DIALS = [
    ("богат", "600-1000 знаци: конкретни факти, барем еден пример или чекор-по-чекор, "
              "и кратка практична забелешка на крајот", 0.55),
    ("многу богат", "900-1400 знаци: разработена материја со поднаслови или нумерирани "
                    "точки, конкретни бројки/имиња каде што ги знаеш, и што луѓето "
                    "најчесто го погрешуваат", 0.25),
    ("директен но целосен", "250-500 знаци: прашањето е навистина едноставно, па одговорот "
                            "е краток — но сепак дава причина или контекст, не гола реченица", 0.20),
]

PERSONA_DIALS = ["обичен корисник, неформално на „ти“",
                 "постар корисник, учтиво",
                 "млад корисник, разговорно со сленг",
                 "корисник кој пишува кратко и директно",
                 "корисник кој дава многу контекст"]

SYSTEM = """Генерираш податоци за обучување на македонски јазичен асистент по име Скажна.
Работиш строго по оваа јазична конституција:

{constitution}

=== ЗЛАТЕН СТАНДАРД (роден регистар — вака треба да звучи асистентот) ===
{gold_examples}

=== ШТО ПРАВИШ ===
Смислуваш ЕДНА реална размена: прашање што вистински македонски корисник би го
поставил, и одговор од асистентот.

ФОРМАТ (строг JSON, ништо друго):
{{"user": "...", "assistant": "..."}}

=== ПРАВИЛА ЗА ОДГОВОРОТ ===
1. ДЛАБОЧИНА: {depth_spec}
2. Секоја реченица носи содржина. Никакви празни уводи („Одлично прашање!"),
   никакви AI-фрази, никакво повторување на прашањето.
3. Конкретност пред општост: имиња, бројки, чекори, примери. Ако не си сигурен
   во факт — кажи го тоа отворено во одговорот наместо да измислуваш.
4. Роден македонски: без преводни конструкции, без несоодветни кирилични букви
   (ђћъщюяйьэыёіїєґў), без латинично-кирилични мешавини во ист збор.
5. Прашањето го пишува {persona}.
{fact_clause}"""

FACT_CLAUSE_HIGH = ("6. ВНИМАНИЕ, висок ризик од измислување: држи се до она што сигурно го "
                    "знаеш. Наместо измислен податок, напиши каде корисникот може да провери.")


def load_topics() -> dict:
    """Minimal YAML reader for the pinned topics file (no pyyaml dependency)."""
    cats, cur = {}, None
    key = None
    for raw in TOPICS.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            continue
        if indent == 2 and line.endswith(":"):
            cur = line[:-1]
            cats[cur] = {"subtopics": [], "fact_risk": "low"}
            key = None
        elif cur and ":" in line and indent >= 4:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            key = k
            if k == "subtopics":
                v = v.strip("[]")
                cats[cur]["subtopics"] += [s.strip() for s in v.split(",") if s.strip()]
            elif k in ("fact_risk", "tier"):
                cats[cur][k] = v
            elif k == "count":
                cats[cur]["count"] = int(v)
        elif cur and key == "subtopics":
            cats[cur]["subtopics"] += [s.strip(" ,[]") for s in line.split(",") if s.strip(" ,[]")]
    return cats


def build_system(depth_spec: str, persona: str, fact_risk: str) -> str:
    constitution = Path("docs/constitution.md").read_text(encoding="utf-8")
    hw = [json.loads(l)["conversations"]
          for l in Path("data/handwritten/handwritten.jsonl").open(encoding="utf-8")]
    anchors = [c for c in hw if len(c) == 2 and 600 <= len(c[1]["content"]) <= 1200][:3]
    gold = "\n\n".join(f"ПРАШАЊЕ: {c[0]['content']}\nОДГОВОР: {c[1]['content']}"
                       for c in anchors)
    return SYSTEM.format(constitution=constitution, gold_examples=gold,
                         depth_spec=depth_spec, persona=persona,
                         fact_clause=FACT_CLAUSE_HIGH if fact_risk == "high" else "")


class RateLimited(Exception):
    pass


class QuotaExceeded(Exception):
    pass


# Generation is capped by TOKENS consumed, not by request count or wall-clock:
# a retry storm on long prompts is the failure mode that actually runs a key
# dry. Every worker keeps its own usage ledger (data/synth/.gemini_usage_*.json)
# and sums ALL ledgers before each call, so parallel workers share one cap.
TOKEN_CAP = 120_000_000  # in + out, across every worker
USAGE_GLOB = "data/synth/.gemini_usage_*.json"
USAGE_PATH = [None]  # set in main() from the worker's --out name
_usage = {"pro_in": 0, "pro_out": 0, "flash_in": 0, "flash_out": 0,
          "lite_in": 0, "lite_out": 0}


def total_tokens() -> int:
    import glob as _g
    tot = 0
    for p in _g.glob(USAGE_GLOB):
        try:
            s = json.load(open(p))
            for tier in ("pro", "flash", "lite"):
                tot += s.get(f"{tier}_in", 0) + s.get(f"{tier}_out", 0)
        except Exception:
            pass
    return tot


def _record(tier: str, usage: dict) -> None:
    _usage[f"{tier}_in"] += usage.get("promptTokenCount", 0)
    _usage[f"{tier}_out"] += usage.get("candidatesTokenCount", 0)
    if USAGE_PATH[0]:
        json.dump(_usage, open(USAGE_PATH[0], "w"))


# newest usable text models on the paid key (probed 2026-08-05): 3.1-pro is the
# top pro; 3.1-flash exists only as -lite, so 3-flash stays the quality flash
PRO_MODELS = ("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-pro-latest")
FLASH_MODELS = ("gemini-3-flash-preview", "gemini-flash-latest", "gemini-2.5-flash")
LITE_MODELS = ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview")


def gemini_call(models: tuple, tier: str, system: str, user: str) -> str:
    if total_tokens() >= TOKEN_CAP:
        raise QuotaExceeded(f"gemini token cap {TOKEN_CAP:,} reached")
    for model in models:
        for suffix in ("_PAID", "_1", "_2"):
            key = os.environ.get("GOOGLE_API_KEY" + suffix)
            if not key:
                continue
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            payload = {"system_instruction": {"parts": [{"text": system}]},
                       "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "generationConfig": {"temperature": 0.9, "maxOutputTokens": 4096}}
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "skazna-data"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.load(r)
                _record(tier, d.get("usageMetadata", {}))
                parts = d["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return text
            except urllib.error.HTTPError as e:
                if e.code in (429, 404, 503):
                    continue
                raise
    raise RateLimited("gemini: every model/key rate-limited")


def gemini(system: str, user: str) -> str:
    return gemini_call(FLASH_MODELS, "flash", system, user)


def gemini_pro(system: str, user: str) -> str:
    # pro is the heaviest tier: reserved for hard slices, degrades to flash on 429
    try:
        return gemini_call(PRO_MODELS, "pro", system, user)
    except RateLimited:
        return gemini_call(FLASH_MODELS, "flash", system, user)


def gemini_lite(system: str, user: str) -> str:
    # 3.1-flash-lite: newest lite tier — good enough for the shortest depth
    # the lightest tier; upgrades to flash if lite is unavailable
    try:
        return gemini_call(LITE_MODELS, "lite", system, user)
    except RateLimited:
        return gemini_call(FLASH_MODELS, "flash", system, user)


def openai_compat(base: str, env_prefix: str, model: str, system: str, user: str) -> str:
    payload = json.dumps({"model": model, "temperature": 0.9, "max_tokens": 4096,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]}).encode()
    for suffix in ("_1", "_2"):
        key = os.environ.get(env_prefix + suffix)
        if not key:
            continue
        req = urllib.request.Request(f"{base}/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "skazna-data",
                                              "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.load(r)
            return (d["choices"][0]["message"].get("content") or "").strip()
        except urllib.error.HTTPError as e:
            if e.code in (402, 429, 500, 503):  # 402: samba trial depleted
                continue
            raise
    raise RateLimited(f"{model}: every key rate-limited")


def glm(system: str, user: str) -> str:
    return openai_compat("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                         "z-ai/glm-5.2", system, user)


def gemma31(system: str, user: str) -> str:
    return openai_compat("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY",
                         "gemma-4-31B-it", system, user)


# GLM-5.2 REMOVED (validation 2026-08-02): 48% of its production rows carried
# mixed-script corruption, CJK tokens, or Serbisms — bakeoff quality did not
# hold at volume. All its rows are quarantined in quarantine_glm.jsonl.
# gemini validated clean on 95/95 rows.
#
# Tier routing: PRO for the slices where quality
# is hardest to fake — fact-risk-high categories and the deepest dial — FLASH
# (the bakeoff winner) for the bulk. gemma-4-31b free tier stays as overflow.
TEACHERS = [("gemini", gemini, 0.80), ("gemma-4-31b", gemma31, 0.20)]
PRO_CATEGORIES = {"mk_culture", "institutions_practical", "summarization"}

# variety dial — the topic seed alone converges to near-identical scenarios
ANGLES = ["почетник е и нема поим од темата", "веќе пробал нешто што не успеало",
          "брза и му треба решение денес", "сака детално да разбере зошто, не само како",
          "скептичен е и бара докази", "има ограничен буџет",
          "прашува за туѓ проблем (родител, дете, колега)", "споредува две опции"]


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
MIXED_WORD = re.compile(r"\b[а-шѓќѕџљњјА-ШЃЌЅЏЉЊЈ]+[a-zA-Z]{2,}[а-шѓќѕџљњј]*\b"
                        r"|\b[a-zA-Z]{2,}[а-шѓќѕџљњј]+\b")
CJK = re.compile(r"[一-鿿぀-ヿ]")


def parse(raw: str) -> tuple:
    m = JSON_RE.search(raw.replace("```json", "").replace("```", ""))
    if not m:
        raise ValueError("no JSON object in reply")
    d = json.loads(m.group(0))
    u, a = d.get("user", "").strip(), d.get("assistant", "").strip()
    if not u or not a:
        raise ValueError("empty user/assistant field")
    return u, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--glm-share", type=float, default=0.30)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    lex = load_approved()
    # decontamination: the frozen vibes prompts must never appear in training
    vibes = [json.loads(l)["prompt"].lower()
             for l in Path("eval/vibes_prompts.jsonl").open(encoding="utf-8")]
    vibes_grams = set()
    for p in vibes:
        w = p.split()
        vibes_grams |= {" ".join(w[i:i + 8]) for i in range(max(len(w) - 7, 1))}

    cats = load_topics()
    weights = [(c, s) for c, s in cats.items() for _ in range(s.get("count", 1000) // 500 or 1)]
    # seed from the output filename: parallel workers MUST diverge (the fixed
    # SEED gave all 4 workers identical topic sequences -> mass near-dups)
    rng = random.Random(f"{SEED}:{args.out}")

    out_path = Path(args.out)
    done = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    print(f"{done} already generated -> target {args.target}", flush=True)
    fh = out_path.open("a", encoding="utf-8")

    dials, dial_w = [d[:2] for d in DEPTH_DIALS], [d[2] for d in DEPTH_DIALS]
    USAGE_PATH[0] = f"data/synth/.gemini_usage_{Path(args.out).stem}.json"
    # resume the ledger — a worker restart must never reset cap accounting
    if Path(USAGE_PATH[0]).exists():
        _usage.update(json.load(open(USAGE_PATH[0])))
    stats = {n: 0 for n, _, _ in TEACHERS}
    stats.update({"gemini-pro": 0, "gemini-lite": 0, "bleed": 0, "parse": 0, "short": 0, "vibes": 0})
    cooldown = {}
    subtopic_counts = {}
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            d = json.loads(l)
            k = (d.get("category"), d.get("subtopic"))
            subtopic_counts[k] = subtopic_counts.get(k, 0) + 1
    delay = 1.0
    i = done
    while i < args.target:
        cat, spec = rng.choice(weights)
        depth_name, depth_spec = rng.choices(dials, weights=dial_w)[0]
        persona = rng.choice(PERSONA_DIALS)
        system = build_system(depth_spec, persona, spec.get("fact_risk", "low"))
        subtopic = rng.choice(spec["subtopics"]) if spec["subtopics"] else cat
        if subtopic_counts.get((cat, subtopic), 0) >= 8:  # variety cap
            continue
        angle = rng.choice(ANGLES)
        user_msg = (f"Тема: {cat} · поттема: {subtopic}.\n"
                    f"Агол: корисникот {angle}.\n"
                    f"Длабочина: {depth_name}.\n"
                    f"ВАЖНО: не наведувај конкретни телефонски броеви, цени или "
                    f"износи од закони освен ако си целосно сигурен — подобро "
                    f"упати каде се проверува.\n"
                    f"Смисли една размена. Врати само JSON.")
        now = time.time()
        avail = [(n, f, w) for n, f, w in TEACHERS if cooldown.get(n, 0) < now]
        if not avail:
            nap = min(min(cooldown.values()) - now, 300)
            print(f"  all teachers cooling down, sleeping {nap:.0f}s", flush=True)
            time.sleep(max(nap, 5))
            continue
        tname, tfn, _ = rng.choices(avail, weights=[w for _, _, w in avail])[0]
        # upgrade to pro on the hard slices (fact-heavy category, or the
        # deepest dial when it landed on gemini anyway)
        if tname == "gemini" and (cat in PRO_CATEGORIES
                                  or spec.get("fact_risk") == "high"
                                  or depth_name == "многу богат"):
            tname, tfn = "gemini-pro", gemini_pro
        elif tname == "gemini" and depth_name == "директен но целосен":
            tname, tfn = "gemini-lite", gemini_lite
        try:
            raw = tfn(system, user_msg)
            delay = max(0.5, delay * 0.9)
        except QuotaExceeded as e:
            print(f"QUOTA STOP: {e} — total {total_tokens():,} tokens", flush=True)
            break
        except RateLimited:
            # park under the BASE teacher name — "gemini-pro" is a routing
            # alias, and parking it under its own key left stale entries that
            # dragged the all-cooling sleep computation negative
            base = "gemini" if tname.startswith("gemini") else tname
            cooldown[base] = now + 900
            print(f"  {base} rate-limited -> cooling 15 min", flush=True)
            continue
        except Exception as e:
            print(f"  {tname} error: {e}", flush=True)
            time.sleep(5)
            continue

        try:
            u, a = parse(raw)
        except Exception:
            stats["parse"] += 1
            continue
        # screen BOTH sides (validation found Russian/mixed-script in questions)
        # plus mixed-script words and CJK anywhere (GLM failure signature)
        both = u + " " + a
        h, l = find_hits(a, lex)
        h2, l2 = find_hits(u, lex)
        if len(h) > 2 or l or len(h2) > 2 or l2 or MIXED_WORD.search(both) or CJK.search(both):
            stats["bleed"] += 1
            continue
        if len(a) < 200:
            stats["short"] += 1
            continue
        low = (u + " " + a).lower()
        if any(g in low for g in vibes_grams):
            stats["vibes"] += 1
            continue

        fh.write(json.dumps({"id": f"synth-{i:06d}", "source": "synth",
                             "category": cat, "subtopic": subtopic,
                             "depth": depth_name, "teacher": tname,
                             "conversations": [{"role": "user", "content": u},
                                               {"role": "assistant", "content": a}]},
                            ensure_ascii=False) + "\n")
        fh.flush()
        stats[tname] += 1
        subtopic_counts[(cat, subtopic)] = subtopic_counts.get((cat, subtopic), 0) + 1
        i += 1
        if i % 25 == 0:
            print(f"[{i}/{args.target}] {stats} tok={total_tokens():,} delay={delay:.1f}s", flush=True)
        time.sleep(delay)
    fh.close()
    print("DONE", stats)


if __name__ == "__main__":
    main()
