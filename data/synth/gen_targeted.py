#!/usr/bin/env python3
"""Targeted consolidation data for the ship build — evidence-driven pack design.

Arena loss analysis of the preceding builds against yak (eval/blind/verdict_*.json)
says every single loss is either a SHORT-FACTUAL ask or a CULTURE item. The depth
categories (објаснувања/пишување/разговор/секојдневни/техничко/јазик) come back
4-0/5-0. So this round goes entirely to brevity + format compliance and grounded
culture, and adds no further depth data.

Kinds:
  kratki   short-answer discipline + explicit format compliance. Emits BOTH an
           SFT row (short, correct, native) and a preference pair whose rejected
           side is the same content padded out — the exact failure mode we lose on.
  kultura  grounded cultural answers (film/food/customs/history) written from a
           verified passage, native register, opinionated where the ask invites it.
  math     worked step-by-step solutions. Problems are TEMPLATE-generated in
           Python so the final answer is provably correct; Gemini only writes the
           Macedonian prose. No trust placed in the model's arithmetic.
  scimcq   ARC-Easy-style science reasoning MCQ (not trivia — reasoning about a
           fact), for the arc_easy_mk gap. Panel-verified downstream.

Every kind is decontaminated against:
  - benchmark test sets (6-gram)      -> eval/lm_eval_tasks/data/*_test_mk.jsonl
  - the FROZEN vibes eval set          -> 8-gram AND 0.6 token-Jaccard
The Jaccard gate matters because the losing vibes prompts are 5-8 words long and
would slip an n-gram check.

Usage is capped by tokens (see TOKEN_CAP), summed across per-kind ledgers:
  .venv/bin/python data/synth/gen_targeted.py --kind kratki --sample 12
  .venv/bin/python data/synth/gen_targeted.py --kind kratki --target 1000
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import time as _t
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved  # noqa: E402
from data.synth.generate_sft import MIXED_WORD, CJK  # noqa: E402

SEED = 6
TOKEN_CAP = 12_000_000   # in + out + thinking, summed across workers
# One ledger PER KIND so concurrent workers never race on the same file, but the
# cap is checked against the SUM of all of them — a per-file cap would let four
# parallel kinds each consume the full cap.
LEDGER = Path("data/synth/.gemini_usage_targeted.json")   # rebound per-kind in main()
_usage = {"in": 0, "out": 0, "think": 0}


def _tok(d) -> int:
    return d["in"] + d["out"] + d["think"]


def tokens_used() -> int:
    """Global usage: this worker's live counters plus every sibling ledger."""
    total = _tok(_usage)
    for p in Path("data/synth").glob(".gemini_usage_targeted_*.json"):
        if p == LEDGER:
            continue
        try:
            total += _tok(json.load(p.open()))
        except Exception:
            pass
    return total


def gemini(system: str, user: str, thinking: bool = False, max_tokens: int = 3072,
           temperature: float = 0.85) -> str:
    if tokens_used() >= TOKEN_CAP:
        raise SystemExit(f"TOKEN CAP {TOKEN_CAP:,} reached ({tokens_used():,} used)")
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
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


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------- decontam ---

def _toks(s: str):
    return [w.lower() for w in WORD_RE.findall(s)]


def load_bench_grams():
    grams = set()
    for f in Path("eval/lm_eval_tasks/data").glob("*_test_mk.jsonl"):
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            fields = [r.get("question", ""), r.get("query", "")] + list(r.get("choices", []))
            for txt in fields:
                w = _toks(txt)
                grams |= {" ".join(w[i:i + 6]) for i in range(max(len(w) - 5, 1))}
    return grams


class VibesGuard:
    """The frozen eval set must never appear in training data — not as an n-gram
    copy and not as a near-paraphrase. The losing prompts are 5-8 words, so an
    8-gram test alone would pass them; the Jaccard gate is what actually bites."""

    def __init__(self, path="eval/vibes_prompts.jsonl", jaccard=0.6):
        self.sets, self.grams, self.j = [], set(), jaccard
        for l in Path(path).open(encoding="utf-8"):
            w = _toks(json.loads(l)["prompt"])
            self.sets.append(set(w))
            self.grams |= {" ".join(w[i:i + 8]) for i in range(max(len(w) - 7, 1))}

    def hits(self, text: str) -> bool:
        w = _toks(text)
        g8 = {" ".join(w[i:i + 8]) for i in range(max(len(w) - 7, 1))}
        if g8 & self.grams:
            return True
        s = set(w)
        if not s:
            return False
        for v in self.sets:
            inter = len(s & v)
            if inter and inter / len(s | v) >= self.j:
                return True
        return False


def clean_json(raw: str):
    m = JSON_RE.search(raw.replace("```json", "").replace("```", "", 1))
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def bleed_ok(text: str, lex) -> bool:
    h, l = find_hits(text, lex)
    return not (len(h) > 2 or l or CJK.search(text) or MIXED_WORD.search(text))


def n_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?…]+", text) if s.strip()])


# ------------------------------------------------------------------ kratki ---

SYS_KRATKI = """Ти пишуваш податоци за КРАТКОСТ И ПОЧИТУВАЊЕ НА ФОРМАТ за Скажна,
македонски јазичен модел. Ова е најслабата точка на моделот: кога корисникот бара
кратко, тој сепак излегува со пасус. Тоа го поправаме сега.

Јазичниот устав важи:
{constitution}

Смисли ЕДНО прашање од типот „{style}" на тема: {topic}.
Прашањето мора да биде такво што ЧОВЕК би одговорил со {limit}.

Потоа напиши ДВА одговора на тоа прашање:
- "short": ТОЧЕН одговор во {limit}. Директен, самоуверен, природен македонски.
  Бројката/името/фактот доаѓа ВЕДНАШ, не по вовед. Без „Секако!", без
  „Одлично прашање", без резиме на крајот. Ако прашањето бара формат
  (на пр. „по една реченица за секој"), ПОЧИТУВАЈ ГО точно.
- "padded": ИСТАТА содржина, но развлечена како што греши слаб модел —
  учтив вовед, контекст што никој не го барал, набројување, заклучок.
  Фактите остануваат исти и точни. 4-8 пати подолго од "short".

ПРАВИЛА:
1. Фактот мора да е ВИСТИНИТ и проверлив. Ако не си сигурен во бројка, избери
   тема каде си сигурен. Никогаш не измислувај специфики.
2. Не пишувај прашање за кое одговорот се менува од година во година.
3. Никогаш не почнувај ниту еден одговор со „Секако", „Се разбира", „Одлично".
Врати строг JSON: {{"question": "...", "short": "...", "padded": "..."}}"""

KRATKI_STYLES = [
    ("фактичко прашање со број", "една реченица со бројката"),
    ("фактичко прашање со име", "една реченица"),
    ("прашање со експлицитен формат (на пр. наброј три, по една реченица за секое)",
     "точно толку реченици колку што бара прашањето"),
    ("прашање од типот што значи Х", "една до две реченици"),
    ("прашање со да/не + краток образложение", "да или не, плус една реченица"),
    ("прашање од типот која е разликата меѓу Х и Y", "две до три реченици"),
    ("прашање што бара само една збор/име", "неколку зборови"),
]

KRATKI_TOPICS = [
    "географија на Македонија", "историја на Македонија", "македонски јазик",
    "македонска книжевност", "македонска музика и филм", "македонска кујна",
    "секојдневен живот во Македонија", "природа и планини", "градови и населби",
    "обичаи и празници", "спорт", "наука", "технологија", "светска историја",
    "здравје и исхрана", "администрација и документи", "сообраќај и патишта",
    "образование", "економија", "растенија и животни",
    # widened: 20 topics x 7 styles capped diversity at ~58% unique questions
    "реки и езера", "клима и време", "археолошки локалитети", "манастири и цркви",
    "народни носии и ракотворби", "занаети", "земјоделство и лозарство",
    "вино и ракија", "пазари и трговија", "банки и пари", "работа и вработување",
    "станови и градење", "мобилни и интернет", "струја и комунални услуги",
    "здравствен систем", "аптеки и лекови", "автобуси и возови", "авиони и патувања",
    "виза и патни документи", "полиција и безбедност", "судство и закони",
    "избори и политички систем", "општини и локална власт", "весници и медиуми",
    "телевизија и радио", "театар и опера", "музеи и галерии", "библиотеки",
    "фудбал и кошарка", "олимписки спортови", "планинарење и природа",
    "лов и риболов", "домашни миленици", "птици и инсекти", "дрвја и шуми",
    "минерали и карпи", "астрономија", "физика во секојдневието", "хемија дома",
    "човечко тело", "исхрана и витамини", "спиење и одмор", "деца и родителство",
    "свадби и крштевки", "погреби и обичаи", "религии во Македонија",
    "јазични недоумици", "правопис и интерпункција", "странски јазици",
    "компјутери и програмирање", "вештачка интелигенција", "автомобили",
    "алати и поправки", "градинарство", "готвење техники", "слатки и колачи",
    "кафе и чај", "мерки и единици", "математика во секојдневието",
]


# ----------------------------------------------------------------- kultura ---

SYS_KULTURA = """Ти пишуваш КУЛТУРНИ одговори за Скажна, македонски јазичен модел.
Ова е категоријата каде моделот губи против конкуренцијата — не поради незнаење,
туку поради здодевна испорака и несигурен регистар.

Јазичниот устав важи:
{constitution}

Подолу е ПРОВЕРЕН ПАСУС за темата „{topic}". Смисли ЕДНО природно прашање што
човек би го поставил за оваа тема, и напиши одговор.

ОДГОВОРОТ мора да:
1. Биде фактички точен и заснован САМО на пасусот за специфики (имиња, години,
   бројки, места). Ако нешто не е во пасусот, не го тврди како факт.
2. Звучи како образован Скопјанец што ја знае темата и има став — не како
   енциклопедија и не како туристички проспект.
3. Почнува од суштината. Без вовед, без „Секако". Без резиме на крајот.
4. Биде 3-8 реченици. Богат, но без полнење.
5. Ако прашањето бара препорака или избор, ЗАЗЕМИ СТАВ и кажи зошто.
6. НИКОГАШ не спомнувај имиња на комерцијални брендови, фирми или производи
   што ги нема во пасусот. Ставот е за суштината, не за маркички.
Врати строг JSON: {{"question": "...", "answer": "..."}}

ПАСУС:
{passage}"""

KULTURA_TOPICS = [
    "Крсте Петков Мисирков", "Гоце Делчев", "Илинден", "Крушевска Република",
    "Галичка свадба", "Блаже Конески", "Кочо Рацин", "Марко Цепенков",
    "Тоше Проески", "Пред дождот", "Медена земја", "ајвар", "тавче гравче",
    "Охридско Езеро", "Кораб", "Охрид", "Битола", "Скопје", "Стоби", "Хераклеа",
    "Самоил", "Охридска архиепископија", "АСНОМ", "кирилица", "Вардар",
    "Даме Груев", "Јане Сандански", "ВМРО", "Битолски конгрес", "сказна",
    "македонски народни песни", "оро", "Крушево", "Струга", "Преспанско Езеро",
    "Мариово", "Пелистер", "Маврово", "Шар Планина", "Дојранско Езеро",
    "Свети Наум", "Климент Охридски", "Кирил и Методиј", "гајда", "зурла",
    "Ванчо Прке", "Никола Карев", "Питу Гули", "македонска везба", "филигран",
]


# -------------------------------------------------------------------- math ---

def gen_math_problem(rng):
    """Template-generated so the answer is provably correct. Gemini writes prose
    only — it is never trusted with the arithmetic."""
    kind = rng.choice(["pct", "prop", "speed", "area", "split", "discount",
                       "avg", "interest", "mix", "time"])
    if kind == "pct":
        base = rng.choice([120, 240, 360, 450, 800, 1200, 2500, 3600]) + 10 * rng.randint(0, 24)
        pct = rng.choice([5, 8, 12, 15, 20, 25, 35, 40, 45, 60, 75])
        ans = base * pct / 100
        setup = f"Колку е {pct}% од {base}?"
        facts = f"{pct}% од {base} = {base} × {pct}/100 = {_num(ans)}"
    elif kind == "prop":
        a, b = rng.randint(2, 14), rng.randint(3, 25)
        tot = a * rng.randint(4, 40)
        ans = Fraction(tot * b, a)
        setup = (f"Ако {a} работници завршуваат работа за {b} дена, "
                 f"колку дена им требаат на {a} работници за {tot // a} такви работи?")
        facts = f"{tot // a} × {b} = {_num(ans)}"
    elif kind == "speed":
        t = rng.choice([2, 3, 4, 5, 6, 8])
        d = t * rng.randint(15, 95)
        ans = Fraction(d, t)
        setup = f"Автомобил поминува {d} км за {t} часа. Колкава е просечната брзина?"
        facts = f"{d} ÷ {t} = {_num(ans)} км/ч"
    elif kind == "area":
        a, b = rng.randint(4, 60), rng.randint(3, 45)
        ans = a * b
        setup = f"Нива е долга {a} метри и широка {b} метри. Колкава ѝ е плоштината?"
        facts = f"{a} × {b} = {_num(ans)} м²"
    elif kind == "split":
        n = rng.choice([3, 4, 5, 6, 8, 12])
        tot = n * rng.randint(40, 900)
        ans = Fraction(tot, n)
        setup = f"{tot} денари се делат на {n} лица подеднакво. Колку добива секој?"
        facts = f"{tot} ÷ {n} = {_num(ans)} денари"
    elif kind == "discount":
        p = 50 * rng.randint(6, 240)
        d = rng.choice([10, 15, 20, 25, 30, 40, 45, 55, 70])
        ans = p * (100 - d) / 100
        setup = f"Производ чини {p} денари и е намален за {d}%. Колку чини сега?"
        facts = f"{p} − {p}×{d}/100 = {p} − {_num(p * d / 100)} = {_num(ans)} денари"
    elif kind == "avg":
        vals = [rng.randint(2, 120) for _ in range(rng.choice([3, 4, 5, 6, 7]))]
        ans = Fraction(sum(vals), len(vals))
        setup = f"Колкав е просекот на броевите {', '.join(map(str, vals))}?"
        facts = f"({' + '.join(map(str, vals))}) ÷ {len(vals)} = {sum(vals)} ÷ {len(vals)} = {_num(ans)}"
    elif kind == "interest":
        p = 1000 * rng.randint(5, 150)
        r = rng.choice([2, 3, 4, 5, 6, 7, 8])
        y = rng.choice([2, 3, 4, 5, 6, 10])
        ans = p * r * y / 100
        setup = (f"Влог од {p} денари носи проста камата од {r}% годишно. "
                 f"Колкава камата се добива за {y} години?")
        facts = f"{p} × {r}/100 × {y} = {_num(ans)} денари"
    elif kind == "mix":
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        pa, pb = 10 * rng.randint(3, 40), 10 * rng.randint(4, 55)
        ans = a * pa + b * pb
        setup = (f"Купуваш {a} килограми по {pa} денари и {b} килограми по {pb} денари. "
                 f"Колку плаќаш вкупно?")
        facts = f"{a}×{pa} + {b}×{pb} = {a * pa} + {b * pb} = {_num(ans)} денари"
    else:  # time
        h, m = rng.randint(1, 11), rng.randint(1, 59)
        add = rng.randint(15, 260)
        total = h * 60 + m + add
        ans = f"{total // 60}:{total % 60:02d}"
        setup = (f"Патувањето почнува во {h}:{m:02d} и трае {add} минути. "
                 f"Во колку часот завршува?")
        facts = f"{h}:{m:02d} + {add} мин = {ans}"
    return setup, str(_num(ans)), facts


def _num(x):
    if isinstance(x, Fraction):
        return x.numerator // x.denominator if x.denominator == 1 else f"{float(x):.2f}".rstrip("0").rstrip(".")
    if isinstance(x, float):
        return int(x) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")
    return x


SYS_MATH = """Ти пишуваш ЗАДАЧИ СО ПОСТАПКА за Скажна, македонски јазичен модел.
Јазичниот устав важи:
{constitution}

Подолу е задача, ТОЧНИОТ одговор и пресметката. Твојата работа е САМО да ја
напишеш постапката на природен македонски — не смееш да ја менуваш ниту задачата
ниту одговорот.

Напиши решение што:
1. Оди чекор по чекор, кратко и јасно. Секој чекор е една реченица.
2. Ја покажува пресметката (бројките, не само зборови).
3. Завршува со јасен конечен одговор во кој ГО СОДРЖИ точниот резултат.
4. Е во 3-6 реченици. Без вовед, без „Секако".
5. Користи запирка за децимали (2,5 а не 2.5) — освен во конечниот резултат,
   каде запиши го БУКВАЛНО како што е даден погоре.
Врати строг JSON: {{"solution": "..."}}

ЗАДАЧА: {problem}
ТОЧЕН ОДГОВОР: {answer}
ПРЕСМЕТКА: {facts}"""


# ------------------------------------------------------------------ scimcq ---

SYS_SCIMCQ = """Ти генерираш прашања за ПРИРОДНИ НАУКИ за Скажна, македонски модел.
Јазичниот устав важи:
{constitution}

Смисли ЕДНО оригинално прашање по природни науки од областа: {area}, на ниво
основно/средно училиште. КЛУЧНО: прашањето мора да бара РАСУДУВАЊЕ за некој
научен факт (зошто се случува нешто, што ќе се случи ако, која постапка е
најдобра, кој доказ го поддржува тоа) — НЕ гола дефиниција и НЕ трик-прашање.

ПРАВИЛА:
1. Оригинално прашање, формулирано од тебе сега. Никогаш препишано од тест.
2. Точно 4 опции, точно една целосно точна. Дистракторите се разумни и слични
   по должина — не смешни.
3. Фактот мора да е ВИСТИНИТ. Без измислени специфики.
4. Природен македонски, без калки од англиски.
Врати строг JSON: {{"question": "...", "choices": ["...","...","...","..."],
"answer_idx": 0-3, "why": "една реченица зошто точниот е точен"}}"""

SCI_AREAS = [
    "биологија — растенија и фотосинтеза", "биологија — животни и адаптации",
    "биологија — човечко тело", "биологија — екосистеми и синџири на исхрана",
    "физика — сила и движење", "физика — енергија и топлина",
    "физика — светлина и звук", "физика — електрицитет и магнетизам",
    "хемија — состојби на материјата", "хемија — мешавини и раствори",
    "хемија — хемиски реакции", "науки за Земјата — временски услови и клима",
    "науки за Земјата — карпи и почва", "науки за Земјата — вода и кружење",
    "астрономија — Сончев систем", "астрономија — Земја, Месечина, Сонце",
    "научен метод и експерименти", "мерење и научни инструменти",
    "заштита на животната средина", "материјали и нивни својства",
]


# -------------------------------------------------------------------- main ---

def find_passages(topics):
    best = {t: None for t in topics}
    with gzip.open("data/processed/mk_corpus_filtered.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "mkwiki":
                continue
            text = r["text"]
            first = text[:120].lower()
            for t in topics:
                if t.lower() in first and len(text) > 600:
                    if best[t] is None or len(text) > len(best[t]):
                        best[t] = text[:2500]
    return {t: p for t, p in best.items() if p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["kratki", "kultura", "math", "scimcq"])
    ap.add_argument("--sample", type=int)
    ap.add_argument("--target", type=int)
    args = ap.parse_args()
    n = args.sample or args.target
    tag = f"sample_{n}" if args.sample else "bulk"

    global LEDGER, _usage
    LEDGER = Path(f"data/synth/.gemini_usage_targeted_{args.kind}.json")
    if LEDGER.exists():
        _usage = json.load(LEDGER.open())

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    lex = load_approved()
    constitution = Path("docs/constitution.md").read_text(encoding="utf-8")
    rng = random.Random(SEED if args.sample else SEED + 1)
    guard = VibesGuard()
    out_path = Path(f"data/synth/v6_{args.kind}_{tag}.jsonl")
    done = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0
    f = out_path.open("a", encoding="utf-8")
    stats = {"ok": done, "rej_parse": 0, "rej_bleed": 0, "rej_decon": 0,
             "rej_vibes": 0, "rej_len": 0}

    bench = load_bench_grams() if args.kind == "scimcq" else set()

    def emit(rec):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        stats["ok"] += 1
        if stats["ok"] % 50 == 0:
            print(f"[{stats['ok']}/{n}] {stats} {tokens_used():,}tok", flush=True)

    if args.kind == "kratki":
        while stats["ok"] < n:
            style, limit = rng.choice(KRATKI_STYLES)
            d = clean_json(gemini(
                SYS_KRATKI.format(constitution=constitution, style=style,
                                  limit=limit, topic=rng.choice(KRATKI_TOPICS)),
                "Генерирај. Само JSON.", max_tokens=2048))
            if not d or not all(d.get(k) for k in ("question", "short", "padded")):
                stats["rej_parse"] += 1
                continue
            q, s, p = d["question"], d["short"], d["padded"]
            if guard.hits(q):
                stats["rej_vibes"] += 1
                continue
            # the whole point: short must actually be short, padded actually padded
            if len(s) > 400 or n_sentences(s) > 4 or len(p) < 2.0 * len(s):
                stats["rej_len"] += 1
                continue
            if not (bleed_ok(q + " " + s, lex) and bleed_ok(p, lex)):
                stats["rej_bleed"] += 1
                continue
            emit({"id": f"v6k-{stats['ok']:05d}", "question": q,
                  "short": s, "padded": p, "style": style})

    elif args.kind == "kultura":
        # TITLE-KEYED passages, not corpus substring matching. Five separate
        # matcher designs (loose, strict-title, RAG, opener-anchored) all pulled
        # namesakes — a school named Јане Сандански, a scientist born in Битола,
        # "Мајст-ОРО-т и Маргарита" for оро. Keying on the exact Wikipedia title
        # makes the failure impossible instead of merely unlikely.
        passages = json.load(open("data/synth/kultura_passages.json",
                                  encoding="utf-8"))
        # hand-verified canon passages win where they exist
        for t, p in json.load(open("data/synth/canon_curated.json",
                                   encoding="utf-8")).items():
            passages[t] = p
        print(f"passages: {len(passages)} (title-keyed + curated)", flush=True)
        items = sorted(passages.items())
        reps = max(1, n // max(len(items), 1) + 1)
        for rep in range(reps):
            for topic, passage in items:
                if stats["ok"] >= n:
                    break
                d = clean_json(gemini(
                    SYS_KULTURA.format(constitution=constitution, topic=topic,
                                       passage=passage),
                    "Генерирај. Само JSON.", max_tokens=2560))
                if not d or not d.get("question") or not d.get("answer"):
                    stats["rej_parse"] += 1
                    continue
                q, a = d["question"], d["answer"]
                if guard.hits(q):
                    stats["rej_vibes"] += 1
                    continue
                if not (2 <= n_sentences(a) <= 12):
                    stats["rej_len"] += 1
                    continue
                if not bleed_ok(q + " " + a, lex):
                    stats["rej_bleed"] += 1
                    continue
                emit({"id": f"v6c-{stats['ok']:05d}", "topic": topic,
                      "question": q, "answer": a})

    elif args.kind == "math":
        while stats["ok"] < n:
            problem, answer, facts = gen_math_problem(rng)
            if guard.hits(problem):
                stats["rej_vibes"] += 1
                continue
            d = clean_json(gemini(
                SYS_MATH.format(constitution=constitution, problem=problem,
                                answer=answer, facts=facts),
                "Реши. Само JSON.", max_tokens=1536, temperature=0.6))
            sol = (d or {}).get("solution", "")
            if not sol:
                stats["rej_parse"] += 1
                continue
            # provable check: the known-correct answer must appear in the prose
            if str(answer) not in sol.replace(",", "."):
                stats["rej_decon"] += 1
                continue
            if not bleed_ok(sol, lex):
                stats["rej_bleed"] += 1
                continue
            emit({"id": f"v6m-{stats['ok']:05d}", "problem": problem,
                  "answer": answer, "solution": sol})

    else:  # scimcq
        while stats["ok"] < n:
            d = clean_json(gemini(
                SYS_SCIMCQ.format(constitution=constitution,
                                  area=rng.choice(SCI_AREAS)),
                "Генерирај. Само JSON.", max_tokens=2048))
            if not d or len(d.get("choices", [])) != 4 or \
                    not isinstance(d.get("answer_idx"), int) or not 0 <= d["answer_idx"] <= 3:
                stats["rej_parse"] += 1
                continue
            blob = d["question"] + " " + " ".join(d["choices"])
            w = _toks(blob)
            g6 = {" ".join(w[i:i + 6]) for i in range(max(len(w) - 5, 1))}
            if g6 & bench:
                stats["rej_decon"] += 1
                continue
            if guard.hits(d["question"]):
                stats["rej_vibes"] += 1
                continue
            if not bleed_ok(blob, lex):
                stats["rej_bleed"] += 1
                continue
            emit({"id": f"v6s-{stats['ok']:05d}", "question": d["question"],
                  "choices": d["choices"], "answer_idx": d["answer_idx"],
                  "why": d.get("why", "")})

    f.close()
    print(f"DONE {stats} | {tokens_used():,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
