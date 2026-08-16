#!/usr/bin/env python3
"""Depth transplant: rewrite deep LVSTCK sft-mk answers into native Macedonian.

Motivation (early arena debrief): our answers were correct but shallow
(median 165 chars vs yak's 594). yak's depth comes from sft-mk content our
ranker rejected as translationese. This keeps the CONTENT (depth, structure,
facts) and replaces the STYLE (native MK per docs/constitution.md).

Two modes:
  --sample N   generate N calibration examples for user review (markdown+jsonl)
  --bulk       full run (only after the user approves the sample; off-peak)

Selection: single-exchange conversations, assistant answer 600-2500 chars,
Cyrillic-dominant, no AI-self-reference boilerplate, bleed-screened.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.filters.bleed_blocklist import find_hits, load_approved  # noqa: E402
from data.synth.teacher_api import Usage, api_key, chat, is_peak_hour  # noqa: E402

SRC = Path("data/downloads/sft_mk/sft_dataset_train.jsonl")
MODEL = "deepseek-v4-pro"  # calibration wants the strong tier; bulk may mix
SEED = 1903

# answers that are mostly the assistant talking about being an AI carry no depth
AI_BOILERPLATE = re.compile(r"вештачка интелигенција|јазичен модел|AI модел", re.IGNORECASE)
CYR = re.compile(r"[а-шѓќѕџљњј]", re.IGNORECASE)
# MT translated inside code blocks (низа( = array(, печати( = print(...) — code
# items are unsalvageable and not our arena deficit; drop them wholesale
CODE = re.compile(r"```|def |import |низа\(|печати\(|\bfor\b.*\bin\b|SELECT |<html", re.IGNORECASE)

SYSTEM = """Ти си врвен македонски писател и лектор — најдобриот жив уредник
за роден македонски стил. Работиш строго по оваа јазична конституција:

{constitution}

=== ЗЛАТЕН СТАНДАРД ===
Вака звучи роден македонски одговор (пишувано од човек, роден говорител —
ова е целниот регистар и тон):

{gold_examples}

=== ЗАДАЧА ===
Одговорот што ќе го добиеш е содржински вреден, но стилски е ПРЕВОД од
англиски — синтаксата, редот на зборовите и фразите му се туѓи. Твојата
работа е да го препишеш како да го напишал роден говорител од нула.

=== ЖЕЛЕЗНИ ПРАВИЛА ===
1. СОДРЖИНАТА Е СВЕТА: секој факт, број, аргумент, пример и чекор останува.
   Листите остануваат листи, табелите табели, редоследот ист.
2. ДОЛЖИНАТА СЕ ЧУВА: препишаното мора да задржи барем 85% од должината.
   Смееш да сечеш САМО чисто полнење (AI-фрази, празни уводи). Ако сечеш
   реченица, прашај се: „носи ли факт?" — ако да, останува, преформулирана.
3. ПРЕВОДНИ БЕЛЕЗИ — активно барај и поправај: калкирани конструкции,
   пасив каде што македонскиот бара актив, „се работи за", туѓ збороред,
   непотребни заменки, англициски врзувачки фрази („сепак, важно е да се
   напомене дека…").
4. Сите ◆-означени правила од конституцијата се МОМЕНТАЛЕН ПАД — провери
   го текстот наспроти секое пред да одговориш.
5. AI-фрази („како вештачка интелигенција/јазичен модел…") се бришат без
   замена; одговорот почнува директно со суштината.
6. Никакви несоодветни кирилични букви (ђћъщюяйьэыёіїєґў), никаков
   латинично-кириличен микс во ист збор.
7. Врати САМО препишаниот одговор — без коментари, без објаснувања."""


def load_candidates() -> list:
    lex = load_approved()
    out = []
    for i, line in enumerate(SRC.open(encoding="utf-8")):
        conv = json.loads(line)["conversations"]
        if len(conv) != 2 or conv[0]["role"] != "user":
            continue
        q, a = conv[0]["content"], conv[1]["content"]
        if not (600 <= len(a) <= 2500) or AI_BOILERPLATE.search(a):
            continue
        cyr_ratio = len(CYR.findall(a)) / max(len(a), 1)
        if cyr_ratio < 0.55:
            continue
        s, l = find_hits(a, lex)
        if len(s) > 2 or l:
            continue
        out.append({"id": f"dt{i:06d}", "prompt": q, "original": a})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--bulk", action="store_true")
    ap.add_argument("--allow-peak", action="store_true")
    ap.add_argument("--out-dir", default="data/synth")
    args = ap.parse_args()

    if args.bulk and is_peak_hour() and not args.allow_peak:
        raise SystemExit("peak hours — bulk runs are scheduled off-peak only")

    constitution = Path("docs/constitution.md").read_text(encoding="utf-8")
    # gold style anchors: the curated anchor set (600-1200 chars,
    # deep enough to demonstrate register without bloating the cached prefix)
    hw = [json.loads(l)["conversations"]
          for l in Path("data/handwritten/handwritten.jsonl").open(encoding="utf-8")]
    anchors = [c for c in hw
               if len(c) == 2 and 600 <= len(c[1]["content"]) <= 1200][:3]
    gold = "\n\n".join(
        f"ПРАШАЊЕ: {c[0]['content']}\nОДГОВОР: {c[1]['content']}" for c in anchors)
    system = SYSTEM.format(constitution=constitution, gold_examples=gold)
    key = api_key()
    usage = Usage()
    lex = load_approved()

    cands = load_candidates()
    print(f"{len(cands)} candidates pass selection", flush=True)
    rng = random.Random(SEED)
    rng.shuffle(cands)
    n = args.sample if args.sample else len(cands)
    picked = cands[:n]

    out_dir = Path(args.out_dir)
    tag = f"sample_{n}" if args.sample else "bulk"
    jl = (out_dir / f"depth_transplant_{tag}.jsonl").open("w", encoding="utf-8")
    md_path = out_dir / f"depth_transplant_{tag}.md"
    md = md_path.open("w", encoding="utf-8")
    md.write("# Depth transplant — calibration sample\n\n"
             "Оцени: дали препишаното звучи родно И ја задржува длабочината?\n\n")

    for k, item in enumerate(picked, 1):
        base_msgs = [{"role": "system", "content": system},
                     {"role": "user", "content": item["prompt"] + "\n\n---ОДГОВОР ЗА ПРЕПИШУВАЊЕ---\n" + item["original"]}]
        # pro burns a lot of hidden reasoning on this long system prompt;
        # 10240 gives headroom (learned: 6144 starved it to empty output)
        try:
            rewritten = chat(key, MODEL, base_msgs, usage, temperature=0.6, max_tokens=10240)
        except ValueError as e:
            print(f"[{k}/{n}] {item['id']} SKIPPED: {e}", flush=True)
            continue
        s, l = find_hits(rewritten, lex)
        # one corrective retry on over-compression or bleed — feedback included
        problems = []
        if len(rewritten) < 0.75 * len(item["original"]):
            problems.append(f"Премногу скрати ({len(item['original'])}→{len(rewritten)} знаци). "
                            "Задржи ја ЦЕЛАТА содржина — правило 2.")
        if len(s) > 2 or l:
            problems.append(f"Несоодветни букви/зборови: {s[:3] + l[:3]}. Правило 6.")
        if problems:
            retry_msgs = base_msgs + [
                {"role": "assistant", "content": rewritten},
                {"role": "user", "content": "Поправи: " + " ".join(problems)
                                            + " Врати ја целата поправена верзија."}]
            try:
                rewritten = chat(key, MODEL, retry_msgs, usage, temperature=0.6, max_tokens=10240)
            except ValueError:
                pass  # keep the first attempt; it stays flagged below
            s, l = find_hits(rewritten, lex)
        flag = f" ⚠️ bleed: {s[:3]}{l[:3]}" if (len(s) > 2 or l) else ""
        if len(rewritten) < 0.75 * len(item["original"]):
            flag += " ⚠️ still-short"
        rec = {**item, "rewritten": rewritten,
               "len_orig": len(item["original"]), "len_new": len(rewritten)}
        jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        md.write(f"## {k}. [{item['id']}] ({rec['len_orig']}→{rec['len_new']} chars){flag}\n\n"
                 f"**Прашање:** {item['prompt']}\n\n"
                 f"<details><summary>Оригинал (sft-mk)</summary>\n\n{item['original']}\n\n</details>\n\n"
                 f"**Препишано:**\n\n{rewritten}\n\n---\n\n")
        print(f"[{k}/{n}] {item['id']} {rec['len_orig']}→{rec['len_new']}{flag}", flush=True)

    jl.close(); md.close()
    print(usage.report())
    print(f"review -> {md_path}")


if __name__ == "__main__":
    main()
