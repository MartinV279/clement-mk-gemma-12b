#!/usr/bin/env python3
"""Factual-recall probe: does CPT actually add Macedonian KNOWLEDGE?

Perplexity measures fluency (common tokens) and saturates early; facts live in
rare tokens. This probe scores knowledge directly: for each item, the model's
likelihood of the CORRECT completion is compared against plausible distractors
(multiple-choice by loglikelihood, no generation, no judging needed).

Items are stable, widely-known Macedonian facts — geography, institutions,
history, culture, procedures. NOT drawn from the frozen vibes set.

Usage (pod):
  python eval/mk_fact_probe.py --model gemma4-12b-text
  python eval/mk_fact_probe.py --model checkpoints/cpt/checkpoint-5100 --peft
"""

import argparse
import json
import os
from pathlib import Path

# (prompt, correct, [distractors]) — completions are single words/short phrases
ITEMS = [
    ("Највисокиот врв во Северна Македонија е", "Кораб", ["Пелистер", "Титов Врв", "Солунска Глава"]),
    ("Главен град на Северна Македонија е", "Скопје", ["Битола", "Охрид", "Куманово"]),
    ("Илинденското востание се случило во", "1903", ["1878", "1913", "1945"]),
    ("Крушевската Република била прогласена во", "1903", ["1918", "1944", "1991"]),
    ("Најголемото природно езеро во земјата е", "Охридското", ["Дојранското", "Мавровското", "Тиквешкото"]),
    ("Реката што минува низ Скопје се вика", "Вардар", ["Црна Река", "Треска", "Брегалница"]),
    ("Лична карта во Северна Македонија се вади во", "МВР", ["УЈП", "ЕВН", "Катастар"]),
    ("Данокот на личен доход го администрира", "УЈП", ["МВР", "НБРМ", "Царина"]),
    ("Електричната енергија во домовите ја дистрибуира", "ЕВН", ["МЕПСО", "ЕЛЕМ", "Телеком"]),
    ("Имотните листови се издаваат во", "Катастар", ["МВР", "Општина", "УЈП"]),
    ("Македонскиот јазик го стандардизирал во своите дела", "Мисирков", ["Конески", "Прличев", "Цепенков"]),
    ("Автор на „За македонцките работи“ е Крсте Петков", "Мисирков", ["Конески", "Рацин", "Шапкарев"]),
    ("Поетот што ја напишал „Бели мугри“ е Кочо", "Рацин", ["Конески", "Јаневски", "Шопов"]),
    ("Пејачот роден во Крушево, трагично загинат во 2007, е Тоше", "Проески", ["Марковски", "Ристески", "Николовски"]),
    ("Галичката свадба традиционално се одржува на", "Петровден", ["Илинден", "Велигден", "Богојавление"]),
    ("Националната валута на Северна Македонија е", "денар", ["евро", "лев", "динар"]),
    ("Централната банка на земјата е", "НБРМ", ["ЕВН", "УЈП", "МЕПСО"]),
    ("Најголемиот град по Скопје по население е", "Битола", ["Охрид", "Струмица", "Прилеп"]),
    ("Манастирот Свети Јован Бигорски се наоѓа близу", "Дебар", ["Охрид", "Струга", "Кичево"]),
    ("Тавче гравче се подготвува од", "грав", ["леќа", "ориз", "компири"]),
    ("Ајварот главно се прави од", "пиперки", ["домати", "краставици", "патлиџани"]),
    ("Универзитетот „Св. Кирил и Методиј“ се наоѓа во", "Скопје", ["Битола", "Штип", "Тетово"]),
    ("Мечкин Камен е спомен-обележје кај", "Крушево", ["Прилеп", "Кичево", "Гостивар"]),
    ("Стариот скопски базен и Камен мост се врзани за реката", "Вардар", ["Лепенец", "Треска", "Пчиња"]),
    ("Зимските гуми во земјата се задолжителни од средината на", "ноември", ["октомври", "декември", "јануари"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--peft", action="store_true")
    ap.add_argument("--base", default="gemma4-12b-text")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, token=os.environ.get("HF_TOKEN"))
    if args.peft:
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                    device_map="auto")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map="auto")
    model.eval()

    def score(prompt: str, completion: str) -> float:
        """Length-normalised loglikelihood of the completion given the prompt."""
        p_ids = tok(prompt, return_tensors="pt").input_ids
        full = tok(prompt + " " + completion, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            logits = model(full).logits[0, :-1].float().log_softmax(-1)
        tgt = full[0, 1:]
        start = p_ids.shape[1] - 1
        lp = logits[range(start, tgt.shape[0]), tgt[start:]].sum().item()
        return lp / max(tgt.shape[0] - start, 1)

    correct = 0
    details = []
    for prompt, gold, distractors in ITEMS:
        options = [gold] + distractors
        scores = {o: score(prompt, o) for o in options}
        pick = max(scores, key=scores.get)
        ok = pick == gold
        correct += ok
        details.append({"prompt": prompt, "gold": gold, "picked": pick, "ok": ok})
        print(f"{'OK ' if ok else 'MISS'} {prompt[:50]:52s} -> {pick} (gold {gold})", flush=True)

    acc = correct / len(ITEMS)
    print(f"\nMODEL: {args.model}")
    print(f"FACT RECALL: {correct}/{len(ITEMS)} = {acc:.1%}  (random baseline 25%)")
    out = Path("eval/fact_probe_results.jsonl")
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"model": args.model, "correct": correct,
                            "total": len(ITEMS), "acc": acc, "details": details},
                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
