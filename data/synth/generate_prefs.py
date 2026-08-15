#!/usr/bin/env python3
"""Preference pairs for ORPO (Phase 4). CLAUDE.md recipe:
rejected = round-trip MT (mk->en->mk) of chosen · bloated-length variants ·
Serbian/Bulgarian bleed variants.

Chosen answers come from the user-validated pools (top-scored synth +
grounded). Bleed variants are constructed mechanically (blocklist inverse
substitutions) with no API call at all. Round-trip and bloat use Flash off-peak.

The playbook's LLM-judge step is replaced by construction guarantees +
mechanical validation + the user's 200-pair spot check (prefs_spotcheck_200.md)
— the rejected variant is worse BY CONSTRUCTION, so judging adds little.

Output:
  data/processed/prefs_train.jsonl  ({prompt, chosen, rejected, kind})
  data/processed/prefs_holdout.jsonl (500 pairs, FROZEN regression set)
  data/processed/prefs_spotcheck_200.md (user)

Usage:  uv run python data/synth/generate_prefs.py --max-tokens 3000000
"""

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.synth.teacher_api import Usage, api_key, chat, is_peak_hour  # noqa: E402

MODEL = "deepseek-v4-flash"
SEED = 20260727
TARGETS = {"roundtrip": 3300, "bloat": 2200, "bleed": 1500}
HOLDOUT = 500

# inverse bleed: standard MK -> Serbian/Bulgarian intrusions (constitution §3)
BLEED_SUBS = [("исто така", "такође"), ("неколку", "неколико"), ("многу", "врло"),
              ("фала", "хвала"), ("точно", "тачно"), ("нешто", "нещо"),
              ("ништо", "нищо"), ("сакам", "искам"), ("се чувствувам", "се осеќам")]
# ("навистина","стварно") removed 2026-07-27: user ruled стварно acceptable
# colloquial MK, so that substitution does not produce a worse answer

BLOAT_PROMPT = """Препиши го одговорот подолу ЛОШО: додај празен вовед („Секако! Одлично прашање!"),
непотребни повторувања, генерички фрази, и заклучок „Се надевам дека ова помага!".
Содржината остави ја иста, само надуј ја двојно. Врати САМО препишаниот текст.

ОДГОВОР:
{answer}"""


def load_chosen(rng):
    pools = []
    for path, min_score in (("data/processed/synth_sft_ranked.jsonl", 0.02),
                            ("data/processed/synth_grounded_clean.jsonl", None)):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if min_score is not None and r.get("quality_score", 0) < min_score:
                continue
            conv = r["conversations"]
            # last user->assistant exchange as the pair context
            for i in range(len(conv) - 1):
                if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant" \
                        and len(conv[i + 1]["content"]) > 150:
                    pools.append({"prompt": conv[i]["content"],
                                  "chosen": conv[i + 1]["content"], "src": r["id"]})
                    break
    rng.shuffle(pools)
    print(f"{len(pools)} candidate chosen answers")
    return pools


def make_bleed(chosen: str, rng) -> str | None:
    subs = [s for s in BLEED_SUBS if s[0] in chosen.lower()]
    if not subs:
        return None
    out = chosen
    for a, b in rng.sample(subs, min(len(subs), 3)):
        out = re.sub(re.escape(a), b, out, count=1, flags=re.IGNORECASE)
    return out if out != chosen else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=3_000_000,
                    help="stop once this many teacher tokens have been consumed")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--allow-peak", action="store_true")
    args = ap.parse_args()
    if is_peak_hour() and not args.allow_peak:
        raise SystemExit("Peak hours — rerun off-peak (see teacher_api.is_peak_hour).")

    key = api_key()
    rng = random.Random(SEED)
    pool = load_chosen(rng)

    out_path = Path("data/processed/prefs_raw.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["src"] for l in out_path.open(encoding="utf-8")}
    f = out_path.open("a", encoding="utf-8")
    usage = Usage()
    lock = threading.Lock()
    stop = threading.Event()
    counters = {"ok": 0, "fail": 0}

    # assign kinds to distinct chosen answers
    tasks = []
    i = 0
    for kind, n in TARGETS.items():
        for _ in range(n):
            if i >= len(pool):
                break
            t = dict(pool[i])
            t["kind"] = kind
            if t["src"] not in done:
                tasks.append(t)
            i += 1
    print(f"{len(tasks)} pairs to build ({len(done)} done)")

    def build(t):
        if stop.is_set():
            return
        try:
            if t["kind"] == "bleed":
                rej = make_bleed(t["chosen"], random.Random(t["src"]))
                if rej is None:
                    with lock:
                        counters["fail"] += 1
                    return
            elif t["kind"] == "bloat":
                rej = chat(key, MODEL, [{"role": "user",
                                         "content": BLOAT_PROMPT.format(answer=t["chosen"][:3000])}],
                           usage, temperature=0.7, max_tokens=2048)
            else:  # roundtrip mk->en->mk
                en = chat(key, MODEL, [{"role": "user", "content":
                    "Translate to English. Return ONLY the translation.\n\n" + t["chosen"][:3000]}],
                    usage, temperature=0.2, max_tokens=2048)
                rej = chat(key, MODEL, [{"role": "user", "content":
                    "Преведи на македонски. Врати САМО превод.\n\n" + en}],
                    usage, temperature=0.2, max_tokens=2048)
            rej = rej.strip()
            if not rej or rej == t["chosen"] or len(rej) < 80:
                with lock:
                    counters["fail"] += 1
                return
        except Exception:
            with lock:
                counters["fail"] += 1
            return
        with lock:
            f.write(json.dumps({"prompt": t["prompt"], "chosen": t["chosen"],
                                "rejected": rej, "kind": t["kind"], "src": t["src"]},
                               ensure_ascii=False) + "\n")
            f.flush()
            counters["ok"] += 1
            if counters["ok"] % 250 == 0:
                print(f"  {counters['ok']} | {usage.total():,} tok", flush=True)
            if usage.total() >= args.max_tokens or (is_peak_hour() and not args.allow_peak):
                stop.set()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(build, tasks))
    f.close()
    print(f"built: {counters['ok']} ok, {counters['fail']} fail | {usage.report()}")

    # split: frozen holdout + train + spot-check file
    pairs = [json.loads(l) for l in out_path.open(encoding="utf-8")]
    rng.shuffle(pairs)
    hold, train = pairs[:HOLDOUT], pairs[HOLDOUT:]
    with open("data/processed/prefs_holdout.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in hold)
    with open("data/processed/prefs_train.jsonl", "w", encoding="utf-8") as ft:
        ft.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in train)
    with open("data/processed/prefs_spotcheck_200.md", "w", encoding="utf-8") as fs:
        fs.write("# 200 preference pairs — spot check\n# CHOSEN must be better than "
                 "REJECTED. Note src ids where it is NOT.\n\n")
        for p in rng.sample(train, min(200, len(train))):
            fs.write(f"---\n\n## {p['src']} ({p['kind']})\n\n**Прашање:** {p['prompt'][:400]}\n\n"
                     f"**CHOSEN:** {p['chosen'][:800]}\n\n**REJECTED:** {p['rejected'][:800]}\n\n")
    print(f"train {len(train)} | holdout {len(hold)} (FROZEN) | spot-check written")


if __name__ == "__main__":
    main()
