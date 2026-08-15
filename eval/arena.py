#!/usr/bin/env python3
"""Blind A/B arena — the user's scoring instrument (CLAUDE.md division of
labor: only the user judges Macedonian quality; this tool just hides labels).

Loads two or more vibes-generation files (eval/generations/*.jsonl), pairs
answers by prompt id, and walks the user through shuffled, label-hidden
head-to-head comparisons in the terminal. Votes are saved after every choice
(resume-safe); model identities are revealed only in the final table.

Usage:
  uv run python eval/arena.py --files eval/generations/A.jsonl eval/generations/B.jsonl
  uv run python eval/arena.py --files ... --pairs-per-prompt all   # >2 models: all pairs

Keys: 1 = left better · 2 = right better · 0 = tie · s = skip · q = quit (resume later)
"""

import argparse
import itertools
import json
import random
from pathlib import Path

VOTES_PATH = Path("eval/arena_votes.jsonl")
SEED = 1903


def load_generations(paths: list) -> dict:
    """{model: {prompt_id: answer}}"""
    by_model = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_model.setdefault(r["model"], {})[r["id"]] = r
    return by_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--votes", default=str(VOTES_PATH))
    args = ap.parse_args()

    by_model = load_generations(args.files)
    models = sorted(by_model)
    if len(models) < 2:
        raise SystemExit(f"need >=2 models, got {models}")

    matchups = []
    for a, b in itertools.combinations(models, 2):
        shared = sorted(set(by_model[a]) & set(by_model[b]))
        for pid in shared:
            matchups.append((pid, a, b))
    rng = random.Random(SEED)
    rng.shuffle(matchups)

    votes_path = Path(args.votes)
    done = set()
    if votes_path.exists():
        for line in votes_path.read_text(encoding="utf-8").splitlines():
            v = json.loads(line)
            done.add((v["prompt_id"], v["model_a"], v["model_b"]))
    todo = [m for m in matchups if (m[0], m[1], m[2]) not in done]
    print(f"{len(matchups)} matchups total, {len(done)} voted, {len(todo)} to go\n")

    with votes_path.open("a", encoding="utf-8") as vf:
        for n, (pid, a, b) in enumerate(todo, 1):
            ra, rb = by_model[a][pid], by_model[b][pid]
            left_is_a = rng.random() < 0.5
            left, right = (ra, rb) if left_is_a else (rb, ra)
            print("=" * 72)
            print(f"[{n}/{len(todo)}] ПРАШАЊЕ: {ra['prompt']}\n")
            print("--- ЛЕВО ---")
            print(left["answer"][:2500], "\n")
            print("--- ДЕСНО ---")
            print(right["answer"][:2500], "\n")
            while True:
                choice = input("1=лево  2=десно  0=нерешено  s=прескокни  q=крај > ").strip().lower()
                if choice in ("1", "2", "0", "s", "q"):
                    break
            if choice == "q":
                print("Прекинато — продолжи подоцна, гласовите се зачувани.")
                break
            if choice == "s":
                continue
            winner = ("tie" if choice == "0"
                      else (a if left_is_a else b) if choice == "1"
                      else (b if left_is_a else a))
            vf.write(json.dumps({"prompt_id": pid, "model_a": a, "model_b": b,
                                 "winner": winner}, ensure_ascii=False) + "\n")
            vf.flush()

    # results table (only over voted pairs)
    wins, ties, games = {m: 0 for m in models}, {m: 0 for m in models}, {m: 0 for m in models}
    if votes_path.exists():
        for line in votes_path.read_text(encoding="utf-8").splitlines():
            v = json.loads(line)
            if v["winner"] == "skip":
                continue
            for m in (v["model_a"], v["model_b"]):
                games[m] += 1
            if v["winner"] == "tie":
                ties[v["model_a"]] += 1
                ties[v["model_b"]] += 1
            elif v["winner"] in wins:
                wins[v["winner"]] += 1
    print("\n===== РЕЗУЛТАТИ (откриени модели) =====")
    for m in sorted(models, key=lambda x: -(wins[x] + 0.5 * ties[x])):
        g = games[m]
        score = (wins[m] + 0.5 * ties[m]) / g if g else 0
        print(f"  {m}: {wins[m]}W {ties[m]}T {g - wins[m] - ties[m]}L "
              f"({g} games, score {score:.1%})")


if __name__ == "__main__":
    main()
