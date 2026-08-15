#!/usr/bin/env python3
"""Deterministic, auto-scorable arithmetic probe — the only Skazna eval that
needs no judge at all.

The arena measures preference and the battery measures multiple-choice ranking;
neither catches "the model reasons fluently and gets the number wrong." One
build's расудување collapse (4-0 -> 0-4 in the arena) was completely invisible
to the battery. This probe makes that failure mode a number.

Scoring: an item is correct if ANY accepted answer string appears in the model's
output, comparing digits only (so 1.056 / 1,056 / 1056 all match) and requiring
the match to fall on a number boundary so "16" does not match "160".

  .venv-train/bin/python eval/run_math_probe.py --model skazna-ship --name clement
"""

import argparse
import json
import re
from pathlib import Path

GEN = {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}


def digits_only(s: str) -> str:
    """1.056 / 1,056 / 1 056 -> 1056. Separators vary by locale and by model."""
    return re.sub(r"[.,\s ]", "", s)


def hit(answer: str, accepted) -> bool:
    hay = digits_only(answer)
    for a in accepted:
        needle = digits_only(str(a))
        if not needle:
            continue
        # number-boundary match: reject 16 inside 160 or 3160
        for m in re.finditer(re.escape(needle), hay):
            before = hay[m.start() - 1] if m.start() else ""
            after = hay[m.end()] if m.end() < len(hay) else ""
            if not before.isdigit() and not after.isdigit():
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--peft", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--probe", default="eval/math_probe.jsonl")
    ap.add_argument("--out", default="eval/results/")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(l) for l in Path(args.probe).open(encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="auto")
    if args.peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft)

    stop_ids = [tok.eos_token_id]
    te = tok.convert_tokens_to_ids("<turn|>")
    if isinstance(te, int) and te >= 0:
        stop_ids.append(te)

    results, correct = [], 0
    for it in items:
        msgs = [{"role": "user", "content": it["q"]}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            marker = "<|turn>model\n"
            pos = text.rfind(marker)
            if pos != -1:
                text = text[:pos + len(marker)]
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        except Exception:
            ids = tok(it["q"], return_tensors="pt").input_ids
        ids = ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, eos_token_id=stop_ids, **GEN)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ans = ans.split("<turn|>")[0].strip()
        ok = hit(ans, it["a"])
        correct += ok
        results.append({"id": it["id"], "q": it["q"], "expected": it["a"],
                        "answer": ans, "correct": ok})
        print(f"  [{it['id']:2}] {'OK ' if ok else 'MISS'} expected {it['a']} | "
              f"{ans[:90].replace(chr(10), ' ')}", flush=True)

    stem = args.name or args.model.replace("/", "__")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) / f"math_probe_{stem}.json"
    score = correct / max(len(items), 1)
    dest.write_text(json.dumps({"model": stem, "correct": correct,
                                "total": len(items), "score": score,
                                "results": results}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nMATH PROBE {stem}: {correct}/{len(items)} = {score:.1%} -> {dest}")


if __name__ == "__main__":
    main()
