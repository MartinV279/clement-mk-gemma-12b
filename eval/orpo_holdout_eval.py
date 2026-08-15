#!/usr/bin/env python3
"""Preference accuracy on the frozen ORPO holdout (prefs_holdout.jsonl).

For each pair: sum logprob of chosen vs rejected continuation given the
prompt (training format). Reports overall + per-kind accuracy.
  .venv/bin/python eval/orpo_holdout_eval.py --model skazna-ship
"""
import argparse
import json

TURN_USER = "<|turn>user\n"
TURN_MODEL = "<|turn>model\n"
TURN_END = "<turn|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/processed/prefs_holdout.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    bos = tok.bos_token or ""

    def seq_logprob(prompt: str, cont: str) -> float:
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        full = tok(prompt + cont, return_tensors="pt", add_special_tokens=False).input_ids
        if full.shape[1] > 3500:
            full = full[:, :3500]
        full = full.cuda()
        with torch.inference_mode():
            logits = model(full).logits[0, :-1].float()
        targets = full[0, 1:]
        lp = torch.log_softmax(logits, -1).gather(1, targets[:, None])[:, 0]
        start = p_ids.shape[1] - 1
        seg = lp[start:]
        return seg.sum().item(), seg.mean().item()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    stats = {}
    for i, r in enumerate(rows):
        prompt = f"{bos}{TURN_USER}{r['prompt']}{TURN_END}{TURN_MODEL}"
        cs, cm = seq_logprob(prompt, r["chosen"] + TURN_END)
        rs, rm = seq_logprob(prompt, r["rejected"] + TURN_END)
        k = r.get("kind", "?")
        stats.setdefault(k, [0, 0, 0])   # [sum-wins, mean-wins, n]
        stats[k][0] += int(cs > rs)
        stats[k][1] += int(cm > rm)
        stats[k][2] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    tot_s = sum(v[0] for v in stats.values())
    tot_m = sum(v[1] for v in stats.values())
    tot_n = sum(v[2] for v in stats.values())
    print(f"\nOVERALL sum-acc {tot_s}/{tot_n} = {tot_s/tot_n:.3f} | len-norm acc {tot_m}/{tot_n} = {tot_m/tot_n:.3f}")
    for k, (ws, wm, n) in sorted(stats.items()):
        print(f"  {k:12} sum {ws/n:.3f} | norm {wm/n:.3f}  (n={n})")


if __name__ == "__main__":
    main()
