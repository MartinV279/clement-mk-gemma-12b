# Benchmark results — raw harness output

The `lm-evaluation-harness` output behind every number in the top-level README,
so the table can be recomputed rather than taken on trust.

```
battery_clement.json             Clement 12B
battery_yak.json                 LVSTCK/domestic-yak-8B-instruct
battery_base_gemma-4-12B.json    google/gemma-4-12B (the untuned base)
math_probe_clement.json          deterministic arithmetic probe, 16/20
gemini_flash_exams_mk.json       gemini-3-flash on exams_mk, as a scale reference
```

All three battery runs used harness 0.4.12, identical task definitions (the
YAMLs in [`../lm_eval_tasks/`](../lm_eval_tasks/)), identical seeds, and
loglikelihood accuracy over the answer options. Per-task `acc_stderr` is in the
files.

Recompute the published table:

```python
import json
f = {"base": "battery_base_gemma-4-12B.json", "yak": "battery_yak.json",
     "clement": "battery_clement.json"}
d = {k: json.load(open(v))["results"] for k, v in f.items()}
tasks = sorted(d["clement"])
for k in d:
    print(k, round(sum(d[k][t]["acc,none"] for t in tasks) / len(tasks), 4))
```

## Reading these honestly

- The averages are **0.5481 / 0.5572 / 0.5622** (base / yak / Clement). The gap
  between Clement and yak is **z = 0.82 — a statistical tie**, not a win.
- Three tasks are natively authored (`exams_mk`, `copa_mk`, `include_mk`); the
  other seven are machine-translated ports of English benchmarks. Clement's
  margin over yak is **+0.0144** on the native three and **+0.0010** on the
  translated seven.
- `gemini_flash_exams_mk.json` used a **generative** protocol (the model outputs
  a letter), not loglikelihood ranking. It is a scale reference for the whole
  open 8–12B class, not a same-protocol row.
- The math probe is auto-scored with no judge: an item counts as correct only if
  an accepted answer appears on a number boundary, so "16" cannot match "160".

## A note on the files

The model label in `battery_clement.json` and `math_probe_clement.json` was
renamed from the internal build id to the release name for publication, and the
harness's machine-environment dump (hostname, local paths, hardware serials) was
removed. Scores, seeds, and task configuration are unchanged; both files record
this in a `_note` field.

## Comparison with published VezilkaLLM results

[VezilkaLLM](https://huggingface.co/finki-ukim/VezilkaLLM) (FINKI/UKIM, 4B, base
model on Gemma-3-4B) publishes benchmark results on its model card. Seven of its
eight tasks overlap with this battery (NQ Open does not). Their published
numbers against ours:

| Task | VezilkaLLM 4B (published) | Clement 12B (this repo) |
|---|---|---|
| ARC Challenge | 0.30 | 0.3473 |
| ARC Easy | 0.50 | 0.5274 |
| BoolQ | 0.72 | 0.7807 |
| HellaSwag | 0.41 | 0.4754 |
| OpenBookQA | 0.25 | 0.3260 |
| PIQA | 0.65 | 0.6872 |
| Winogrande | 0.59 | 0.6140 |

Read it with two caveats, stated so you do not have to discover them:

- **These are cross-setup numbers, not a same-harness re-run.** Usefully, their
  card also lists domestic-yak-8B, which we *did* measure ourselves — their yak
  numbers differ from our yak numbers by roughly 1–3 points per task (e.g. ARC
  Easy 0.52 published vs 0.5484 measured here). Treat differences under ~3
  points as harness noise. Clement's margins over VezilkaLLM's published
  numbers are ~3–9 points.
- **The models are not size-matched.** VezilkaLLM is a 4B base model and states
  itself to be a foundation for future fine-tuning; Clement is a 12B
  instruction-tuned model. A larger adapted model outperforming a smaller base
  model is expected — the comparison is included for completeness of the
  Macedonian-model landscape, not as a like-for-like contest.
