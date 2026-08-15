#!/usr/bin/env python3
"""Phase 4 ORPO (TRL ORPOTrainer via Unsloth). Pinned: 1 epoch, LR 5e-6,
save<=500 steps, frozen 500-pair holdout excluded (prefs_holdout.jsonl).

Data: data/processed/prefs_train.jsonl {prompt, chosen, rejected, kind}.
The chat template comes from the model dir (training-format, no thought
channel). Reference-free ORPO: one model in memory.

Usage (pod):
  .venv-train/bin/python train/run_orpo.py --base-model skazna-sft-merged \
      --out checkpoints/orpo
Smoke (local 8GB):
  python train/run_orpo.py --smoke
"""

import argparse
import json
from pathlib import Path

TURN_USER = "<|turn>user\n"
TURN_MODEL = "<|turn>model\n"
TURN_END = "<turn|>\n"


def load_pairs(path: str, tokenizer) -> "Dataset":
    from datasets import Dataset
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    bos = tokenizer.bos_token or ""

    def fmt(r):
        return {
            "prompt": f"{bos}{TURN_USER}{r['prompt']}{TURN_END}{TURN_MODEL}",
            "chosen": f"{r['chosen']}{TURN_END}",
            "rejected": f"{r['rejected']}{TURN_END}",
        }
    return Dataset.from_list([fmt(r) for r in rows])



class _MetricsToFile:
    """Append every Trainer log dict to a JSONL file.

    Exists because report_to=[] meant NO metric ever reached disk: not a tracker,
    and not even stdout (verified empirically on the anneal run — grep -c loss = 0,
    only tqdm bars). Every earlier round's loss curve is therefore unrecoverable.

    Deliberately dependency-free and fully exception-swallowing: a telemetry
    writer must never be able to kill a training run. That is also why this is
    not report_to=["wandb"] — a failed wandb auth raises inside TRL.
    """

    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            import json as _j
            rec = dict(logs or {})
            rec["step"] = state.global_step
            rec["epoch"] = state.epoch
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(_j.dumps(rec) + "\n")
        except Exception:
            pass

    # TrainerCallback protocol: everything else is a no-op
    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="skazna-sft-merged")
    ap.add_argument("--data", default="data/processed/prefs_train.jsonl")
    ap.add_argument("--out", default="checkpoints/orpo")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.base_model = "Qwen/Qwen3-0.6B"
        args.out = "checkpoints/smoke-orpo"
        args.batch_size, args.grad_accum = 1, 2

    from unsloth import FastLanguageModel
    import torch
    from trl import ORPOConfig, ORPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=4096, load_in_4bit=True, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=64, use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none")

    ds = load_pairs(args.data, tokenizer)
    if args.smoke:
        ds = ds.select(range(8))

    cfg = ORPOConfig(
        output_dir=args.out,
        num_train_epochs=1,
        learning_rate=5e-6,                       # pinned
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=512 if args.smoke else 4096,
        max_prompt_length=128 if args.smoke else 1024,
        beta=0.1,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_steps=100, save_total_limit=2,
        bf16=True, optim="adamw_8bit",
        max_steps=3 if args.smoke else -1,
        report_to=[],
    )
    trainer = ORPOTrainer(model=model, args=cfg, train_dataset=ds,
                          processing_class=tokenizer)
    trainer.add_callback(_MetricsToFile(str(Path(args.out) / 'metrics_orpo.jsonl')))
    ckpts = sorted(Path(args.out).glob("checkpoint-*")) if args.resume else []
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(args.out + "/final")
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
