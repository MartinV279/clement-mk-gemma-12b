#!/usr/bin/env python3
"""Facts-anneal: short continued-pretrain over paraphrase texts (variant A).
LoRA on the merged CPT base, low LR, packing, 3 epochs over ~15k short texts.
  .venv-train/bin/python train/run_anneal.py --base-model skazna-base-merged \
      --data data/processed/anneal.jsonl --out checkpoints/anneal"""
import argparse, json


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
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", default="data/processed/anneal.jsonl")
    ap.add_argument("--out", default="checkpoints/anneal")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from pathlib import Path
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=2048, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=64, use_rslora=True,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth")

    from datasets import Dataset
    texts = [json.loads(l)["text"] for l in open(args.data, encoding="utf-8")]
    ds = Dataset.from_dict({"text": [t + tokenizer.eos_token for t in texts]})

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=3,
                    per_device_train_batch_size=8, gradient_accumulation_steps=2,
                    learning_rate=1e-5, lr_scheduler_type="cosine",
                    warmup_ratio=0.03, logging_steps=10, save_steps=200,
                    save_total_limit=2, bf16=True, max_length=2048,
                    packing=True, dataset_text_field="text", report_to=[], seed=42)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tokenizer)
    trainer.add_callback(_MetricsToFile(str(Path(args.out) / 'metrics_anneal.jsonl')))
    ckpts = sorted(Path(args.out).glob("checkpoint-*")) if args.resume else []
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(args.out + "/final")
    print(f"anneal done -> {args.out}")

if __name__ == "__main__":
    main()
