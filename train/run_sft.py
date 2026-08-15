#!/usr/bin/env python3
"""Phase 3 SFT runner — skazna-sft (option C: SFT-first on gemma-4-12B-it).

Data: data/processed/sft_mix.jsonl ({conversations: [{role, content}...]}).
Chat template is read from the checkpoint at runtime (CLAUDE.md rule 2);
docs/constitution.md is injected as the system prompt (or prepended to the
first user turn if the template rejects a system role). Loss is computed on
assistant turns only (unsloth train_on_responses_only).

Smoke (mandatory before pod):
  .venv-train/bin/python train/run_sft.py --smoke
Pod:
  .venv-train/bin/python train/run_sft.py --base-model gemma4-12b-it-text \
      --out checkpoints/sft
"""

import argparse
import json
import os
import random
from pathlib import Path

import yaml

SMOKE_MODEL = "Qwen/Qwen3-0.6B"
SEED = 42


def render(tokenizer, conversations, system_text):
    """Apply the checkpoint's own chat template; fall back to prepending the
    system text into the first user message when the template has no system role."""
    msgs = [{"role": "system", "content": system_text}] + conversations
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
    except Exception:
        conv = [dict(m) for m in conversations]
        conv[0]["content"] = system_text + "\n\n" + conv[0]["content"]
        text = tokenizer.apply_chat_template(conv, tokenize=False)
    # avoid double-BOS: SFTTrainer re-tokenizes with special tokens
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return text



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
    ap.add_argument("--config", default="train/sft_config.yaml")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--data", default="data/processed/sft_mix.jsonl")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override num_train_epochs from config")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lora_r = int(raw["lora"]["r"])
    lr = float(raw["learning_rate"])
    epochs = args.epochs or int(raw.get("num_train_epochs", 2))

    base = args.base_model or SMOKE_MODEL
    if args.smoke:
        base = SMOKE_MODEL

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        base, max_seq_length=args.max_seq, load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"))
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    system_text = Path("docs/constitution.md").read_text(encoding="utf-8")
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.smoke:
        rows = rows[:64]
    random.Random(SEED).shuffle(rows)
    texts = [render(tokenizer, r["conversations"], system_text) for r in rows]

    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})

    from trl import SFTConfig, SFTTrainer
    targs = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=1 if args.smoke else args.batch_size,
        gradient_accumulation_steps=2 if args.smoke else args.grad_accum,
        num_train_epochs=epochs,
        max_steps=10 if args.smoke else -1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke else 10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        max_length=1024 if args.smoke else args.max_seq,
        packing=False,
        dataset_text_field="text",
        report_to=[],
        seed=SEED,
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                         processing_class=tokenizer)
    trainer.add_callback(_MetricsToFile(str(Path(args.out) / 'metrics_sft.jsonl')))

    # loss on assistant turns only — marker strings read from the actual template
    try:
        from unsloth.chat_templates import train_on_responses_only
        tmpl = tokenizer.chat_template or ""
        if "<|turn>" in tmpl:                  # gemma-4 family (new markers)
            trainer = train_on_responses_only(
                trainer, instruction_part="<|turn>user\n",
                response_part="<|turn>model\n")
        elif "<start_of_turn>" in tmpl:        # gemma<=3 family
            trainer = train_on_responses_only(
                trainer, instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n")
        elif "<|im_start|>" in tmpl:           # qwen family (smoke)
            trainer = train_on_responses_only(
                trainer, instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n")
        else:
            print("WARNING: unknown template markers — training on full sequences")
    except Exception as e:
        print(f"WARNING: train_on_responses_only unavailable ({e}) — full-sequence loss")

    ckpts = sorted(Path(args.out).glob("checkpoint-*")) if args.resume else []
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(args.out + "/final")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
