#!/usr/bin/env python3
"""Phase 2 CPT runner — dual backend, WSD-capable.

Reads train/cpt_config.yaml. That file is not shipped (it names this
project's own data sources); copy train/cpt_config.example.yaml to
train/cpt_config.yaml and fill in your mixture.

Schedule strategy (user-approved 2026-07-27, replaces plain cosine):
  phase 1 (stable):  --scheduler constant   -> run for as long as planned
  phase 2 (decay):   --scheduler linear --init-adapter <phase1_final> \
                     --max-steps <~8% of phase 1>  -> anneals LR to 0
This makes the stopping point a runtime decision instead of a pre-committed
schedule, so a faster-than-expected run uses MORE tokens rather than fewer.

  --backend hf       plain transformers+peft+bitsandbytes QLoRA
                     (supports gemma4_unified; ~1.5-2x slower)
  --backend unsloth  Unsloth fast path (does NOT support gemma4_unified as of
                     2026.7.5 — usable only if the text-tower extraction spike
                     succeeds, or for smoke tests on small models)

Pinned hyperparameters come from train/cpt_config.yaml (CLAUDE.md rule 3).
Every run checkpoints every <=500 steps and supports --resume (rule 5).
Local smoke test (MANDATORY before any pod, CLAUDE.md Local hardware):
  uv run python train/run_cpt.py --backend hf --smoke

Data: pre-packed mixture parquet shards (data/mixture/), input_ids @8192.
"""

import argparse
import glob
import os
from pathlib import Path

import yaml

SMOKE_OVERRIDES = {
    "base_model": "Qwen/Qwen3-0.6B",
    "max_seq_length": 1024,   # 8GB local GPU
    "lora_r": 16,
    "max_steps": 12,
    "save_steps": 5,
    "batch_size": 1,
    "grad_accum": 2,
}


def load_cfg(path: str, smoke: bool) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = {
        "base_model": raw["base_model"],
        "load_in_4bit": raw.get("load_in_4bit", True),
        "lora_r": raw["lora"]["r"],
        "use_rslora": raw["lora"].get("use_rslora", True),
        "train_embed": raw["lora"].get("train_embed_tokens", True),
        "lr": float(raw["learning_rate"]),
        "emb_lr": float(raw.get("embedding_learning_rate", raw["learning_rate"])),
        "warmup_ratio": float(raw.get("warmup_ratio", 0.01)),
        "max_seq_length": int(raw.get("max_seq_length", 8192)),
        "save_steps": min(500, int(raw.get("checkpointing", {}).get("save_steps", 500))),
        "max_steps": -1,
        "batch_size": 1,
        "grad_accum": 8,
    }
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    return cfg


def load_dataset_blocks(seq_len: int, smoke: bool, base_model: str = None):
    from datasets import Dataset, load_dataset
    if smoke:
        # mixture blocks are GEMMA-tokenized; the smoke model has a different
        # vocab — tokenize fresh corpus text with the smoke model's tokenizer
        import gzip
        import json
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
        buf, blocks = [], []
        with gzip.open("data/processed/mk_corpus_filtered.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                buf.extend(tok(json.loads(line)["text"][:20000],
                               add_special_tokens=False)["input_ids"])
                buf.append(tok.eos_token_id or 0)
                while len(buf) >= seq_len:
                    blocks.append(buf[:seq_len])
                    del buf[:seq_len]
                if len(blocks) >= 64:
                    return Dataset.from_dict({"input_ids": blocks})
    files = sorted(glob.glob("data/mixture/mixture-*.parquet"))
    assert files, "mixture not built"
    return load_dataset("parquet", data_files=files, split="train", streaming=True)


def collate(features):
    import torch
    ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def build_hf(cfg):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if cfg["load_in_4bit"] else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"), attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)
    modules_to_save = ["embed_tokens", "lm_head"] if cfg["train_embed"] else None
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_r"],
                      target_modules="all-linear", use_rslora=cfg["use_rslora"],
                      modules_to_save=modules_to_save, task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def make_optimizer(model, cfg):
    """Decoupled LR: LoRA adapters at lr, embed/lm_head at emb_lr (~10x lower)."""
    import bitsandbytes as bnb
    emb, lora = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (emb if ("embed_tokens" in n or "lm_head" in n) else lora).append(p)
    groups = [{"params": lora, "lr": cfg["lr"]}]
    if emb:
        groups.append({"params": emb, "lr": cfg["emb_lr"]})
    return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), weight_decay=0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/cpt_config.yaml")
    ap.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--base-model-override", default=None,
                    help="e.g. extracted gemma4-12b-text for the unsloth path")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    # WSD support: phase 1 runs "constant" (stable) for as long as the run
    # is planned to last; phase 2 resumes those weights with "linear" decay to zero.
    ap.add_argument("--scheduler", choices=["cosine", "constant", "linear"],
                    default=None, help="overrides cpt_config lr_scheduler_type")
    ap.add_argument("--init-adapter", default=None,
                    help="continue from an existing adapter (WSD decay phase)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    cfg = load_cfg(args.config, args.smoke)
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.base_model_override:
        cfg["base_model"] = args.base_model_override
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum:
        cfg["grad_accum"] = args.grad_accum
    out_dir = args.out or ("checkpoints/smoke" if args.smoke else "checkpoints/cpt")

    import torch
    from transformers import Trainer, TrainingArguments

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        model, _tok = FastLanguageModel.from_pretrained(
            cfg["base_model"], max_seq_length=cfg["max_seq_length"],
            load_in_4bit=cfg["load_in_4bit"], token=os.environ.get("HF_TOKEN"))
        targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        if cfg["train_embed"] and not args.smoke:
            targets += ["embed_tokens", "lm_head"]  # pinned: trained at emb_lr
        if args.init_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.init_adapter,
                                              is_trainable=True)
            print(f"continuing from adapter: {args.init_adapter}")
        else:
            model = FastLanguageModel.get_peft_model(
                model, r=cfg["lora_r"], lora_alpha=cfg["lora_r"],
                target_modules=targets,
                use_rslora=cfg["use_rslora"], use_gradient_checkpointing="unsloth")
    else:
        model = build_hf(cfg)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = load_dataset_blocks(cfg["max_seq_length"], args.smoke, cfg["base_model"])
    targ_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=(args.scheduler or "cosine"),
        warmup_ratio=(0.0 if args.init_adapter else cfg["warmup_ratio"]),
        logging_steps=1 if args.smoke else 10,
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        bf16=True,
        report_to=[],
        seed=42,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
    )
    if args.backend == "unsloth" and not args.smoke:
        # decoupled embedding LR via Unsloth's own trainer (pinned decision)
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        targs = UnslothTrainingArguments(embedding_learning_rate=cfg["emb_lr"],
                                         **targ_kwargs)
        trainer = UnslothTrainer(model=model, args=targs, train_dataset=ds,
                                 data_collator=collate)
    else:
        targs = TrainingArguments(**targ_kwargs)
        trainer = Trainer(model=model, args=targs, train_dataset=ds,
                          data_collator=collate,
                          optimizers=(make_optimizer(model, cfg), None))
    ckpts = sorted(Path(out_dir).glob("checkpoint-*")) if args.resume else []
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(out_dir + "/final")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
