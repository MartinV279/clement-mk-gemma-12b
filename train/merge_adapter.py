#!/usr/bin/env python3
"""Merge a LoRA adapter into full base weights.

Used twice in the pipeline:
  1. after CPT decay  -> skazna-base (so SFT trains on a real base, not a
     stacked adapter, which keeps the SFT LoRA clean and comparable across rounds)
  2. after ORPO       -> the shippable model that gets quantised to GGUF

Usage:
  python train/merge_adapter.py --base gemma4-12b-text \
      --adapter checkpoints/skazna-base/final --out skazna-base-merged
"""

import argparse
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base: {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu",
        token=os.environ.get("HF_TOKEN"))
    print(f"applying adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging...", flush=True)
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base, token=os.environ.get("HF_TOKEN"))
    tok.save_pretrained(out)
    # carry the chat template over if the base has one as a separate file
    src_tmpl = Path(args.base) / "chat_template.jinja"
    if src_tmpl.exists():
        shutil.copy(src_tmpl, out / "chat_template.jinja")
    print(f"MERGED -> {out}", flush=True)


if __name__ == "__main__":
    main()
