#!/usr/bin/env python3
"""Extract the text tower from google/gemma-4-12B (gemma4_unified) into a
standalone `gemma4` text checkpoint that Unsloth/transformers 5.5 can train.

Verified feasible 2026-07-25: all 666 `model.language_model.*` tensors map
1:1 onto transformers 5.5's Gemma4 text implementation (layer_scalar incl.);
vision/audio embedders (11 tensors) are dropped; embeddings are tied (no
separate lm_head in the unified checkpoint).

ACCEPTANCE TEST (mandatory, run right after): perplexity parity — the
extracted model must produce (near-)identical perplexity to the unified
original on the same Macedonian text. See --verify.

Runs on the POD (needs ~50GB free disk + both venvs). Usage:
  python train/extract_text_model.py --src <unified_dir_or_repo> --dst gemma4-12b-text
  python train/extract_text_model.py --verify gemma4-12b-text   # in each venv
"""

import argparse
import json
import os
from pathlib import Path

PREFIX = "model.language_model."
DROP_PREFIXES = ("model.vision_embedder.", "model.embed_vision.", "model.embed_audio.")


def extract(src: str, dst: str) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not Path(src).exists():
        src = snapshot_download(src, token=os.environ.get("HF_TOKEN"),
                                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.jinja"])
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # config: unified text_config -> standalone gemma4 text config
    uni = json.loads((src / "config.json").read_text())
    text_cfg = dict(uni.get("text_config") or uni)
    text_cfg["model_type"] = "gemma4_text"  # NOT "gemma4" (that is the April
    # E-series per-layer-embedding arch; parity fails catastrophically with it)
    text_cfg["architectures"] = ["Gemma4TextForCausalLM"]
    text_cfg["tie_word_embeddings"] = True
    for k in ("vision_config", "audio_config"):
        text_cfg.pop(k, None)
    (dst / "config.json").write_text(json.dumps(text_cfg, indent=2))

    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (src / f).exists():
            (dst / f).write_bytes((src / f).read_bytes())

    st_files = sorted(src.glob("*.safetensors"))
    out, dropped, kept = {}, 0, 0
    for stf in st_files:
        with safe_open(stf, framework="pt") as f:
            for name in f.keys():
                if name.startswith(DROP_PREFIXES):
                    dropped += 1
                    continue
                if name.startswith(PREFIX):
                    out["model." + name[len(PREFIX):]] = f.get_tensor(name)
                    kept += 1
                else:
                    print(f"  unexpected tensor (dropped): {name}")
                    dropped += 1
    save_file(out, dst / "model.safetensors", metadata={"format": "pt"})
    print(f"extracted {kept} tensors ({dropped} dropped) -> {dst}")


def verify(model_dir: str) -> None:
    """Perplexity on fixed MK sentences — run in BOTH venvs and compare."""
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sents = [
        "Охридското Езеро е едно од најстарите езера во Европа и дом на многу ендемични видови.",
        "Македонскиот јазик е јужнословенски јазик со богата литературна традиција.",
        "Скопје е главен град на Северна Македонија и нејзин најголем економски центар.",
    ]
    tok = AutoTokenizer.from_pretrained(model_dir, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"))
    model.eval()
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            loss = model(ids, labels=ids).loss.item()
        print(f"ppl {math.exp(loss):10.3f}  {s[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="google/gemma-4-12B")
    ap.add_argument("--dst", default="gemma4-12b-text")
    ap.add_argument("--verify", default=None, metavar="MODEL_DIR")
    args = ap.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        extract(args.src, args.dst)


if __name__ == "__main__":
    main()
