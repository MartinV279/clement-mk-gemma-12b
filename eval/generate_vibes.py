#!/usr/bin/env python3
"""Generate answers to the vibes prompts from one model (Phase 0 baselines +
every later checkpoint). Runs on the rented pod for 8-12B models; the local
8GB box may only run Q4-quantized baselines via llama.cpp (separate path).

Output feeds eval/arena.py for blind user scoring. One JSONL line per prompt:
  {"id", "model", "prompt", "answer", "gen_params"}

Usage (pod):
  uv run python eval/generate_vibes.py --model google/gemma-4-12B-it \
      --prompts eval/vibes_prompts.jsonl --out eval/generations/
"""

import argparse
import json
import os
from pathlib import Path

GEN_PARAMS = {"max_new_tokens": 768, "temperature": 0.7, "top_p": 0.95, "do_sample": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--peft", default=None, help="adapter dir to apply on top of --model")
    ap.add_argument("--name", default=None, help="output file stem (defaults to model id)")
    ap.add_argument("--prompts", default="eval/vibes_prompts.jsonl")
    ap.add_argument("--out", default="eval/generations/")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_PARAMS["max_new_tokens"])
    args = ap.parse_args()

    import torch
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    gen_params = {**GEN_PARAMS, "max_new_tokens": args.max_new_tokens}

    prompts = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=token, torch_dtype="auto", device_map="auto"
    )
    if args.peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.model.replace("/", "__")
    out_path = out_dir / f"{stem}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(prompts, 1):
            messages = [{"role": "user", "content": item["prompt"]}]
            # Chat template read from the checkpoint itself, never from memory
            # (CLAUDE.md hard rule 2). Base (pt) checkpoints have none -> raw
            # prompt. transformers 5.x raises AttributeError on missing
            # attributes, so probe via apply_chat_template directly.
            try:
                try:
                    # reasoning models (Qwen3 etc.): plain answers only — thinking
                    # pollutes the arena and eats the token budget
                    text = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False,
                    )
                # gemma-4 canonical template appends an empty thought-channel
                # scaffold after <|turn>model that SFT never trained on -> the
                # model emits junk fragments. Cut the prompt at the model marker
                # so it matches the training render exactly.
                marker = "<|turn>model\n"
                pos = text.rfind(marker)
                if pos != -1:
                    text = text[: pos + len(marker)]
                # text already carries <bos> from the template
                inputs = tokenizer(text, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(model.device)
            except Exception:
                inputs = tokenizer(item["prompt"], return_tensors="pt").input_ids.to(model.device)

            # stop at end-of-turn, not just EOS — without this the model closes
            # its turn and hallucinates a follow-up user question until the cap
            stop_ids = [tokenizer.eos_token_id]
            turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
            if isinstance(turn_end, int) and turn_end >= 0:
                stop_ids.append(turn_end)
            with torch.inference_mode():
                output = model.generate(inputs, eos_token_id=stop_ids, **gen_params)
            answer = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
            answer = answer.split("<turn|>")[0].strip()
            # defensive: strip any leaked reasoning block
            if "</think>" in answer:
                answer = answer.split("</think>", 1)[1]

            f.write(json.dumps({
                "id": item["id"],
                "model": args.model,
                "prompt": item["prompt"],
                "answer": answer.strip(),
                "gen_params": gen_params,
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] {item['id']}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
