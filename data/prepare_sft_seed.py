#!/usr/bin/env python3
"""Screen LVSTCK/sft-mk with the approved automated filters and prepare the
user review batch (CLAUDE.md: filters propose, the USER disposes).

Pipeline per conversation: langID -> bleed screen -> KenLM ppl -> decontamination
-> embedding dedup. Outputs:
  data/processed/sft_seed_pool.jsonl   — surviving conversations (candidate pool)
  data/processed/sft_seed_stats.json   — per-filter drop counts
  data/processed/review_batch_300.md   — 300 random survivors formatted for
                                         fast reading; the user deletes bad ones
                                         by listing their ids

Usage:
  uv run python data/prepare_sft_seed.py [--input data/downloads/sft_mk/sft_dataset_train.jsonl]
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "filters"))
from bleed_blocklist import find_hits, load_approved  # noqa: E402
from decontaminate import contamination_sets, ngrams, normalize_words  # noqa: E402
from kenlm_filter import doc_perplexity  # noqa: E402
from langid_filter import predict_lang  # noqa: E402

KENLM_THRESHOLD = 3595       # user-approved 2026-07-23 (data/filters/approved.yaml)
LANGID_MIN_CONF = 0.65
MAX_SCRIPT_CHARS = 2
NGRAM = 8
DEDUP_THRESHOLD = 0.92
SEED = 42


def conv_text(conv: list) -> str:
    return "\n".join(turn["content"] for turn in conv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/downloads/sft_mk/sft_dataset_train.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--prefix", default="sft_seed", help="output filename prefix")
    ap.add_argument("--review-size", type=int, default=300)
    ap.add_argument("--kenlm-threshold", type=float, default=KENLM_THRESHOLD,
                    help="use 32000 for conversational data (approved.yaml)")
    args = ap.parse_args()

    import fasttext
    import kenlm
    import torch
    from sentence_transformers import SentenceTransformer

    lid = fasttext.load_model("data/models/lid.176.bin")
    lm = kenlm.Model("data/models/kenlm_mk_5gram.bin")
    lexical = load_approved()
    grams, shorts = contamination_sets(["eval/vibes_prompts.jsonl"], NGRAM)

    stats = Counter()
    survivors = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            conv = json.loads(line)["conversations"]
            text = conv_text(conv)

            labels, confs = predict_lang(lid, text[:2000])
            if not labels or labels[0] != "__label__mk" or confs[0] <= LANGID_MIN_CONF:
                stats["drop_langid"] += 1
                continue
            script_hits, lex_hits = find_hits(text, lexical)
            if len(script_hits) > MAX_SCRIPT_CHARS or lex_hits:
                stats["drop_bleed"] += 1
                continue
            if doc_perplexity(lm, text) > args.kenlm_threshold:
                stats["drop_kenlm"] += 1
                continue
            words = normalize_words(text)
            if (ngrams(words, NGRAM) & grams) or any(s in " ".join(words) for s in shorts):
                stats["drop_decontaminate"] += 1
                continue
            rec = json.loads(line)
            rec.setdefault("id", i)
            survivors.append(rec)

    # embedding dedup on survivors (local GPU — approved workload)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                device="cuda" if torch.cuda.is_available() else "cpu")
    texts = [conv_text(s["conversations"])[:2000] for s in survivors]
    emb = model.encode(texts, batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)
    keep = [True] * len(survivors)
    chunk = 1024
    for start in range(0, len(texts), chunk):
        end = min(start + chunk, len(texts))
        sims = emb[start:end] @ emb[:end].T
        for i in range(start, end):
            if not keep[i]:
                continue
            row = sims[i - start, :i]
            if row.numel():
                mask = torch.tensor(keep[:i], device=row.device)
                if bool((row[mask] > DEDUP_THRESHOLD).any()):
                    keep[i] = False
    deduped = [s for s, k in zip(survivors, keep) if k]
    stats["drop_dedup"] = len(survivors) - len(deduped)
    stats["kept"] = len(deduped)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.prefix}_pool.jsonl").open("w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out / f"{args.prefix}_stats.json").write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")

    rng = random.Random(SEED)
    batch = rng.sample(deduped, min(args.review_size, len(deduped)))
    with (out / f"{args.prefix}_review_{args.review_size}.md").open("w", encoding="utf-8") as f:
        f.write("# SFT seed review batch — random survivors of the automated filters\n\n"
                "Read fast; note the **id** of every conversation that is bad Macedonian\n"
                "(translationese, bleed, stiff, wrong) — those ids get deleted from the pool.\n\n")
        for s in batch:
            f.write(f"---\n\n## id {s['id']}\n\n")
            for turn in s["conversations"]:
                who = "**Корисник:**" if turn["role"] == "user" else "**Асистент:**"
                f.write(f"{who} {turn['content']}\n\n")
    print("FINAL:", dict(stats))
    print(f"pool: {out}/{args.prefix}_pool.jsonl | review: {out}/{args.prefix}_review_{args.review_size}.md")


if __name__ == "__main__":
    main()
