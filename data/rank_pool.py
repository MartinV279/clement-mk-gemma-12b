#!/usr/bin/env python3
"""Rank an SFT pool by similarity to the USER's good/bad review labels
(nearest-centroid over multilingual sentence embeddings). Proposes a cut;
the user confirms via a boundary-sample verification batch.

The user's 300-sample verdicts are the ONLY quality ground truth here —
this just extrapolates them (CLAUDE.md: filters propose, user disposes).

Usage:
  uv run python data/rank_pool.py --pool data/processed/synth_sft_pool.jsonl \
      --review data/processed/synth_sft_review_300.md \
      --bad-ids data/processed/synth_sft_bad_ids.txt --prefix synth_sft
"""

import argparse
import json
import re
from pathlib import Path

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def conv_text(conv):
    return "\n".join(t["content"] for t in conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--bad-ids", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--boundary-sample", type=int, default=40)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    by_id = {str(r["id"]): r for r in pool}

    reviewed_ids = [m.group(1) for m in
                    re.finditer(r"^## id (\S+)", Path(args.review).read_text(encoding="utf-8"), re.M)]
    bad_ids = {l.strip() for l in open(args.bad_ids, encoding="utf-8")
               if l.strip() and not l.startswith("#")}
    good_ids = [i for i in reviewed_ids if i not in bad_ids and i in by_id]
    bad_in_pool = [i for i in bad_ids if i in by_id]
    print(f"labels: {len(good_ids)} good, {len(bad_in_pool)} bad (of {len(reviewed_ids)} reviewed)")

    st = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def embed(texts):
        return st.encode(texts, batch_size=256, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)

    good_c = embed([conv_text(by_id[i]["conversations"])[:2000] for i in good_ids]).mean(0)
    bad_c = embed([conv_text(by_id[i]["conversations"])[:2000] for i in bad_in_pool]).mean(0)
    good_c, bad_c = good_c / good_c.norm(), bad_c / bad_c.norm()

    emb = embed([conv_text(r["conversations"])[:2000] for r in pool])
    score = (emb @ good_c) - (emb @ bad_c)  # >0 leans good

    # sanity: how well does the signal separate the labeled data?
    idx = {str(r["id"]): k for k, r in enumerate(pool)}
    g = torch.tensor([score[idx[i]] for i in good_ids])
    b = torch.tensor([score[idx[i]] for i in bad_in_pool])
    thr = 0.0
    acc = ((g > thr).float().mean() + (b <= thr).float().mean()) / 2
    print(f"labeled separation: good median {g.median():.3f}, bad median {b.median():.3f}, "
          f"balanced acc @0 = {acc:.0%}")

    ranked = sorted(zip(pool, score.tolist()), key=lambda x: -x[1])
    out = Path(args.out_dir)
    with (out / f"{args.prefix}_ranked.jsonl").open("w", encoding="utf-8") as f:
        for r, s in ranked:
            r2 = dict(r)
            r2["quality_score"] = round(s, 4)
            # drop everything the user explicitly flagged
            if str(r["id"]) in bad_ids:
                continue
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    # boundary verification batch: samples nearest the proposed threshold
    near = sorted(ranked, key=lambda x: abs(x[1] - thr))[:args.boundary_sample]
    with (out / f"{args.prefix}_boundary_{args.boundary_sample}.md").open("w", encoding="utf-8") as f:
        f.write(f"# Boundary samples (score ~ {thr}) — is the proposed cut line right?\n"
                f"# Mark ids that are BAD; if most below-zero ones are bad and most\n"
                f"# above-zero ones are fine, the threshold works.\n\n")
        for r, s in near:
            f.write(f"---\n\n## id {r['id']} (score {s:+.3f})\n\n")
            for t in r["conversations"]:
                who = "**К:**" if t["role"] == "user" else "**А:**"
                f.write(f"{who} {t['content']}\n\n")

    n_above = sum(1 for _, s in ranked if s > thr)
    print(f"ranked pool written ({len(ranked)} rows, flagged removed); "
          f"{n_above} above the 0.0 line ({n_above/len(ranked):.0%})")
    print(f"boundary batch: {out}/{args.prefix}_boundary_{args.boundary_sample}.md")


if __name__ == "__main__":
    main()
