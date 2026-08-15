# Blind arena — raw record

Everything needed to recompute the published head-to-head result. Nothing here
is a summary: these are the artifacts the tally was produced from.

```
pairs_clement-vs-yak.jsonl    50 items: prompt, category, a `check` rubric, ans_1, ans_2
key_clement-vs-yak.json       which model wrote ans_1 for each item — SEALED during voting
votes_clement-vs-yak.jsonl    one vote per item: winner, confidence, written rationale, flags
verdict_clement-vs-yak.json   raw votes + final tally
reveal.py                     joins votes with the key and prints the tally
```

```bash
python eval/blind/reveal.py
```

## Protocol

1. **The prompt set was frozen first.** 50 prompts written by the native speaker
   before any training, covering ten categories (everyday questions, explanations,
   writing help, reasoning, traps, culture, language, technical, conversation,
   short-answer). It was never trained on, and every training mix was screened
   against it continuously with 8-gram overlap plus a 0.6 token-Jaccard gate —
   the gate matters because the shortest prompts here are 5–8 words and would
   slip an n-gram check.
2. **Answers were generated, then blinded.** For each item the two answers were
   shuffled into ans_1/ans_2 and the mapping written to a key file that was not
   read until every vote was cast.
3. **Each item carries a `check`** — a rubric written with the prompt, stating
   what a good answer must actually get right. This is what keeps a vote from
   collapsing into "which one sounds nicer".
4. **Voting was self-blinded**, one item at a time, with a written rationale
   recorded for every vote before the key was opened.
5. **Then the key was unsealed** and the tally computed mechanically.

## Honest notes on the method

- **The judge was an LLM**, voting under the protocol above on the native
  speaker's delegation, with the rationales spot-checked and endorsed by them
  afterwards. It is not a panel of independent native speakers, and it should
  not be read as one. What the sealed-key design does guarantee is that the
  judge could not know which model it was rewarding while it voted.
- **Both models answered under their own chat template** at matched sampling
  settings. Answers are included in full, so any disagreement with a vote can be
  checked against the actual text.
- **The losses are as informative as the wins** and are not hidden: culture is
  1–4 against us, and the rationales in the vote file say why. See the
  limitations section of the top-level README.
- The model label in `key_*.json` and `verdict_*.json` was renamed from the
  internal build id to the release name for publication. Vote values and
  ans_1/ans_2 assignments are untouched; both files record the rename in a
  `_note` / `note` field.
