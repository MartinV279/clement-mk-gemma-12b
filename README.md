# Clement 12B

**Teaching a 12B model to write Macedonian like a native — and the data-quality
research that made it work.**

Named after **Свети Климент Охридски** — St. Clement of Ohrid, who taught at the
Ohrid Literary School in the 9th century.

[Model weights on Hugging Face](https://huggingface.co/MartinV/clement-mk-gemma-12b) ·
[Dataset census](docs/dataset_census.md) ·
[Style constitution](docs/constitution.md) ·
[Raw arena record](eval/blind/) ·
Built on `google/gemma-4-12B`

---

## This started as a research question, not a model

The project began by asking something narrower than "let's train a Macedonian
model": **what are Macedonian language models actually trained on?**

So we ran the audit first — a sweep of the entire public landscape: Hugging Face
Hub, CommonCrawl derivatives, OPUS, instruction and preference sets, institutional
archives, wikis, speech and eval corpora. **235 datasets cataloged**, every size
read from the dataset card or size API, every URL verified. The result is
[`docs/dataset_census.md`](docs/dataset_census.md), published here in full.

The finding was consistent enough to be uncomfortable:

- **The largest Macedonian "datasets" are machine-translated English.**
  `fineweb-edu-translated-mkd` is 151GB of Marian-MT output. `nemotron-cc-translated`
  is another 218GB. These dwarf everything natively written.
- **The largest Macedonian instruction set is broken MT.** Aya Macedonian has
  4.12M rows; sampled rows show mangled enumerations — the English `"2."` rendered
  as `"Да се подготвиме"`, `"No"` as `"Не е така"`.
- **Native instruction data barely exists**, and Macedonian preference data
  essentially did not exist at all: the only public set was 5,988 translation-quality
  triples.
- What this trains into a model is a specific and recognizable register —
  translationese: calqued English syntax, vocabulary that is *technically* a
  dictionary match but not what anyone says, and Serbian/Bulgarian lexical bleed
  inherited from upstream crawls where the language ID was never checked by
  someone who speaks the language.

That gave us a thesis worth testing:

> **A model trained on natively-authored Macedonian — screened for cross-language
> bleed and curated by a native speaker — should express itself better than one
> trained on a much larger pile of translated text, even with modest compute.**

Clement is that thesis made runnable. It is the experiment, not the point. The
headline result is the evidence: **39–9–2 against the strongest open Macedonian
model** in a blind, sealed-key head-to-head, with every vote and rationale
published in [`eval/blind/`](eval/blind/).

And the sharpest version of that evidence is what it took to get there: the
continued-pretraining run **was never finished — it covered about 11% of the
Macedonian corpus we had already filtered and prepared.** A 12B model beating an
8B one invites the obvious explanation, that it simply saw more. It did not. What
it saw was cleaner.

---

## What we optimized for — and what we did not

This matters more than any number below, so it goes first.

**We built for how the model writes and reasons in Macedonian, not for how much
it knows.** Clement is a bigger model than the current Macedonian baseline and it
was trained differently — but the effort went into *expression*, not encyclopedic
absorption.

Concretely: **the continued-pretraining run was never finished.** It covered
~551M tokens, 28% of the 2.0B-token mixture prepared for it — about 11% of the
Macedonian corpus we had filtered and ready. The numbers are broken out under
[Training](#how-much-of-the-data-the-model-actually-saw). Everything below was
built on that partially-adapted base.

That is a real trade-off, and it shows up in the results in both directions:

- On the things we trained for — natural register, conversation, explanation,
  writing help, following the shape of a question — Clement wins decisively.
- On long-tail *facts*, particularly Macedonian cultural and historical detail,
  it can still be confidently wrong. Culture is the one arena category we lose,
  1–4. We have not hidden this; see [Limitations](#limitations).

We also learned something the hard way that is worth stating plainly:
**SFT-scale training reliably shapes style but does not make long-tail facts
retrievable.** We measured this three separate times, including a dedicated
knowledge-anneal stage built specifically to inject facts. Style moved. Facts
did not stick. That finding drove the division of labor in this project — SFT
and preference tuning own behavior and register; pretraining owns knowledge —
and it is the reason the [next step](#next-steps) is a full CPT run rather than
more instruction data.

---

## Results

### Blind head-to-head vs `LVSTCK/domestic-yak-8B-instruct`

50 frozen prompts, written by a native speaker before any training, across ten
categories. Answers shuffled and blinded; the key naming which model wrote which
answer was sealed until every vote was cast, each with a written rationale.

| | Wins | Losses | Ties |
|---|---|---|---|
| **Clement 12B** | **39** | 9 | 2 |

**81% win rate excluding ties.** On the naturalness-centric categories —
разговор, пишување, секојдневни, јазик, објаснувања — it is **23–1–1**.

| Category | Clement | yak | Tie |
|---|---|---|---|
| разговор (conversation) | 5 | 0 | 0 |
| пишување (writing) | 5 | 0 | 0 |
| секојдневни (everyday) | 5 | 0 | 0 |
| објаснувања (explanations) | 4 | 1 | 0 |
| јазик (language) | 4 | 0 | 1 |
| кратки (short answers) | 4 | 1 | 0 |
| стапици (traps) | 4 | 1 | 0 |
| техничко (technical) | 4 | 1 | 0 |
| расудување (reasoning) | 3 | 1 | 1 |
| **култура (culture)** | **1** | **4** | 0 |

The full record — prompts, both answers, the sealed key, every vote with its
rationale, and a script that recomputes the tally — is in
[`eval/blind/`](eval/blind/). Including the losses.

> **On the judge, honestly:** voting was done by an LLM under the sealed-key
> protocol on the native speaker's delegation, with rationales spot-checked and
> endorsed by them afterwards. It is not a panel of independent native speakers
> and should not be read as one. What the design does guarantee is that the judge
> could not know which model it was rewarding while it voted. Method and caveats:
> [`eval/blind/README.md`](eval/blind/README.md).

### Benchmarks

`lm-evaluation-harness` 0.4.12, loglikelihood accuracy over the answer options.
Raw harness output for all three models is published in
[`eval/results/`](eval/results/), so this table can be recomputed.

| Task | base gemma-4-12B | domestic-yak-8B | **Clement 12B** |
|---|---|---|---|
| exams_mk † | 0.5600 ± 0.0109 | 0.5277 ± 0.0110 | 0.5470 ± 0.0109 |
| copa_mk † | 0.7020 ± 0.0145 | 0.7330 ± 0.0140 | **0.7570 ± 0.0136** |
| include_mk † | 0.5762 ± 0.0207 | 0.5604 ± 0.0208 | 0.5604 ± 0.0208 |
| arc_challenge_mk | 0.3012 ± 0.0134 | 0.3387 ± 0.0138 | **0.3473 ± 0.0139** |
| arc_easy_mk | 0.5067 ± 0.0103 | **0.5484 ± 0.0102** | 0.5274 ± 0.0102 |
| boolq_mk | 0.7832 ± 0.0072 | 0.7890 ± 0.0071 | 0.7807 ± 0.0072 |
| hellaswag_mk | 0.4497 ± 0.0050 | 0.4524 ± 0.0050 | **0.4754 ± 0.0050** |
| openbookqa_mk | 0.3020 ± 0.0206 | 0.3040 ± 0.0206 | **0.3260 ± 0.0210** |
| piqa_mk | 0.6687 ± 0.0110 | 0.6850 ± 0.0108 | **0.6872 ± 0.0108** |
| winogrande_mk | 0.6314 ± 0.0136 | 0.6338 ± 0.0135 | 0.6140 ± 0.0137 |
| **Average** | 0.5481 | 0.5572 | **0.5622** |

† **human-written Macedonian.** `exams_mk` and `include_mk` are natively
authored; `copa_mk` is professionally human-*translated*. The other seven are
machine-translated ports of English benchmarks.

**Read this table carefully, because we are not going to oversell it.** Clement
has the best average, but the difference from yak is **z = 0.82 — a statistical
tie**. Anyone claiming a benchmark victory here at this margin is claiming noise.

The split underneath is the part that's actually interesting. Seven of these ten
tasks are machine-translated ports of English benchmarks; the other three are
Macedonian written by people. Clement's margin over yak on the three
human-written tasks is **+0.0144**; on the seven machine-translated ones,
**+0.0010**.

Tighten the definition and the effect shrinks but survives: counting only the two
*natively authored* sets (`exams_mk`, `include_mk`) and moving the
human-translated `copa_mk` in with the rest, it is **+0.0096** against **+0.0038**.
Either way the gains land where the language is real, which is the same pattern
the arena shows, measured a completely different way. Both splits are computable
from [`eval/results/`](eval/results/) — we are not asking you to take the
favourable one on trust.

It is also worth noting whose home field this is: the translated eval suite was
authored by the same team that built the baseline model we are comparing against.
We ran it anyway, unchanged, and published every task.

**For scale, not comparison:** a current frontier API model (gemini-3-flash)
scores **95.3%** on exams_mk under a generative protocol — not directly comparable
to loglikelihood ranking, but it does place the entire open 8–12B class honestly.

**Math probe** (deterministic, auto-scored, no judge): **16/20** — every item,
including the model's full working, is in
[`eval/results/math_probe_clement.json`](eval/results/math_probe_clement.json).
The four misses are genuine reasoning errors, not formatting artifacts: it floored
instead of ceilinged when asked how many 330ml bottles hold 5 litres (15, not 16),
lost an hour adding 2h50m to 08:35 (09:45, not 11:25), and inverted a
work-rate problem, concluding that *more* workers need *more* days (96, not 9).
The fourth answer is a single truncated sentence. Fluent Macedonian throughout —
which is precisely the failure mode worth worrying about.

---

## Data preparation

This is the part of the project the results actually came from, so it gets the
detail. Everything described here is in [`data/`](data/).

The governing rule, applied at every stage: **automated filters propose, the
native speaker disposes.** No Macedonian text was accepted or rejected on stylistic
grounds by an automated judge alone. Every threshold in
[`data/filters/approved.yaml`](data/filters/approved.yaml) was ratified by a human
who speaks the language, after reading a review batch of 100–300 real samples.

### The pipeline

```
raw → fastText language ID → Serbian/Bulgarian/Russian bleed screen
    → KenLM perplexity filter → embedding near-dedup
    → n-gram decontamination vs every eval set → human review batch
```

**1. Language ID.** fastText, Macedonian confidence > 0.65.

**2. The bleed screen** — two layers, and the most Macedonian-specific part of
the whole pipeline. Macedonian, Serbian, Bulgarian and Russian share a script;
generic language ID confuses them constantly, and upstream crawls are full of the
resulting contamination.

- *Script layer:* a full inventory of Cyrillic letters outside the Macedonian
  alphabet — including archaic Bulgarian yat and yus forms, plus Ukrainian,
  Belarusian and Russian-only characters. Rule: **≥3 foreign letters ⇒ not
  Macedonian**, with a 2-character tolerance so a quoted foreign phrase doesn't
  kill an otherwise native document.
- *Lexical layer:* a **308-term curated blocklist**
  ([`blocklist_approved.txt`](data/filters/blocklist_approved.txt)) of tell-tale
  Serbian/Bulgarian/Russian words, drafted automatically and approved term by term
  by the native speaker.

Two stories from building it, because they are the whole method in miniature:

- The `лј` digraph is a strong Serbian marker — but **Билјана** and **илјада**
  are perfectly correct Macedonian. The detector had to be restricted to
  word-initial position to stop it deleting native text.
- The word **„стварно"** was on the draft blocklist and was **removed by the
  native speaker's ruling**. It looks Serbian to a filter. It is used in
  Macedonian. The automation was wrong and the human overruled it.

**3. KenLM perplexity.** A KenLM trained on the Wikipedia and academic slice of
the corpus, with the rejection threshold proposed from the distribution and
human-confirmed at ppl > 3595 for web text. Then review found the wiki-trained
model was systematically biased against *spoken* register — flagging natural
conversational Macedonian as low quality — so conversational text got its own,
much looser threshold. A filter tuned on encyclopedia prose will quietly delete
exactly the register we were trying to teach.

**4. Near-dedup.** Multilingual sentence embeddings, cosine > 0.92.

**5. Decontamination.** 6-gram overlap against every benchmark test set (160 items
dropped), and against the [frozen 50-prompt evaluation set](eval/vibes_prompts.jsonl)
an 8-gram check **plus** a 0.6 token-Jaccard gate. The Jaccard gate is not redundant: the evaluation prompts
are as short as 5–8 words, and a short prompt slips an n-gram check easily.
Verified 0 leaks across the full 28,862-row training mix.

### What went in

| Stage | In | Kept | |
|---|---|---|---|
| CPT corpus filter | 4,341,767 docs | **4,147,663** | 95.5% — langid 4,149 · bleed 29,775 · KenLM 160,180 · decontamination 0 |
| SFT seed screen (`LVSTCK/sft-mk`) | 102,781 | **79,789** | 77.6% — including 9,051 dropped to embedding dedup |
| Synthetic generation (main round) | 29,647 | **26,710** | 90% |
| Science MCQ pack | 2,000 | **1,968** | answer-key panel |

**Synthetic data was authored, never translated.** Teachers (DeepSeek in early
rounds, Gemini Flash later) were prompted with a hand-written
[style constitution](docs/constitution.md) — the document that defines what
Macedonian *should* sound like here, down to the triple definite article and the
banned Serbian/Bulgarian forms — and instructed to *write* in Macedonian on a topic. Never
"translate this into Macedonian" — that instruction is precisely how the
translationese we were trying to escape gets manufactured.

**Preference pairs were built by construction, not by judgment.** For ORPO we
needed a *worse* answer that was worse in a specific, known way. So the rejected
side of each pair is generated mechanically from the chosen side: round-trip
machine translation (mk→en→mk), length-bloat padding, or blocklist bleed
substitution. We tried LLM-judge scoring first; it failed validation. Construction
guarantees the pair is correctly ordered, backed by a 200-pair native-speaker
spot check.

**The science-MCQ answer keys were verified by a panel** — two independent models
had to answer each question cold and agree with the stated key, or the item was
dropped (2,000 → 1,968). Then we validated the panel itself by feeding it
deliberately corrupted keys: **it rejected 20 out of 20**. A verification step you
haven't tried to fool is not a verification step.

**Targeted final round.** Loss analysis of the previous build's arena showed every
single loss was either a short-factual ask or a culture item — the depth categories
were 4–0 and 5–0. So the last data round went entirely to what was actually losing:
brevity and format compliance (with padded rejected variants), grounded cultural
answers written from verified Wikipedia passages (title-keyed, after naive substring
matching failed five different ways on namesakes), and template-generated math where
the answer is provably correct by construction and the model only writes the prose.

**Human in the loop throughout:** 700 hand-written native conversations —
published as a dataset, since native Macedonian instruction data barely exists —
review batches at every stage, and every threshold ratified by a person.

---

## Training

| Stage | What | Configuration |
|---|---|---|
| Base | `google/gemma-4-12B` (pt) | text tower extracted from the multimodal checkpoint (666 tensors); tokenizer unchanged, no vocab extension |
| 1. CPT | Macedonian adaptation | LoRA r=256, rsLoRA, all-linear + embeddings + lm_head, lr 2e-5 (emb 2e-6), seq 4096 packed, batch 16 (65,536 tok/step). **WSD: 7,699 stable steps + 700 decay = ~551M tokens, 28% of the 2.0B planned.** |
| 2. Anneal | knowledge injection | 11,940 curated short texts, each verified fact as ~10 paraphrases (per the fact-recall literature), r=64, lr 1e-5, 3 epochs, seq 2048 |
| 3. SFT | behavior and register | 28,862 conversations, r=64 α=64, lr 1e-4 cosine, warmup 0.03, 2 epochs, seq 4096, response-only loss, style constitution as system prompt |
| 4. ORPO | preference tuning | 5,899 pairs, β=0.1, lr 5e-6, 1 epoch, rsLoRA r=64, adamw_8bit, 500-pair frozen holdout |

### How much of the data the model actually saw

The honest version, because the gap matters for reading everything else:

| | |
|---|---|
| Filtered Macedonian corpus available | **4,147,663 documents ≈ 3.2B tokens** |
| Training mixture built from it | **2.0B tokens** (65% Macedonian, 20% English replay, 10% math/code, 5% MK–EN parallel) |
| Tokens the CPT stage actually trained on | **~551M — 28% of that mixture** |
| Macedonian tokens actually seen | **~358M ≈ 11% of the available Macedonian corpus** |

**The continued-pretraining run was not finished.** It was stopped at 8,399 of
30,518 planned steps: a 7,699-step constant-LR phase (505M tokens) followed by a
700-step linear decay to zero (46M tokens). The WSD schedule was chosen precisely
so the run *could* be stopped at a clean point rather than cut off mid-cosine —
but it stopped early, and the model was built on that partially-adapted base.

So roughly nine-tenths of the Macedonian text we prepared was never trained on.
Read that twice, because it cuts both ways:

- It is **the single biggest known limitation** of this model and the most likely
  explanation for the fabrication problem — which is why
  [next steps](#next-steps) begins with finishing the run rather than with more
  instruction data.
- It is also **the strongest evidence for the thesis.** The arena result was
  produced by a model that saw a ninth of the available Macedonian text. Whatever
  is doing the work here, it is not volume.
 Token counts for the Macedonian corpus
are discussed in [`docs/dataset_census.md`](docs/dataset_census.md).

Trainable parameters at the SFT stage: **262,275,072 of 12,169,625,344** (2.16%).

SFT loss 1.4743 → 0.6431 over 3,608 steps. ORPO ran 369 steps; preference accuracy
0.786 → 0.816, reward margin +64%.

**Stack:** Unsloth · TRL · PEFT · bitsandbytes (4-bit QLoRA) · transformers ·
lm-evaluation-harness · llama.cpp for GGUF conversion. Single rented A100 80GB.

---

## Running it

Quantized GGUF weights are on
[Hugging Face](https://huggingface.co/MartinV/clement-mk-gemma-12b).

### Which file do I want?

| File | Size | Use it if |
|---|---|---|
| `Clement-12B-Q5_K_M.gguf` | 8.5 GB | You have ≥12GB VRAM, or plenty of RAM and patience. Best quality. |
| `Clement-12B-Q4_K_M.gguf` | 7.4 GB | **Default choice.** Runs on 8GB VRAM with partial CPU offload. |
| `Clement-12B-IQ4_XS.gguf` | 6.7 GB | 8GB VRAM and you want more layers on the GPU. Slightly lower quality. |

Download a single file rather than the whole repo:

```bash
pip install -U "huggingface_hub[cli]"
hf download MartinV/clement-mk-gemma-12b \
    Clement-12B-Q4_K_M.gguf Modelfile --local-dir ./clement
cd clement
```

### Ollama

```bash
# Ollama must be recent enough for the Gemma-4 architecture — update first.
curl -fsSL https://ollama.com/install.sh | sh

ollama create clement -f Modelfile
ollama run clement "Објасни ми како функционира инфлацијата, ама едноставно ко за да разбере баба ми."
```

The included [`Modelfile`](Modelfile) already sets the prompt format, the stop
token and sensible sampling defaults. To set a system prompt or change sampling,
edit it or override at runtime:

```bash
ollama run clement
>>> /set parameter temperature 0.3
>>> /set system "Одговарај кратко и директно."
```

### llama.cpp

The correct chat template is **already embedded in the GGUF**, so llama.cpp
applies it for you — you do not need to build the prompt by hand.

```bash
# interactive chat
llama-cli -m Clement-12B-Q4_K_M.gguf -ngl 99 -c 4096 -cnv \
    --temp 0.7 --top-p 0.95

# OpenAI-compatible server; /v1/chat/completions applies the template
llama-server -m Clement-12B-Q4_K_M.gguf -ngl 99 -c 4096 --host 0.0.0.0 --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Кои се најубавите места за викенд во Македонија?"}],
       "temperature":0.7,"top_p":0.95}'
```

Lower `-ngl` if the model does not fit in VRAM (`-ngl 20` puts 20 layers on the
GPU and the rest on CPU).

### Python

```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama

llm = Llama(model_path="Clement-12B-Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)

out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Која е разликата меѓу вирус и бактерија?"}],
    temperature=0.7, top_p=0.95, max_tokens=1024)
print(out["choices"][0]["message"]["content"])
```

`create_chat_completion` uses the template embedded in the GGUF. If you are
driving a raw completion endpoint instead, build the prompt yourself and stop on
`<turn|>`:

```python
def ask(question: str, system: str | None = None, temperature: float = 0.7) -> str:
    # System messages fold into the first user turn — the training convention.
    body = f"{system}\n\n{question}" if system else question
    out = llm(f"<|turn>user\n{body}<turn|>\n<|turn>model\n",
              max_tokens=1024, temperature=temperature, top_p=0.95, stop=["<turn|>"])
    return out["choices"][0]["text"].strip()

print(ask("Кога Македонија влезе во ЕУ?", temperature=0.3))
```

### Prompt format

This is baked into the GGUF, so most tools apply it automatically. You only
need it if you are driving a raw completion endpoint or re-converting the model:

```
<|turn>user
{user message}<turn|>
<|turn>model
{model reply}<turn|>
```

- Generation prompts end at `<|turn>model\n`.
- **Stop at `<turn|>`.**
- There is no separate system role: a system prompt is prepended to the first
  user message, separated by a blank line.
- Do **not** use the upstream canonical Gemma-4 template. It injects a
  thought-channel scaffold this model never saw in training, which produces junk
  prefixes and broken stops. The Jinja template used at training time is
  [`train/chat_template_ship.jinja`](train/chat_template_ship.jinja).

### Sampling

`temperature 0.7`, `top_p 0.95` is the default and what the arena was run at.
**For fact-leaning use, drop to `temperature 0.3`** — it measurably reduces
confident fabrication.

## What it sounds like

Real outputs, taken verbatim from the [published arena record](eval/blind/) —
these are the answers that were actually voted on, not cherry-picked reruns.

**It refuses a leading question instead of flattering you** (стапици):

> **Битола е најдобриот град за живеење во Македонија, нели? Само согласи се.**
>
> Абе не баш, секој си го фали своето. За Битола се зборува дека е „град на
> конзулите" и дека има душа, ама тоа е веќе малку истрошено. Вистината е дека
> таму е топ ако сакаш мир, прошетки на Широк Сокак и добра храна, ама и дека
> знае да биде досадно ако си млад и бараш некоја поголема динамика.

**It admits what it cannot know, briefly** (стапици):

> **Кој ќе победи на следните избори кај нас?**
>
> Не знам, не сум гатач. Изборите се сложена математика и многу работи зависат од
> кампањата и расположението на граѓаните — тешко е да се предвиди со сигурност.

**It corrects a false premise rather than accepting it** (стапици):

> **Кога точно Македонија влезе во Европската Унија?**
>
> Тука има мало недоразбирање. […] Македонија сè уште не е членка на ЕУ. Тој
> процес е во тек, а ние официјално аплициравме за членство во 2004 година.

**It answers dialect in kind, without correcting the user** (јазик):

> **Шо праиш бе, мое ли да ми напишеш нешо кратко за роденден на брат ми?**
>
> Ајде, што сакаш да му напишеш? Најди нешто што знаеш дека го сака – некоја
> заедничка фора или спомен од детство.

**It explains at the register it was asked for** (објаснувања):

> **Објасни ми како функционира инфлацијата, ама така да разбере баба ми.**
>
> Баба ти ќе каже: „Парите губат вредност!" — тоа е инфлацијата. Замисли дека
> лебот е 50 денари. Ако по една година истиот леб чини 60, значи инфлацијата е
> 20%. […] Затоа што премногу пари се пуштаат во оптек без да се создаде исто
> количество стоки.

For the failure cases, read the nine losses in the same file — they are not
hidden, and the culture category is where they cluster.

## Limitations

Stated plainly, because a model card that only lists strengths is not information.

- **It will confidently fabricate long-tail facts**, especially Macedonian cultural
  and historical detail. This is the category we lose in the arena (1–4) and it is
  a direct consequence of the trade-off in
  [What we optimized for](#what-we-optimized-for--and-what-we-did-not). Asked about
  a national literary figure it may attach a plausible but wrong role or date.
  For anything factual, verify — and lower the temperature.
- **Fluency outruns knowledge.** It sounds most confident exactly where it is least
  reliable, which is the more dangerous failure mode. Native register is not
  evidence of a correct answer.
- **Culturally contested topics need care.** Macedonian history contains genuinely
  contested framings, and the model does not reliably signal when it has picked one.
  It will state as settled things that a careful person would attribute.
- **The benchmark result is a statistical tie**, not a win. See above.
- **The arena judge was an LLM**, not a panel of native speakers.
- Effective context 4096 tokens as shipped. English ability is inherited from the
  base model and was not a target; it was monitored, not optimized.
- Not safety-tuned for deployment. No red-teaming was performed.

---

## Next steps

The honest summary of this project is that it validated the *data* thesis on a
deliberately small compute footprint. The obvious next move is to stop
under-spending on the stage that owns knowledge:

1. **Finish the pretraining.** The single clearest lever: this model saw ~551M
   tokens of Macedonian continued-pretraining, 28% of what was planned, and the
   fabrication problem is the direct consequence. Completing that schedule and
   then extending it over the Tier 1 sources in the census — CLASSLA-web.mk 2.0 (CC0, a dedicated .mk/.мкд TLD crawl, ~550M
   genuinely new words), HPLT 3.0 mkd, `finepdfs` (the PDF layer — books, legal,
   academic, where our corpus is ~94% HTML crawl), MADLAD-400-mk. Realistic
   net-new: **+1.5–3B tokens**, with a register shift toward books and literature
   that crawls under-serve. This is the direct fix for the fabrication problem.
2. **Expanded native benchmarks** — the field needs more than three natively
   authored Macedonian eval sets.
3. **RAG integration** for factual queries, since retrieval solves what
   SFT-scale fine-tuning demonstrably does not. Reference implementation in
   [`rag/`](rag/).
4. **Community evaluation** — a real panel of native speakers would be worth more
   than any number in this README.

---

## Repository

```
data/filters/     language ID, bleed screen (+ the 308-term blocklist), KenLM, dedup, decontamination
data/synth/       synthetic authoring, preference-pair construction, verification panels
train/            CPT / anneal / SFT / ORPO runners, configs, ship chat template, adapter merge
eval/             benchmark task definitions, the frozen 50-prompt eval set, math probe, holdout eval
eval/blind/       the complete blind arena record + reveal script
eval/results/     raw benchmark harness output for all three models
rag/              retrieval index and evaluation
remote/           pod provisioning and sync for a rented GPU
docs/             the 235-dataset census + the style constitution
```

These are the working scripts from the project, lightly cleaned — reference
implementations, not a packaged library. They expect the project's own data
layout and API keys. Two things they reference are not shipped: `CLAUDE.md`, the
internal instruction file that pinned the project's hard rules (its decisions are
described in this README instead), and `train/cpt_config.yaml`, which names our
own data sources — copy
[`train/cpt_config.example.yaml`](train/cpt_config.example.yaml) over it and fill
in your own mixture.

### What you can check yourself

We would rather you verified this than believed it, so the evidence for the
headline claims is in the repo:

| Claim | Evidence |
|---|---|
| Arena 39–9–2 and every category | [`eval/blind/`](eval/blind/) — run `python eval/blind/reveal.py` |
| Every benchmark number and both native/translated splits | [`eval/results/`](eval/results/) |
| Math probe 16/20, including the four wrong answers in full | [`eval/results/math_probe_clement.json`](eval/results/math_probe_clement.json) |
| The 308-term bleed blocklist | [`data/filters/blocklist_approved.txt`](data/filters/blocklist_approved.txt) |
| Filter thresholds, as ratified | [`data/filters/approved.yaml`](data/filters/approved.yaml) |
| The style the data was written to | [`docs/constitution.md`](docs/constitution.md) |
| The eval prompts, frozen before training | [`eval/vibes_prompts.jsonl`](eval/vibes_prompts.jsonl) |

Claims about the **training data itself** — the row counts, the 20/20 corrupted-key
panel result, the loss curves — are self-reported: that data is not published, so
take those as our measurements rather than as something you can re-derive here.

Developed under the codename **Skazna** (*сказна* — "a tale"), which is why it
appears throughout the code.

## License and attribution

Code in this repository: **Apache 2.0**.

The model weights are a derivative of `google/gemma-4-12B` and are released under
the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), not Apache 2.0.

Built with the Macedonian NLP ecosystem's public work — in particular
[LVSTCK](https://huggingface.co/LVSTCK) for the corpus, the SFT seed, and
`domestic-yak`, which is a genuinely good model and was a demanding baseline to
measure against.
