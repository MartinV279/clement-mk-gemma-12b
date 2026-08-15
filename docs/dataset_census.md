# Macedonian dataset census — 2026-07-31

A parallel sweep across eight sources (HF Hub, CommonCrawl derivatives, OPUS,
instruction/preference sets, institutional archives, wiki/community, speech+eval,
plus a completeness critic pass). **235 datasets cataloged**, all sizes read from
dataset cards / API size endpoints, all URLs HTTP-verified, July 2026.

This is the audit the whole project grew out of: it is what answers "what is
Macedonian actually being modeled from?" — and the answer, in Tier 3 and the
SFT section below, is *mostly machine-translated English*.

**What this document is:** the decision-useful summary of that catalog, not the
catalog itself. It names the ~25 sources that changed a decision — everything
worth ingesting, everything deliberately rejected, and why. The remaining
entries were screened and dropped as redundant, unlicensed, or too small to
matter, and the raw catalog is not published.

**On the corpus figures below:** these are read from the LVSTCK dataset card and
describe the *source* corpus as published, before our filtering. What we
actually trained on is the filtered survivor set — 4,147,663 documents of the
4,341,767 we started from.

## What our corpus actually contains (provenance, from the LVSTCK card)

`LVSTCK/macedonian-corpus-raw` = **HPLT-2 (42.2%) + FineWeb-2 (37.7%) +
MaCoCu-mk 2.0 (13.9%) + university-PDF slice "MMORE" (4.1%) + Wikipedia (2.0%)
+ SETimes + Common Voice**. So HPLT-2, FineWeb-2, MaCoCu and mk.wikipedia are
already inside our 1.5B tokens — re-ingesting them buys nothing.
`LVSTCK/sft-mk` already absorbs alpaca-mk, ultrachat-sft-mk (16k), Capybara-mk,
dolly-mk, Platypus-MK — those are spent too.

## Tier 1 — big net-new raw text (the CPT case)

| Source | Size | License | Why it's new |
|---|---|---|---|
| **CLASSLA-web.mk 2.0** (CLARIN.SI 11356/2079) | 691M words / 2.11M texts | **CC0** | *Highest value-density find.* Dedicated **2024 crawl of the .mk/.мкд TLDs** — not CommonCrawl. LVSTCK's only TLD input was the 2021 MaCoCu crawl (= CLASSLA-web 1.0); the 2.0 paper measures **79.9% of mk texts unique vs 1.0** → ~550M genuinely new words. Ships genre + IPTC topic labels. |
| **HPLT 3.0 mkd_Cyrl** | 6.79M docs / 5.93B tokens | CC0 | LVSTCK used HPLT-2 (3.57M docs); 3.0 is ~1.9×. Same crawl lineage → heavy dedup mandatory, but the residual is plausibly the largest new pool anywhere. Data at data.hplt-project.org (HF repo is a pointer). |
| **HuggingFaceFW/finepdfs mkd** | 166k docs / 3.29GB (~20KB/doc) | ODC-BY | **PDF-layer extraction** — books, legal, gov reports, academic. Our corpus is ~94% HTML crawl; LVSTCK's PDF slice was only 1.48GB. Different pipeline → low overlap. `finepdfs-edu` (18k docs / 0.62GB) is the quality-filtered subset. |
| MADLAD-400 mk clean | 1.4M docs / ~4.5B SP tokens | ODC-BY | Different filter chain than FineWeb-2 → est. 0.3–0.6B net-new words. Authors admit audits by non-speakers — needs our mk-vs-bg LID + bleed screen re-pass. |
| GlotCC-V1 mkd | 335k docs / 1.22GB | CC0 | CC-derived, moderate overlap; cheap to dedup-test. |

Quality tooling for the above: **JQL-AI/hplt2_edu_scores + fw2_edu_scores**
(doc-level edu-quality scores for 7.7M/8.9M mkd docs) and
**LumiOpen/hpltv2-llama33-edu-annotation** (500k mkd docs WITH text + Llama-3.3-70B
quality labels) — a free quality filter / classifier training set for any HPLT ingest.

## Tier 2 — register gaps (small but irreplaceable)

| Source | Size | Notes |
|---|---|---|
| procesaur/**Wiki.mk** | 210k records, refreshed 2026-05 | wikipedia + **wikisource + wikibooks** splits pre-cleaned; the latter two (public-domain literature, textbooks) are absent from our corpus. CC-BY-SA. |
| mk.**wikisource** dump | 3,371 texts / 1.3M words | Edited, native, public-domain literary prose. Tiny; upsample. |
| **MANU Electronic corpus of MK literary texts** | 16MB plain text | Digitized "135 volumes of Macedonian literature". License unstated — ask MANU before use. Koneski collected works also digitized (CC BY-NC, 13 vols PDF). |
| **opendata.sobranie.mk** | 29 datasets, plenary 2020–2024 JSON | Parliament proceedings, portal states CC0. Formal-register native speech-in-text. Sobranie stenographic PDF archive goes back to 1991 (no license stated). |
| OpenSubtitles v2024 mk mono | 10.6M sentences / 67.8M words | The conversational register we lack. **License unclear** (OPUS redistribution of opensubtitles.org) — usable for research CPT, flag for any HF release. |

## Tier 3 — parallel (we used 2 sources; the catalog has ~15)

- **HPLT v3 parallel en-mk: 5.82M pairs, CC0** — the standout; dwarfs MaCoCu+SETIMES.
- opus-100 en-mk 1.0M · OPUS wikimedia 695k · OpenSubtitles en-mk 5.0M (license ~) · CCMatrix 12.0M (noisy, mined)
- Clean small: GlobalVoices 55k (CC-BY) · QED 68k · Tatoeba 81k · TED2020 44k (NC-ND — skip for release)
- Neighbor transfer: SETIMES bg-mk/hr-mk/el-mk ~200k each.

## SFT / preference — the honest picture

Native MK instruction data barely exists. The MT pile is a trap:
**Aya macedonian (4.12M rows) is broken MT** — sampled rows show mangled
enumerations ("2." → "Да се подготвиме", "No" → "Не е така"). Do not touch.
Same class: xP3x, Bactrian-X, saillab taco. **Helsinki fineweb-edu-translated
mkd (151GB!)** and nemotron-cc-translated (218GB) are Marian-MT output —
translationese at scale, exactly what the constitution bans.

Worth taking:

| Dataset | Size | Verdict |
|---|---|---|
| **PaDaS-Lab/webfaq-v2 mkd** | 46,388 QA pairs | **Native human-written** FAQ from schema.org markup on real MK sites. Domain-skewed (commercial/gov) — flavour additive, not a base. |
| milanvelinovski/mk-ultrachat42k | 42.8k multi-turn | MT but Apache-2.0 and ~26k conversations additive over sft-mk's UltraChat slice. Multi-turn is our thinnest register. Sample before committing. |
| ilijalichkovski makstat family | ~1k QA | Human-refined QA on MK state statistics. No license on refined repo — ping author. |
| medical-o1-mk 1,068 · gsm8k_mk 7,473 · lr-sum 2,223 | small | Niche coverage; gsm8k_mk contaminates GSM8K-style evals — track it. |
| **haoranxu/X-ALMA-Preference** | 5,988 mk triples, MIT | **The only MK preference data in existence** — translation-quality only. Confirms our own 5,686 generated pairs are the necessary path for ORPO. |

## Evals — the fact-probe ceiling fix

Three **natively-authored** benchmarks exist beyond the MT'd LVSTCK ports —
add to held-out suite, keep out of all training data + decontamination lists:

1. **mhardalov/exams crosslingual_mk — 2,075 real MK high-school/entrance exam questions** (native, curriculum knowledge; `with_para` variant for open-book)
2. **classla/COPA-MK — 1,000** professionally human-translated (causal commonsense)
3. **CohereLabs/include-base-44 "North Macedonian" — 571** natively-authored regional-knowledge questions

These directly solve the fact-probe headroom problem (base already at 68% on our
25 items) and defend against "scores well on MT benchmarks, doesn't know Macedonia".

## Dead ends checked

No ParlaMint-MK release · classla/mak_na_konac is HR/SR speech despite the name ·
no MK legal corpus on HF beyond LVSTCK's gazette scrape · Projekat Rastko is
©-reserved · Knigoteka states all rights reserved · no MK DPO/chat-preference
data anywhere.

## Implications — the roadmap this produced

Nothing was changed mid-chain; these were the branches identified for after the
first arena, and they remain the next steps:

- **An expanded CPT mixture**: CLASSLA-web.mk 2.0 + finepdfs + HPLT3-delta
  + MADLAD-delta, MinHash-deduped against the current corpus, same filter
  chain (bleed screen, KenLM, LID, vibes decontam **+ the 3 new eval sets**).
  Realistic net-new MK: **+1.5–3B tokens — could double the MK pool**, with a
  register shift toward books/legal/literature the crawls under-serve.
- **Further SFT candidates**: webfaq native QA + sampled ultrachat42k + makstat.
- **Either way**: port EXAMS-mk / COPA-MK / INCLUDE-mk into `eval/lm_eval_tasks/`
  — cheap, high-signal, and usable for the head-to-head table immediately.
    (This was done: all three are in the published benchmark battery.)
