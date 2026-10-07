# Hiver SDE Intern take-home: data foundation

This repo is the data foundation of an AI customer-support agent built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Phase 1 turns raw tweets into cleaned tweets, reconstructs conversations, normalizes them into support cases, and ranks brands. Phase 2 (the section [VirginTrains pipeline](#virgintrains-pipeline-phase-2)) narrows to the chosen brand, builds episode-level support cases, discovers a *candidate* intent taxonomy, and prepares leakage-safe evaluation splits.

There is no LLM, agent, or UI yet. Extraction and resolution labels are deterministic and rule-based; intent discovery uses sentence embeddings + clustering only to propose candidates.

```
data/raw/twcs.csv
   │  ingestion.loader          validate schema, type-convert, dedupe, keep raw text + source_row
   ▼
tweets (cleaned)
   │  ingestion.reconstruction  connected components over reply links, ordered by timestamp
   ▼
data/processed/tweets_clean.parquet   (every tweet + conversation_id / parent / turn_index)
   │  ingestion.cases           one case per conversation, rule-based resolution labels
   ▼
data/processed/cases.parquet          (nested full_turns with full provenance)
   │  evaluation.brand_ranking  per-brand metrics -> percentile-weighted score
   ▼
data/processed/brand_ranking.{csv,json}
```

## Project layout

```
src/
  common/logging_utils.py       logging setup
  ingestion/schema.py           observed column contract + validation
  ingestion/loader.py           read CSV/Parquet, clean, summarize
  ingestion/reconstruction.py   thread reconstruction
  ingestion/resolution.py       rule-based resolution labels
  ingestion/cases.py            support-case builder (streams to Parquet/JSONL)
  ingestion/text.py             normalization helpers (never applied to stored raw text)
  taxonomy/keyword_intents.py   seed keyword intents (used only as a diversity proxy)
  evaluation/brand_ranking.py   brand metrics + scoring
  -- phase 2 (VirginTrains) --
  ingestion/roles.py            customer / brand_agent / other_agent roles (+ hidden-agent detection)
  ingestion/brand_filter.py     keep conversations the brand took part in
  ingestion/episodes.py         owner attribution + time-gap splitting -> one episode per customer
  ingestion/resolution_signals.py  regex signals with quoted evidence -> resolution_type
  ingestion/episode_cases.py    case records (schema, related cases, parquet/jsonl writers)
  ingestion/brand_pipeline.py   filter -> roles -> episodes -> cases -> output tables
  taxonomy/entity_masking.py    mask places/times/seats before embedding
  taxonomy/discovery.py         normalization, TF-IDF, embeddings, k sweep, KMeans
  taxonomy/taxonomy_builder.py  cluster report + draft taxonomy from curated labels
  evaluation/splits.py          leakage-safe train / dev / golden splits
  evaluation/sampling.py        deterministic coverage sampling
  evaluation/inspection.py      markdown rendering of cases
  -- retrieval prototype --
  ingestion/resolution_memory.py  support cases -> resolution episodes (problem, response, summary, provenance)
  retrieval.py                  BM25 + sentence-embedding + hybrid retriever with a soft candidate-intent signal
  evaluation/retrieval.py       proxy retrieval evaluation (Recall@k, MRR, stress test, examples)
  -- support agent --
  agent/                        schemas, classifier, risk policy, generator, grounding check, orchestration (support_agent.py)
  models/                       LanguageModel interface + the single Gemini REST client
configs/
  support_agent.yaml                 model settings and ALL risk-policy thresholds
  virgintrains_cluster_labels.yaml   hand-curated names for the clusters (matched by anchor terms)
  virgintrains_intents.yaml          GENERATED candidate taxonomy (everything uncertain = NEEDS_REVIEW)
scripts/
  profile_dataset.py            dataset profile -> stdout + data/processed/profile.json
  build_cases.py                reconstruction + cases -> data/processed/
  rank_brands.py                brand report -> data/processed/brand_ranking.{csv,json}
  build_virgintrains.py  discover_intents.py  prepare_splits.py  generate_taxonomy.py  virgintrains_report.py
  run_virgintrains_pipeline.py  runs the five phase-2 steps in order
tests/                          pytest unit tests (87)
data/raw/  data/processed/  data/golden/  reports/
```

## Setup

You need Python 3.10 or newer. These steps were tested on Python 3.11 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

To get the data, download `twcs.csv` from Kaggle into `data/raw/`. Either do it manually, or use `kagglehub`:

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'); shutil.copy(p + '/twcs/twcs.csv', 'data/raw/twcs.csv')"
```

The scripts add `src/` to `sys.path` themselves, so `pip install -e .` is optional.

## Expected input format

The input is a CSV (optionally `.csv.gz`) or a Parquet file with these columns. They come from the real `twcs.csv` header.

| column | observed type | notes |
|---|---|---|
| `tweet_id` | integer | unique in the full file |
| `author_id` | string | brand handle for outbound tweets, anonymized number for customers |
| `inbound` | `"True"`/`"False"` | True means customer, False means brand |
| `created_at` | `Tue Oct 31 22:10:47 +0000 2017` | parsed to UTC |
| `text` | string | stored untouched |
| `response_tweet_id` | nullable, comma-separated ids | e.g. `"9,6,10"` |
| `in_response_to_tweet_id` | nullable integer | |

A missing required column raises `SchemaError`. Extra columns trigger a warning and are then ignored. Rows that are invalid at the value level (a non-numeric id, or an `inbound` value other than True/False) are dropped and counted in the cleaning report. A timestamp that can't be parsed is kept as NaT, and that tweet sorts last within its thread.

## Commands

```bash
# 1. Profile the dataset (about 2.5 min on the full 2.8M rows)
python scripts/profile_dataset.py --input data/raw/twcs.csv

# 2. Reconstruct conversations and build support cases (about 2 min)
python scripts/build_cases.py --input data/raw/twcs.csv
#    only one brand's cases, plus JSONL for reading or labelling:
python scripts/build_cases.py --input data/raw/twcs.csv --brands AmazonHelp --jsonl

# 3. Rank brands (about 30 s; needs the outputs of step 2)
python scripts/rank_brands.py --top 15
#    tweak gates and weights:
python scripts/rank_brands.py --min-conversations 5000 --weights '{"multi_turn_density": 0.3}'

# Tests
python -m pytest -q
```

`--nrows N` (on the profile and build scripts) gives a fast debug run. It truncates the file, so some threads lose turns. For a quick smoke test, Kaggle also ships a 93-row `sample.csv`.

## Outputs

### `cases.parquet`: one row per conversation

| field | meaning |
|---|---|
| `case_id`, `conversation_id` | `case_<root_tweet_id>` and `conv_<root_tweet_id>` |
| `brand`, `brand_source` | brand handle; `agent_reply` (it replied), `mention` (customer @-mentioned it, no reply) or `unknown` |
| `brands_involved`, `customer_ids`, `customer_count`, `is_multi_party` | multi-party means more than one customer or more than one brand |
| `customer_messages`, `agent_messages` | raw texts in turn order |
| `full_turns` | list of turns with `tweet_id, author_id, role, inbound, created_at, created_at_raw, text, in_response_to_tweet_id, parent_tweet_id, response_tweet_id_raw, source_row` |
| `tweet_ids`, `turn_count`, `customer_turn_count`, `agent_turn_count`, `speaker_switches` | |
| `first_timestamp`, `last_timestamp`, `first_response_minutes` | |
| `is_reconstructable`, `is_multi_turn`, `has_missing_parent`, `starts_with_customer` | quality flags |
| `resolved`, `resolution_type`, `resolution_summary` | rule-based; see below |

Provenance is preserved at every stage. Each turn carries its original `tweet_id`, its raw text and timestamp string, the raw `response_tweet_id` string, and `source_row` (its 0-based row in the input file).

### Resolution labels (`ingestion/resolution.py`)

The labels describe how the public thread ended. They are weak labels, not ground truth.

| type | rule | resolved |
|---|---|---|
| `no_response` | no brand turn | no |
| `customer_confirmed` | the customer speaks last and thanks/confirms with no negation (e.g. "thanks for nothing" doesn't count) | yes |
| `customer_followup_unanswered` | the customer speaks last, with anything else | no |
| `agent_closed` | the agent speaks last with a closing phrase and no question | yes |
| `redirected_to_dm` | the agent's last turn asks the customer to DM | no |
| `redirected_to_channel` | the agent's last turn gives a URL, phone, email, or help-center pointer | no |
| `agent_awaiting_customer` | the agent asked a question and got no public reply | no |
| `agent_answered_unconfirmed` | the agent replied last, matching none of the above | no |

### Brand ranking (`evaluation/brand_ranking.py`)

Metrics are computed per brand. "Reconstructable" means one customer and one brand, with at least one turn from each.

| metric | default weight |
|---|---|
| `conversations` (volume) | 0.10 |
| `reconstructable_conversations` | 0.15 |
| `reconstructable_rate` | 0.05 |
| `multi_turn_density`: share of reconstructable cases with 3+ turns | 0.20 |
| `template_repeat_rate`: share of agent replies whose normalized 8-token prefix appears 5+ times (repeated resolution playbook) | 0.15 |
| `intent_diversity`: normalized entropy of seed-keyword intents of the first customer message | 0.15 |
| `resolved_rate` | 0.10 |
| `non_dm_rate` (1 − DM-redirect rate; DM'd cases hide the actual resolution) | 0.10 |

The score is the weighted mean of each metric's percentile rank among eligible brands. To be eligible, a brand needs at least 1,000 conversations and at least 85% of customer openers in Latin script. The text metrics use a fixed random sample of 5,000 per brand (seed 42), so big brands don't win just by having more text. The JSON report also includes `resolution_mix`, `top_intents`, and `top_agent_templates` for each brand, to help with the manual review.

## Example output (full dataset)

Here's the output of `profile_dataset.py`, abridged:

```
Total tweets:            2,811,774
Date range:              2008-05-08 -> 2017-12-03 (99.3% on or after 2017-10-01)
Unique authors / brands: 702,777 / 108
Inbound / outbound:      1,537,843 / 1,273,931  (ratio 1.2072)
Missing values:          response_tweet_id=1,040,629  in_response_to_tweet_id=794,335  (others 0)
% with in_response_to:   71.75%
% with response ids:     62.99%
% with any relationship: 100.0%
Refs to missing tweets:  parent=3,862  response=172,500
Conversations:           798,197   (single-tweet: 0)
Size mean/median/p90/max 3.52 / 2 / 6 / 1390
```

Here's the output of `build_cases.py`:

```
cases 798,197 | reconstructable 798,110 | multi-turn (3+ turns) 362,787
resolution_types: redirected_to_dm 248,585 | agent_answered_unconfirmed 197,811 |
  redirected_to_channel 140,668 | customer_followup_unanswered 102,998 |
  agent_awaiting_customer 39,838 | agent_closed 38,741 | customer_confirmed 29,556
```

Here's the output of `rank_brands.py --top 10` (78 of 108 brands are eligible):

```
rank brand            score  convs   recon  multi_turn  avg_turns resolved dm_rate template_rep intent_div
 1   AskAmex          0.718   4,286   4,084  0.776      4.57      0.280    0.023   0.308        0.736
 2   Tesco            0.695  16,659  15,564  0.673      4.01      0.105    0.315   0.049        0.774
 3   VirginTrains     0.688  14,813  12,984  0.564      3.74      0.185    0.033   0.036        0.784
 4   Uber_Support     0.686  41,891  40,157  0.332      2.86      0.043    0.593   0.574        0.871
 5   AmazonHelp       0.674  82,477  78,763  0.603      4.29      0.061    0.010   0.043        0.689
 6   XboxSupport      0.667  13,399  11,612  0.566      3.49      0.098    0.238   0.230        0.747
 7   BofA_Help        0.655   6,550   6,080  0.558      3.36      0.067    0.115   0.301        0.821
 8   sainsburys       0.652  10,832  10,108  0.535      3.58      0.158    0.305   0.061        0.766
 9   British_Airways  0.644  16,389  15,439  0.506      3.41      0.166    0.114   0.009        0.735
10   GWRHelp          0.642  10,676   9,577  0.605      3.80      0.158    0.019   0.009        0.737
```

## VirginTrains pipeline (phase 2)

```bash
pip install -r requirements.txt            # adds pyyaml, scikit-learn, sentence-transformers
python scripts/run_virgintrains_pipeline.py --input data/raw/twcs.csv     # about 5 min (GPU) / longer on CPU
python scripts/run_virgintrains_pipeline.py --from taxonomy               # after editing the cluster labels
python -m pytest -q
```

Steps (each is also a standalone script): `build_virgintrains.py` → `discover_intents.py` → `prepare_splits.py` → `generate_taxonomy.py` → `virgintrains_report.py`. Embeddings are cached in `data/processed/cache/`; every step is deterministic (seed 42), and a re-run reproduces the split files byte for byte.

| output | what |
|---|---|
| `data/processed/virgintrains_conversations.parquet` | one row per reconstructed thread, turns nested in chronological order with `case_id`/`role` per turn |
| `data/processed/virgintrains_tweets.parquet` | one row per tweet, with role, owner, case, exclusion reason, raw timestamp and response ids |
| `data/processed/virgintrains_cases.parquet` | one row per customer episode (the required case fields plus evidence, flags, related cases) |
| `data/processed/virgintrains_case_sample.csv`, `reports/virgintrains_case_inspection.md` | 450 coverage-sampled cases (train + dev only) and a readable rendering |
| `data/processed/virgintrains_intent_clusters.json`, `reports/virgintrains_intent_clusters.md` | per-cluster detail |
| `configs/virgintrains_intents.yaml` | candidate taxonomy (`CANDIDATE_NOT_GROUND_TRUTH`) until human calibration; see [Taxonomy calibration and freeze](#taxonomy-calibration-and-freeze-phase-3) |
| `data/processed/splits/*`, `data/golden/virgintrains_golden_candidates.*` | train/retrieval, dev/calibration, and 250 unlabeled golden candidates (blank `human_*` columns) |
| `reports/virgintrains_eda.md` | statistics, quality issues, recommended taxonomy |

How cases are built, in one paragraph: a conversation is a connected component of the reply graph. Within it each customer tweet belongs to its author, and each agent tweet to the customer found by walking up the reply chain, then by `@customer_id` mention, then by the thread's only customer. A gap above 24 h between a customer's consecutive tweets starts a new case, flagged `is_continuation` and linked to the previous one. Agent tweets that answer nobody (announcements) are kept as `context_tweet_ids` rather than turns. Some accounts flagged `inbound=True` are really other operators' agents (they sign off with `^XX`); the raw flag is kept and a derived `role` is used.

Resolution types come from regexes over the brand's turns, with the matching sentence quoted as evidence (`resolution_evidence`), and `resolved` additionally requires a positive thread ending. Nothing is generated or paraphrased.

**Leakage rules for the splits.** Cases that share a customer or a thread are grouped and the group is hashed to a split, so no customer appears in two splits. Clusters are fit without the golden pool. The golden set is stratified from that pool (not prevalence-faithful; use `golden_stratum_weight`), and any train/dev case whose opening text or context tweet matches a golden case is excluded. `verify_no_leakage` runs at the end of `prepare_splits.py` and fails the run on any overlap. Do not tune thresholds, prompts, or retrieval on the golden set.

## Taxonomy calibration and freeze (phase 3)

The candidate taxonomy is calibrated by a **human** on a sample of the `golden_pool_reserve` split, then frozen, and only then are the 250 golden cases prepared for labelling. There is no LLM anywhere in this phase and no label is ever generated by code.

```bash
# 1. Draw the 200-case calibration sample (reserve only, golden-adjacent cases excluded) + labelling pack.  Done; deterministic.
python scripts/build_taxonomy_calibration.py
#    -> data/processed/taxonomy_calibration.csv  (fill the human_* columns)
#    -> data/processed/taxonomy_calibration_labeling.md  (guidelines + every case, readable)
#    -> data/processed/taxonomy_calibration_guide.md     (the reviewer guide on its own; `--guide-only` regenerates it without touching the CSV)

# 1b. Label the 200 cases (human, local, no extra dependencies). Opens http://127.0.0.1:8765/
python scripts/label_taxonomy_calibration.py
python scripts/label_taxonomy_calibration.py --check   # verify files and print progress; starts nothing, writes nothing

# 2. After labelling: candidate vs human report (exits 2 until labels exist)
python scripts/compare_taxonomy.py
#    -> reports/taxonomy_calibration_report.md, data/processed/taxonomy_calibration_report.json, *_confusion_matrix.csv

# 3. Record decisions (merges / splits / renames / edits) in configs/virgintrains_taxonomy_decisions.yaml and sign it
python scripts/finalize_taxonomy.py --dry-run     # checks everything, writes nothing
python scripts/finalize_taxonomy.py               # writes HUMAN_CALIBRATED configs/virgintrains_intents.yaml + frozen configs/virgintrains_taxonomy_v1.yaml

# 4. Only now: 250 golden cases for human labelling (gold_* columns blank)
python scripts/prepare_golden_labeling.py          # -> data/golden/virgintrains_golden_v1.csv
python -m pytest -q
```

**Labeling tool.** A stdlib HTTP server bound to `127.0.0.1` (random per-run token, Host check, no new dependency). It shows one case at a time (conversation oldest-first, customer and agent visually distinct), progress `N / 200`, previous / next / next-unlabelled / jump, and a form for the five `human_*` fields. The system's candidate intent is **not** sent to the browser until you press *Reveal* (each reveal is logged), and the reference drawer lists all intents alphabetically with their definition and confusables. Save writes only the `human_*` cells of that row to `data/processed/taxonomy_calibration.csv`, atomically, after re-verifying the file; there is no auto-fill. `human_intent` must be a current intent or `NEW:<snake_case>` (with a note). Resume = start it again; it opens at the first unlabelled case. The audit trail is `data/processed/taxonomy_calibration_label_audit.jsonl` (timestamp, labeler, before/after, whether the suggestion was revealed first) and the previous save is kept as `taxonomy_calibration.csv.bak`. Before every read and write the tool checks that case ids, tweet ids, order and sampling columns still match a fingerprint in the manifest and that no case belongs to the golden set, and it refuses to touch a modified sample. It never reads or writes `data/golden/`. Stopping and restarting is safe at any point; two windows cannot run at once (the port is the lock).

Guards: the finalizer refuses unless at least 95% of calibration rows are fully labelled with valid values, the decisions file is signed (`reviewed_by`, `reviewed_on`), the golden set hash equals the one recorded when the sample was drawn, and the final registry has 8-12 intents plus the fallback with examples only from `train_retrieval`. A frozen taxonomy cannot be re-frozen; `content_sha256` detects any later edit. `prepare_golden_labeling.py` refuses without a verified frozen taxonomy, never alters the golden sampling (hash-checked), and never overwrites gold labels. The golden CSV adds `source_tweet_ids`, `conversation_id` and `golden_stratum_weight` after the requested columns for provenance, and shows no system-suggested intent.

Calibration sample design: 1,240 reserve cases, minus 128 that share a group with a golden case, 22 continuations and 8 that do not start with a customer, leaves 1,082 eligible. 200 are drawn with quotas that over-represent the confusable intents; within an intent, half are low-margin boundary cases and half are diverse coverage. `stratum_weight` re-balances within an intent only (the sample is not prevalence-faithful). Cases are presented in a shuffled order and the system suggestion is shown after the conversation to limit anchoring.

## Evidence-Grounded RAG

```text
Twitter conversations
        |
Support cases              (episodes per customer, rule-based resolution signals)
        |
Resolution memory          (problem + brand response + summary + provenance, train split only)
        |
BM25 + embeddings          (hybrid score, optional soft candidate-intent bonus)
        |
Historical evidence        (similar past cases and what the brand actually did)
```

The system retrieves **historical support resolutions** (what a customer asked, what the brand replied, how it ended), not generic documents. Nothing is generated yet: there is no LLM, escalation policy, or UI in this step.

> The taxonomy is currently a candidate taxonomy derived from data exploration; human taxonomy calibration is intentionally not used as a blocking dependency for this prototype. The candidate intent is used only as a soft retrieval signal, never as ground truth.

```bash
python scripts/build_resolution_memory.py   # -> data/processed/virgintrains_resolution_memory.parquet (+ manifest)
python scripts/evaluate_retrieval.py        # -> BM25 vs embeddings vs hybrid vs hybrid + intent; reports/virgintrains_retrieval_evaluation.{md,json}
```

```python
from retrieval import ResolutionRetriever

retriever = ResolutionRetriever.from_files("data/processed/virgintrains_resolution_memory.parquet",
                                           "data/processed/splits/virgintrains_split_assignments.csv",
                                           cache_dir="data/processed/cache")
results = retriever.search(query=customer_message, intent="service_status_delay_enquiry", top_k=5)  # intent is optional
```

Design notes:

- **Memory** has one record per train-split support case: `customer_problem` (what the customer wrote before the first brand reply, urls and mentions removed), `historical_response` (the brand reply that carried the resolution, original wording, sign-offs removed), a compact `resolution_summary`, `resolution_type`, `resolved`, `dm_redirect`, `escalation_signal` (what past agents *did*, not a policy), and the source/response tweet ids. Cases with no brand reply or only a weak reply (clarifying question, `other`, unresolved, fallback intent, continuation) are kept but flagged `in_primary_corpus = False` and are never retrieved.
- **Safety:** only `train_retrieval` cases are written to the memory and the retriever refuses any other split or any golden/reserve/dev id. Retrieval is deterministic (ties break on corpus order), uses no LLM, and has configurable `top_k`, `sem_weight` and `intent_weight`. Embeddings (all-MiniLM-L6-v2) are cached under `data/processed/cache/`.
- **Intent-aware retrieval** adds a small bonus to the normalised hybrid score when the candidate intent matches. It is not a filter, so strong cross-intent evidence can still win, and an unknown or missing intent simply means query-only retrieval.
- **Evaluation is a proxy**, not human relevance ground truth. Queries are non-golden `dev_calibration` cases; a historical case is "relevant" when it has the same *candidate* intent and the same rule-derived resolution type. Weights are tuned on one half of the queries and every reported number is on the other half. Golden cases are never used. See `reports/virgintrains_retrieval_evaluation.md` for the numbers, the wrong-intent stress test, and failure modes.

## Evidence-grounded support agent with risk-aware abstention

```text
message -> intent (Gemini, candidate taxonomy) -> retrieve similar past cases -> risk policy (deterministic)
        -> AUTO_HANDLE: Gemini reply from the evidence -> grounding check -> reply        (any failure -> ESCALATE, no reply)
        -> ESCALATE:    no reply, evidence and reasons handed to a human
```

The key goes in a git-ignored `.env` file at the repo root (copy `.env.example`), or in the shell environment, which takes precedence. `GEMINI_MODEL` optionally overrides the model name.

```bash
cp .env.example .env             # then set GEMINI_API_KEY=... in .env
python scripts/run_support_agent.py --message "The wifi on my train keeps dropping" [--context "earlier conversation"]
```

- The **historical resolution memory is the only evidence source**; retrieval happens before generation, and every returned case keeps its `case_id`, source tweet ids, resolution type and scores.
- The **candidate intent taxonomy is not human-validated ground truth**, so the classifier is a prediction of a candidate intent. Its confidence is the model's own, uncalibrated, estimate.
- The system **prefers escalation to unsupported automation**. `src/agent/policy.py` is deterministic and separate from Gemini: it auto-handles only when classifier confidence, retrieval similarity, the number of usable cases and their agreement all pass, and no trigger fires (multi-intent, low-information message, a never-auto intent, sensitive wording). Each decision lists its reasons. Gemini's grounding verifier can only turn AUTO_HANDLE into ESCALATE.
- All thresholds and the never-auto intents are in `configs/support_agent.yaml`. They are conservative starting values, not tuned or validated. Policy uses raw cosine similarity because the normalised hybrid score is always about 1.0 for the top result.
- The **golden evaluation set has not been used** for development. There is no evaluation of reply quality or escalation accuracy yet, and the Gemini calls have not been exercised against a live key. Tests use fakes (`pytest -q`).

## Live smoke test

A diagnostic that runs the **complete agent against the real Gemini API** on about 12 representative VirginTrains openers (status, booking, seat, Delay Repay, wifi, catering, complaint, praise, short message, multi-intent, boundary case, likely escalation). It checks that the pipeline works end to end and prints every decision for a human to read. It is **not the benchmark**: no accuracy or score is computed, and the historical resolution type and candidate intent it prints are weak metadata, not human ground truth.

```bash
# needs GEMINI_API_KEY in .env or the environment (see above)
python scripts/smoke_test_agent.py --limit 12 --seed 42     # --select-only lists the cases without calling Gemini
```

Cases come from `dev_calibration` only and are checked before any model call; golden, reserve and excluded cases are never used. Selection is rule-based and deterministic. Only the opening customer message is sent to the agent. Output goes to the terminal only; live outputs are deliberately not committed because they are model-dependent and are not evaluation ground truth.

## Assumptions and known limitations

- **VirginTrains specifics.** Taxonomy clusters are weak (silhouette about 0.05) and only loosely agree with a TF-IDF clustering; read `reports/virgintrains_eda.md` section 4 before relying on them. Resolution labels are English-only heuristics; `refund` means a refund was discussed (including refusals). The largest group of linked cases holds about 16% of all cases and therefore lands in a single split. `golden_pool_reserve` cases are held back and must not be used for retrieval or prompts.
- **What counts as a brand.** A brand is any author of an outbound tweet (`inbound == False`); there are 108. Inbound tweets don't record which brand they're addressed to, so a case's brand comes from whoever replied. If no brand replied, it falls back to an @-mention of a known handle.
- **What counts as a conversation.** A conversation is a connected component of the reply graph. Links come from both `in_response_to_tweet_id` and `response_tweet_id`. Branches, such as one customer tweet answered by two agents, stay inside a single conversation.
- **Dangling references.** Links to tweets that aren't in the file (172,500 response references and 3,862 parent references) are flagged, not treated as errors. The public file is a sample, so dangling ids are expected.
- **No unanswered tweets.** Every tweet in the public file has at least one reply link, which means there are no single-tweet conversations in the data. As a result, `no_response` cases essentially never occur on the full file. That says something about how the dataset was sampled, not about the brands.
- **Multi-party threads.** 54,642 conversations involve more than one customer, and the largest has 972 customers and 1,390 tweets (many people replying under a single brand tweet during an outage). They're flagged with `is_multi_party` and excluded from the ranking's reconstructable metrics, but they aren't split apart yet.
- **Resolution labels are heuristics.** They're English-only regexes, and "resolved" here means only "visibly resolved in public". Many brands resolve issues in DMs, so their true resolution rate is higher than these labels suggest.
- **Seed intents are a proxy.** The keyword intents exist only to compare intent diversity between brands. About 25–45% of openers fall into `other`. This lexicon is not the agent's taxonomy.
- **The Latin-script gate is not a language filter.** It doesn't detect language: Safaricom_Care (Swahili/English) and idea_cares (Hinglish) pass it. Check language manually for any brand you shortlist.
- **Caveats on the top rank.** AskAmex is first partly because it sends templated follow-ups ("Hi, I never heard back from you…"). Those inflate both its multi-turn density and its template repetition, and it has about 4k conversations, the lowest volume in the top 10.
