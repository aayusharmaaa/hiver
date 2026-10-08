# Reference: data contracts and pipeline details

Supporting detail for the [README](../README.md). Nothing here changes a result reported there.

## Input format (from the real `twcs.csv` header)

| column | observed type | notes |
|---|---|---|
| `tweet_id` | integer | unique |
| `author_id` | string | brand handle for outbound tweets, anonymised number for customers |
| `inbound` | `"True"`/`"False"` | True = customer |
| `created_at` | `Tue Oct 31 22:10:47 +0000 2017` | parsed to UTC |
| `text` | string | stored untouched |
| `response_tweet_id` | nullable, comma-separated ids | |
| `in_response_to_tweet_id` | nullable integer | |

If a required column is missing, the loader raises `SchemaError`. Extra columns trigger a warning. Rows with an invalid id or
`inbound` value are dropped and counted. Timestamps that can't be parsed become NaT and sort last in their thread.
`--nrows N` gives a fast debug run.

## How VirginTrains cases are built

A conversation is a connected component of the reply graph. Customer tweets belong to their author. Agent tweets belong to
the customer found by walking up the reply chain, then by `@mention`, then by the thread's only customer. A gap of more than
24 hours starts a new case, flagged `is_continuation`. Agent tweets that answer nobody (announcements) are kept as
`context_tweet_ids`. Some accounts flagged `inbound=True` are really other operators' agents (they sign off `^XX`); the raw
flag is kept and a derived `role` is used.

Each case carries:

- `case_id`, `brand`, `conversation_id`;
- `customer_messages`, `agent_messages`;
- `full_turns`: every turn with its tweet id, role, raw text, timestamp and source row;
- `turn_count`, `first_timestamp`, `last_timestamp`;
- `resolved`, `resolution_type`, `resolution_summary`, `resolution_evidence`;
- `dm_redirect`, `source_tweet_ids`.

Outputs: `data/processed/virgintrains_{conversations,tweets,cases}.parquet`. [`reports/virgintrains_eda.md`](../reports/virgintrains_eda.md)
has the numbers: 14,853 conversations, 65,810 tweets, 17,913 cases, 16% visibly resolved and 4% DM-redirected.

## Resolution memory

Only train-split cases with *strong* rule-based resolution evidence enter the primary memory: 10,380 of 14,213 train
episodes. The rest are kept out, with a reason:

| reason | cases |
|---|---|
| brand never replied | 1,210 |
| weak: other | 989 |
| weak: clarification only | 526 |
| weak: unresolved | 497 |
| continuation without an opening message | 376 |
| fallback intent `unclear` | 232 |
| uninformative | 3 |

## Brand ranking (phase 1)

Each metric is converted to a percentile among the eligible brands (at least 1,000 conversations and 85% Latin-script
openers), then combined with these weights:

| metric | weight |
|---|---|
| volume | 0.10 |
| reconstructable conversations | 0.15 |
| reconstructable rate | 0.05 |
| multi-turn density | 0.20 |
| template repeat rate | 0.15 |
| intent diversity | 0.15 |
| resolved rate | 0.10 |
| non-DM rate | 0.10 |

Top 3 of 78 eligible brands: AskAmex 0.718, Tesco 0.695, VirginTrains 0.688. The output is
`data/processed/brand_ranking.{csv,json}`.

## Golden labeling tool

[`scripts/label_golden_eval.py`](../scripts/label_golden_eval.py) is a small local web tool (127.0.0.1 only). It shows one
case at a time: the full conversation oldest-first and the candidate taxonomy, clearly marked *provisional*. **It never
shows model output**: no predicted intent, cluster, retrieved evidence, policy decision or draft reply. The labeler fills in:

- `gold_intent`: a candidate intent, or `NEW:<snake_case>` with a note if none fits;
- `gold_should_escalate`: yes or no, judged against what *this* AI can actually do;
- `gold_resolution_type`;
- `gold_confidence` (optional);
- `human_notes`.

Before every read and write the tool re-checks the sample's fingerprints, writes only the human columns, keeps an audit log,
and refuses to run if the CSV was edited by hand. Every save in the audit log records its `source` (`human`, or a confirmed
or corrected assistant draft). `python scripts/label_golden_eval.py --check` prints the counts.

The assistant drafts for orders 101–250 are in
[`virgintrains_golden_v1_assistant_drafts.csv`](../data/golden/virgintrains_golden_v1_assistant_drafts.csv) and were loaded with
[`scripts/import_golden_drafts.py`](../scripts/import_golden_drafts.py). The assistant saw only the conversation and the
taxonomy, never agent output, and nothing from the first 100 was changed. When a human opens a draft in the tool it shows a
banner; saving it unchanged records a confirmation, and editing it records a correction.

## Taxonomy-calibration path (built, optional, not used)

There is a 200-case calibration sample drawn from `golden_pool_reserve`, with a local labeling tool (the system suggestion
is hidden until you click Reveal), a comparison report and a guarded finalizer that writes a frozen taxonomy file. I chose
not to run it; the golden labels plus the taxonomy review ([`reports/golden_taxonomy_review.md`](../reports/golden_taxonomy_review.md))
cover the same need with one labeling pass. Scripts: `build_taxonomy_calibration.py`, `label_taxonomy_calibration.py`,
`compare_taxonomy.py`, `finalize_taxonomy.py`.

## Script index

| script | what it does |
|---|---|
| `profile_dataset.py`, `build_cases.py`, `rank_brands.py` | phase 1: profile all brands, rebuild threads, rank brands |
| `run_virgintrains_pipeline.py` | runs `build_virgintrains` → `discover_intents` → `prepare_splits` → `generate_taxonomy` → `virgintrains_report` |
| `build_resolution_memory.py`, `evaluate_retrieval.py` | memory + proxy retrieval evaluation |
| `evaluate_intents.py` | golden intent classification: majority class, TF-IDF + logistic regression, the agent's LLM classifier |
| `evaluate_agent.py` | end-to-end agent on a stratified 50-case golden slice: routing, generation, grounding, judge inputs |
| `judge_replies.py` | LLM-as-judge on the agent's draft replies, blind human rating sheet, judge-vs-human agreement |
| `analyze_failures.py` | failure-mode report from existing artifacts (no model calls) |
| `run_copilot_ui.py` | local Support Copilot UI over the agent-run results |
| `run_support_agent.py`, `smoke_test_agent.py` | the agent on one message / on 12 representative dev cases |
| `prepare_golden_eval.py`, `label_golden_eval.py`, `import_golden_drafts.py`, `golden_taxonomy_review.py` | golden pack, blind labeling, draft import, post-labeling taxonomy review |
| `build_taxonomy_calibration.py`, `label_taxonomy_calibration.py`, `compare_taxonomy.py`, `finalize_taxonomy.py` | optional taxonomy-calibration path (built, not used) |
