# An evidence-grounded support agent for VirginTrains

This is my take-home for the Hiver SDE Intern role. The brief asks for an AI support agent for one brand from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. The agent should:

1. classify incoming messages into intents defined from the data;
2. draft replies grounded in how the brand actually resolved similar issues;
3. decide whether to auto-handle or escalate, with a reason.

The brief also asks me to prove the agent can be trusted.

I picked **VirginTrains**. The system is built as an *evidence-grounded support agent with risk-aware abstention*: it only
replies when it has strong historical evidence, and otherwise it hands the case to a human with its reasons.

> **Where things stand (honestly).** The data pipeline, the intent taxonomy, retrieval, the agent and a blind labeling tool for
> the golden set are built and tested (`pytest -q`: 490 passed). Retrieval has a proxy evaluation with baselines. The 250-case
> golden set is sampled, frozen and leakage-checked. **100 cases are hand-labelled; the other 150 are AI-assistant drafts
> awaiting human review** (see [Golden evaluation set](#golden-evaluation-set)). The end-to-end evaluation harness and the
> LLM-as-judge are **not built yet**; I didn't want to build them before the labels were final. The
> [deliverables table](#deliverables-vs-the-brief) shows exactly what is done and what isn't.

**Contents**

- [Quick start](#quick-start)
- [Deliverables vs the brief](#deliverables-vs-the-brief)
- [Problem framing: what "good" means here](#problem-framing-what-good-means-here)
- [Architecture](#architecture)
- [Results so far](#results-so-far)
- [What is misleading about my headline number?](#what-is-misleading-about-my-headline-number)
- [Failure modes I already know about](#failure-modes-i-already-know-about)
- [Golden evaluation set](#golden-evaluation-set)
- [Decision log](#decision-log)
- [What I'd do next with one more week](#what-id-do-next-with-one-more-week)
- [Repo map](#repo-map)
- [Reference: data contracts and pipeline details](#reference-data-contracts-and-pipeline-details)
- [Assumptions, limitations and credits](#assumptions-limitations-and-credits)

---

## Quick start

You need Python 3.10+. I tested on 3.11 on Windows.

```bash
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                      # no data or API key needed; about 1 minute
```

**Reproduce the headline retrieval numbers.** You need `twcs.csv` (2.8M tweets) in `data/raw/`.

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'); shutil.copy(p + '/twcs/twcs.csv', 'data/raw/twcs.csv')"
python scripts/run_virgintrains_pipeline.py --input data/raw/twcs.csv   # cases, intents, splits, EDA (about 5 min on a GPU, longer on CPU)
python scripts/build_resolution_memory.py                                # historical resolution memory (train split only)
python scripts/evaluate_retrieval.py                                     # -> reports/virgintrains_retrieval_evaluation.md
```

Every step is seeded (42), so a re-run reproduces the split files byte for byte. Embeddings are cached in `data/processed/cache/`.

**Talk to the agent.** This needs a Gemini key. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The `.env` file is git-ignored.

```bash
python scripts/run_support_agent.py --message "The wifi on my train keeps dropping"
python scripts/smoke_test_agent.py --limit 12 --seed 42        # 12 representative dev cases end to end; --select-only skips Gemini
```

**Label the golden set.** This is local and blind, and opens http://127.0.0.1:8766/.

```bash
python scripts/label_golden_eval.py            # --check verifies the files and prints progress
```

---

## Deliverables vs the brief

| brief asks for | status | where |
|---|---|---|
| Runnable pipeline, reproducible in under 15 min | ✅ built (raw tweets → cases → taxonomy → splits → memory → agent); the main pipeline took about 5 min on my GPU, CPU is slower | [Quick start](#quick-start), [`scripts/run_virgintrains_pipeline.py`](scripts/run_virgintrains_pipeline.py) |
| Intents defined from the data | ✅ 10 candidate intents plus a fallback, still marked *candidate* (not human-validated) | [`configs/virgintrains_intents.yaml`](configs/virgintrains_intents.yaml), [`reports/virgintrains_eda.md`](reports/virgintrains_eda.md) §4 and §8 |
| Grounded reply drafting | ✅ built and unit-tested; the live run was cut short by the Gemini free-tier quota | [`src/agent/generator.py`](src/agent/generator.py), [`src/agent/grounding.py`](src/agent/grounding.py) |
| Auto-handle vs escalate, with a reason | ✅ deterministic policy; every decision lists its reasons | [`src/agent/policy.py`](src/agent/policy.py), [`configs/support_agent.yaml`](configs/support_agent.yaml) |
| Golden set of 150–250 hand-labelled examples | ⚠️ 250 cases sampled, frozen and leakage-checked; **100 hand-labelled, 150 AI-assistant drafts pending human review** | [Golden evaluation set](#golden-evaluation-set) |
| Evaluation harness: metrics, LLM judge, judge-vs-human agreement | ⚠️ retrieval harness done; **agent harness and LLM judge not built yet** (they need the golden labels) | [`src/evaluation/retrieval.py`](src/evaluation/retrieval.py) |
| Results vs a trivial and a simple baseline | ⚠️ done for retrieval (random, BM25, embeddings, hybrid); agent-level baselines pending | [Results so far](#results-so-far) |
| Top 5 failure modes | ⚠️ retrieval and data failure modes documented below; agent failure modes need the golden run | [Failure modes](#failure-modes-i-already-know-about) |
| "What is misleading about my headline number?" | ✅ | [below](#what-is-misleading-about-my-headline-number) |
| Decision log (10–15 items) | ✅ | [Decision log](#decision-log) |

---

## Problem framing: what "good" means here

VirginTrains customers tweet about a few recurring things: is my train running, the train is late or packed, Delay Repay,
booking and seats, the wifi, first-class catering, and a lot of praise and banter.
**Many of those cannot be answered safely from history alone.** "Is the 17:30 to Euston cancelled?" needs live data, and a
refund needs an account lookup. A reply that sounds confident but is wrong is worse than no reply at all.

So for this brand I define **good** as:

1. **Never be confidently wrong.** If the agent auto-handles a message, the reply must be supported by what VirginTrains
   actually said in similar past cases: no invented links, amounts, policies or "I've refunded you".
2. **Escalate the right things.** Live status, money, complaints about staff, safety and accessibility go to a human, with
   a reason a human can act on.
3. **Automate only the boring, safe part.** That means general information, self-service links, wifi steps and thanking
   people. That's where the historical replies are repetitive and safe to reuse.

On this dataset, precision on auto-handled messages matters far more than coverage. I would rather escalate 70% of traffic
and be right on the other 30% than the reverse.

**What I chose not to build.** There is no frontend beyond the local labeling tool, and no Hiver API integration, database,
Docker or multi-agent setup. There is no reranker or fine-tuning, and the agent has no live train data or account access.
That missing access is exactly why the policy escalates those intents.

---

## Architecture

```text
 twcs.csv (2.8M tweets, 108 brands)
   │  ingestion/loader + reconstruction       validate schema, dedupe, rebuild reply threads (provenance kept)
   ▼
 brand ranking ──► VirginTrains (manual pick)  evaluation/brand_ranking.py
   │  ingestion/brand_pipeline                 roles → episodes (one per customer, 24h gap) → support cases
   ▼
 17,913 support cases ──► rule-based resolution signals with quoted evidence (no LLM)
   │
   ├─► taxonomy/discovery     entity-masked MiniLM embeddings + KMeans (k=12) → 10 candidate intents + fallback
   ├─► evaluation/splits      customer/thread-grouped hashing → train 14,213 · dev 2,210 · golden 250 · reserve 1,240
   ▼
 resolution memory (train only)   problem · what the brand replied · resolution type · tweet ids
   │
   ▼
 ┌──────────────────────────── support agent (src/agent/support_agent.py) ───────────────────────────┐
 │ message → Gemini classifier (candidate intents, JSON, temp 0)                                     │
 │         → hybrid retrieval (BM25 + embeddings + soft intent bonus, top 5)                         │
 │         → deterministic risk policy (confidence, similarity, evidence count/agreement, triggers)  │
 │             ├─ ESCALATE    → no reply; reasons + evidence for the human                           │
 │             └─ AUTO_HANDLE → Gemini reply from evidence → grounding check → reply                 │
 │                                    (link/amount check + Gemini verifier; any failure → ESCALATE)  │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

A few things about how it fits together:

- **The evidence is the brand's own history.** Retrieval returns past *resolutions* (what the customer asked, what
  VirginTrains replied, how it ended), not generic documents. Every evidence item keeps its `case_id` and source tweet ids, so
  any reply can be traced back to real tweets.
- **The policy is deterministic and separate from the LLM.** Gemini classifies and drafts; plain code decides. All
  thresholds live in one file, [`configs/support_agent.yaml`](configs/support_agent.yaml).
  - Five intents never auto-handle: status, disruption complaints, Delay Repay, service complaints and unclear messages.
  - Sensitive wording also forces escalation: legal, safety, accessibility, theft, refunds.
- **Grounding can only make the agent more cautious.** A deterministic check rejects any link or £ amount that isn't in the
  evidence. A Gemini verifier then lists unsupported claims. If either fails, or the verifier can't run, the decision becomes
  ESCALATE. Nothing can turn an ESCALATE into an AUTO_HANDLE.
- **One model wrapper.** All Gemini calls go through [`src/models/gemini.py`](src/models/gemini.py), which uses plain REST
  with timeouts, retries and JSON mode, behind a small `LanguageModel` interface. Tests use fakes and never need a key.

Pointers into the code:
[`classifier.py`](src/agent/classifier.py) ·
[`retrieval.py`](src/retrieval.py) ·
[`policy.py`](src/agent/policy.py) ·
[`generator.py`](src/agent/generator.py) ·
[`grounding.py`](src/agent/grounding.py) ·
[`schemas.py`](src/agent/schemas.py) (`Classification`, `Evidence`, `Decision`, `GroundingResult`, `AgentResult`).

---

## Results so far

### Retrieval (the headline number, for now)

Retrieval is the part I could measure without human labels. The setup:

- **Queries:** 857 held-out `dev_calibration` cases.
- **Corpus:** 10,380 train-split resolutions.
- **Relevance:** a retrieved case counts as relevant if it has the same candidate intent *and* the same rule-derived
  resolution type.
- **Tuning:** weights were tuned on a separate half of dev queries.

Full report: [`reports/virgintrains_retrieval_evaluation.md`](reports/virgintrains_retrieval_evaluation.md).

| strategy | R@1 | R@5 | MRR (95% CI) |
|---|---|---|---|
| Random (trivial baseline) | 0.053 | 0.237 | 0.152 |
| BM25 (simple baseline) | 0.223 | 0.551 | 0.372 (0.349–0.396) |
| Embeddings (MiniLM) | 0.287 | 0.616 | 0.443 (0.418–0.470) |
| Hybrid (0.9 semantic) | 0.299 | 0.623 | 0.445 (0.420–0.472) |
| **Hybrid + soft intent bonus (0.15)** | **0.407** | **0.784** | **0.572 (0.547–0.598)** |

**Stress test.** If 30% of the predicted intents are wrong, the soft bonus still beats query-only retrieval (MRR 0.519 vs 0.445).
A hard intent filter drops below it (0.404). That's why the intent is only a nudge and never a filter.

### Agent

- **Unit level:** fully tested with fakes. Those tests cover:
  - policy paths;
  - multi-intent and weak-evidence escalation;
  - grounding failure forcing escalation;
  - provenance flowing through end to end;
  - behaviour with no API key.
- **Live:** a 12-case smoke test on `dev_calibration` showed classification, retrieval and policy escalation working against
  real Gemini. Cases that passed the policy reached generation. The free-tier quota ran out before a full run, so
  **I have not yet observed AUTO_HANDLE with a passing grounding check live**.
- **No accuracy, escalation precision or reply-quality numbers yet.** Those come from the golden set.

---

## What is misleading about my headline number?

The 0.572 MRR / 0.784 Recall@5 above looks good. Here is why you shouldn't read too much into it:

1. **It's partly self-confirming.** The "+ intent" strategy uses the candidate intent, and so does the relevance rule.
   Retrieval gets rewarded for agreeing with the same clustering that defines "relevant". On intent-only relevance that row
   jumps to R@1 0.953, which mostly says the bonus works, not that the evidence is useful.
2. **The labels behind it are heuristics.** Both the intent (an unsupervised cluster, silhouette ≈ 0.05) and the resolution
   type (regexes over the brand's reply) are machine-made. No human has judged a single retrieved case as useful.
3. **"Relevant" isn't "helpful".** A past reply with the same intent and resolution type can still be wrong for this
   customer, for example a different route or a disruption that has since ended.
4. **It measures retrieval, not the agent.** A perfect top 5 still has to get through the policy, the generator and the
   grounding check. The number the brief actually cares about is how often an auto-handled reply is right, and that doesn't
   exist yet.
5. **It's seven weeks of 2017 tweets.** Almost all the data is from Oct–Dec 2017, so recurring disruption templates are
   over-represented and there is no test of how it holds up over time.

---

## Failure modes I already know about

These come from retrieval and data analysis. The agent-level top 5 will come from the golden run.

1. **Right topic, wrong handling.** 178 of the 185 retrieval misses found the same intent but a different resolution type.
   Example: "can I use Virgin from Watford with this??" was historically a *refund*, but retrieval brought back "next train"
   answers. *Hypothesis:* the opening message doesn't contain what decides the handling (ticket type, refund eligibility).
2. **Sarcasm reads as praise or chitchat.** "on what planet can you justify this price? … hand-waited on by Tom Hardy" is a
   fare complaint, but it clustered as `chitchat_non_support`. *Hypothesis:* short-text embeddings miss irony; the
   classifier should do better, and the golden labels will show whether it does.
3. **Mixed messages.** "I appreciate your 2-hour service… sort out your air conditioning" was filed as praise. *Hypothesis:*
   one opener with two intents; the policy's multi-intent trigger is the safety net.
4. **Invisible resolutions.** About 4% of cases move to DM, and whatever was resolved there is invisible to us. Refunds and
   complaints are therefore under-represented as resolved and over-represented as "unresolved".
5. **Routes instead of intents.** Before I masked places and times, the clusters split by route (Euston–Manchester vs
   London–Glasgow) rather than by what people wanted. Masking fixed it, but residual entity bias is likely.

---

## Golden evaluation set

**Sampling.** I took 250 cases from a held-out golden pool, stratified by:

- candidate cluster;
- resolution type;
- conversation length;
- DM redirect;
- resolved.

That deliberately over-represents rare and hard cases; `golden_stratum_weight` restores natural prevalence.
The pool is separated *before* clustering, and splits are assigned by grouping customers and threads. So no golden customer,
conversation, thread, tweet or near-duplicate opener appears in train or dev. Train or dev cases that copied a golden opener
or context tweet are dropped. [`evaluation/splits.py`](src/evaluation/splits.py) checks all of this, and it currently passes
with zero overlaps.

**Labeling.** [`scripts/label_golden_eval.py`](scripts/label_golden_eval.py) is a small local web tool (127.0.0.1 only). It
shows one case at a time: the full conversation oldest-first and the candidate taxonomy, clearly marked *provisional*.
**It never shows model output**: no predicted intent, cluster, retrieved evidence, policy decision or draft reply. The
labeler fills in:

- `gold_intent`: a candidate intent, or `NEW:<snake_case>` with a note if none fits;
- `gold_should_escalate`: yes or no, judged against what *this* AI can actually do;
- `gold_resolution_type`;
- `gold_confidence` (optional);
- `human_notes`.

Before every read and write the tool re-checks the sample's fingerprints, writes only the human columns, keeps an audit log,
and refuses to run if the CSV was edited by hand.

**Status and provenance.** Not every label came from a human, and the files say so:

- **Orders 1–100 are hand-labelled** in the blind tool.
- **Orders 101–250 are AI-assistant drafts.** To save time, an AI coding assistant read those conversations and drafted labels
  following the conventions in the first 100. They are in
  [`virgintrains_golden_v1_assistant_drafts.csv`](data/golden/virgintrains_golden_v1_assistant_drafts.csv) and were loaded
  with [`scripts/import_golden_drafts.py`](scripts/import_golden_drafts.py). The assistant saw only the conversation and the
  taxonomy, never agent output, and nothing from the first 100 was changed. A draft is **not** a human label.
- Every save in the audit log records its `source`. When a human opens a draft in the tool it shows a banner; saving it
  unchanged records a confirmation, and editing it records a correction. `python scripts/label_golden_eval.py --check`
  prints the counts (right now: 100 human, 150 unreviewed drafts).
- Reviewing a draft is weaker than labeling blind, because the draft can anchor the reviewer. So results will be reported
  twice: on the 100 human-labelled cases alone, and on all 250 together with how many drafts were confirmed or corrected.

The golden set has not been used for any tuning: not retrieval weights, prompts, thresholds or the taxonomy. When labeling is
finished,
[`scripts/golden_taxonomy_review.py`](scripts/golden_taxonomy_review.py) compares candidate intents against human ones.

---

## Decision log

1. **VirginTrains over higher-ranked brands.** It ranked 3rd of 78 eligible brands, but its replies happen *in public*: only
   3.3% go to DM, against 31.5% for Tesco. You can't learn resolutions you can't see. AskAmex ranked 1st mostly because of
   templated follow-ups.
2. **One case per customer episode, not per thread.** Busy VirginTrains threads mix several customers and other operators'
   agents. I attribute each tweet to a customer through the reply chain and @mentions, and treat a gap of more than 24 hours as
   a new episode.
3. **Rule-based resolution labels with quoted evidence, no LLM.** These are weak labels, but every one points to the exact
   sentence that triggered it, so they're auditable and nothing is paraphrased.
4. **Mask entities before clustering.** Without it the clusters were train routes, not intents.
5. **Taxonomy stays a *candidate*.** Clusters are soft (silhouette ≈ 0.05), so I marked uncertain fields `NEEDS_REVIEW` and
   never called them ground truth.
6. **Skipped human taxonomy calibration on purpose.** I built a 200-case calibration workflow, then fast-forwarded past it to
   get an end-to-end agent. The golden labels allow `NEW:` intents and feed a taxonomy review afterwards, so the taxonomy can
   still change.
7. **Split by customer and thread groups, golden pool carved out first.** Hashing cases at random would leak the same
   customer, or the same templated reply, across splits.
8. **Retrieve resolutions, not documents.** The question is "what did VirginTrains do last time?", so a memory record is
   problem → brand reply → outcome, from the train split only.
9. **Intent is a soft bonus, never a filter.** The stress test shows filters fall apart once the classifier is wrong.
10. **Tune on one half of dev, report on the other.** Golden is never touched for tuning.
11. **Deterministic policy, separate from the LLM.** LLM confidence is uncalibrated, so the auto-handle decision is made by
    configurable thresholds that list their reasons, not by the model.
12. **Some intents always escalate.** Live status, Delay Repay and complaints need live data, account access or human
    judgment that this system doesn't have. Getting them wrong costs more than escalating them.
13. **Grounding is one-directional and fails closed.** It can only downgrade to ESCALATE, and a verifier error counts as
    "not grounded".
14. **REST client instead of the Gemini SDK, one wrapper.** It's easy to fake in tests, has no SDK version drift, and is one
    place to swap providers.
15. **Blind golden labeling against a provisional taxonomy.** Showing the labeler predictions would anchor them, and the
    evaluation would end up measuring agreement with the model.

---

## What I'd do next with one more week

1. **Finish reviewing the 150 drafted golden labels**, then run the taxonomy review and decide on merges and renames *before* evaluating.
2. **Build the evaluation harness** on golden. It would measure:
   - intent accuracy and macro-F1;
   - escalation precision and recall;
   - the precision of auto-handled replies (the one that matters).

   It would compare against a trivial baseline (always escalate, or always the majority intent) and a simple one (BM25 top-1
   reply plus keyword rules).
3. **LLM-as-judge for reply quality**, with a short rubric: grounded, answers the question, no invented commitments, tone.
   I'd label about 50 replies by hand to measure how well the judge agrees with a human (Cohen's κ) before trusting it.
4. **Calibrate the policy on dev, not golden.** Sweep the thresholds for an auto-handle precision target and report the
   coverage that buys.
5. **Finish the live smoke test** with a paid key, to see AUTO_HANDLE plus grounding PASS end to end.

---

## Repo map

```text
src/
  ingestion/     loader, schema, thread reconstruction, roles, episodes, resolution signals, case builder, resolution memory
  taxonomy/      entity masking, discovery (TF-IDF / embeddings / KMeans), taxonomy builder, registry, finalize
  evaluation/    brand ranking, splits + leakage checks, sampling, retrieval eval, smoke test, calibration and golden labeling tools
  agent/         schemas, classifier, policy, generator, grounding, support_agent (orchestration), config
  models/        LanguageModel interface + Gemini REST client
  retrieval.py   BM25 + embeddings + hybrid retriever
configs/         support_agent.yaml (all thresholds), virgintrains_intents.yaml (candidate taxonomy), cluster labels
scripts/         one entry point per step (see below)
reports/         EDA, intent clusters, case inspection, retrieval evaluation
data/            raw/ (twcs.csv, not committed) · processed/ · golden/
tests/           489 tests, no network or API key needed
```

| script | what it does |
|---|---|
| `profile_dataset.py`, `build_cases.py`, `rank_brands.py` | phase 1: profile all brands, rebuild threads, rank brands |
| `run_virgintrains_pipeline.py` | runs `build_virgintrains` → `discover_intents` → `prepare_splits` → `generate_taxonomy` → `virgintrains_report` |
| `build_resolution_memory.py`, `evaluate_retrieval.py` | memory + proxy retrieval evaluation |
| `run_support_agent.py`, `smoke_test_agent.py` | the agent on one message / on 12 representative dev cases |
| `prepare_golden_eval.py`, `label_golden_eval.py`, `golden_taxonomy_review.py` | golden pack, blind labeling, post-labeling taxonomy review |
| `build_taxonomy_calibration.py`, `label_taxonomy_calibration.py`, `compare_taxonomy.py`, `finalize_taxonomy.py` | optional taxonomy-calibration path (built, not used yet) |

---

## Reference: data contracts and pipeline details

<details>
<summary><b>Input format</b> (from the real <code>twcs.csv</code> header)</summary>

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
</details>

<details>
<summary><b>How VirginTrains cases are built</b></summary>

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

Outputs: `data/processed/virgintrains_{conversations,tweets,cases}.parquet`. Numbers:
[`reports/virgintrains_eda.md`](reports/virgintrains_eda.md) has 14,853 conversations, 65,810 tweets, 17,913 cases, 16%
visibly resolved and 4% DM-redirected.
</details>

<details>
<summary><b>Brand ranking</b> (phase 1)</summary>

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
</details>

<details>
<summary><b>Taxonomy-calibration path</b> (built, optional, not used yet)</summary>

There is a 200-case calibration sample drawn from `golden_pool_reserve`, with a local labeling tool (the system suggestion
is hidden until you click Reveal), a comparison report and a guarded finalizer that writes a frozen
`virgintrains_taxonomy_v1.yaml`. I chose not to run it; the golden labels plus the taxonomy review cover the same need with
one labeling pass. The scripts are listed in the [repo map](#repo-map).
</details>

---

## Assumptions, limitations and credits

- **The public view only.** Anything resolved in DMs, by phone or in person is invisible, so "resolved" is a lower bound.
- **English-only heuristics.** The resolution signals are regexes. `refund` means a refund was *discussed*, including
  refusals.
- **The dataset is a sample.** Some reply links point to tweets that aren't in the file; these are flagged, not treated as
  errors. Every tweet has at least one link, so there are no single-tweet threads.
- **One big linked group.** The largest group of linked cases holds about 16% of all cases, so it lands in a single split.
- **Thresholds are starting values**, chosen conservatively on dev data. They are not tuned to a validated target.
- **Credits.**
  - Data: Customer Support on Twitter (Kaggle, thoughtvector).
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
  - Clustering and TF-IDF: scikit-learn.
  - LLM: Google Gemini (`gemini-2.5-flash`) over REST.
  - BM25 is my own implementation in [`src/retrieval.py`](src/retrieval.py).
  - Written with an AI coding assistant; I can walk through and change any part of it.
