# Hiver Support Agent — Evidence-Grounded Customer Support Copilot

> Hiver SDE Intern take-home
>
> An evidence-grounded AI support agent that classifies incoming customer requests, retrieves historically similar resolutions, decides whether a case is safe to auto-handle, and generates a response only when the evidence supports it.

## Overview

This project builds a conservative customer-support agent on the [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

Rather than treating the LLM as the source of truth, the system uses historical support interactions as a **resolution memory**. The LLM proposes an intent and a response; deterministic policy and grounding checks decide whether that response is safe to surface.

The current implementation focuses on **VirginTrains**, selected from the dataset after comparing brands on conversation volume, multi-turn interaction density, visible resolution patterns, and DM reliance.

## Architecture

```
                         Customer message
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Context / case state│
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Gemini intent       │
                    │ classifier          │
                    └──────────┬──────────┘
                               │ candidate intent
                               ▼
             ┌───────────────────────────────────┐
             │ Hybrid historical retrieval       │
             │ BM25 + sentence embeddings       │
             │ + soft candidate-intent signal   │
             └───────────────┬───────────────────┘
                             │ evidence
                             ▼
                    ┌─────────────────────┐
                    │ Deterministic risk  │
                    │ / escalation policy │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              AUTO-HANDLE             ESCALATE
                    │
                    ▼
          ┌─────────────────────┐
          │ Gemini response     │
          │ generation          │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Grounding verifier  │
          │ deterministic + LLM │
          └──────────┬──────────┘
                     │
             PASS ───┴─── FAIL
              │             │
              ▼             ▼
        suggested reply   ESCALATE
```

The central design rule is **evidence before generation**: retrieval supplies historical evidence; it does not grant the model permission to make commitments.

## What the system does

### 1. Reconstructs support cases

The raw dataset is a reply graph rather than a clean ticket table. The ingestion pipeline:

- validates and cleans the TWCS schema;
- reconstructs conversations from reply links;
- derives customer, VirginTrains-agent, and other-operator roles;
- splits long customer interactions into support episodes using a 24-hour gap rule;
- preserves tweet-level provenance throughout.

This produces episode-level cases instead of treating every tweet as an independent training example.

### 2. Builds a resolution memory

Historical VirginTrains episodes are transformed into compact evidence records containing:

- customer problem;
- historical brand response;
- rule-derived resolution summary/type;
- candidate intent;
- source tweet IDs and provenance.

Cases with no meaningful brand response or weak evidence are excluded from the primary retrieval corpus.

### 3. Creates a candidate intent taxonomy

The taxonomy was derived from clustering and manual inspection of representative cases. It is intentionally labelled:

`CANDIDATE_NOT_GROUND_TRUTH`

because the human calibration/finalization step has not yet been completed.

The current candidate taxonomy contains ten support intents plus a fallback:

| Intent | Approx. cases |
|---|---:|
| `service_status_delay_enquiry` | 4,251 |
| `journey_disruption_complaint` | 2,093 |
| `chitchat_non_support` | 2,047 |
| `praise_positive_feedback` | 1,481 |
| `ticket_booking_query` | 1,360 |
| `seat_reservation_issue` | 1,096 |
| `customer_service_complaint` | 1,090 |
| `delay_repay_refund_claim` | 884 |
| `first_class_catering_issue` | 707 |
| `onboard_wifi_issue` | 519 |
| `unclear_or_media_only` | 347 |

These counts are dataset-derived and are not human-labelled class frequencies.

### 4. Retrieves similar historical resolutions

Retrieval combines:

- BM25 lexical matching;
- sentence-transformer semantic similarity;
- normalized hybrid scoring;
- a **soft**, rather than hard, candidate-intent signal.

The intent signal is deliberately soft. A wrong intent prediction should degrade retrieval rather than completely block useful evidence.

A proxy evaluation on held-out development queries found:

| Strategy | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Random | 0.053 | 0.156 | 0.237 | 0.152 |
| BM25 | 0.223 | 0.447 | 0.551 | 0.372 |
| Embedding | 0.287 | 0.522 | 0.616 | 0.443 |
| Hybrid | 0.299 | 0.518 | 0.623 | 0.445 |
| Hybrid + intent | **0.407** | **0.690** | **0.784** | **0.572** |

**Important:** this is a proxy retrieval benchmark, not human relevance ground truth. Relevance uses candidate intent + rule-derived resolution type, so the intent-assisted number is partly circular and should not be interpreted as end-to-end support quality.

### 5. Makes an explicit automation decision

The agent separates **reasoning** from **authority**.

A deterministic policy can force escalation for:

- ambiguous or low-information requests;
- multi-intent requests;
- low classifier confidence;
- weak or insufficient evidence;
- evidence that conflicts with the predicted candidate intent;
- sensitive wording or exception-like cases;
- intents where public historical replies do not justify autonomous handling.

The policy is intentionally conservative: a failed safety check can downgrade `AUTO_HANDLE` to `ESCALATE`, but the grounding verifier cannot upgrade an escalation.

### 6. Generates grounded replies

Gemini is used for response generation only after retrieval and policy checks.

The generated response is then checked by:

1. deterministic checks for unsupported claims such as links or monetary amounts not present in the evidence;
2. a Gemini grounding verifier for broader unsupported claims.

A grounding failure results in escalation rather than a potentially fabricated public reply.

## Data and leakage controls

The project treats evaluation leakage as a first-class engineering concern.

Cases are partitioned into:

```
train_retrieval
       │
       ├── retrieval corpus
       │
       ▼
dev_calibration
       │
       ▼
golden / golden_pool_reserve
```

Split assignment groups cases sharing customer, conversation, or related provenance so a single customer's interactions cannot quietly appear across evaluation boundaries.

Additional checks cover case, customer, conversation, group, tweet, and opening-text overlap. The live smoke-test workflow also verifies that each selected query is non-golden before making a model call.

The 250-case golden candidate set is frozen and currently **unlabelled**. It is not used for tuning.

## Dataset profile

The full TWCS dump contains approximately **2.81M tweets**, **798K reconstructed conversations**, and **108 brands**.

For VirginTrains specifically, the pipeline produced roughly:

- 14.8K reconstructed conversations;
- 65.8K tweets;
- 17.9K support episodes;
- 91.9% of cases with a VirginTrains reply;
- ~16% conservatively resolved by the rule-based resolution model;
- ~4% DM redirects.

The dataset is noisy. In particular, `brand replied != resolved`, and the public thread ending is only a weak proxy for whether the customer's issue was actually solved.

## Brand selection

VirginTrains was selected rather than simply choosing the largest brand.

The ranking considered conversation volume, reconstructability, multi-turn density, repeated response templates, intent diversity, conservative resolution rate, and non-DM handling.

VirginTrains ranked highly while retaining enough multi-turn structure and visible resolution evidence to support a meaningful historical-resolution agent.

## Live smoke test

Before final evaluation, the repository includes a leakage-safe live smoke test covering representative scenarios such as status, booking, seats, Delay Repay, Wi-Fi, catering, complaints, praise, low-information, multi-intent, ambiguous, and escalation-like cases.

This uses the **real Gemini API**; mocks are intentionally not treated as live validation.

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="YOUR_KEY"
python scripts/smoke_test_agent.py --limit 12 --seed 42

# Preview selected cases without calling Gemini
python scripts/smoke_test_agent.py --select-only --limit 14
```

The smoke test does not create evaluation scores or modify golden/taxonomy artifacts. It is a diagnostic gate for the end-to-end pipeline.

> **Current status:** live Gemini validation must be run locally with `GEMINI_API_KEY`. The repository does not claim that the live smoke test has passed until that run succeeds.

## Setup

Python 3.10+ is required. The pipeline was developed against Python 3.11.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Download `twcs.csv` from Kaggle into `data/raw/`.

Alternatively, using `kagglehub`:

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'); shutil.copy(p + '/twcs/twcs.csv', 'data/raw/twcs.csv')"
```

The scripts add `src/` to `sys.path`; editable installation is optional.

## Reproduce the pipeline

### Full data foundation

```bash
python scripts/profile_dataset.py --input data/raw/twcs.csv
python scripts/build_cases.py --input data/raw/twcs.csv
python scripts/rank_brands.py --top 15
```

### VirginTrains pipeline

```bash
python scripts/run_virgintrains_pipeline.py --input data/raw/twcs.csv
```

This runs the main VirginTrains processing stages and generates the intermediate artifacts used by taxonomy discovery, splitting, and retrieval.

### Retrieval evaluation

```bash
python scripts/evaluate_retrieval.py
```

The resulting report lives under `reports/virgintrains_retrieval_evaluation.*`.

### Tests

```bash
pytest -q
```

## Repository structure

```text
hiver/
├── configs/
│   ├── virgintrains_intents.yaml        # candidate intent taxonomy
│   ├── virgintrains_cluster_labels.yaml
│   ├── virgintrains_labeling_examples.yaml
│   └── virgintrains_taxonomy_decisions.yaml
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py                    # input validation + loading
│   │   ├── reconstruction.py             # reply-graph reconstruction
│   │   ├── roles.py                      # role inference
│   │   ├── episodes.py                   # customer episode construction
│   │   ├── cases.py / episode_cases.py   # case materialization
│   │   ├── resolution.py                 # baseline resolution rules
│   │   ├── resolution_signals.py         # evidence-backed resolution signals
│   │   └── resolution_memory.py           # retrieval corpus construction
│   │
│   ├── taxonomy/
│   │   ├── discovery.py                  # clustering / discovery
│   │   ├── taxonomy_builder.py           # draft taxonomy generation
│   │   ├── registry.py                   # taxonomy validation
│   │   └── finalize.py                   # guarded human freeze workflow
│   │
│   ├── retrieval.py                      # BM25 + semantic hybrid retrieval
│   │
│   ├── agent/
│   │   ├── schemas.py                    # typed agent results
│   │   ├── classifier.py                 # Gemini intent classification
│   │   ├── policy.py                     # deterministic risk policy
│   │   ├── generator.py                  # grounded response generation
│   │   ├── grounding.py                  # response grounding verification
│   │   └── support_agent.py              # end-to-end orchestrator
│   │
│   ├── models/
│   │   ├── base.py                       # model abstraction
│   │   └── gemini.py                     # Gemini REST client
│   │
│   └── evaluation/
│       ├── splits.py                     # leakage-safe split construction
│       ├── sampling.py                  # deterministic sampling
│       ├── retrieval.py                  # retrieval benchmark + stress tests
│       ├── brand_ranking.py              # brand selection
│       ├── smoke_selection.py            # live smoke-test case selection
│       └── smoke_report.py               # smoke-test execution/reporting
│
├── scripts/
│   ├── run_virgintrains_pipeline.py
│   ├── evaluate_retrieval.py
│   ├── smoke_test_agent.py
│   └── ...
│
├── reports/
│   ├── virgintrains_eda.md
│   ├── virgintrains_intent_clusters.md
│   └── virgintrains_retrieval_evaluation.md
│
├── data/
│   ├── raw/                              # local input, not committed
│   ├── processed/                        # generated artifacts
│   └── golden/                           # frozen golden candidates
│
└── tests/
```

## Testing status

The latest local run reported **424 passing tests and 1 skipped test**.

The suite covers ingestion, conversation reconstruction, role attribution, resolution signals, taxonomy discovery/freeze guards, leakage-safe splits, retrieval, agent policy, grounding behavior, model-client behavior, and smoke-test selection/reporting.

## Limitations and honest status

This repository deliberately distinguishes what has been measured from what has only been implemented.

**Not yet claimed:**

- final intent accuracy/F1 on human-labelled data;
- final auto-handle safety rate;
- reply-quality benchmark on a human-labelled golden set;
- LLM-judge/human agreement;
- live Gemini smoke-test success until an API-backed run is completed.

The biggest modelling limitations are also explicit:

1. **The taxonomy is provisional.** Several VirginTrains clusters are genuinely ambiguous, especially around delay/status, complaints, seats, and refunds.
2. **Resolution labels are weak.** They are inferred from public conversation endings and regex evidence, not internal ticket outcomes.
3. **Retrieval evaluation is proxy-based.** It is useful for comparing retrieval strategies, but it is not equivalent to human relevance judgement.
4. **LLM confidence is not calibrated.** A model-reported confidence score should not be interpreted as a statistically calibrated probability.
5. **Historical fidelity is not the same as helpfulness.** A response can resemble prior brand behavior and still be unhelpful to a customer.

## What is intentionally next

The intended next evaluation stage is:

```
live Gemini smoke test
        ↓
freeze implementation
        ↓
human-label 150–250 golden cases
        ↓
classification + escalation + reply evaluation
        ↓
LLM-as-judge / human agreement
        ↓
failure analysis
        ↓
small reviewer UI
```

The golden set should remain untouched until the implementation and prompts are frozen.

## Design decisions worth reading

A few choices define the project more than the individual libraries:

- **Episode-level evidence over tweet-level chunks:** support resolutions are conversational workflows, so retrieval uses compact historical episodes.
- **Soft intent conditioning:** wrong intents should not destroy retrieval.
- **Deterministic policy outside the LLM:** the model does not decide its own safety boundary.
- **Conservative abstention:** uncertain cases are escalated rather than guessed.
- **Grounding as a downgrade-only gate:** verification can block an unsafe reply, never authorize a risky one.
- **Provenance everywhere:** every retrieved resolution remains traceable to source tweets.

---

Built for the Hiver SDE Intern take-home by Aayush Sharma.
