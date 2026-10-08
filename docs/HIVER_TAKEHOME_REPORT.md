# Hiver SDE Intern Take-home Report

## Executive summary

**Thesis: evidence before generation.** The safest support automation is not the model that answers the most tickets; it is the system that knows when evidence is insufficient and hands the case to a human before it invents an answer.

### Headline results

| Benchmark | Result |
|---|---:|
| Intent classification - 100 blind human labels | **60.0% accuracy / 0.533 macro-F1** |
| TF-IDF + Logistic Regression baseline | 54.0% / 0.466 |
| Majority baseline | 15.0% / 0.024 |
| Retrieval - 857 dev queries | **MRR 0.572 / R@5 0.784** |
| E2E - 50 stratified cases | **22% auto-handle / 20.0% safe automation** |
| E2E escalation recall | **92.3%** |
| False auto-handle rate | **9.1%** |

## 1. Problem framing

VirginTrains has recurring support patterns: service status, disruptions, Delay Repay, booking and seats, wifi, catering, complaints and praise. But some requests require information historical tweets cannot safely provide: live train status, account-specific refunds, formal complaints, and safety/accessibility issues.

So the objective is **safe automation**, not raw coverage.

**Good:** auto-handle only when historical evidence supports the response and policy permits automation.

**Bad:** a fluent answer that is wrong, over-promises, invents policy, or implies an account action was completed.

**Out of scope on purpose:** Twitter/Hiver APIs, auth, deployment, vector databases, rerankers, fine-tuning, live train data and account access.

## 2. System design

Customer message -> thread context -> intent -> hybrid retrieval -> deterministic risk policy -> grounded reply -> grounding verification.

The critical choice is **evidence before generation**. Retrieval supplies precedent; deterministic policy decides whether automation is allowed; generation writes from that evidence; grounding can only downgrade AUTO_HANDLE to ESCALATE.

## 3. Dataset and golden evaluation

I reconstructed **17,913 support cases** from the Customer Support on Twitter corpus. The candidate taxonomy contains 10 intents plus an unclear/media-only fallback and is explicitly treated as a candidate taxonomy, not ground truth.

The golden set contains **250 frozen cases**. **100 were independently labelled by a human without model outputs; 150 began as assistant drafts and were subsequently human-reviewed and confirmed.** The 100 blind set is the primary benchmark; the 250-case result is secondary because draft-assisted review can introduce anchoring.

## 4. Results vs baselines

- **Intent:** GPT-OSS 120B scored 60.0% / 0.533 on the 100 blind cases, ahead of TF-IDF + Logistic Regression at 54.0% / 0.466 and majority at 15.0% / 0.024.
- **Retrieval:** hybrid + soft intent bonus reached MRR 0.572 / R@5 0.784 on 857 dev queries. BM25 alone reached 0.372 / 0.551.
- **Routing:** on the 50-case stratified end-to-end slice, 11/50 were auto-handled. One of those 11 should have escalated, giving a 9.1% false-auto rate. 12/13 must-escalate cases were correctly escalated, but 27/37 automatable cases were still sent to a human.
- **Reply quality:** 14 drafts received both LLM-judge and human scores. Weighted kappa was 0.26 correctness, 0.00 groundedness, 0.52 actionability and 0.16 brand alignment.

The important result is therefore not “60% intent accuracy.” The current system is safer than its coverage would suggest, but much too conservative.

## 5. Top five failure modes

### 1. Blanket policy over-escalates

case_2396788 was positive feedback mentioning a wheelchair. A sensitive-word rule blocked it despite strong evidence.

**Hypothesis:** policy should be tied to required action, not isolated keywords.

### 2. Intent boundaries are genuinely ambiguous

case_2639881 is both a status question and a disruption complaint. Gold: journey_disruption_complaint; model: service_status_delay_enquiry.

**Hypothesis:** define intents around the action the brand must take.

### 3. Confidence is not calibrated

case_2607464 was a sarcastic complaint classified as chitchat at 0.97 confidence.

**Hypothesis:** self-reported LLM confidence is not a probability; calibrate it or replace it with retrieval-agreement signals.

### 4. Retrieval finds the topic, not the handling

case_527334 retrieved the right broad intent but the wrong resolution type; the first relevant case appeared at rank 246.

**Hypothesis:** retrieve on conversation context and diversify by historical resolution type.

### 5. The agent cannot see all of the real context

case_482898 involves another operator; case_1845809 is effectively media-only.

**Hypothesis:** add explicit operator detection and low-information/media abstention.

## 6. What is misleading about my headline number?

- 60% is on 100 primary human-labelled cases, not the full traffic distribution.
- The 250-case number is weaker evidence because 150 labels began as assistant drafts.
- The taxonomy itself is imperfect: 131/250 candidate-vs-gold disagreements show boundary ambiguity.
- Retrieval metrics are proxy metrics, not human relevance judgements.
- The E2E sample is 50 stratified cases, not a fleet-wide estimate.
- Coverage and safety trade off: 22% automation includes heavy over-escalation.
- The LLM judge is only a rough first screen: n=14 and groundedness kappa = 0.00.
- Historical resolution labels are heuristic because DM/phone outcomes are invisible.

## 7. What I would do with one more week

1. Make routing action-based rather than intent-wide.
2. Calibrate abstention using dev reliability curves and retrieval agreement.
3. Give the classifier compact conversation context.
4. Retrieve on thread context and diversify by resolution type.
5. Grow E2E evaluation from 50 to 250 cases, add a temporal holdout, and collect 50+ human reply ratings.
6. Add explicit lost-property / other-operator paths and formalize the top confusion pairs.

## Bottom line

This is not a chatbot that answers tweets; it is a conservative support decision system. The strongest evidence from the take-home is the architecture: retrieve precedent before generation, keep escalation policy deterministic, fail closed on grounding, and measure automation by safety as well as coverage. The current bottleneck is over-escalation and evaluation scale - exactly the next problems to attack.
