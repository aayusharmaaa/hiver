"""Render the human labeling pack (guidelines + one block per calibration case)."""

from __future__ import annotations

import pandas as pd

from evaluation.inspection import ROLE_LABEL
from evaluation.taxonomy_calibration import ESCALATION_SIGNALS, FALLBACK, HUMAN_COLUMNS, HUMAN_RESOLVED_VALUES

RESOLUTION_DEFINITIONS = {
    "refund": "the agent discusses a refund (including refusing one)",
    "compensation": "the agent discusses compensation or a claim, e.g. Delay Repay",
    "account_action": "the agent says it has changed, or will change, a booking or account",
    "redirected_to_dm": "the agent asks the customer to continue in direct messages",
    "redirected_to_other_operator": "the agent hands the customer to another train operator",
    "escalated": "the agent refers the customer to Customer Relations, a formal complaint route, or forwards it internally",
    "troubleshooting": "the agent gives steps to try",
    "self_service": "the agent points to a link, the app, the website, or live updates",
    "feedback_acknowledged": "the agent thanks the customer or says feedback will be passed on",
    "information_provided": "the agent answers with information that does not fit a more specific type above",
    "clarification_requested": "the agent only asks a question",
    "other": "the agent replied, but nothing above fits",
    "unresolved": "the brand never replied, or the customer was left hanging",
    "unclear": "you cannot tell",
}

TIE_BREAKS = [
    "Label the customer's **main request**. If the conversation shows the real request is different from the opening line, label the real request and say so in `human_notes`.",
    "If two intents both fit, choose the one the agent would have to act on first, and name the other in `human_notes` as `also: <intent>`.",
    "Use `unclear_or_media_only` only when you cannot tell what the customer wants even after reading the whole conversation. Short but clear requests (\"any update?\", \"refund?\") belong to their normal intent.",
    "Do not choose an intent because it is the system's suggestion. Decide first, then compare.",
    "If none of the intents fit and you would write a new one, put `NEW:<snake_case_name>` in `human_intent` and describe it in `human_notes`.",
]

DECISION_QUESTIONS = [
    "Is there a question or request the agent can act on? If yes, pick the intent for the **topic of that request** (status, ticket, refund claim, seat, Wi-Fi, catering...).",
    "If there is no request, is the customer venting about a journey or service? Pick the complaint intent that matches what they are unhappy about.",
    "If there is no request and no complaint, is it thanks or a compliment? `praise_positive_feedback`.",
    "If it is banter, a joke, a photo, or general chatter that asks for nothing: `chitchat_non_support`.",
    "Only if you still cannot say what the message is for: `unclear_or_media_only`.",
]

RESOLUTION_PRECEDENCE = (
    "When two types could apply, take the **most specific** one: `refund`, `compensation`, `account_action`, `escalated`, and the "
    "redirects come before `troubleshooting`; `troubleshooting` before `self_service`; `self_service` before `feedback_acknowledged`; "
    "`information_provided` and `clarification_requested` are the general cases; `other` is the last resort."
)

ESCALATION_DEFINITIONS = {
    "none": "no sign that the case needs more than a normal reply",
    "formal_complaint_or_customer_relations": "the customer wants to complain formally, or the agent points to Customer Relations / a complaints form, or the case is already with them",
    "safety_or_vulnerability": "physical safety, harassment or aggression, a stranded or distressed customer, or someone who says they are vulnerable",
    "accessibility_or_assistance": "the customer needs, or reports a failure of, assisted travel, disability access, or help boarding",
    "legal_media_or_ombudsman_threat": "the customer threatens legal action, the press, an ombudsman or a regulator",
    "repeated_unresolved_contact": "the customer says they already contacted the brand (more than once, or a long time ago) without a useful answer",
    "needs_account_or_booking_lookup": "the agent cannot help without the customer's booking, account, or claim details",
    "other": "none of the above fits; explain in `human_notes`",
}

NOTE_CONVENTIONS = [
    "`also: <intent_name>`: a second intent also fits. The comparison tool reads this, so write the exact intent name, one `also:` per intent.",
    "`real request: ...`: the opening line is not the real request.",
    "`unsure: ...`: a genuine coin-flip between two labels (name both).",
    "`resolution: ...`: your resolution type or resolved label differs from what the system would guess, and why.",
    "`new intent: <name>: <one-line definition>`: when you used `NEW:<name>`.",
]


def _clean(text: object) -> str:
    return " ".join(str(text).split())


def render_examples(sections: list[dict]) -> list[str]:
    lines = [
        "## Worked examples (from the train split)",
        "",
        "Every case below is a **train** case, never one of the cases you are labelling and never a golden case. The quotes are copied from the "
        "dataset. Each reading shows how the rules apply and is one defensible reading, not ground truth. The system's suggestion is printed "
        "so you can see how often it disagrees with a sensible reading.",
        "",
    ]
    for section in sections:
        lines += [f"### {section['title']}", ""]
        if section["intro"]:
            lines += [section["intro"], ""]
        for ex in section["cases"]:
            lines.append(f"**`{ex['case_id']}`**  ")
            for t in ex["turns"]:
                lines.append(f"> **{t['role']}:** {t['text']}  ")
            if ex["truncated"]:
                lines.append("> (conversation shortened)  ")
            lines += [
                "",
                f"- System suggestion (not authoritative): `{ex['candidate_intent'] or 'none'}`",
                f"- Reading: {ex['reading']}",
                "",
            ]
    return lines


def render_guidelines(taxonomy: dict, confusable_notes: list[dict], n_cases: int, examples: list[dict] | None = None) -> str:
    intents = list(taxonomy["intents"]) + ([taxonomy["fallback"]] if taxonomy.get("fallback") else [])
    lines = [
        "# VirginTrains taxonomy calibration: labeling guide",
        "",
        f"You will label **{n_cases}** customer support cases from VirginTrains' Twitter support account. Your labels are the human reference: "
        "they are compared with the system's candidate intents to see which intents are real, which are confused with each other, and "
        "which should be merged, split or renamed before the taxonomy is frozen.",
        "",
        "The cases come from a held-back reserve. None of them is one of the 250 golden evaluation cases.",
        "",
        "## 1. Ground rules",
        "",
        "- **Decide from the conversation first.** In the labeling tool the system's suggestion stays hidden until you click *Reveal*. "
        "Label, then (if you want) reveal it and compare.",
        "- **Do not follow the suggestion blindly.** It comes from clustering opening messages, it is sometimes plainly wrong (see the worked "
        "examples), and the point of this exercise is to find where it is wrong. High agreement is not the goal; honest labels are.",
        "- **Use only what is visible** in the public conversation. Do not guess about DMs, phone calls, or later events.",
        "- **Do not use model output** (ChatGPT, Gemini or any other) to fill labels. This phase must be human-labelled.",
        "- **Never leave a half-labelled case.** The four required fields are `human_intent`, `human_resolution_type`, `human_resolved`, "
        "`human_escalation_signal`; `human_notes` is optional except where this guide says otherwise.",
        "- When you are unsure, give your best label and write a note. A confident wrong label is worse than a flagged uncertain one.",
        "",
        "## 2. How to run the labeling",
        "",
        "```",
        "python scripts/label_taxonomy_calibration.py",
        "```",
        "",
        "- One case at a time, with progress `X / " + str(n_cases) + "`, previous / next, and a jump list.",
        "- Each *Save* checkpoints to `data/processed/taxonomy_calibration.csv` immediately. Close the tool whenever you like; "
        "start it again and it opens at your first unlabelled case.",
        "- The tool writes only the five `human_*` columns. Case ids, tweet ids and sampling metadata cannot be changed from it, "
        "and it refuses to start if they were changed on disk.",
        "- `data/processed/taxonomy_calibration_label_audit.jsonl` records every save (time, before/after, whether you had revealed the suggestion).",
        "- Alternative without the tool: fill the five `human_*` columns directly in the CSV (matched by `case_id`) using the same vocabulary. "
        "Do not edit any other column.",
        "",
        "## 3. What counts as the primary customer intent",
        "",
        "The **primary intent is the customer's main request or complaint as the conversation shows it**, not the first words, not the agent's "
        "reply, and not the topic of the disruption that provoked it.",
        "",
        *[f"{i}. {q}" for i, q in enumerate(DECISION_QUESTIONS, start=1)],
        "",
        "Multi-intent cases:",
        "",
        "- Choose the request the **agent would have to act on first**, and add `also: <intent_name>` to `human_notes`.",
        "- A complaint plus an explicit question (\"how do I claim a refund?\"): the explicit question usually decides.",
        "- Sarcasm is read as what it means (\"thanks for 0 minutes to change trains\" is a complaint).",
        "- If the opening line is not the real request, label the real request and note `real request: ...`.",
        "- Later turns count: the customer often clarifies what they wanted.",
        "- Disruption is context, not an intent by itself: someone stranded by a delay who asks whether their ticket is valid asks a ticket question.",
        "",
        "## 4. The intents",
        "",
    ]
    for it in intents:
        inc = it["inclusion_criteria"] if it["inclusion_criteria"] != ["NEEDS_REVIEW"] else ["(none written)"]
        exc = it["exclusion_criteria"] if it["exclusion_criteria"] != ["NEEDS_REVIEW"] else ["(none written)"]
        lines += [
            f"### `{it['name']}`" + ("  (fallback)" if it.get("fallback") else ""),
            "",
            _clean(it["definition"]),
            "",
            "- **Fits when:** " + " ".join(f"{x}" for x in inc),
            "- **Does not fit when:** " + " ".join(f"{x}" for x in exc),
            "",
        ]
    lines += ["## 5. Choosing between confusable intents", "", "Notes for the pairs the system confuses most:", ""]
    for n in confusable_notes:
        a, b = n["pair"]
        lines.append(f"- `{a}` vs `{b}`: {_clean(n['note'])}")
    lines += ["", "General tie-breaks:", "", *[f"- {t}" for t in TIE_BREAKS], ""]
    lines += [
        "## 6. When to use the fallback `unclear_or_media_only`",
        "",
        "- Use it when, **after reading the whole conversation**, you cannot tell what the customer wants: a bare mention, a photo or link with no words, "
        "or text too short to interpret.",
        "- Do **not** use it for a message that is clear but awkward to classify, for a short but clear request (\"any update?\"), or for a "
        "non-support message (that is `chitchat_non_support`).",
        "- Label what you can see; do not guess what a link or image might contain.",
        "",
        "## 7. When to use `NEW:<name>`",
        "",
        "- Only when **no existing intent fits and you would expect the same kind of case to recur**. Use `snake_case`, e.g. `NEW:lost_property`.",
        "- A note is **required**: describe the new intent in one line (`new intent: <name>: <definition>`).",
        "- For a one-off, pick the closest existing intent and explain in `human_notes`.",
        "- A `NEW:` label is a proposal. Whether it becomes an intent is decided later by the reviewer, after the comparison report.",
        "",
        "## 8. `human_resolution_type`: what the agent did",
        "",
        "Pick the agent's main move from the agent's words. Do not copy the system's `auto_resolution_type`.",
        "",
        *[f"- `{k}`: {v}" for k, v in RESOLUTION_DEFINITIONS.items()],
        "",
        RESOLUTION_PRECEDENCE,
        "",
        "## 9. `human_resolved`",
        "",
        f"`{'` / `'.join(HUMAN_RESOLVED_VALUES)}`.",
        "",
        "- **yes**: the request was visibly met or closed in the public conversation (the customer confirms, or the agent's reply fully answers it and nothing is left open).",
        "- **no**: something is visibly still open, the customer pushes back, or the brand never replied.",
        "- **unclear**: you cannot tell from what is public, for example the thread moves to DMs.",
        "",
        "A polite \"thanks\" after a reply that did not answer the question is not a resolution.",
        "",
        "## 10. `human_escalation_signal`",
        "",
        "Describes a signal visible in the conversation, not a policy about what to do. Choose the strongest that applies; mention others in `human_notes`.",
        "",
        *[f"- `{k}`: {v}" for k, v in ESCALATION_DEFINITIONS.items()],
        "",
        "## 11. `human_notes`",
        "",
        "Free text. Write a note whenever the case is ambiguous, two intents fit, the opening line is misleading, or you used `NEW:`. "
        "These notes feed the disagreement report. Suggested prefixes:",
        "",
        *[f"- {c}" for c in NOTE_CONVENTIONS],
        "",
    ]
    if examples:
        lines += render_examples(examples)
    lines += [
        "## 12. Before you finish",
        "",
        "- All 200 cases show as labelled in the tool (the progress bar reads 200 / 200).",
        "- Every `NEW:` label has a note.",
        "- You did not change anything outside the `human_*` columns.",
        "- Do not edit the decisions file yet: merge and split decisions are made *after* the comparison report, by the reviewer, in `configs/virgintrains_taxonomy_decisions.yaml`.",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def render_case(order: int, row: pd.Series) -> str:
    lines = [
        f"### {order:03d} · `{row['case_id']}`",
        "",
        f"**First customer message:** {_clean(row['first_customer_message']) or '(none)'}",
        "",
        "**Conversation**",
        "",
    ]
    for t in row["turns"]:
        stamp = "" if pd.isna(t["created_at"]) else pd.Timestamp(t["created_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{t['turn_index'] + 1}. **{ROLE_LABEL.get(t['role'], t['role'])}** `{t['tweet_id']}` {stamp}: {_clean(t['text'])}")
    runner = f"; runner-up `{row['runner_up_intent']}`" if isinstance(row["runner_up_intent"], str) and row["runner_up_intent"] else ""
    lines += [
        "",
        "> **System suggestion (label the message first; do not anchor on this)**  ",
        f"> Candidate intent: `{row['candidate_intent']}` (cluster {row['candidate_cluster_id']} `{row['candidate_cluster_name']}`{runner}, margin {row['cluster_margin']})  ",
        f"> Auto resolution: `{row['auto_resolution_type']}`, resolved: {row['auto_resolved']}, DM redirect: {row['auto_dm_redirect']}, outcome `{row['auto_resolution_outcome']}`  ",
        f"> {_clean(row['auto_resolution_summary'])}",
        "",
        f"Fill in CSV row `{row['case_id']}`: " + ", ".join(f"`{c}`" for c in HUMAN_COLUMNS),
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def render_labeling_markdown(
    frame: pd.DataFrame, turns_by_case: dict[str, list[dict]], taxonomy: dict, confusable_notes: list[dict], examples: list[dict] | None = None
) -> str:
    """`frame` is the calibration CSV frame in labeling order."""
    parts = [render_guidelines(taxonomy, confusable_notes, len(frame), examples)]
    for order, (_, row) in enumerate(frame.iterrows(), start=1):
        r = row.copy()
        r["turns"] = turns_by_case[row["case_id"]]
        parts.append(render_case(order, r))
    return "\n".join(parts)


__all__ = ["render_labeling_markdown", "render_guidelines", "RESOLUTION_DEFINITIONS", "FALLBACK"]
