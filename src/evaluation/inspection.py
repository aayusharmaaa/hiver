"""Human-readable rendering of support cases for manual inspection."""

from __future__ import annotations

import pandas as pd

ROLE_LABEL = {"customer": "CUSTOMER", "brand_agent": "AGENT", "other_agent": "OTHER-AGENT"}


def _clean(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def conversation_text(turns: list[dict]) -> str:
    """Plain one-line-per-turn rendering (used in the CSV sample)."""
    return "\n".join(f"[{ROLE_LABEL.get(t['role'], t['role'])} {t['tweet_id']}] {_clean(t['text'])}" for t in turns)


def render_case_markdown(row: pd.Series, cluster_name: str | None = None) -> str:
    turns = list(row["full_turns"])
    first_ts = row["first_timestamp"]
    lines = [
        f"### {row['case_id']}  ·  {row['resolution_type']}  ·  {'resolved' if row['resolved'] else 'not resolved'}"
        + ("  ·  DM redirect" if row["dm_redirect"] else ""),
        "",
        f"**First customer message:** {_clean(row['opening_message'] or '(none)')}",
        "",
        "**Conversation**",
        "",
    ]
    for t in turns:
        stamp = "" if pd.isna(t["created_at"]) else pd.Timestamp(t["created_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{t['turn_index'] + 1}. **{ROLE_LABEL.get(t['role'], t['role'])}** `{t['tweet_id']}` {stamp}: {_clean(t['text'])}")
    lines += [
        "",
        f"**Resolution:** {_clean(row['resolution_summary'])}",
        "",
        "**Metadata:** "
        + " · ".join(
            [
                f"conversation `{row['conversation_id']}`",
                f"customer `{row['customer_id']}`",
                f"turns {row['turn_count']} ({row['length_bucket']})",
                f"period {row['time_period']}",
                f"outcome `{row['resolution_outcome']}`",
                f"split `{row['split']}`",
                f"cluster {row['cluster_id']}" + (f" ({cluster_name})" if cluster_name else ""),
                f"first {'' if pd.isna(first_ts) else pd.Timestamp(first_ts).strftime('%Y-%m-%d %H:%M')}",
                f"continuation {bool(row['is_continuation'])}",
                f"thread customers {row['thread_customer_count']}",
                f"source tweets {list(row['source_tweet_ids'])}",
            ]
        ),
        "",
        "---",
        "",
    ]
    return "\n".join(lines)
