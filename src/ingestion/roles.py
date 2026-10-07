"""Role assignment for tweets in a brand's conversations.

The dataset's `inbound` flag is not fully reliable: some accounts of other operators are
flagged `inbound=True` even though they answer customers with agent-style sign-offs
("... ^KM"). We keep the raw flag untouched and add a derived `role`:

    customer      inbound tweet from a normal customer
    brand_agent   outbound tweet from the target brand
    other_agent   outbound tweet from another brand, or an `inbound=True` account that
                  behaves like an agent (caret sign-off on most of its tweets)
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from ingestion import schema as S

logger = logging.getLogger(__name__)

ROLE = "role"
CUSTOMER = "customer"
BRAND_AGENT = "brand_agent"
OTHER_AGENT = "other_agent"

SIGNATURE = re.compile(r"\^[A-Za-z]{2,3}\s*$")


def detect_hidden_agents(
    tweets: pd.DataFrame, min_tweets: int = 5, min_signature_rate: float = 0.6
) -> set[str]:
    """Authors flagged inbound whose tweets mostly end in a ^XX agent sign-off."""
    inbound = tweets.loc[tweets[S.INBOUND]]
    if inbound.empty:
        return set()
    signed = inbound[S.TEXT].astype(str).str.contains(SIGNATURE, regex=True)
    stats = signed.groupby(inbound[S.AUTHOR_ID]).agg(["size", "mean"])
    hidden = stats[(stats["size"] >= min_tweets) & (stats["mean"] >= min_signature_rate)]
    found = set(hidden.index.astype(str))
    logger.info("Detected %d inbound-flagged accounts that behave like agents: %s", len(found), sorted(found))
    return found


def assign_roles(tweets: pd.DataFrame, brand: str, hidden_agents: set[str]) -> pd.Series:
    author = tweets[S.AUTHOR_ID].astype(str)
    outbound = ~tweets[S.INBOUND].astype(bool)
    role = pd.Series(CUSTOMER, index=tweets.index, dtype=object)
    role[outbound & (author == brand)] = BRAND_AGENT
    role[(outbound & (author != brand)) | author.isin(hidden_agents)] = OTHER_AGENT
    role[(author == brand)] = BRAND_AGENT
    return role
