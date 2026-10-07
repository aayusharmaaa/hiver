"""Select the conversations that belong to one brand."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

import pandas as pd

from ingestion import schema as S
from ingestion.reconstruction import CONVERSATION_ID

logger = logging.getLogger(__name__)


@dataclass
class BrandFilterStats:
    brand: str
    conversations_total: int = 0
    conversations_with_brand_reply: int = 0
    conversations_mention_only: int = 0
    conversations_selected: int = 0
    tweets_selected: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def select_brand_conversations(conversations: pd.DataFrame, brand: str) -> tuple[pd.DataFrame, BrandFilterStats]:
    """Keep whole conversations (never partial threads) in which `brand` is involved.

    A conversation is selected when
      * the brand authored at least one tweet in it, or
      * no outbound tweet exists at all and a customer tweet @-mentions the brand
        (an unanswered tweet to the brand).
    Full reply threads are kept so that nothing is lost at the filtering step.
    """
    stats = BrandFilterStats(brand=brand, conversations_total=int(conversations[CONVERSATION_ID].nunique()))
    brand_lower = brand.lower()
    by_brand = conversations[S.AUTHOR_ID].astype(str).str.lower() == brand_lower
    with_reply = set(conversations.loc[by_brand, CONVERSATION_ID])

    has_outbound = set(conversations.loc[~conversations[S.INBOUND], CONVERSATION_ID])
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    mentions = conversations[S.INBOUND] & conversations[S.TEXT].astype(str).str.contains(mention, regex=True)
    mention_only = set(conversations.loc[mentions, CONVERSATION_ID]) - has_outbound - with_reply

    selected = with_reply | mention_only
    out = conversations.loc[conversations[CONVERSATION_ID].isin(selected)].copy()
    stats.conversations_with_brand_reply = len(with_reply)
    stats.conversations_mention_only = len(mention_only)
    stats.conversations_selected = len(selected)
    stats.tweets_selected = len(out)
    logger.info("Brand filter: %s", stats.to_dict())
    return out, stats
