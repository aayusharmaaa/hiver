"""Mask rail-specific entities so embeddings reflect the customer's goal, not the route.

Without masking, KMeans on opening messages splits by geography ("euston manchester" vs
"london glasgow") instead of intent. The place lexicon covers stations/cities on the West
Coast and East Coast networks that appear in the dataset; it is intentionally generous.
"""

from __future__ import annotations

import re

PLACES = sorted(
    {
        "london", "euston", "kings cross", "king's cross", "kx", "kgx", "eus", "stancras", "st pancras",
        "manchester", "piccadilly", "picc", "mcr", "man", "glasgow", "central", "edinburgh", "waverley",
        "birmingham", "new street", "bham", "brum", "liverpool", "lime street", "preston", "crewe", "carlisle",
        "wolverhampton", "wolves", "stoke", "stoke on trent", "lancaster", "warrington", "bank quay", "wigan",
        "watford", "watford junction", "milton keynes", "mk", "rugby", "coventry", "bangor", "holyhead",
        "chester", "llandudno", "llandudno junction", "stafford", "oxenholme", "penrith", "lockerbie",
        "motherwell", "newcastle", "york", "leeds", "bradford", "doncaster", "peterborough", "stevenage",
        "inverness", "aberdeen", "sheffield", "nottingham", "derby", "northampton", "tamworth", "lichfield",
        "nuneaton", "sandwell", "dudley", "birmingham international", "bhi", "runcorn", "lockerbie", "wrexham",
        "shrewsbury", "harrogate", "darlington", "durham", "berwick", "newark", "grantham", "hull", "lincoln",
        "stirling", "perth", "dundee", "oban", "fort william", "blackpool", "wakefield", "bolton", "macclesfield",
        "stockport", "rochdale", "hemel hempstead", "hemel", "berkhamsted", "tring", "leighton buzzard",
        "bletchley", "wembley", "willesden", "harrow", "kilburn", "ealing", "reading", "oxford", "scotland",
        "wales", "england", "uk", "heathrow", "gatwick", "airport", "nottingham", "ashford",
    },
    key=len,
    reverse=True,
)
_PLACE_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in PLACES) + r")\b")
_TIME_RE = re.compile(r"\b\d{1,2}[:.]?\d{2}\s?(?:am|pm)?\b|\b\d{1,2}\s?(?:am|pm)\b")
_COACH_RE = re.compile(r"\b(?:coach|carriage|seat|platform|row|table)\s+[a-z]?\d*[a-z]?\b")
_DAY_RE = re.compile(r"\b(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b")
_COST_RE = re.compile(r"£\s?\d[\d,.]*")


def mask_entities(text: str) -> str:
    """Expects already lowercased text. Replaces entities with placeholder tokens."""
    t = _COST_RE.sub(" <money> ", text)
    t = _TIME_RE.sub(" <time> ", t)
    t = _COACH_RE.sub(" <seat> ", t)
    t = _DAY_RE.sub(" <day> ", t)
    t = _PLACE_RE.sub(" <place> ", t)
    return " ".join(t.split())
