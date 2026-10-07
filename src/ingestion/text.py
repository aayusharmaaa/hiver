"""Deterministic text normalization. Never applied to stored raw text, only to derived keys."""

from __future__ import annotations

import html
import re

_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_DIGITS = re.compile(r"\d+")
# Agent sign-offs such as "-KC", "^JK", "~Ana", "*TM" at the end of a reply.
_SIGNATURE = re.compile(r"(?:\s|(?<=[.!?)]))[\^\-–—~*]\s*[A-Z][A-Za-z]{0,11}\.?\s*$")
_NON_WORD = re.compile(r"[^a-z0-9<>' ]+")
_SPACES = re.compile(r"\s+")


def strip_mentions(text: str) -> str:
    return _SPACES.sub(" ", _MENTION.sub(" ", text)).strip()


def normalize_for_matching(text: str) -> str:
    """Lowercase, drop mentions and sign-offs, mask urls, collapse whitespace."""
    text = _SIGNATURE.sub("", html.unescape(text or ""))
    text = _MENTION.sub(" ", text)
    text = _URL.sub(" <url> ", text)
    text = text.lower().replace("’", "'")
    return _SPACES.sub(" ", text).strip()


def template_key(text: str, n_tokens: int = 8) -> str:
    """A coarse key used to detect repeated (templated) agent replies.

    Digits are masked and punctuation removed so that "DM us your order #123" and
    "DM us your order #456" collapse to the same key.
    """
    norm = normalize_for_matching(text)
    norm = _DIGITS.sub("0", norm)
    norm = _NON_WORD.sub(" ", norm)
    tokens = norm.split()
    return " ".join(tokens[:n_tokens])


def latin_share(text: str) -> float:
    """Share of alphabetic characters that are ASCII letters (rough English/Latin-script proxy)."""
    letters = [c for c in strip_mentions(_URL.sub(" ", text or "")) if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() for c in letters) / len(letters)
