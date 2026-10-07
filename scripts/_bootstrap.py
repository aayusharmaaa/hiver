"""Make `src/` importable when scripts are run directly without `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tweets contain emoji; avoid UnicodeEncodeError on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "twcs.csv"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
