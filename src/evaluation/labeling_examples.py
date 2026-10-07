"""Worked examples for the labeling guide, pulled from the *train* split only.

The YAML file holds just case ids and a hand-written reading. The conversation text comes from the data at build time,
so a quote can never drift from the dataset, and any case that is not a `train_retrieval` case (golden, reserve, dev,
or one of the calibration cases) is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.inspection import ROLE_LABEL
from evaluation.splits import TRAIN

MAX_TURNS = 6
MAX_CHARS = 400


class ExampleError(ValueError):
    pass


def _clip(text: object, n: int = MAX_CHARS) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def load_guide_examples(
    path: str | Path,
    cases: pd.DataFrame,
    split_of: pd.Series,
    forbidden_ids: set[str],
    candidate_of: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Resolve the example sections. `cases` needs `case_id` and `full_turns`; `split_of` maps case_id -> split name."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    by_id = cases.drop_duplicates("case_id").set_index("case_id")["full_turns"]
    problems: list[str] = []
    out: list[dict[str, Any]] = []
    for section in spec["sections"]:
        resolved = []
        for ex in section["cases"]:
            cid = ex["case_id"]
            if cid in forbidden_ids:
                problems.append(f"{cid}: is a golden/calibration case")
            elif split_of.get(cid) != TRAIN:
                problems.append(f"{cid}: split is {split_of.get(cid)!r}, examples must be {TRAIN}")
            elif cid not in by_id.index:
                problems.append(f"{cid}: not found in the case table")
            else:
                turns = list(by_id[cid])
                resolved.append(
                    {
                        "case_id": cid,
                        "turns": [
                            {"role": ROLE_LABEL.get(t["role"], t["role"]), "tweet_id": t["tweet_id"], "text": _clip(t["text"])}
                            for t in turns[:MAX_TURNS]
                        ],
                        "truncated": len(turns) > MAX_TURNS,
                        "candidate_intent": (candidate_of or {}).get(cid),
                        "reading": " ".join(str(ex["reading"]).split()),
                    }
                )
        out.append({"key": section["key"], "title": section["title"], "intro": " ".join(str(section.get("intro", "")).split()), "cases": resolved})
    if problems:
        raise ExampleError("Invalid guide examples: " + "; ".join(problems))
    return out
