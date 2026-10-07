"""Deterministic coverage-oriented sampling over several categorical dimensions.

The sample is for *inspection and labelling coverage*, not for estimating prevalence: rarer
levels are over-represented on purpose (square-root allocation sits between proportional
and uniform). Use `stratum_weights` to re-weight if prevalence estimates are needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LENGTH_BUCKETS = ((0, 2, "short_1-2"), (3, 4, "medium_3-4"), (5, 8, "long_5-8"), (9, 10**9, "very_long_9+"))
PERIOD_START = pd.Timestamp("2017-10-09", tz="UTC")


def length_bucket(turn_count: int) -> str:
    for lo, hi, label in LENGTH_BUCKETS:
        if lo <= turn_count <= hi:
            return label
    return "unknown"


def time_period(ts: pd.Timestamp | None) -> str:
    """Week-start label (Monday) for the main collection window; older tweets are pooled."""
    if ts is None or pd.isna(ts):
        return "unknown_time"
    if ts < PERIOD_START:
        return "before_2017-10-09"
    return (ts - pd.Timedelta(days=ts.weekday())).strftime("week_%Y-%m-%d")


def add_coverage_columns(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["length_bucket"] = out["turn_count"].map(length_bucket)
    out["time_period"] = [time_period(t) for t in out["first_timestamp"]]
    return out


def _sqrt_targets(codes: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(codes).astype(float)
    weights = np.sqrt(counts)
    return np.maximum(n * weights / weights.sum(), 1.0)


def diversity_sample(frame: pd.DataFrame, dimensions: list[str], n: int, seed: int = 42) -> pd.Index:
    """Greedy selection that repeatedly picks the row filling the most under-represented levels."""
    n = min(n, len(frame))
    if n == 0:
        return frame.index[:0]
    rng = np.random.default_rng(seed)
    codes, targets, counts = [], [], []
    for dim in dimensions:
        c, _ = pd.factorize(frame[dim].astype(str))
        codes.append(c)
        targets.append(_sqrt_targets(c, n))
        counts.append(np.zeros(c.max() + 1))

    chosen = np.zeros(len(frame), dtype=bool)
    noise = rng.random(len(frame)) * 1e-6
    picked: list[int] = []
    for _ in range(n):
        score = noise.copy()
        for c, t, cn in zip(codes, targets, counts):
            score += ((t - cn) / t)[c]
        score[chosen] = -np.inf
        i = int(np.argmax(score))
        chosen[i] = True
        picked.append(i)
        for c, cn in zip(codes, counts):
            cn[c[i]] += 1
    return frame.index[np.array(sorted(picked))]


def stratum_weights(population: pd.DataFrame, sample: pd.DataFrame, column: str) -> pd.Series:
    """Post-stratification weight per sampled row: population share / sample share of its level."""
    pop = population[column].astype(str).value_counts(normalize=True)
    smp = sample[column].astype(str).value_counts(normalize=True)
    ratio = (pop / smp).fillna(0.0)
    return sample[column].astype(str).map(ratio)
