"""Safe, auditable storage for the human calibration labels.

The labeling tool never writes anything except the five `human_*` cells of the calibration CSV (plus a rolling `.bak` and an
append-only audit log). Before every write the store re-reads the CSV from disk and re-verifies it, so:

  * source tweet ids, case ids and every other provenance/sampling column cannot change (fingerprint from the manifest);
  * the case set still equals the sampled set and none of it is golden (hashes from the manifest, split assignments);
  * edits made to the CSV by other means (e.g. a spreadsheet) are never overwritten with a stale in-memory copy.

Nothing here generates or suggests a label. Values only come from the reviewer.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.splits import GOLDEN, RESERVE
from evaluation.taxonomy_calibration import (
    ESCALATION_SIGNALS,
    HUMAN_COLUMNS,
    HUMAN_RESOLUTION_TYPES,
    HUMAN_RESOLVED_VALUES,
    source_columns_sha256,
)
from taxonomy.registry import case_ids_sha256

CSV_NAME = "taxonomy_calibration.csv"
AUDIT_NAME = "taxonomy_calibration_label_audit.jsonl"
REQUIRED_FIELDS = ["human_intent", "human_resolution_type", "human_resolved", "human_escalation_signal"]
NEW_PATTERN = re.compile(r"^NEW:[a-z][a-z0-9_]{2,39}$")
MAX_NOTES = 2000
_TURN = re.compile(r"^\[(CUSTOMER|AGENT|OTHER-AGENT) (\d+)\] ?(.*)$")
SOURCE_COLUMNS_REQUIRED = [
    "case_id", "labeling_order", "first_customer_message", "conversation", "candidate_intent", "source_tweet_ids", "source_split",
    "conversation_id", "customer_id", "group_id", "sampling_stratum", "selection_reason", "stratum_weight", "sampling_seed",
]


class LabelStoreError(RuntimeError):
    """The calibration file or its provenance is not safe to write to."""


class LabelValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class Vocabulary:
    intents: tuple[str, ...]
    resolution_types: tuple[str, ...] = tuple(HUMAN_RESOLUTION_TYPES)
    resolved: tuple[str, ...] = tuple(HUMAN_RESOLVED_VALUES)
    escalation_signals: tuple[str, ...] = tuple(ESCALATION_SIGNALS)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "intents": list(self.intents),
            "resolution_types": list(self.resolution_types),
            "resolved": list(self.resolved),
            "escalation_signals": list(self.escalation_signals),
        }


def vocabulary_from_registry(taxonomy: dict) -> Vocabulary:
    names = [i["name"] for i in taxonomy["intents"]]
    if taxonomy.get("fallback"):
        names.append(taxonomy["fallback"]["name"])
    return Vocabulary(intents=tuple(names))


def validate_label(values: dict[str, Any], vocab: Vocabulary) -> dict[str, str]:
    """Normalise and validate one label. Raises LabelValidationError listing every problem."""
    errors: dict[str, str] = {}
    clean: dict[str, str] = {}

    intent = str(values.get("human_intent", "") or "").strip()
    if not intent:
        errors["human_intent"] = "required"
    elif intent.lower().startswith("new:"):
        name = intent[4:].strip().lower().replace("-", "_").replace(" ", "_")
        candidate = f"NEW:{name}"
        if not NEW_PATTERN.match(candidate):
            errors["human_intent"] = "NEW:<name> must be snake_case: a letter, then 2-39 letters, digits or underscores"
        elif name in vocab.intents:
            errors["human_intent"] = f"{name!r} already exists; pick it from the list instead of NEW:"
        else:
            clean["human_intent"] = candidate
    elif intent.lower() in vocab.intents:
        clean["human_intent"] = intent.lower()
    else:
        errors["human_intent"] = f"{intent!r} is not one of the candidate intents or NEW:<name>"

    for field, allowed in (
        ("human_resolution_type", vocab.resolution_types),
        ("human_resolved", vocab.resolved),
        ("human_escalation_signal", vocab.escalation_signals),
    ):
        raw = str(values.get(field, "") or "").strip().lower()
        if not raw:
            errors[field] = "required"
        elif raw not in allowed:
            errors[field] = f"{raw!r} is not allowed"
        else:
            clean[field] = raw

    notes = str(values.get("human_notes", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(notes) > MAX_NOTES:
        errors["human_notes"] = f"too long ({len(notes)} > {MAX_NOTES} characters)"
    elif clean.get("human_intent", "").startswith("NEW:") and not notes:
        errors["human_notes"] = "a note describing the new intent is required with NEW:<name>"
    clean["human_notes"] = notes

    if errors:
        raise LabelValidationError(errors)
    return clean


def case_status(row: pd.Series | dict) -> str:
    required = [str(row[c]).strip() != "" for c in REQUIRED_FIELDS]
    notes = str(row["human_notes"]).strip() != ""
    if all(required):
        return "labelled"
    if any(required) or notes:
        return "partial"
    return "unlabelled"


def parse_conversation(text: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line in str(text).split("\n"):
        m = _TURN.match(line)
        if m:
            turns.append({"role": m.group(1), "tweet_id": m.group(2), "text": html.unescape(m.group(3))})
        elif turns and line.strip():
            turns[-1]["text"] += " " + html.unescape(line.strip())
    return turns


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class LabelStore:
    def __init__(
        self,
        csv_path: str | Path,
        manifest_path: str | Path,
        vocab: Vocabulary,
        *,
        assignments_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        labeler: str = "",
    ):
        self.csv_path = Path(csv_path).resolve()
        if self.csv_path.name != CSV_NAME:
            raise LabelStoreError(f"refusing to write to {self.csv_path.name!r}: the only writable file is {CSV_NAME}")
        if any(part.lower() == "golden" for part in self.csv_path.parts):
            raise LabelStoreError("the golden directory is never writable from the labeling tool")
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.vocab = vocab
        self.assignments_path = Path(assignments_path) if assignments_path else None
        self.audit_path = Path(audit_path) if audit_path else self.csv_path.with_name(AUDIT_NAME)
        self.backup_path = self.csv_path.with_name(self.csv_path.name + ".bak")
        self.labeler = labeler
        self.session_id = uuid.uuid4().hex[:12]
        self._lock = threading.RLock()
        self._frame = self._read_verified()
        self._revealed = self._load_revealed()

    def start_session(self) -> None:
        """Record that a labelling session began. Not called by read-only checks."""
        self._audit({"event": "session_start", "n_cases": len(self._frame), "labelled": self.counts()["labelled"]})

    def _read_verified(self) -> pd.DataFrame:
        frame = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        missing = [c for c in SOURCE_COLUMNS_REQUIRED + HUMAN_COLUMNS if c not in frame]
        if missing:
            raise LabelStoreError(f"calibration CSV is missing columns: {missing}")
        if list(frame.columns[-len(HUMAN_COLUMNS):]) != HUMAN_COLUMNS:
            raise LabelStoreError("the human_* columns must be the last five columns, in the original order")
        if frame["case_id"].duplicated().any():
            raise LabelStoreError("duplicate case_id rows in the calibration CSV")
        if case_ids_sha256(frame["case_id"]) != self.manifest.get("calibration_case_ids_sha256"):
            raise LabelStoreError("the set of case ids differs from the sampled set recorded in the manifest (rows added, removed or renamed)")
        expected = self.manifest.get("source_columns_sha256")
        if not expected:
            raise LabelStoreError("manifest has no source_columns_sha256; rebuild the calibration pack before labelling")
        if source_columns_sha256(frame) != expected:
            raise LabelStoreError(
                "provenance or sampling columns differ from the generated sample (case ids, tweet ids, order or metadata were edited). "
                "Restore the CSV from the .bak file or from git; the tool will not write to a modified sample."
            )
        if set(frame["source_split"]) != {RESERVE}:
            raise LabelStoreError(f"every calibration case must come from {RESERVE}")
        if self.assignments_path is not None:
            a = pd.read_csv(self.assignments_path, usecols=["case_id", "split"])
            golden = set(a.loc[a["split"] == GOLDEN, "case_id"])
            if golden & set(frame["case_id"]):
                raise LabelStoreError("a golden case is present in the calibration set")
            if case_ids_sha256(golden) != self.manifest.get("golden_case_ids_sha256"):
                raise LabelStoreError("the golden set no longer matches the one recorded when the calibration sample was drawn")
            split_of = a.set_index("case_id")["split"]
            if (frame["case_id"].map(split_of) != RESERVE).any():
                raise LabelStoreError(f"a calibration case is not in {RESERVE}")
        return frame

    def _load_revealed(self) -> set[str]:
        revealed: set[str] = set()
        if self.audit_path.exists():
            for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "reveal":
                    revealed.add(rec["case_id"])
        return revealed

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"ts": _now(), "labeler": self.labeler, "session": self.session_id, **record}
        with open(self.audit_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _write_atomic(self, frame: pd.DataFrame) -> None:
        original = self.csv_path.read_bytes()
        line_end = "\r\n" if b"\r\n" in original[:5000] else "\n"
        self.backup_path.write_bytes(original)
        tmp = self.csv_path.with_name(f".{self.csv_path.name}.{self.session_id}.tmp")
        text = frame.to_csv(index=False, lineterminator=line_end)
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp, self.csv_path)
                return
            except PermissionError:
                time.sleep(0.25 * (attempt + 1))
        tmp.unlink(missing_ok=True)
        raise LabelStoreError("could not replace the calibration CSV (is it open and locked in another program?)")

    def _index(self, frame: pd.DataFrame, case_id: str) -> int:
        hits = frame.index[frame["case_id"] == case_id]
        if len(hits) != 1:
            raise LabelStoreError(f"unknown case_id {case_id!r}")
        return int(hits[0])

    def counts(self, frame: pd.DataFrame | None = None) -> dict[str, int]:
        frame = self._frame if frame is None else frame
        statuses = frame.apply(case_status, axis=1)
        return {
            "total": len(frame),
            "labelled": int((statuses == "labelled").sum()),
            "partial": int((statuses == "partial").sum()),
            "unlabelled": int((statuses == "unlabelled").sum()),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._frame = self._read_verified()
            statuses = self._frame.apply(case_status, axis=1)
            cases = [{"case_id": r.case_id, "order": int(r.labeling_order), "status": s} for r, s in zip(self._frame.itertuples(), statuses)]
            first_open = next((i for i, c in enumerate(cases) if c["status"] != "labelled"), None)
            return {"counts": self.counts(self._frame), "cases": cases, "first_open_index": first_open, "labeler": self.labeler}

    def get_case(self, case_id: str) -> dict[str, Any]:
        """Everything needed to label a case. Deliberately contains no candidate/system-suggested information."""
        with self._lock:
            self._frame = self._read_verified()
            i = self._index(self._frame, case_id)
            row = self._frame.loc[i]
            return {
                "case_id": case_id,
                "index": i,
                "total": len(self._frame),
                "order": int(row["labeling_order"]),
                "first_timestamp": row["first_timestamp"],
                "turns": parse_conversation(row["conversation"]),
                "labels": {c: row[c] for c in HUMAN_COLUMNS},
                "status": case_status(row),
                "suggestion_revealed": case_id in self._revealed,
            }

    def get_suggestion(self, case_id: str) -> dict[str, Any]:
        """The machine's suggestion for one case. Calling this is recorded in the audit log (reveal)."""
        with self._lock:
            frame = self._read_verified()
            i = self._index(frame, case_id)
            row = frame.loc[i]
            margins = pd.to_numeric(frame["cluster_margin"], errors="coerce")
            margin = float(pd.to_numeric(row["cluster_margin"], errors="coerce"))
            if case_id not in self._revealed:
                self._revealed.add(case_id)
                self._audit({"event": "reveal", "case_id": case_id, "labelled_at_reveal": case_status(row) == "labelled"})
            return {
                "candidate_intent": row["candidate_intent"],
                "candidate_cluster": row["candidate_cluster_name"],
                "runner_up_intent": row["runner_up_intent"] or None,
                "runner_up_cluster": row["runner_up_cluster_name"] or None,
                "cluster_margin": margin,
                "margin_percentile_in_sample": round(float((margins <= margin).mean()), 3),
                "auto_resolution_type": row["auto_resolution_type"],
                "auto_resolved": row["auto_resolved"],
                "auto_dm_redirect": row["auto_dm_redirect"],
                "auto_resolution_outcome": row["auto_resolution_outcome"],
                "auto_resolution_summary": row["auto_resolution_summary"],
                "sampling_stratum": row["sampling_stratum"],
                "selection_reason": row["selection_reason"],
            }

    def save_label(self, case_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            clean = validate_label(values, self.vocab)
            frame = self._read_verified()
            i = self._index(frame, case_id)
            before = {c: frame.at[i, c] for c in HUMAN_COLUMNS}
            if before == {c: clean[c] for c in HUMAN_COLUMNS}:
                self._frame = frame
                return {"saved": False, "reason": "unchanged", "counts": self.counts(frame)}
            for c in HUMAN_COLUMNS:
                frame.at[i, c] = clean[c]
            self._write_atomic(frame)
            self._frame = frame
            self._audit(
                {
                    "event": "save",
                    "case_id": case_id,
                    "before": before,
                    "after": {c: clean[c] for c in HUMAN_COLUMNS},
                    "suggestion_revealed_before_save": case_id in self._revealed,
                    "labelled_total": self.counts(frame)["labelled"],
                }
            )
            return {"saved": True, "counts": self.counts(frame), "status": "labelled"}

    def clear_label(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            frame = self._read_verified()
            i = self._index(frame, case_id)
            before = {c: frame.at[i, c] for c in HUMAN_COLUMNS}
            if not any(before.values()):
                self._frame = frame
                return {"cleared": False, "counts": self.counts(frame)}
            for c in HUMAN_COLUMNS:
                frame.at[i, c] = ""
            self._write_atomic(frame)
            self._frame = frame
            self._audit({"event": "clear", "case_id": case_id, "before": before})
            return {"cleared": True, "counts": self.counts(frame)}
