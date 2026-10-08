"""Golden evaluation set: build the human-labeling pack for the frozen 250 golden candidates, and store the labels safely.

The pack (data/golden/virgintrains_golden_v1.csv) holds only what a labeler needs to see (opening message, conversation,
provenance ids) plus five blank human columns. It deliberately contains no candidate intent, cluster, historical resolution
label or any agent output, so the labels are independent of the system being evaluated.

The intent vocabulary is the CANDIDATE taxonomy (configs/virgintrains_intents.yaml, not human-validated) used as a reference
list, plus NEW:<snake_case> for anything it does not cover. Labels are never pre-filled and are written only by `GoldenLabelStore`,
which re-verifies the frozen sample before every read and write and refuses CSV edits that did not go through the tool.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.inspection import conversation_text
from evaluation.labeling_store import MAX_NOTES, NEW_PATTERN, LabelStoreError, LabelValidationError, parse_conversation
from evaluation.splits import DEV, GOLDEN, TRAIN, LeakageError, verify_no_leakage
from taxonomy.registry import case_ids_sha256, intent_names, sha256_file

PACK_CSV = "virgintrains_golden_v1.csv"
PACK_MANIFEST = "virgintrains_golden_v1.manifest.json"
AUDIT_LOG = "virgintrains_golden_v1_label_audit.jsonl"
CANDIDATES_PARQUET = "virgintrains_golden_candidates.parquet"
EXPECTED_CASES = 250
PACK_SEED = 42

SOURCE_COLUMNS = ["case_id", "labeling_order", "first_customer_message", "conversation", "first_timestamp", "source_tweet_ids", "conversation_id"]
GOLD_COLUMNS = ["gold_intent", "gold_should_escalate", "gold_resolution_type", "gold_confidence", "human_notes"]
REQUIRED_GOLD = ["gold_intent", "gold_should_escalate", "gold_resolution_type"]
ESCALATE_VALUES = ("yes", "no")
CONFIDENCE_VALUES = ("high", "medium", "low")

TAXONOMY_HEADING = "Candidate taxonomy — provisional; choose the best-supported intent. You may use NEW:<snake_case> if none fits."

LABEL_DEFINITIONS = {
    "gold_intent": "The primary customer-support intent of the incoming message (the customer's main request, judged from their own words).",
    "gold_should_escalate": (
        "yes = route to a human; no = safe for THIS project's support AI to auto-handle. The AI can only reply on Twitter using "
        "evidence from past VirginTrains replies: it has no live train data, cannot look up bookings or accounts, cannot issue refunds "
        "or compensation, and cannot take operational action."
    ),
    "gold_resolution_type": "The type of handling that would appropriately resolve the request (not necessarily what the brand actually did).",
    "gold_confidence": "Optional: how sure you are of this label.",
    "human_notes": "Free text. Required when gold_intent is NEW:<name> (explain why no candidate intent fits).",
}

GOLD_RESOLUTION_TYPES: dict[str, str] = {
    "information_provided": "answer with general information (policy, how things work, what to expect)",
    "self_service": "point to a link, the app, the website, a form or live-updates page",
    "troubleshooting": "give steps to try (e.g. reconnecting to the wifi)",
    "refund": "a refund needs to be discussed or processed",
    "compensation": "compensation or a claim (e.g. Delay Repay) needs to be discussed",
    "feedback_acknowledged": "thank the customer or acknowledge and pass on their feedback",
    "clarification_requested": "ask the customer for missing information before anything else can be done",
    "escalated": "a person must take over: account/booking lookup, DM, Customer Relations, formal complaint, another operator, staff action",
    "unresolved": "nothing reasonable can resolve it (no actionable request, or outside what the brand can do)",
    "other": "none of the above fits",
}

ESCALATION_GUIDANCE = [
    "requests that depend on live service status (is my train running / delayed right now)",
    "anything needing an account, booking or reference lookup",
    "disputed or specific refunds and compensation",
    "formal complaints, staff conduct, safety, accessibility, legal or media threats, other exceptional cases",
    "ambiguous or multi-intent requests where one reply cannot safely cover everything",
    "too little context to know what the customer needs",
    "requests that need an operational action (fix the heating, hold a train, find lost property)",
]


class GoldenIntegrityError(RuntimeError):
    """The golden candidates or the labeling pack are not what was frozen. Nothing may be written."""


# --------------------------------------------------------------------------------------------------------------------
# Vocabulary and validation
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldVocabulary:
    intents: tuple[str, ...]
    should_escalate: tuple[str, ...] = ESCALATE_VALUES
    resolution_types: tuple[str, ...] = tuple(GOLD_RESOLUTION_TYPES)
    confidence: tuple[str, ...] = CONFIDENCE_VALUES

    @classmethod
    def from_registry(cls, taxonomy: dict) -> "GoldVocabulary":
        return cls(intents=tuple(intent_names(taxonomy)))

    def as_dict(self) -> dict[str, list[str]]:
        return {"intents": list(self.intents), "should_escalate": list(self.should_escalate), "resolution_types": list(self.resolution_types), "confidence": list(self.confidence)}


def validate_gold_label(values: dict[str, Any], vocab: GoldVocabulary) -> dict[str, str]:
    """Normalise and validate one label; raises LabelValidationError listing every problem. NEW:<name> is kept exactly."""
    errors: dict[str, str] = {}
    clean: dict[str, str] = {}

    intent = str(values.get("gold_intent", "") or "").strip()
    if not intent:
        errors["gold_intent"] = "required"
    elif intent.lower().startswith("new:"):
        name = intent[4:].strip()
        if not NEW_PATTERN.match(f"NEW:{name}"):
            errors["gold_intent"] = "NEW:<name> must be snake_case: a lower-case letter, then 2-39 lower-case letters, digits or underscores"
        elif name in vocab.intents:
            errors["gold_intent"] = f"{name!r} is already a candidate intent; choose it from the list instead of NEW:"
        else:
            clean["gold_intent"] = f"NEW:{name}"
    elif intent in vocab.intents:
        clean["gold_intent"] = intent
    else:
        errors["gold_intent"] = f"{intent!r} is not a candidate intent or NEW:<name>"

    for field, allowed, required in (
        ("gold_should_escalate", vocab.should_escalate, True),
        ("gold_resolution_type", vocab.resolution_types, True),
        ("gold_confidence", vocab.confidence, False),
    ):
        raw = str(values.get(field, "") or "").strip().lower()
        if not raw and required:
            errors[field] = "required"
        elif raw and raw not in allowed:
            errors[field] = f"{raw!r} is not allowed"
        else:
            clean[field] = raw

    notes = str(values.get("human_notes", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(notes) > MAX_NOTES:
        errors["human_notes"] = f"too long ({len(notes)} > {MAX_NOTES} characters)"
    elif clean.get("gold_intent", "").startswith("NEW:") and not notes:
        errors["human_notes"] = "explain why no candidate intent fits (required with NEW:<name>)"
    clean["human_notes"] = notes

    unknown = sorted(set(values) - set(GOLD_COLUMNS))
    if unknown:
        errors["_"] = f"unknown fields: {unknown}"
    if errors:
        raise LabelValidationError(errors)
    return {c: clean[c] for c in GOLD_COLUMNS}


def case_status(row: pd.Series | dict) -> str:
    required = [str(row[c]).strip() != "" for c in REQUIRED_GOLD]
    optional = [str(row[c]).strip() != "" for c in ("gold_confidence", "human_notes")]
    if all(required):
        return "labelled"
    return "partial" if any(required) or any(optional) else "unlabelled"


# --------------------------------------------------------------------------------------------------------------------
# Building and verifying the pack
# --------------------------------------------------------------------------------------------------------------------
def labeling_order(case_ids: list[str], seed: int = PACK_SEED) -> dict[str, int]:
    """Deterministic shuffled order (1-based) so the labeling sequence does not follow the sampling strata."""
    ranked = sorted(case_ids, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    return {c: i + 1 for i, c in enumerate(ranked)}


def build_pack_frame(golden: pd.DataFrame, seed: int = PACK_SEED) -> pd.DataFrame:
    """Labeling rows for the golden candidates: what a labeler needs, provenance ids, and blank human columns. Nothing else."""
    order = labeling_order(list(golden["case_id"]), seed)
    frame = pd.DataFrame(
        {
            "case_id": golden["case_id"].to_numpy(),
            "labeling_order": golden["case_id"].map(order).to_numpy(),
            "first_customer_message": golden["opening_message"].fillna("").to_numpy(),
            "conversation": [conversation_text(list(t)) for t in golden["full_turns"]],
            "first_timestamp": golden["first_timestamp"].astype(str).to_numpy() if "first_timestamp" in golden else "",
            "source_tweet_ids": [json.dumps([int(x) for x in ids]) for ids in golden["source_tweet_ids"]],
            "conversation_id": golden["conversation_id"].astype(str).to_numpy(),
        }
    )
    for col in GOLD_COLUMNS:
        frame[col] = ""
    return frame.sort_values("labeling_order").reset_index(drop=True)[SOURCE_COLUMNS + GOLD_COLUMNS]


def source_fingerprint(frame: pd.DataFrame) -> str:
    """sha256 of every non-human column, row order included, computed on the CSV as re-read as strings."""
    source = frame[[c for c in frame.columns if c not in GOLD_COLUMNS]].astype(str)
    payload = json.dumps({"columns": list(source.columns), "rows": source.values.tolist()}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_golden_candidates(
    golden: pd.DataFrame,
    candidates_sha256: str,
    split_manifest: dict,
    assigned: pd.DataFrame,
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, int]:
    """Check the frozen golden candidates; return the checks performed (all zero). Raise GoldenIntegrityError otherwise.

    `assigned` is the split assignments joined with the case table (customer_id, conversation_id, group_id, source/context tweet ids,
    opening_message), as `evaluation.splits.verify_no_leakage` expects.
    """
    if candidates_sha256 != split_manifest.get("file_sha256", {}).get("golden_candidates"):
        raise GoldenIntegrityError("the golden candidates file differs from the one recorded in the split manifest (the sample was altered)")
    ids = list(golden["case_id"])
    if len(ids) != expected_cases or split_manifest.get("counts", {}).get(GOLDEN) != expected_cases:
        raise GoldenIntegrityError(f"expected exactly {expected_cases} golden candidates, found {len(ids)} (manifest: {split_manifest.get('counts', {}).get(GOLDEN)})")
    if len(set(ids)) != len(ids):
        raise GoldenIntegrityError("duplicate case ids among the golden candidates")
    golden_split = set(assigned.loc[assigned["split"] == GOLDEN, "case_id"])
    if set(ids) != golden_split:
        raise GoldenIntegrityError("golden candidate membership differs from the split assignments")
    split_of = assigned.set_index("case_id")["split"]
    in_train_or_dev = [c for c in ids if split_of.get(c) in (TRAIN, DEV)]
    if in_train_or_dev:
        raise GoldenIntegrityError(f"{len(in_train_or_dev)} golden case(s) are in train/dev, e.g. {in_train_or_dev[0]}")
    try:
        checks = verify_no_leakage(assigned)
    except LeakageError as exc:
        raise GoldenIntegrityError(str(exc)) from exc
    return {"golden_in_train_or_dev": 0, **checks}


def load_assigned(processed_dir: str | Path) -> pd.DataFrame:
    p = Path(processed_dir)
    assignments = pd.read_csv(p / "splits" / "virgintrains_split_assignments.csv", dtype={"customer_id": str})
    cases = pd.read_parquet(p / "virgintrains_cases.parquet", columns=["case_id", "opening_message", "source_tweet_ids", "context_tweet_ids"])
    return assignments.merge(cases, on="case_id", how="left")


def build_manifest(frame: pd.DataFrame, *, candidates_sha256: str, registry_path: Path, registry_status: str | None, checks: dict[str, int], vocab: GoldVocabulary) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "workflow": "golden_eval_v1 (human labels against the candidate taxonomy as a reference)",
        "n_cases": int(len(frame)),
        "golden_case_ids_sha256": case_ids_sha256(frame["case_id"]),
        "golden_candidates_parquet_sha256": candidates_sha256,
        "source_columns_sha256": source_fingerprint(frame),
        "labeling_order_seed": PACK_SEED,
        "taxonomy_reference": {"path": registry_path.name, "status": registry_status, "sha256": sha256_file(registry_path), "note": TAXONOMY_HEADING},
        "allowed": vocab.as_dict() | {"new_intent_pattern": "NEW:<snake_case>"},
        "label_definitions": LABEL_DEFINITIONS,
        "resolution_type_definitions": GOLD_RESOLUTION_TYPES,
        "leakage_checks_all_zero": checks,
        "notes": [
            "Human columns start blank and are written only by scripts/label_golden_eval.py (audited).",
            "No candidate intent, cluster, historical resolution label or agent output is stored in the pack or shown to the labeler.",
            "Golden cases are never used for retrieval, prompts, thresholds or taxonomy changes; final evaluation only.",
        ],
    }


def prepare_golden_pack(golden_dir: str | Path, processed_dir: str | Path, registry_path: str | Path, expected_cases: int = EXPECTED_CASES) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the frozen golden candidates and write the blank labeling pack + manifest. Nothing is written if a check fails.

    Refuses to overwrite a pack once any label (or audit record) exists.
    """
    golden_dir, processed_dir, registry_path = Path(golden_dir), Path(processed_dir), Path(registry_path)
    out_csv, out_manifest, audit = golden_dir / PACK_CSV, golden_dir / PACK_MANIFEST, golden_dir / AUDIT_LOG
    if replay_audit(audit):
        raise GoldenIntegrityError(f"{audit} records labels: labeling has started; refusing to rebuild the golden pack")
    if out_csv.exists():
        existing = pd.read_csv(out_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if any((existing[c].str.strip() != "").any() for c in GOLD_COLUMNS if c in existing):
            raise GoldenIntegrityError(f"{out_csv} already contains labels; refusing to overwrite")

    parquet = golden_dir / CANDIDATES_PARQUET
    golden = pd.read_parquet(parquet)
    split_manifest = json.loads((processed_dir / "splits" / "virgintrains_split_manifest.json").read_text(encoding="utf-8"))
    checks = verify_golden_candidates(golden, sha256_file(parquet), split_manifest, load_assigned(processed_dir), expected_cases)

    taxonomy = yaml.safe_load(registry_path.read_text(encoding="utf-8"))["taxonomy"]
    vocab = GoldVocabulary.from_registry(taxonomy)
    frame = build_pack_frame(golden)
    golden_dir.mkdir(parents=True, exist_ok=True)
    tmp = golden_dir / f".{PACK_CSV}.prepare.tmp"
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    written = pd.read_csv(tmp, dtype=str, keep_default_na=False)
    manifest = build_manifest(written, candidates_sha256=sha256_file(parquet), registry_path=registry_path, registry_status=taxonomy.get("status"), checks=checks, vocab=vocab)
    os.replace(tmp, out_csv)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return written, manifest


# --------------------------------------------------------------------------------------------------------------------
# Label store
# --------------------------------------------------------------------------------------------------------------------
def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def replay_audit(path: Path) -> dict[str, dict[str, str]]:
    """Label state implied by the audit log (last save/clear per case). Raises on an unreadable log."""
    state: dict[str, dict[str, str]] = {}
    if not path.exists():
        return state
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            raise LabelStoreError(f"audit log line {n} is not valid JSON; restore it before labeling") from None
        if rec.get("event") == "save":
            state[rec["case_id"]] = {c: str(rec["after"].get(c, "")) for c in GOLD_COLUMNS}
        elif rec.get("event") == "clear":
            state[rec["case_id"]] = {c: "" for c in GOLD_COLUMNS}
    return state


class GoldenLabelStore:
    """Reads and writes ONLY the five human columns of virgintrains_golden_v1.csv, re-verifying the frozen sample each time."""

    def __init__(self, golden_dir: str | Path, vocab: GoldVocabulary, *, split_manifest_path: str | Path, assignments_path: str | Path, registry_path: str | Path, labeler: str = ""):
        self.golden_dir = Path(golden_dir).resolve()
        self.csv_path = self.golden_dir / PACK_CSV
        self.manifest_path = self.golden_dir / PACK_MANIFEST
        self.audit_path = self.golden_dir / AUDIT_LOG
        self.backup_path = self.golden_dir / (PACK_CSV + ".bak")
        self.candidates_path = self.golden_dir / CANDIDATES_PARQUET
        self.split_manifest_path = Path(split_manifest_path)
        self.assignments_path = Path(assignments_path)
        self.registry_path = Path(registry_path)
        if not self.manifest_path.exists() or not self.csv_path.exists():
            raise LabelStoreError(f"{PACK_CSV} / {PACK_MANIFEST} not found. Run: python scripts/prepare_golden_eval.py")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.vocab = vocab
        if list(vocab.intents) != list(self.manifest["allowed"]["intents"]):
            raise LabelStoreError("the intent vocabulary differs from the one recorded when the pack was prepared")
        self.labeler = labeler
        self.session_id = uuid.uuid4().hex[:12]
        self._lock = threading.RLock()
        self._frame = self._read_verified()

    def start_session(self) -> None:
        self._audit({"event": "session_start", "n_cases": len(self._frame), "labelled": self.counts()["labelled"]})

    def _read_verified(self) -> pd.DataFrame:
        m = self.manifest
        split_manifest = json.loads(self.split_manifest_path.read_text(encoding="utf-8"))
        candidates_sha = sha256_file(self.candidates_path)
        if candidates_sha != m["golden_candidates_parquet_sha256"] or candidates_sha != split_manifest["file_sha256"]["golden_candidates"]:
            raise LabelStoreError("the golden candidates file changed since the pack was prepared; refusing to continue")
        if sha256_file(self.registry_path) != m["taxonomy_reference"]["sha256"]:
            raise LabelStoreError("the candidate taxonomy file changed since the pack was prepared; refusing to continue")
        a = pd.read_csv(self.assignments_path, usecols=["case_id", "split"])
        if case_ids_sha256(a.loc[a["split"] == GOLDEN, "case_id"]) != m["golden_case_ids_sha256"]:
            raise LabelStoreError("the golden split membership changed since the pack was prepared")
        frame = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if list(frame.columns) != SOURCE_COLUMNS + GOLD_COLUMNS:
            raise LabelStoreError(f"unexpected columns in {PACK_CSV}; expected {SOURCE_COLUMNS + GOLD_COLUMNS}")
        if frame["case_id"].duplicated().any() or len(frame) != m["n_cases"]:
            raise LabelStoreError("rows were added, removed or duplicated in the golden pack")
        if case_ids_sha256(frame["case_id"]) != m["golden_case_ids_sha256"]:
            raise LabelStoreError("the case ids in the golden pack differ from the frozen golden set")
        if source_fingerprint(frame) != m["source_columns_sha256"]:
            raise LabelStoreError("messages, conversations, provenance or order in the golden pack were edited; restore it from git")
        expected = replay_audit(self.audit_path)
        blank = {c: "" for c in GOLD_COLUMNS}
        edited = [r.case_id for r in frame.itertuples(index=False) if {c: getattr(r, c) for c in GOLD_COLUMNS} != expected.get(r.case_id, blank)]
        if edited:
            raise LabelStoreError(f"{len(edited)} label(s) in {PACK_CSV} were changed outside the labeling tool (e.g. {edited[0]}); restore the file from the .bak or git")
        return frame

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
        tmp = self.golden_dir / f".{PACK_CSV}.{self.session_id}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(frame.to_csv(index=False, lineterminator=line_end))
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp, self.csv_path)
                return
            except PermissionError:
                time.sleep(0.25 * (attempt + 1))
        tmp.unlink(missing_ok=True)
        raise LabelStoreError(f"could not replace {PACK_CSV} (is it open in another program?)")

    def _index(self, frame: pd.DataFrame, case_id: str) -> int:
        hits = frame.index[frame["case_id"] == case_id]
        if len(hits) != 1:
            raise LabelStoreError(f"unknown case_id {case_id!r}")
        return int(hits[0])

    def counts(self, frame: pd.DataFrame | None = None) -> dict[str, int]:
        frame = self._frame if frame is None else frame
        statuses = frame.apply(case_status, axis=1) if len(frame) else pd.Series(dtype=str)
        return {"total": len(frame), **{s: int((statuses == s).sum()) for s in ("labelled", "partial", "unlabelled")}}

    def verified_frame(self) -> pd.DataFrame:
        with self._lock:
            self._frame = self._read_verified()
            return self._frame.copy()

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._frame = self._read_verified()
            statuses = self._frame.apply(case_status, axis=1)
            cases = [{"case_id": r.case_id, "order": int(r.labeling_order), "status": s} for r, s in zip(self._frame.itertuples(), statuses)]
            first_open = next((i for i, c in enumerate(cases) if c["status"] != "labelled"), None)
            return {"counts": self.counts(self._frame), "cases": cases, "first_open_index": first_open, "labeler": self.labeler}

    def get_case(self, case_id: str) -> dict[str, Any]:
        """What the labeler sees for one case: the conversation and their own labels. No candidate or system information."""
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
                "labels": {c: row[c] for c in GOLD_COLUMNS},
                "status": case_status(row),
            }

    def save_label(self, case_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            clean = validate_gold_label(values, self.vocab)
            frame = self._read_verified()
            i = self._index(frame, case_id)
            before = {c: frame.at[i, c] for c in GOLD_COLUMNS}
            if before == clean:
                self._frame = frame
                return {"saved": False, "reason": "unchanged", "counts": self.counts(frame), "status": case_status(frame.loc[i])}
            for c in GOLD_COLUMNS:
                frame.at[i, c] = clean[c]
            self._write_atomic(frame)
            self._frame = frame
            self._audit({"event": "save", "case_id": case_id, "before": before, "after": clean, "labelled_total": self.counts(frame)["labelled"]})
            return {"saved": True, "counts": self.counts(frame), "status": "labelled"}

    def clear_label(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            frame = self._read_verified()
            i = self._index(frame, case_id)
            before = {c: frame.at[i, c] for c in GOLD_COLUMNS}
            if not any(before.values()):
                self._frame = frame
                return {"cleared": False, "counts": self.counts(frame)}
            for c in GOLD_COLUMNS:
                frame.at[i, c] = ""
            self._write_atomic(frame)
            self._frame = frame
            self._audit({"event": "clear", "case_id": case_id, "before": before})
            return {"cleared": True, "counts": self.counts(frame)}


# --------------------------------------------------------------------------------------------------------------------
# Reference material and guide (shown to the labeler; contains no system output)
# --------------------------------------------------------------------------------------------------------------------
def build_reference(taxonomy: dict[str, Any], confusable_notes: list[dict]) -> dict[str, Any]:
    from evaluation.labeling_support import build_reference as candidate_cards

    cards = candidate_cards(taxonomy, confusable_notes)
    return {
        "heading": TAXONOMY_HEADING,
        "status": taxonomy.get("status"),
        "intents": [
            {k: c[k] for k in ("name", "is_fallback", "definition", "fits_when", "does_not_fit_when", "confusable_intents", "historical_examples")}
            for c in cards["intents"]
        ],
        "confusable_notes": cards["confusable_notes"],
        "label_definitions": LABEL_DEFINITIONS,
        "resolution_types": GOLD_RESOLUTION_TYPES,
        "escalation_guidance": ESCALATION_GUIDANCE,
    }


def guide_markdown() -> str:
    lines = [
        "# Golden evaluation set: labeling guide",
        "",
        "You are creating the reference labels the support agent will be evaluated against. Label each case from the conversation alone: "
        "the tool never shows what the system predicts, and nothing is pre-filled.",
        "",
        "## Fields",
        "",
    ]
    lines += [f"- **`{k}`**: {v}" for k, v in LABEL_DEFINITIONS.items()]
    lines += ["", "## Intent", "", f"> {TAXONOMY_HEADING}", "",
              "- Pick the customer's *primary* request. If there is a clear second request, mention it in the notes.",
              "- The list is a provisional, data-derived draft. Do not force a case into the nearest intent: use `NEW:<snake_case>` and explain in the notes.",
              "- `unclear_or_media_only` is for messages with no usable request (mention only, photo or link only).",
              "", "## Should escalate", "",
              "Judge against what this project's AI can actually do. " + LABEL_DEFINITIONS["gold_should_escalate"], "",
              "These usually lean towards **yes** (route to a human), but decide case by case:", ""]
    lines += [f"- {g}" for g in ESCALATION_GUIDANCE]
    lines += ["", "Lean towards **no** when a general, evidence-backed reply (information, a link, simple steps, thanks) would genuinely resolve it.",
              "", "## Resolution type", "", "What handling would appropriately resolve the request:", ""]
    lines += [f"- `{k}`: {v}" for k, v in GOLD_RESOLUTION_TYPES.items()]
    lines += ["", "## Conversation", "", "Oldest message first. The brand's historical replies are shown as context; they are what happened, not "
              "necessarily the right handling, and not a hint about the label."]
    return "\n".join(lines) + "\n"
