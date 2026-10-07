"""Synthetic registry / calibration fixtures. Nothing here touches the real data or the real labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluation.taxonomy_calibration import HUMAN_COLUMNS
from taxonomy.registry import case_ids_sha256, sha256_file

INTENTS = [
    "service_status_delay_enquiry", "journey_disruption_complaint", "chitchat_non_support", "customer_service_complaint",
    "delay_repay_refund_claim", "ticket_booking_query", "seat_reservation_issue", "praise_positive_feedback",
    "first_class_catering_issue", "onboard_wifi_issue",
]
FALLBACK = "unclear_or_media_only"


def intent_entry(name: str, n: int = 100, train_ids: tuple[str, ...] = ()) -> dict:
    others = [i for i in INTENTS if i != name][:2]
    return {
        "name": name,
        "status": "candidate_NEEDS_REVIEW",
        "definition": f"Definition of {name}.",
        "inclusion_criteria": [f"{name} fits"],
        "exclusion_criteria": [f"{name} does not fit"],
        "source_clusters": [{"cluster_id": INTENTS.index(name) if name in INTENTS else -1, "name": name}],
        "n_cases": n,
        "pct_resolved": 0.2,
        "pct_dm_redirect": 0.05,
        "positive_examples": [{"text": f"pos {name} {i}", "case_id": cid} for i, cid in enumerate(train_ids)] or ["NEEDS_REVIEW"],
        "negative_examples": [{"text": "neg", "case_id": train_ids[0], "belongs_to": others[0]}] if train_ids else ["NEEDS_REVIEW"],
        "common_resolution_patterns": [f'"reply" (n=5, template: t_{name})'],
        "resolution_types": {"information_provided": 0.6, "self_service": 0.4},
        "required_information": [{"item": "NEEDS_REVIEW", "evidence": "none"}],
        "known_confusable_intents": [{"intent": others[0], "evidence": "x", "distinguishing_note": "NEEDS_REVIEW"}],
        "escalation_triggers": [{"trigger": "t", "source": "historical_dataset", "policy": "NEEDS_REVIEW"}],
        "safe_to_auto_handle": "NEEDS_REVIEW",
        "auto_handle_evidence": {"hint": "unclear", "reason": "NEEDS_REVIEW"},
    }


def candidate_taxonomy(train_ids: tuple[str, ...] = ("train_1", "train_2")) -> dict:
    fb = intent_entry(FALLBACK, 20, train_ids)
    fb["fallback"] = True
    fb["known_confusable_intents"] = [{"intent": "any", "evidence": "x", "distinguishing_note": "NEEDS_REVIEW"}]
    return {
        "taxonomy": {
            "brand": "VirginTrains",
            "registry_schema_version": 2,
            "status": "CANDIDATE_NOT_GROUND_TRUTH",
            "warning": "candidate",
            "calibration": {"status": "NOT_YET_CALIBRATED"},
            "intents": [intent_entry(n, 100 - i, train_ids) for i, n in enumerate(INTENTS)],
            "fallback": fb,
        }
    }


def write_world(root: Path, labelled: bool = True, n_cal: int = 40) -> dict[str, Path]:
    """A miniature processed/golden/configs tree, enough for finalize_taxonomy + prepare_golden_labeling."""
    processed, golden_dir, configs = root / "processed", root / "golden", root / "configs"
    (processed / "splits").mkdir(parents=True)
    golden_dir.mkdir()
    configs.mkdir()

    cal_ids = [f"cal_{i}" for i in range(n_cal)]
    golden_ids = [f"gold_{i}" for i in range(10)]
    assignments = pd.DataFrame(
        [{"case_id": c, "split": "golden_pool_reserve", "split_reason": "", "group_id": c, "customer_id": f"u_{c}", "conversation_id": f"conv_{c}", "cluster_id": 0} for c in cal_ids]
        + [{"case_id": c, "split": "golden_eval", "split_reason": "", "group_id": c, "customer_id": f"u_{c}", "conversation_id": f"conv_{c}", "cluster_id": 0} for c in golden_ids]
        + [{"case_id": c, "split": "train_retrieval", "split_reason": "", "group_id": c, "customer_id": f"u_{c}", "conversation_id": f"conv_{c}", "cluster_id": 0} for c in ("train_1", "train_2")]
    )
    assignments.to_csv(processed / "splits" / "virgintrains_split_assignments.csv", index=False)
    pd.DataFrame({"case_id": ["train_1", "train_2"]}).to_parquet(processed / "splits" / "virgintrains_train_retrieval.parquet")

    turns = lambda cid: [{"turn_index": 0, "tweet_id": 1, "author_id": "u", "role": "customer", "text": f"hello {cid}", "created_at": None}]  # noqa: E731
    pd.DataFrame(
        {
            "case_id": golden_ids,
            "opening_message": [f"gold message {i}" for i in range(10)],
            "full_turns": [turns(c) for c in golden_ids],
            "source_tweet_ids": [[100 + i] for i in range(10)],
            "conversation_id": [f"conv_{c}" for c in golden_ids],
            "golden_stratum_weight": [1.0] * 10,
        }
    ).to_parquet(golden_dir / "virgintrains_golden_candidates.parquet")

    rows = []
    for i, cid in enumerate(cal_ids):
        intent = INTENTS[i % len(INTENTS)]
        row = {
            "case_id": cid, "labeling_order": i + 1, "first_customer_message": f"message {i}", "conversation": "[CUSTOMER 1] hi",
            "candidate_intent": intent, "auto_resolution_type": "information_provided", "auto_resolved": "False", "stratum_weight": "2.0",
        }
        row.update({c: "" for c in HUMAN_COLUMNS})
        if labelled:
            row.update(
                {
                    "human_intent": intent if i % 5 else INTENTS[(i + 1) % len(INTENTS)],
                    "human_resolution_type": "information_provided",
                    "human_resolved": "no",
                    "human_escalation_signal": "none",
                    "human_notes": "",
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(processed / "taxonomy_calibration.csv", index=False)
    (processed / "taxonomy_calibration_manifest.json").write_text(
        json.dumps({"golden_case_ids_sha256": case_ids_sha256(golden_ids), "calibration_case_ids_sha256": case_ids_sha256(cal_ids)}), encoding="utf-8"
    )

    registry = configs / "virgintrains_intents.yaml"
    registry.write_text(yaml.safe_dump(candidate_taxonomy(), sort_keys=False), encoding="utf-8")
    decisions = configs / "virgintrains_taxonomy_decisions.yaml"
    decisions.write_text(yaml.safe_dump({"reviewed_by": "Test Reviewer", "reviewed_on": "2026-10-08", "merges": [], "renames": {}, "drop_to_fallback": [], "new_intents": [], "edits": {}}), encoding="utf-8")
    return {"processed": processed, "golden": golden_dir, "configs": configs, "registry": registry, "decisions": decisions, "frozen": configs / "virgintrains_taxonomy_v1.yaml", "reports": root / "reports"}


def write_calibration_world(root: Path, n: int = 12) -> dict:
    """A miniature, fully consistent calibration pack (CSV + manifest + assignments + golden files), all human_* blank."""
    from evaluation.taxonomy_calibration import source_columns_sha256

    processed, golden_dir = root / "data" / "processed", root / "data" / "golden"
    (processed / "splits").mkdir(parents=True)
    golden_dir.mkdir(parents=True)
    ids = [f"case_{1000 + i}" for i in range(n)]
    golden_ids = [f"case_{9000 + i}" for i in range(5)]
    rows = []
    for i, cid in enumerate(ids):
        rows.append(
            {
                "case_id": cid, "labeling_order": i + 1, "first_customer_message": f"first message {i} &amp; more",
                "conversation": f"[CUSTOMER {1000 + i}] first message {i} &amp; more\n[AGENT {5000 + i}] sorry to hear, comma, \"quoted\" ^AB\n[OTHER-AGENT {6000 + i}] not ours",
                "candidate_intent": INTENTS[i % len(INTENTS)], "candidate_cluster_id": i % 12, "candidate_cluster_name": f"cluster {i % 12}",
                "runner_up_cluster_id": (i + 1) % 12, "runner_up_cluster_name": f"cluster {(i + 1) % 12}", "runner_up_intent": INTENTS[(i + 1) % len(INTENTS)],
                "cluster_margin": round(0.01 * (i + 1), 4), "auto_resolution_type": "information_provided", "auto_resolved": False, "auto_dm_redirect": False,
                "auto_resolution_outcome": "answered", "auto_resolution_summary": "agent answered", "turn_count": 3, "length_bucket": "short",
                "first_timestamp": "2017-10-24 11:33:55+00:00", "customer_id": f"u{i}", "conversation_id": f"conv{i}", "group_id": f"g{i}",
                "source_tweet_ids": json.dumps([1000 + i, 5000 + i]), "source_split": "golden_pool_reserve", "sampling_stratum": INTENTS[i % len(INTENTS)],
                "selection_reason": "diverse_coverage", "stratum_quota": 5, "stratum_allocated": 5, "stratum_eligible": 40, "stratum_selected": 5,
                "stratum_weight": 8.0, "sampling_seed": 42,
            }
        )
    frame = pd.DataFrame(rows)
    for col in HUMAN_COLUMNS:
        frame[col] = ""
    csv = processed / "taxonomy_calibration.csv"
    frame.to_csv(csv, index=False, encoding="utf-8")
    on_disk = pd.read_csv(csv, dtype=str, keep_default_na=False)

    assignments = pd.DataFrame(
        [{"case_id": c, "split": "golden_pool_reserve", "group_id": f"g{i}", "customer_id": f"u{i}", "conversation_id": f"conv{i}"} for i, c in enumerate(ids)]
        + [{"case_id": c, "split": "golden_eval", "group_id": f"gg{i}", "customer_id": f"gu{i}", "conversation_id": f"gconv{i}"} for i, c in enumerate(golden_ids)]
        + [{"case_id": "case_1", "split": "train_retrieval", "group_id": "gt", "customer_id": "ut", "conversation_id": "ct"}]
    )
    assignments_path = processed / "splits" / "virgintrains_split_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    manifest = {
        "calibration_case_ids_sha256": case_ids_sha256(ids),
        "source_columns_sha256": source_columns_sha256(on_disk),
        "golden_case_ids_sha256": case_ids_sha256(golden_ids),
        "notes": ["human_* columns are empty. Do not pre-fill them with model output."],
    }
    manifest_path = processed / "taxonomy_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    golden_files = {
        golden_dir / "virgintrains_golden_candidates.parquet": pd.DataFrame({"case_id": golden_ids, "opening_message": ["g"] * 5}),
    }
    golden_paths = []
    for path, df in golden_files.items():
        df.to_parquet(path)
        golden_paths.append(path)
    csv_golden = golden_dir / "virgintrains_golden_candidates.csv"
    pd.DataFrame({"case_id": golden_ids, "human_intent": [""] * 5}).to_csv(csv_golden, index=False)
    golden_paths.append(csv_golden)
    return {
        "csv": csv, "manifest": manifest_path, "assignments": assignments_path, "golden_files": golden_paths, "processed": processed,
        "golden_dir": golden_dir, "ids": ids, "golden_ids": golden_ids, "taxonomy": candidate_taxonomy()["taxonomy"],
    }


def sha(path: Path) -> str:
    return sha256_file(path)


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
