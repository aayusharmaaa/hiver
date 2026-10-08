"""Read-only checks of the real calibration pack and labeling guide. Nothing here writes to data/ or configs/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evaluation.labeling_store import AUDIT_NAME, LabelStore, vocabulary_from_registry
from evaluation.labeling_examples import ExampleError, load_guide_examples
from evaluation.splits import GOLDEN, RESERVE, TRAIN
from evaluation.taxonomy_calibration import ESCALATION_SIGNALS, HUMAN_COLUMNS, HUMAN_RESOLUTION_TYPES, HUMAN_RESOLVED_VALUES, source_columns_sha256
from taxonomy.registry import registry_status

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CSV = PROCESSED / "taxonomy_calibration.csv"
MANIFEST = PROCESSED / "taxonomy_calibration_manifest.json"
ASSIGNMENTS = PROCESSED / "splits" / "virgintrains_split_assignments.csv"
GUIDE = PROCESSED / "taxonomy_calibration_guide.md"
LABELING_MD = PROCESSED / "taxonomy_calibration_labeling.md"
REGISTRY = ROOT / "configs" / "virgintrains_intents.yaml"
EXAMPLES = ROOT / "configs" / "virgintrains_labeling_examples.yaml"
CASES = PROCESSED / "virgintrains_cases.parquet"

needs_pack = pytest.mark.skipif(not all(p.exists() for p in (CSV, MANIFEST, ASSIGNMENTS, REGISTRY)), reason="calibration pack not generated")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_pack
def test_real_calibration_csv_matches_its_manifest_and_opens_read_only() -> None:
    before = digest(CSV)
    taxonomy = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["taxonomy"]
    store = LabelStore(CSV, MANIFEST, vocabulary_from_registry(taxonomy), assignments_path=ASSIGNMENTS)
    counts = store.counts()
    assert counts["total"] == 200 and sum(counts[k] for k in ("labelled", "partial", "unlabelled")) == 200
    frame = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    assert source_columns_sha256(frame) == json.loads(MANIFEST.read_text())["source_columns_sha256"]
    assert list(frame.columns[-5:]) == HUMAN_COLUMNS
    assert set(frame["source_split"]) == {RESERVE}
    assert digest(CSV) == before, "opening the labeling store must not modify the calibration CSV"


@needs_pack
def test_real_calibration_cases_are_all_reserve_and_none_golden() -> None:
    frame = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    assign = pd.read_csv(ASSIGNMENTS).set_index("case_id")["split"]
    assert set(assign.loc[frame["case_id"]]) == {RESERVE}
    assert not (set(frame["case_id"]) & set(assign[assign == GOLDEN].index))


@needs_pack
def test_taxonomy_is_still_a_candidate_and_nothing_is_frozen() -> None:
    assert registry_status(REGISTRY) == "CANDIDATE_NOT_GROUND_TRUTH"
    assert not (ROOT / "configs" / "virgintrains_taxonomy_v1.yaml").exists()
    assert not (ROOT / "data" / "golden" / "virgintrains_golden_v1.meta.json").exists(), "the freeze-gated golden script must not have run"


@needs_pack
def test_opening_the_real_store_leaves_no_audit_or_backup_files() -> None:
    taxonomy = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["taxonomy"]
    audit_before = (PROCESSED / AUDIT_NAME).exists()
    LabelStore(CSV, MANIFEST, vocabulary_from_registry(taxonomy), assignments_path=ASSIGNMENTS).state()
    assert (PROCESSED / AUDIT_NAME).exists() == audit_before


@pytest.mark.skipif(not (EXAMPLES.exists() and ASSIGNMENTS.exists() and CASES.exists() and CSV.exists()), reason="pipeline artifacts not generated")
def test_every_guide_example_is_a_train_case_and_not_golden_or_calibration() -> None:
    assign = pd.read_csv(ASSIGNMENTS).set_index("case_id")["split"]
    ids = [c["case_id"] for s in yaml.safe_load(EXAMPLES.read_text(encoding="utf-8"))["sections"] for c in s["cases"]]
    assert ids
    calibration = set(pd.read_csv(CSV, dtype=str, keep_default_na=False)["case_id"])
    for cid in ids:
        assert assign.get(cid) == TRAIN, f"{cid} is {assign.get(cid)!r}, guide examples must be train cases"
        assert cid not in calibration


@pytest.mark.skipif(not (GUIDE.exists() and LABELING_MD.exists() and CASES.exists() and EXAMPLES.exists()), reason="pipeline artifacts not generated")
def test_guide_quotes_are_verbatim_dataset_text() -> None:
    cases = pd.read_parquet(CASES, columns=["case_id", "full_turns"]).set_index("case_id")["full_turns"]
    spec = yaml.safe_load(EXAMPLES.read_text(encoding="utf-8"))["sections"]
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())
    checked = 0
    for section in spec:
        for ex in section["cases"]:
            first = " ".join(str(cases[ex["case_id"]][0]["text"]).split())[:60]
            assert first in guide, f"{ex['case_id']}: opening text is not in the guide verbatim"
            checked += 1
    assert checked >= 20


@pytest.mark.skipif(not (GUIDE.exists() and LABELING_MD.exists()), reason="guide not generated")
def test_guide_covers_every_required_concept_and_all_vocabulary_values() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    lowered = guide.lower()
    for needle in (
        "primary customer intent", "multi-intent", "confusable", "unclear_or_media_only", "NEW:<", "resolution type", "resolved",
        "escalation signal", "blindly", "ambiguous", "also:", "taxonomy_calibration_label_audit",
    ):
        assert needle.lower() in lowered, f"guide is missing: {needle}"
    for value in (*HUMAN_RESOLUTION_TYPES, *HUMAN_RESOLVED_VALUES, *ESCALATION_SIGNALS):
        assert value in guide, f"guide does not define the allowed value {value!r}"
    assert LABELING_MD.read_text(encoding="utf-8").startswith(guide.split("\n", 1)[0]) or "Calibration" in LABELING_MD.read_text(encoding="utf-8")[:300]


def test_loader_rejects_golden_calibration_and_non_train_examples(tmp_path) -> None:
    spec = {"sections": [{"key": "k", "title": "t", "cases": [{"case_id": cid, "reading": "r"} for cid in ("t1", "d1", "r1", "g1", "c1", "missing")]}]}
    path = tmp_path / "examples.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    turn = [{"role": "customer", "tweet_id": "1", "text": "hello"}]
    cases = pd.DataFrame({"case_id": ["t1", "d1", "r1", "g1", "c1"], "full_turns": [turn] * 5})
    split_of = pd.Series({"t1": TRAIN, "d1": "dev_calibration", "r1": RESERVE, "g1": GOLDEN, "c1": TRAIN})
    with pytest.raises(ExampleError) as err:
        load_guide_examples(path, cases, split_of, {"c1", "g1"})
    message = str(err.value)
    for bad in ("d1", "r1", "g1", "c1", "missing"):
        assert bad in message
    assert "t1:" not in message

    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.safe_dump({"sections": [{"key": "k", "title": "t", "cases": [{"case_id": "t1", "reading": "r"}]}]}), encoding="utf-8")
    out = load_guide_examples(ok, cases, split_of, {"c1", "g1"})
    assert out[0]["cases"][0]["turns"][0] == {"role": "CUSTOMER", "tweet_id": "1", "text": "hello"}
