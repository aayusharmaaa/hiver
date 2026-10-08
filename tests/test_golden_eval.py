"""Golden evaluation labeling workflow: integrity gates, blind UI payloads, label store, taxonomy review.

Synthetic worlds only, except the `real_*` tests at the end, which check the prepared pack in data/golden (read-only).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
import yaml
from taxonomy_fixtures import FALLBACK, INTENTS, candidate_taxonomy

from evaluation.golden_eval import (
    AUDIT_LOG,
    CANDIDATES_PARQUET,
    GOLD_COLUMNS,
    PACK_CSV,
    PACK_MANIFEST,
    SOURCE_COLUMNS,
    TAXONOMY_HEADING,
    GoldenIntegrityError,
    GoldenLabelStore,
    GoldVocabulary,
    build_reference,
    guide_markdown,
    prepare_golden_pack,
    replay_audit,
    source_fingerprint,
    validate_gold_label,
)
from evaluation.golden_labeling_ui import GOLDEN_INDEX_PAGE
from evaluation.golden_review import ReviewNotReadyError, build_review_markdown, review_frame
from evaluation.labeling_store import LabelStoreError, LabelValidationError, parse_conversation
from evaluation.labeling_ui import make_handler, md_to_html
from taxonomy.registry import case_ids_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
N = 12
GOOD = {"gold_intent": "ticket_booking_query", "gold_should_escalate": "no", "gold_resolution_type": "information_provided", "gold_confidence": "high", "human_notes": ""}
PREDICTION_WORDS = ("suggest", "predict", "reveal", "candidate_intent", "cluster", "retriev", "policy", "draft", "grounding", "auto_handle", "evidence_score", "reply")


# --------------------------------------------------------------------------------------------------------------------
# Synthetic world
# --------------------------------------------------------------------------------------------------------------------
def write_golden_world(root: Path, n: int = N) -> dict:
    processed, golden_dir, configs = root / "data" / "processed", root / "data" / "golden", root / "configs"
    (processed / "splits").mkdir(parents=True)
    golden_dir.mkdir(parents=True)
    configs.mkdir()
    gold = [f"case_{9000 + i}" for i in range(n)]
    train, dev = ["case_1", "case_2"], ["case_3"]

    def row(cid: str, split: str, i: int) -> dict:
        return {"case_id": cid, "split": split, "split_reason": "", "group_id": f"g_{cid}", "customer_id": f"u_{cid}", "conversation_id": f"conv_{cid}", "cluster_id": i % 5}

    assignments = pd.DataFrame([row(c, "golden_eval", i) for i, c in enumerate(gold)] + [row(c, "train_retrieval", 0) for c in train] + [row(c, "dev_calibration", 0) for c in dev])
    assignments.to_csv(processed / "splits" / "virgintrains_split_assignments.csv", index=False)
    all_ids = gold + train + dev

    def opener(i: int) -> str:
        return "opening message " + "".join(chr(97 + int(d)) for d in str(i)) + " please help"

    pd.DataFrame(
        {
            "case_id": all_ids,
            "opening_message": [opener(i) for i in range(len(all_ids))],
            "source_tweet_ids": [[10_000 + i] for i in range(len(all_ids))],
            "context_tweet_ids": [[20_000 + i] for i in range(len(all_ids))],
        }
    ).to_parquet(processed / "virgintrains_cases.parquet")

    def turns(i: int, cid: str) -> list[dict]:
        return [
            {"turn_index": 0, "tweet_id": 10_000 + i, "role": "customer", "text": f"{opener(i)} for {cid} &amp; more"},
            {"turn_index": 1, "tweet_id": 30_000 + i, "role": "brand_agent", "text": "sorry to hear, \"quoted\" ^AB"},
        ]

    golden = pd.DataFrame(
        {
            "case_id": gold,
            "conversation_id": [f"conv_{c}" for c in gold],
            "customer_id": [f"u_{c}" for c in gold],
            "opening_message": [opener(i) for i in range(n)],
            "full_turns": [turns(i, c) for i, c in enumerate(gold)],
            "first_timestamp": pd.to_datetime(["2017-10-24 11:33:55+00:00"] * n),
            "source_tweet_ids": [[10_000 + i, 30_000 + i] for i in range(n)],
            "cluster_id": [i % 5 for i in range(n)],
            "resolution_type": ["refund"] * n,
            "resolved": [True] * n,
            "golden_stratum_weight": [2.0] * n,
        }
    )
    parquet = golden_dir / CANDIDATES_PARQUET
    golden.to_parquet(parquet)
    split_manifest = processed / "splits" / "virgintrains_split_manifest.json"
    split_manifest.write_text(json.dumps({"counts": {"golden_eval": n}, "file_sha256": {"golden_candidates": sha256_file(parquet)}}), encoding="utf-8")
    registry = configs / "virgintrains_intents.yaml"
    registry.write_text(yaml.safe_dump(candidate_taxonomy(), sort_keys=False), encoding="utf-8")
    return {"processed": processed, "golden_dir": golden_dir, "registry": registry, "parquet": parquet, "split_manifest": split_manifest,
            "assignments": processed / "splits" / "virgintrains_split_assignments.csv", "ids": gold, "n": n}


def prepare(world: dict) -> tuple[pd.DataFrame, dict]:
    return prepare_golden_pack(world["golden_dir"], world["processed"], world["registry"], expected_cases=world["n"])


def vocab() -> GoldVocabulary:
    return GoldVocabulary.from_registry(candidate_taxonomy()["taxonomy"])


def open_store(world: dict) -> GoldenLabelStore:
    return GoldenLabelStore(world["golden_dir"], vocab(), split_manifest_path=world["split_manifest"], assignments_path=world["assignments"], registry_path=world["registry"], labeler="tester")


def reseal_parquet(world: dict, frame: pd.DataFrame) -> None:
    """Rewrite the candidates AND the split manifest hash, to test the checks behind the hash gate."""
    frame.to_parquet(world["parquet"])
    m = json.loads(world["split_manifest"].read_text(encoding="utf-8"))
    m["file_sha256"]["golden_candidates"] = sha256_file(world["parquet"])
    world["split_manifest"].write_text(json.dumps(m), encoding="utf-8")


@pytest.fixture()
def world(tmp_path: Path) -> dict:
    return write_golden_world(tmp_path)


@pytest.fixture()
def prepared(world: dict) -> dict:
    prepare(world)
    return world


# --------------------------------------------------------------------------------------------------------------------
# Preparing the pack
# --------------------------------------------------------------------------------------------------------------------
def test_prepare_writes_one_blank_row_per_golden_case_and_nothing_else(world) -> None:
    frame, manifest = prepare(world)
    on_disk = pd.read_csv(world["golden_dir"] / PACK_CSV, dtype=str, keep_default_na=False)
    assert list(on_disk.columns) == SOURCE_COLUMNS + GOLD_COLUMNS
    assert len(on_disk) == N and on_disk["case_id"].is_unique and set(on_disk["case_id"]) == set(world["ids"])
    assert (on_disk[GOLD_COLUMNS] == "").all().all(), "human columns must start blank"
    assert sorted(on_disk["labeling_order"].astype(int)) == list(range(1, N + 1))
    assert manifest["golden_case_ids_sha256"] == case_ids_sha256(world["ids"])
    assert manifest["golden_candidates_parquet_sha256"] == sha256_file(world["parquet"])
    assert manifest["source_columns_sha256"] == source_fingerprint(on_disk)
    assert manifest["taxonomy_reference"]["status"] == "CANDIDATE_NOT_GROUND_TRUTH"
    assert all(v == 0 for v in manifest["leakage_checks_all_zero"].values())


def test_pack_hides_clusters_historical_labels_and_weights(prepared) -> None:
    text = (prepared["golden_dir"] / PACK_CSV).read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for word in ("cluster", "resolution_type,", "resolved", "weight", "candidate", "predict", "reply"):
        assert word not in header.replace("gold_resolution_type", ""), word


def test_prepare_does_not_touch_the_candidates_or_splits(world) -> None:
    before = {p: sha256_file(p) for p in (world["parquet"], world["split_manifest"], world["assignments"], world["registry"])}
    prepare(world)
    assert {p: sha256_file(p) for p in before} == before


def test_prepare_refuses_if_the_candidates_changed_after_the_split(world) -> None:
    golden = pd.read_parquet(world["parquet"])
    golden.loc[0, "opening_message"] = "altered"
    golden.to_parquet(world["parquet"])
    with pytest.raises(GoldenIntegrityError, match="split manifest"):
        prepare(world)
    assert not (world["golden_dir"] / PACK_CSV).exists()


def test_prepare_requires_the_exact_case_count(world) -> None:
    with pytest.raises(GoldenIntegrityError, match="exactly"):
        prepare_golden_pack(world["golden_dir"], world["processed"], world["registry"], expected_cases=N + 1)
    with pytest.raises(GoldenIntegrityError, match="exactly 250"):
        prepare_golden_pack(world["golden_dir"], world["processed"], world["registry"])
    assert not (world["golden_dir"] / PACK_CSV).exists()


def test_prepare_refuses_duplicate_case_ids(world) -> None:
    golden = pd.read_parquet(world["parquet"])
    reseal_parquet(world, pd.concat([golden.iloc[:-1], golden.iloc[[0]]], ignore_index=True))
    with pytest.raises(GoldenIntegrityError, match="duplicate"):
        prepare(world)


def test_prepare_refuses_changed_membership(world) -> None:
    golden = pd.read_parquet(world["parquet"])
    golden.loc[0, "case_id"] = "case_3"
    reseal_parquet(world, golden)
    with pytest.raises(GoldenIntegrityError, match="membership"):
        prepare(world)


@pytest.mark.parametrize("split", ["train_retrieval", "dev_calibration"])
def test_prepare_refuses_a_golden_case_assigned_to_train_or_dev(world, split) -> None:
    a = pd.read_csv(world["assignments"])
    a.loc[a["case_id"] == world["ids"][0], "split"] = split
    a.to_csv(world["assignments"], index=False)
    with pytest.raises(GoldenIntegrityError):
        prepare(world)


@pytest.mark.parametrize("column", ["customer_id", "conversation_id", "group_id"])
def test_prepare_refuses_shared_customer_conversation_or_group(world, column) -> None:
    a = pd.read_csv(world["assignments"], dtype=str)
    a.loc[a["case_id"] == "case_1", column] = a.loc[a["case_id"] == world["ids"][0], column].iloc[0]
    a.to_csv(world["assignments"], index=False)
    with pytest.raises(GoldenIntegrityError):
        prepare(world)
    assert not (world["golden_dir"] / PACK_CSV).exists()


@pytest.mark.parametrize("column", ["source_tweet_ids", "context_tweet_ids", "opening_message"])
def test_prepare_refuses_shared_tweets_or_opener(world, column) -> None:
    cases_path = world["processed"] / "virgintrains_cases.parquet"
    cases = pd.read_parquet(cases_path)
    values = list(cases[column])
    values[cases.index[cases["case_id"] == "case_3"][0]] = values[cases.index[cases["case_id"] == world["ids"][0]][0]]
    cases[column] = values
    cases.to_parquet(cases_path)
    with pytest.raises(GoldenIntegrityError):
        prepare(world)


def test_prepare_never_overwrites_once_labeling_started(prepared) -> None:
    store = open_store(prepared)
    store.save_label(prepared["ids"][0], GOOD)
    with pytest.raises(GoldenIntegrityError, match="labeling has started"):
        prepare(prepared)
    (prepared["golden_dir"] / AUDIT_LOG).unlink()
    with pytest.raises(GoldenIntegrityError, match="already contains labels"):
        prepare(prepared)


def test_prepare_is_repeatable_while_blank(prepared) -> None:
    first = (prepared["golden_dir"] / PACK_CSV).read_bytes()
    open_store(prepared).start_session()
    prepare(prepared)
    assert (prepared["golden_dir"] / PACK_CSV).read_bytes() == first


def test_old_freeze_gated_script_refuses_to_overwrite_the_new_pack(prepared) -> None:
    before = sha256_file(prepared["golden_dir"] / PACK_CSV)
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_golden_labeling.py"), "--golden-dir", str(prepared["golden_dir"])], capture_output=True, text=True)
    assert proc.returncode == 1 and "prepare_golden_eval" in proc.stderr + proc.stdout
    assert sha256_file(prepared["golden_dir"] / PACK_CSV) == before


# --------------------------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------------------------
def test_validation_accepts_candidate_intents_and_the_fallback() -> None:
    assert validate_gold_label(GOOD, vocab()) == GOOD
    assert validate_gold_label({**GOOD, "gold_intent": FALLBACK, "gold_confidence": ""}, vocab())["gold_intent"] == FALLBACK


def test_new_intent_is_preserved_exactly_and_needs_notes() -> None:
    with pytest.raises(LabelValidationError) as err:
        validate_gold_label({**GOOD, "gold_intent": "NEW:lost_property"}, vocab())
    assert "human_notes" in err.value.errors
    out = validate_gold_label({**GOOD, "gold_intent": "NEW:lost_property", "human_notes": "left a bag on the train"}, vocab())
    assert out["gold_intent"] == "NEW:lost_property"


@pytest.mark.parametrize(
    "change, field",
    [
        ({"gold_intent": ""}, "gold_intent"),
        ({"gold_intent": "made_up_intent"}, "gold_intent"),
        ({"gold_intent": "NEW:Bad Name", "human_notes": "x"}, "gold_intent"),
        ({"gold_intent": f"NEW:{INTENTS[0]}", "human_notes": "x"}, "gold_intent"),
        ({"gold_should_escalate": ""}, "gold_should_escalate"),
        ({"gold_should_escalate": "maybe"}, "gold_should_escalate"),
        ({"gold_resolution_type": ""}, "gold_resolution_type"),
        ({"gold_resolution_type": "redirected_to_dm"}, "gold_resolution_type"),
        ({"gold_confidence": "certain"}, "gold_confidence"),
        ({"human_notes": "x" * 2001}, "human_notes"),
        ({"reply": "Hi there, your train is on time"}, "_"),
        ({"predicted_intent": "ticket_booking_query"}, "_"),
    ],
)
def test_validation_rejects_bad_values_and_any_non_human_field(change, field) -> None:
    with pytest.raises(LabelValidationError) as err:
        validate_gold_label({**GOOD, **change}, vocab())
    assert field in err.value.errors


# --------------------------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------------------------
def test_store_writes_only_the_human_columns_and_audits(prepared) -> None:
    store = open_store(prepared)
    csv = prepared["golden_dir"] / PACK_CSV
    fingerprint = source_fingerprint(pd.read_csv(csv, dtype=str, keep_default_na=False))
    res = store.save_label(prepared["ids"][3], {**GOOD, "human_notes": "  two requests\r\nsecond line  "})
    assert res["saved"] and res["counts"]["labelled"] == 1
    after = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert source_fingerprint(after) == fingerprint
    row = after.set_index("case_id").loc[prepared["ids"][3]]
    assert row["gold_intent"] == "ticket_booking_query" and row["human_notes"] == "two requests\nsecond line"
    assert (after.loc[after["case_id"] != prepared["ids"][3], GOLD_COLUMNS] == "").all().all()
    assert replay_audit(prepared["golden_dir"] / AUDIT_LOG)[prepared["ids"][3]]["gold_intent"] == "ticket_booking_query"
    assert (prepared["golden_dir"] / (PACK_CSV + ".bak")).exists()
    assert store.clear_label(prepared["ids"][3])["cleared"]
    assert (pd.read_csv(csv, dtype=str, keep_default_na=False)[GOLD_COLUMNS] == "").all().all()
    open_store(prepared)


def test_store_survives_crlf_conversion_of_the_pack(prepared) -> None:
    csv = prepared["golden_dir"] / PACK_CSV
    open_store(prepared).save_label(prepared["ids"][0], {**GOOD, "human_notes": "line one\nline two"})
    csv.write_bytes(csv.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    store = open_store(prepared)
    store.save_label(prepared["ids"][1], GOOD)
    assert open_store(prepared).counts()["labelled"] == 2


def test_store_never_prefills_and_reports_progress(prepared) -> None:
    store = open_store(prepared)
    state = store.state()
    assert state["counts"] == {"total": N, "labelled": 0, "partial": 0, "unlabelled": N}
    assert all(v == "" for v in store.get_case(prepared["ids"][0])["labels"].values())


def test_store_refuses_labels_edited_outside_the_tool(prepared) -> None:
    csv = prepared["golden_dir"] / PACK_CSV
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df.loc[0, "gold_intent"] = "ticket_booking_query"
    df.to_csv(csv, index=False)
    with pytest.raises(LabelStoreError, match="outside the labeling tool"):
        open_store(prepared)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda w: pd.read_csv(w["golden_dir"] / PACK_CSV, dtype=str, keep_default_na=False).assign(conversation=lambda d: d["conversation"].str.upper()).to_csv(w["golden_dir"] / PACK_CSV, index=False),
        lambda w: pd.read_csv(w["golden_dir"] / PACK_CSV, dtype=str, keep_default_na=False).iloc[1:].to_csv(w["golden_dir"] / PACK_CSV, index=False),
        lambda w: pd.read_csv(w["golden_dir"] / PACK_CSV, dtype=str, keep_default_na=False).assign(extra="x").to_csv(w["golden_dir"] / PACK_CSV, index=False),
        lambda w: reseal_parquet(w, pd.read_parquet(w["parquet"]).assign(opening_message="changed")),
        lambda w: w["registry"].write_text(w["registry"].read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8"),
        lambda w: pd.read_csv(w["assignments"]).assign(split=lambda d: d["split"].where(d["case_id"] != w["ids"][0], "golden_pool_reserve")).to_csv(w["assignments"], index=False),
    ],
    ids=["conversation_edited", "row_removed", "column_added", "candidates_changed", "taxonomy_changed", "membership_changed"],
)
def test_store_refuses_any_change_to_the_frozen_sample(prepared, tamper) -> None:
    store = open_store(prepared)
    tamper(prepared)
    with pytest.raises(LabelStoreError):
        store.save_label(prepared["ids"][0], GOOD)
    with pytest.raises(LabelStoreError):
        store.get_case(prepared["ids"][0])


# --------------------------------------------------------------------------------------------------------------------
# Blind UI
# --------------------------------------------------------------------------------------------------------------------
class Client:
    def __init__(self, base: str, token: str):
        self.base, self.token = base, token

    def call(self, path: str, method: str = "GET", body: dict | None = None, headers: dict | None = None):
        hdrs = {"X-Label-Token": self.token, "Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


@pytest.fixture()
def server(prepared):
    store = open_store(prepared)
    reference = build_reference(candidate_taxonomy()["taxonomy"], [])
    allowed: set[str] = set()
    handler = make_handler(store, reference, md_to_html(guide_markdown()), "tok", allowed, index_page=GOLDEN_INDEX_PAGE, with_suggestion=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    allowed.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Client(f"http://127.0.0.1:{port}", "tok"), prepared
    httpd.shutdown()
    httpd.server_close()


def test_ui_payloads_contain_only_the_conversation_and_human_labels(server) -> None:
    client, world = server
    status, body = client.call("/api/state")
    state = json.loads(body)
    assert status == 200 and set(state) == {"counts", "cases", "first_open_index", "labeler", "vocab"}
    assert all(set(c) == {"case_id", "order", "status"} for c in state["cases"])
    status, body = client.call(f"/api/case/{world['ids'][0]}")
    case = json.loads(body)
    assert set(case) == {"case_id", "index", "total", "order", "first_timestamp", "turns", "labels", "status"}
    assert set(case["labels"]) == set(GOLD_COLUMNS) and all(v == "" for v in case["labels"].values())
    assert [t["role"] for t in case["turns"]] == ["CUSTOMER", "AGENT"] and case["turns"][0]["text"].endswith("& more")
    for word in ("cluster", "candidate_intent", "refund", "weight", "predict", "suggest"):
        assert word not in body.decode("utf-8"), word


def test_ui_has_no_suggestion_route_or_prediction_controls(server) -> None:
    client, world = server
    assert client.call(f"/api/suggestion/{world['ids'][0]}", "POST", {})[0] == 404
    status, page = client.call("/")
    page = page.decode("utf-8").lower()
    assert status == 200 and "golden evaluation labeling" in page
    for word in PREDICTION_WORDS:
        assert word not in page, word


def test_ui_reference_is_the_provisional_taxonomy_without_system_output(server) -> None:
    client, _ = server
    ref = json.loads(client.call("/api/reference")[1])
    assert ref["heading"] == TAXONOMY_HEADING
    assert set(ref) == {"heading", "status", "intents", "confusable_notes", "label_definitions", "resolution_types", "escalation_guidance"}
    assert {c["name"] for c in ref["intents"]} == set(INTENTS) | {FALLBACK}
    assert all(set(c) == {"name", "is_fallback", "definition", "fits_when", "does_not_fit_when", "confusable_intents", "historical_examples"} for c in ref["intents"])


def test_ui_saves_through_the_store_and_validates(server) -> None:
    client, world = server
    cid = world["ids"][1]
    assert client.call(f"/api/save/{cid}", "POST", {**GOOD, "gold_should_escalate": ""})[0] == 422
    status, body = client.call(f"/api/save/{cid}", "POST", GOOD)
    assert status == 200 and json.loads(body)["counts"]["labelled"] == 1
    assert json.loads(client.call(f"/api/case/{cid}")[1])["labels"] == GOOD
    assert client.call("/api/state", headers={"X-Label-Token": "wrong"})[0] == 403


def test_golden_modules_do_not_load_the_agent_models_or_retrieval() -> None:
    code = (
        "import sys; import evaluation.golden_eval, evaluation.golden_labeling_ui, evaluation.golden_review; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('agent', 'models', 'retrieval') or 'gemini' in m))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


# --------------------------------------------------------------------------------------------------------------------
# Taxonomy review (after labeling)
# --------------------------------------------------------------------------------------------------------------------
def labelled_review_frame(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cand = INTENTS[i % 4]
        gold = cand if i % 3 else INTENTS[(i + 1) % 4]
        if i == 5:
            gold = "NEW:lost_property"
        rows.append({"case_id": f"case_{i}", "first_customer_message": f"msg {i}", "gold_intent": gold, "gold_should_escalate": "no",
                     "gold_resolution_type": "information_provided", "gold_confidence": "low" if i % 4 == 0 else "high",
                     "human_notes": "left my bag" if i == 5 else "", "_cand": cand})
    return pd.DataFrame(rows)


def test_review_refuses_until_every_case_is_labelled() -> None:
    df = labelled_review_frame()
    df.loc[0, "gold_resolution_type"] = ""
    with pytest.raises(ReviewNotReadyError):
        review_frame(df.drop(columns="_cand"), dict(zip(df["case_id"], df["_cand"])))


def test_review_report_has_the_requested_sections_and_no_metrics() -> None:
    df = labelled_review_frame()
    frame = review_frame(df.drop(columns="_cand"), dict(zip(df["case_id"], df["_cand"])))
    md = build_review_markdown(frame, registry_status="CANDIDATE_NOT_GROUND_TRUTH")
    for heading in ("## Intent counts", "## Candidate vs gold disagreement", "## NEW intents", "## Major confusion pairs", "## Difficult boundary examples", "## Merge / split / rename signals"):
        assert heading in md
    assert "NEW:lost_property" in md and "left my bag" in md
    for word in ("accuracy:", "f1", "precision", "recall"):
        assert word not in md.lower().replace("reports no accuracy figures", "")


# --------------------------------------------------------------------------------------------------------------------
# Real artifacts (read-only)
# --------------------------------------------------------------------------------------------------------------------
REAL_GOLDEN = ROOT / "data" / "golden"
REAL_SPLITS = ROOT / "data" / "processed" / "splits"
real = pytest.mark.skipif(not (REAL_GOLDEN / PACK_MANIFEST).exists() or not (REAL_SPLITS / "virgintrains_split_assignments.csv").exists(), reason="golden pack not prepared")


@real
def test_real_pack_is_the_frozen_250_with_no_train_or_dev_overlap() -> None:
    pack = pd.read_csv(REAL_GOLDEN / PACK_CSV, dtype=str, keep_default_na=False)
    manifest = json.loads((REAL_GOLDEN / PACK_MANIFEST).read_text(encoding="utf-8"))
    split_manifest = json.loads((REAL_SPLITS / "virgintrains_split_manifest.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(REAL_SPLITS / "virgintrains_split_assignments.csv", dtype=str)
    assert len(pack) == 250 and pack["case_id"].is_unique and list(pack.columns) == SOURCE_COLUMNS + GOLD_COLUMNS
    assert set(pack["case_id"]) == set(assignments.loc[assignments["split"] == "golden_eval", "case_id"])
    assert sha256_file(REAL_GOLDEN / CANDIDATES_PARQUET) == split_manifest["file_sha256"]["golden_candidates"] == manifest["golden_candidates_parquet_sha256"]
    assert manifest["golden_case_ids_sha256"] == case_ids_sha256(pack["case_id"]) and manifest["source_columns_sha256"] == source_fingerprint(pack)
    golden = assignments[assignments["case_id"].isin(pack["case_id"])]
    others = assignments[assignments["split"].isin(["train_retrieval", "dev_calibration"])]
    assert (golden["split"] == "golden_eval").all()
    for col in ("customer_id", "conversation_id", "group_id"):
        assert not set(golden[col]) & set(others[col]), col
    assert all(v == 0 for v in manifest["leakage_checks_all_zero"].values())


@real
def test_real_pack_conversations_render_every_turn_oldest_first() -> None:
    pack = pd.read_csv(REAL_GOLDEN / PACK_CSV, dtype=str, keep_default_na=False).set_index("case_id")
    golden = pd.read_parquet(REAL_GOLDEN / CANDIDATES_PARQUET, columns=["case_id", "full_turns"])
    for cid, turns in zip(golden["case_id"], golden["full_turns"]):
        parsed = parse_conversation(pack.at[cid, "conversation"])
        assert [p["tweet_id"] for p in parsed] == [str(t["tweet_id"]) for t in turns], cid
        assert {p["role"] for p in parsed} <= {"CUSTOMER", "AGENT", "OTHER-AGENT"}
        assert parsed[0]["role"] == "CUSTOMER" or turns[0]["role"] != "customer"


@real
def test_real_pack_labels_came_only_through_the_tool() -> None:
    registry = ROOT / "configs" / "virgintrains_intents.yaml"
    store = GoldenLabelStore(
        REAL_GOLDEN, GoldVocabulary.from_registry(yaml.safe_load(registry.read_text(encoding="utf-8"))["taxonomy"]),
        split_manifest_path=REAL_SPLITS / "virgintrains_split_manifest.json", assignments_path=REAL_SPLITS / "virgintrains_split_assignments.csv", registry_path=registry,
    )
    frame = store.verified_frame()
    if not (REAL_GOLDEN / AUDIT_LOG).exists():
        assert (frame[GOLD_COLUMNS] == "").all().all(), "labels exist without an audit trail"
