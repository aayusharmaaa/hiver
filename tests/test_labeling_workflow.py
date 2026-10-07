"""Safety tests for the human labeling workflow. All data here is synthetic and lives in tmp dirs."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
from taxonomy_fixtures import INTENTS, write_calibration_world

from evaluation.labeling_store import (
    AUDIT_NAME,
    LabelStore,
    LabelStoreError,
    LabelValidationError,
    case_status,
    parse_conversation,
    validate_label,
    vocabulary_from_registry,
)
from evaluation.labeling_ui import make_handler, md_to_html
from evaluation.taxonomy_calibration import HUMAN_COLUMNS, source_columns_sha256
from evaluation.taxonomy_comparison import load_labelled

GOOD = {
    "human_intent": "ticket_booking_query",
    "human_resolution_type": "information_provided",
    "human_resolved": "yes",
    "human_escalation_signal": "none",
    "human_notes": "also: service_status_delay_enquiry",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def world(tmp_path):
    return write_calibration_world(tmp_path)


def open_store(world, **kw) -> LabelStore:
    return LabelStore(
        world["csv"], world["manifest"], vocabulary_from_registry(world["taxonomy"]), assignments_path=world["assignments"], labeler="tester", **kw
    )


def read(world) -> pd.DataFrame:
    return pd.read_csv(world["csv"], dtype=str, keep_default_na=False)


def source_part(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=HUMAN_COLUMNS)


def audit_events(world) -> list[dict]:
    path = world["csv"].with_name(AUDIT_NAME)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_unlabelled_calibration_data_is_valid_and_untouched_by_opening_it(world) -> None:
    before = digest(world["csv"])
    store = open_store(world)
    state = store.state()
    assert state["counts"] == {"total": 12, "labelled": 0, "partial": 0, "unlabelled": 12}
    assert state["first_open_index"] == 0 and [c["order"] for c in state["cases"]] == list(range(1, 13))
    load = load_labelled(read(world), list(store.vocab.intents), "unclear_or_media_only")
    assert load.n_complete == 0 and load.problems == []
    assert digest(world["csv"]) == before, "opening the store must not write the CSV"
    assert not world["csv"].with_name(AUDIT_NAME).exists(), "read-only use leaves no audit file"
    assert not world["csv"].with_name("taxonomy_calibration.csv.bak").exists()


def test_saving_preserves_every_provenance_field_and_changes_only_the_target_row(world) -> None:
    original = read(world)
    original_bytes = world["csv"].read_bytes().split(b"\n")
    store = open_store(world)
    target = world["ids"][4]
    result = store.save_label(target, GOOD)
    assert result["saved"] and result["counts"]["labelled"] == 1

    after = read(world)
    pd.testing.assert_frame_equal(source_part(after), source_part(original))
    assert source_columns_sha256(after) == json.loads(world["manifest"].read_text())["source_columns_sha256"]
    changed = after[(after[HUMAN_COLUMNS] != original[HUMAN_COLUMNS]).any(axis=1)]
    assert list(changed["case_id"]) == [target]
    assert dict(changed.iloc[0][HUMAN_COLUMNS]) == GOOD
    new_lines = world["csv"].read_bytes().split(b"\n")
    assert len(new_lines) == len(original_bytes)
    assert sum(a != b for a, b in zip(new_lines, original_bytes)) == 1, "exactly one CSV line differs"
    assert str(json.loads(after.loc[after["case_id"] == target, "source_tweet_ids"].iloc[0])[0]) == target.removeprefix("case_")


def test_save_then_clear_restores_the_original_bytes(world) -> None:
    original = world["csv"].read_bytes()
    store = open_store(world)
    store.save_label(world["ids"][0], GOOD)
    assert world["csv"].read_bytes() != original
    store.clear_label(world["ids"][0])
    assert world["csv"].read_bytes() == original, "the sample is byte-reproducible except for explicitly edited human_* fields"


@pytest.mark.parametrize(
    "patch, field",
    [
        ({"human_intent": "made_up_intent"}, "human_intent"),
        ({"human_intent": ""}, "human_intent"),
        ({"human_intent": "NEW:"}, "human_intent"),
        ({"human_intent": "NEW:ab"}, "human_intent"),
        ({"human_intent": "NEW:9bad_start"}, "human_intent"),
        ({"human_intent": "NEW:ticket_booking_query", "human_notes": "dup"}, "human_intent"),
        ({"human_intent": "NEW:lost_property", "human_notes": ""}, "human_notes"),
        ({"human_resolution_type": "nonsense"}, "human_resolution_type"),
        ({"human_resolution_type": ""}, "human_resolution_type"),
        ({"human_resolved": "maybe"}, "human_resolved"),
        ({"human_resolved": ""}, "human_resolved"),
        ({"human_escalation_signal": "panic"}, "human_escalation_signal"),
        ({"human_escalation_signal": ""}, "human_escalation_signal"),
        ({"human_notes": "x" * 2001}, "human_notes"),
    ],
)
def test_invalid_labels_are_rejected_and_nothing_is_written(world, patch, field) -> None:
    store = open_store(world)
    before = digest(world["csv"])
    with pytest.raises(LabelValidationError) as err:
        store.save_label(world["ids"][0], {**GOOD, **patch})
    assert field in err.value.errors
    assert digest(world["csv"]) == before
    assert [e for e in audit_events(world) if e["event"] == "save"] == []


def test_validation_reports_all_problems_at_once_and_normalises_new_intents() -> None:
    vocab = vocabulary_from_registry({"intents": [{"name": n} for n in INTENTS[:-1]], "fallback": {"name": INTENTS[-1]}})
    with pytest.raises(LabelValidationError) as err:
        validate_label({"human_intent": "zzz", "human_resolved": "?"}, vocab)
    assert {"human_intent", "human_resolution_type", "human_resolved", "human_escalation_signal"} <= set(err.value.errors)
    ok = validate_label({**GOOD, "human_intent": " new: Lost Property ", "human_notes": "new intent: lost_property: left items"}, vocab)
    assert ok["human_intent"] == "NEW:lost_property"
    assert validate_label({**GOOD, "human_intent": "TICKET_BOOKING_QUERY"}, vocab)["human_intent"] == "ticket_booking_query"
    assert validate_label({**GOOD, "human_notes": "a\r\nb"}, vocab)["human_notes"] == "a\nb"


def test_every_saved_label_passes_the_comparison_tools_validation(world) -> None:
    store = open_store(world)
    for i, cid in enumerate(world["ids"][:3]):
        store.save_label(cid, {**GOOD, "human_intent": INTENTS[i]})
    store.save_label(world["ids"][3], {**GOOD, "human_intent": "new:lost_property", "human_notes": "new intent: lost_property: ..."})
    load = load_labelled(read(world), list(store.vocab.intents), "unclear_or_media_only")
    assert load.problems == [] and load.n_complete == 4
    assert "new:lost_property" in set(load.frame["human_intent"])


def test_partial_labeling_resumes_where_the_reviewer_stopped(world) -> None:
    first = open_store(world)
    for cid in world["ids"][:3]:
        first.save_label(cid, GOOD)
    reopened = open_store(world)
    state = reopened.state()
    assert state["counts"]["labelled"] == 3 and state["first_open_index"] == 3
    assert reopened.get_case(world["ids"][1])["labels"] == GOOD
    assert reopened.get_case(world["ids"][5])["status"] == "unlabelled"
    reopened.save_label(world["ids"][3], {**GOOD, "human_resolved": "no"})
    assert open_store(world).state()["counts"]["labelled"] == 4


def test_cases_labelled_outside_the_tool_are_respected_and_partials_flagged(world) -> None:
    store = open_store(world)
    frame = read(world)
    frame.loc[0, HUMAN_COLUMNS] = list(GOOD.values())
    frame.loc[1, ["human_intent", "human_notes"]] = ["ticket_booking_query", "half done"]
    frame.to_csv(world["csv"], index=False)
    state = store.state()
    assert state["counts"] == {"total": 12, "labelled": 1, "partial": 1, "unlabelled": 10}
    assert state["first_open_index"] == 1
    assert case_status(read(world).loc[1]) == "partial"


def test_external_edits_to_other_rows_are_not_clobbered_by_a_save(world) -> None:
    store = open_store(world)
    store.state()
    frame = read(world)
    frame.loc[7, HUMAN_COLUMNS] = list(GOOD.values())
    frame.to_csv(world["csv"], index=False)
    store.save_label(world["ids"][2], GOOD)
    after = read(world)
    assert after.loc[7, "human_intent"] == GOOD["human_intent"], "a stale in-memory copy must never overwrite newer edits"
    assert after.loc[2, "human_intent"] == GOOD["human_intent"]


def test_tampered_provenance_blocks_all_writes(world) -> None:
    store = open_store(world)
    frame = read(world)
    frame.loc[3, "source_tweet_ids"] = "[1, 2]"
    frame.to_csv(world["csv"], index=False)
    tampered = digest(world["csv"])
    with pytest.raises(LabelStoreError, match="provenance"):
        store.save_label(world["ids"][0], GOOD)
    with pytest.raises(LabelStoreError):
        store.clear_label(world["ids"][0])
    assert digest(world["csv"]) == tampered, "the tool must not write to a modified sample"
    with pytest.raises(LabelStoreError, match="provenance"):
        open_store(world)


@pytest.mark.parametrize("column", ["case_id", "labeling_order", "conversation", "candidate_intent", "customer_id", "stratum_weight", "source_split"])
def test_changing_any_non_human_column_is_detected(world, column) -> None:
    frame = read(world)
    frame.loc[5, column] = "golden_eval" if column == "source_split" else frame.loc[5, column] + "x"
    frame.to_csv(world["csv"], index=False)
    with pytest.raises(LabelStoreError):
        open_store(world)


def test_a_golden_case_cannot_enter_the_calibration_set(world) -> None:
    frame = read(world)
    frame.loc[0, "case_id"] = world["golden_ids"][0]
    frame.to_csv(world["csv"], index=False)
    with pytest.raises(LabelStoreError):
        open_store(world)

    fresh = write_calibration_world(world["csv"].parents[2] / "second")
    a = pd.read_csv(fresh["assignments"])
    a.loc[a["case_id"] == fresh["ids"][0], "split"] = "golden_eval"
    a.to_csv(fresh["assignments"], index=False)
    with pytest.raises(LabelStoreError, match="golden"):
        open_store(fresh)

    third = write_calibration_world(world["csv"].parents[2] / "third")
    a = pd.read_csv(third["assignments"])
    a.loc[a["case_id"] == third["golden_ids"][0], "split"] = "train_retrieval"
    a.to_csv(third["assignments"], index=False)
    with pytest.raises(LabelStoreError, match="golden set no longer matches"):
        open_store(third)


def test_non_reserve_calibration_case_is_refused(world) -> None:
    a = pd.read_csv(world["assignments"])
    a.loc[a["case_id"] == world["ids"][0], "split"] = "train_retrieval"
    a.to_csv(world["assignments"], index=False)
    with pytest.raises(LabelStoreError):
        open_store(world)


def test_the_tool_only_writes_the_calibration_csv_and_never_golden_files(world) -> None:
    golden_before = {p: digest(p) for p in world["golden_files"]}
    assignments_before = digest(world["assignments"])
    manifest_before = digest(world["manifest"])
    store = open_store(world)
    store.start_session()
    for cid in world["ids"]:
        store.get_case(cid)
        store.get_suggestion(cid)
    store.save_label(world["ids"][0], GOOD)
    store.clear_label(world["ids"][0])
    store.save_label(world["ids"][1], GOOD)
    assert {p: digest(p) for p in world["golden_files"]} == golden_before
    assert digest(world["assignments"]) == assignments_before and digest(world["manifest"]) == manifest_before
    written = {p.name for p in world["processed"].iterdir() if p.is_file()}
    assert written == {"taxonomy_calibration.csv", "taxonomy_calibration.csv.bak", AUDIT_NAME, "taxonomy_calibration_manifest.json"}, written
    assert not list(world["processed"].glob(".*tmp")), "no temp files are left behind"


def test_store_refuses_to_target_any_other_file_or_the_golden_directory(world, tmp_path) -> None:
    vocab = vocabulary_from_registry(world["taxonomy"])
    other = world["processed"] / "other.csv"
    other.write_bytes(world["csv"].read_bytes())
    with pytest.raises(LabelStoreError, match="only writable file"):
        LabelStore(other, world["manifest"], vocab)
    in_golden = world["golden_dir"] / "taxonomy_calibration.csv"
    in_golden.write_bytes(world["csv"].read_bytes())
    with pytest.raises(LabelStoreError, match="golden"):
        LabelStore(in_golden, world["manifest"], vocab)


def test_no_human_label_is_ever_generated(world) -> None:
    original = world["csv"].read_bytes()
    store = open_store(world)
    store.start_session()
    for cid in world["ids"]:
        case = store.get_case(cid)
        assert all(v == "" for v in case["labels"].values()) and case["status"] == "unlabelled"
        store.get_suggestion(cid)
    store.state()
    assert world["csv"].read_bytes() == original
    frame = read(world)
    assert (frame[HUMAN_COLUMNS] == "").all(axis=None)
    assert [e["event"] for e in audit_events(world)].count("save") == 0


def test_the_case_view_never_contains_the_system_suggestion(world) -> None:
    case = open_store(world).get_case(world["ids"][0])
    dumped = json.dumps(case)
    for forbidden in ("candidate", "runner_up", "auto_", "cluster", "sampling", "selection_reason"):
        assert forbidden not in dumped, forbidden
    assert [t["role"] for t in case["turns"]] == ["CUSTOMER", "AGENT", "OTHER-AGENT"]
    assert case["turns"][0]["text"] == "first message 0 & more", "HTML entities are decoded for display"
    assert "quoted" in case["turns"][1]["text"]


def test_audit_log_records_reveal_before_save_and_before_after_values(world) -> None:
    store = open_store(world)
    store.start_session()
    a, b = world["ids"][0], world["ids"][1]
    store.save_label(a, GOOD)
    store.get_suggestion(b)
    store.get_suggestion(b)
    store.save_label(b, {**GOOD, "human_resolved": "no"})
    store.save_label(b, {**GOOD, "human_resolved": "unclear"})
    events = audit_events(world)
    assert events[0]["event"] == "session_start" and events[0]["labeler"] == "tester"
    assert [e["event"] for e in events].count("reveal") == 1, "repeated reveals of one case are logged once"
    saves = [e for e in events if e["event"] == "save"]
    assert [s["suggestion_revealed_before_save"] for s in saves] == [False, True, True]
    assert saves[0]["before"]["human_intent"] == "" and saves[0]["after"] == {c: GOOD[c] for c in HUMAN_COLUMNS}
    assert saves[2]["before"]["human_resolved"] == "no" and saves[2]["after"]["human_resolved"] == "unclear"
    assert all("ts" in e for e in events)
    assert open_store(world)._revealed == {b}, "reveal history survives a restart"


def test_saving_identical_values_is_a_no_op(world) -> None:
    store = open_store(world)
    store.save_label(world["ids"][0], GOOD)
    snapshot = digest(world["csv"])
    n_events = len(audit_events(world))
    assert store.save_label(world["ids"][0], GOOD) == {"saved": False, "reason": "unchanged", "counts": store.counts()}
    assert digest(world["csv"]) == snapshot and len(audit_events(world)) == n_events


def test_backup_keeps_the_previous_version(world) -> None:
    store = open_store(world)
    store.save_label(world["ids"][0], GOOD)
    first = world["csv"].read_bytes()
    store.save_label(world["ids"][1], GOOD)
    assert world["csv"].with_name("taxonomy_calibration.csv.bak").read_bytes() == first


def test_windows_line_endings_are_preserved(world) -> None:
    for terminator in ("\r\n", "\n"):
        read(world).to_csv(world["csv"], index=False, lineterminator=terminator)
        store = open_store(world)
        store.save_label(world["ids"][0], GOOD)
        data = world["csv"].read_bytes()
        if terminator == "\r\n":
            assert data.count(b"\r\n") == 13, "header + 12 rows keep CRLF; newlines inside cells stay LF"
        else:
            assert b"\r" not in data
        store.clear_label(world["ids"][0])


def test_conversation_parsing_handles_multiline_text_and_entities() -> None:
    turns = parse_conversation("[CUSTOMER 1] a &lt;b&gt;\ncontinued line\n[AGENT 2] ok")
    assert turns == [{"role": "CUSTOMER", "tweet_id": "1", "text": "a <b> continued line"}, {"role": "AGENT", "tweet_id": "2", "text": "ok"}]


def test_markdown_renderer_escapes_html() -> None:
    out = md_to_html("# Title\n\n- **bold** `<script>alert(1)</script>`\n\n<img src=x onerror=alert(1)>")
    assert "<script>" not in out and "<img" not in out and "&lt;script&gt;" in out and "<strong>bold</strong>" in out


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
def server(world):
    store = open_store(world)
    allowed: set[str] = set()
    token = "test-token"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, {"intents": [], "disclaimer": ""}, "<p>guide</p>", token, allowed))
    port = httpd.server_address[1]
    allowed.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield Client(f"http://127.0.0.1:{port}", token), world, store
    httpd.shutdown()
    httpd.server_close()


def test_http_requires_token_and_a_local_host_header(server) -> None:
    client, world, _ = server
    assert client.call("/api/state", headers={"X-Label-Token": "wrong"})[0] == 403
    assert client.call("/api/state", headers={"X-Label-Token": ""})[0] == 403
    status, _ = client.call("/api/state", headers={"Host": "evil.example:80"})
    assert status == 403
    assert client.call("/api/state")[0] == 200
    status, body = client.call("/")
    assert status == 200 and b"test-token" in body and b"Taxonomy calibration labeling" in body
    assert client.call("/guide")[0] == 200


def test_http_save_validation_and_routing(server) -> None:
    client, world, store = server
    cid = world["ids"][0]
    status, body = client.call(f"/api/save/{cid}", "POST", {**GOOD, "human_intent": "bogus"})
    assert status == 422 and "human_intent" in json.loads(body)["errors"]
    assert read(world).loc[0, "human_intent"] == ""
    status, body = client.call(f"/api/save/{cid}", "POST", GOOD)
    assert status == 200 and json.loads(body)["counts"]["labelled"] == 1
    assert read(world).loc[0, "human_intent"] == GOOD["human_intent"]
    assert client.call(f"/api/save/{cid}", "GET")[0] == 405
    assert client.call(f"/api/case/{cid}", "POST", {})[0] == 405
    assert client.call("/api/save/not_a_case", "POST", GOOD)[0] == 404
    assert client.call("/api/save/..%2f..%2fetc", "POST", GOOD)[0] == 404
    assert client.call("/api/save/case_424242", "POST", GOOD)[0] == 409, "unknown but well-formed ids are refused"
    assert client.call("/nope")[0] == 404
    state = json.loads(client.call("/api/state")[1])
    assert state["counts"]["labelled"] == 1 and state["vocab"]["intents"] == list(store.vocab.intents)


def test_http_suggestion_is_a_separate_audited_request(server) -> None:
    client, world, _ = server
    cid = world["ids"][2]
    status, body = client.call(f"/api/case/{cid}")
    assert status == 200 and "candidate" not in body.decode()
    assert not any(e["event"] == "reveal" for e in audit_events(world))
    status, body = client.call(f"/api/suggestion/{cid}", "POST", {})
    sug = json.loads(body)
    assert status == 200 and sug["candidate_intent"] == INTENTS[2] and 0 < sug["margin_percentile_in_sample"] <= 1
    assert [e["case_id"] for e in audit_events(world) if e["event"] == "reveal"] == [cid]
    assert (read(world)[HUMAN_COLUMNS] == "").all(axis=None), "revealing never fills a label"


def test_http_cannot_write_provenance_fields_through_the_api(server) -> None:
    client, world, _ = server
    before = source_part(read(world))
    cid = world["ids"][0]
    payload = {**GOOD, "case_id": "case_1", "source_tweet_ids": "[1]", "candidate_intent": "x", "source_split": "golden_eval", "labeling_order": "99"}
    assert client.call(f"/api/save/{cid}", "POST", payload)[0] == 200
    pd.testing.assert_frame_equal(source_part(read(world)), before)


def test_http_clear_and_oversized_body(server) -> None:
    client, world, _ = server
    cid = world["ids"][0]
    client.call(f"/api/save/{cid}", "POST", GOOD)
    status, body = client.call(f"/api/clear/{cid}", "POST", {})
    assert status == 200 and json.loads(body)["cleared"] is True
    assert (read(world)[HUMAN_COLUMNS] == "").all(axis=None)
    assert client.call(f"/api/save/{cid}", "POST", {**GOOD, "human_notes": "x" * 70000})[0] == 409


def test_reference_cards_decode_html_entities_and_stay_neutral() -> None:
    from evaluation.labeling_support import build_reference

    taxonomy = {
        "intents": [
            {"name": "b_intent", "definition": "d", "positive_examples": [{"case_id": "case_1", "text": "touch &amp; go"}], "negative_examples": []},
            {"name": "a_intent", "definition": "d"},
        ],
        "fallback": {"name": "unclear_or_media_only", "definition": "d"},
    }
    ref = build_reference(taxonomy, [])
    names = [c["name"] for c in ref["intents"]]
    assert names == ["a_intent", "b_intent", "unclear_or_media_only"], "alphabetical, fallback last, never ranked by suggestion"
    assert ref["intents"][1]["historical_examples"][0]["text"] == "touch & go"