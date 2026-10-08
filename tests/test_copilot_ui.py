import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from ui.copilot import PAGE, CopilotApp, _Server, build_tickets, make_handler, parse_conversation, ticket_status

CONVERSATION = "[CUSTOMER 11] @VirginTrains is wifi free?\nsecond line\n[AGENT 12] @1 Yes ^AB\n[OTHER-AGENT 13] @1 Not on our trains ^XY"


def case(case_id="case_11", **kw):
    return {"case_id": case_id, "labeling_order": 1, "provenance": "human", "blind_human": True, "text": "@VirginTrains is wifi free?",
            "gold_intent": "onboard_wifi_issue", "gold_should_escalate": "no", "gold_resolution_type": "information_provided",
            "conversation": CONVERSATION, "first_timestamp": "2017-11-05 22:35:12+00:00", "gold_confidence": None, **kw}


def record(case_id="case_11", action="AUTO_HANDLE", **kw):
    return {"case_id": case_id, "final_action": action, "classification_failed": False, "reasons": [], "evidence": [], **kw}


def test_parse_conversation_roles_and_continuation_lines():
    turns = parse_conversation(CONVERSATION)
    assert [t["role"] for t in turns] == ["customer", "brand", "other_operator"]
    assert turns[0]["text"] == "@VirginTrains is wifi free?\nsecond line"
    assert turns[1]["tweet_id"] == "12"
    assert parse_conversation("") == []


@pytest.mark.parametrize("rec,status", [
    (None, "pending"),
    (record(), "auto_handled"),
    (record(action="ESCALATE"), "escalated"),
    (record(action="ESCALATE", classification_failed=True), "model_error"),
    (record(action="ESCALATE", model_error="429"), "model_error"),
])
def test_ticket_status(rec, status):
    assert ticket_status(rec) == status


def test_build_tickets_marks_cached_results_and_keeps_gold_separate():
    cases = pd.DataFrame([case("case_11"), case("case_22")])
    tickets = build_tickets(cases, [record("case_11")])
    assert [(t["case_id"], t["status"], t["source"]) for t in tickets] == [("case_11", "auto_handled", "cached"), ("case_22", "pending", None)]
    assert tickets[0]["gold"]["intent"] == "onboard_wifi_issue" and tickets[0]["gold"]["blind_human"] is True
    assert tickets[0]["gold"]["confidence"] is None
    assert "judge" not in json.dumps(tickets)


def test_run_requires_live_runner_and_known_case():
    app = CopilotApp(build_tickets(pd.DataFrame([case()]), []), {"cached_results": 0})
    assert app.payload()["meta"]["live_available"] is False
    with pytest.raises(RuntimeError):
        app.run("case_11")
    with pytest.raises(KeyError):
        app.run("case_99")


def test_live_run_replaces_ticket_and_labels_source():
    app = CopilotApp(build_tickets(pd.DataFrame([case()]), []), {}, live_runner=lambda cid: record(cid, action="ESCALATE"))
    t = app.run("case_11")
    assert (t["status"], t["source"]) == ("escalated", "live")
    assert app.payload()["tickets"][0]["source"] == "live"


@pytest.fixture
def server():
    calls = []

    def runner(cid):
        calls.append(cid)
        if cid == "case_22":
            raise RuntimeError("Groq rate limit reached")
        return record(cid)

    app = CopilotApp(build_tickets(pd.DataFrame([case("case_11"), case("case_22")]), []), {"slice_size": 2}, live_runner=runner)
    srv = _Server(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()
    srv.server_close()


def _get(url, method="GET"):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_routes(server):
    base, calls = server
    status, body = _get(base + "/?v=1")
    assert status == 200 and b"Support Copilot" in body
    status, body = _get(base + "/api/tickets")
    assert status == 200 and len(json.loads(body)["tickets"]) == 2
    status, body = _get(base + "/api/run/case_11", "POST")
    assert status == 200 and json.loads(body)["source"] == "live"
    status, body = _get(base + "/api/run/case_22", "POST")
    assert status == 503 and "rate limit" in json.loads(body)["error"]
    assert _get(base + "/api/run/case_99", "POST")[0] == 404
    assert _get(base + "/api/run/../../etc", "POST")[0] == 404
    assert _get(base + "/nope")[0] == 404
    assert calls == ["case_11", "case_22"]


def test_page_escapes_data_and_never_mentions_judge_scores():
    html = PAGE.read_text(encoding="utf-8")
    assert "function esc(" in html
    assert "judge" not in html.lower()
