"""Support Copilot demo: a local, read-mostly view over real VirginTrains golden cases and the agent's results.

Tickets are the 50-case end-to-end evaluation slice. A ticket shows the cached result from `scripts/evaluate_agent.py` when
one exists (labelled "cached"), otherwise it is "not analysed yet" and can be run live with the real agent (Groq), which
writes to a separate cache so demo use never touches the evaluation artifacts. Nothing is invented: messages and threads
come from the golden pack; analysis comes from the agent. Judge scores are never shown (they must stay hidden from human
raters). Served on 127.0.0.1 only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pandas as pd

logger = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("copilot.html")
_LINE = re.compile(r"^\[(CUSTOMER|AGENT|OTHER-AGENT) (\d+)\] ?(.*)$")
ROLE = {"CUSTOMER": "customer", "AGENT": "brand", "OTHER-AGENT": "other_operator"}

AUTO_HANDLED, ESCALATED, PENDING, MODEL_ERROR = "auto_handled", "escalated", "pending", "model_error"


def parse_conversation(text: str) -> list[dict[str, str]]:
    """Golden `conversation` text -> turns. Lines without a role tag continue the previous turn."""
    turns: list[dict[str, str]] = []
    for line in str(text or "").splitlines():
        m = _LINE.match(line)
        if m:
            turns.append({"role": ROLE[m.group(1)], "tweet_id": m.group(2), "text": m.group(3)})
        elif turns and line.strip():
            turns[-1]["text"] += "\n" + line
    return turns


def ticket_status(record: dict[str, Any] | None) -> str:
    if record is None:
        return PENDING
    if record.get("classification_failed") or record.get("model_error"):
        return MODEL_ERROR
    return AUTO_HANDLED if record["final_action"] == "AUTO_HANDLE" else ESCALATED


def build_ticket(case: dict[str, Any], record: dict[str, Any] | None, source: str | None) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "opened_at": str(case.get("first_timestamp") or ""),
        "message": case["text"],
        "turns": parse_conversation(case.get("conversation", "")),
        "gold": {
            "intent": case["gold_intent"], "should_escalate": case["gold_should_escalate"], "resolution_type": case["gold_resolution_type"],
            "confidence": case.get("gold_confidence") or None, "provenance": case["provenance"], "blind_human": bool(case["blind_human"]),
        },
        "status": ticket_status(record),
        "source": source if record is not None else None,
        "result": record,
    }


def build_tickets(cases: pd.DataFrame, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["case_id"]: r for r in records}
    rows = cases.astype(object).where(cases.notna(), None).to_dict("records")
    return [build_ticket(c, by_id.get(c["case_id"]), "cached" if c["case_id"] in by_id else None) for c in rows]


def load_records(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class CopilotApp:
    """Ticket state plus an optional live runner `(case_id) -> record`. Live runs are serialised."""

    def __init__(self, tickets: list[dict[str, Any]], meta: dict[str, Any], live_runner: Callable[[str], dict[str, Any]] | None = None):
        self.tickets = {t["case_id"]: t for t in tickets}
        self.order = [t["case_id"] for t in tickets]
        self.meta = {**meta, "live_available": live_runner is not None}
        self.live_runner = live_runner
        self.lock = threading.Lock()

    def payload(self) -> dict[str, Any]:
        return {"meta": self.meta, "tickets": [self.tickets[c] for c in self.order]}

    def run(self, case_id: str) -> dict[str, Any]:
        if case_id not in self.tickets:
            raise KeyError(case_id)
        if self.live_runner is None:
            raise RuntimeError("Live mode is off: set GROQ_API_KEY in .env and restart.")
        with self.lock:
            record = self.live_runner(case_id)
            ticket = {**self.tickets[case_id], "result": record, "status": ticket_status(record), "source": "live"}
            self.tickets[case_id] = ticket
            return ticket


def make_handler(app: CopilotApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/tickets":
                self._json(HTTPStatus.OK, app.payload())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            m = re.fullmatch(r"/api/run/(case_\d+)", self.path)
            if not m:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                self._json(HTTPStatus.OK, app.run(m.group(1)))
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown case"})
            except Exception as exc:  # the page shows the message; the server keeps running
                logger.warning("live run failed: %s", exc)
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)[:300]})

    return Handler


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server silently share the port; fail instead.
    allow_reuse_address = os.name != "nt"


def serve(app: CopilotApp, port: int = 8770, open_browser: bool = True) -> None:
    server = _Server(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"Support Copilot: {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
