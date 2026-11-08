"""The `precedent-mcp` server: two read-only tools over stdio, on top of `run_card`."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from subprocess import PIPE
from typing import Any

import pytest
from library_root_synthetic import write_library_root

from precedent import mcp_server
from precedent.engine.card import DISCLAIMER

CENTRAL_BANK_US_HIGH_DICT = {
    "event_type": "central_bank",
    "actor_kind": "central_bank",
    "country_group": "US",
    "severity_band": "high",
    "is_scheduled": False,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def library_root(tmp_path: Path) -> Path:
    return write_library_root(tmp_path)


def test_precedents_returns_card_json(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    data = mcp_server.precedents(key=CENTRAL_BANK_US_HIGH_DICT, horizon=5)
    assert data["schema_version"] == "1.0" and data["disclaimer"] == DISCLAIMER


def test_untyped_text_returns_the_refusal_and_the_key_flags(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    text = mcp_server.card(text="zzzz qqqq")
    assert "typed as       UNTYPED" in text and "--event-type" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"text": "x", "key": CENTRAL_BANK_US_HIGH_DICT},
        {"key": {**CENTRAL_BANK_US_HIGH_DICT, "colour": "red"}},
    ],
)
def test_bad_arguments_raise(kwargs, monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    with pytest.raises(ValueError):
        mcp_server.precedents(**kwargs)
    with pytest.raises(ValueError):
        mcp_server.card(**kwargs)


def test_card_for_a_key_is_the_text_card_without_the_hint(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    text = mcp_server.card(key=CENTRAL_BANK_US_HIGH_DICT, horizon=5)
    assert DISCLAIMER in text and "--event-type" not in text


def send(proc: subprocess.Popen, message: dict[str, Any]) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


_LINE_QUEUES: dict[int, queue.Queue[str | None]] = {}


def _lines_of(proc: subprocess.Popen) -> queue.Queue[str | None]:
    """One reader thread per process feeds one queue, so no response is lost between reads."""
    if id(proc) not in _LINE_QUEUES:
        lines: queue.Queue[str | None] = queue.Queue()
        _LINE_QUEUES[id(proc)] = lines

        def pump() -> None:
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
    return _LINE_QUEUES[id(proc)]


def read_response(proc: subprocess.Popen, response_id: int, timeout_s: float) -> dict[str, Any]:
    """Read JSON lines from stdout until the response with `response_id`; fail on timeout."""
    lines = _lines_of(proc)
    while True:
        try:
            line = lines.get(timeout=timeout_s)
        except queue.Empty:
            pytest.fail(f"no response with id {response_id} within {timeout_s} s")
        if line is None:
            pytest.fail(f"server closed stdout before response {response_id}")
        if not line.strip():
            continue
        message = json.loads(line)
        if message.get("id") == response_id:
            return message


def test_server_lists_two_tools_over_stdio(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "precedent.mcp_server"],
        stdin=PIPE,
        stdout=PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={
            **os.environ,
            "PRECEDENT_DATA_DIR": str(tmp_path),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    try:
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        read_response(proc, 1, timeout_s=20)
        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = sorted(t["name"] for t in read_response(proc, 2, timeout_s=20)["result"]["tools"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    assert names == ["card", "precedents"]


def test_run_calls_main_and_returns_zero(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(mcp_server, "main", lambda: calls.append(1))
    assert mcp_server.run([]) == 0
    assert calls == [1]


def test_core_modules_import_no_provider_and_no_research_extra():
    for path in [
        *(REPO_ROOT / "src/precedent/engine").glob("*.py"),
        *(REPO_ROOT / "src/precedent/libraries").glob("*.py"),
        REPO_ROOT / "src/precedent/mcp_server.py",
    ]:
        if path.name in {"precedent_factors.py", "embedder.py"}:
            continue
        text = path.read_text()
        for banned in (
            "anthropic_typer",
            "ollama_typer",
            "api.anthropic.com",
            "import qlib",
            "import lightgbm",
        ):
            assert banned not in text, (path, banned)
