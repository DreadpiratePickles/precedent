"""The `precedent-mcp` server: two read-only tools over stdio, on top of `run_card`."""

from __future__ import annotations

import asyncio
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
from mcp import Client

from precedent import mcp_server
from precedent.engine.card import DISCLAIMER, KEY_FLAGS_HINT

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


async def _call_through_client(name: str, arguments: dict[str, Any]):
    async with Client(mcp_server.server) as client:
        return await client.call_tool(name, arguments)


def call_tool(name: str, arguments: dict[str, Any]):
    return asyncio.run(_call_through_client(name, arguments))


def error_text(result) -> str:
    assert result.is_error
    return " ".join(block.text for block in result.content)


@pytest.mark.parametrize("tool", ["precedents", "card"])
def test_client_sees_the_message_when_neither_text_nor_key_is_given(
    tool, monkeypatch, library_root
):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    message = error_text(call_tool(tool, {}))
    assert "exactly one of `text` and `key`" in message


@pytest.mark.parametrize("tool", ["precedents", "card"])
def test_client_sees_the_message_for_an_unknown_key_field(tool, monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    key = {**CENTRAL_BANK_US_HIGH_DICT, "colour": "red"}
    message = error_text(call_tool(tool, {"key": key}))
    assert "unknown event key field(s): colour" in message


@pytest.mark.parametrize("tool", ["precedents", "card"])
def test_client_sees_the_message_for_a_missing_library(tool, monkeypatch, tmp_path):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    message = error_text(call_tool(tool, {"key": CENTRAL_BANK_US_HIGH_DICT}))
    assert "precedent fetch" in message and "precedent build" in message


def test_client_sees_the_message_for_a_bad_horizon(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    message = error_text(call_tool("precedents", {"key": CENTRAL_BANK_US_HIGH_DICT, "horizon": 7}))
    assert "horizon" in message


def test_client_sees_the_message_and_key_flags_for_a_typer_config_error(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    monkeypatch.setenv("PRECEDENT_TYPER", "nonsense")
    message = error_text(call_tool("card", {"text": "Fed cuts rates"}))
    assert "unknown typer" in message and KEY_FLAGS_HINT in message


def test_precedents_through_the_client_returns_the_card_dict(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    result = call_tool("precedents", {"key": CENTRAL_BANK_US_HIGH_DICT, "horizon": 5})
    assert not result.is_error
    assert json.loads(result.content[0].text)["schema_version"] == "1.0"


def test_card_through_the_client_returns_text_ending_with_the_disclaimer(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    result = call_tool("card", {"key": CENTRAL_BANK_US_HIGH_DICT, "horizon": 5})
    assert not result.is_error
    assert result.content[0].text.rstrip().endswith(DISCLAIMER)


def send(proc: subprocess.Popen, message: dict[str, Any]) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


class ResponseReader:
    """Owns the one reader thread and queue of a server process, so no response is lost."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._pump, args=(proc,), daemon=True)
        self._thread.start()

    def _pump(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def read_response(self, response_id: int, timeout_s: float) -> dict[str, Any]:
        """Read JSON lines until the response with `response_id`; fail on timeout."""
        while True:
            try:
                line = self.lines.get(timeout=timeout_s)
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
        reader = ResponseReader(proc)
        reader.read_response(1, timeout_s=20)
        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = sorted(t["name"] for t in reader.read_response(2, timeout_s=20)["result"]["tools"])
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


def test_untypable_text_with_no_library_returns_the_refusal_card(monkeypatch, tmp_path):
    # V-4 / P-47: the amendment says untypable text gets the refusal and the key flags, library
    # or not
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    text = mcp_server.card(text="zzzz qqqq")
    assert "typed as       UNTYPED" in text and "--event-type" in text
    data = mcp_server.precedents(text="zzzz qqqq")
    assert data["typed_as"] == "UNTYPED" and data["refused"] is True and data["covers"] == []


def test_typable_text_with_no_library_is_still_a_tool_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    message = error_text(
        call_tool("card", {"text": "The Federal Reserve raised interest rates unexpectedly"})
    )
    assert "precedent fetch" in message


def test_a_keyerror_from_retrieval_becomes_a_tool_error(monkeypatch, library_root):
    # M4 / P-47: a session missing from the cutoffs raises KeyError inside find_precedents
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))

    def missing_session(*args, **kwargs):
        raise KeyError("response_complete_session 2020-02-01 is not in session_cutoffs_utc")

    monkeypatch.setattr(mcp_server, "run_card", missing_session)
    message = error_text(call_tool("precedents", {"key": CENTRAL_BANK_US_HIGH_DICT}))
    assert "response_complete_session 2020-02-01" in message


def test_ask_still_exports_the_card_api():
    from precedent.engine import ask, card_build

    assert ask.run_card is card_build.run_card
    assert ask.refused_card_without_library is card_build.refused_card_without_library
    assert ask.UNTYPED == card_build.UNTYPED == "UNTYPED"
