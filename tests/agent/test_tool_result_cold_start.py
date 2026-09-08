"""Exercise real cold-start result persistence without model/provider calls."""

import hashlib
import json
from pathlib import Path
import re
import shlex
import socket
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import tool_executor
from tools import terminal_tool as terminal
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import enforce_turn_budget, maybe_persist_tool_result


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("network access is not part of result persistence")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    yield
    assert not attempts


@pytest.fixture
def cold_local(tmp_path, monkeypatch):
    temporary = tmp_path / "sandbox temporary"
    temporary.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setenv("TERMINAL_LOCAL_PERSISTENT", "false")
    monkeypatch.setattr(terminal, "_active_environments", {})
    monkeypatch.setattr(terminal, "_last_activity", {})
    monkeypatch.setattr(terminal, "_creation_locks", {})
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    yield "cold-result-task", temporary
    for environment in terminal._active_environments.values():
        environment.cleanup()


def large_result():
    note = {
        "id": "00000000-0000-0000-0000-000000000001",
        "updateSequenceNumber": 123,
        "content": "<en-note>" + "fixture \u00e9\n" * 25_000 + "</en-note>",
        "attributes": {"sourceURL": "https://example.invalid/source"},
    }
    return json.dumps({"result": "A readable note", "structuredContent": {"note": note}}, ensure_ascii=False)


def assert_saved_exact(message, expected, task_id, temporary):
    match = re.search(r"Full output saved to: ([^\n]+)", message)
    assert match, message[:200]
    path = Path(match.group(1))
    assert path.parent == temporary / "hermes-results"
    assert path.read_bytes() == expected.encode("utf-8")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    environment = terminal.get_active_env(task_id)
    result = environment.execute(f"sha256sum {shlex.quote(str(path))}", timeout=10)
    assert result["returncode"] == 0
    assert result["output"].split()[0] == hashlib.sha256(expected.encode()).hexdigest()
    return path


@pytest.mark.linux_only
def test_first_large_json_result_uses_real_local_stdin_and_is_retrievable(cold_local):
    task_id, temporary = cold_local
    payload = large_result()
    assert len(payload.encode()) > 128 * 1024
    assert terminal.get_active_env(task_id) is None
    result = maybe_persist_tool_result(payload, "mcp__fixture__get_note", "first-note", task_id=task_id)
    path = assert_saved_exact(result, payload, task_id, temporary)
    restored = json.loads(path.read_text())["structuredContent"]["note"]
    assert restored["updateSequenceNumber"] == 123
    assert restored["attributes"]["sourceURL"] == "https://example.invalid/source"


def test_small_result_and_small_turn_never_create_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("small result must not initialize a backend")
    monkeypatch.setattr(terminal, "ensure_task_env", forbidden)
    assert maybe_persist_tool_result("small", "fixture", "small", task_id="cold") == "small"
    messages = [{"role": "tool", "content": "small", "tool_call_id": "small"}]
    enforce_turn_budget(messages, task_id="cold")
    assert messages[0]["content"] == "small"


@pytest.mark.linux_only
def test_aggregate_only_overflow_initializes_local_environment(cold_local):
    task_id, temporary = cold_local
    payload = "a" * 80_000
    messages = [{"role": "tool", "content": payload, "tool_call_id": f"aggregate-{i}"} for i in range(3)]
    assert terminal.get_active_env(task_id) is None
    enforce_turn_budget(messages, task_id=task_id)
    persisted = [message for message in messages if "<persisted-output>" in message["content"]]
    assert persisted
    for message in persisted:
        assert_saved_exact(message["content"], payload, task_id, temporary)
    assert sum(len(message["content"]) for message in messages) <= 200_000


@pytest.mark.parametrize("failure", [None, "creation", "write"])
def test_remote_environment_is_used_without_host_fallback(cold_local, monkeypatch, failure):
    task_id, temporary = cold_local
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    calls = []
    class Remote:
        def get_temp_dir(self):
            return "/remote-only/tmp"
        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"returncode": 1 if failure == "write" else 0}
        def cleanup(self):
            pass
    def create(**kwargs):
        assert kwargs["env_type"] == "ssh"
        if failure == "creation":
            raise RuntimeError("remote unavailable")
        return Remote()
    monkeypatch.setattr(terminal, "_create_environment", create)
    payload = large_result()
    result = maybe_persist_tool_result(payload, "fixture", "remote-note", task_id=task_id)
    assert not (temporary / "hermes-results").exists()
    if failure == "creation":
        assert not calls and "could not be saved" in result
        assert terminal.get_active_env(task_id) is None
    else:
        assert len(calls) == 1 and calls[0][1]["stdin_data"] == payload
        if failure == "write":
            assert "could not be saved" in result
        else:
            assert "Full output saved to: /remote-only/tmp/hermes-results/remote-note.txt" in result


@pytest.fixture
def agent(monkeypatch):
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256_000),
        patch("agent.context_compressor.get_model_context_length", return_value=256_000),
    ):
        instance = AIAgent(api_key="fixture", base_url="https://example.invalid/v1",
                           provider="openai-compat", model="fixture/model", quiet_mode=True,
                           skip_context_files=True, skip_memory=True)
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance._append_guardrail_observation = lambda name, args, result, **kwargs: result
    yield instance
    instance.close()


@pytest.mark.linux_only
@pytest.mark.parametrize("mode", ["sequential", "concurrent", "segmented"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_executor_paths_preserve_cold_results(agent, cold_local, monkeypatch, mode, aggregate):
    task_id, temporary = cold_local
    payload = "aggregate fixture " * 5000 if aggregate else large_result()
    if aggregate:
        assert len(payload) < 100_000
    calls = [SimpleNamespace(id=f"tool-{i}", function=SimpleNamespace(name="mcp__fixture__get_note", arguments="{}"))
             for i in range(3 if aggregate else 1)]
    agent.valid_tool_names = {"mcp__fixture__get_note"}
    agent._invoke_tool = lambda *args, **kwargs: payload
    monkeypatch.setattr("run_agent.handle_function_call", lambda *args, **kwargs: payload)
    monkeypatch.setattr(tool_executor, "_budget_for_agent", lambda _: BudgetConfig())
    messages = []
    assistant = SimpleNamespace(tool_calls=calls)
    if mode == "segmented":
        tool_executor.execute_tool_calls_segmented(agent, assistant, messages, task_id,
                                                  segments=[("sequential", calls[:1]), ("parallel", calls[1:])])
    else:
        getattr(tool_executor, f"execute_tool_calls_{mode}")(agent, assistant, messages, task_id)
    assert len(messages) == len(calls)
    persisted = [message for message in messages if "<persisted-output>" in message["content"]]
    assert persisted
    for message in persisted:
        expected = (
            tool_executor.make_tool_result_message("mcp__fixture__get_note", payload, message["tool_call_id"])["content"]
            if aggregate else payload
        )
        assert_saved_exact(message["content"], expected, task_id, temporary)
