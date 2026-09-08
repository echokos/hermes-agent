"""Integration coverage for Cron required-tool dependency outcomes."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cron.scheduler as scheduler
from tools import mcp_tool


REQUIRED = "mcp__nirvana__get_tasks"


class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


def _result(text: str, *, error: bool = False):
    return SimpleNamespace(
        content=[_Block(text)],
        isError=error,
        structuredContent=None,
        meta=None,
    )


@pytest.fixture
def mcp_runtime(monkeypatch):
    servers = {}

    def install(server_name: str, *responses, tool_name="get_tasks"):
        session = SimpleNamespace(call_tool=AsyncMock(side_effect=responses))
        server = SimpleNamespace(session=session, _rpc_lock=None)
        servers[server_name] = server
        mcp_tool._servers[server_name] = server
        return mcp_tool._make_tool_handler(server_name, tool_name, 10.0)

    def run(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

        async def scoped():
            for server in servers.values():
                server._rpc_lock = asyncio.Lock()
            return await coro

        return asyncio.run(scoped())

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run)
    yield install
    for name in servers:
        mcp_tool._servers.pop(name, None)
        mcp_tool._server_error_counts.pop(name, None)
        mcp_tool._server_breaker_opened_at.pop(name, None)


def _job(**updates):
    job = {
        "id": "daily-note",
        "name": "Daily note",
        "workflow_id": "daily-note-workflow",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "123"},
        "required_tool_dependencies": [REQUIRED],
        "failure_ownership": {
            "technical_owner": "root",
            "director": "aurora",
        },
    }
    job.update(updates)
    return job


def _run_cron(monkeypatch, tmp_path, run_job, job=None):
    delivered = []
    marked = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "run_job", run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: tmp_path / "out")
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content),
    )
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda job_id, success, error=None, **kwargs: marked.append(
            (job_id, success, error, kwargs)
        ),
    )

    assert scheduler.run_one_job(job or _job()) is True
    intake_path = tmp_path / "cron" / "operational-failures.jsonl"
    events = (
        [json.loads(line) for line in intake_path.read_text().splitlines()]
        if intake_path.exists()
        else []
    )
    return delivered, marked, events


@pytest.mark.parametrize("mode", [None, "always", "when_invoked"])
def test_caught_mcp_401_keeps_partial_artifact_and_records_dependency_failure(
    monkeypatch, tmp_path, mcp_runtime, mode,
):
    handler = mcp_runtime("nirvana", _result("Unauthorized 401 raw-provider", error=True))

    def run_job(_job, **_kwargs):
        assert "error" in json.loads(handler({}))
        return True, "saved partial", "Useful partial daily note", None

    delivered, marked, events = _run_cron(
        monkeypatch, tmp_path, run_job,
        _job(required_tool_dependency_mode=mode),
    )

    assert marked[0][1] is True
    assert delivered and delivered[0].startswith("Useful partial daily note")
    assert "repair intake was recorded" in delivered[0]
    assert "owner acceptance is not yet verified here" in delivered[0]
    assert events[0]["failure_type"] == "required_tool_dependency"
    assert events[0]["dependency_outcome"]["failed"] == [
        {"tool": REQUIRED, "reasons": ["tool_error"]}
    ]
    assert "raw-provider" not in json.dumps(events[0])


def test_missing_required_call_cannot_emit_false_recovery(
    monkeypatch, tmp_path,
):
    delivered, marked, events = _run_cron(
        monkeypatch,
        tmp_path,
        lambda _job, **_kwargs: (True, "out", "Partial result", None),
    )

    assert marked[0][1] is True
    assert events[0]["status"] == "failure"
    assert events[0]["dependency_outcome"]["missing"] == [REQUIRED]
    assert "degraded" in delivered[0]


@pytest.mark.parametrize("branch", ["preserve_nonempty_next_things", "same_day_noop"])
def test_conditional_no_call_preserves_result_without_false_recovery(
    monkeypatch, tmp_path, mcp_runtime, branch,
):
    required = mcp_runtime("nirvana", _result("Unauthorized", error=True))
    note = mcp_runtime("evernote", _result("Existing Next Things"), tool_name="get_note")
    job = _job(required_tool_dependency_mode="when_invoked")

    def failing_run(_job, **_kwargs):
        required({})
        return True, "partial", "Partial note", None

    _run_cron(monkeypatch, tmp_path, failing_run, job)
    intake = tmp_path / "cron" / "operational-failures.jsonl"
    previous_intake = intake.read_bytes()

    def preserving_run(_job, **_kwargs):
        note({"guid": "existing-note"})
        return True, "Existing Next Things", branch, None

    delivered, marked, events = _run_cron(
        monkeypatch, tmp_path, preserving_run, job,
    )

    assert marked[0][1] is True
    assert marked[0][3]["dependency_status"] == "not_observed"
    assert marked[0][3]["dependency_outcome"]["missing"] == [REQUIRED]
    assert delivered == [branch]
    assert intake.read_bytes() == previous_intake
    assert [event["status"] for event in events] == ["failure"]


def test_conditional_silent_no_call_is_not_a_failure(monkeypatch, tmp_path):
    delivered, marked, events = _run_cron(
        monkeypatch, tmp_path,
        lambda _job, **_kwargs: (True, "out", "[SILENT]", None),
        _job(required_tool_dependency_mode="when_invoked"),
    )
    assert delivered == []
    assert events == []
    assert marked[0][3]["dependency_status"] == "not_observed"


def test_conditional_partial_observation_cannot_emit_recovery(
    monkeypatch, tmp_path, mcp_runtime,
):
    handler = mcp_runtime("nirvana", _result("tasks"))

    def run_job(_job, **_kwargs):
        handler({})
        return True, "out", "Preserved note", None

    delivered, marked, events = _run_cron(
        monkeypatch, tmp_path, run_job,
        _job(
            required_tool_dependency_mode="when_invoked",
            required_tool_dependencies=[REQUIRED, "mcp__evernote__get_note"],
        ),
    )
    assert delivered == ["Preserved note"]
    assert events == []
    assert marked[0][3]["dependency_status"] == "not_observed"
    assert marked[0][3]["dependency_outcome"]["successful"] == [REQUIRED]


def test_silent_model_response_cannot_hide_missing_dependency(
    monkeypatch, tmp_path,
):
    delivered, marked, events = _run_cron(
        monkeypatch,
        tmp_path,
        lambda _job, **_kwargs: (True, "out", "[SILENT]", None),
    )

    assert marked[0][1] is True
    assert len(events) == 1
    assert delivered == []


def test_unrelated_mcp_error_is_ignored_after_required_call_succeeds(
    monkeypatch, tmp_path, mcp_runtime,
):
    required = mcp_runtime("nirvana", _result("tasks"))
    unrelated = mcp_runtime("other", _result("unrelated failure", error=True))

    def run_job(_job, **_kwargs):
        required({})
        unrelated({})
        return True, "out", "Complete result", None

    delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)

    assert marked[0][1] is True
    assert delivered == ["Complete result"]
    assert events[0]["status"] == "recovered"


def test_later_successful_retry_is_the_terminal_dependency_outcome(
    monkeypatch, tmp_path, mcp_runtime,
):
    handler = mcp_runtime(
        "nirvana",
        _result("Unauthorized", error=True),
        _result("tasks"),
    )

    def run_job(_job, **_kwargs):
        handler({})
        handler({})
        return True, "out", "Complete after retry", None

    delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)

    assert marked[0][1] is True
    assert delivered == ["Complete after retry"]
    assert events[0]["status"] == "recovered"


def test_success_for_distinct_arguments_does_not_clear_failed_invocation(
    monkeypatch, tmp_path, mcp_runtime,
):
    handler = mcp_runtime(
        "nirvana",
        _result("Unauthorized", error=True),
        _result("due tasks"),
    )

    def run_job(_job, **_kwargs):
        handler({"filter": "starred", "private": "must-not-persist"})
        handler({"filter": "due_today"})
        return True, "out", "Partial due-today result", None

    delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)

    assert marked[0][1] is True
    assert "degraded" in delivered[0]
    assert events[0]["dependency_outcome"]["failed"] == [
        {"tool": REQUIRED, "reasons": ["tool_error"]}
    ]
    assert "must-not-persist" not in json.dumps(events[0])
    assert "starred" not in json.dumps(events[0])


@pytest.mark.parametrize("mode", ["always", "when_invoked"])
def test_detached_pending_distinct_call_keeps_completed_artifact_degraded(
    monkeypatch, tmp_path, mcp_runtime, mode,
):
    from tools.thread_context import propagate_context_to_thread

    handler = mcp_runtime(
        "nirvana",
        _result("completed A"),
        _result("late completed B"),
    )
    run_on_mcp_loop = mcp_tool._run_on_mcp_loop
    pending_started = threading.Event()
    release_pending = threading.Event()
    calls = 0

    def blocked_second_call(coro_or_factory, timeout=30):
        nonlocal calls
        calls += 1
        if calls == 2:
            pending_started.set()
            assert release_pending.wait(5)
        return run_on_mcp_loop(coro_or_factory, timeout=timeout)

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", blocked_second_call)
    detached = []

    def run_job(_job, **_kwargs):
        handler({"filter": "completed_a"})
        worker = threading.Thread(
            target=propagate_context_to_thread(
                lambda: handler({"filter": "detached_b"})
            ),
            daemon=True,
        )
        worker.start()
        detached.append(worker)
        assert pending_started.wait(5)
        return True, "out", "Partial result from A", None

    delivered, marked, events = _run_cron(
        monkeypatch, tmp_path, run_job, _job(required_tool_dependency_mode=mode),
    )
    try:
        assert marked[0][1] is True
        assert "degraded" in delivered[0]
        assert events[0]["dependency_outcome"]["failed"] == [
            {"tool": REQUIRED, "reasons": ["pending"]}
        ]
    finally:
        release_pending.set()
        detached[0].join(5)
    assert not detached[0].is_alive()
    assert marked[0][3]["dependency_outcome"]["failed"] == [
        {"tool": REQUIRED, "reasons": ["pending"]}
    ]


def test_finalized_summary_ignores_late_worker_completion():
    from tools.required_dependency_runtime import (
        activate,
        mark_pending,
        mark_success,
        reset,
    )

    token, state = activate([REQUIRED])
    try:
        attempt = mark_pending(REQUIRED, {"filter": "detached"})
        summary = state.finalize()
        mark_success(attempt)

        assert state.finalize() is summary
        assert summary["failed"] == [
            {"tool": REQUIRED, "reasons": ["pending"]}
        ]
    finally:
        reset(token)


def test_older_same_args_completion_cannot_erase_newer_pending_attempt():
    from tools.required_dependency_runtime import (
        activate,
        mark_pending,
        mark_success,
        reset,
    )

    token, state = activate([REQUIRED])
    try:
        older = mark_pending(REQUIRED, {"filter": "same"})
        newer = mark_pending(REQUIRED, {"filter": "same"})
        mark_success(older)

        summary = state.finalize()
        assert summary["failed"] == [
            {"tool": REQUIRED, "reasons": ["pending"]}
        ]

        mark_success(newer)
        assert state.finalize() is summary
        assert summary["failed"] == [
            {"tool": REQUIRED, "reasons": ["pending"]}
        ]
    finally:
        reset(token)


def test_multiple_distinct_completed_calls_satisfy_dependency(
    monkeypatch, tmp_path, mcp_runtime,
):
    handler = mcp_runtime(
        "nirvana",
        _result("starred tasks"),
        _result("due tasks"),
    )

    def run_job(_job, **_kwargs):
        handler({"filter": "starred"})
        handler({"filter": "due_today"})
        return True, "out", "Complete result", None

    delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)

    assert marked[0][1] is True
    assert delivered == ["Complete result"]
    assert events[0]["status"] == "recovered"


def test_successful_call_result_text_that_looks_like_error_stays_successful(
    monkeypatch, tmp_path, mcp_runtime,
):
    handler = mcp_runtime("nirvana", _result('{"error":"domain data"}'))

    def run_job(_job, **_kwargs):
        assert "result" in json.loads(handler({"filter": "all"}))
        return True, "out", "Complete result", None

    delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)

    assert marked[0][1] is True
    assert delivered == ["Complete result"]
    assert events[0]["status"] == "recovered"


@pytest.mark.parametrize("failure_mode", ["transport", "circuit"])
def test_transport_and_circuit_failures_are_authoritative_dependency_failures(
    monkeypatch, tmp_path, failure_mode,
):
    mcp_tool._servers.pop("nirvana", None)
    if failure_mode == "circuit":
        mcp_tool._server_error_counts["nirvana"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        mcp_tool._server_breaker_opened_at["nirvana"] = mcp_tool.time.monotonic()
    else:
        mcp_tool._server_error_counts.pop("nirvana", None)
        mcp_tool._server_breaker_opened_at.pop("nirvana", None)
    handler = mcp_tool._make_tool_handler("nirvana", "get_tasks", 10.0)

    def run_job(_job, **_kwargs):
        assert "error" in json.loads(handler({}))
        return True, "out", "Partial", None

    try:
        _delivered, marked, events = _run_cron(monkeypatch, tmp_path, run_job)
        reasons = events[0]["dependency_outcome"]["failed"][0]["reasons"]
        assert reasons == [
            "circuit_open" if failure_mode == "circuit" else "transport_unavailable"
        ]
        assert marked[0][1] is True
    finally:
        mcp_tool._server_error_counts.pop("nirvana", None)
        mcp_tool._server_breaker_opened_at.pop("nirvana", None)


@pytest.mark.parametrize("mode", [None, "when_invoked"])
def test_whole_run_failure_remains_suppressed_with_dependency_observation(
    monkeypatch, tmp_path, mode,
):
    delivered, marked, events = _run_cron(
        monkeypatch,
        tmp_path,
        lambda _job, **_kwargs: (False, "partial", "", "whole run failed"),
        _job(required_tool_dependency_mode=mode),
    )

    assert delivered == []
    assert marked[0][1] is False
    assert events[0]["failure_type"] == "execution"


@pytest.mark.parametrize("mode", [None, "always", "when_invoked"])
@pytest.mark.parametrize("invoked", [True, False])
def test_dependency_health_is_persisted_with_native_job_run_lock(
    monkeypatch, tmp_path, mcp_runtime, mode, invoked,
):
    from cron import jobs as jobs_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handler = mcp_runtime("nirvana", _result("tasks"))
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)

    def run_job(_job, **_kwargs):
        if invoked:
            handler({"filter": "all"})
        return True, "out", "Complete result", None

    monkeypatch.setattr(
        scheduler,
        "run_job",
        run_job,
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: tmp_path / "out")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)

    with jobs_mod.use_cron_store(tmp_path):
        job = jobs_mod.create_job(
            prompt="Collect tasks",
            schedule="every 1h",
            name="Daily note",
            deliver="local",
            workflow_slug="daily-note-workflow",
            required_tool_dependencies=[REQUIRED],
            required_tool_dependency_mode=mode,
            failure_ownership={"technical_owner": "root", "director": "aurora"},
        )
        assert scheduler.run_one_job(job) is True
        stored = jobs_mod.get_job(job["id"])

    assert stored["last_status"] == "ok"
    expected_status = (
        "healthy" if invoked else "not_observed" if mode == "when_invoked" else "degraded"
    )
    assert stored["last_dependency_status"] == expected_status
    assert stored["last_dependency_outcome"] == {
        "required": [REQUIRED],
        "successful": [REQUIRED] if invoked else [],
        "failed": [],
        "missing": [] if invoked else [REQUIRED],
    }
    intake = tmp_path / "cron" / "operational-failures.jsonl"
    events = [json.loads(line) for line in intake.read_text().splitlines()] if intake.exists() else []
    assert [event["status"] for event in events] == (
        ["recovered"] if invoked else [] if mode == "when_invoked" else ["failure"]
    )
