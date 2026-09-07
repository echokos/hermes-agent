"""Terminal-review turns stop after a durable host verdict."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.coordination_budget import scoped_coordination_budget
from agent.conversation_loop import _terminal_review_tool_round_completed
from hermes_cli import kanban_db as kb
from hermes_cli.workforce_handoffs import acknowledge_handoff, create_handoff
from hermes_cli.workforce_org import load_organization
from run_agent import AIAgent


_ORGANIZATION = """
schema_version: 1
agents:
  - agent: elliott
    display_name: Elliott
    status: artifact
    operational: false
    manager: null
    direct_reports: [aurora]
    mission: Own the system
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: aurora
    display_name: Aurora
    status: active
    operational: true
    function: Chief of Staff
    manager: elliott
    direct_reports: [director]
    mission: Coordinate requests
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: director
    display_name: Director
    status: active
    operational: true
    function: Product Director
    manager: aurora
    direct_reports: [builder]
    mission: Direct delivery
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: builder
    display_name: Builder
    status: active
    operational: true
    function: Software Developer
    manager: director
    direct_reports: []
    mission: Implement work
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
"""


def _tool_definitions(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _tool_response(tool_call: SimpleNamespace) -> SimpleNamespace:
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def terminal_review_context(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    organization_dir = home / "organization"
    organization_dir.mkdir(parents=True)
    (organization_dir / "organization.yaml").write_text(_ORGANIZATION)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    organization = load_organization()
    now = int(time.time())
    iso = lambda timestamp: datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

    with kb.connect_closing() as conn:
        created = create_handoff(
            conn,
            source_agent="director",
            target_agent="builder",
            expected_outcome="Repair the failing scheduled workflow",
            acceptance_test="Two distinct later executions succeed",
            evidence_references=["workflow:test"],
            acknowledgment_deadline=iso(now + 60),
            checkpoint_at=iso(now + 240),
            organization=organization,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "builder",
                "director": "director",
                "workflow_id": "scheduled-repair",
                "event_id": "failure-1",
            },
            requires_source_acceptance=True,
        )
        task_id = created["task_id"]
        request = kb.create_owned_failure_coordination_request(
            conn,
            root_task_id=task_id,
            organization=organization,
            now=now,
        )
        acknowledge_handoff(
            conn,
            task_id,
            actor="builder",
            organization=organization,
            now=now + 1,
        )
        owner = kb.claim_task(conn, task_id, claimer="builder:test")
        assert owner is not None
        for ordinal in range(1, 18):
            assert (
                kb.charge_coordination_model_call(
                    conn,
                    request.id,
                    purpose="work",
                    task_id=task_id,
                    now=now + 2,
                )
                == ordinal
            )
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_required",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "required_successes": 2,
                },
            )
            kb._append_event(
                conn,
                task_id,
                "workforce_handoff_recovery_verified",
                {
                    "failure_event_id": "failure-1",
                    "failure_order": 10,
                    "success_event_ids": ["success-1", "success-2"],
                    "success_orders": [11, 12],
                    "required_successes": 2,
                },
            )
        assert kb.request_review(
            conn,
            task_id,
            summary="Repair complete; recovery evidence attached",
            expected_run_id=owner.current_run_id,
        )
        reviewer, _ = kb.claim_task_for_dispatch(
            conn,
            task_id,
            review=True,
            organization=organization,
            now=now + 300,
        )
        assert reviewer is not None
        # ``run_kanban_once`` marks an owned-failure root as terminal review
        # after its generic claim. This direct DB fixture mirrors that
        # dispatcher-derived worker prompt input.
        reviewer.coordination_purpose = "terminal_review"

        source_path = tmp_path / "verified-source.txt"
        source_path.write_text("VERIFIED_SOURCE_OK\n")
        artifact_path = tmp_path / "review-artifact.txt"
        artifact_path.write_text(f"Source to verify: {source_path}\n")
        kb.add_attachment(
            conn,
            task_id,
            filename=artifact_path.name,
            stored_path=str(artifact_path),
            content_type="text/plain",
            size=artifact_path.stat().st_size,
            uploaded_by="builder",
        )
        preface = kb.build_terminal_review_worker_prompt(conn, reviewer, now=now + 300)
        assert preface is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer.current_run_id))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_PROFILE", "director")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", request.id)
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", task_id)
    monkeypatch.setenv("HERMES_COORDINATION_PURPOSE", "terminal_review")
    return (
        home,
        request.id,
        task_id,
        reviewer.current_run_id,
        preface,
        str(artifact_path),
        str(source_path),
    )


def _new_agent(*tool_names: str):
    if not tool_names:
        tool_names = ("read_file", "kanban_complete")
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_tool_definitions(*tool_names),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are a terminal reviewer."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.session_id = "terminal-review-session"
    return agent


def test_terminal_review_complete_stops_after_final_reserved_tool_call(
    terminal_review_context,
):
    """Three real provider admissions end at call 20, with no call 21."""
    (
        _home,
        request_id,
        task_id,
        _run_id,
        preface,
        artifact_path,
        source_path,
    ) = terminal_review_context
    agent = _new_agent()
    agent.max_iterations = 3

    provider_calls = 0

    def provider_response(*_args, **kwargs):
        nonlocal provider_calls
        messages = kwargs["messages"]
        if provider_calls == 0:
            request_text = str(messages)
            assert "[HERMES_HOST_TERMINAL_REVIEW_V1]" in request_text
            assert artifact_path in request_text
            assert os.environ["HERMES_COORDINATION_REQUEST_ROOT"] == request_id
            assert os.environ["HERMES_COORDINATION_TASK_ID"] == task_id
            assert os.environ["HERMES_COORDINATION_PURPOSE"] == "terminal_review"
            response = _tool_response(
                _tool_call("read_file", {"path": artifact_path}, "call-18")
            )
        elif provider_calls == 1:
            artifact_result = messages[-1]
            assert artifact_result["role"] == "tool"
            assert artifact_result["name"] == "read_file"
            assert source_path in artifact_result["content"]
            response = _tool_response(
                _tool_call("read_file", {"path": source_path}, "call-19")
            )
        elif provider_calls == 2:
            source_result = messages[-1]
            assert source_result["role"] == "tool"
            assert source_result["name"] == "read_file"
            assert "VERIFIED_SOURCE_OK" in source_result["content"]
            response = _tool_response(
                _tool_call(
                    "kanban_complete",
                    {"summary": "Source accepted verified repair"},
                    "call-20",
                )
            )
        else:
            raise AssertionError("terminal verdict must prevent provider call 21")
        provider_calls += 1
        return response

    agent.client.chat.completions.create.side_effect = provider_response

    persisted = {}

    def persist(messages, *_args):
        persisted["messages"] = [dict(message) for message in messages]

    with (
        patch.object(agent, "_persist_session", side_effect=persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.title_generator.maybe_auto_title") as title_call,
        patch("agent.turn_finalizer._record_kanban_budget_exhausted") as record_timeout,
        scoped_coordination_budget(),
    ):
        result = agent.run_conversation(preface, task_id=task_id)

    assert agent.client.chat.completions.create.call_count == 3
    title_call.assert_not_called()
    record_timeout.assert_not_called()
    assert result["api_calls"] == 3
    assert result["final_response"] == ""
    assert result["completed"] is True
    assert result["turn_exit_reason"] == "terminal_review_verdict"
    assert persisted["messages"][-1]["role"] == "assistant"
    assert (
        persisted["messages"][-1]["content"]
        == "Kanban terminal verdict recorded by host."
    )
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"
        request = kb.get_coordination_request(conn, request_id)
        assert request is not None
        assert request.status == "completed"
        assert request.model_calls_used == 20
        with pytest.raises(kb.CoordinationBudgetExceeded):
            kb.charge_coordination_model_call(
                conn,
                request_id,
                purpose="terminal_review",
                task_id=task_id,
            )


def test_terminal_review_request_review_is_not_a_clean_stop(terminal_review_context):
    """A normal worker handoff is not a source-review verdict."""
    _home, request_id, task_id, _run_id, _preface, _artifact, _source = (
        terminal_review_context
    )
    messages = [
        {
            "role": "tool",
            "name": "kanban_request_review",
            "content": json.dumps({"ok": True, "task_id": task_id, "status": "review"}),
        }
    ]

    with scoped_coordination_budget(
        request_root_id=request_id,
        task_id=task_id,
        purpose="terminal_review",
        db_path=kb.kanban_db_path(),
    ):
        assert _terminal_review_tool_round_completed(messages) is False


@pytest.mark.parametrize(
    ("purpose", "env", "messages"),
    [
        (
            None,
            {},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "work",
            {},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_KANBAN_TASK": None},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_KANBAN_TASK": "wrong-task"},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_SESSION_SOURCE": None},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_SESSION_SOURCE": "cli"},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_KANBAN_RUN_ID": None},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_KANBAN_RUN_ID": "0"},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {"HERMES_KANBAN_RUN_ID": "invalid"},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": True}}],
        ),
        (
            "terminal_review",
            {},
            [{"role": "tool", "name": "kanban_complete", "content": {"ok": False}}],
        ),
        (
            "terminal_review",
            {},
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "kanban_complete"}}],
                }
            ],
        ),
    ],
    ids=[
        "no-active-scope",
        "non-review-purpose",
        "missing-task",
        "wrong-task",
        "missing-source",
        "wrong-source",
        "missing-run",
        "zero-run",
        "invalid-run",
        "rejected-result",
        "invocation-only",
    ],
)
def test_terminal_review_guard_rejects_untrusted_context(
    terminal_review_context,
    monkeypatch,
    purpose,
    env,
    messages,
):
    """Environment strings never substitute for the active dispatcher scope."""
    _home, request_id, task_id, _run_id, _preface, _artifact, _source = (
        terminal_review_context
    )
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    if purpose is None:
        assert _terminal_review_tool_round_completed(messages) is False
        return
    with scoped_coordination_budget(
        request_root_id=request_id,
        task_id=task_id,
        purpose=purpose,
        db_path=kb.kanban_db_path(),
    ):
        assert _terminal_review_tool_round_completed(messages) is False


def test_terminal_review_request_changes_stops_after_one_final_call(
    terminal_review_context,
):
    """A real rework verdict needs no narrative follow-up at call 20."""
    _home, request_id, task_id, _run_id, preface, _artifact, _source = (
        terminal_review_context
    )
    with kb.connect_closing() as conn:
        for ordinal in (18, 19):
            assert (
                kb.charge_coordination_model_call(
                    conn,
                    request_id,
                    purpose="terminal_review",
                    task_id=task_id,
                )
                == ordinal
            )

    agent = _new_agent("kanban_request_changes")
    agent.max_iterations = 1
    agent.client.chat.completions.create.return_value = _tool_response(
        _tool_call(
            "kanban_request_changes",
            {"reason": "The verified source needs an additional assertion."},
            "call-20-changes",
        )
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.turn_finalizer._record_kanban_budget_exhausted") as record_timeout,
        scoped_coordination_budget(),
    ):
        result = agent.run_conversation(preface, task_id=task_id)

    assert agent.client.chat.completions.create.call_count == 1
    record_timeout.assert_not_called()
    assert result["completed"] is True
    assert result["final_response"] == ""
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"
        request = kb.get_coordination_request(conn, request_id)
        assert request is not None
        assert request.status == "active"
        assert request.model_calls_used == 20
