"""Work-scope review handoffs stop after the host commits the transition."""

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
from agent.conversation_loop import _work_review_tool_round_completed
from agent.tool_dispatch_helpers import _plan_tool_batch_segments
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
    direct_reports: [director]
    mission: Own the system
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    buzz_rooms: []
  - agent: director
    display_name: Director
    status: active
    operational: true
    function: Director
    manager: elliott
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
    function: Developer
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


def _tool_response(*tool_calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=list(tool_calls)),
                finish_reason="tool_calls",
            )
        ],
        model="test/model",
        usage=None,
    )


def _new_agent(*tool_names: str) -> AIAgent:
    tool_names = tool_names or ("kanban_request_review",)
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
    agent._cached_system_prompt = "You are an implementation worker."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.session_id = "work-review-session"
    return agent


@pytest.fixture
def work_review_context(tmp_path, monkeypatch):
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
            expected_outcome="Repair the bounded owned failure",
            acceptance_test="Attach repair evidence and request review",
            evidence_references=["workflow:test"],
            acknowledgment_deadline=iso(now + 60),
            checkpoint_at=iso(now + 240),
            organization=organization,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "builder",
                "director": "director",
                "workflow_id": "bounded-repair",
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
        worker = kb.claim_task(conn, task_id, claimer="builder:test")
        assert worker is not None and worker.current_run_id is not None
        for ordinal in range(1, 17):
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

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.current_run_id))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", request.id)
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", task_id)
    monkeypatch.setenv("HERMES_COORDINATION_PURPOSE", "work")
    return request.id, task_id, worker.current_run_id


def _review_result(task_id: str, run_id: int, *, ok: bool = True) -> dict:
    return {
        "role": "tool",
        "name": "kanban_request_review",
        "content": json.dumps(
            {
                "ok": ok,
                "task_id": task_id,
                "run_id": run_id,
                "status": "review",
            }
        ),
    }


def test_work_review_handoff_stops_at_call_seventeen(work_review_context):
    """A real host review transition ends the work turn without call 18."""
    request_id, task_id, run_id = work_review_context
    agent = _new_agent()
    agent.max_iterations = 1
    agent.client.chat.completions.create.return_value = _tool_response(
        _tool_call(
            "kanban_request_review",
            {"summary": "Repair evidence is attached and ready for review."},
            "call-17-review",
        )
    )
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
        result = agent.run_conversation("Repair the owned failure.", task_id=task_id)

    assert agent.client.chat.completions.create.call_count == 1
    title_call.assert_not_called()
    record_timeout.assert_not_called()
    assert result["api_calls"] == 1
    assert result["completed"] is True
    assert result["final_response"] == ""
    assert result["turn_exit_reason"] == "work_review_handoff"
    assert persisted["messages"][-1]["role"] == "assistant"
    assert persisted["messages"][-1]["content"] == "Kanban review handoff recorded by host."

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.current_run_id is None
        request = kb.get_coordination_request(conn, request_id)
        assert request is not None
        assert request.status == "active"
        assert request.model_calls_used == 17
        assert request.transient_retries_used == 0
        events = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]
    assert "review_requested" in events
    assert "crashed" not in events
    assert "completed" not in events
    assert str(run_id) == os.environ["HERMES_KANBAN_RUN_ID"]


def test_work_review_handoff_survives_pending_steer_at_call_seventeen(
    work_review_context,
):
    """A post-tool steer must not turn a verified handoff into call 18."""
    _, task_id, _ = work_review_context
    agent = _new_agent()
    agent._pending_steer = "Preserve this operator correction."
    response = _tool_response(
        _tool_call(
            "kanban_request_review",
            {"summary": "Repair evidence is attached and ready for review."},
            "call-17-review",
        )
    )

    result = _run_review_batch(agent, task_id=task_id, responses=(response,), scope=True)

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "work_review_handoff"
    assert result["final_response"] == ""
    tool_messages = [m for m in agent._session_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Preserve this operator correction." in str(tool_messages[0].get("content"))
    assert agent._session_messages[-1]["role"] == "assistant"
    assert agent._session_messages[-1]["content"] == "Kanban review handoff recorded by host."


def _run_review_batch(
    agent: AIAgent,
    *,
    task_id: str,
    responses: tuple[SimpleNamespace, ...],
    scope: bool,
) -> dict:
    agent.max_iterations = 1
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.title_generator.maybe_auto_title"),
        patch("agent.turn_finalizer._record_kanban_budget_exhausted"),
    ):
        if scope:
            with scoped_coordination_budget():
                return agent.run_conversation("Repair the owned failure.", task_id=task_id)
        return agent.run_conversation("Repair the owned failure.", task_id=task_id)


def test_work_review_handoff_skips_mutating_suffix_in_segmented_batch(
    work_review_context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A persisted host review result prevents later batch mutations."""
    _, task_id, implementation_run_id = work_review_context
    source = tmp_path / "source.txt"
    source.write_text("evidence")
    before = tmp_path / "before.txt"
    after_one = tmp_path / "after-one.txt"
    after_two = tmp_path / "after-two.txt"
    agent = _new_agent("read_file", "write_file", "kanban_request_review")
    response = _tool_response(
        _tool_call("read_file", {"path": str(source)}, "call-17-read"),
        _tool_call("write_file", {"path": str(before), "content": "before"}, "call-17-before"),
        _tool_call(
            "kanban_request_review",
            {"summary": "Evidence is attached and ready for review."},
            "call-17-review",
        ),
        _tool_call("write_file", {"path": str(after_one), "content": "must not land"}, "call-17-after-one"),
        _tool_call("write_file", {"path": str(after_two), "content": "must not land"}, "call-17-after-two"),
    )
    segments = _plan_tool_batch_segments(response.choices[0].message.tool_calls)
    assert [kind for kind, _ in segments] == ["parallel", "sequential", "parallel"]
    real_request_review = kb.request_review

    def request_review_then_claim(conn, handoff_task_id, **kwargs):
        outcome = real_request_review(conn, handoff_task_id, **kwargs)
        if outcome[0]:
            claimed = kb.claim_review_task(conn, handoff_task_id, claimer="director:race")
            assert claimed is not None
            assert claimed.current_run_id != implementation_run_id
        return outcome

    monkeypatch.setattr(kb, "request_review", request_review_then_claim)

    result = _run_review_batch(agent, task_id=task_id, responses=(response,), scope=True)

    assert before.read_text() == "before"
    assert not after_one.exists()
    assert not after_two.exists()
    assert result["turn_exit_reason"] == "work_review_handoff"
    tool_messages = [m for m in agent._session_messages if m.get("role") == "tool"]
    assert [m.get("tool_call_id") for m in tool_messages] == [
        "call-17-read", "call-17-before", "call-17-review", "call-17-after-one", "call-17-after-two",
    ]
    assert [m.get("name") for m in tool_messages] == [
        "read_file", "write_file", "kanban_request_review", "write_file", "write_file",
    ]
    assert [m.get("effect_disposition") for m in tool_messages[-2:]] == ["none", "none"]
    assert all("was not started" in str(m.get("content")) for m in tool_messages[-2:])
    assert json.loads(str(tool_messages[2]["content"])) == {
        "ok": True,
        "task_id": task_id,
        "run_id": implementation_run_id,
        "status": "review",
    }
    assert agent.client.chat.completions.create.call_count == 1


def test_work_review_handoff_skips_mutating_suffix_in_sequential_batch(
    work_review_context,
    tmp_path: Path,
):
    """The all-sequential planner path also leaves a paired skipped result."""
    _, task_id, _ = work_review_context
    source = tmp_path / "source.txt"
    source.write_text("evidence")
    after = tmp_path / "after.txt"
    agent = _new_agent("read_file", "write_file", "kanban_request_review")
    response = _tool_response(
        _tool_call("read_file", {"path": str(source)}, "call-17-read"),
        _tool_call(
            "kanban_request_review",
            {"summary": "Evidence is attached and ready for review."},
            "call-17-review",
        ),
        _tool_call("write_file", {"path": str(after), "content": "must not land"}, "call-17-after"),
    )
    segments = _plan_tool_batch_segments(response.choices[0].message.tool_calls)
    assert [kind for kind, _ in segments] == ["sequential"]

    result = _run_review_batch(agent, task_id=task_id, responses=(response,), scope=True)

    assert not after.exists()
    assert result["turn_exit_reason"] == "work_review_handoff"
    tool_messages = [m for m in agent._session_messages if m.get("role") == "tool"]
    assert [m.get("tool_call_id") for m in tool_messages] == [
        "call-17-read", "call-17-review", "call-17-after",
    ]
    assert tool_messages[-1].get("effect_disposition") == "none"


def test_rejected_review_does_not_skip_suffix(
    work_review_context,
    tmp_path: Path,
):
    """A host rejection cannot suppress later calls in the same batch."""
    _, task_id, _ = work_review_context
    after = tmp_path / "after.txt"
    agent = _new_agent("write_file", "kanban_request_review")
    response = _tool_response(
        _tool_call("kanban_request_review", {"summary": ""}, "call-review"),
        _tool_call("write_file", {"path": str(after), "content": "landed"}, "call-after"),
    )

    _run_review_batch(agent, task_id=task_id, responses=(response,), scope=True)

    assert after.read_text() == "landed"
    tool_messages = [m for m in agent._session_messages if m.get("role") == "tool"]
    assert [m.get("tool_call_id") for m in tool_messages] == ["call-review", "call-after"]
    assert tool_messages[-1].get("effect_disposition") != "none"


def test_ordinary_review_without_active_scope_does_not_skip_suffix(
    work_review_context,
    tmp_path: Path,
):
    """Environment-shaped ordinary calls cannot activate the executor stop."""
    _, task_id, _ = work_review_context
    after = tmp_path / "after.txt"
    agent = _new_agent("write_file", "kanban_request_review")
    assistant_message = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "kanban_request_review",
                {"summary": "Evidence is attached and ready for review."},
                "call-review",
            ),
            _tool_call(
                "write_file",
                {"path": str(after), "content": "landed"},
                "call-after",
            ),
        ]
    )
    messages: list[dict] = []

    with patch.object(agent, "_flush_messages_to_session_db", return_value=True):
        stopped = agent._execute_tool_calls(assistant_message, messages, task_id)

    assert stopped is False
    assert after.read_text() == "landed"
    assert [m.get("tool_call_id") for m in messages] == ["call-review", "call-after"]
    assert messages[-1].get("effect_disposition") != "none"


@pytest.mark.parametrize(
    ("scope_purpose", "environment", "result"),
    [
        (None, {}, "valid"),
        ("terminal_review", {}, "valid"),
        ("work", {"HERMES_COORDINATION_REQUEST_ROOT": "wrong-root"}, "valid"),
        ("work", {"HERMES_COORDINATION_TASK_ID": "wrong-task"}, "valid"),
        ("work", {"HERMES_COORDINATION_PURPOSE": "terminal_review"}, "valid"),
        ("work", {"HERMES_KANBAN_TASK": "wrong-task"}, "valid"),
        ("work", {"HERMES_SESSION_SOURCE": "cli"}, "valid"),
        ("work", {"HERMES_KANBAN_RUN_ID": "0"}, "valid"),
        ("work", {}, "wrong-task"),
        ("work", {}, "wrong-run"),
        ("work", {}, "rejected"),
        ("work", {}, "invocation-only"),
    ],
    ids=[
        "no-active-context",
        "terminal-review-context",
        "wrong-root-environment",
        "wrong-task-environment",
        "wrong-purpose-environment",
        "wrong-kanban-task",
        "wrong-source",
        "invalid-run",
        "wrong-result-task",
        "wrong-result-run",
        "rejected-result",
        "invocation-only",
    ],
)
def test_work_review_guard_rejects_untrusted_context(
    work_review_context,
    monkeypatch,
    scope_purpose,
    environment,
    result,
):
    request_id, task_id, run_id = work_review_context
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    if result == "wrong-task":
        messages = [_review_result("other-task", run_id)]
    elif result == "wrong-run":
        messages = [_review_result(task_id, run_id + 1)]
    elif result == "rejected":
        messages = [_review_result(task_id, run_id, ok=False)]
    elif result == "invocation-only":
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "kanban_request_review"}}],
            }
        ]
    else:
        messages = [_review_result(task_id, run_id)]

    if scope_purpose is None:
        assert _work_review_tool_round_completed(messages) is False
        return
    with scoped_coordination_budget(
        request_root_id=request_id,
        task_id=task_id,
        purpose=scope_purpose,
        db_path=kb.kanban_db_path(),
    ):
        assert _work_review_tool_round_completed(messages) is False
