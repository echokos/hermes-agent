from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent import coordination_budget as budget
from agent import relay_llm, relay_runtime
from hermes_cli import kanban_db as kb


@pytest.fixture
def budget_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    for key in ("REQUEST_ROOT", "TASK_ID", "PURPOSE"):
        monkeypatch.delenv(f"HERMES_COORDINATION_{key}", raising=False)
    monkeypatch.setattr(relay_runtime, "resolve_execution_context", lambda _: (None, None, None))
    monkeypatch.setattr(relay_runtime, "active_turn", lambda: None)
    org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))
    with kb.connect_closing(db) as conn:
        root = kb.create_task(conn, title="Return result", assignee="coordinator", session_id="s")
        kb.add_notify_sub(conn, task_id=root, platform="buzz", chat_id="origin", delivery_mode="wake")
        accepted = kb.create_coordination_request(
            conn, root_task_id=root, origin_session_id="s", origin_message_id="m",
            organization=org, max_model_calls=6, final_model_call_reserve=2,
        )
    return SimpleNamespace(db=db, task=root, root=accepted.id)


def scope(budget_request, **kwargs):
    return budget.scoped_coordination_budget(
        request_root_id=budget_request.root, task_id=budget_request.task, db_path=budget_request.db, **kwargs
    )


def used(budget_request):
    with kb.connect_closing(budget_request.db) as conn:
        return kb.get_coordination_request(conn, budget_request.root).model_calls_used


def test_sync_async_and_stream_admit_before_provider(budget_request):
    seen = []

    def provider(_):
        seen.append(used(budget_request))
        return "ok"

    async def async_provider(body):
        return provider(body)

    with scope(budget_request):
        assert relay_llm.execute_current({}, provider, name="openai", model_name="test") == "ok"
        assert asyncio.run(relay_llm.execute_current_async(
            {}, async_provider, name="openai", model_name="test"
        )) == "ok"
        assert list(relay_llm.stream(
            {}, lambda body: iter([provider(body)]), session_id="s", name="openai",
            model_name="test", finalizer=dict,
        )) == ["ok"]
        assert list(relay_llm.stream_current(
            {}, lambda body: iter([provider(body)]), name="openai", model_name="test",
            finalizer=dict,
        )) == ["ok"]
        with pytest.raises(kb.CoordinationBudgetExceeded):
            relay_llm.execute_current({}, provider, name="openai", model_name="test")
    assert seen == [1, 2, 3, 4]
    assert used(budget_request) == 4


def test_failed_attempt_and_fallback_each_charge(budget_request):
    def fail(_):
        raise ConnectionError("provider unavailable")

    with scope(budget_request):
        with pytest.raises(ConnectionError):
            relay_llm.execute({}, fail, session_id="s", name="openai", model_name="test")
        relay_llm.execute({}, lambda _: "fallback", session_id="s", name="other", model_name="test")
    assert used(budget_request) == 2


def test_outer_codex_wrapper_is_not_a_second_physical_call(budget_request):
    def inner(_):
        return list(relay_llm.stream(
            {}, lambda _: iter(["event"]), session_id="s", name="codex",
            model_name="test", finalizer=dict,
        ))

    with scope(budget_request):
        assert relay_llm.execute(
            {}, inner, session_id="s", name="codex", model_name="test",
            metadata={"physical_attempt": False},
        ) == ["event"]
    assert used(budget_request) == 1


def test_origin_discovers_accepted_root(budget_request, monkeypatch):
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": {
        "HERMES_SESSION_ID": "s", "HERMES_SESSION_MESSAGE_ID": "m"
    }.get(key, default))
    with budget.scoped_coordination_budget():
        assert budget.charge_provider_attempt() == 1
        with budget.scoped_coordination_budget(session_id="delegated-child"):
            assert budget.charge_provider_attempt() == 2
    assert budget.charge_provider_attempt() is None


def test_root_accepted_later_in_real_tool_thread_and_rotation_keeps_origin(budget_request, monkeypatch):
    from gateway import session_context
    from tools.thread_context import propagate_context_to_thread

    values = {"HERMES_SESSION_ID": "s", "HERMES_SESSION_MESSAGE_ID": "later"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))

    def accept():
        session, message = budget.current_coordination_origin()
        org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))
        with kb.connect_closing(budget_request.db) as conn:
            task = kb.create_task(conn, title="Later accepted root", assignee="coordinator", session_id=session)
            kb.add_notify_sub(conn, task_id=task, platform="buzz", chat_id="origin", delivery_mode="wake")
            return kb.create_coordination_request(
                conn, root_task_id=task, origin_session_id=session,
                origin_message_id=message, organization=org,
            )

    with budget.scoped_coordination_budget():
        assert budget.charge_provider_attempt() is None
        values["HERMES_SESSION_ID"] = "rotated-session"
        with ThreadPoolExecutor(max_workers=1) as executor:
            accepted = executor.submit(propagate_context_to_thread(accept)).result()
        assert budget.current_coordination_origin() == ("s", "later")
        assert budget.charge_provider_attempt() == 1
    with kb.connect_closing(budget_request.db) as conn:
        assert kb.get_coordination_request(conn, accepted.id).model_calls_used == 1
    assert used(budget_request) == 0


def test_api_origin_uses_stable_request_chat_id(budget_request, monkeypatch):
    from gateway import session_context

    values = {
        "HERMES_SESSION_PLATFORM": "api_server", "HERMES_SESSION_CHAT_ID": "s",
        "HERMES_SESSION_ID": "internal-session", "HERMES_SESSION_MESSAGE_ID": "m",
    }
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))
    with budget.scoped_coordination_budget():
        assert budget.current_coordination_origin() == ("s", "m")
        assert budget.charge_provider_attempt() == 1


def test_copied_threads_share_budget_and_closed_turn_is_denied(budget_request):
    with scope(budget_request):
        contexts = [contextvars.copy_context() for _ in range(8)]

        def attempt(ctx):
            try:
                return ctx.run(budget.charge_provider_attempt)
            except kb.CoordinationBudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            outcomes = list(executor.map(attempt, contexts))
        retained = contextvars.copy_context()
    assert sorted(x for x in outcomes if x is not None) == [1, 2, 3, 4]
    with pytest.raises(kb.CoordinationBudgetExceeded, match="owning turn ended"):
        retained.run(budget.charge_provider_attempt)
    assert used(budget_request) == 4


def test_headless_env_applies_before_turn_and_invalid_envelope_fails(budget_request, monkeypatch):
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", budget_request.root)
    with pytest.raises(ValueError, match="incomplete"):
        budget.charge_provider_attempt()
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", budget_request.task)
    assert budget.charge_provider_attempt() == 1


def test_explicit_scope_does_not_inherit_other_worker_env(budget_request, monkeypatch):
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", "invalid-other-root")
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", "invalid-other-task")
    with scope(budget_request):
        assert budget.charge_provider_attempt() == 1
    assert used(budget_request) == 1
