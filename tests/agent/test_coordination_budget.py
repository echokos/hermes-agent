from __future__ import annotations

import asyncio
import contextvars
import threading
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


def test_current_execution_reads_only_live_bound_context(budget_request, monkeypatch):
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", "env-root")
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", "env-task")
    monkeypatch.setenv("HERMES_COORDINATION_PURPOSE", "terminal_review")

    # Process environment alone is not proof of an active validated turn.
    assert budget.current_coordination_execution() is None

    with scope(budget_request, purpose="terminal_review"):
        assert budget.current_coordination_execution() == (
            budget_request.root,
            budget_request.task,
            "terminal_review",
        )

    # Closed scopes never remain observable to post-turn code.
    assert budget.current_coordination_execution() is None


def test_current_execution_rejects_unbound_capture_scope(budget_request, monkeypatch):
    for key in ("REQUEST_ROOT", "TASK_ID", "PURPOSE"):
        monkeypatch.delenv(f"HERMES_COORDINATION_{key}", raising=False)

    with budget.scoped_coordination_budget(session_id="ordinary-session"):
        assert budget.current_coordination_execution() is None


def test_declared_acceptance_blocks_materialization_until_success_or_new_turn(
    budget_request, monkeypatch,
):
    from gateway import session_context

    values = {
        "HERMES_SESSION_ID": "ordinary-session",
        "HERMES_SESSION_MESSAGE_ID": "ordinary-message",
    }
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": values.get(key, default),
    )

    with budget.scoped_coordination_budget():
        with budget.coordination_materialization_binding() as binding:
            assert binding == (None, ("ordinary-session", "ordinary-message"))

        budget.register_declared_coordination_acceptance(declared=True)
        with pytest.raises(
            ValueError, match="requires successful coordination acceptance"
        ):
            with budget.coordination_materialization_binding():
                pass

        # Failure is sticky for this user turn and cannot reopen an unbudgeted
        # materialization lane after the executor returns.
        with pytest.raises(
            ValueError, match="requires successful coordination acceptance"
        ):
            with budget.coordination_materialization_binding():
                pass

    # A fresh user turn has no declared coordination requirement.
    with budget.scoped_coordination_budget():
        with budget.coordination_materialization_binding() as binding:
            assert binding == (None, ("ordinary-session", "ordinary-message"))


def test_uncoordinated_materialization_blocks_later_same_turn_acceptance(
    budget_request, monkeypatch,
):
    from gateway import session_context

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": {
            "HERMES_SESSION_ID": "materialized-session",
            "HERMES_SESSION_MESSAGE_ID": "materialized-message",
        }.get(key, default),
    )

    with budget.scoped_coordination_budget():
        with budget.coordination_materialization_binding():
            budget.register_uncoordinated_materialization(
                created=True,
                request_root_id=None,
            )
        with pytest.raises(
            ValueError, match="after uncoordinated workforce materialization"
        ):
            with budget.coordination_acceptance_binding():
                pass


@pytest.mark.parametrize(
    ("created", "request_root_id"),
    [(False, None), (False, "cr_existing"), (True, "cr_existing")],
)
def test_retry_or_coordinated_materialization_does_not_block_acceptance(
    budget_request, monkeypatch, created, request_root_id,
):
    from gateway import session_context

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": {
            "HERMES_SESSION_ID": "nonblocking-session",
            "HERMES_SESSION_MESSAGE_ID": "nonblocking-message",
        }.get(key, default),
    )

    with budget.scoped_coordination_budget():
        budget.register_uncoordinated_materialization(
            created=created,
            request_root_id=request_root_id,
        )
        with budget.coordination_acceptance_binding() as binding:
            assert binding.existing_request_root_id == ""


@pytest.mark.parametrize("report_to_origin", [True, "true", "yes", 1])
def test_acceptance_declaration_requires_a_coordination_object(report_to_origin):
    assert budget.declares_coordination_acceptance(
        "kanban_create",
        {"coordination": {}, "report_to_origin": report_to_origin},
    )
    assert not budget.declares_coordination_acceptance(
        "kanban_create",
        {"report_to_origin": report_to_origin},
    )
    assert not budget.declares_coordination_acceptance(
        "kanban_create",
        {"coordination": [], "report_to_origin": report_to_origin},
    )


def test_root_accepted_later_in_real_tool_thread_and_rotation_keeps_origin(budget_request, monkeypatch):
    from gateway import session_context
    from tools.thread_context import propagate_context_to_thread

    values = {"HERMES_SESSION_ID": "s", "HERMES_SESSION_MESSAGE_ID": "later"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))

    def accept():
        session, message = budget.current_coordination_origin()
        org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))
        with (
            budget.coordination_acceptance_binding() as binding,
            kb.connect_closing(budget_request.db) as conn,
            kb.write_txn(conn),
        ):
            task = kb.create_task(conn, title="Later accepted root", assignee="coordinator", session_id=session)
            kb.add_notify_sub(conn, task_id=task, platform="buzz", chat_id="origin", delivery_mode="wake")
            accepted = kb.create_coordination_request(
                conn, root_task_id=task, origin_session_id=session,
                origin_message_id=message, organization=org,
                accepting_model_calls=binding.model_calls,
                acceptance_scope_id=binding.scope_id,
            )
            binding.accept(accepted)
        return accepted

    with budget.scoped_coordination_budget():
        assert budget.charge_provider_attempt() is None
        values["HERMES_SESSION_ID"] = "rotated-session"
        with ThreadPoolExecutor(max_workers=1) as executor:
            accepted = executor.submit(propagate_context_to_thread(accept)).result()
        assert budget.current_coordination_origin() == ("s", "later")
        assert budget.charge_provider_attempt() == 2
    with kb.connect_closing(budget_request.db) as conn:
        assert kb.get_coordination_request(conn, accepted.id).model_calls_used == 2
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


def test_acceptance_binding_rolls_back_without_losing_provisional_usage(budget_request, monkeypatch):
    from gateway import session_context

    values = {"HERMES_SESSION_ID": "rollback", "HERMES_SESSION_MESSAGE_ID": "m"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": values.get(key, default))
    org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))

    def accept(conn, binding):
        task = kb.create_task(conn, title="Accepted atomically", assignee="coordinator", session_id="rollback")
        kb.add_notify_sub(conn, task_id=task, platform="buzz", chat_id="origin", delivery_mode="wake")
        accepted = kb.create_coordination_request(
            conn, root_task_id=task, origin_session_id="rollback", origin_message_id="m",
            organization=org, accepting_model_calls=binding.model_calls,
            acceptance_scope_id=binding.scope_id,
        )
        binding.accept(accepted)
        return accepted

    with budget.scoped_coordination_budget(session_id="unaccepted"):
        assert budget.charge_provider_attempt() is None
        with pytest.raises(RuntimeError, match="transaction failed"):
            with budget.coordination_acceptance_binding() as binding, kb.connect_closing(budget_request.db) as conn, kb.write_txn(conn):
                assert binding.model_calls == 1
                accept(conn, binding)
                raise RuntimeError("transaction failed")
        assert budget.current_coordination_request_id() == ""
        with budget.coordination_acceptance_binding() as binding, kb.connect_closing(budget_request.db) as conn, kb.write_txn(conn):
            assert binding.model_calls == 1
            accepted = accept(conn, binding)
        assert budget.current_coordination_request_id() == accepted.id
        assert budget.charge_provider_attempt() == 2
        with budget.coordination_acceptance_binding() as retry:
            assert retry.model_calls == 1
            retry.accept(accepted)
        with kb.connect_closing(budget_request.db) as conn:
            assert kb.get_coordination_request(conn, accepted.id).model_calls_used == 2
        assert used(budget_request) == 0


def test_delegate_spawn_and_root_acceptance_are_mutually_exclusive(budget_request, monkeypatch):
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": default)
    barrier = threading.Barrier(2)
    accepted = SimpleNamespace(id=budget_request.root, root_task_id=budget_request.task)

    def accept():
        barrier.wait(timeout=5)
        try:
            with budget.coordination_acceptance_binding() as binding:
                binding.accept(accepted)
            return "accepted"
        except ValueError:
            return "denied"

    def delegate():
        barrier.wait(timeout=5)
        try:
            budget.admit_delegate_spawn()
            return "delegated"
        except ValueError:
            return "denied"

    with budget.scoped_coordination_budget(session_id="unaccepted"):
        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(context.run, fn) for context, fn in zip(contexts, (accept, delegate))]
            outcomes = [future.result(timeout=10) for future in futures]
    assert outcomes.count("denied") == 1
    assert outcomes.count("accepted") + outcomes.count("delegated") == 1


@pytest.mark.parametrize("settle_on_exit", [False, True])
def test_independent_same_origin_scopes_settle_each_initial_call(budget_request, monkeypatch, settle_on_exit):
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": {
        "HERMES_SESSION_ID": "independent", "HERMES_SESSION_MESSAGE_ID": "m",
    }.get(key, default))
    org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))
    first, second = contextvars.Context(), contextvars.Context()
    managers = [budget.scoped_coordination_budget(), budget.scoped_coordination_budget()]
    for ctx, manager in zip((first, second), managers):
        ctx.run(manager.__enter__)
        assert ctx.run(budget.charge_provider_attempt) is None

    def accept():
        with budget.coordination_acceptance_binding() as binding, kb.connect_closing(budget_request.db) as conn, kb.write_txn(conn):
            task = kb.create_task(conn, title="Shared request", assignee="coordinator", session_id="independent")
            kb.add_notify_sub(conn, task_id=task, platform="buzz", chat_id="origin", delivery_mode="wake")
            accepted = kb.create_coordination_request(
                conn, root_task_id=task, origin_session_id="independent", origin_message_id="m",
                organization=org, accepting_model_calls=binding.model_calls,
                acceptance_scope_id=binding.scope_id,
            )
            binding.accept(accepted)
        return accepted

    accepted = first.run(accept)
    try:
        if not settle_on_exit:
            assert second.run(budget.charge_provider_attempt) == 3
    finally:
        for ctx, manager in zip((first, second), managers):
            ctx.run(manager.__exit__, None, None, None)
    with kb.connect_closing(budget_request.db) as conn:
        assert kb.get_coordination_request(conn, accepted.id).model_calls_used == (2 if settle_on_exit else 3)


def test_acceptance_commit_uncertainty_does_not_double_debit(budget_request, monkeypatch):
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": {
        "HERMES_SESSION_ID": "uncertain", "HERMES_SESSION_MESSAGE_ID": "m",
    }.get(key, default))
    org = SimpleNamespace(validate_execution_profile=lambda _: SimpleNamespace(agent="coordinator"))
    with budget.scoped_coordination_budget():
        assert budget.charge_provider_attempt() is None
        with pytest.raises(RuntimeError, match="lost commit acknowledgement"):
            with budget.coordination_acceptance_binding() as binding:
                with kb.connect_closing(budget_request.db) as conn, kb.write_txn(conn):
                    task = kb.create_task(conn, title="Committed root", assignee="coordinator", session_id="uncertain")
                    kb.add_notify_sub(conn, task_id=task, platform="buzz", chat_id="origin", delivery_mode="wake")
                    accepted = kb.create_coordination_request(
                        conn, root_task_id=task, origin_session_id="uncertain", origin_message_id="m",
                        organization=org, accepting_model_calls=binding.model_calls,
                        acceptance_scope_id=binding.scope_id,
                    )
                    binding.accept(accepted)
                raise RuntimeError("lost commit acknowledgement")
        assert budget.current_coordination_request_id() == accepted.id
        assert budget.charge_provider_attempt() == 2
    with kb.connect_closing(budget_request.db) as conn:
        assert kb.get_coordination_request(conn, accepted.id).model_calls_used == 2


def test_closed_unaccepted_turn_cannot_make_late_provider_attempt(budget_request, monkeypatch):
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": default)
    with budget.scoped_coordination_budget(session_id="unaccepted"):
        retained = contextvars.copy_context()
    with pytest.raises(kb.CoordinationBudgetExceeded, match="owning turn ended"):
        retained.run(budget.charge_provider_attempt)


def test_detached_scope_keeps_root_limits_after_parent_closes(budget_request):
    def exhaust_work_budget():
        assert [budget.charge_provider_attempt() for _ in range(4)] == [1, 2, 3, 4]
        budget.charge_provider_attempt()

    with scope(budget_request):
        detached = budget.bind_detached_coordination_scope(exhaust_work_budget)

    with pytest.raises(kb.CoordinationBudgetExceeded, match="reserve preserved"):
        detached()
    assert used(budget_request) == 4


def test_detached_scope_observes_request_cancellation(budget_request):
    with scope(budget_request):
        detached = budget.bind_detached_coordination_scope(
            budget.charge_provider_attempt
        )

    with kb.connect_closing(budget_request.db) as conn:
        conn.execute(
            "UPDATE coordination_requests SET status='cancelled' WHERE id=?",
            (budget_request.root,),
        )
        conn.commit()

    with pytest.raises(kb.CoordinationBudgetExceeded, match="request is cancelled"):
        detached()
    assert used(budget_request) == 0


def test_background_review_gets_detached_budget_lifetime(budget_request, monkeypatch):
    from agent import background_review

    observed = {}

    def review_attempt(*_args):
        observed["execution"] = budget.current_coordination_execution()
        observed["charged"] = budget.charge_provider_attempt()

    monkeypatch.setattr(background_review, "_run_review_in_thread", review_attempt)
    with scope(budget_request):
        target, _prompt = background_review.spawn_background_review_thread(
            SimpleNamespace(),
            messages_snapshot=[],
            review_memory=True,
            task_cfg={},
        )

    target()
    assert observed == {
        "execution": (budget_request.root, budget_request.task, "work"),
        "charged": 1,
    }
    assert used(budget_request) == 1


@pytest.mark.parametrize("remaining", [1, 2])
def test_auxiliary_stream_fallback_charges_second_physical_attempt(budget_request, monkeypatch, remaining):
    from agent import auxiliary_client as aux

    attempts = []

    def create(**kwargs):
        attempts.append(used(budget_request))
        if kwargs.get("stream"):
            raise ValueError("stream not supported")
        return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: True)
    monkeypatch.setattr(aux, "_client_streams_internally", lambda _: False)
    with scope(budget_request):
        for _ in range(4 - remaining):
            budget.charge_provider_attempt()

        @aux._relay_auxiliary_call
        def run(task):
            return aux._relay_sync_completion(
                client, {"model": "test"}, provider="openai", api_mode="chat_completions",
                create=lambda kwargs: aux._create_with_progress(client, kwargs),
            )

        if remaining == 1:
            with pytest.raises(kb.CoordinationBudgetExceeded):
                run("compression")
        else:
            assert run("compression") == "ok"
    assert attempts == ([4] if remaining == 1 else [3, 4])
    assert used(budget_request) == 4


def test_actual_codex_moa_aggregator_stream_is_budgeted(budget_request, monkeypatch):
    from agent import auxiliary_client as aux

    attempts = []
    completed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="result"))])

    def create(**kwargs):
        attempts.append(used(budget_request))
        return completed

    client = object.__new__(aux.CodexAuxiliaryClient)
    client.base_url = "https://example.invalid/v1"
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *args, **kwargs: (client, "test"))
    with scope(budget_request):
        for _ in range(4):
            assert aux.call_llm(
                "moa_aggregator", provider="openai", model="test", api_key="test",
                messages=[{"role": "user", "content": "test"}], stream=True,
            ) is completed
        with pytest.raises(kb.CoordinationBudgetExceeded):
            aux.call_llm(
                "moa_aggregator", provider="openai", model="test", api_key="test",
                messages=[{"role": "user", "content": "test"}], stream=True,
            )
    assert attempts == [1, 2, 3, 4]
