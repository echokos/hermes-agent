"""Canonical coordination polling and receipt-backed final-return ownership."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_coordination_requests import ORGANIZATION


class Runner(GatewayKanbanWatchersMixin):
    def __init__(self, adapter=None):
        self._running = True
        self.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
        self._profile_adapters = {}
        self._kanban_coordination_jobs = {}

    def _active_profile_name(self):
        return "aurora"

    def _owns_kanban_dispatcher_lock(self):
        return False

    def _authorization_adapter(self, platform, profile):
        assert profile == "aurora"
        return self.adapters.get(platform)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "organization").mkdir(parents=True)
    (home / "organization" / "organization.yaml").write_text(ORGANIZATION)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    kb.init_db()
    return home / "kanban.db"


def ready_request(board):
    with kb.connect_closing(board) as conn:
        root = kb.create_task(
            conn, title="Final result", assignee="aurora", session_id="origin-session",
        )
        kb.add_notify_sub(
            conn, task_id=root, platform="telegram", chat_id="origin-chat",
            notifier_profile="aurora", delivery_mode="notify+wake", chat_type="dm",
        )
        request = kb.create_coordination_request(
            conn, root_task_id=root, origin_session_id="origin-session",
            origin_message_id="origin-message",
        )
        child = kb.create_task(
            conn, title="Verified repair", assignee="builder",
            coordination_source_task_id=root,
        )
        assert kb.complete_task(conn, child, summary="verified")
        return root, request


async def finish_tick(runner):
    await runner._kanban_coordination_tick()
    if runner._kanban_coordination_jobs:
        await asyncio.gather(*runner._kanban_coordination_jobs.values())


def test_final_receipt_advances_only_after_terminal_root(board, monkeypatch):
    root, request = ready_request(board)
    adapter = SimpleNamespace(send=AsyncMock())
    runner = Runner(adapter)

    async def wake(_adapter, **kwargs):
        assert _adapter is adapter
        assert kwargs["session_id"] == "origin-session"
        assert kwargs["source"].chat_id == "origin-chat"
        assert kwargs["source"].chat_type == "dm"
        envelope = kwargs["coordination_context"]
        assert set(envelope) == {
            "request_root_id", "task_id", "event_id", "responsible_agent", "db_path",
        }
        assert all(isinstance(value, str) for value in envelope.values())
        with kb.connect_closing(board) as conn:
            assert kb.get_coordination_request(conn, request.id).status == "return_pending"
            assert kb.list_notify_subs(conn, root)[0]["last_event_id"] < int(envelope["event_id"])
            assert kb.complete_task(conn, root, summary="verified final")
        return SimpleNamespace(state="acknowledged", returned_message_id="platform:123")

    send_wake = AsyncMock(side_effect=wake)
    monkeypatch.setattr("gateway.wake.deliver_wake", send_wake)
    asyncio.run(finish_tick(runner))
    with kb.connect_closing(board) as conn:
        assert kb.get_coordination_request(conn, request.id).status == "completed"
        assert kb.list_notify_subs(conn, root)[0]["last_event_id"] == int(
            send_wake.call_args.kwargs["coordination_context"]["event_id"],
        )
    asyncio.run(finish_tick(Runner(adapter)))
    assert send_wake.await_count == 1
    adapter.send.assert_not_awaited()


@pytest.mark.parametrize("state,receipt", [("pending", ""), ("uncertain", ""), ("acknowledged", "")])
def test_unconfirmed_return_retains_request_and_cursor(board, monkeypatch, state, receipt):
    root, request = ready_request(board)
    with kb.connect_closing(board) as conn:
        original_cursor = kb.list_notify_subs(conn, root)[0]["last_event_id"]
    adapter = SimpleNamespace(send=AsyncMock())
    monkeypatch.setattr(
        "gateway.wake.deliver_wake",
        AsyncMock(return_value=SimpleNamespace(state=state, returned_message_id=receipt)),
    )
    asyncio.run(finish_tick(Runner(adapter)))
    with kb.connect_closing(board) as conn:
        assert kb.get_coordination_request(conn, request.id).status == "return_pending"
        assert kb.list_notify_subs(conn, root)[0]["last_event_id"] == original_cursor
    adapter.send.assert_not_awaited()


def test_readiness_runs_without_any_adapter(board):
    _, request = ready_request(board)
    asyncio.run(finish_tick(Runner()))
    with kb.connect_closing(board) as conn:
        assert kb.get_coordination_request(conn, request.id).status == "return_pending"


@pytest.mark.parametrize("paused", [False, True])
def test_real_owned_failure_is_claimed_without_chat_subscription(board, monkeypatch, paused):
    from hermes_cli.workforce_handoffs import create_handoff

    now = datetime.now(timezone.utc)
    with kb.connect_closing(board) as conn:
        created = create_handoff(
            conn, source_agent="director", target_agent="builder",
            expected_outcome="Repair the owned failure", acceptance_test="Later probe succeeds",
            evidence_references=["execution:failure-one"],
            acknowledgment_deadline=(now + timedelta(minutes=2)).isoformat(),
            checkpoint_at=(now + timedelta(minutes=20)).isoformat(),
            requires_source_acceptance=True,
            context={
                "kind": "owned_operational_failure", "technical_owner": "builder",
                "director": "director", "workflow_id": "test-owned-failure",
                "event_id": "failure-one",
            },
        )
        assert kb.list_notify_subs(conn) == []
    runner = Runner()
    runner._active_profile_name = lambda: "builder"
    runner._kanban_pickup_owned_failure = AsyncMock()
    monkeypatch.setattr("gateway.kanban_watchers._kanban_dispatch_allowed", lambda: not paused)
    asyncio.run(finish_tick(runner))
    if paused:
        runner._kanban_pickup_owned_failure.assert_not_awaited()
    else:
        runner._kanban_pickup_owned_failure.assert_awaited_once()
        pickup = runner._kanban_pickup_owned_failure.call_args.args[0]
        assert pickup["task_id"] == created["task_id"]
        with kb.connect_closing(board) as conn:
            request = kb.get_coordination_request(conn, pickup["request_root_id"])
            assert request.kind == "owned_operational_failure"
            assert request.responsible_agent == "builder"
            assert kb.list_notify_subs(conn) == []
        # A restarted gateway cannot claim the same one-shot pickup again.
        restarted = Runner()
        restarted._active_profile_name = lambda: "builder"
        restarted._kanban_pickup_owned_failure = AsyncMock()
        asyncio.run(finish_tick(restarted))
        restarted._kanban_pickup_owned_failure.assert_not_awaited()


def test_pickup_without_adapter_or_sub_does_not_block_next_tick(board, monkeypatch):
    runner = Runner()
    monkeypatch.setattr(kb, "has_coordination_tick_work", lambda *a, **k: True)
    monkeypatch.setattr(kb, "prepare_coordination_final_return_deliveries", lambda *a, **k: [])
    claims = []

    def claim(conn, *, target_agent):
        claims.append(target_agent)
        return dict(task_id="t_one", request_root_id="cr_one", target_agent=target_agent, source_agent="builder")

    monkeypatch.setattr("hermes_cli.workforce_handoffs.claim_owned_failure_handoff_pickup", claim)

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def pickup(data):
            assert data["database_path"] == board
            started.set()
            await release.wait()

        runner._kanban_pickup_owned_failure = pickup
        await runner._kanban_coordination_tick()
        await started.wait()
        await runner._kanban_coordination_tick()
        assert claims == ["aurora"]
        release.set()
        await asyncio.gather(*runner._kanban_coordination_jobs.values())

    asyncio.run(scenario())


def test_watcher_shutdown_cancels_and_reaps_background_jobs(monkeypatch):
    runner = Runner()
    reaped = []

    async def scenario():
        started = asyncio.Event()

        async def job():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                reaped.append(True)

        async def loop(interval):
            runner._kanban_coordination_jobs["aurora"] = asyncio.create_task(job())
            await started.wait()

        runner._kanban_notifier_loop = loop
        await runner._kanban_notifier_watcher()

    asyncio.run(scenario())
    assert reaped == [True]
    assert runner._kanban_coordination_jobs == {}


def test_legacy_notifier_never_claims_or_passively_sends_origin_root(board, monkeypatch):
    root, request = ready_request(board)
    with kb.connect_closing(board) as conn:
        kb.begin_coordination_final_return_if_ready(conn, request.id)
        kb.complete_task(conn, root, summary="root now terminal")
        cursor = kb.list_notify_subs(conn, root)[0]["last_event_id"]
    adapter = SimpleNamespace(send=AsyncMock())
    runner = Runner(adapter)
    runner._kanban_coordination_tick = AsyncMock()
    real_sleep = asyncio.sleep

    async def sleep(delay):
        if delay != 5:
            runner._running = False
            await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))
    adapter.send.assert_not_awaited()
    with kb.connect_closing(board) as conn:
        assert kb.list_notify_subs(conn, root)[0]["last_event_id"] == cursor
