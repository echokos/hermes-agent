"""End-to-end cron Photon delivery idempotency regressions."""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _config_and_job():
    from gateway.config import Platform

    photon = Platform("photon")
    pconfig = SimpleNamespace(enabled=True, extra={})
    config = MagicMock()
    config.platforms = {photon: pconfig}
    job = {
        "id": "photon-cron",
        "execution_id": "execution-photon-1",
        "deliver": "origin",
        "origin": {"platform": "photon", "chat_id": "+15555550123"},
    }
    return photon, config, job


def _run_on_gateway_loop(coro, _loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except BaseException as exc:  # the scheduler must classify real async failures
        future.set_exception(exc)
    return future


def test_accepted_but_http_500_is_unknown_and_never_falls_back_or_replays(monkeypatch, tmp_path):
    import cron.executions as executions
    from cron.scheduler import _deliver_result

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    photon, config, job = _config_and_job()
    loop = MagicMock()
    loop.is_running.return_value = True
    sink_effects = []

    async def accepted_then_500(*_args, **_kwargs):
        sink_effects.append("primary")
        return SimpleNamespace(
            success=False,
            error="sidecar HTTP 500 after provider accepted",
            raw_response={"http_status": 500},
        )

    adapter = MagicMock()
    adapter.send = accepted_then_500

    async def standalone_send(*_args, **_kwargs):
        sink_effects.append("fallback")
        return {"success": True}

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=_run_on_gateway_loop), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(side_effect=standalone_send)) as standalone:
        first = _deliver_result(job, "one final response", adapters={photon: adapter}, loop=loop)
        second = _deliver_result(job, "one final response", adapters={photon: adapter}, loop=loop)

    assert first is not None and "unknown" in first
    assert second is not None and "unknown" in second
    assert sink_effects == ["primary"]
    standalone.assert_not_awaited()


def test_confirmed_absence_uses_one_fallback_then_blocks_process_retry(monkeypatch, tmp_path):
    import cron.executions as executions
    from cron.scheduler import _deliver_result

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    photon, config, job = _config_and_job()
    loop = MagicMock()
    loop.is_running.return_value = True
    sink_effects = []

    async def rejected_before_acceptance(*_args, **_kwargs):
        return SimpleNamespace(
            success=False,
            error="sidecar rejected invalid request",
            raw_response={"http_status": 400},
        )

    adapter = MagicMock()
    adapter.send = rejected_before_acceptance

    async def standalone_send(*_args, **_kwargs):
        sink_effects.append("fallback")
        return {"success": True}

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=_run_on_gateway_loop), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(side_effect=standalone_send)) as standalone:
        assert _deliver_result(job, "one final response", adapters={photon: adapter}, loop=loop) is None
        replay = _deliver_result(job, "one final response", adapters={photon: adapter}, loop=loop)

    assert replay is not None and "already" in replay
    assert sink_effects == ["fallback"]
    assert standalone.await_count == 1


def test_run_body_passes_each_persisted_execution_id_to_photon_delivery(monkeypatch, tmp_path):
    """Same-content executions must not share a Photon delivery identity."""
    import cron.executions as executions
    import cron.scheduler as scheduler

    delivery_contexts = []
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_args: True)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (True, "# output", "unchanged alert", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    def reserve_delivery(job, content, **_kwargs):
        identity, target = scheduler._photon_delivery_identity(
            job, "+155****0123", None, content
        )
        claimed = executions.claim_delivery(
            identity,
            execution_id=job["execution_id"],
            job_id=job["id"],
            platform="photon",
            target_fingerprint=target,
        )
        if claimed["claimed"]:
            executions.resolve_delivery(identity, "confirmed_sent")
        delivery_contexts.append((dict(job), identity, target))
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", reserve_delivery)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        for _ in range(2):
            assert scheduler._run_one_job_body(
                {
                    "id": "photon-cron",
                    "deliver": "origin",
                    "origin": {"platform": "photon", "chat_id": "+155****0123"},
                },
                admitted_run=SimpleNamespace(_context=lambda: None),
            ) is True

    contexts, identities, targets = map(list, zip(*delivery_contexts))
    execution_ids = [context["execution_id"] for context in contexts]
    assert len(set(execution_ids)) == 2
    assert identities[0] != identities[1]
    assert executions.claim_delivery(
        identities[0],
        execution_id=execution_ids[0],
        job_id="photon-cron",
        platform="photon",
        target_fingerprint=targets[0],
    ) == {"claimed": False, "state": "confirmed_sent"}


def _delivery_state(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT state FROM deliveries").fetchone()[0]


def test_confirmed_live_success_commits_terminal_ledger_state(monkeypatch, tmp_path):
    """A confirmed live Photon send must not leave its reservation pending."""
    import cron.executions as executions
    from cron.scheduler import _deliver_result

    db_path = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db_path)
    photon, config, job = _config_and_job()
    loop = MagicMock()
    loop.is_running.return_value = True
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, raw_response={"message_id": "masked"})
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=_run_on_gateway_loop):
        assert _deliver_result(job, "one final response", adapters={photon: adapter}, loop=loop) is None

    assert _delivery_state(db_path) == "confirmed_sent"


def test_pre_dispatch_failure_is_absent_then_uses_one_fallback(monkeypatch, tmp_path):
    """A scheduler rejection happens before dispatch and is safe to retry once."""
    import cron.executions as executions
    from cron.scheduler import _deliver_result

    db_path = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db_path)
    photon, config, job = _config_and_job()
    loop = MagicMock()
    loop.is_running.return_value = True
    fallback = AsyncMock(return_value={"success": True})

    def reject_before_dispatch(coro, _loop):
        coro.close()
        return None

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("agent.async_utils.safe_schedule_threadsafe", side_effect=reject_before_dispatch), \
         patch("tools.send_message_tool._send_to_platform", new=fallback):
        assert _deliver_result(job, "one final response", adapters={photon: MagicMock()}, loop=loop) is None

    fallback.assert_awaited_once()
    assert _delivery_state(db_path) == "confirmed_sent"
