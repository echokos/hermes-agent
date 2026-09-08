import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, patch

import pytest

from cron.operational_outcomes import capture_outcome_notice
from gateway import delivery_ledger as dl
from gateway import operational_outcomes as notices
from gateway.platforms.base import SendResult
from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_coordination_requests import kanban_home, organization
from tests.hermes_cli.test_owned_failure_decision_review import owner_run, send_to_review


@pytest.fixture
def accepted(kanban_home, organization, monkeypatch, request):
    home = kanban_home / "profiles" / "aurora"
    monkeypatch.setenv("HERMES_HOME", str(home))
    job = {"id": "daily-note", "deliver": "telegram:12345", "failure_ownership": {
        "technical_owner": "builder", "director": "aurora", "return_outcome_to_origin": True,
    }}
    notice = capture_outcome_notice(job, source_profile="aurora", execution_id="execution-one")
    (home / "cron").mkdir(exist_ok=True)
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))
    with kb.connect_closing() as conn:
        owner, _ = owner_run(conn, organization)
        body = json.loads(owner.body)
        body["context"].update({"source": {"kind": "profile_cron", "scope": "aurora", "id": job["id"]},
                                "outcome_notice": notice})
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(body), owner.id))
        if getattr(request, "param", None) == "recovery":
            with kb.write_txn(conn):
                kb._append_event(conn, owner.id, "workforce_handoff_recovery_verified", {
                    "failure_event_id": "failure-one", "failure_order": 10,
                    "success_event_ids": ["success-one", "success-two"], "success_orders": [11, 12],
                    "required_successes": 2,
                })
            assert kb.request_review(conn, owner.id, summary="recovery", expected_run_id=owner.current_run_id)
            reviewer, _ = kb.claim_task_for_dispatch(conn, owner.id, review=True, organization=organization)
            assert kb.complete_task(conn, owner.id, summary="PRIVATE WORKER PROSE", expected_run_id=reviewer.current_run_id)
        else:
            proposal = send_to_review(conn, owner)
            reviewer, _ = kb.claim_task_for_dispatch(conn, owner.id, review=True, organization=organization)
            assert kb.block_task(conn, owner.id, reason="reviewed", expected_run_id=reviewer.current_run_id,
                                 decision_review={"proposal_event_id": proposal["proposal_event_id"], "outcome": "accepted"})
    event = {"event_id": "failure-one", "source_kind": "profile_cron", "source_scope": "aurora",
             "source_id": job["id"], "execution_id": "execution-one", "outcome_notice": notice}
    (home / "cron" / "operational-failures.jsonl").write_text(json.dumps(event) + "\n")
    outcomes, _ = notices.collect_operational_outcomes("aurora")
    assert len(outcomes) == 1
    return outcomes[0], home


def test_real_review_route_intake_and_receipt_end_to_end(accepted):
    outcome, _ = accepted
    assert notices.validate_operational_outcome_authority(outcome) == outcome
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="actual-id")
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "acknowledged"
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "acknowledged"
    assert adapter.send.await_count == 1
    assert adapter.send.call_args.kwargs["chat_id"] == "12345"
    assert "incident is not repaired" in adapter.send.call_args.kwargs["content"]
    assert notices.collect_operational_outcomes("aurora")[0] == []
    assert dl.sweep_recoverable() == []


@pytest.mark.parametrize("field,value", [("content", "forged"), ("event_id", 1), ("execution_profile", "builder")])
def test_modified_candidate_has_no_send_authority(accepted, field, value):
    outcome, _ = accepted
    outcome = {**outcome, field: value}
    adapter = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(notices.deliver_operational_outcome(outcome, adapter))
    adapter.send.assert_not_called()


@pytest.mark.parametrize("change", ["route", "intake", "episode"])
def test_revoked_authority_before_claim_prevents_send(accepted, change):
    outcome, home = accepted
    if change == "route":
        path = home / "cron" / "jobs.json"
        data = json.loads(path.read_text())
        data["jobs"][0]["deliver"] = "telegram:99999"
        path.write_text(json.dumps(data))
    elif change == "intake":
        (home / "cron" / "operational-failures.jsonl").write_text("")
    else:
        with kb.connect_closing() as conn, kb.write_txn(conn):
            kb._append_event(conn, outcome["task_id"], "workforce_handoff_recovery_required", {
                "failure_event_id": "new-failure", "failure_order": 20, "required_successes": 2,
            })
    adapter = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(notices.deliver_operational_outcome(outcome, adapter))
    adapter.send.assert_not_called()


@pytest.mark.parametrize("result", [None, SendResult(success=True), SendResult(success=False, error_kind="transient")])
def test_missing_receipt_is_uncertain_and_never_replayed(accepted, result):
    outcome, _ = accepted
    adapter = AsyncMock()
    adapter.send.return_value = result
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "uncertain"
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "uncertain"
    assert adapter.send.await_count == 1


def test_definite_rejection_has_backoff_and_cumulative_retry_limit(accepted, monkeypatch):
    outcome, _ = accepted
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=False, error_kind="rate_limited")
    now = dl.time.time()
    monkeypatch.setattr(dl.time, "time", lambda: now)
    for attempt in range(3):
        assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "pending"
        assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "pending"
        assert adapter.send.await_count == attempt + 1
        now += 120
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "pending"
    assert adapter.send.await_count == 3


def test_simultaneous_claims_send_only_once(accepted):
    outcome, _ = accepted
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="remote-id")
    async def race():
        return await asyncio.gather(*(notices.deliver_operational_outcome(deepcopy(outcome), adapter) for _ in range(5)))
    asyncio.run(race())
    assert adapter.send.await_count == 1


def test_real_gateway_tick_delivers_reviewed_outcome(accepted):
    from tests.gateway.test_kanban_notifier_coordination import Runner, finish_tick

    outcome, _ = accepted
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="tick-receipt")
    runner = Runner(adapter)
    asyncio.run(finish_tick(runner))
    asyncio.run(finish_tick(runner))
    assert adapter.send.await_count == 1
    record = dl.get_operational_outcome_delivery(outcome["request_root_id"], outcome["event_id"], outcome["route"]["route_key"])
    assert record.returned_message_id == "platform-message:telegram:tick-receipt"


def test_outcome_profile_probe_finds_route_without_writable_open(accepted):
    with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
        assert notices.operational_outcome_profiles({"aurora", "builder"}) == {"aurora"}
    spy_connect.assert_not_called()


def test_outcome_profile_probe_tolerates_legacy_schema(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    path.touch()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))

    with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
        assert notices.operational_outcome_profiles({"aurora"}) == set()
    spy_connect.assert_not_called()


def test_missing_profile_adapter_does_not_borrow_another(accepted):
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from tests.gateway.test_kanban_notifier_coordination import Runner, finish_tick

    class MultiplexRunner(Runner):
        _authorization_adapter = GatewayAuthorizationMixin._authorization_adapter

        def _active_profile_name(self):
            return "builder"

    adapter = AsyncMock()
    runner = MultiplexRunner(adapter)
    runner._profile_adapters = {"aurora": {}}
    asyncio.run(finish_tick(runner))
    adapter.send.assert_not_called()


@pytest.mark.parametrize("accepted", ["recovery"], indirect=True)
def test_source_accepted_recovery_uses_receipt_without_worker_prose(accepted):
    outcome, _ = accepted
    assert "recovery verified" in outcome["content"]
    assert "Routine: daily-note" in outcome["content"]
    assert "PRIVATE WORKER PROSE" not in outcome["content"]
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True, message_id="recovery-id")
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "acknowledged"


def test_timeout_marks_uncertain_without_retry(accepted):
    outcome, _ = accepted
    adapter = AsyncMock()
    adapter.send.side_effect = TimeoutError("private transport detail")
    with pytest.raises(TimeoutError):
        asyncio.run(notices.deliver_operational_outcome(outcome, adapter))
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "uncertain"
    record = dl.get_operational_outcome_delivery(outcome["request_root_id"], outcome["event_id"], outcome["route"]["route_key"])
    assert record.last_error == "send_interrupted"
    assert adapter.send.await_count == 1


def test_dead_sender_becomes_uncertain_not_replayed(accepted, monkeypatch):
    outcome, _ = accepted
    first = dl.claim_operational_outcome_delivery(outcome)
    assert first.send_claimed
    monkeypatch.setattr(dl, "_owner_alive", lambda *args: False)
    adapter = AsyncMock()
    assert asyncio.run(notices.deliver_operational_outcome(outcome, adapter)) == "uncertain"
    adapter.send.assert_not_called()


def test_substituting_another_intake_event_cannot_rebind_origin(accepted):
    outcome, home = accepted
    path = home / "cron" / "operational-failures.jsonl"
    event = json.loads(path.read_text())
    event["event_id"] = "another-event"
    path.write_text(json.dumps(event) + "\n")
    with kb.connect_closing() as conn, kb.write_txn(conn):
        body = json.loads(kb.get_task(conn, outcome["task_id"]).body)
        body["context"]["event_id"] = event["event_id"]
        conn.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(body), outcome["task_id"]))
    assert notices.collect_operational_outcomes("aurora")[0] == []
