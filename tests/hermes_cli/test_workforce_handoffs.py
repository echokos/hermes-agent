from pathlib import Path
import json
import time

import pytest

from hermes_cli import kanban_db
from hermes_cli.workforce_handoffs import (
    acknowledge_handoff,
    claim_owned_failure_handoff_pickup,
    create_handoff,
    record_checkpoint,
    sweep_overdue_handoffs,
)
from hermes_cli.workforce_org import load_organization


ORG = load_organization(Path(__file__).parents[2] / "workforce" / "organization.yaml")


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with kanban_db.connect_closing() as connection:
        yield connection


def _iso(epoch: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def test_cross_director_handoff_requires_aurora(conn):
    now = int(time.time())
    with pytest.raises(ValueError, match="route through Aurora"):
        create_handoff(
            conn,
            source_agent="emily",
            target_agent="xenia",
            expected_outcome="Validate data",
            acceptance_test="Evidence attached",
            evidence_references=[],
            acknowledgment_deadline=_iso(now + 60),
            checkpoint_at=_iso(now + 120),
            organization=ORG,
        )


@pytest.mark.parametrize(
    ("source", "target"),
    [("milena", "grace"), ("emily", "aurora"), ("sage", "emily"), ("grace", "aurora")],
)
def test_reporting_line_and_executive_peer_handoffs_route_internally(conn, source, target):
    now = int(time.time())
    created = create_handoff(
        conn,
        source_agent=source,
        target_agent=target,
        expected_outcome="Resolve one internal dependency",
        acceptance_test="The accountable manager records a disposition",
        evidence_references=["kanban:t_source"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
    )
    assert created["source_agent"] == source
    assert created["target_agent"] == target


def test_receiver_must_acknowledge_and_stalled_checkpoint_notifies_aurora_chloe(conn):
    now = int(time.time())
    created = create_handoff(
        conn,
        source_agent="aurora",
        target_agent="emily",
        expected_outcome="Prepare product evidence",
        acceptance_test="Packet contains source links",
        evidence_references=["kanban:source"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
    )
    task_id = created["task_id"]
    assert kanban_db.get_task(conn, task_id).status == "triage"
    with pytest.raises(ValueError, match="receiving agent"):
        acknowledge_handoff(conn, task_id, actor="xenia", organization=ORG, now=now + 10)
    accepted = acknowledge_handoff(
        conn, task_id, actor="emily", organization=ORG, now=now + 10
    )
    assert accepted["state"] == "accepted"
    assert kanban_db.get_task(conn, task_id).status == "ready"
    stalled = sweep_overdue_handoffs(
        conn, actor="chloe", organization=ORG, now=now + 121
    )
    assert stalled == [{
        "task_id": task_id,
        "state": "stalled",
        "notify": ["aurora", "chloe"],
        "decision_owner": "aurora",
    }]
    assert kanban_db.get_task(conn, task_id).status == "blocked"


def test_checkpoint_moves_deadline_without_changing_authority(conn):
    now = int(time.time())
    created = create_handoff(
        conn,
        source_agent="emily",
        target_agent="sage",
        expected_outcome="Review product evidence",
        acceptance_test="Findings linked",
        evidence_references=[],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
    )
    acknowledge_handoff(
        conn, created["task_id"], actor="sage", organization=ORG, now=now + 10
    )
    result = record_checkpoint(
        conn,
        created["task_id"],
        actor="sage",
        evidence_references=["repo:commit"],
        next_checkpoint_at=_iso(now + 240),
        organization=ORG,
        now=now + 100,
    )
    assert result["state"] == "active"
    assert result["checkpoint_at"] == now + 240


def test_owned_failure_pickup_is_one_shot_bounded_and_keeps_ack_pending(conn):
    now = int(time.time())
    created = create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome="Repair scheduled integration",
        acceptance_test="Two later executions succeed",
        evidence_references=["execution:failure-1"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 3600),
        organization=ORG,
        context={
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
            "workflow_id": "scheduled-integration",
            "event_id": "failure-1",
        },
        requires_source_acceptance=True,
    )
    task_id = created["task_id"]

    pickup = claim_owned_failure_handoff_pickup(
        conn, target_agent="alina", organization=ORG, now=now + 1,
    )
    duplicate = claim_owned_failure_handoff_pickup(
        conn, target_agent="alina", organization=ORG, now=now + 2,
    )

    assert pickup == {
        "task_id": task_id,
        "target_agent": "alina",
        "source_agent": "aurora",
        "request_root_id": pickup["request_root_id"],
        "claimed_at": now + 1,
    }
    assert duplicate is None
    task = kanban_db.get_task(conn, task_id)
    assert task.status == "triage"
    assert task.request_root_id == pickup["request_root_id"]
    assert json.loads(task.body)["state"] == "pending_acknowledgment"
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    request = kanban_db.get_coordination_request(
        conn, pickup["request_root_id"]
    )
    assert request.kind == "owned_operational_failure"
    assert request.max_model_calls == 20
    assert request.final_model_call_reserve == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
        "AND kind = 'workforce_handoff_pickup_claimed'",
        (task_id,),
    ).fetchone()[0] == 1

    accepted = acknowledge_handoff(
        conn, task_id, actor="alina", organization=ORG, now=now + 3,
    )
    assert accepted["state"] == "accepted"
    assert kanban_db.get_task(conn, task_id).status == "ready"


def test_pickup_ignores_generic_and_expired_handoffs(conn):
    now = int(time.time())
    create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome="Ordinary handoff",
        acceptance_test="Done",
        evidence_references=[],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
    )
    create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome="Expired operational failure",
        acceptance_test="Recovered",
        evidence_references=[],
        acknowledgment_deadline=_iso(now - 10),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
        context={
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
        },
        requires_source_acceptance=True,
        allow_overdue=True,
    )
    assert claim_owned_failure_handoff_pickup(
        conn, target_agent="alina", organization=ORG, now=now,
    ) is None


def test_pickup_skips_malformed_first_candidate_and_requires_literal_flag(conn):
    now = int(time.time())
    malformed_body = {
        "kind": "workforce_handoff",
        "state": "pending_acknowledgment",
        "source_agent": "aurora",
        "target_agent": "alina",
        "acknowledgment_deadline": "not-an-integer",
        "requires_source_acceptance": True,
        "context": {
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
        },
    }
    malformed = kanban_db.create_task(
        conn,
        title="Malformed first candidate",
        body=json.dumps(malformed_body),
        assignee="alina",
        triage=True,
    )
    truthy_flag_body = dict(malformed_body)
    truthy_flag_body["acknowledgment_deadline"] = now + 60
    truthy_flag_body["requires_source_acceptance"] = "true"
    truthy = kanban_db.create_task(
        conn,
        title="Truthy flag is not authority",
        body=json.dumps(truthy_flag_body),
        assignee="alina",
        triage=True,
    )
    malformed_checkpoint_body = dict(malformed_body)
    malformed_checkpoint_body["acknowledgment_deadline"] = now + 60
    malformed_checkpoint_body["checkpoint_at"] = "not-an-integer"
    malformed_checkpoint = kanban_db.create_task(
        conn,
        title="Malformed checkpoint",
        body=json.dumps(malformed_checkpoint_body),
        assignee="alina",
        triage=True,
    )
    invalid_factory_body = dict(malformed_body)
    invalid_factory_body["acknowledgment_deadline"] = now + 60
    invalid_factory_body["checkpoint_at"] = now + 120
    invalid_factory_body["context"] = {
        **malformed_body["context"],
        "technical_owner": "sage",
    }
    invalid_factory = kanban_db.create_task(
        conn,
        title="Factory-invalid context",
        body=json.dumps(invalid_factory_body),
        assignee="alina",
        triage=True,
    )
    valid = create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome="Repair valid later candidate",
        acceptance_test="Two later executions succeed",
        evidence_references=["execution:failure-valid"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 3600),
        organization=ORG,
        context={
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
            "event_id": "failure-valid",
        },
        requires_source_acceptance=True,
    )["task_id"]
    with kanban_db.write_txn(conn):
        for created_at, task_id in enumerate(
            (malformed, truthy, malformed_checkpoint, invalid_factory, valid),
            start=1,
        ):
            conn.execute(
                "UPDATE tasks SET created_at = ? WHERE id = ?",
                (created_at, task_id),
            )

    pickup = claim_owned_failure_handoff_pickup(
        conn, target_agent="alina", organization=ORG, now=now,
    )

    assert pickup is not None
    assert pickup["task_id"] == valid
    assert kanban_db.get_task(conn, malformed).request_root_id is None
    assert kanban_db.get_task(conn, truthy).request_root_id is None
    assert kanban_db.get_task(conn, malformed_checkpoint).request_root_id is None
    assert kanban_db.get_task(conn, invalid_factory).request_root_id is None


def test_owned_failure_review_wait_is_not_marked_stalled(conn):
    now = int(time.time())
    task_id = create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome="Repair and wait for recovery proof",
        acceptance_test="Two later executions succeed",
        evidence_references=["execution:failure-1"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 120),
        organization=ORG,
        context={
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
            "event_id": "failure-1",
        },
        requires_source_acceptance=True,
    )["task_id"]
    claim_owned_failure_handoff_pickup(
        conn, target_agent="alina", organization=ORG, now=now + 1,
    )
    acknowledge_handoff(
        conn, task_id, actor="alina", organization=ORG, now=now + 2,
    )
    owner_run = kanban_db.claim_task(conn, task_id, claimer="alina:test")
    assert owner_run is not None
    assert kanban_db.request_review(
        conn,
        task_id,
        summary="repair applied",
        expected_run_id=owner_run.current_run_id,
    )
    with kanban_db.write_txn(conn):
        kanban_db._append_event(
            conn,
            task_id,
            "workforce_handoff_recovery_required",
            {
                "failure_event_id": "failure-1",
                "failure_order": 10,
                "required_successes": 2,
            },
        )

    assert sweep_overdue_handoffs(
        conn, actor="chloe", organization=ORG, now=now + 86_400,
    ) == []
    assert kanban_db.get_task(conn, task_id).status == "review"
    with pytest.raises(
        kanban_db.CoordinationLaunchDeferred, match="recovery verification"
    ):
        kanban_db.claim_task_for_dispatch(
            conn, task_id, review=True, organization=ORG, now=now + 86_400,
        )

    with kanban_db.write_txn(conn):
        kanban_db._append_event(
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
    reviewer, _ = kanban_db.claim_task_for_dispatch(
        conn, task_id, review=True, organization=ORG, now=now + 86_401,
    )
    assert reviewer is not None
    assert reviewer.assignee == "aurora"


def test_source_acceptance_requires_target_ack_and_source_review_run(conn):
    now = int(time.time())
    created = create_handoff(
        conn,
        source_agent="emily",
        target_agent="sage",
        expected_outcome="Repair the scheduled product workflow",
        acceptance_test="Two distinct scheduled executions succeed",
        evidence_references=["cron:job-1"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + 240),
        organization=ORG,
        requires_source_acceptance=True,
    )
    task_id = created["task_id"]

    # A target cannot close its own work without returning it to the source.
    acknowledge_handoff(
        conn, task_id, actor="sage", organization=ORG, now=now + 1
    )
    owner_run = kanban_db.claim_task(conn, task_id, claimer="sage:test")
    assert owner_run is not None
    assert kanban_db.complete_task(
        conn,
        task_id,
        summary="Implementation finished without review",
        expected_run_id=owner_run.current_run_id,
    ) is False

    ok, reason = kanban_db.request_review(
        conn,
        task_id,
        summary="Repair evidence attached",
        reviewer="xenia",
        expected_run_id=owner_run.current_run_id,
        with_reason=True,
    )
    assert ok is False
    assert reason == "handoff review must return to its source"

    assert kanban_db.request_review(
        conn,
        task_id,
        summary="Repair evidence attached",
        expected_run_id=owner_run.current_run_id,
    )
    assert kanban_db.get_task(conn, task_id).assignee == "emily"
    review_run = kanban_db.claim_review_task(
        conn, task_id, claimer="emily:test"
    )
    assert review_run is not None
    assert kanban_db.complete_task(
        conn,
        task_id,
        summary="Accepted after source-owned verification",
        expected_run_id=review_run.current_run_id,
    )
