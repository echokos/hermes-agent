from pathlib import Path
import time

import pytest

from hermes_cli import kanban_db
from hermes_cli.workforce_handoffs import (
    acknowledge_handoff,
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
