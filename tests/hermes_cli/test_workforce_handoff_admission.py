from contextlib import contextmanager
import json
from pathlib import Path
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


def _create_handoff(
    conn,
    *,
    now: int,
    label: str,
    owned: bool,
    checkpoint_seconds: int = 120,
) -> str:
    context = None
    if owned:
        context = {
            "kind": "owned_operational_failure",
            "technical_owner": "alina",
            "director": "aurora",
            "workflow_id": f"workflow-{label}",
            "event_id": f"failure-{label}",
        }
    return create_handoff(
        conn,
        source_agent="aurora",
        target_agent="alina",
        expected_outcome=f"Repair {label}",
        acceptance_test="A verified execution succeeds",
        evidence_references=[f"execution:{label}"],
        acknowledgment_deadline=_iso(now + 60),
        checkpoint_at=_iso(now + checkpoint_seconds),
        organization=ORG,
        context=context,
        requires_source_acceptance=owned,
    )["task_id"]


def _accept_owned(conn, *, now: int, label: str) -> tuple[str, str]:
    task_id = _create_handoff(conn, now=now, label=label, owned=True)
    pickup = claim_owned_failure_handoff_pickup(
        conn,
        target_agent="alina",
        organization=ORG,
        now=now + 1,
    )
    assert pickup is not None and pickup["task_id"] == task_id
    acknowledge_handoff(
        conn,
        task_id,
        actor="alina",
        organization=ORG,
        now=now + 2,
    )
    return task_id, pickup["request_root_id"]


def _ledger(conn) -> dict[str, tuple[tuple, ...]]:
    return {
        "events": tuple(
            tuple(row)
            for row in conn.execute("SELECT * FROM task_events ORDER BY id")
        ),
        "requests": tuple(
            tuple(row)
            for row in conn.execute("SELECT * FROM coordination_requests ORDER BY id")
        ),
        "runs": tuple(
            tuple(row)
            for row in conn.execute("SELECT * FROM task_runs ORDER BY id")
        ),
    }


def _prepare_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        kanban_db,
        "_resolve_dispatch_profile",
        lambda assignee: assignee,
    )
    monkeypatch.setattr(
        kanban_db,
        "_memory_pressure_level",
        lambda sample=None: "unknown",
    )


def test_overdue_owned_handoff_cannot_reenter_dispatch(conn, monkeypatch):
    now = int(time.time())
    task_id = _create_handoff(conn, now=now, label="expired-owned", owned=True)
    assert sweep_overdue_handoffs(
        conn,
        actor="chloe",
        organization=ORG,
        now=now + 61,
    )[0]["state"] == "acknowledgment_overdue"
    baseline = _ledger(conn)

    assert kanban_db.recompute_ready(conn) == 0
    assert kanban_db.promote_task(
        conn,
        task_id,
        actor="operator",
        force=True,
        dry_run=True,
    ) == (False, "workforce handoff state is not launchable")
    assert kanban_db.promote_task(
        conn,
        task_id,
        actor="operator",
        force=True,
    ) == (False, "workforce handoff state is not launchable")
    assert kanban_db.unblock_task(conn, task_id) is False
    assert kanban_db.get_task(conn, task_id).status == "blocked"
    assert _ledger(conn) == baseline

    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    forced_ready_ledger = _ledger(conn)
    assert kanban_db.claim_task(conn, task_id) is None
    with pytest.raises(
        kanban_db.CoordinationLaunchRefused,
        match="workforce handoff state is not launchable",
    ):
        kanban_db.reserve_coordination_launch(conn, task_id, organization=ORG)
    with pytest.raises(kanban_db.CoordinationLaunchRefused):
        kanban_db.claim_task_for_dispatch(conn, task_id, organization=ORG)
    _prepare_dispatch(monkeypatch)
    spawned: list[str] = []
    result = kanban_db.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace: spawned.append(task.id),
        reconcile_orphans=False,
    )
    assert spawned == []
    assert result.spawned == []
    assert result.coordination_guardrails == []
    assert result.coordination_deferred == [
        (task_id, "workforce handoff state is not launchable")
    ]
    assert kanban_db.has_spawnable_ready(conn) is False
    assert kanban_db.get_task(conn, task_id).current_run_id is None
    assert _ledger(conn) == forced_ready_ledger

    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    forced_review_ledger = _ledger(conn)
    assert kanban_db.claim_review_task(conn, task_id) is None
    with pytest.raises(kanban_db.CoordinationLaunchRefused):
        kanban_db.claim_task_for_dispatch(
            conn,
            task_id,
            review=True,
            organization=ORG,
        )
    result = kanban_db.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace: spawned.append(task.id),
        reconcile_orphans=False,
    )
    assert result.coordination_guardrails == []
    assert result.coordination_deferred == [
        (task_id, "workforce handoff state is not launchable")
    ]
    assert kanban_db.has_spawnable_review(conn) is False
    assert _ledger(conn) == forced_review_ledger


def test_stalled_owned_handoff_with_request_is_inert(conn, monkeypatch):
    now = int(time.time())
    task_id, request_id = _accept_owned(conn, now=now, label="stalled-owned")
    record_checkpoint(
        conn,
        task_id,
        actor="alina",
        evidence_references=["repair:started"],
        organization=ORG,
        now=now + 10,
    )
    assert sweep_overdue_handoffs(
        conn,
        actor="chloe",
        organization=ORG,
        now=now + 121,
    )[0]["state"] == "stalled"
    baseline = _ledger(conn)

    assert kanban_db.recompute_ready(conn) == 0
    assert kanban_db.promote_task(
        conn,
        task_id,
        actor="operator",
        force=True,
    ) == (False, "workforce handoff state is not launchable")
    assert kanban_db.unblock_task(conn, task_id) is False
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    forced_ready_ledger = _ledger(conn)
    assert kanban_db.claim_task(conn, task_id) is None

    _prepare_dispatch(monkeypatch)
    result = kanban_db.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace: pytest.fail("stalled handoff spawned"),
        reconcile_orphans=False,
    )
    assert result.coordination_guardrails == []
    assert result.coordination_deferred == [
        (task_id, "workforce handoff state is not launchable")
    ]
    assert kanban_db.get_coordination_request(conn, request_id).status == "active"
    assert _ledger(conn) == forced_ready_ledger
    assert baseline["requests"] == forced_ready_ledger["requests"]


@pytest.mark.parametrize(
    "link_fault",
    ["null", "missing", "wrong_kind", "wrong_root"],
)
def test_invalid_owned_request_link_refuses_without_touching_ledger(
    conn,
    monkeypatch,
    link_fault,
):
    now = int(time.time())
    task_id, request_id = _accept_owned(conn, now=now, label=link_fault)
    with kanban_db.write_txn(conn):
        if link_fault == "null":
            conn.execute(
                "UPDATE tasks SET request_root_id = NULL WHERE id = ?",
                (task_id,),
            )
        elif link_fault == "missing":
            conn.execute(
                "UPDATE tasks SET request_root_id = 'cr_missing' WHERE id = ?",
                (task_id,),
            )
        elif link_fault == "wrong_kind":
            conn.execute(
                "UPDATE coordination_requests SET kind = 'origin_request' WHERE id = ?",
                (request_id,),
            )
        else:
            conn.execute(
                "UPDATE coordination_requests SET root_task_id = 't_other' WHERE id = ?",
                (request_id,),
            )
    baseline = _ledger(conn)

    assert kanban_db.claim_task(conn, task_id) is None
    with pytest.raises(kanban_db.CoordinationLaunchRefused):
        kanban_db.reserve_coordination_launch(conn, task_id, organization=ORG)
    with pytest.raises(kanban_db.CoordinationLaunchRefused):
        kanban_db.claim_task_for_dispatch(conn, task_id, organization=ORG)
    _prepare_dispatch(monkeypatch)
    result = kanban_db.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace: pytest.fail("invalid handoff spawned"),
        reconcile_orphans=False,
    )

    assert result.spawned == []
    assert result.coordination_guardrails == []
    assert result.coordination_deferred[0][0] == task_id
    assert kanban_db.get_task(conn, task_id).status == "ready"
    assert _ledger(conn) == baseline


def test_valid_owned_and_public_handoffs_keep_launch_paths(conn):
    now = int(time.time())
    owned_id, request_id = _accept_owned(conn, now=now, label="valid-owned")
    record_checkpoint(
        conn,
        owned_id,
        actor="alina",
        evidence_references=["repair:active"],
        organization=ORG,
        now=now + 3,
    )
    owner, reservation = kanban_db.claim_task_for_dispatch(
        conn,
        owned_id,
        organization=ORG,
        now=now + 4,
    )
    assert owner is not None
    assert reservation is not None and reservation.request_root_id == request_id
    with kanban_db.write_txn(conn):
        kanban_db._append_event(
            conn,
            owned_id,
            "workforce_handoff_recovery_required",
            {
                "failure_event_id": "failure-valid-owned",
                "failure_order": 10,
                "required_successes": 2,
            },
        )
        kanban_db._append_event(
            conn,
            owned_id,
            "workforce_handoff_recovery_verified",
            {
                "failure_event_id": "failure-valid-owned",
                "failure_order": 10,
                "success_event_ids": ["success-1", "success-2"],
                "success_orders": [11, 12],
                "required_successes": 2,
            },
        )
    assert kanban_db.request_review(
        conn,
        owned_id,
        summary="Recovery evidence ready for source review",
        expected_run_id=owner.current_run_id,
    )
    assert kanban_db.mark_coordination_guardrail(
        conn,
        request_id,
        task_id=owned_id,
        reason="enter terminal source review",
        now=now + 5,
    )
    assert kanban_db.get_coordination_request(conn, request_id).status == "return_pending"
    reviewer, terminal_reservation = kanban_db.claim_task_for_dispatch(
        conn,
        owned_id,
        review=True,
        organization=ORG,
        now=now + 86_400,
    )
    assert reviewer is not None
    assert terminal_reservation is not None

    public_id = _create_handoff(conn, now=now, label="valid-public", owned=False)
    acknowledge_handoff(
        conn,
        public_id,
        actor="alina",
        organization=ORG,
        now=now + 2,
    )
    public, public_reservation = kanban_db.claim_task_for_dispatch(
        conn,
        public_id,
        organization=ORG,
    )
    assert public is not None
    assert public.request_root_id is None
    assert public_reservation is None

    public_active_id = _create_handoff(
        conn,
        now=now,
        label="active-public",
        owned=False,
    )
    acknowledge_handoff(
        conn,
        public_active_id,
        actor="alina",
        organization=ORG,
        now=now + 2,
    )
    active = record_checkpoint(
        conn,
        public_active_id,
        actor="alina",
        evidence_references=["public:active"],
        organization=ORG,
        now=now + 3,
    )
    assert active["state"] == "active"
    public_active, active_reservation = kanban_db.claim_task_for_dispatch(
        conn,
        public_active_id,
        organization=ORG,
    )
    assert public_active is not None
    assert public_active.request_root_id is None
    assert active_reservation is None


def test_public_pending_overdue_and_stalled_handoffs_are_not_launchable(conn):
    now = int(time.time())
    pending_id = _create_handoff(conn, now=now, label="pending-public", owned=False)
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (pending_id,))
    assert kanban_db.claim_task(conn, pending_id) is None

    overdue_id = _create_handoff(conn, now=now, label="overdue-public", owned=False)
    stalled_id = _create_handoff(conn, now=now, label="stalled-public", owned=False)
    acknowledge_handoff(
        conn,
        stalled_id,
        actor="alina",
        organization=ORG,
        now=now + 2,
    )
    record_checkpoint(
        conn,
        stalled_id,
        actor="alina",
        evidence_references=["public:started"],
        organization=ORG,
        now=now + 3,
    )
    swept = sweep_overdue_handoffs(
        conn,
        actor="chloe",
        organization=ORG,
        now=now + 61,
    )
    assert {row["task_id"] for row in swept} == {pending_id, overdue_id}
    assert kanban_db.recompute_ready(conn) == 0
    assert kanban_db.promote_task(
        conn,
        overdue_id,
        actor="operator",
        force=True,
    ) == (False, "workforce handoff state is not launchable")
    assert sweep_overdue_handoffs(
        conn,
        actor="chloe",
        organization=ORG,
        now=now + 121,
    ) == [
        {
            "task_id": stalled_id,
            "state": "stalled",
            "notify": ["aurora", "chloe"],
            "decision_owner": "aurora",
        }
    ]
    assert kanban_db.promote_task(
        conn,
        stalled_id,
        actor="operator",
        force=True,
    ) == (False, "workforce handoff state is not launchable")


@pytest.mark.parametrize("bad_state", [[], {}])
def test_malformed_handoff_state_does_not_starve_valid_task(
    conn,
    monkeypatch,
    bad_state,
):
    malformed_id = kanban_db.create_task(
        conn,
        title="Malformed workforce handoff",
        body=json.dumps(
            {
                "kind": "workforce_handoff",
                "state": bad_state,
                "context": {"kind": "owned_operational_failure"},
            }
        ),
        assignee="alina",
    )
    valid_id = kanban_db.create_task(
        conn,
        title="Independent valid task",
        assignee="alina",
    )
    malformed_ledger = _ledger(conn)
    _prepare_dispatch(monkeypatch)
    spawned: list[str] = []

    assert kanban_db.has_spawnable_ready(conn) is True
    result = kanban_db.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace: spawned.append(task.id),
        reconcile_orphans=False,
    )

    assert spawned == [valid_id]
    assert [row[0] for row in result.spawned] == [valid_id]
    assert result.coordination_deferred == [
        (malformed_id, "workforce handoff state is not launchable")
    ]
    assert kanban_db.get_task(conn, malformed_id).status == "ready"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
        (malformed_id,),
    ).fetchone()[0] == 0
    assert tuple(
        tuple(row)
        for row in conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (malformed_id,),
        )
    ) == tuple(
        row for row in malformed_ledger["events"] if row[1] == malformed_id
    )


def test_promote_rechecks_handoff_state_inside_write_transaction(
    conn,
    monkeypatch,
):
    now = int(time.time())
    task_id, _ = _accept_owned(conn, now=now, label="promote-race")
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
    events_before = _ledger(conn)["events"]
    original_write_txn = kanban_db.write_txn
    injected = False

    @contextmanager
    def inject_expiry(connection, *args, **kwargs):
        nonlocal injected
        if not injected:
            payload = json.loads(kanban_db.get_task(connection, task_id).body)
            payload["state"] = "stalled"
            connection.execute(
                "UPDATE tasks SET body = ? WHERE id = ?",
                (json.dumps(payload), task_id),
            )
            connection.commit()
            injected = True
        with original_write_txn(connection, *args, **kwargs):
            yield

    monkeypatch.setattr(kanban_db, "write_txn", inject_expiry)
    assert kanban_db.promote_task(
        conn,
        task_id,
        actor="operator",
        force=True,
    ) == (False, "workforce handoff state is not launchable")
    assert injected is True
    assert kanban_db.get_task(conn, task_id).status == "blocked"
    assert _ledger(conn)["events"] == events_before


def test_reservation_rechecks_handoff_link_inside_write_transaction(
    conn,
    monkeypatch,
):
    now = int(time.time())
    task_id, _ = _accept_owned(conn, now=now, label="reservation-race")
    original_write_txn = kanban_db.write_txn
    injected = False

    @contextmanager
    def inject_bad_link(connection, *args, **kwargs):
        nonlocal injected
        if not injected:
            connection.execute(
                "UPDATE tasks SET request_root_id = NULL WHERE id = ?",
                (task_id,),
            )
            connection.commit()
            injected = True
        with original_write_txn(connection, *args, **kwargs):
            yield

    baseline = _ledger(conn)
    monkeypatch.setattr(kanban_db, "write_txn", inject_bad_link)
    with pytest.raises(
        kanban_db.CoordinationLaunchRefused,
        match="has no coordination request",
    ):
        kanban_db.reserve_coordination_launch(
            conn,
            task_id,
            organization=ORG,
            now=now + 3,
        )
    assert injected is True
    assert _ledger(conn) == baseline
