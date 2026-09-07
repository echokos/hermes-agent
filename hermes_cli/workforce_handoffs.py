"""Durable, organization-aware workforce handoffs backed by Hermes Kanban."""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3
import time
from typing import Any

from hermes_cli import kanban_db
from hermes_cli.sqlite_util import write_txn
from hermes_cli.workforce_org import WorkforceOrganization, load_organization


HANDOFF_KIND = "workforce_handoff"


def _timestamp(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("deadline is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadlines must include a timezone")
    return int(parsed.timestamp())


def _body(task) -> dict[str, Any]:
    try:
        value = json.loads(task.body or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("task is not a structured workforce handoff") from exc
    if not isinstance(value, dict) or value.get("kind") != HANDOFF_KIND:
        raise ValueError("task is not a workforce handoff")
    return value


def _authorized_route(org: WorkforceOrganization, source: str, target: str) -> None:
    sender = org.validate_execution_profile(source)
    receiver = org.validate_execution_profile(target)
    if (
        sender.agent == "aurora"
        or receiver.manager == sender.agent
        or sender.manager == receiver.agent
        or {sender.agent, receiver.agent} == {"aurora", "grace"}
    ):
        return
    raise ValueError("cross-team handoffs and non-report assignments must route through Aurora")


def create_handoff(
    conn: sqlite3.Connection,
    *,
    source_agent: str,
    target_agent: str,
    expected_outcome: str,
    acceptance_test: str,
    evidence_references: list[str],
    acknowledgment_deadline: str,
    checkpoint_at: str,
    organization: WorkforceOrganization | None = None,
    context: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    requires_source_acceptance: bool = False,
    allow_overdue: bool = False,
    max_runtime_seconds: int | None = None,
) -> dict[str, Any]:
    org = organization or load_organization()
    _authorized_route(org, source_agent, target_agent)
    source = org.resolve_profile(source_agent).agent
    target = org.resolve_profile(target_agent).agent
    ack_at = _timestamp(acknowledgment_deadline)
    checkpoint = _timestamp(checkpoint_at)
    if checkpoint <= ack_at:
        raise ValueError("checkpoint must be after the acknowledgment deadline")
    now = int(time.time())
    if ack_at <= now and not allow_overdue:
        raise ValueError("acknowledgment deadline must be in the future")
    payload = {
        "kind": HANDOFF_KIND,
        "state": "pending_acknowledgment",
        "source_agent": source,
        "target_agent": target,
        "expected_outcome": str(expected_outcome).strip(),
        "acceptance_test": str(acceptance_test).strip(),
        "evidence_references": list(evidence_references),
        "acknowledgment_deadline": ack_at,
        "checkpoint_at": checkpoint,
        "created_at": now,
        "notification_targets": ["aurora", "chloe"],
        "requires_source_acceptance": bool(requires_source_acceptance),
    }
    if context is not None:
        if not isinstance(context, dict):
            raise ValueError("context must be an object")
        payload["context"] = context
    if not payload["expected_outcome"] or not payload["acceptance_test"]:
        raise ValueError("expected_outcome and acceptance_test are required")
    task_id = kanban_db.create_task(
        conn,
        title=f"Handoff: {payload['expected_outcome'][:120]}",
        body=json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        assignee=target,
        created_by=source,
        workspace_kind="scratch",
        triage=True,
        idempotency_key=idempotency_key or (
            f"workforce-handoff:{source}:{target}:"
            f"{ack_at}:{payload['expected_outcome']}"
        ),
        max_runtime_seconds=max_runtime_seconds,
    )
    return {"task_id": task_id, **payload}


def acknowledge_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    organization: WorkforceOrganization | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    org = organization or load_organization()
    actor_id = org.validate_execution_profile(actor).agent
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    payload = _body(task)
    if actor_id != payload["target_agent"]:
        raise ValueError("only the receiving agent can acknowledge a handoff")
    if payload["state"] != "pending_acknowledgment":
        raise ValueError(f"handoff cannot be acknowledged from {payload['state']}")
    accepted_at = int(now if now is not None else time.time())
    if accepted_at > int(payload["acknowledgment_deadline"]):
        raise ValueError("acknowledgment deadline has passed; Aurora must review the overdue handoff")
    payload.update({"state": "accepted", "acknowledged_at": accepted_at})
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET body = ?, status = 'ready' WHERE id = ?",
            (json.dumps(payload, indent=2, sort_keys=True), task_id),
        )
        kanban_db._append_event(
            conn, task_id, "workforce_handoff_acknowledged", {"actor": actor_id}
        )
    kanban_db.notify_task_updated(conn, task_id, ("body", "status"))
    return {"task_id": task_id, **payload}


def record_checkpoint(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    evidence_references: list[str],
    next_checkpoint_at: str | None = None,
    organization: WorkforceOrganization | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    org = organization or load_organization()
    actor_id = org.validate_execution_profile(actor).agent
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    payload = _body(task)
    allowed = {payload["target_agent"], payload["source_agent"], "aurora"}
    if actor_id not in allowed:
        raise ValueError("only the receiver, sender, or Aurora may record a checkpoint")
    if payload["state"] not in {"accepted", "active"}:
        raise ValueError(f"checkpoint cannot be recorded from {payload['state']}")
    recorded_at = int(now if now is not None else time.time())
    payload["state"] = "active"
    payload["last_checkpoint_at"] = recorded_at
    payload["checkpoint_evidence"] = list(evidence_references)
    if next_checkpoint_at:
        next_at = _timestamp(next_checkpoint_at)
        if next_at <= recorded_at:
            raise ValueError("next checkpoint must be in the future")
        payload["checkpoint_at"] = next_at
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET body = ? WHERE id = ?",
            (json.dumps(payload, indent=2, sort_keys=True), task_id),
        )
        kanban_db._append_event(
            conn,
            task_id,
            "workforce_handoff_checkpoint",
            {"actor": actor_id, "evidence_count": len(evidence_references)},
        )
    kanban_db.notify_task_updated(conn, task_id, ("body",))
    return {"task_id": task_id, **payload}


def sweep_overdue_handoffs(
    conn: sqlite3.Connection,
    *,
    actor: str,
    organization: WorkforceOrganization | None = None,
    now: int | None = None,
) -> list[dict[str, Any]]:
    org = organization or load_organization()
    actor_id = org.validate_execution_profile(actor).agent
    if actor_id not in {"aurora", "chloe"}:
        raise ValueError("only Aurora or Chloe may perform the mechanical overdue sweep")
    current = int(now if now is not None else time.time())
    rows = conn.execute(
        "SELECT id FROM tasks WHERE status NOT IN ('done','archived') AND body LIKE ?",
        ('%"kind": "workforce_handoff"%',),
    ).fetchall()
    changed: list[dict[str, Any]] = []
    for row in rows:
        task = kanban_db.get_task(conn, row["id"])
        if task is None:
            continue
        payload = _body(task)
        if (
            payload["state"] == "pending_acknowledgment"
            and int(payload["acknowledgment_deadline"]) < current
        ):
            state = "acknowledgment_overdue"
            event = "workforce_handoff_acknowledgment_overdue"
        elif (
            payload["state"] in {"accepted", "active"}
            and int(payload["checkpoint_at"]) < current
        ):
            state = "stalled"
            event = "workforce_handoff_stalled"
        else:
            continue
        payload.update({"state": state, "flagged_at": current, "flagged_by": actor_id})
        with write_txn(conn):
            conn.execute(
                "UPDATE tasks SET body = ?, status = 'blocked' WHERE id = ?",
                (json.dumps(payload, indent=2, sort_keys=True), task.id),
            )
            kanban_db._append_event(
                conn,
                task.id,
                event,
                {"actor": actor_id, "notify": ["aurora", "chloe"]},
            )
        kanban_db.notify_task_updated(conn, task.id, ("body", "status"))
        changed.append(
            {
                "task_id": task.id,
                "state": state,
                "notify": ["aurora", "chloe"],
                "decision_owner": "aurora",
            }
        )
    return changed
