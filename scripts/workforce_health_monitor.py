#!/usr/bin/env python3
"""Deterministically turn Hermes operational failures into owned repair work.

The monitor spends no model tokens while healthy. Opted-in scheduler and host
failures create or attach one owner handoff immediately; legacy Cron jobs retain
the two-failure threshold. Stable execution identities prevent replay fanout.
Two later successful executions are recovery evidence, while owner acknowledgment
and director acceptance remain mandatory for an owned incident to resolve.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any

from agent.redact import redact_sensitive_text
from cron.operational_failures import profile_failure_event
from hermes_cli import kanban_db
from hermes_cli.workforce_handoffs import (
    create_handoff,
    sweep_overdue_handoffs,
)
from hermes_cli.workforce_org import load_organization
from hermes_constants import get_hermes_home


FAILURE_STATUSES = {"error", "failed", "unknown"}
ACTIVE_TASK_STATUSES = {"triage", "todo", "ready", "running", "blocked", "review", "scheduled"}
INTAKE_FILENAME = "operational-failures.jsonl"


def _safe_error(value: Any) -> str:
    return redact_sensitive_text(str(value or "unspecified failure"), force=True)[:800]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _jobs(profile: Path) -> list[dict[str, Any]]:
    raw = _read_json(profile / "cron" / "jobs.json", {})
    rows = raw.get("jobs", []) if isinstance(raw, dict) else raw
    return [row for row in rows if isinstance(row, dict)]


def _ledger_enabled_at(job: dict[str, Any]) -> str:
    ownership = job.get("failure_ownership")
    text = (
        str(ownership.get("enabled_at") or "").strip()
        if isinstance(ownership, dict)
        else ""
    )
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        return ""
    return parsed.astimezone(timezone.utc).isoformat()


def _new_intake_events(
    path: Path,
    cursors: dict[str, Any],
    *,
    profile: Path | None = None,
) -> list[dict[str, Any]]:
    """Consume only complete records after a durable per-file byte cursor.

    An inode replacement or truncation restarts from offset zero. Stable
    execution event ids in monitor state make that replay inert even after an
    incident has reached a terminal state. A torn final write is deliberately
    not advanced, so it is retried after the writer finishes.
    """
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        return []
    row = cursors.setdefault(key, {})
    previous_identity = (row.get("device"), row.get("inode"))
    identity = (stat.st_dev, stat.st_ino)
    offset = int(row.get("offset") or 0)
    if previous_identity != identity or stat.st_size < offset:
        offset = 0
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return []
    consumed = 0
    events: list[dict[str, Any]] = []
    for raw_line in data.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            break
        consumed += len(raw_line)
        try:
            event = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict) or event.get("schema_version") != 1:
            continue
        event = {
            **event,
            "_intake_path": key,
            "_intake_offset": offset + consumed,
        }
        if profile is not None:
            event["_profile_home"] = str(profile)
            event.setdefault("source_scope", profile.name)
        events.append(event)
    row.update({"device": stat.st_dev, "inode": stat.st_ino, "offset": offset + consumed})
    return events


def _ledger_intake_events(
    profile: Path,
    job: dict[str, Any],
    cursor: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct opted-in events from the existing durable execution ledger.

    JSONL and SQLite are deliberately separate durability boundaries. If the
    append fails but the scheduler later terminalizes its execution row, this
    projection recovers the event on the next monitor pass. Event identity is
    execution-scoped, so a normally appended row and its ledger projection
    collapse to one event.
    """
    database = profile / "cron" / "executions.db"
    if not database.is_file():
        return []
    conn: sqlite3.Connection | None = None
    enabled_at = _ledger_enabled_at(job)
    initializing = not cursor.get("initialized")
    try:
        conn = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        job_id = str(job.get("id") or "")
        if cursor.get("initialized"):
            finished_at = str(cursor.get("finished_at") or "")
            execution_id = str(cursor.get("execution_id") or "")
            rows = conn.execute(
                "SELECT * FROM executions WHERE job_id = ? "
                "AND status IN ('completed','failed','unknown') "
                "AND finished_at IS NOT NULL "
                "AND (finished_at > ? OR (finished_at = ? AND id > ?)) "
                "ORDER BY finished_at, id LIMIT 256",
                (job_id, finished_at, finished_at, execution_id),
            ).fetchall()
        else:
            if enabled_at:
                rows = conn.execute(
                    "SELECT * FROM executions WHERE job_id = ? "
                    "AND status IN ('completed','failed','unknown') "
                    "AND finished_at IS NOT NULL "
                    "AND julianday(finished_at) >= julianday(?) "
                    "ORDER BY finished_at, id LIMIT 256",
                    (job_id, enabled_at),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM executions WHERE job_id = ? "
                    "AND status IN ('completed','failed','unknown') "
                    "AND finished_at IS NOT NULL "
                    "ORDER BY finished_at DESC, id DESC LIMIT 1",
                    (job_id,),
                ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()
    cursor["initialized"] = True
    if initializing and not rows and enabled_at and not cursor.get("finished_at"):
        cursor.update({"finished_at": enabled_at, "execution_id": ""})
    if rows:
        newest = rows[-1]
        # Bootstrap without an explicit opt-in baseline treats a latest success
        # as the healthy starting point instead of reopening older incidents.
        if not cursor.get("finished_at") and not enabled_at and len(rows) == 1:
            bootstrap_healthy = str(newest["status"] or "") == "completed"
        else:
            bootstrap_healthy = False
        cursor.update({
            "finished_at": str(newest["finished_at"] or ""),
            "execution_id": str(newest["id"] or ""),
        })
        if bootstrap_healthy:
            return []
    events: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        status = str(values.get("status") or "")
        outcome = "recovered" if status == "completed" else "failure"
        error = (
            "scheduler execution completed"
            if outcome == "recovered"
            else values.get("error") or f"scheduler execution {status or 'failed'}"
        )
        event = profile_failure_event(
            job,
            error,
            execution_id=str(values.get("id") or ""),
            outcome=outcome,
            source_scope=profile.name,
            occurred_at=(
                values.get("finished_at")
                or values.get("started_at")
                or values.get("claimed_at")
            ),
        )
        if event is not None:
            event["_recovered_from_execution_ledger"] = True
            events.append(event)
    return events


def _two_recent_successes(profile: Path, job_id: str) -> bool:
    database = profile / "cron" / "executions.db"
    if not database.is_file():
        return False
    conn = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT status FROM executions WHERE job_id=? "
            "ORDER BY claimed_at DESC, id DESC LIMIT 2",
            (job_id,),
        ).fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return len(rows) == 2 and all(str(row[0]) == "completed" for row in rows)


def _active_matching_task(conn: sqlite3.Connection, job_id: str, job_name: str):
    name_probe = " ".join(job_name.split())[:80]
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status IN ("
        + ",".join("?" for _ in ACTIVE_TASK_STATUSES)
        + ") AND (body LIKE ? OR title LIKE ? OR body LIKE ?) "
        "ORDER BY created_at DESC LIMIT 1",
        (*sorted(ACTIVE_TASK_STATUSES), f"%{job_id}%", f"%{name_probe}%", f"%{name_probe}%"),
    ).fetchone()
    return rows


def _record_failure(
    conn: sqlite3.Connection,
    *,
    profile_name: str,
    job: dict[str, Any],
    state_row: dict[str, Any],
) -> tuple[str, bool]:
    job_id = str(job.get("id") or "")
    job_name = str(job.get("name") or job_id)
    existing = _active_matching_task(conn, job_id, job_name)
    if existing:
        task_id = str(existing["id"])
        created = False
    else:
        episode = int(state_row.get("episode", 0)) + 1
        error = _safe_error(job.get("last_error"))
        body = json.dumps(
            {
                "kind": "recurring_workflow_failure",
                "profile": profile_name,
                "job_id": job_id,
                "job_name": job_name,
                "failure_streak": int(job.get("failure_streak") or 0),
                "last_status": str(job.get("last_status") or ""),
                "last_run_at": str(job.get("last_run_at") or ""),
                "sanitized_error": error,
                "decision_owner": "aurora",
                "required_outcome": "Route the defect to the canonical technical owner, repair it, and verify two consecutive successful executions.",
                "reporting": "Internal control-plane incident. Do not tell Elliott he is the blocker and do not deliver raw failure chatter to an Elliott-visible room.",
            },
            indent=2,
            sort_keys=True,
        )
        task_id = kanban_db.create_task(
            conn,
            title=f"Repair recurring Cron failure: {profile_name} / {job_name}"[:180],
            body=body,
            assignee="aurora",
            created_by="workforce-health-monitor",
            workspace_kind="scratch",
            initial_status="running",
            idempotency_key=f"workforce-health:cron:{profile_name}:{job_id}:episode:{episode}",
            max_runtime_seconds=900,
        )
        state_row["episode"] = episode
        created = True
    kanban_db.add_comment(
        conn,
        task_id,
        "workforce-health-monitor",
        "Recurring failure detected deterministically: "
        f"profile={profile_name}; job={job_name} ({job_id}); "
        f"failure_streak={int(job.get('failure_streak') or 0)}; "
        f"last_status={job.get('last_status')}; last_run_at={job.get('last_run_at')}; "
        f"sanitized_error={_safe_error(job.get('last_error'))}",
    )
    return task_id, created


def _event_key(event: dict[str, Any]) -> str:
    return ":".join((
        "intake",
        str(event.get("source_kind") or "unknown"),
        str(event.get("source_scope") or "unknown"),
        str(event.get("workflow_id") or "unknown"),
        str(event.get("source_id") or "unknown"),
    ))


def _event_already_processed(
    row: dict[str, Any], source_order: int, event_id: str,
) -> bool:
    watermark = int(row.get("last_processed_order") or -1)
    return source_order < watermark or (
        source_order == watermark
        and event_id in row.get("last_processed_event_ids", [])
    )


def _mark_event_processed(
    row: dict[str, Any], source_order: int, event_id: str,
) -> None:
    watermark = int(row.get("last_processed_order") or -1)
    if source_order > watermark:
        row["last_processed_order"] = source_order
        row["last_processed_event_ids"] = [event_id]
        return
    event_ids = row.setdefault("last_processed_event_ids", [])
    if event_id not in event_ids:
        event_ids.append(event_id)


def _event_owner(org, event: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Return canonical owner/director or Aurora with an auditable defect."""
    missing: list[str] = []
    try:
        owner = org.validate_execution_profile(str(event.get("technical_owner") or ""))
    except ValueError:
        owner = None
        missing.append("technical_owner")
    try:
        director = org.validate_execution_profile(str(event.get("director") or ""))
    except ValueError:
        director = None
        missing.append("director")
    if owner is None or director is None:
        return "aurora", "aurora", missing
    if director.agent != "aurora" and owner.manager != director.agent:
        return "aurora", "aurora", [*missing, "director_route"]
    return owner.agent, director.agent, missing


def _active_intake_task(
    conn: sqlite3.Connection,
    event: dict[str, Any],
    *,
    task_kind: str,
    match_ownership: bool = True,
):
    """Find the exact active incident after monitor-state loss."""
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status IN ("
        + ",".join("?" for _ in ACTIVE_TASK_STATUSES)
        + ") AND body LIKE ? ORDER BY created_at DESC, id DESC",
        (*sorted(ACTIVE_TASK_STATUSES), '%"kind": "workforce_handoff"%'),
    ).fetchall()
    expected_source = {
        "kind": str(event.get("source_kind") or "unknown"),
        "scope": str(event.get("source_scope") or "unknown"),
        "id": str(event.get("source_id") or "unknown"),
    }
    for row in rows:
        try:
            payload = json.loads(row["body"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        context = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(context, dict):
            continue
        if (
            context.get("kind") == task_kind
            and context.get("workflow_id") == str(event.get("workflow_id") or "unknown")
            and context.get("source") == expected_source
            and (
                not match_ownership
                or context.get("ownership_key")
                == str(event.get("ownership_key") or "")
            )
        ):
            return kanban_db.get_task(conn, str(row["id"]))
    return None


def _existing_incident_route(
    conn: sqlite3.Connection,
    task,
    event: dict[str, Any],
    *,
    owner: str,
    director: str,
) -> tuple[str, str, bool]:
    """Keep one active incident's actors stable across configuration changes."""
    try:
        payload = json.loads(task.body or "{}")
    except (TypeError, json.JSONDecodeError):
        return owner, director, False
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        return owner, director, False
    existing_owner = str(payload.get("target_agent") or owner)
    existing_director = str(payload.get("source_agent") or director)
    ownership_changed = context.get("ownership_key") != str(
        event.get("ownership_key") or ""
    )
    if ownership_changed:
        kanban_db.add_comment(
            conn,
            task.id,
            "workforce-health-monitor",
            "Failure ownership changed while this incident is active. "
            f"The existing handoff remains {existing_director}->{existing_owner} "
            "until director acceptance; the new declaration is "
            f"{director}->{owner} and applies to a later incident after closure. "
            f"event_id={event.get('event_id')}",
        )
    return existing_owner, existing_director, ownership_changed


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _record_intake_failure(
    conn: sqlite3.Connection,
    *,
    org,
    event: dict[str, Any],
    state_row: dict[str, Any],
) -> tuple[str, bool, bool, str, str]:
    """Create one owner card, or a visible configuration incident if invalid."""
    owner, director, ownership_errors = _event_owner(org, event)
    invalid = str(event.get("status")) != "failure" or bool(ownership_errors)
    source_kind = str(event.get("source_kind") or "unknown")
    source_id = str(event.get("source_id") or "unknown")
    workflow_id = str(event.get("workflow_id") or "unknown")
    task_key = _event_key(event)
    task_kind = "failure_ownership_configuration" if invalid else "owned_operational_failure"
    existing_id = str(state_row.get("task_id") or "")
    existing = kanban_db.get_task(conn, existing_id) if existing_id else None
    if existing and existing.status in ACTIVE_TASK_STATUSES:
        owner, director, changed = _existing_incident_route(
            conn, existing, event, owner=owner, director=director,
        )
        return existing_id, False, invalid or changed, owner, director
    existing = None

    episode = int(state_row.get("episode", 0)) + 1
    existing = _active_intake_task(conn, event, task_kind=task_kind)
    if existing is None:
        existing = _active_intake_task(
            conn, event, task_kind=task_kind, match_ownership=False,
        )
    if existing is not None:
        owner, director, changed = _existing_incident_route(
            conn, existing, event, owner=owner, director=director,
        )
        return existing.id, False, invalid or changed, owner, director
    context = {
        "kind": task_kind,
        "workflow_id": workflow_id,
        "source": {
            "kind": source_kind,
            "scope": str(event.get("source_scope") or "unknown"),
            "id": source_id,
        },
        "technical_owner": owner,
        "director": director,
        "severity": str(event.get("severity") or "warning"),
        "attempt": int(event.get("attempt") or 1),
        "dedupe_key": str(event.get("dedupe_key") or ""),
        "sanitized_error": _safe_error(event.get("sanitized_error")),
        "evidence_ref": event.get("evidence_ref"),
        "ack_deadline": event.get("ack_deadline"),
        "checkpoint_at": event.get("checkpoint_at"),
        "execution_id": str(event.get("execution_id") or ""),
        "event_id": str(event.get("event_id") or ""),
        "failure_order": int(event.get("source_order") or 0),
        "ownership_key": str(event.get("ownership_key") or ""),
        "recovery_successes_required": int(
            event.get("recovery_successes_required") or 2
        ),
        "required_outcome": (
            "Restore the integration and attach bounded verification evidence. "
            "The technical owner must acknowledge the incident; the director "
            "must accept the final repair before it can be reported upward."
        ),
        "recovery_gate": (
            "A later successful execution is evidence only. It cannot complete "
            "this owner-assigned incident or bypass director acceptance."
        ),
        "ownership_errors": ownership_errors + list(event.get("missing_fields") or []),
    }
    ack_deadline = int(event.get("ack_deadline") or int(time.time()) + 900)
    checkpoint_at = int(event.get("checkpoint_at") or int(time.time()) + 3600)
    if checkpoint_at <= ack_deadline:
        checkpoint_at = ack_deadline + 60
        invalid = True
        context["ownership_errors"] = [
            *context["ownership_errors"], "checkpoint_at",
        ]
    task_idempotency = (
        f"workforce-health:{task_key}:event:{event.get('event_id') or episode}"
    )
    prior = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
        (task_idempotency,),
    ).fetchone()
    created_handoff = create_handoff(
        conn,
        source_agent=director,
        target_agent=owner,
        expected_outcome=(
            f"Configure valid failure ownership for {workflow_id} / {source_id}"
            if invalid
            else f"Repair and verify {workflow_id} / {source_id}"
        ),
        acceptance_test=(
            "The declared owner and director route are valid, then a new failure "
            "is owned through this control plane."
            if invalid
            else (
                "Attach repair evidence and prove the configured number of "
                "distinct successful executions after the last failure."
            )
        ),
        evidence_references=[
            str(event.get("evidence_ref") or f"operational-event:{event.get('event_id')}"),
        ],
        acknowledgment_deadline=_iso(ack_deadline),
        checkpoint_at=_iso(checkpoint_at),
        organization=org,
        context=context,
        idempotency_key=task_idempotency,
        requires_source_acceptance=True,
        allow_overdue=True,
        max_runtime_seconds=max(60, min(3600, checkpoint_at - int(time.time()))),
    )
    task_id = str(created_handoff["task_id"])
    created = prior is None
    state_row["episode"] = episode
    if created:
        kanban_db.add_comment(
            conn,
            task_id,
            "workforce-health-monitor",
            "Durable failure intake accepted as a director-to-owner handoff: "
            f"workflow={workflow_id}; source={source_kind}:{source_id}; "
            f"technical_owner={owner}; director={director}; "
            f"sanitized_error={_safe_error(event.get('sanitized_error'))}",
        )
    return task_id, created, invalid, owner, director


def _record_recovery_requirement(
    conn: sqlite3.Connection,
    task_id: str,
    event: dict[str, Any],
) -> None:
    """Persist the latest failure episode on the task's durable event stream."""
    failure_event_id = str(event.get("event_id") or "")
    if not failure_event_id:
        return
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'workforce_handoff_recovery_required' ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and str(payload.get("failure_event_id") or "") == failure_event_id
        ):
            return
    with kanban_db.write_txn(conn):
        kanban_db._append_event(
            conn,
            task_id,
            "workforce_handoff_recovery_required",
            {
                "failure_event_id": failure_event_id,
                "failure_order": int(event.get("source_order") or 0),
                "required_successes": int(
                    event.get("recovery_successes_required") or 2
                ),
            },
        )


def _record_recovery_verification(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    failure_event_id: str,
    failure_order: int,
    success_event_ids: list[str],
    success_orders: list[int],
    required: int,
) -> None:
    """Attach exact post-failure execution evidence to the incident task."""
    with kanban_db.write_txn(conn):
        kanban_db._append_event(
            conn,
            task_id,
            "workforce_handoff_recovery_verified",
            {
                "failure_event_id": failure_event_id,
                "failure_order": int(failure_order),
                "success_event_ids": list(success_event_ids),
                "success_orders": [int(value) for value in success_orders],
                "required_successes": int(required),
            },
        )


def _sync_intake_lifecycle(
    conn: sqlite3.Connection, findings: dict[str, Any], now: int, org,
) -> int:
    """Verify exact handoff actors and raise only actionable Aurora exceptions."""
    sweep_overdue_handoffs(conn, actor="aurora", organization=org, now=now)
    overdue = 0
    for key, row in findings.items():
        if not key.startswith("intake:") or not row.get("task_id"):
            continue
        task = kanban_db.get_task(conn, str(row["task_id"]))
        if task is None:
            continue
        events = conn.execute(
            "SELECT id, run_id, kind, payload FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (task.id,),
        ).fetchall()
        owner = str(row.get("technical_owner") or "")
        director = str(row.get("director") or "")
        acknowledged_event_id = None
        review_event_id = None
        accepted_event_id = None
        for event in events:
            try:
                payload = json.loads(event["payload"] or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if (
                event["kind"] == "workforce_handoff_acknowledged"
                and str(payload.get("actor") or "") == owner
            ):
                acknowledged_event_id = int(event["id"])
                continue
            if (
                acknowledged_event_id is not None
                and event["kind"] == "review_requested"
                and str(payload.get("implementer") or "") == owner
                and str(payload.get("reviewer") or "") == director
            ):
                review_event_id = int(event["id"])
                continue
            if (
                review_event_id is not None
                and event["kind"] == "completed"
                and int(event["id"]) > review_event_id
                and event["run_id"] is not None
            ):
                run = conn.execute(
                    "SELECT profile, outcome FROM task_runs WHERE id = ?",
                    (int(event["run_id"]),),
                ).fetchone()
                claimed = conn.execute(
                    "SELECT payload FROM task_events WHERE task_id = ? "
                    "AND run_id = ? AND kind = 'claimed' ORDER BY id DESC LIMIT 1",
                    (task.id, int(event["run_id"])),
                ).fetchone()
                try:
                    claim_payload = (
                        json.loads(claimed["payload"] or "{}") if claimed else {}
                    )
                except (TypeError, json.JSONDecodeError):
                    claim_payload = {}
                if (
                    run is not None
                    and str(run["profile"] or "") == director
                    and str(run["outcome"] or "") == "completed"
                    and isinstance(claim_payload, dict)
                    and claim_payload.get("source_status") == "review"
                ):
                    accepted_event_id = int(event["id"])
        owner_acknowledged = acknowledged_event_id is not None
        director_review = review_event_id is not None
        director_accepted = accepted_event_id is not None
        row["owner_acknowledged"] = owner_acknowledged
        row["director_review_requested"] = director_review
        row["director_accepted"] = director_accepted
        row["ack_overdue"] = bool(
            not owner_acknowledged and task.status == "blocked"
        )
        overdue += int(row["ack_overdue"])
        if director_accepted and row.get("recovery_verified"):
            row["status"] = "resolved"
            row["resolved_at"] = now

        if task.status != "blocked" or task.block_kind != "needs_input":
            continue
        prior_exception_id = str(row.get("aurora_exception_task_id") or "")
        prior_exception = (
            kanban_db.get_task(conn, prior_exception_id)
            if prior_exception_id
            else None
        )
        if prior_exception and prior_exception.status in ACTIVE_TASK_STATUSES:
            continue
        blocked = next(
            (event for event in reversed(events) if event["kind"] == "blocked"),
            None,
        )
        try:
            blocked_payload = json.loads(blocked["payload"] or "{}") if blocked else {}
        except (TypeError, json.JSONDecodeError):
            blocked_payload = {}
        reason = _safe_error(
            blocked_payload.get("reason")
            if isinstance(blocked_payload, dict)
            else "human input required"
        )
        exception = create_handoff(
            conn,
            source_agent=director,
            target_agent="aurora",
            expected_outcome=f"Resolve the human-only blocker for {task.id}",
            acceptance_test=(
                "Aurora records one specific decision or access request; raw log "
                "pointers and repeated unchanged alerts are not acceptance."
            ),
            evidence_references=[f"kanban:{task.id}"],
            acknowledgment_deadline=_iso(now + 900),
            checkpoint_at=_iso(now + 3600),
            organization=org,
            context={
                "kind": "operational_failure_human_exception",
                "incident_task_id": task.id,
                "reason": reason,
            },
            idempotency_key=f"workforce-health:human-exception:{task.id}",
            max_runtime_seconds=900,
        )
        row["aurora_exception_task_id"] = str(exception["task_id"])
    return overdue


def run(*, organization: Path, database: Path, state_path: Path) -> dict[str, Any]:
    org = load_organization(organization, validate_profiles=True)
    state = _read_json(state_path, {"schema_version": 2, "findings": {}})
    state["schema_version"] = 2
    findings = state.setdefault("findings", {})
    cursors = state.setdefault("intake_cursors", {})
    ledger_cursors = state.setdefault("ledger_cursors", {})
    state.pop("processed_events", None)
    now = int(time.time())
    detected = attached = created = recovered = invalid_ownership = 0

    # Host producers write under the canonical Hermes state directory. The
    # monitor's own cursor file may intentionally live elsewhere (the shipped
    # unit uses ``workforce-control/``), so it cannot define the intake path.
    intake: list[dict[str, Any]] = _new_intake_events(
        get_hermes_home() / "state" / INTAKE_FILENAME, cursors,
    )
    for agent in org.operational_agents(include_planned=False):
        profile = Path(str(agent.profile_path))
        intake.extend(_new_intake_events(
            profile / "cron" / INTAKE_FILENAME, cursors, profile=profile,
        ))
        for job in _jobs(profile):
            if isinstance(job.get("failure_ownership"), dict):
                ledger_key = f"{profile}:{job.get('id') or ''}"
                intake.extend(_ledger_intake_events(
                    profile,
                    job,
                    ledger_cursors.setdefault(ledger_key, {}),
                ))

    intake.sort(key=lambda event: (
        int(event.get("source_order") or 0), str(event.get("event_id") or ""),
    ))

    with kanban_db.connect_closing(database) as conn:
        # Stable execution ids, rather than byte offsets, are authoritative.
        # This makes inode replacement/truncation replay-safe and lets the
        # existing execution ledger reconstruct an append that failed.
        for event in intake:
            event_id = str(event.get("event_id") or "")
            key = _event_key(event)
            row = findings.setdefault(key, {"episode": 0, "status": "healthy"})
            source_order = int(event.get("source_order") or 0)
            if not event_id or _event_already_processed(row, source_order, event_id):
                continue
            signature = str(event.get("dedupe_key") or event.get("event_id") or "")
            is_recovery = str(event.get("status") or "") == "recovered"
            if is_recovery:
                if (
                    row.get("status") in {"active", "recovered_pending_acceptance"}
                    and source_order > int(row.get("last_failure_order") or -1)
                ):
                    successes = row.setdefault("recovery_success_event_ids", [])
                    success_orders = row.setdefault("recovery_success_orders", {})
                    if event_id not in successes:
                        successes.append(event_id)
                    success_orders[event_id] = source_order
                    required = int(row.get("recovery_successes_required") or 2)
                    if len(successes) >= required and not row.get("recovery_verified"):
                        task_id = str(row.get("task_id") or "")
                        if task_id:
                            _record_recovery_verification(
                                conn,
                                task_id,
                                failure_event_id=str(
                                    row.get("episode_event_id") or ""
                                ),
                                failure_order=int(
                                    row.get("last_failure_order") or 0
                                ),
                                success_event_ids=list(successes),
                                success_orders=[
                                    int(success_orders.get(value) or 0)
                                    for value in successes
                                ],
                                required=required,
                            )
                            kanban_db.add_comment(
                                conn,
                                task_id,
                                "workforce-health-monitor",
                                "Scheduler-path recovery evidence accepted: "
                                f"{len(successes)} distinct post-failure executions "
                                f"completed successfully (required={required}). The "
                                "incident still requires technical-owner acknowledgment "
                                "and director acceptance.",
                            )
                        row.update({
                            "status": "recovered_pending_acceptance",
                            "recovered_at": now,
                            "recovery_verified": True,
                        })
                        recovered += 1
                _mark_event_processed(row, source_order, event_id)
                continue
            task_id, was_created, was_invalid, owner, director = _record_intake_failure(
                conn, org=org, event=event, state_row=row
            )
            _record_recovery_requirement(conn, task_id, event)
            row.update({
                "status": "active",
                "task_id": task_id,
                "monitor_created": was_created,
                "signature": signature,
                "first_seen_at": row.get("first_seen_at") or now,
                "last_seen_at": now,
                "last_failure_order": source_order,
                "episode_event_id": event_id,
                "technical_owner": owner,
                "director": director,
                "ack_deadline": event.get("ack_deadline"),
                "checkpoint_at": event.get("checkpoint_at"),
                "recovery_successes_required": int(
                    event.get("recovery_successes_required") or 2
                ),
                "recovery_success_event_ids": [],
                "recovery_success_orders": {},
                "recovery_verified": False,
            })
            _mark_event_processed(row, source_order, event_id)
            detected += 1
            created += int(was_created)
            attached += int(not was_created)
            invalid_ownership += int(was_invalid)

        for agent in org.operational_agents(include_planned=False):
            profile = Path(str(agent.profile_path))
            for job in _jobs(profile):
                job_id = str(job.get("id") or "").strip()
                if not job_id:
                    continue
                # Opted-in jobs are handled exclusively by ordered intake.
                # Do not let the legacy mutable job-streak scan create a
                # second incident for the same source.
                if isinstance(job.get("failure_ownership"), dict):
                    continue
                key = f"cron:{profile.name}:{job_id}"
                row = findings.setdefault(key, {"episode": 0, "status": "healthy"})
                is_failure = (
                    str(job.get("last_status") or "").lower() in FAILURE_STATUSES
                    and int(job.get("failure_streak") or 0) >= 2
                )
                signature = hashlib.sha256(
                    (str(job.get("last_status")) + "\0" + _safe_error(job.get("last_error"))).encode()
                ).hexdigest()
                if is_failure:
                    detected += 1
                    if row.get("status") != "active" or row.get("signature") != signature:
                        task_id, was_created = _record_failure(
                            conn,
                            profile_name=profile.name,
                            job=job,
                            state_row=row,
                        )
                        row.update({
                            "status": "active",
                            "task_id": task_id,
                            "monitor_created": was_created,
                            "signature": signature,
                            "first_seen_at": row.get("first_seen_at") or now,
                            "last_seen_at": now,
                        })
                        created += int(was_created)
                        attached += int(not was_created)
                    else:
                        row["last_seen_at"] = now
                    continue

                if row.get("status") == "active" and _two_recent_successes(profile, job_id):
                    task_id = str(row.get("task_id") or "")
                    task = kanban_db.get_task(conn, task_id) if task_id else None
                    if task:
                        kanban_db.add_comment(
                            conn,
                            task_id,
                            "workforce-health-monitor",
                            "Deterministic recovery evidence: the two most recent scheduler executions completed successfully. This proves scheduler-path recovery only; linked business-outcome acceptance remains with its accountable owner.",
                        )
                        if row.get("monitor_created") and task.status in ACTIVE_TASK_STATUSES:
                            kanban_db.complete_task(
                                conn,
                                task_id,
                                result="Recurring scheduler failure cleared after two consecutive completed executions.",
                                summary="Recurring scheduler failure cleared; two consecutive executions completed.",
                            )
                    row.update({"status": "recovered", "recovered_at": now})
                    recovered += 1

        _sync_intake_lifecycle(conn, findings, now, org)

    state["updated_at"] = now
    _write_state(state_path, state)
    return {
        "detected": detected,
        "created": created,
        "attached": attached,
        "recovered": recovered,
        "state": str(state_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organization", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(
        organization=args.organization,
        database=args.database,
        state_path=args.state,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
