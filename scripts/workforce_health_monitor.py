#!/usr/bin/env python3
"""Deterministically turn recurring Hermes Cron failures into owned repair work.

The monitor spends no model tokens while healthy. A failure must recur at least
twice before it is attached to an existing active task or creates one canonical
Aurora triage card. State prevents repeat comments and card fanout. Two later
successful executions close only cards created by this monitor; pre-existing
repair work receives recovery evidence but keeps its own acceptance lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_cli import kanban_db
from hermes_cli.workforce_org import load_organization


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


def _new_intake_events(
    path: Path,
    cursors: dict[str, Any],
    *,
    profile: Path | None = None,
) -> list[dict[str, Any]]:
    """Consume only complete records after a durable per-file byte cursor.

    An inode replacement or truncation restarts from offset zero; exact Kanban
    idempotency then makes replay safe. A torn final write is deliberately not
    advanced, so it is retried on the next monitor tick after the writer
    finishes (or remains ignored until replaced).
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
    return owner.agent, director.agent, missing


def _record_intake_failure(
    conn: sqlite3.Connection,
    *,
    org,
    event: dict[str, Any],
    state_row: dict[str, Any],
) -> tuple[str, bool, bool]:
    """Create one owner card, or a visible configuration incident if invalid."""
    owner, director, ownership_errors = _event_owner(org, event)
    invalid = str(event.get("status")) != "failure" or bool(ownership_errors)
    source_kind = str(event.get("source_kind") or "unknown")
    source_id = str(event.get("source_id") or "unknown")
    workflow_id = str(event.get("workflow_id") or "unknown")
    task_key = _event_key(event)
    existing_id = str(state_row.get("task_id") or "")
    existing = kanban_db.get_task(conn, existing_id) if existing_id else None
    if existing and existing.status in ACTIVE_TASK_STATUSES:
        return existing_id, False, invalid

    episode = int(state_row.get("episode", 0)) + 1
    task_kind = "failure_ownership_configuration" if invalid else "owned_operational_failure"
    body = {
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
    title_prefix = "Configure failure ownership" if invalid else "Repair operational failure"
    task_id = kanban_db.create_task(
        conn,
        title=f"{title_prefix}: {workflow_id} / {source_id}"[:180],
        body=json.dumps(body, indent=2, sort_keys=True),
        assignee="aurora" if invalid else owner,
        created_by="workforce-health-monitor",
        workspace_kind="scratch",
        initial_status="running",
        idempotency_key=f"workforce-health:{task_key}:episode:{episode}",
        max_runtime_seconds=900,
    )
    state_row["episode"] = episode
    kanban_db.add_comment(
        conn,
        task_id,
        "workforce-health-monitor",
        "Durable failure intake accepted: "
        f"workflow={workflow_id}; source={source_kind}:{source_id}; "
        f"technical_owner={owner}; director={director}; "
        f"sanitized_error={_safe_error(event.get('sanitized_error'))}",
    )
    return task_id, True, invalid


def _sync_intake_lifecycle(
    conn: sqlite3.Connection, findings: dict[str, Any], now: int,
) -> int:
    """Project existing Kanban lifecycle events into monitor state.

    The owner acknowledgement is the canonical claim event. The only accepted
    final route is the existing review lifecycle: owner requests review with
    the declared director, then that director's review-assigned task reaches a
    terminal completed state. The monitor records those facts; it never
    completes an owner incident on its own.
    """
    overdue = 0
    for key, row in findings.items():
        if not key.startswith("intake:") or not row.get("task_id"):
            continue
        task = kanban_db.get_task(conn, str(row["task_id"]))
        if task is None:
            continue
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (task.id,),
        ).fetchall()
        owner_acknowledged = any(event["kind"] == "claimed" for event in events)
        director_review = False
        director = str(row.get("director") or "")
        for event in events:
            if event["kind"] != "review_requested":
                continue
            try:
                payload = json.loads(event["payload"] or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict) and payload.get("reviewer") == director:
                director_review = True
        row["owner_acknowledged"] = owner_acknowledged
        row["director_review_requested"] = director_review
        row["director_accepted"] = bool(director_review and task.status == "done")
        deadline = row.get("ack_deadline")
        try:
            expired = not owner_acknowledged and deadline is not None and now > int(deadline)
        except (TypeError, ValueError):
            expired = False
        if expired and not row.get("ack_overdue"):
            kanban_db.add_comment(
                conn, task.id, "workforce-health-monitor",
                "Acknowledgement deadline elapsed without a canonical task claim. "
                "This remains an internal ownership-control exception.",
            )
            row["ack_overdue"] = True
        overdue += int(bool(row.get("ack_overdue")))
    return overdue


def run(*, organization: Path, database: Path, state_path: Path) -> dict[str, Any]:
    org = load_organization(organization, validate_profiles=True)
    state = _read_json(state_path, {"schema_version": 1, "findings": {}})
    findings = state.setdefault("findings", {})
    cursors = state.setdefault("intake_cursors", {})
    now = int(time.time())
    detected = attached = created = recovered = invalid_ownership = 0

    intake: list[dict[str, Any]] = _new_intake_events(
        state_path.parent / INTAKE_FILENAME, cursors,
    )
    for agent in org.operational_agents(include_planned=False):
        profile = Path(str(agent.profile_path))
        intake.extend(_new_intake_events(
            profile / "cron" / INTAKE_FILENAME, cursors, profile=profile,
        ))

    with kanban_db.connect_closing(database) as conn:
        # Intake is authoritative for opted-in jobs. The append cursor makes
        # old failure history inert after it has been consumed once; a later
        # recovery is only meaningful after a failure in that same source.
        for event in intake:
            key = _event_key(event)
            row = findings.setdefault(key, {"episode": 0, "status": "healthy"})
            signature = str(event.get("dedupe_key") or event.get("event_id") or "")
            offset = int(event.get("_intake_offset") or 0)
            is_recovery = str(event.get("status") or "") == "recovered"
            if is_recovery and row.get("status") == "active" and offset > int(
                row.get("last_failure_offset") or -1
            ):
                task_id = str(row.get("task_id") or "")
                if task_id:
                    kanban_db.add_comment(
                        conn,
                        task_id,
                        "workforce-health-monitor",
                        "Scheduler-path recovery evidence: the two most recent executions completed successfully. The owner-assigned repair remains open pending technical-owner acknowledgement and director acceptance.",
                    )
                row.update({
                    "status": "recovered_pending_acceptance",
                    "recovered_at": now,
                    "last_recovery_offset": offset,
                })
                recovered += 1
                continue
            if is_recovery:
                # A success without a known failure is normal operating
                # evidence, not an incident or a reason to wake anyone.
                continue
            if row.get("status") == "active" and row.get("signature") == signature:
                row["last_seen_at"] = now
                continue
            task_id, was_created, was_invalid = _record_intake_failure(
                conn, org=org, event=event, state_row=row
            )
            row.update({
                "status": "active",
                "task_id": task_id,
                "monitor_created": was_created,
                "signature": signature,
                "first_seen_at": row.get("first_seen_at") or now,
                "last_seen_at": now,
                "last_failure_offset": offset,
                "director": str(event.get("director") or ""),
                "ack_deadline": event.get("ack_deadline"),
            })
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

        overdue_acknowledgements = _sync_intake_lifecycle(conn, findings, now)

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
