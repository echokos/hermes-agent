"""Receipt-backed delivery of source-reviewed operational outcomes."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
from pathlib import Path

from hermes_cli import kanban_db as kb


def operational_outcome_profiles(profiles: set[str]) -> set[str]:
    path = kb.kanban_db_path(kb.DEFAULT_BOARD).resolve()
    if not profiles or not path.exists():
        return set()
    with kb.connect_closing(path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT CASE WHEN json_valid(t.body) THEN "
            "json_extract(t.body,'$.context.outcome_notice.source_profile') END AS profile "
            "FROM tasks t WHERE EXISTS (SELECT 1 FROM task_events e WHERE e.task_id=t.id "
            "AND e.kind IN ('workforce_handoff_decision_accepted','completed'))",
        ).fetchall()
    return {row["profile"] for row in rows if row["profile"] in profiles}


def _source_event(home: Path, context: dict, notice: dict) -> bool:
    path = home / "cron" / "operational-failures.jsonl"
    try:
        with path.open("rb") as stream:
            # Intake is append-only. Bound each record without loading the archive.
            total = 0
            for _ in range(100_000):
                line = stream.readline(65_537)
                if not line:
                    return False
                if len(line) > 65_536:
                    return False
                total += len(line)
                if total > 16 * 1024 * 1024:
                    return False
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict) or event.get("event_id") != context.get("event_id"):
                    continue
                return (
                    event.get("source_kind") == "profile_cron"
                    and event.get("source_scope") == notice["source_profile"]
                    and event.get("source_id") == notice["source_id"]
                    and event.get("execution_id") == notice["execution_id"]
                    and event.get("outcome_notice") == notice
                )
    except OSError:
        pass
    return False


def _canonical_outcomes(conn, task_id: str, profile: str) -> list[dict]:
    from cron.jobs import get_job
    from cron.operational_outcomes import validate_outcome_notice
    from hermes_constants import get_hermes_home

    task = kb.get_task(conn, task_id)
    if task is None:
        return []
    try:
        context = json.loads(task.body)["context"]
        accepted = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "AND kind='coordination_internal_request_accepted' ORDER BY id LIMIT 1", (task_id,),
        ).fetchone()
        if accepted is None or json.loads(accepted["payload"]).get("failure_event_id") != context.get("event_id"):
            return []
        notice = context["outcome_notice"]
        source = context["source"]
        if source != {"kind": "profile_cron", "scope": profile, "id": notice["source_id"]}:
            return []
        if notice["source_profile"] != profile or not _source_event(get_hermes_home(), context, notice):
            return []
        job = get_job(notice["source_id"])
        if job is None:
            return []
        validate_outcome_notice(notice, job, source_profile=profile)
        snapshot = kb.owned_failure_recovery_outcome_snapshot(conn, task_id)
        if snapshot is not None:
            content = "Operational recovery verified. The routine's technical owner supplied recovery evidence and its director accepted the repair."
        else:
            snapshot = kb.owned_failure_decision_outcome_snapshot(conn, task_id)
            if snapshot is None or snapshot["outcome"] != "accepted":
                return []
            decision = snapshot["reserved_decision"]
            content = (
                f"Action required: {decision['integration']}\n"
                f"Account: {decision['account']}\n{decision['action']}\n\n"
                "This action was reviewed by the routine's director. The incident is not repaired; no authorization has been granted on your behalf."
            )
        if len(content) > 1800:
            return []
        label = str(job.get("name") or job["id"])
        label = re.sub(r"[^A-Za-z0-9 _()./-]", "", label)[:96].strip()
        content = f"Routine: {label}\n\n{content}"
        return [{
            "request_root_id": snapshot["request_root_id"], "task_id": task_id,
            "event_id": snapshot["event_id"], "execution_profile": profile,
            "route": route, "content": content,
        } for route in notice["routes"]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return []


def validate_operational_outcome_authority(outcome: dict) -> dict:
    """Rebuild authority from canonical state immediately before the send claim."""
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home

    profile = outcome.get("execution_profile")
    if not isinstance(profile, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", profile):
        raise ValueError("invalid operational outcome profile")
    if get_hermes_home().resolve() != get_profile_dir(profile).resolve():
        raise ValueError("operational outcome profile scope mismatch")
    with kb.connect_closing(kb.kanban_db_path(kb.DEFAULT_BOARD).resolve()) as conn:
        for bound in _canonical_outcomes(conn, outcome.get("task_id"), profile):
            if bound == outcome:
                return bound
    raise ValueError("operational outcome no longer has canonical authority")


def collect_operational_outcomes(profile: str, after: int = 0) -> tuple[list[dict], int]:
    """Page fairly through terminal events; the existing outbox owns deduplication."""
    from gateway.delivery_ledger import get_operational_outcome_delivery

    path = kb.kanban_db_path(kb.DEFAULT_BOARD).resolve()
    if not path.exists():
        return [], 0
    with kb.connect_closing(path) as conn:
        rows = conn.execute(
            "SELECT e.id, e.task_id FROM task_events e JOIN tasks t ON t.id=e.task_id "
            "WHERE e.id>? AND e.kind IN ('workforce_handoff_decision_accepted','completed') "
            "AND CASE WHEN json_valid(t.body) THEN json_extract(t.body,'$.context.outcome_notice.source_profile') END=? "
            "ORDER BY e.id LIMIT 64", (after, profile),
        ).fetchall()
        result = []
        for row in rows:
            for outcome in _canonical_outcomes(conn, row["task_id"], profile):
                if outcome["event_id"] != row["id"]:
                    continue
                record = get_operational_outcome_delivery(
                    outcome["request_root_id"], outcome["event_id"], outcome["route"]["route_key"],
                )
                if record is None or record.state in {"pending", "sending"}:
                    result.append(outcome)
        return result, rows[-1]["id"] if len(rows) == 64 else 0


async def deliver_operational_outcome(outcome: dict, adapter) -> str:
    from gateway.delivery_ledger import (
        acknowledge_operational_outcome_delivery, claim_operational_outcome_delivery,
        settle_operational_outcome_send,
    )
    from gateway.platforms.base import SendResult

    outcome = deepcopy(outcome)
    record = await asyncio.to_thread(claim_operational_outcome_delivery, outcome)
    if not record.send_claimed:
        return record.state
    route = outcome["route"]
    identity = (outcome["request_root_id"], outcome["event_id"], route["route_key"])
    try:
        result = await asyncio.wait_for(adapter.send(
            chat_id=route["chat_id"], content=outcome["content"],
            metadata={"thread_id": route["thread_id"]} if route["thread_id"] else None,
        ), timeout=15)
    except BaseException:
        settle_operational_outcome_send(*identity, claim_token=record.claim_token, error="send_interrupted")
        raise
    if isinstance(result, SendResult) and result.success is True and result.message_id:
        receipt = f"platform-message:{route['platform']}:{result.message_id}"
        acknowledge_operational_outcome_delivery(*identity, returned_message_id=receipt)
        return "acknowledged"
    rejected = isinstance(result, SendResult) and result.success is False and result.error_kind in {
        "rate_limited", "bad_format", "forbidden", "not_found", "too_long",
    }
    return settle_operational_outcome_send(
        *identity, claim_token=record.claim_token, definitely_rejected=rejected,
        error="send_rejected" if rejected else "missing_platform_receipt",
    ).state
