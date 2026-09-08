"""Fail-closed capability check for the internal handoff pickup CLI turn."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_ENV_KEYS = (
    "HERMES_COORDINATION_REQUEST_ROOT",
    "HERMES_COORDINATION_TASK_ID",
    "HERMES_COORDINATION_PURPOSE",
    "HERMES_WORKFORCE_HANDOFF_PICKUP_TASK",
    "HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET",
    "HERMES_WORKFORCE_HANDOFF_PICKUP_SOURCE",
    "HERMES_KANBAN_DB",
)

_PICKUP_ENV_KEYS = (
    "HERMES_WORKFORCE_HANDOFF_PICKUP_TASK",
    "HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET",
    "HERMES_WORKFORCE_HANDOFF_PICKUP_SOURCE",
)


def _scope_env() -> dict[str, str] | None:
    values = {key: os.environ.get(key, "") for key in _ENV_KEYS}
    # Coordination roots are inherited by ordinary repair and review workers.
    # Only the three pickup-only fields opt a turn into this capability clamp.
    pickup_present = [key for key in _PICKUP_ENV_KEYS if values[key]]
    if not pickup_present:
        return None
    if any(not values[key] for key in _ENV_KEYS):
        raise ValueError("pickup scope metadata is incomplete")
    return values


def _canonical_agent(value: str) -> str:
    from hermes_cli.workforce_org import load_organization

    candidate = str(value or "").strip().casefold()
    if not candidate or candidate != str(value or "").strip():
        raise ValueError("pickup agent is not canonical")
    return load_organization().validate_execution_profile(candidate).agent


def _active_profile_matches(target: str) -> bool:
    """Require the running profile, not only child-controlled scope fields."""
    from hermes_cli.profiles import get_active_profile_name

    try:
        return (
            _canonical_agent(os.environ.get("HERMES_PROFILE", "")) == target
            and _canonical_agent(get_active_profile_name()) == target
        )
    except Exception:
        return False


def _valid_identifier(value: str, prefix: str) -> bool:
    return (
        value.startswith(prefix)
        and len(value) <= 160
        and all(char.isascii() and (char.isalnum() or char in "_-") for char in value)
    )


def _payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("pickup payload is missing")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("pickup payload is invalid")
    return value


def _durable_claim_matches(scope: dict[str, str]) -> bool:
    """Read fresh task/event state instead of trusting child environment data."""
    from hermes_cli import kanban_db

    db_path = Path(scope["HERMES_KANBAN_DB"])
    if not db_path.is_absolute() or not db_path.is_file():
        return False
    task_id = scope["HERMES_COORDINATION_TASK_ID"]
    root_id = scope["HERMES_COORDINATION_REQUEST_ROOT"]
    target = _canonical_agent(scope["HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET"])
    source = _canonical_agent(scope["HERMES_WORKFORCE_HANDOFF_PICKUP_SOURCE"])
    with kanban_db.connect_closing(db_path) as conn:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            return False
        body = _payload(task.body)
        context = body.get("context")
        if (
            body.get("kind") != "workforce_handoff"
            or body.get("state") != "pending_acknowledgment"
            or body.get("target_agent") != target
            or body.get("source_agent") != source
            or not isinstance(context, dict)
            or context.get("kind") != "owned_operational_failure"
            or getattr(task, "request_root_id", None) != root_id
        ):
            return False
        for event in reversed(kanban_db.list_events(conn, task_id)):
            if event.kind != "workforce_handoff_pickup_claimed":
                continue
            payload = event.payload
            return bool(
                isinstance(payload, dict)
                and payload.get("actor") == target
                and payload.get("target_agent") == target
                and payload.get("source_agent") == source
                and payload.get("request_root_id") == root_id
                and payload.get("claim_kind") == "owned_operational_failure"
            )
    return False


def pickup_scope_denial(name: str, args: dict[str, Any]) -> str | None:
    """Return a reason when an internal pickup turn may not execute a tool."""
    try:
        scope = _scope_env()
        if scope is None:
            return None
        task_id = scope["HERMES_COORDINATION_TASK_ID"]
        root_id = scope["HERMES_COORDINATION_REQUEST_ROOT"]
        if (
            scope["HERMES_COORDINATION_PURPOSE"] != "work"
            or scope["HERMES_WORKFORCE_HANDOFF_PICKUP_TASK"] != task_id
            or not _valid_identifier(task_id, "t_")
            or not _valid_identifier(root_id, "cr_")
            or not _active_profile_matches(
                _canonical_agent(scope["HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET"])
            )
            or not _durable_claim_matches(scope)
        ):
            return "workforce handoff pickup scope is invalid"
        if (
            name != "workforce_handoff"
            or not isinstance(args, dict)
            or args != {"action": "acknowledge", "task_id": task_id}
        ):
            return "only acknowledgment of the claimed workforce handoff is allowed"
        return None
    except Exception:
        return "workforce handoff pickup scope is invalid"
