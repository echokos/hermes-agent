"""Restricted, triage-only proactive workforce opportunity intake."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli import kanban_db
from hermes_cli.workforce_org import (
    WorkforceOrganizationError,
    active_workforce_agent,
)
from plugins.workforce_control.store import record_signal
from tools.registry import registry, tool_error, tool_result


def _enabled() -> bool:
    try:
        source = active_workforce_agent()
        return (
            source.operational
            and source.status in {"active", "planned"}
            and source.agent != "mel"
        )
    except Exception:
        return False


def _required_text(args: dict[str, Any], name: str) -> str:
    value = str(args.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _preflight(args: dict[str, Any]):
    """Reject impossible records before the bounded-write reservation."""
    source = active_workforce_agent()
    if not source.operational or source.status not in {"active", "planned"}:
        raise PermissionError(f"{source.agent} is not eligible to submit workforce signals")
    if source.agent == "mel":
        raise PermissionError("Mel may develop alternatives but may not route or launch work")
    _required_text(args, "expected_outcome")
    _required_text(args, "observation")
    if source.agent == "chloe":
        if not str(args.get("aurora_assignment_id") or "").strip():
            raise ValueError("Chloe requires an explicit aurora_assignment_id for mechanical intake")
        if str(args.get("department_recommendation") or "").strip():
            raise ValueError("Chloe may record facts but may not provide a recommendation")
    else:
        _required_text(args, "estimated_effort")
        _required_text(args, "department_recommendation")
    return source


def _handle(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        source = _preflight(args)
        recommendation = str(args.get("department_recommendation") or "").strip()
        aurora_assignment_id = str(args.get("aurora_assignment_id") or "").strip()
        if source.agent == "chloe":
            if not aurora_assignment_id:
                raise ValueError("Chloe requires an explicit aurora_assignment_id for mechanical intake")
            if recommendation:
                raise ValueError("Chloe may record facts but may not provide a recommendation")
            recommendation = "not provided; mechanical record under Aurora direction"
        elif not recommendation:
            raise ValueError("department_recommendation is required")
        packet = {
            "kind": "workforce_signal",
            "decision_owner": "aurora",
            "launch_authorized": False,
            "source_agent": source.agent,
            "source_department": source.department,
            "expected_outcome": _required_text(args, "expected_outcome"),
            "approved_goal": str(args.get("approved_goal") or "unknown").strip() or "unknown",
            "observation": _required_text(args, "observation"),
            "evidence_references": list(args.get("evidence_references") or []),
            "estimated_effort": (
                "not applicable; factual observation under Aurora direction"
                if source.agent == "chloe"
                else _required_text(args, "estimated_effort")
            ),
            "dependencies": list(args.get("dependencies") or []),
            "risks": list(args.get("risks") or []),
            "needed_capabilities": list(args.get("needed_capabilities") or []),
            "department_recommendation": recommendation,
            "aurora_assignment_id": aurora_assignment_id or None,
        }
        with kanban_db.connect_closing() as conn:
            recorded = record_signal(
                conn,
                source_agent=source.agent,
                expected_outcome=packet["expected_outcome"],
                goal_ref=packet["approved_goal"],
                observation=packet["observation"],
                evidence_references=packet["evidence_references"],
                action_class=str(args.get("action_class") or "opportunity"),
                target_ref=str(args.get("target_ref") or ""),
                packet=packet,
            )
        from tools.workforce_signal_runtime import mark_success
        mark_success()
        return tool_result(
            success=True, signal_id=recorded["task_id"], status=recorded["status"],
            assignee=recorded["assignee"], decision_owner="aurora",
            source_agent=source.agent, launch_authorized=False,
            duplicate_key=recorded["stable_key"], created=recorded["created"],
        )
    except (ValueError, WorkforceOrganizationError, OSError) as exc:
        from tools.workforce_signal_runtime import mark_failure
        mark_failure(str(exc))
        return tool_error(str(exc))


WORKFORCE_SIGNAL_SCHEMA = {
    "name": "workforce_signal",
    "description": (
        "Record a concrete observation for Aurora's triage. Chloe must submit "
        "only directed facts with an aurora_assignment_id and no recommendation. "
        "This never approves, prioritizes, dispatches, or launches work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expected_outcome": {"type": "string"},
            "approved_goal": {"type": "string", "default": "unknown"},
            "observation": {"type": "string"},
            "evidence_references": {"type": "array", "items": {"type": "string"}},
            "estimated_effort": {"type": "string"},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "needed_capabilities": {"type": "array", "items": {"type": "string"}},
            "department_recommendation": {"type": "string"},
            "aurora_assignment_id": {"type": "string"},
            "action_class": {"type": "string", "default": "opportunity"},
            "target_ref": {"type": "string"},
        },
        "required": ["expected_outcome", "observation"],
        "additionalProperties": False,
    },
}


registry.register(
    name="workforce_signal", toolset="workforce",
    schema=WORKFORCE_SIGNAL_SCHEMA, handler=_handle,
    check_fn=_enabled, emoji="📡",
    preflight=_preflight,
)
