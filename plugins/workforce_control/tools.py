"""Agent-facing tools for controlled planning, reconciliation, and learning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from agent.coordination_budget import (
    coordination_materialization_binding,
    register_uncoordinated_materialization,
)
from hermes_cli import kanban_db
from hermes_cli.workforce_org import active_workforce_agent
from plugins.workforce_control.store import (
    apply_reconciliation,
    complete_vision_review,
    current_goal_snapshot,
    list_vision_reviews,
    materialize_plan,
    publish_goal_snapshot,
    propose_reconciliation,
    record_correction,
    record_plan,
    request_vision_review,
    runtime_state,
)
from tools.registry import tool_error, tool_result


def _actor() -> str:
    return active_workforce_agent().agent


def _enabled() -> bool:
    try:
        return active_workforce_agent().operational
    except Exception:
        return False


def _plan(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        with kanban_db.connect_closing() as conn:
            result = record_plan(conn, actor=_actor(), payload=args)
        return tool_result(success=True, **result)
    except Exception as exc:
        return tool_error(str(exc))


def _materialize(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        with coordination_materialization_binding() as (
            coordination_context,
            coordination_origin,
        ):
            with kanban_db.connect_closing() as conn:
                result = materialize_plan(
                    conn, actor=_actor(), plan_id=str(args.get("plan_id") or ""),
                    current_state_evidence=list(args.get("current_state_evidence") or []),
                    current_state_evidence_at=args.get("current_state_evidence_at"),
                    confirmed_execution_ready=bool(args.get("confirmed_execution_ready")),
                    coordination_context=coordination_context,
                    coordination_origin=coordination_origin,
                )
                register_uncoordinated_materialization(
                    created=result.get("created") is True,
                    request_root_id=result.get("request_root_id"),
                )
        return tool_result(success=True, **result)
    except Exception as exc:
        return tool_error(str(exc))


def _reconcile(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        action = str(args.get("action") or "propose")
        with kanban_db.connect_closing() as conn:
            if action == "status":
                result: Any = runtime_state(conn)
            elif action in {"shadow", "propose"}:
                result = propose_reconciliation(
                    conn, actor=_actor(), observations=list(args.get("observations") or []),
                    mode="shadow" if action == "shadow" else "proposed",
                )
            elif action == "apply":
                result = apply_reconciliation(conn, actor=_actor(), action_ids=list(args.get("action_ids") or []))
            else:
                raise ValueError("action must be status, shadow, propose, or apply")
        return tool_result(success=True, action=action, result=result)
    except Exception as exc:
        return tool_error(str(exc))


def _correct(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        with kanban_db.connect_closing() as conn:
            result = record_correction(
                conn, actor=_actor(), classification=str(args.get("classification") or ""),
                scope=str(args.get("scope") or ""), description=str(args.get("description") or ""),
                provenance_ref=str(args.get("provenance_ref") or ""),
                privacy_class=str(args.get("privacy_class") or "organizational"),
                rule_target=args.get("rule_target"), regression_ref=args.get("regression_ref"),
                supersedes_id=args.get("supersedes_id"),
            )
        return tool_result(success=True, **result)
    except Exception as exc:
        return tool_error(str(exc))


def _goals(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        action = str(args.get("action") or "read")
        with kanban_db.connect_closing() as conn:
            if action == "read":
                snapshot = current_goal_snapshot(
                    conn, max_age_hours=int(args.get("max_age_hours") or 36)
                )
                return tool_result(success=True, action=action, snapshot=snapshot)
            if action != "publish":
                raise ValueError("action must be read or publish")
            result = publish_goal_snapshot(
                conn,
                actor=_actor(),
                source_guid=str(args.get("source_guid") or ""),
                source_title=str(args.get("source_title") or ""),
                source_updated_at=args.get("source_updated_at"),
                goals=list(args.get("goals") or []),
            )
        return tool_result(success=True, action=action, **result)
    except Exception as exc:
        return tool_error(str(exc))


def _vision(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        action = str(args.get("action") or "list")
        with kanban_db.connect_closing() as conn:
            if action == "list":
                result: Any = list_vision_reviews(
                    conn,
                    status=str(args.get("status") or "pending"),
                    limit=int(args.get("limit") or 10),
                )
            elif action == "request":
                result = request_vision_review(
                    conn,
                    actor=_actor(),
                    source_ref=str(args.get("source_ref") or ""),
                    goal_ref=str(args.get("goal_ref") or ""),
                    brief=str(args.get("brief") or ""),
                    evidence_references=list(args.get("evidence_references") or []),
                )
            elif action == "respond":
                result = complete_vision_review(
                    conn,
                    actor=_actor(),
                    review_id=str(args.get("review_id") or ""),
                    response={
                        "reframe": args.get("reframe"),
                        "ten_x_option": args.get("ten_x_option"),
                        "assumptions": list(args.get("assumptions") or []),
                        "value_case": args.get("value_case"),
                        "risks": list(args.get("risks") or []),
                        "smallest_test": args.get("smallest_test"),
                    },
                )
            else:
                raise ValueError("action must be list, request, or respond")
        return tool_result(success=True, action=action, result=result)
    except Exception as exc:
        return tool_error(str(exc))


def _buzz_events(
    *, lookback_minutes: int, per_room_limit: int, max_events: int
) -> dict[str, Any]:
    """Read a small, metadata-safe window from the active profile's watched rooms."""
    from hermes_cli.config import load_config_readonly
    from plugins.platforms.buzz.adapter import _configured_channels, _resolve_private_key

    config = load_config_readonly()
    buzz = (((config.get("gateway") or {}).get("platforms") or {}).get("buzz") or {})
    extra = buzz.get("extra") or {}
    channel_ids = _configured_channels(extra)
    if not channel_ids:
        raise RuntimeError("the active profile has no watched Buzz rooms")
    cli = str(extra.get("cli_path") or "/home/elliott/.local/bin/buzz")
    if not Path(cli).is_file():
        raise RuntimeError("Buzz CLI is unavailable")
    private_key = _resolve_private_key(extra)
    if not private_key:
        raise RuntimeError("the active profile has no usable Buzz identity")
    relay = str(extra.get("relay_url") or "").strip()
    if not relay:
        raise RuntimeError("the active profile has no Buzz relay URL")
    env = os.environ.copy()
    env["BUZZ_RELAY_URL"] = relay
    env["BUZZ_PRIVATE_KEY"] = private_key

    names: dict[str, str] = {}
    listed = subprocess.run(
        [cli, "--format", "compact", "channels", "list"],
        text=True, capture_output=True, timeout=30, env=env, check=False,
    )
    if listed.returncode == 0:
        try:
            payload = json.loads(listed.stdout)
            if isinstance(payload, dict):
                payload = payload.get("channels") or payload.get("items") or []
            for item in payload if isinstance(payload, list) else []:
                channel_id = str(item.get("id") or item.get("channel_id") or "")
                if channel_id:
                    names[channel_id] = str(item.get("name") or item.get("title") or channel_id)
        except (TypeError, ValueError):
            pass

    since = int(time.time()) - max(15, min(int(lookback_minutes), 360)) * 60
    events: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for channel_id in channel_ids:
        completed = subprocess.run(
            [
                cli, "--format", "compact", "messages", "get",
                "--channel", channel_id, "--since", str(since),
                "--limit", str(max(1, min(int(per_room_limit), 10))),
                "--kinds", "9",
            ],
            text=True, capture_output=True, timeout=30, env=env, check=False,
        )
        if completed.returncode != 0:
            errors.append({"room": names.get(channel_id, channel_id), "error": "read_failed"})
            continue
        try:
            payload = json.loads(completed.stdout)
            if isinstance(payload, dict):
                payload = payload.get("messages") or payload.get("events") or payload.get("items") or []
        except (TypeError, ValueError):
            errors.append({"room": names.get(channel_id, channel_id), "error": "invalid_response"})
            continue
        for item in payload if isinstance(payload, list) else []:
            if int(item.get("kind") or 9) != 9:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            events.append({
                "room": names.get(channel_id, channel_id),
                "room_id": channel_id,
                "event_id": str(item.get("id") or ""),
                "created_at": int(item.get("created_at") or 0),
                "author": str(item.get("display_name") or item.get("name") or item.get("pubkey") or "unknown")[:96],
                "content": content[:600],
            })
    events.sort(key=lambda value: (value["created_at"], value["room_id"]))
    bounded_max = max(1, min(int(max_events), 20))
    return {
        "since": since,
        "rooms_checked": len(channel_ids),
        "events": events[-bounded_max:],
        "errors": errors,
    }


def _observe_buzz(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        actor = _actor()
        if actor not in {"chloe", "milena", "aurora"}:
            raise PermissionError(
                "Buzz workforce observation is restricted to Chloe, Milena, and Aurora"
            )
        result = _buzz_events(
            lookback_minutes=int(args.get("lookback_minutes") or 180),
            per_room_limit=int(args.get("per_room_limit") or 6),
            max_events=int(args.get("limit") or 20),
        )
        return tool_result(success=True, **result)
    except Exception as exc:
        return tool_error(str(exc))


PLAN_SCHEMA = {
    "name": "workforce_plan", "description": "Draft one goal-aligned outcome and bounded execution graph without creating execution cards.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "goal_ref": {"type": "string"}, "goal_evidence_at": {"type": "string"},
        "desired_outcome": {"type": "string"}, "acceptance_test": {"type": "string"},
        "priority_rationale": {"type": "string"}, "checkpoint": {"type": "string"},
        "capacity_assessment": {"type": "string"}, "deadline_dependencies": {"type": "string"},
        "displaced_work": {"type": "string"}, "unresolved_decisions": {"type": "array", "items": {"type": "string"}},
        "defer_or_stop": {"type": "string"}, "evidence_references": {"type": "array", "items": {"type": "string"}},
        "nodes": {"type": "array", "maxItems": 8, "items": {"type": "object"}},
    }, "required": ["title","goal_ref","desired_outcome","acceptance_test","priority_rationale","checkpoint","capacity_assessment","deadline_dependencies","displaced_work","unresolved_decisions","defer_or_stop","nodes"], "additionalProperties": False},
}

MATERIALIZE_SCHEMA = {
    "name": "workforce_materialize", "description": "After current-state revalidation, atomically materialize an approved Aurora draft. Reserved actions remain gated.",
    "parameters": {"type": "object", "properties": {
        "plan_id": {"type": "string"}, "current_state_evidence": {"type": "array", "items": {"type": "string"}},
        "current_state_evidence_at": {"type": "string"}, "confirmed_execution_ready": {"type": "boolean"},
    }, "required": ["plan_id","current_state_evidence","current_state_evidence_at","confirmed_execution_ready"], "additionalProperties": False},
}

RECONCILE_SCHEMA = {
    "name": "workforce_reconcile", "description": "Record shadow/proposed current-state dispositions or apply a high-confidence Aurora-approved transition.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["status","shadow","propose","apply"]},
        "observations": {"type": "array", "items": {"type": "object"}},
        "action_ids": {"type": "array", "items": {"type": "string"}},
    }, "required": ["action"], "additionalProperties": False},
}

CORRECT_SCHEMA = {
    "name": "workforce_correct", "description": "Record a correction with provenance, privacy, scope, precedence, and regression linkage.",
    "parameters": {"type": "object", "properties": {
        "classification": {"type": "string"}, "scope": {"type": "string"}, "description": {"type": "string"},
        "provenance_ref": {"type": "string"}, "privacy_class": {"type": "string", "enum": ["organizational","personal_private","relationship_private"]},
        "rule_target": {"type": "string"}, "regression_ref": {"type": "string"}, "supersedes_id": {"type": "string"},
    }, "required": ["classification","scope","description","provenance_ref","privacy_class"], "additionalProperties": False},
}

GOALS_SCHEMA = {
    "name": "workforce_goals",
    "description": "Read the current workforce-safe goals projection, or let Aurora publish a verified projection from the canonical Evernote note.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["read", "publish"]},
        "max_age_hours": {"type": "integer", "minimum": 1, "maximum": 168},
        "source_guid": {"type": "string"}, "source_title": {"type": "string"},
        "source_updated_at": {"type": ["string", "integer"]},
        "goals": {"type": "array", "maxItems": 24, "items": {"type": "object", "properties": {
            "goal_id": {"type": "string"}, "title": {"type": "string"},
            "desired_outcome": {"type": "string"}, "priority": {"type": "string"},
            "status": {"type": "string"},
            "departments": {"type": "array", "items": {"type": "string"}},
        }, "required": ["goal_id", "title", "desired_outcome"], "additionalProperties": False}},
    }, "required": ["action"], "additionalProperties": False},
}

VISION_SCHEMA = {
    "name": "workforce_vision",
    "description": "Operate Mel's bounded 10x end-layer. Aurora requests a review; Mel returns a provocation and smallest test. This never approves or launches work.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["list", "request", "respond"]},
        "status": {"type": "string", "enum": ["pending", "completed", "cancelled"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        "source_ref": {"type": "string"}, "goal_ref": {"type": "string"},
        "brief": {"type": "string"},
        "evidence_references": {"type": "array", "items": {"type": "string"}},
        "review_id": {"type": "string"}, "reframe": {"type": "string"},
        "ten_x_option": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "value_case": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "smallest_test": {"type": "string"},
    }, "required": ["action"], "additionalProperties": False},
}

BUZZ_OBSERVE_SCHEMA = {
    "name": "workforce_observe_buzz",
    "description": "Read a bounded recent window from the active Chloe, Milena, or Aurora profile's configured Buzz rooms for factual friction, contradiction, commitment, and cross-functional-gap observation. This does not post or mutate anything.",
    "parameters": {"type": "object", "properties": {
        "lookback_minutes": {"type": "integer", "minimum": 15, "maximum": 360},
        "per_room_limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
    }, "additionalProperties": False},
}

TOOLS = (
    ("workforce_plan", PLAN_SCHEMA, _plan, _enabled, "🧭"),
    ("workforce_materialize", MATERIALIZE_SCHEMA, _materialize, _enabled, "🏗️"),
    ("workforce_reconcile", RECONCILE_SCHEMA, _reconcile, _enabled, "🔄"),
    ("workforce_correct", CORRECT_SCHEMA, _correct, _enabled, "🧠"),
    ("workforce_goals", GOALS_SCHEMA, _goals, _enabled, "🎯"),
    ("workforce_vision", VISION_SCHEMA, _vision, _enabled, "🔭"),
    ("workforce_observe_buzz", BUZZ_OBSERVE_SCHEMA, _observe_buzz, _enabled, "👀"),
)
