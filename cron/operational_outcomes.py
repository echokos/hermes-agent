"""Trusted delivery bindings for reviewed operational outcomes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


_PROFILE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ORIGIN_FIELDS = ("platform", "chat_id", "thread_id", "user_id", "user_id_alt", "chat_type", "scope_id")
_ROUTE_FIELDS = ("platform", "chat_id", "thread_id")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identifier(value: Any, *, optional: bool = False) -> str:
    if value is None and optional:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid operational outcome routing identifier")
    result = str(value).strip()
    if (not result and not optional) or len(result) > 256 or any(
        ord(char) < 32 for char in result
    ):
        raise ValueError("invalid operational outcome routing identifier")
    return result


def capture_outcome_notice(
    job: dict[str, Any], *, source_profile: str, execution_id: str,
) -> dict[str, Any] | None:
    """Capture the persisted job's resolved destinations under its profile scope.

    No caller-supplied destination is accepted. Unroutable opt-ins remain
    explicit intake evidence rather than breaking operational failure intake.
    """
    ownership = job.get("failure_ownership")
    if not isinstance(ownership, dict) or ownership.get("return_outcome_to_origin") is not True:
        return None
    try:
        profile = _identifier(source_profile)
        if not _PROFILE.fullmatch(profile):
            raise ValueError("invalid operational outcome source profile")
        source_id = _identifier(job.get("id"))
        execution = _identifier(execution_id)
        from cron.scheduler import _resolve_delivery_targets, _resolve_origin

        origin = _resolve_origin(job) or {}
        origin_fields = {
            key: _identifier(origin.get(key), optional=True)
            for key in _ORIGIN_FIELDS
        }
        resolved = _resolve_delivery_targets(job)
        if not resolved or len(resolved) > 4:
            raise ValueError("operational outcome requires one to four concrete destinations")
        routes = []
        for target in resolved:
            route = {
                key: _identifier(target.get(key), optional=key == "thread_id")
                for key in _ROUTE_FIELDS
            }
            route["platform"] = route["platform"].lower()
            if route["platform"] in {"local", "api_server", "cli", "terminal"}:
                raise ValueError("operational outcome requires receipt-bearing push delivery")
            route["route_key"] = _fingerprint(route)
            if route not in routes:
                routes.append(route)
        routes.sort(key=lambda route: route["route_key"])
        return {
            "version": 1,
            "status": "bound",
            "source_profile": profile,
            "source_id": source_id,
            "execution_id": execution,
            "routes": routes,
            "routing_fingerprint": _fingerprint({
                "deliver": job.get("deliver", "local"),
                "origin": origin_fields,
                "technical_owner": ownership.get("technical_owner"),
                "director": ownership.get("director"),
                "routes": routes,
            }),
        }
    except (TypeError, ValueError, KeyError):
        return {"version": 1, "status": "unroutable", "reason": "invalid_or_missing_source_route"}


def validate_outcome_notice(
    notice: Any, job: dict[str, Any], *, source_profile: str,
) -> dict[str, Any]:
    """Refuse stale or re-routed delivery instead of guessing another recipient."""
    if not isinstance(notice, dict) or notice.get("status") != "bound":
        raise ValueError("operational outcome has no bound source route")
    current = capture_outcome_notice(
        job, source_profile=source_profile,
        execution_id=notice.get("execution_id"),
    )
    if current != notice:
        raise ValueError("operational outcome source route changed")
    return current
