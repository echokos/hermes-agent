"""Host enforcement tests for the internal handoff pickup turn."""

from __future__ import annotations

import json

from tools.registry import ToolRegistry


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


def _clear_scope(monkeypatch) -> None:
    from tools.workforce_handoff_pickup_scope import _ENV_KEYS

    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _install_scope(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", "cr_pickup_123")
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", "t_pickup_123")
    monkeypatch.setenv("HERMES_COORDINATION_PURPOSE", "work")
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_TASK", "t_pickup_123")
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET", "alina")
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_SOURCE", "aurora")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/pickup-test.db")


def _registry() -> tuple[ToolRegistry, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []
    registry = ToolRegistry()
    for name in ("workforce_handoff", "terminal"):
        registry.register(
            name=name,
            toolset="workforce",
            schema=_schema(name),
            handler=lambda args, _name=name, **_kwargs: (
                calls.append((_name, dict(args))), json.dumps({"ok": True})
            )[1],
        )
    return registry, calls


def test_pickup_scope_allows_only_the_exact_claimed_acknowledgment(monkeypatch):
    _clear_scope(monkeypatch)
    _install_scope(monkeypatch)
    monkeypatch.setattr(
        "tools.workforce_handoff_pickup_scope._durable_claim_matches",
        lambda _scope: True,
    )
    registry, calls = _registry()

    accepted = json.loads(registry.dispatch(
        "workforce_handoff", {"action": "acknowledge", "task_id": "t_pickup_123"}
    ))
    denied_tool = json.loads(registry.dispatch("terminal", {}))
    denied_action = json.loads(registry.dispatch(
        "workforce_handoff", {"action": "sweep", "task_id": "t_pickup_123"}
    ))
    denied_extra = json.loads(registry.dispatch(
        "workforce_handoff",
        {"action": "acknowledge", "task_id": "t_pickup_123", "evidence_references": []},
    ))

    assert accepted == {"ok": True}
    assert calls == [("workforce_handoff", {"action": "acknowledge", "task_id": "t_pickup_123"})]
    for denied in (denied_tool, denied_action, denied_extra):
        assert denied["error_type"] == "workforce_handoff_pickup_scope_denied"


def test_partial_pickup_metadata_fails_closed_for_every_tool(monkeypatch):
    _clear_scope(monkeypatch)
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_TASK", "t_pickup_123")
    registry, calls = _registry()

    denied = json.loads(registry.dispatch("terminal", {}))

    assert denied["error_type"] == "workforce_handoff_pickup_scope_denied"
    assert calls == []


def test_normal_sessions_are_unchanged_without_pickup_metadata(monkeypatch):
    _clear_scope(monkeypatch)
    # A gateway commonly pins its board DB; that alone must not enable this
    # pickup-only scope guard.
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/shared-kanban.db")
    registry, calls = _registry()

    assert json.loads(registry.dispatch("terminal", {})) == {"ok": True}
    assert calls == [("terminal", {})]
