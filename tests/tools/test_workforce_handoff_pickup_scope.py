"""Host enforcement tests for the internal handoff pickup turn."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

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
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    monkeypatch.setenv("HERMES_PROFILE", "alina")
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
    monkeypatch.setattr(
        "tools.workforce_handoff_pickup_scope._active_profile_matches",
        lambda _target: True,
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
    # Ordinary repair/review workers inherit all coordination fields.  They
    # must remain unrestricted unless a pickup-only field is also present.
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", "cr_normal_123")
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", "t_normal_123")
    monkeypatch.setenv("HERMES_COORDINATION_PURPOSE", "work")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/shared-kanban.db")
    registry, calls = _registry()

    assert json.loads(registry.dispatch("terminal", {})) == {"ok": True}
    assert calls == [("terminal", {})]


def test_pickup_scope_rejects_a_process_running_as_another_profile(monkeypatch):
    _clear_scope(monkeypatch)
    _install_scope(monkeypatch)
    monkeypatch.setattr(
        "tools.workforce_handoff_pickup_scope._durable_claim_matches",
        lambda _scope: True,
    )
    monkeypatch.setattr(
        "tools.workforce_handoff_pickup_scope._active_profile_matches",
        lambda _target: False,
    )
    registry, calls = _registry()

    denied = json.loads(registry.dispatch(
        "workforce_handoff", {"action": "acknowledge", "task_id": "t_pickup_123"}
    ))

    assert denied["error_type"] == "workforce_handoff_pickup_scope_denied"
    assert calls == []


def test_active_pickup_profile_requires_both_profile_env_and_active_home(monkeypatch):
    from tools.workforce_handoff_pickup_scope import _active_profile_matches

    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    monkeypatch.setenv("HERMES_PROFILE", "alina")
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "alina")
    assert _active_profile_matches("alina") is True

    monkeypatch.setenv("HERMES_PROFILE", "aurora")
    assert _active_profile_matches("alina") is False


def test_only_a_validated_pickup_envelope_eagerly_exposes_the_handoff_tool(monkeypatch):
    from agent.agent_init import _pickup_eager_tool_names

    _clear_scope(monkeypatch)
    assert _pickup_eager_tool_names(None) is None
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", "t_pickup_123")
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_TASK", "t_pickup_123")
    monkeypatch.setattr(
        "tools.workforce_handoff_pickup_scope.pickup_scope_denial",
        lambda _name, _args: None,
    )

    eager = _pickup_eager_tool_names({"terminal"})

    assert eager == frozenset({"terminal", "workforce_handoff"})


def test_real_registry_acknowledges_only_a_durably_claimed_pickup(monkeypatch, tmp_path):
    """The host guard admits the real handler only after the factory claim."""
    from hermes_cli import kanban_db
    from hermes_cli.workforce_handoffs import (
        claim_owned_failure_handoff_pickup,
        create_handoff,
    )
    from hermes_cli.workforce_org import load_organization
    import tools.workforce_handoff_tool  # noqa: F401 -- registers the real handler
    from tools.registry import registry as real_registry

    _clear_scope(monkeypatch)
    home = tmp_path / ".hermes"
    (home / "profiles" / "alina").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )
    org = load_organization()
    now = int(time.time())
    iso = lambda offset: datetime.fromtimestamp(now + offset, timezone.utc).isoformat()
    db_path = home / "kanban.db"
    with kanban_db.connect_closing(db_path) as conn:
        created = create_handoff(
            conn,
            source_agent="aurora",
            target_agent="alina",
            expected_outcome="Repair the owned operational failure",
            acceptance_test="A later probe succeeds",
            evidence_references=["execution:failure-1"],
            acknowledgment_deadline=iso(60),
            checkpoint_at=iso(3600),
            organization=org,
            context={
                "kind": "owned_operational_failure",
                "technical_owner": "alina",
                "director": "aurora",
                "workflow_id": "owned-failure-test",
                "event_id": "failure-1",
            },
            requires_source_acceptance=True,
        )
        pickup = claim_owned_failure_handoff_pickup(
            conn, target_agent="alina", organization=org, now=now + 1
        )

    assert pickup is not None
    _install_scope(monkeypatch)
    monkeypatch.setenv("HERMES_COORDINATION_REQUEST_ROOT", pickup["request_root_id"])
    monkeypatch.setenv("HERMES_COORDINATION_TASK_ID", created["task_id"])
    monkeypatch.setenv("HERMES_WORKFORCE_HANDOFF_PICKUP_TASK", created["task_id"])
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_PROFILE", "alina")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home / "profiles" / "alina")
    try:
        accepted = json.loads(real_registry.dispatch(
            "workforce_handoff",
            {"action": "acknowledge", "task_id": created["task_id"]},
        ))
    finally:
        reset_hermes_home_override(token)

    assert accepted.get("success") is True, accepted
    with kanban_db.connect_closing(db_path) as conn:
        assert json.loads(kanban_db.get_task(conn, created["task_id"]).body)["state"] == "accepted"
