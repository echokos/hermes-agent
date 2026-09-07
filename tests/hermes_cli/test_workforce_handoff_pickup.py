"""CLI-process boundaries for owned workforce handoff pickup."""

from __future__ import annotations

from hermes_cli.workforce_handoff_pickup import _pickup_command, _pickup_env


def test_pickup_command_is_one_turn_tool_sourced_workforce_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup._resolve_hermes_argv",
        lambda: ["hermes"],
    )

    command = _pickup_command(
        target_agent="alina", request_root_id="cr_pickup_123", task_id="t_pickup_123"
    )

    assert command[:5] == ["hermes", "-p", "alina", "--cli", "chat"]
    assert "-Q" in command
    assert command[command.index("--max-turns") + 1] == "1"
    assert command[command.index("-t") + 1] == "workforce"
    assert command[command.index("-c") + 1] == "workforce-handoff:cr_pickup_123:alina"


def test_pickup_env_scrubs_worker_identity_and_sets_exact_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "unrelated-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "9")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "stale-lock")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "telegram")
    monkeypatch.setattr(
        "hermes_cli.workforce_handoff_pickup.resolve_profile_env", lambda _target: "/profiles/alina"
    )

    env = _pickup_env(
        database_path=tmp_path / "kanban.db",
        task_id="t_pickup_123",
        request_root_id="cr_pickup_123",
        target_agent="alina",
        source_agent="aurora",
    )

    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env
    assert env["HERMES_SESSION_SOURCE"] == "tool"
    assert env["HERMES_COORDINATION_REQUEST_ROOT"] == "cr_pickup_123"
    assert env["HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET"] == "alina"
