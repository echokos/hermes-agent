"""Regression coverage for canonical workforce assignees in Kanban dispatch."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def canonical_root_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create root→main and personal-only canonical workforce identities."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "main").mkdir(parents=True)
    (home / "profiles" / "amy").mkdir(parents=True)
    organization = tmp_path / "organization.yaml"
    organization.write_text(
        """
schema_version: 1
agents:
  - agent: elliott
    display_name: Elliott
    status: friend
    operational: false
    department: null
    function: null
    manager: null
    direct_reports: [aurora]
    mission: Executive sponsor
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: null
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: null
  - agent: aurora
    display_name: Aurora
    status: active
    operational: true
    department: Product
    function: Manager
    manager: elliott
    direct_reports: [root]
    mission: Manage delivery
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: elliott
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: /profiles/aurora
  - agent: root
    display_name: Root
    status: active
    operational: true
    department: Operations
    function: Infrastructure
    manager: aurora
    direct_reports: []
    mission: Own infrastructure
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: aurora
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: /profiles/main
  - agent: amy
    display_name: Amy
    status: friend
    operational: false
    department: null
    function: Personal-only
    manager: null
    direct_reports: []
    mission: Personal continuity only
    owned_outcomes: []
    authority: []
    prohibited_actions: [receive operational work]
    escalation_target: null
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: /profiles/amy
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(organization))
    kb.init_db()
    return home


def _park_in_review(conn, task_id: str) -> None:
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="implementation ready",
        expected_run_id=claimed.current_run_id,
    )


def test_canonical_root_is_spawnable_in_ready_and_review_lanes(
    canonical_root_home: Path,
) -> None:
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="ready", assignee="root")
        review_id = kb.create_task(conn, title="review", assignee="root")
        _park_in_review(conn, review_id)

        assert kb.has_spawnable_ready(conn) is True
        assert kb.has_spawnable_review(conn) is True

        result = kb.dispatch_once(conn, dry_run=True, max_in_progress=2)

    assert {task_id for task_id, _assignee, _workspace in result.spawned} == {
        ready_id,
        review_id,
    }
    assert result.skipped_nonspawnable == []
    # The board remains canonical: dispatch does not rewrite ownership to main.
    assert {assignee for _task_id, assignee, _workspace in result.spawned} == {"root"}


def test_default_spawn_resolves_canonical_root_only_for_worker_launch(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="launch", assignee="root")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb._default_spawn(task, str(workspace)) == 4242

    cmd = captured["cmd"]
    env = captured["env"]
    assert cmd[1:3] == ["-p", "main"]
    assert env["HERMES_PROFILE"] == "main"
    assert env["HERMES_HOME"] == str(canonical_root_home / "profiles" / "main")
    assert task.assignee == "root"


def test_direct_profiles_stay_spawnable_and_unknown_lanes_stay_skipped(
    canonical_root_home: Path,
) -> None:
    with kb.connect() as conn:
        direct_id = kb.create_task(conn, title="direct", assignee="main")
        unknown_id = kb.create_task(conn, title="control plane", assignee="orion-cc")
        result = kb.dispatch_once(conn, dry_run=True)

    assert direct_id in [task_id for task_id, _assignee, _workspace in result.spawned]
    assert unknown_id in result.skipped_nonspawnable


def test_non_operational_workforce_assignees_are_rejected_at_create(
    canonical_root_home: Path,
) -> None:
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="cannot receive operational Kanban work"):
            kb.create_task(conn, title="personal work", assignee="amy")
        task_id = kb.create_task(conn, title="operational work", assignee="root")
        with pytest.raises(ValueError, match="cannot receive operational Kanban work"):
            kb.assign_task(conn, task_id, "amy")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "root"


def test_non_operational_workforce_assignees_are_rejected_in_all_dispatch_lanes(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted pre-guard assignments are never launched and leave evidence."""
    organization = os.environ["HERMES_WORKFORCE_ORG"]
    monkeypatch.delenv("HERMES_WORKFORCE_ORG")
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="persisted ready", assignee="amy")
        review_id = kb.create_task(conn, title="persisted review", assignee="amy")
        _park_in_review(conn, review_id)

    monkeypatch.setenv("HERMES_WORKFORCE_ORG", organization)

    def unexpected_spawn(*_args, **_kwargs):
        pytest.fail("the dispatcher must not launch personal-only profiles")

    with kb.connect() as conn:
        result = kb.dispatch_once(conn, spawn_fn=unexpected_spawn)
        rejected = {task_id: (assignee, reason) for task_id, assignee, reason in result.rejected_non_operational}
        assert set(rejected) == {ready_id, review_id}
        assert {assignee for assignee, _reason in rejected.values()} == {"amy"}
        assert all("amy is friend" in reason for _assignee, reason in rejected.values())

        # The task stays ready/review for a valid operational reassignment,
        # while the event trail makes the dispatcher rejection inspectable.
        events = conn.execute(
            "SELECT task_id, payload FROM task_events WHERE kind = 'dispatch_rejected' "
            "ORDER BY task_id"
        ).fetchall()
        assert {event["task_id"] for event in events} == {ready_id, review_id}
        assert all(json.loads(event["payload"])["assignee"] == "amy" for event in events)

        # Repeated dispatcher ticks do not flood task events with the same
        # unchanged invalid assignment.
        kb.dispatch_once(conn, spawn_fn=unexpected_spawn)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'dispatch_rejected'"
        ).fetchone()[0] == 2
