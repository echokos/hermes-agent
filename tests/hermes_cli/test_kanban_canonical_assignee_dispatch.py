"""Regression coverage for canonical workforce assignees in Kanban dispatch."""

from __future__ import annotations

import builtins
from dataclasses import replace
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb


@pytest.fixture
def canonical_root_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create root->main plus a colliding, unrelated root profile directory."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "main").mkdir(parents=True)
    (home / "profiles" / "root").mkdir(parents=True)
    (home / "profiles" / "aurora").mkdir(parents=True)
    (home / "profiles" / "amy").mkdir(parents=True)
    (home / "profiles" / "legacy").mkdir(parents=True)
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
    assert kb._resolve_dispatch_profile("root") == "main"
    assert (canonical_root_home / "profiles" / "root").is_dir()
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


@pytest.mark.parametrize("dry_run", [False, True])
def test_per_profile_cap_combines_canonical_owner_and_runtime_alias(
    canonical_root_home: Path,
    dry_run: bool,
) -> None:
    """Canonical ``root`` and declared alias ``main`` share one worker cap."""
    with kb.connect() as conn:
        root_id = kb.create_task(
            conn, title="canonical owner", assignee="root", priority=10
        )
        alias_id = kb.create_task(
            conn, title="runtime alias", assignee="main", priority=0
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: 4242,
            dry_run=dry_run,
            max_in_progress_per_profile=1,
        )

        root = kb.get_task(conn, root_id)
        alias = kb.get_task(conn, alias_id)

    assert [item[:2] for item in result.spawned] == [(root_id, "root")]
    assert result.skipped_per_profile_capped == [(alias_id, "main", 1)]
    expected_root_status = "ready" if dry_run else "running"
    assert root is not None
    assert root.status == expected_root_status
    assert root.assignee == "root"
    assert alias is not None and alias.status == "ready" and alias.assignee == "main"


def test_per_profile_cap_maps_existing_canonical_work_to_runtime_alias(
    canonical_root_home: Path,
) -> None:
    with kb.connect() as conn:
        root_id = kb.create_task(conn, title="already running", assignee="root")
        assert kb.claim_task(conn, root_id) is not None
        alias_id = kb.create_task(conn, title="runtime alias", assignee="main")

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail(
                "runtime alias exceeded the concrete profile cap"
            ),
            max_in_progress_per_profile=1,
        )

    assert result.spawned == []
    assert result.skipped_per_profile_capped == [(alias_id, "main", 1)]


def test_claim_time_profile_cap_uses_fresh_reassigned_runtime(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reassignment cannot reserve request budget or claim past the cap."""
    from hermes_cli.workforce_org import load_organization

    with kb.connect() as conn:
        running_id = kb.create_task(conn, title="already running", assignee="root")
        assert kb.claim_task(conn, running_id) is not None
        request_root_id = kb.create_task(
            conn,
            title="coordinate work",
            assignee="aurora",
            session_id="capacity-race-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=request_root_id,
            platform="buzz",
            chat_id="capacity-race-chat",
            notifier_profile="aurora",
            delivery_mode="wake",
        )
        request = kb.create_coordination_request(
            conn,
            root_task_id=request_root_id,
            origin_session_id="capacity-race-session",
            origin_message_id="capacity-race-message",
            organization=load_organization(),
            now=100,
        )
        candidate_id = kb.create_task(
            conn,
            title="raced candidate",
            assignee="aurora",
            coordination_source_task_id=request_root_id,
        )
        reserved_before = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'coordination_launch_reserved'",
            (request_root_id,),
        ).fetchone()[0]

    original_resolve = kb._resolve_dispatch_profile
    reassigned = False

    def resolve_then_reassign(assignee, **kwargs):
        nonlocal reassigned
        profile = original_resolve(assignee, **kwargs)
        if assignee == "aurora" and not reassigned:
            with kb.connect() as writer:
                with kb.write_txn(writer):
                    writer.execute(
                        "UPDATE tasks SET assignee = 'main' WHERE id = ?",
                        (candidate_id,),
                    )
            reassigned = True
        return profile

    monkeypatch.setattr(kb, "_resolve_dispatch_profile", resolve_then_reassign)
    with kb.connect() as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail(
                "fresh cap refusal reached spawn"
            ),
            max_in_progress_per_profile=1,
        )
        candidate = kb.get_task(conn, candidate_id)
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (candidate_id,)
        ).fetchone()[0]
        reserved_after = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'coordination_launch_reserved'",
            (request_root_id,),
        ).fetchone()[0]
        current_request = kb.get_coordination_request(conn, request.id)

    assert reassigned is True
    assert result.spawned == []
    assert result.skipped_per_profile_capped == [(candidate_id, "main", 1)]
    assert candidate is not None
    assert candidate.status == "ready"
    assert candidate.assignee == "main"
    assert candidate.claim_lock is None
    assert candidate.current_run_id is None
    assert runs == 0
    assert reserved_after == reserved_before == 0
    assert current_request is not None
    assert current_request.leaf_launches_used == 0


def test_per_profile_cap_keeps_colliding_canonical_profiles_independent(
    canonical_root_home: Path,
) -> None:
    """Canonical ``main -> foo`` does not share ``root -> main`` capacity."""
    organization_path = Path(os.environ["HERMES_WORKFORCE_ORG"])
    organization = yaml.safe_load(organization_path.read_text(encoding="utf-8"))
    root = next(
        agent for agent in organization["agents"] if agent["agent"] == "root"
    )
    aurora = next(
        agent for agent in organization["agents"] if agent["agent"] == "aurora"
    )
    aurora["direct_reports"].append("main")
    organization["agents"].append({
        **root,
        "agent": "main",
        "display_name": "Canonical Main",
        "direct_reports": [],
        "profile_path": str(canonical_root_home / "profiles" / "foo"),
    })
    organization_path.write_text(
        yaml.safe_dump(organization, sort_keys=False),
        encoding="utf-8",
    )
    (canonical_root_home / "profiles" / "foo").mkdir()

    with kb.connect() as conn:
        root_id = kb.create_task(conn, title="root runtime", assignee="root")
        main_id = kb.create_task(conn, title="main runtime", assignee="main")
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: 4242,
            max_in_progress_per_profile=1,
        )

    assert kb._resolve_dispatch_profile("root") == "main"
    assert kb._resolve_dispatch_profile("main") == "foo"
    assert {item[:2] for item in result.spawned} == {
        (root_id, "root"),
        (main_id, "main"),
    }
    assert result.skipped_per_profile_capped == []


def test_review_reservation_uses_real_canonical_runtime_alias(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    with kb.connect() as conn:
        alias_ready = kb.create_task(
            conn, title="same runtime", assignee="main", priority=10
        )
        other_ready = kb.create_task(
            conn, title="other runtime", assignee="legacy", priority=0
        )
        review_id = kb.create_task(conn, title="canonical review", assignee="root")
        _park_in_review(conn, review_id)
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: 4242,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

    spawned_ids = [item[0] for item in result.spawned]
    assert alias_ready not in spawned_ids
    assert spawned_ids == [other_ready, review_id]


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


def test_actual_review_dispatch_runs_root_on_main_profile(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review launch", assignee="root")
        _park_in_review(conn, task_id)
        result = kb.dispatch_once(conn, max_spawn=1)
        task = kb.get_task(conn, task_id)

    assert result.spawned[0][:2] == (task_id, "root")
    assert captured["cmd"][1:3] == ["-p", "main"]
    assert captured["env"]["HERMES_PROFILE"] == "main"
    assert captured["env"]["HERMES_HOME"] == str(
        canonical_root_home / "profiles" / "main"
    )
    assert task is not None
    assert task.status == "running"
    assert task.assignee == "root"


def test_direct_profiles_stay_spawnable_and_unknown_lanes_stay_skipped(
    canonical_root_home: Path,
) -> None:
    assert kb._resolve_dispatch_profile("main") == "main"
    assert kb._resolve_dispatch_profile("legacy") == "legacy"
    with kb.connect() as conn:
        direct_id = kb.create_task(conn, title="direct", assignee="main")
        legacy_id = kb.create_task(conn, title="legacy", assignee="legacy")
        unknown_id = kb.create_task(conn, title="control plane", assignee="orion-cc")
        result = kb.dispatch_once(conn, dry_run=True)

    assert direct_id in [task_id for task_id, _assignee, _workspace in result.spawned]
    assert legacy_id in [task_id for task_id, _assignee, _workspace in result.spawned]
    assert unknown_id in result.skipped_nonspawnable


def test_dispatch_profile_refuses_inactive_profileless_and_ambiguous_agents(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.workforce_org as workforce_org

    organization = workforce_org.load_organization()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="ambiguous runtime", assignee="root")
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb._resolve_dispatch_profile("amy") is None

    profileless = replace(
        organization.agents["root"],
        agent="profileless",
        display_name="Profileless",
        profile_path=None,
    )
    profileless_org = replace(
        organization,
        agents={**organization.agents, "profileless": profileless},
    )
    (canonical_root_home / "profiles" / "profileless").mkdir()
    monkeypatch.setattr(
        workforce_org,
        "load_organization",
        lambda *args, **kwargs: profileless_org,
    )
    assert kb._resolve_dispatch_profile("profileless") is None

    duplicate = replace(
        organization.agents["root"],
        agent="duplicate",
        display_name="Duplicate Runtime",
    )
    ambiguous_org = replace(
        organization,
        agents={**organization.agents, "duplicate": duplicate},
    )
    monkeypatch.setattr(
        workforce_org,
        "load_organization",
        lambda *args, **kwargs: ambiguous_org,
    )
    assert kb._resolve_dispatch_profile("root") is None
    assert kb._resolve_dispatch_profile("main") is None
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("ambiguous task reached Popen"),
    )
    with pytest.raises(ValueError, match="no launchable Hermes profile"):
        kb._default_spawn(task, str(canonical_root_home))


def test_dispatch_profile_distinguishes_absent_invalid_and_unavailable_org(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization_path = Path(os.environ["HERMES_WORKFORCE_ORG"])

    organization_path.unlink()
    assert kb._resolve_dispatch_profile("legacy") == "legacy"
    assert kb._resolve_dispatch_profile("missing") is None

    organization_path.write_text("not: [valid", encoding="utf-8")
    assert kb._resolve_dispatch_profile("legacy") is None

    original_import = builtins.__import__

    def unavailable_workforce(name, *args, **kwargs):
        if name == "hermes_cli.workforce_org":
            raise ImportError("workforce organization unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_workforce)
    assert kb._resolve_dispatch_profile("legacy") is None
    monkeypatch.setattr(builtins, "__import__", original_import)

    def unavailable_profiles(name, *args, **kwargs):
        if name == "hermes_cli.profiles":
            raise ImportError("profiles unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_profiles)
    assert kb._resolve_dispatch_profile("root") == "root"


def test_default_spawn_refuses_missing_declared_runtime_without_popen(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="missing declared runtime", assignee="root")
        task = kb.get_task(conn, task_id)
    assert task is not None
    (canonical_root_home / "profiles" / "main").rmdir()
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("refused task reached Popen"),
    )

    with kb.connect() as conn:
        assert kb.has_spawnable_ready(conn) is False
        result = kb.dispatch_once(conn)
    assert result.spawned == []
    assert result.skipped_nonspawnable == [task_id]
    with pytest.raises(ValueError, match="no launchable Hermes profile"):
        kb._default_spawn(task, str(tmp_path))


def test_default_spawn_only_keeps_missing_org_legacy_fallback(
    canonical_root_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy late profile", assignee="late-profile")
        task = kb.get_task(conn, task_id)
    assert task is not None
    organization_path = Path(os.environ["HERMES_WORKFORCE_ORG"])
    organization_path.unlink()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured: dict = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert kb._default_spawn(task, str(tmp_path)) == 4244
    assert captured["cmd"][1:3] == ["-p", "late-profile"]
    assert captured["env"]["HERMES_PROFILE"] == "late-profile"

    organization_path.write_text("not: [valid", encoding="utf-8")
    captured.clear()
    with pytest.raises(ValueError, match="no launchable Hermes profile"):
        kb._default_spawn(task, str(tmp_path))
    assert captured == {}


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
