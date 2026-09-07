import json
from pathlib import Path
import sqlite3

import yaml

from hermes_cli import kanban_db
from cron.operational_failures import (
    append_host_failure, append_profile_failure, append_profile_recovery,
)
from scripts.workforce_health_monitor import run


def _fixture(tmp_path: Path):
    profiles = tmp_path / "profiles"
    agents = []
    for agent_id in ("aurora", "worker"):
        profile = profiles / agent_id
        (profile / "cron").mkdir(parents=True)
        (profile / "AGENTS.md").write_text(f"# {agent_id}\n")
        (profile / "config.yaml").write_text("{}\n")
        (profile / "cron" / "jobs.json").write_text('{"jobs": []}\n')
        agents.append({
            "agent": agent_id,
            "display_name": agent_id.title(),
            "status": "active",
            "operational": True,
            "department": None,
            "function": "Chief of Staff" if agent_id == "aurora" else "Specialist",
            "manager": "elliott" if agent_id == "aurora" else "aurora",
            "direct_reports": ["worker"] if agent_id == "aurora" else [],
            "mission": "test",
            "owned_outcomes": ["test"],
            "authority": ["test"],
            "prohibited_actions": [],
            "escalation_target": "elliott" if agent_id == "aurora" else "aurora",
            "cross_team_request_path": "test",
            "buzz_rooms": [],
            "profile_path": str(profile),
        })
    organization = tmp_path / "organization.yaml"
    organization.write_text(yaml.safe_dump({
        "schema_version": 1,
        "workforce_contract_version": "test",
        "reserved_approvals": [],
        "agents": [
            {
                "agent": "elliott", "display_name": "Elliott", "status": "artifact",
                "operational": False, "department": None, "function": "Owner",
                "manager": None, "direct_reports": ["aurora"], "mission": "test",
                "owned_outcomes": ["test"], "authority": ["test"],
                "prohibited_actions": [], "escalation_target": None,
                "cross_team_request_path": None, "buzz_rooms": [], "profile_path": None,
            },
            *agents,
        ],
    }, sort_keys=False))
    database = tmp_path / "kanban.db"
    with kanban_db.connect_closing(database):
        pass
    return organization, database, tmp_path / "state.json", profiles / "worker"


def _write_failure(profile: Path):
    job = {
        "id": "job-1", "name": "Critical collector", "enabled": True,
        "last_status": "error", "last_error": "token=secret-value boom",
        "last_run_at": "2026-08-21T08:00:00-05:00", "failure_streak": 2,
    }
    (profile / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))


def _write_successes(profile: Path):
    job = {
        "id": "job-1", "name": "Critical collector", "enabled": True,
        "last_status": "ok", "last_error": None,
        "last_run_at": "2026-08-21T09:00:00-05:00", "failure_streak": 0,
    }
    (profile / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))
    conn = sqlite3.connect(profile / "cron" / "executions.db")
    conn.execute(
        "CREATE TABLE executions (id TEXT PRIMARY KEY, job_id TEXT, status TEXT, claimed_at TEXT)"
    )
    conn.executemany(
        "INSERT INTO executions VALUES (?,?,?,?)",
        [
            ("2", "job-1", "completed", "2026-08-21T09:00:00-05:00"),
            ("1", "job-1", "completed", "2026-08-21T08:30:00-05:00"),
        ],
    )
    conn.commit()
    conn.close()


def test_recurring_failure_creates_once_and_closes_after_two_successes(tmp_path: Path):
    organization, database, _state, profile = _fixture(tmp_path)
    state = tmp_path / "state" / "workforce-health.json"
    _write_failure(profile)

    first = run(organization=organization, database=database, state_path=state)
    second = run(organization=organization, database=database, state_path=state)
    assert first == {"detected": 1, "created": 1, "attached": 0, "recovered": 0, "state": str(state)}
    assert second["created"] == 0

    with kanban_db.connect_closing(database) as conn:
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
        assert len(tasks) == 1
        task_id = tasks[0]["id"]
        assert tasks[0]["assignee"] == "aurora"
        assert "secret-value" not in (tasks[0]["body"] or "")
        assert len(kanban_db.list_comments(conn, task_id)) == 1

    _write_successes(profile)
    recovery = run(organization=organization, database=database, state_path=state)
    assert recovery["recovered"] == 1
    with kanban_db.connect_closing(database) as conn:
        assert kanban_db.get_task(conn, task_id).status == "done"


def test_failure_attaches_to_existing_active_repair_instead_of_fanout(tmp_path: Path):
    organization, database, state, profile = _fixture(tmp_path)
    _write_failure(profile)
    with kanban_db.connect_closing(database) as conn:
        existing = kanban_db.create_task(
            conn,
            title="Repair Critical collector",
            body="Existing canonical repair for job-1",
            assignee="worker",
        )

    result = run(organization=organization, database=database, state_path=state)
    assert result["attached"] == 1
    assert result["created"] == 0
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert len(kanban_db.list_comments(conn, existing)) == 1


def test_opted_in_profile_failure_is_owned_deduplicated_and_recovery_cannot_complete_it(
    tmp_path: Path, monkeypatch,
):
    """A restart sees the same durable event but never creates a second card."""
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "GitHub collector",
        "workflow_id": "grace-github-collection",
        "failure_streak": 1,
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora", "severity": "warning",
        },
    }
    append_profile_failure(profile, job, "upstream timeout token=not-for-task")

    first = run(organization=organization, database=database, state_path=state)
    second = run(organization=organization, database=database, state_path=state)
    assert first["created"] == 1
    assert second["created"] == 0

    with kanban_db.connect_closing(database) as conn:
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
        assert len(tasks) == 1
        task_id = tasks[0]["id"]
        assert tasks[0]["assignee"] == "worker"
        assert "not-for-task" not in (tasks[0]["body"] or "")

    _write_successes(profile)
    append_profile_recovery(profile, job)
    recovered = run(organization=organization, database=database, state_path=state)
    assert recovered["recovered"] == 1
    with kanban_db.connect_closing(database) as conn:
        # This proves only the scheduler path recovered. An accountable owner
        # and director must still acknowledge/accept the repair.
        assert kanban_db.get_task(conn, task_id).status in {"ready", "running", "review"}
    # The historical failure line is behind the durable cursor; restarting the
    # monitor after recovery cannot reopen or recreate it.
    idle = run(organization=organization, database=database, state_path=state)
    assert idle["created"] == 0
    assert idle["recovered"] == 0


def test_missing_host_ownership_creates_internal_aurora_configuration_incident(tmp_path: Path):
    organization, database, _state, _profile = _fixture(tmp_path)
    state = tmp_path / "state" / "workforce-health.json"
    append_host_failure(tmp_path, {
        "workflow_id": "op-onecli-sync",
        "source_id": "op-onecli-sync",
        "director": "aurora",
        "error": "host command failed",
    })
    result = run(organization=organization, database=database, state_path=state)
    assert result["created"] == 1
    with kanban_db.connect_closing(database) as conn:
        task = conn.execute("SELECT * FROM tasks").fetchone()
        assert task["assignee"] == "aurora"
        assert "failure_ownership_configuration" in (task["body"] or "")


def test_host_recovery_event_is_ordered_and_does_not_complete_owner_incident(tmp_path: Path):
    organization, database, _state, _profile = _fixture(tmp_path)
    state = tmp_path / "state" / "workforce-health.json"
    failure = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "worker", "director": "aurora", "error": "exit 1",
    }
    append_host_failure(tmp_path, failure)
    first = run(organization=organization, database=database, state_path=state)
    assert first["created"] == 1
    recovery = append_host_failure(tmp_path, {**failure, "outcome": "recovered"})
    assert recovery["status"] == "recovered"
    result = run(organization=organization, database=database, state_path=state)
    assert result["recovered"] == 1
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert conn.execute("SELECT status FROM tasks").fetchone()[0] != "done"
