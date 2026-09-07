import json
from pathlib import Path
import sqlite3
import time

import yaml

from hermes_cli import kanban_db
from hermes_cli.workforce_handoffs import acknowledge_handoff
from hermes_cli.workforce_org import load_organization
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
    append_profile_failure(
        profile,
        job,
        "upstream timeout token=not-for-task",
        execution_id="failure-1",
    )

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
        assert json.loads(tasks[0]["body"])["kind"] == "workforce_handoff"
        assert tasks[0]["status"] == "triage"

    append_profile_recovery(profile, job, execution_id="success-1")
    first_success = run(organization=organization, database=database, state_path=state)
    assert first_success["recovered"] == 0
    append_profile_recovery(profile, job, execution_id="success-2")
    recovered = run(organization=organization, database=database, state_path=state)
    assert recovered["recovered"] == 1
    with kanban_db.connect_closing(database) as conn:
        assert kanban_db.get_task(conn, task_id).status == "triage"
    # The historical failure line is behind the durable cursor; restarting the
    # monitor after recovery cannot reopen or recreate it.
    idle = run(organization=organization, database=database, state_path=state)
    assert idle["created"] == 0
    assert idle["recovered"] == 0


def test_missing_host_ownership_creates_internal_aurora_configuration_incident(
    tmp_path: Path, monkeypatch,
):
    organization, database, _state, _profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = tmp_path / "workforce-control" / "health-monitor-state.json"
    append_host_failure(tmp_path, {
        "workflow_id": "op-onecli-sync",
        "source_id": "op-onecli-sync",
        "director": "aurora",
        "error": "host command failed",
        "execution_id": "host-failure-1",
    })
    result = run(organization=organization, database=database, state_path=state)
    assert result["created"] == 1
    with kanban_db.connect_closing(database) as conn:
        task = conn.execute("SELECT * FROM tasks").fetchone()
        assert task["assignee"] == "aurora"
        assert "failure_ownership_configuration" in (task["body"] or "")


def test_host_recovery_event_is_ordered_and_does_not_complete_owner_incident(
    tmp_path: Path, monkeypatch,
):
    organization, database, _state, _profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = tmp_path / "workforce-control" / "health-monitor-state.json"
    failure = {
        "workflow_id": "op-onecli-sync", "source_id": "op-onecli-sync",
        "technical_owner": "worker", "director": "aurora", "error": "exit 1",
        "execution_id": "host-failure-1",
    }
    append_host_failure(tmp_path, failure)
    first = run(organization=organization, database=database, state_path=state)
    assert first["created"] == 1
    recovery = append_host_failure(tmp_path, {
        **failure, "execution_id": "host-success-1", "outcome": "recovered",
    })
    assert recovery["status"] == "recovered"
    result = run(organization=organization, database=database, state_path=state)
    assert result["recovered"] == 0
    append_host_failure(tmp_path, {
        **failure, "execution_id": "host-success-2", "outcome": "recovered",
    })
    result = run(organization=organization, database=database, state_path=state)
    assert result["recovered"] == 1
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert conn.execute("SELECT status FROM tasks").fetchone()[0] != "done"


def test_owned_repair_requires_two_actual_successes_before_director_completion(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1",
        "name": "Collector",
        "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker",
            "director": "aurora",
        },
    }
    append_profile_failure(profile, job, "timeout", execution_id="failure-1")
    assert run(
        organization=organization, database=database, state_path=state,
    )["created"] == 1

    with kanban_db.connect_closing(database) as conn:
        task_id = conn.execute("SELECT id FROM tasks").fetchone()[0]
        acknowledge_handoff(
            conn,
            task_id,
            actor="worker",
            organization=load_organization(organization),
        )
        owner_run = kanban_db.claim_task(conn, task_id, claimer="worker:test")
        assert owner_run is not None
        assert kanban_db.request_review(
            conn,
            task_id,
            summary="Repair ready for scheduler-path verification.",
            expected_run_id=owner_run.current_run_id,
        )
        review_run = kanban_db.claim_review_task(
            conn, task_id, claimer="aurora:test"
        )
        assert review_run is not None
        assert not kanban_db.complete_task(
            conn,
            task_id,
            summary="Premature acceptance before the recovery gate.",
            expected_run_id=review_run.current_run_id,
        )
        assert kanban_db.get_task(conn, task_id).status == "running"

    append_profile_recovery(profile, job, execution_id="success-1")
    assert run(
        organization=organization, database=database, state_path=state,
    )["recovered"] == 0
    append_profile_recovery(profile, job, execution_id="success-2")
    assert run(
        organization=organization, database=database, state_path=state,
    )["recovered"] == 1

    with kanban_db.connect_closing(database) as conn:
        evidence = [
            event
            for event in kanban_db.list_events(conn, task_id)
            if event.kind == "workforce_handoff_recovery_verified"
        ]
        assert len(evidence) == 1
        assert len(set(evidence[0].payload["success_event_ids"])) == 2
        assert len(set(evidence[0].payload["success_orders"])) == 2
        first_recovery_order = max(evidence[0].payload["success_orders"])

    append_profile_failure(profile, job, "timeout again", execution_id="failure-2")
    assert run(
        organization=organization, database=database, state_path=state,
    )["created"] == 0

    with kanban_db.connect_closing(database) as conn:
        assert not kanban_db.complete_task(
            conn,
            task_id,
            summary="Stale recovery evidence cannot satisfy the new failure.",
            expected_run_id=review_run.current_run_id,
        )

    append_profile_recovery(profile, job, execution_id="success-3")
    assert run(
        organization=organization, database=database, state_path=state,
    )["recovered"] == 0
    append_profile_recovery(profile, job, execution_id="success-4")
    assert run(
        organization=organization, database=database, state_path=state,
    )["recovered"] == 1

    with kanban_db.connect_closing(database) as conn:
        evidence = [
            event
            for event in kanban_db.list_events(conn, task_id)
            if event.kind == "workforce_handoff_recovery_verified"
        ]
        assert len(evidence) == 2
        assert min(evidence[-1].payload["success_orders"]) > first_recovery_order
        assert kanban_db.complete_task(
            conn,
            task_id,
            summary="Director accepted after two scheduler executions.",
            expected_run_id=review_run.current_run_id,
        )


def _accept_owner_repair(database: Path, task_id: str, organization: Path):
    with kanban_db.connect_closing(database) as conn:
        acknowledge_handoff(
            conn,
            task_id,
            actor="worker",
            organization=load_organization(organization),
        )
        owner_run = kanban_db.claim_task(conn, task_id, claimer="worker:test")
        assert owner_run is not None
        assert kanban_db.request_review(
            conn,
            task_id,
            summary="Repair complete; execution evidence attached.",
            reviewer="aurora",
            expected_run_id=owner_run.current_run_id,
        )
        review_run = kanban_db.claim_review_task(
            conn, task_id, claimer="aurora:test"
        )
        assert review_run is not None
        assert kanban_db.complete_task(
            conn,
            task_id,
            summary="Director verified the bounded repair.",
            expected_run_id=review_run.current_run_id,
        )


def test_completed_episode_replay_is_inert_but_new_execution_opens_new_incident(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "GitHub collector",
        "workflow_id": "grace-github-collection",
        "failure_streak": 1,
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
        },
    }
    failure = append_profile_failure(
        profile, job, "timeout", execution_id="failure-1"
    )
    assert run(organization=organization, database=database, state_path=state)["created"] == 1
    with kanban_db.connect_closing(database) as conn:
        first_task = conn.execute("SELECT id FROM tasks").fetchone()[0]
    append_profile_recovery(profile, job, execution_id="success-1")
    second_success = append_profile_recovery(
        profile, job, execution_id="success-2"
    )
    assert run(organization=organization, database=database, state_path=state)["recovered"] == 1
    _accept_owner_repair(database, first_task, organization)
    run(organization=organization, database=database, state_path=state)
    state_value = json.loads(state.read_text())
    finding = next(iter(state_value["findings"].values()))
    assert finding["status"] == "resolved"
    assert "processed_events" not in state_value
    assert finding["last_processed_event_ids"] == [second_success["event_id"]]

    intake_path = profile / "cron" / "operational-failures.jsonl"
    replay_line = json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n"
    replacement = profile / "cron" / "replacement.jsonl"
    replacement.write_text(replay_line)
    replacement.replace(intake_path)
    replay = run(organization=organization, database=database, state_path=state)
    assert replay["created"] == 0
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1

    append_profile_failure(profile, job, "timeout", execution_id="failure-2")
    next_episode = run(organization=organization, database=database, state_path=state)
    assert next_episode["created"] == 1
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2


def test_execution_ledger_recovers_a_lost_append_and_requires_two_successes(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "Collector",
        "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
            "enabled_at": "2026-09-07T11:59:00+00:00",
        },
    }
    (profile / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))
    conn = sqlite3.connect(profile / "cron" / "executions.db")
    conn.execute(
        "CREATE TABLE executions (id TEXT PRIMARY KEY, job_id TEXT, status TEXT, "
        "claimed_at TEXT, started_at TEXT, finished_at TEXT, error TEXT)"
    )
    conn.executemany(
        "INSERT INTO executions VALUES (?,?,?,?,?,?,?)",
        [
            ("failed-1", "job-1", "failed", "2026-09-07T12:00:00+00:00", None, "2026-09-07T12:01:00+00:00", "timeout"),
            ("success-1", "job-1", "completed", "2026-09-07T12:02:00+00:00", None, "2026-09-07T12:03:00+00:00", None),
            ("success-2", "job-1", "completed", "2026-09-07T12:04:00+00:00", None, "2026-09-07T12:05:00+00:00", None),
        ],
    )
    conn.commit()
    conn.close()

    result = run(organization=organization, database=database, state_path=state)
    assert result["created"] == 1
    assert result["recovered"] == 1
    assert not (profile / "cron" / "operational-failures.jsonl").exists()
    state_value = json.loads(state.read_text())
    finding = next(iter(state_value["findings"].values()))
    assert len(finding["recovery_success_event_ids"]) == 2
    assert finding["recovery_verified"] is True
    ledger_cursor = next(iter(state_value["ledger_cursors"].values()))
    assert ledger_cursor["execution_id"] == "success-2"

    run(organization=organization, database=database, state_path=state)
    after_idle = json.loads(state.read_text())
    assert next(iter(after_idle["ledger_cursors"].values())) == ledger_cursor

    conn = sqlite3.connect(profile / "cron" / "executions.db")
    conn.execute(
        "INSERT INTO executions VALUES (?,?,?,?,?,?,?)",
        (
            "success-3", "job-1", "completed",
            "2026-09-07T12:06:00+00:00", None,
            "2026-09-07T12:07:00+00:00", None,
        ),
    )
    conn.commit()
    conn.close()
    run(organization=organization, database=database, state_path=state)
    after_newer = json.loads(state.read_text())
    assert next(iter(after_newer["ledger_cursors"].values()))["execution_id"] == "success-3"


def test_first_opt_in_bootstraps_healthy_without_reopening_old_failure(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "Collector", "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
        },
    }
    (profile / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}))
    conn = sqlite3.connect(profile / "cron" / "executions.db")
    conn.execute(
        "CREATE TABLE executions (id TEXT PRIMARY KEY, job_id TEXT, status TEXT, "
        "claimed_at TEXT, started_at TEXT, finished_at TEXT, error TEXT)"
    )
    conn.executemany(
        "INSERT INTO executions VALUES (?,?,?,?,?,?,?)",
        [
            ("failed-old", "job-1", "failed", "2026-09-06T12:00:00+00:00", None, "2026-09-06T12:01:00+00:00", "timeout"),
            ("success-1", "job-1", "completed", "2026-09-06T12:02:00+00:00", None, "2026-09-06T12:03:00+00:00", None),
            ("success-2", "job-1", "completed", "2026-09-06T12:04:00+00:00", None, "2026-09-06T12:05:00+00:00", None),
        ],
    )
    conn.commit()
    conn.close()

    result = run(organization=organization, database=database, state_path=state)

    assert result["detected"] == 0
    assert result["created"] == 0
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_state_loss_reconciles_exact_active_incident_without_fanout(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "Collector", "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
        },
    }
    append_profile_failure(profile, job, "timeout", execution_id="failed-1")
    assert run(organization=organization, database=database, state_path=state)["created"] == 1
    state.unlink()
    append_profile_failure(profile, job, "timeout", execution_id="failed-2")

    result = run(organization=organization, database=database, state_path=state)

    assert result["created"] == 0
    with kanban_db.connect_closing(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_active_incident_owner_change_is_recorded_without_silent_relabel(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "name": "Collector", "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
        },
    }
    append_profile_failure(profile, job, "timeout", execution_id="failed-1")
    run(organization=organization, database=database, state_path=state)
    job["failure_ownership"] = {
        "technical_owner": "aurora", "director": "aurora",
    }
    append_profile_failure(profile, job, "timeout", execution_id="failed-2")

    result = run(organization=organization, database=database, state_path=state)

    assert result["created"] == 0
    with kanban_db.connect_closing(database) as conn:
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
        assert len(tasks) == 1
        assert tasks[0]["assignee"] == "worker"
        comments = kanban_db.list_comments(conn, tasks[0]["id"])
        assert "ownership changed" in comments[-1].body.lower()
    finding = next(iter(json.loads(state.read_text())["findings"].values()))
    assert finding["technical_owner"] == "worker"
    assert finding["director"] == "aurora"


def test_human_input_block_creates_one_actionable_aurora_handoff_without_chloe(
    tmp_path: Path, monkeypatch,
):
    organization, database, state, profile = _fixture(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "job-1", "workflow_id": "collector-workflow",
        "failure_ownership": {
            "technical_owner": "worker", "director": "aurora",
        },
    }
    append_profile_failure(profile, job, "auth access required", execution_id="failed-1")
    run(organization=organization, database=database, state_path=state)
    with kanban_db.connect_closing(database) as conn:
        incident_id = conn.execute("SELECT id FROM tasks").fetchone()[0]
        acknowledge_handoff(
            conn,
            incident_id,
            actor="worker",
            organization=load_organization(organization),
        )
        owner_run = kanban_db.claim_task(conn, incident_id, claimer="worker:test")
        assert owner_run is not None
        assert kanban_db.block_task(
            conn,
            incident_id,
            reason="Elliott must grant access to the disabled account",
            kind="needs_input",
            expected_run_id=owner_run.current_run_id,
        )
    run(organization=organization, database=database, state_path=state)
    run(organization=organization, database=database, state_path=state)
    with kanban_db.connect_closing(database) as conn:
        tasks = conn.execute("SELECT * FROM tasks ORDER BY created_at, id").fetchall()
        assert len(tasks) == 2
        exception = next(row for row in tasks if row["id"] != incident_id)
        payload = json.loads(exception["body"])
        assert exception["assignee"] == "aurora"
        assert payload["context"]["kind"] == "operational_failure_human_exception"
        assert "grant access" in payload["context"]["reason"]


def test_overdue_acknowledgment_is_swept_by_aurora_when_chloe_is_offline(
    tmp_path: Path, monkeypatch,
):
    organization, database, _state, _profile = _fixture(tmp_path)
    state = tmp_path / "state" / "workforce-health.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    append_host_failure(tmp_path, {
        "workflow_id": "host-check", "source_id": "host-check",
        "technical_owner": "worker", "director": "aurora",
        "execution_id": "failed-1", "error": "exit 1",
        "occurred_at": int(time.time()) - 3600,
        "ack_timeout_seconds": 60, "repair_timeout_seconds": 120,
    })
    run(organization=organization, database=database, state_path=state)
    with kanban_db.connect_closing(database) as conn:
        task = conn.execute("SELECT * FROM tasks").fetchone()
        payload = json.loads(task["body"])
    assert task["status"] == "blocked"
    assert payload["state"] == "acknowledgment_overdue"
    assert payload["flagged_by"] == "aurora"
