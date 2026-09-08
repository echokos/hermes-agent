from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import runbook_store
from hermes_cli import workflow_registry as reg
from hermes_cli.workflow_runtime import (
    build_runbook_agent_prompt,
    link_existing_cron_job,
    sync_runbook_cron_jobs,
)
from hermes_constants import get_hermes_home


def _metadata() -> dict:
    return {
        "id": "wf-daily-brief",
        "slug": "daily-brief",
        "title": "Daily Brief",
        "purpose": "Prepare a concise daily brief.",
        "owner_profile": "default",
        "status": "active",
        "runtime": {"kind": "hermes", "ref": "gateway"},
        "schedules": [
            {
                "id": "morning",
                "profile": "default",
                "schedule": "every 1h",
                "deliver": "local",
                "step_key": "collect",
            }
        ],
        "steps": [
            {
                "step_key": "collect",
                "name": "Collect context",
                "description": "Read current operating context.",
            }
        ],
        "inputs": {},
        "outputs": {},
        "permitted_writes": [],
        "approval_rules": {},
        "retry": {"max_attempts": 2},
        "timeout": {"seconds": 1800},
        "deduplication": {"strategy": "date"},
        "related": {},
    }


def _save_runbook() -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: openrouter\n"
        "  default: test-model\n"
        "agent:\n"
        "  reasoning_effort: medium\n"
        "  speed: standard\n",
        encoding="utf-8",
    )
    runbook_store.save_runbook(
        _metadata(),
        "# Daily Brief\n\n## Procedure\n\n1. Collect context.\n",
        approved_by="dashboard",
    )
    with reg.connect_closing() as conn:
        reg.create_definition(
            conn,
            id="wf-daily-brief",
            slug="daily-brief",
            name="Daily Brief",
            owner_profile="default",
            status="active",
            runtime_kind="hermes",
        )
        reg.replace_steps(
            conn,
            "wf-daily-brief",
            [{"step_key": "collect", "position": 0, "name": "Collect context"}],
        )


def test_build_runbook_agent_prompt_contains_protocol() -> None:
    _save_runbook()

    prompt = build_runbook_agent_prompt("daily-brief", step_key="collect")

    assert "Run Hermes workflow `daily-brief`." in prompt
    assert "Execute step `collect`: Collect context" in prompt
    assert "Standing assignment reference: `workflow:wf-daily-brief:step:collect`." in prompt
    assert "[WORKFLOW_STATUS:completed]" in prompt
    assert "[WORKFLOW_STATUS:blocked]" in prompt
    assert "[WORKFLOW_STATUS:failed]" in prompt
    assert "## Procedure" in prompt


@pytest.mark.parametrize(
    ("owner", "executor", "directed"),
    [("aurora", "chloe", True), ("default", "chloe", False), ("aurora", "default", False)],
)
def test_prompt_binds_chloe_intake_to_canonical_aurora_step(owner, executor, directed) -> None:
    _save_runbook()
    metadata = _metadata()
    metadata["owner_profile"] = owner
    metadata["steps"][0]["executor_profile"] = executor
    runbook_store.save_runbook(metadata, "# Assigned observation\n", approved_by="dashboard")

    prompt = build_runbook_agent_prompt("daily-brief", step_key="collect")

    assert ("aurora_assignment_id: `workflow:wf-daily-brief:step:collect`" in prompt) is directed
    if directed:
        assert "omit department_recommendation and estimated_effort" in prompt
        assert "does not authorize recommendations, execution, or a signal on a quiet run" in prompt
    assert "aurora_assignment_id" not in build_runbook_agent_prompt("daily-brief")


def test_sync_runbook_cron_jobs_creates_profile_job_and_registry_link() -> None:
    _save_runbook()

    synced = sync_runbook_cron_jobs("daily-brief")

    assert len(synced) == 1
    job = synced[0]
    assert job["profile"] == "default"
    assert job["workflow_id"] == "wf-daily-brief"
    assert job["workflow_step_key"] == "collect"
    assert job["workflow_schedule_id"] == "morning"
    assert job["track_workflow_status"] is True
    assert "RUNBOOK.md" in job["prompt"]

    from cron.jobs import list_jobs

    jobs = list_jobs(include_disabled=True)
    assert jobs[0]["id"] == job["id"]

    with reg.connect_closing() as conn:
        rows = conn.execute("SELECT * FROM workflow_schedules").fetchall()
    assert len(rows) == 1
    assert rows[0]["cron_job_id"] == job["id"]


def test_sync_runbook_cron_jobs_updates_existing_job() -> None:
    _save_runbook()
    first = sync_runbook_cron_jobs("daily-brief")[0]
    metadata = _metadata()
    metadata["schedules"][0]["schedule"] = "every 2h"
    runbook_store.save_runbook(
        metadata,
        "# Daily Brief\n\n## Procedure\n\n1. Collect updated context.\n",
        approved_by="dashboard",
    )

    second = sync_runbook_cron_jobs("daily-brief")[0]

    assert second["id"] == first["id"]
    assert second["schedule_display"] == "every 120m"
    assert "Collect updated context" in second["prompt"]


def test_sync_runbook_cron_jobs_persists_runtime_budgets() -> None:
    _save_runbook()
    metadata = _metadata()
    metadata["runtime"]["max_iterations"] = 12
    metadata["runtime"]["tool_budget"] = {
        "max_calls": 8,
        "max_writes": 1,
        "max_detail_reads": 3,
        "max_list_items": 20,
        "allowed_tools": ["kanban_list", "workforce_signal"],
    }
    runbook_store.save_runbook(
        metadata,
        "# Daily Brief\n\n## Procedure\n\n1. Collect context.\n",
        approved_by="dashboard",
    )

    job = sync_runbook_cron_jobs("daily-brief")[0]

    assert job["max_iterations"] == 12
    assert job["runtime_tool_budget"] == metadata["runtime"]["tool_budget"]


def test_sync_runbook_cron_jobs_projects_trusted_dependency_ownership() -> None:
    _save_runbook()
    metadata = _metadata()
    schedule = metadata["schedules"][0]
    schedule["required_tool_dependencies"] = ["mcp__nirvana__get_tasks"]
    schedule["failure_ownership"] = {
        "technical_owner": "root",
        "director": "aurora",
        "return_outcome_to_origin": True,
    }
    runbook_store.save_runbook(
        metadata,
        "# Daily Brief\n\n## Procedure\n\n1. Collect context.\n",
        approved_by="dashboard",
    )

    job = sync_runbook_cron_jobs("daily-brief")[0]

    assert job["required_tool_dependencies"] == ["mcp__nirvana__get_tasks"]
    assert job["failure_ownership"] == schedule["failure_ownership"]


def test_omitted_runbook_ownership_does_not_clear_existing_metadata() -> None:
    _save_runbook()
    first_metadata = _metadata()
    first_metadata["schedules"][0]["failure_ownership"] = {
        "technical_owner": "root",
        "director": "aurora",
        "return_outcome_to_origin": True,
    }
    runbook_store.save_runbook(
        first_metadata,
        "# Daily Brief\n",
        approved_by="dashboard",
    )
    first = sync_runbook_cron_jobs("daily-brief")[0]

    runbook_store.save_runbook(
        _metadata(),
        "# Daily Brief\n",
        approved_by="dashboard",
    )
    second = sync_runbook_cron_jobs("daily-brief")[0]

    assert second["id"] == first["id"]
    assert second["failure_ownership"] == first["failure_ownership"]


def test_link_existing_cron_job_only_adds_registry_identity() -> None:
    _save_runbook()
    from cron import jobs as cron_jobs

    original = cron_jobs.create_job(
        prompt="Preserve this exact prompt.",
        schedule="every 3h",
        name="Existing job",
        deliver="local",
        provider="openrouter",
        model="test-model",
        reasoning_effort="medium",
        speed="standard",
    )

    linked = link_existing_cron_job(
        "daily-brief",
        profile="default",
        cron_job_id=original["id"],
        schedule_id="existing",
        step_key="collect",
    )

    for field in (
        "prompt",
        "schedule",
        "schedule_display",
        "deliver",
        "provider",
        "model",
        "reasoning_effort",
        "speed",
        "enabled",
        "track_workflow_status",
    ):
        assert linked.get(field) == original.get(field), field
    assert linked["workflow_id"] == "wf-daily-brief"
    assert linked["workflow_slug"] == "daily-brief"
    assert linked["workflow_step_key"] == "collect"
    assert linked["workflow_schedule_id"] == "existing"
