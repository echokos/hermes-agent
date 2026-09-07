"""Runtime helpers connecting canonical runbooks to cron execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from hermes_cli import runbook_store
from hermes_cli.runbook_schema import ParsedRunbook
from hermes_cli import workflow_registry as registry


def build_runbook_agent_prompt(
    slug: str,
    *,
    step_key: str | None = None,
    trigger_context: str | None = None,
) -> str:
    """Build the cron agent instruction for a canonical runbook."""
    path = runbook_store.runbook_path(slug)
    parsed = runbook_store.read_runbook(path)
    metadata = parsed.metadata
    steps = metadata["steps"]
    selected_step = None
    if step_key:
        selected_step = next(
            (step for step in steps if step.get("step_key") == step_key),
            None,
        )
        if selected_step is None:
            raise ValueError(f"runbook {slug!r} has no step {step_key!r}")
    lines = [
        f"Run Hermes workflow `{metadata['slug']}`.",
        "",
        f"Title: {metadata['title']}",
        f"Purpose: {metadata['purpose']}",
        f"Owner profile: {metadata['owner_profile']}",
        f"Canonical source: {path}",
        "",
        "Follow the RUNBOOK.md below as authoritative operating procedure.",
        "If you complete the workflow, end the final response with exactly:",
        "[WORKFLOW_STATUS:completed]",
        "If the workflow cannot safely continue without human input, end with exactly:",
        "[WORKFLOW_STATUS:blocked]",
        "If the workflow fails or cannot complete, end with exactly:",
        "[WORKFLOW_STATUS:failed]",
    ]
    if selected_step is not None:
        assignment_ref = f"workflow:{metadata['id']}:step:{selected_step['step_key']}"
        lines.extend(
            [
                "",
                f"Execute step `{selected_step['step_key']}`: {selected_step['name']}",
                f"Standing assignment reference: `{assignment_ref}`.",
            ]
        )
        if selected_step.get("description"):
            lines.append(str(selected_step["description"]))
        if (
            metadata.get("owner_profile") == "aurora"
            and selected_step.get("executor_profile") == "chloe"
        ):
            lines.extend([
                "For a material factual workforce_signal within this assigned step, "
                f"use aurora_assignment_id: `{assignment_ref}`.",
                "Include the observation, expected factual outcome, and evidence references; "
                "omit department_recommendation and estimated_effort. This reference "
                "does not authorize recommendations, execution, or a signal on a quiet run.",
            ])
    if trigger_context:
        lines.extend(["", "Trigger context:", trigger_context.strip()])
    lines.extend(["", "RUNBOOK.md:", "", parsed.body.strip()])
    return "\n".join(lines).strip() + "\n"


def sync_runbook_cron_jobs(slug: str) -> list[dict[str, Any]]:
    """Create or update cron jobs declared in a runbook's frontmatter."""
    parsed = runbook_store.read_runbook(runbook_store.runbook_path(slug))
    schedules = parsed.metadata.get("schedules") or []
    synced: list[dict[str, Any]] = []
    for schedule in schedules:
        if not isinstance(schedule, dict):
            continue
        schedule_text = _schedule_text(schedule)
        if not schedule_text:
            continue
        synced.append(_sync_one_schedule(parsed, schedule, schedule_text))
    return synced


def link_existing_cron_job(
    slug: str,
    *,
    profile: str,
    cron_job_id: str,
    schedule_id: str,
    step_key: str,
) -> dict[str, Any]:
    """Attach registry identity to a cron job without changing its behavior."""
    parsed = runbook_store.read_runbook(runbook_store.runbook_path(slug))
    metadata = parsed.metadata
    updates = {
        "workflow_id": metadata["id"],
        "workflow_slug": metadata["slug"],
        "workflow_step_key": step_key,
        "workflow_schedule_id": schedule_id,
        "runbook_slug": metadata["slug"],
    }
    with _cron_store_for_profile(profile):
        from cron import jobs as cron_jobs

        existing = next(
            (
                job
                for job in cron_jobs.list_jobs(include_disabled=True)
                if job.get("id") == cron_job_id
            ),
            None,
        )
        if existing is None:
            raise FileNotFoundError(f"cron job not found: {profile}/{cron_job_id}")
        job = cron_jobs.update_job(cron_job_id, updates)
    if job is None:
        raise RuntimeError(f"cron job disappeared during linkage: {profile}/{cron_job_id}")
    with registry.connect_closing() as conn:
        registry.link_schedule(
            conn,
            metadata["id"],
            profile=profile,
            cron_job_id=cron_job_id,
            enabled=bool(job.get("enabled", True)),
        )
    return job


def control_linked_cron_jobs(
    links: list[dict[str, Any]],
    *,
    action: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Pause/resume linked cron jobs and compensate if any mutation fails.

    Registry links preserve whether a schedule is intended to be enabled. A
    workflow resume therefore never enables a schedule whose canonical link is
    disabled.
    """
    if action not in {"pause", "start", "resume"}:
        raise ValueError(f"unsupported workflow control action: {action}")
    changed: list[tuple[str, str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    try:
        for link in links:
            if action != "pause" and not bool(link.get("enabled", True)):
                continue
            profile = str(link["profile"])
            job_id = str(link["cron_job_id"])
            with _cron_store_for_profile(profile):
                from cron import jobs as cron_jobs

                existing = next(
                    (
                        job
                        for job in cron_jobs.list_jobs(include_disabled=True)
                        if job.get("id") == job_id
                    ),
                    None,
                )
                if existing is None:
                    raise FileNotFoundError(f"linked cron job not found: {profile}/{job_id}")
                snapshot = {
                    key: existing.get(key)
                    for key in (
                        "enabled", "state", "paused_at", "paused_reason", "next_run_at"
                    )
                }
                if action == "pause":
                    updated = cron_jobs.pause_job(job_id, reason)
                else:
                    updated = cron_jobs.resume_job(job_id)
                if updated is None:
                    raise RuntimeError(f"cron control failed: {profile}/{job_id}")
            changed.append((profile, job_id, snapshot))
            results.append(
                {
                    "profile": profile,
                    "cron_job_id": job_id,
                    "enabled": bool(updated.get("enabled", True)),
                    "state": updated.get("state"),
                }
            )
    except Exception:
        for profile, job_id, snapshot in reversed(changed):
            try:
                with _cron_store_for_profile(profile):
                    from cron import jobs as cron_jobs

                    cron_jobs.update_job(job_id, snapshot)
            except Exception:
                # Preserve the original failure; the dashboard audit/error path
                # makes the partial-control condition visible for recovery.
                pass
        raise
    return results


def _schedule_text(schedule: dict[str, Any]) -> str:
    for key in ("schedule", "cron", "value"):
        value = schedule.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sync_one_schedule(
    parsed: ParsedRunbook,
    schedule: dict[str, Any],
    schedule_text: str,
) -> dict[str, Any]:
    metadata = parsed.metadata
    schedule_id = str(
        schedule.get("id")
        or schedule.get("schedule_id")
        or schedule.get("cron_job_id")
        or schedule.get("name")
        or "default"
    )
    profile = str(schedule.get("profile") or metadata["owner_profile"])
    step_key = str(
        schedule.get("step_key")
        or metadata["steps"][0]["step_key"]
    )
    prompt = str(
        schedule.get("prompt")
        or build_runbook_agent_prompt(metadata["slug"], step_key=step_key)
    )
    job_name = str(schedule.get("name") or metadata["title"])
    deliver = schedule.get("deliver")
    enabled = bool(schedule.get("enabled", True))
    runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {}
    payload = {
        "prompt": prompt,
        "schedule": schedule_text,
        "name": job_name,
        "deliver": str(deliver) if deliver else "local",
        "workflow_id": metadata["id"],
        "workflow_slug": metadata["slug"],
        "workflow_step_key": step_key,
        "workflow_schedule_id": schedule_id,
        "runbook_slug": metadata["slug"],
        "track_workflow_status": True,
        "enabled_toolsets": schedule.get("enabled_toolsets"),
        "max_iterations": schedule.get("max_iterations", runtime.get("max_iterations")),
        "runtime_tool_budget": schedule.get(
            "runtime_tool_budget", runtime.get("tool_budget")
        ),
        "workdir": schedule.get("workdir"),
        "provider": schedule.get("provider"),
        "model": schedule.get("model"),
        "reasoning_effort": schedule.get("reasoning_effort"),
        "speed": schedule.get("speed"),
        "base_url": schedule.get("base_url"),
    }
    with _cron_store_for_profile(profile):
        from cron import jobs as cron_jobs

        existing = _find_existing_cron_job(
            cron_jobs.list_jobs(include_disabled=True),
            metadata["id"],
            schedule_id,
            schedule.get("cron_job_id"),
        )
        if existing:
            job = cron_jobs.update_job(existing["id"], _cron_update_payload(payload))
            if job is None:
                raise RuntimeError(f"cron job disappeared during update: {existing['id']}")
        else:
            job = cron_jobs.create_job(**payload)
        if enabled:
            if not job.get("enabled", True) or job.get("state") == "paused":
                job = cron_jobs.resume_job(job["id"])
        else:
            job = cron_jobs.pause_job(job["id"], "disabled in RUNBOOK.md schedule")
    if job is None:
        raise RuntimeError(f"cron job sync failed for runbook schedule {schedule_id}")
    with registry.connect_closing() as conn:
        registry.link_schedule(
            conn,
            metadata["id"],
            profile=profile,
            cron_job_id=job["id"],
            enabled=enabled,
        )
    result = dict(job)
    result["profile"] = profile
    return result


def _cron_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _find_existing_cron_job(
    jobs: list[dict[str, Any]],
    workflow_id: str,
    schedule_id: str,
    cron_job_id: Any,
) -> dict[str, Any] | None:
    cron_job_id_text = str(cron_job_id).strip() if cron_job_id else ""
    for job in jobs:
        if cron_job_id_text and job.get("id") == cron_job_id_text:
            return job
        if (
            job.get("workflow_id") == workflow_id
            and job.get("workflow_schedule_id") == schedule_id
        ):
            return job
    return None


@contextmanager
def _cron_store_for_profile(profile: str):
    from cron import jobs as cron_jobs
    from hermes_cli import profiles as profiles_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    canon = profiles_mod.normalize_profile_name(profile)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise FileNotFoundError(f"Profile {canon!r} does not exist")
    home = profiles_mod.get_profile_dir(canon)
    token = set_hermes_home_override(str(home))
    try:
        with cron_jobs.use_cron_store(home):
            yield
    finally:
        reset_hermes_home_override(token)
