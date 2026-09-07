"""Bounded, internal CLI pickup for one owned workforce handoff."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from hermes_cli._subprocess_compat import IS_WINDOWS, kill_process_tree
from hermes_cli.kanban_db import _resolve_hermes_argv
from hermes_cli.profiles import resolve_profile_env
from hermes_cli.workforce_org import load_organization


PICKUP_TIMEOUT_SECONDS = 120
_SESSION_TITLE_PREFIX = "workforce-handoff:"


@dataclass(frozen=True)
class WorkforceHandoffPickupResult:
    acknowledged: bool
    timed_out: bool
    returncode: int | None
    log_path: Path
    reason: str | None = None


def _canonical_agent(value: str, *, field: str) -> str:
    candidate = str(value or "").strip().casefold()
    if not candidate or candidate != str(value or "").strip():
        raise ValueError(f"{field} must be a canonical workforce agent")
    return load_organization().validate_execution_profile(candidate).agent


def _bounded_identifier(value: str, *, prefix: str) -> str:
    candidate = str(value or "").strip()
    if not candidate.startswith(prefix) or len(candidate) > 160:
        raise ValueError(f"{prefix} identifier is invalid")
    if not all(char.isascii() and (char.isalnum() or char in "_-") for char in candidate):
        raise ValueError(f"{prefix} identifier is invalid")
    return candidate


def _pickup_log_path(database_path: Path, task_id: str) -> Path:
    log_dir = database_path.resolve().parent / "workforce-handoff-pickups"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        log_dir.chmod(0o700)
    except OSError:
        pass
    path = log_dir / f"{task_id}.log"
    if path.exists() and not path.is_file():
        raise ValueError("pickup log path is not a regular file")
    return path


def _pickup_env(
    *,
    database_path: Path,
    task_id: str,
    request_root_id: str,
    target_agent: str,
    source_agent: str,
) -> dict[str, str]:
    env = dict(os.environ)
    from gateway.session_context import _VAR_MAP

    for key in _VAR_MAP:
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith("HERMES_KANBAN_"):
            env.pop(key, None)

    env.update({
        "HERMES_HOME": resolve_profile_env(target_agent),
        "HERMES_PROFILE": target_agent,
        # This must exist before cmd_chat resolves --continue/create-if-missing.
        "HERMES_SESSION_SOURCE": "tool",
        "HERMES_KANBAN_DB": str(database_path.resolve()),
        "HERMES_COORDINATION_REQUEST_ROOT": request_root_id,
        "HERMES_COORDINATION_TASK_ID": task_id,
        "HERMES_COORDINATION_PURPOSE": "work",
        "HERMES_WORKFORCE_HANDOFF_PICKUP_TASK": task_id,
        "HERMES_WORKFORCE_HANDOFF_PICKUP_TARGET": target_agent,
        "HERMES_WORKFORCE_HANDOFF_PICKUP_SOURCE": source_agent,
    })
    env.pop("HERMES_TUI", None)
    return env


def _pickup_command(*, target_agent: str, request_root_id: str, task_id: str) -> list[str]:
    return [
        *_resolve_hermes_argv(),
        "-p", target_agent,
        "--cli",
        "chat",
        "-Q",
        "-c", f"{_SESSION_TITLE_PREFIX}{request_root_id}:{target_agent}",
        "--create-if-missing",
        "--no-restore-cwd",
        "-t", "workforce",
        "--max-turns", "1",
        "-q", (
            "Acknowledge exactly the assigned workforce handoff "
            f"{task_id} using workforce_handoff. Do not take any other action."
        ),
    ]


def _fresh_acknowledgment(
    *, database_path: Path, task_id: str, target_agent: str
) -> bool:
    """Require the child to have changed this exact handoff through its tool."""
    from hermes_cli import kanban_db

    try:
        with kanban_db.connect_closing(database_path) as conn:
            task = kanban_db.get_task(conn, task_id)
            if task is None:
                return False
            payload = json.loads(task.body or "{}")
            if not isinstance(payload, dict) or (
                payload.get("kind") != "workforce_handoff"
                or payload.get("state") != "accepted"
                or payload.get("target_agent") != target_agent
            ):
                return False
            return any(
                event.kind == "workforce_handoff_acknowledged"
                and isinstance(event.payload, dict)
                and event.payload.get("actor") == target_agent
                for event in kanban_db.list_events(conn, task_id)
            )
    except Exception:
        return False


async def run_workforce_handoff_pickup(
    *,
    task_id: str,
    request_root_id: str,
    target_agent: str,
    source_agent: str,
    database_path: Path,
) -> WorkforceHandoffPickupResult:
    """Run one bounded silent-owner turn and verify its durable acknowledgment."""
    task_id = _bounded_identifier(task_id, prefix="t_")
    request_root_id = _bounded_identifier(request_root_id, prefix="cr_")
    target_agent = _canonical_agent(target_agent, field="target_agent")
    source_agent = _canonical_agent(source_agent, field="source_agent")
    db_path = Path(database_path).expanduser()
    if not db_path.is_absolute() or not db_path.is_file():
        raise ValueError("database_path must be an existing absolute file")
    log_path = _pickup_log_path(db_path, task_id)
    env = _pickup_env(
        database_path=db_path,
        task_id=task_id,
        request_root_id=request_root_id,
        target_agent=target_agent,
        source_agent=source_agent,
    )
    command = _pickup_command(
        target_agent=target_agent, request_root_id=request_root_id, task_id=task_id
    )
    with open(log_path, "ab", buffering=0) as log_file:
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
        try:
            proc = subprocess.Popen(  # noqa: S603 -- fixed CLI argv and validated ids
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
        except OSError as exc:
            return WorkforceHandoffPickupResult(
                acknowledged=False,
                timed_out=False,
                returncode=None,
                log_path=log_path,
                reason=f"spawn failed: {type(exc).__name__}",
            )
        try:
            returncode = await asyncio.wait_for(
                asyncio.to_thread(proc.wait), timeout=PICKUP_TIMEOUT_SECONDS
            )
        except TimeoutError:
            kill_process_tree(proc)
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=1)
            except Exception:
                pass
            return WorkforceHandoffPickupResult(
                acknowledged=False,
                timed_out=True,
                returncode=None,
                log_path=log_path,
                reason="pickup timed out",
            )

    acknowledged = returncode == 0 and _fresh_acknowledgment(
        database_path=db_path, task_id=task_id, target_agent=target_agent
    )
    return WorkforceHandoffPickupResult(
        acknowledged=acknowledged,
        timed_out=False,
        returncode=returncode,
        log_path=log_path,
        reason=None if acknowledged else "pickup exited without a durable acknowledgment",
    )
