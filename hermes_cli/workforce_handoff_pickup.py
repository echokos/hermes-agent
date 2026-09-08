"""Bounded, internal CLI pickup for one owned workforce handoff."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, BinaryIO

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


def _canonical_execution_profile(
    value: str | None, *, target_agent: str,
) -> str:
    """Validate the concrete profile directory for a canonical target actor."""
    from hermes_cli.profiles import normalize_profile_name

    organization = load_organization()
    target = organization.validate_execution_profile(target_agent)
    if value is None:
        if not target.profile_path:
            raise ValueError("target workforce agent has no execution profile")
        candidate = normalize_profile_name(Path(target.profile_path).name)
    else:
        candidate = normalize_profile_name(value)
    resolved = organization.validate_execution_profile(candidate)
    declared_profile = (
        Path(resolved.profile_path).name.casefold() if resolved.profile_path else ""
    )
    if resolved.agent != target_agent or declared_profile != candidate:
        raise ValueError("execution_profile does not match the target workforce agent")
    return candidate


def _bounded_identifier(value: str, *, prefix: str) -> str:
    candidate = str(value or "").strip()
    if not candidate.startswith(prefix) or len(candidate) > 160:
        raise ValueError(f"{prefix} identifier is invalid")
    if not all(char.isascii() and (char.isalnum() or char in "_-") for char in candidate):
        raise ValueError(f"{prefix} identifier is invalid")
    return candidate


def _open_pickup_log(database_path: Path, task_id: str) -> tuple[Path, BinaryIO]:
    """Atomically create one owner-only, no-follow log for the one-shot pickup."""
    if IS_WINDOWS:
        raise OSError("workforce handoff pickup requires POSIX descriptor permissions")
    log_dir = database_path.resolve().parent / "workforce-handoff-pickups"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = log_dir / f"{task_id}.log"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(log_dir, directory_flags)
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise ValueError("pickup log directory is not a real directory")
            os.fchmod(directory_fd, 0o700)
            fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    else:
        directory_stat = os.lstat(log_dir)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("pickup log directory is not a real directory")
        os.chmod(log_dir, 0o700)
        fd = os.open(path, flags, 0o600)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("pickup log path is not a regular file")
        os.fchmod(fd, 0o600)
        return path, os.fdopen(fd, "ab", buffering=0)
    except BaseException:
        os.close(fd)
        raise


async def _wait_for_process(proc: subprocess.Popen, timeout: float) -> int:
    """Bound the worker thread itself, not only the coroutine awaiting it."""
    return await asyncio.to_thread(proc.wait, timeout=timeout)


async def _terminate_and_reap(proc: subprocess.Popen) -> int | None:
    """Kill the dedicated group and make a bounded best effort to reap it."""
    kill_process_tree(proc)
    try:
        return await _wait_for_process(proc, 1)
    except subprocess.TimeoutExpired:
        return getattr(proc, "returncode", None)


def _pickup_env(
    *,
    database_path: Path,
    task_id: str,
    request_root_id: str,
    target_agent: str,
    execution_profile: str | None = None,
    source_agent: str,
) -> dict[str, str]:
    profile = execution_profile or target_agent
    env = dict(os.environ)
    from gateway.session_context import _VAR_MAP

    for key in _VAR_MAP:
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith("HERMES_KANBAN_"):
            env.pop(key, None)

    env.update({
        "HERMES_HOME": resolve_profile_env(profile),
        "HERMES_PROFILE": profile,
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


def _pickup_command(
    *,
    target_agent: str,
    request_root_id: str,
    task_id: str,
    execution_profile: str | None = None,
) -> list[str]:
    profile = execution_profile or target_agent
    return [
        *_resolve_hermes_argv(),
        "-p", profile,
        "--cli",
        "chat",
        "-Q",
        "-c", f"{_SESSION_TITLE_PREFIX}{request_root_id}:{target_agent}",
        "--create-if-missing",
        "--no-restore-cwd",
        "-t", "workforce",
        "--max-turns", "2",
        "-q", (
            "Acknowledge exactly the assigned workforce handoff "
            f"{task_id} using workforce_handoff. Do not take any other action."
        ),
    ]


def _fresh_acknowledgment(
    *,
    database_path: Path,
    task_id: str,
    request_root_id: str,
    target_agent: str,
    source_agent: str,
) -> bool:
    """Require an acknowledgment after this exact pickup claim.

    The dispatcher can advance the task from ``accepted`` immediately after
    the tool commits, so event ordering is authoritative rather than its
    current body state or the child's eventual exit code.
    """
    from hermes_cli import kanban_db

    try:
        with kanban_db.connect_closing(database_path) as conn:
            task = kanban_db.get_task(conn, task_id)
            if task is None:
                return False
            payload = json.loads(task.body or "{}")
            if not isinstance(payload, dict) or payload.get("kind") != "workforce_handoff":
                return False
            claim_seen = False
            for event in kanban_db.list_events(conn, task_id):
                if event.kind == "workforce_handoff_pickup_claimed":
                    claim = event.payload
                    claim_seen = bool(
                        isinstance(claim, dict)
                        and claim.get("actor") == target_agent
                        and claim.get("target_agent") == target_agent
                        and claim.get("source_agent") == source_agent
                        and claim.get("request_root_id") == request_root_id
                        and claim.get("claim_kind") == "owned_operational_failure"
                    )
                    continue
                if (
                    claim_seen
                    and event.kind == "workforce_handoff_acknowledged"
                    and isinstance(event.payload, dict)
                    and event.payload.get("actor") == target_agent
                ):
                    return True
            return False
    except Exception:
        return False


async def run_workforce_handoff_pickup(
    *,
    task_id: str,
    request_root_id: str,
    target_agent: str,
    execution_profile: str | None = None,
    source_agent: str,
    database_path: Path,
) -> WorkforceHandoffPickupResult:
    """Run one bounded silent-owner turn and verify its durable acknowledgment."""
    task_id = _bounded_identifier(task_id, prefix="t_")
    request_root_id = _bounded_identifier(request_root_id, prefix="cr_")
    target_agent = _canonical_agent(target_agent, field="target_agent")
    source_agent = _canonical_agent(source_agent, field="source_agent")
    execution_profile = _canonical_execution_profile(
        execution_profile,
        target_agent=target_agent,
    )
    db_path = Path(database_path).expanduser()
    if not db_path.is_absolute() or not db_path.is_file():
        raise ValueError("database_path must be an existing absolute file")
    env = _pickup_env(
        database_path=db_path,
        task_id=task_id,
        request_root_id=request_root_id,
        target_agent=target_agent,
        execution_profile=execution_profile,
        source_agent=source_agent,
    )
    command = _pickup_command(
        target_agent=target_agent,
        execution_profile=execution_profile,
        request_root_id=request_root_id,
        task_id=task_id,
    )
    log_path, log_file = _open_pickup_log(db_path, task_id)
    with log_file:
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
            returncode = await _wait_for_process(proc, PICKUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            returncode = await _terminate_and_reap(proc)
            acknowledged = _fresh_acknowledgment(
                database_path=db_path,
                task_id=task_id,
                request_root_id=request_root_id,
                target_agent=target_agent,
                source_agent=source_agent,
            )
            return WorkforceHandoffPickupResult(
                acknowledged=acknowledged,
                timed_out=True,
                returncode=returncode,
                log_path=log_path,
                reason=(
                    "pickup timed out after durable acknowledgment"
                    if acknowledged else "pickup timed out"
                ),
            )
        except asyncio.CancelledError:
            # Do not leave the tool-scoped child alive when its caller exits.
            await _terminate_and_reap(proc)
            raise

    acknowledged = _fresh_acknowledgment(
        database_path=db_path,
        task_id=task_id,
        request_root_id=request_root_id,
        target_agent=target_agent,
        source_agent=source_agent,
    )
    return WorkforceHandoffPickupResult(
        acknowledged=acknowledged,
        timed_out=False,
        returncode=returncode,
        log_path=log_path,
        reason=(
            None if returncode == 0 and acknowledged
            else f"pickup exited with {returncode} after durable acknowledgment"
            if acknowledged
            else "pickup exited without a durable acknowledgment"
        ),
    )
