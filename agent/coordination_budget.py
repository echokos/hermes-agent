"""Request-wide provider admission, inherited by existing turn/thread contexts."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoordinationScope:
    db_path: Path
    request_root_id: str = ""
    task_id: str = ""
    purpose: str = "work"
    origin_session_id: str = ""
    origin_message_id: str = ""
    closed: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


_scope: ContextVar[CoordinationScope | None] = ContextVar(
    "coordination_budget_scope", default=None
)


def _environment_scope(session_id: str = "") -> CoordinationScope:
    from gateway.session_context import get_session_env
    from hermes_cli.kanban_db import kanban_db_path

    root = os.environ.get("HERMES_COORDINATION_REQUEST_ROOT", "")
    task = os.environ.get("HERMES_COORDINATION_TASK_ID", "")
    purpose = os.environ.get("HERMES_COORDINATION_PURPOSE", "work")
    if (root or task or purpose != "work") and not (root and task):
        raise ValueError("incomplete coordination execution scope")
    origin_session_id = (
        get_session_env("HERMES_SESSION_CHAT_ID", "")
        if get_session_env("HERMES_SESSION_PLATFORM", "") == "api_server"
        else ""
    ) or get_session_env("HERMES_SESSION_ID", "") or session_id
    return CoordinationScope(
        db_path=kanban_db_path(), request_root_id=root, task_id=task,
        purpose=purpose,
        origin_session_id=origin_session_id,
        origin_message_id=get_session_env("HERMES_SESSION_MESSAGE_ID", ""),
    )


def current_coordination_origin() -> tuple[str, str]:
    """Return the turn's stable origin, even after compression rotates sessions."""
    scope = _scope.get()
    if scope is None:
        return "", ""
    return scope.origin_session_id, scope.origin_message_id


@contextmanager
def scoped_coordination_budget(
    *, session_id: str = "", request_root_id: str | None = None,
    task_id: str = "", purpose: str = "work", db_path: Path | None = None,
):
    """Bind a trusted worker/final wake, or capture the current origin turn.

    Nested in-process agents inherit the same budget and lifetime. Explicit
    roots are reserved for host dispatch/wake code, never model tool arguments.
    """
    inherited = _scope.get()
    if inherited is not None and request_root_id is None:
        yield inherited
        return
    if request_root_id is not None:
        if not request_root_id or not task_id or db_path is None:
            raise ValueError("explicit coordination scope requires root, task and DB")
        scope = CoordinationScope(
            db_path=Path(db_path), request_root_id=request_root_id,
            task_id=task_id, purpose=purpose,
        )
    else:
        scope = _environment_scope(session_id)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        scope.closed.set()
        _scope.reset(token)


def _current_scope() -> CoordinationScope | None:
    scope = _scope.get()
    if scope is None:
        # Headless worker initialization may perform auxiliary work before the
        # conversation loop. Its process-scoped dispatch envelope still applies.
        if not any(os.environ.get(key) for key in (
            "HERMES_COORDINATION_REQUEST_ROOT", "HERMES_COORDINATION_TASK_ID",
            "HERMES_COORDINATION_PURPOSE",
        )):
            return None
        scope = _environment_scope()
    return scope


def _resolve_request_root(scope: CoordinationScope) -> str:
    """Resolve under scope.lock; tool-thread acceptance is visible via SQLite."""
    from hermes_cli import kanban_db

    if scope.request_root_id:
        return scope.request_root_id
    if not (scope.origin_session_id and scope.origin_message_id):
        return ""
    if not scope.db_path.exists():
        return ""
    candidate = kanban_db.coordination_request_id(
        scope.origin_session_id, scope.origin_message_id
    )
    with kanban_db.connect_closing(scope.db_path) as conn:
        request = kanban_db.get_coordination_request(conn, candidate)
        if request is None:
            return ""
    scope.request_root_id = candidate
    scope.task_id = request.root_task_id
    return candidate


def current_coordination_request_id() -> str:
    """Non-charging lookup for deterministic dispatch/tool admission guards."""
    scope = _current_scope()
    if scope is None:
        return ""
    with scope.lock:
        return _resolve_request_root(scope)


def charge_provider_attempt() -> int | None:
    """Reserve one call durably before network I/O; never fail open for a root."""
    from hermes_cli import kanban_db

    scope = _current_scope()
    if scope is None:
        return None
    with scope.lock:
        root_id = _resolve_request_root(scope)
        if not root_id:
            return None
        if scope.closed.is_set():
            raise kanban_db.CoordinationBudgetExceeded(root_id, "owning turn ended")
        with kanban_db.connect_closing(scope.db_path) as conn:
            return kanban_db.charge_coordination_model_call(
                conn, root_id, purpose=scope.purpose, task_id=scope.task_id or None,
            )
