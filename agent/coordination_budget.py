"""Request-wide provider admission, inherited by existing turn/thread contexts."""

from __future__ import annotations

import os
import threading
import uuid
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
    provisional_model_calls: int = 0
    settled_provisional_model_calls: int = 0
    acceptance_scope_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    coordination_acceptance_required: bool = False
    unbudgeted_delegation_started: bool = False
    uncoordinated_materialization_committed: bool = False
    closed: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)


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


def current_coordination_execution() -> tuple[str, str, str] | None:
    """Return the active bound request, task, and purpose without resolving it.

    Unlike :func:`current_coordination_request_id`, this read-only view never
    falls back to process environment, queries SQLite, settles provisional
    calls, or otherwise mutates coordination accounting. It is suitable for
    post-tool guards that must prove they are still inside the validated turn
    scope rather than merely trusting inherited environment variables.
    """
    scope = _scope.get()
    if scope is None:
        return None
    with scope.lock:
        if scope.closed.is_set() or not scope.request_root_id or not scope.task_id:
            return None
        return scope.request_root_id, scope.task_id, scope.purpose


def declares_coordination_acceptance(function_name: str, arguments: object) -> bool:
    """Return whether one parsed native tool call declares root acceptance."""
    if function_name != "kanban_create" or not isinstance(arguments, dict):
        return False
    if not isinstance(arguments.get("coordination"), dict):
        return False
    report = arguments.get("report_to_origin")
    return report is True or str(report).strip().lower() in {"true", "1", "yes"}


def register_declared_coordination_acceptance(*, declared: bool) -> None:
    """Make a parsed acceptance declaration mandatory for this user turn."""
    if not declared:
        return
    scope = _scope.get()
    if scope is None:
        return
    with scope.lock:
        if scope.closed.is_set():
            raise ValueError("coordination turn already ended")
        scope.coordination_acceptance_required = True


def register_uncoordinated_materialization(
    *, created: bool, request_root_id: str | None,
) -> None:
    """Fence later same-turn acceptance after newly committed unbound work."""
    if not created or str(request_root_id or "").strip():
        return
    scope = _scope.get()
    if scope is None:
        return
    with scope.lock:
        if scope.closed.is_set():
            raise ValueError("coordination turn already ended")
        scope.uncoordinated_materialization_committed = True


@contextmanager
def coordination_materialization_binding():
    """Fence task materialization against same-turn request acceptance.

    A parsed coordination declaration makes acceptance mandatory for the rest
    of the turn. Materialization may proceed only after the request commits;
    otherwise it fails before opening the materialization transaction and the
    draft remains recoverable on a later tool round or user turn.
    """
    scope = _current_scope()
    if scope is None:
        yield None, ("", "")
        return
    with scope.lock:
        if scope.closed.is_set():
            raise ValueError("coordination turn already ended")
        _resolve_request_root(scope)
        execution = None
        if scope.request_root_id and scope.task_id:
            execution = (
                scope.request_root_id,
                scope.task_id,
                scope.purpose,
            )
        if execution is None and scope.coordination_acceptance_required:
            raise ValueError(
                "workforce materialization requires successful coordination "
                "acceptance; retry after kanban_create succeeds"
            )
        yield execution, (scope.origin_session_id, scope.origin_message_id)


@dataclass
class CoordinationAcceptanceBinding:
    model_calls: int
    scope_id: str = ""
    existing_request_root_id: str = ""
    request_root_id: str = ""
    task_id: str = ""

    def accept(self, request) -> None:
        if self.existing_request_root_id and self.existing_request_root_id != request.id:
            raise ValueError("a running turn cannot replace its coordination root")
        self.request_root_id = request.id
        self.task_id = request.root_task_id


@contextmanager
def coordination_acceptance_binding():
    """Serialize provisional admission with the accepting SQLite transaction.

    Enter this context BEFORE the DB write transaction and call binding.accept
    inside it. Pass binding.model_calls to the trusted DB factory, which checks
    the requested caps and debits that already-observed usage atomically. The
    runtime binding is published only after the DB transaction commits.
    """
    scope = _scope.get()
    if scope is None:
        yield CoordinationAcceptanceBinding(model_calls=0)
        return
    with scope.lock:
        if scope.closed.is_set():
            raise ValueError("coordination turn already ended")
        if scope.unbudgeted_delegation_started:
            raise ValueError("cannot accept a request after unbudgeted delegation started")
        if scope.uncoordinated_materialization_committed:
            raise ValueError(
                "cannot accept a request after uncoordinated workforce "
                "materialization committed"
            )
        binding = CoordinationAcceptanceBinding(
            model_calls=scope.provisional_model_calls,
            scope_id=scope.acceptance_scope_id,
            existing_request_root_id=scope.request_root_id,
        )
        yield binding
        if binding.request_root_id:
            if scope.request_root_id and scope.request_root_id != binding.request_root_id:
                raise ValueError("a running turn cannot replace its coordination root")
            scope.request_root_id = binding.request_root_id
            scope.task_id = binding.task_id
            scope.settled_provisional_model_calls = binding.model_calls


def admit_delegate_spawn() -> None:
    """Fence generic delegation against acceptance of a bounded workforce root."""
    scope = _current_scope()
    if scope is None:
        return
    with scope.lock:
        if scope.closed.is_set():
            raise ValueError("coordination turn already ended")
        if _resolve_request_root(scope):
            raise ValueError("coordinated child work must use budgeted Kanban dispatch")
        scope.unbudgeted_delegation_started = True


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
        try:
            # Another turn may accept the origin while this turn is finishing.
            with scope.lock:
                if scope.provisional_model_calls > scope.settled_provisional_model_calls:
                    _resolve_request_root(scope)
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
        _settle_provisional_calls(scope)
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
    _settle_provisional_calls(scope)
    return candidate


def _settle_provisional_calls(scope: CoordinationScope) -> None:
    from hermes_cli import kanban_db

    if scope.provisional_model_calls <= scope.settled_provisional_model_calls:
        return
    with kanban_db.connect_closing(scope.db_path) as conn:
        kanban_db.settle_coordination_acceptance_calls(
            conn, scope.request_root_id, scope.acceptance_scope_id,
            scope.provisional_model_calls,
        )
    scope.settled_provisional_model_calls = scope.provisional_model_calls


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
        if scope.closed.is_set():
            raise kanban_db.CoordinationBudgetExceeded(scope.request_root_id, "owning turn ended")
        root_id = _resolve_request_root(scope)
        if not root_id:
            scope.provisional_model_calls += 1
            return None
        with kanban_db.connect_closing(scope.db_path) as conn:
            return kanban_db.charge_coordination_model_call(
                conn, root_id, purpose=scope.purpose, task_id=scope.task_id or None,
            )
