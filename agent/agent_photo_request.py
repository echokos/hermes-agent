"""One-turn authorization state for direct agent-photo requests."""

from __future__ import annotations

import hashlib
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


_CURRENT_AUTHORIZATION: ContextVar["AgentPhotoRequestAuthorization | None"] = (
    ContextVar("agent_photo_request_authorization", default=None)
)
_CURRENT_RUN: ContextVar["AgentPhotoRequestRun | None"] = ContextVar(
    "agent_photo_request_run", default=None
)
_ACTIVE_RUNS_LOCK = threading.Lock()
_ACTIVE_RUNS_ATTRIBUTE = "_agent_photo_request_runs"


@dataclass(frozen=True)
class AgentPhotoRequestOrigin:
    """Immutable identity of the authored request that opened this turn."""

    text: str
    text_sha256: str
    session_id: str
    turn_id: str
    profile_name: str
    profile_path: str
    user_message_index: int


class AgentPhotoRequestAuthorization:
    """Atomically offers one direct request to the agent-photo resolver."""

    __slots__ = ("origin", "_active", "_lock", "_provenance", "_state")

    def __init__(self, origin: AgentPhotoRequestOrigin) -> None:
        self.origin = origin
        self._active = True
        self._lock = threading.Lock()
        self._provenance = None
        self._state = "fresh"

    def begin_review(
        self,
        *,
        session_id: str,
        turn_id: str,
        subject: Any,
    ) -> tuple[str, str]:
        """Claim the single review slot without holding a lock during I/O."""
        profile_name = subject.get("profile_name") if isinstance(subject, dict) else None
        profile_path = subject.get("profile_path") if isinstance(subject, dict) else None
        with self._lock:
            if not self._active:
                return ("expired", "")
            if (
                session_id != self.origin.session_id
                or turn_id != self.origin.turn_id
                or profile_name != self.origin.profile_name
                or profile_path != self.origin.profile_path
            ):
                return ("mismatch", "")
            if self._state != "fresh":
                return ("error" if self._state == "error" else "used", "")
            self._state = "reviewing"
            return ("review", self.origin.text)

    def finish_review(self, verdict: str) -> str:
        """Commit a classifier result only while this turn remains active."""
        with self._lock:
            if not self._active or self._state != "reviewing":
                return "unavailable"
            if verdict == "requested":
                self._state = "approved"
                return "requested"
            if verdict in {"not_requested", "ambiguous"}:
                self._state = "not_authorized"
                return verdict
            self._state = "error"
            return "error"

    def issue_provenance(self, issue) -> Any:
        """Issue and register exact provenance atomically with turn activity."""
        with self._lock:
            if not self._active or self._state != "approved":
                return None
            provenance = issue()
            self._provenance = provenance
            self._state = "consumed"
            return provenance

    def is_active(self) -> bool:
        """Return whether the originating run still permits this request."""
        with self._lock:
            return self._active

    def invalidate(self) -> None:
        """Make copied/late worker references permanently non-authorizing."""
        with self._lock:
            self._active = False
            if self._provenance is not None:
                from tools.approval import _revoke_tool_approval_provenance

                _revoke_tool_approval_provenance(self._provenance)


class AgentPhotoRequestRun:
    """Own the authorization created by exactly one conversation run."""

    __slots__ = ("_active", "_authorization", "_lock")

    def __init__(self) -> None:
        self._active = True
        self._authorization: AgentPhotoRequestAuthorization | None = None
        self._lock = threading.Lock()

    def bind(self, authorization: AgentPhotoRequestAuthorization) -> bool:
        with self._lock:
            if not self._active or self._authorization is not None:
                return False
            self._authorization = authorization
            return True

    def is_active(self) -> bool:
        """Return whether work captured from this run may still continue."""
        with self._lock:
            return self._active

    def finish(self) -> None:
        with self._lock:
            self._active = False
            authorization = self._authorization
            if authorization is not None:
                authorization.invalidate()
            self._authorization = None


@dataclass(frozen=True)
class _AgentPhotoRequestContextTokens:
    authorization: Token[AgentPhotoRequestAuthorization | None]
    run: Token[AgentPhotoRequestRun | None]


def start_agent_photo_request_run(agent: Any) -> tuple[
    AgentPhotoRequestRun,
    _AgentPhotoRequestContextTokens,
]:
    """Start an isolated run and clear any authorization inherited by context."""
    run = AgentPhotoRequestRun()
    with _ACTIVE_RUNS_LOCK:
        active_runs = getattr(agent, _ACTIVE_RUNS_ATTRIBUTE, None)
        if not isinstance(active_runs, set):
            active_runs = set()
            setattr(agent, _ACTIVE_RUNS_ATTRIBUTE, active_runs)
        active_runs.add(run)
    run_token = _CURRENT_RUN.set(run)
    try:
        authorization_token = _CURRENT_AUTHORIZATION.set(None)
    except BaseException:
        _CURRENT_RUN.reset(run_token)
        with _ACTIVE_RUNS_LOCK:
            active_runs.discard(run)
        run.finish()
        raise
    return run, _AgentPhotoRequestContextTokens(authorization_token, run_token)


def bind_agent_photo_request(
    run: AgentPhotoRequestRun,
    text: Any,
    *,
    session_id: str,
    turn_id: str,
    user_message_index: int,
) -> AgentPhotoRequestAuthorization | None:
    """Bind trusted authored text to the active profile and turn."""
    if not isinstance(run, AgentPhotoRequestRun):
        return None
    if not isinstance(text, str) or not text.strip() or not session_id or not turn_id:
        return None
    try:
        from hermes_cli.profiles import get_active_profile_name
        from hermes_constants import get_hermes_home

        profile_name = get_active_profile_name()
        profile_path = str(get_hermes_home().expanduser().resolve())
    except Exception:
        return None
    clean = text.strip()
    authorization = AgentPhotoRequestAuthorization(
        AgentPhotoRequestOrigin(
            text=clean,
            text_sha256=hashlib.sha256(clean.encode("utf-8")).hexdigest(),
            session_id=session_id,
            turn_id=turn_id,
            profile_name=profile_name,
            profile_path=profile_path,
            user_message_index=user_message_index,
        )
    )
    if not run.bind(authorization):
        authorization.invalidate()
        return None
    _CURRENT_AUTHORIZATION.set(authorization)
    return authorization


def finish_agent_photo_request_run(
    agent: Any,
    run: AgentPhotoRequestRun,
    token: _AgentPhotoRequestContextTokens,
) -> None:
    """Invalidate only this run's capability and restore its caller context."""
    try:
        run.finish()
    finally:
        try:
            with _ACTIVE_RUNS_LOCK:
                active_runs = getattr(agent, _ACTIVE_RUNS_ATTRIBUTE, None)
                if isinstance(active_runs, set):
                    active_runs.discard(run)
        finally:
            try:
                _CURRENT_AUTHORIZATION.reset(token.authorization)
            finally:
                _CURRENT_RUN.reset(token.run)


def revoke_agent_photo_request_runs(agent: Any) -> None:
    """Revoke unconsumed request authority on every active run for this agent."""
    with _ACTIVE_RUNS_LOCK:
        active_runs = tuple(getattr(agent, _ACTIVE_RUNS_ATTRIBUTE, ()) or ())
    for run in active_runs:
        if isinstance(run, AgentPhotoRequestRun):
            run.finish()


def get_current_agent_photo_request_authorization() -> AgentPhotoRequestAuthorization | None:
    """Return the turn-start snapshot propagated into tool worker contexts."""
    return _CURRENT_AUTHORIZATION.get()


def get_current_agent_photo_request_run() -> AgentPhotoRequestRun | None:
    """Return the shared lifetime guard propagated into tool worker contexts."""
    return _CURRENT_RUN.get()
