"""Host-observed required-workforce-signal outcome state for one Cron turn."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass
class RequiredSignalState:
    required: bool = False
    attempted: bool = False
    failure: str | None = None
    completed: bool = False


_ACTIVE: ContextVar[RequiredSignalState | None] = ContextVar(
    "required_workforce_signal", default=None
)


def activate(
    required: bool,
    *,
    observe_attempts: bool = False,
) -> tuple[Token, RequiredSignalState | None]:
    state = (
        RequiredSignalState(required=required)
        if required or observe_attempts
        else None
    )
    return _ACTIVE.set(state), state


def reset(token: Token) -> None:
    _ACTIVE.reset(token)


def mark_attempted() -> None:
    state = _ACTIVE.get()
    if state is not None:
        state.attempted = True


def mark_failure(message: str) -> None:
    state = _ACTIVE.get()
    if state is not None and not state.completed:
        state.attempted = True
        state.failure = str(message)[:800]


def mark_success() -> None:
    state = _ACTIVE.get()
    if state is not None:
        # A validation-only rejection is recoverable within the same model
        # turn because registry preflight runs before write reservation. Once a
        # later call commits the required signal, that earlier rejection must
        # not poison the host-observed outcome.
        state.failure = None
        state.attempted = True
        state.completed = True
