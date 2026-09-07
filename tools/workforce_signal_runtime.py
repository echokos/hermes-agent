"""Host-observed required-workforce-signal outcome state for one Cron turn."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass
class RequiredSignalState:
    failure: str | None = None
    completed: bool = False


_ACTIVE: ContextVar[RequiredSignalState | None] = ContextVar(
    "required_workforce_signal", default=None
)


def activate(required: bool) -> tuple[Token, RequiredSignalState | None]:
    state = RequiredSignalState() if required else None
    return _ACTIVE.set(state), state


def reset(token: Token) -> None:
    _ACTIVE.reset(token)


def mark_failure(message: str) -> None:
    state = _ACTIVE.get()
    if state is not None and state.failure is None:
        state.failure = str(message)[:800]


def mark_success() -> None:
    state = _ACTIVE.get()
    if state is not None:
        state.completed = True
