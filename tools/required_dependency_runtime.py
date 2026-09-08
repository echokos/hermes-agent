"""Host-observed outcomes for exact required tool dependencies in one Cron run."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
from threading import Lock
from typing import Any, Iterable


@dataclass
class RequiredDependencyState:
    required: tuple[str, ...]
    outcomes: dict[str, dict[str, dict[int, str]]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _finalized: dict[str, object] | None = field(default=None, repr=False)
    _generation: int = field(default=0, repr=False)

    def begin(self, tool_name: str, invocation: str) -> int | None:
        if tool_name not in self.required:
            return None
        with self._lock:
            if self._finalized is not None:
                return None
            self._generation += 1
            generation = self._generation
            by_argument = self.outcomes.setdefault(tool_name, {})
            by_argument.setdefault(invocation, {})[generation] = "pending"
            return generation

    def complete(
        self,
        tool_name: str,
        invocation: str,
        generation: int,
        outcome: str,
    ) -> None:
        with self._lock:
            if self._finalized is not None:
                return
            attempts = self.outcomes.get(tool_name, {}).get(invocation, {})
            if generation in attempts:
                attempts[generation] = outcome

    def _summary_locked(self) -> dict[str, object]:
        successful = []
        failed = []
        missing = []
        for name in self.required:
            by_argument = self.outcomes.get(name)
            if not by_argument:
                missing.append(name)
                continue
            unresolved = set()
            for attempts in by_argument.values():
                if "pending" in attempts.values():
                    unresolved.add("pending")
                    continue
                latest = attempts[max(attempts)]
                if latest != "success":
                    unresolved.add(latest)
            if unresolved:
                failed.append({"tool": name, "reasons": sorted(unresolved)})
            else:
                successful.append(name)
        return {
            "required": list(self.required),
            "successful": successful,
            "failed": failed,
            "missing": missing,
        }

    def finalize(self) -> dict[str, object]:
        """Freeze and return the sole authoritative summary for this run."""
        with self._lock:
            if self._finalized is None:
                self._finalized = self._summary_locked()
            return self._finalized

    def all_succeeded(self) -> bool:
        summary = self.finalize()
        return bool(self.required) and not summary["failed"] and not summary["missing"]


_ACTIVE: ContextVar[RequiredDependencyState | None] = ContextVar(
    "required_tool_dependencies", default=None
)


@dataclass(frozen=True, repr=False)
class InvocationAttempt:
    state: RequiredDependencyState
    tool_name: str
    invocation: str
    generation: int


def activate(
    required: Iterable[str] | None,
) -> tuple[Token, RequiredDependencyState | None]:
    names = tuple(dict.fromkeys(str(name).strip() for name in (required or ()) if str(name).strip()))
    state = RequiredDependencyState(required=names) if names else None
    return _ACTIVE.set(state), state


def reset(token: Token) -> None:
    _ACTIVE.reset(token)


def _invocation_identity(args: Any) -> str:
    # MCP arguments originate as JSON. The canonical form can contain private
    # values, so it must remain only in the turn-local state above.
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mark_success(attempt: InvocationAttempt | None) -> None:
    if attempt is not None:
        attempt.state.complete(
            attempt.tool_name,
            attempt.invocation,
            attempt.generation,
            "success",
        )


def mark_pending(tool_name: str, args: Any) -> InvocationAttempt | None:
    state = _ACTIVE.get()
    if state is None:
        return None
    invocation = _invocation_identity(args)
    generation = state.begin(tool_name, invocation)
    if generation is None:
        return None
    return InvocationAttempt(state, tool_name, invocation, generation)


def mark_failure(attempt: InvocationAttempt | None, reason: str) -> None:
    if attempt is not None:
        attempt.state.complete(
            attempt.tool_name,
            attempt.invocation,
            attempt.generation,
            str(reason or "tool_error")[:80],
        )
