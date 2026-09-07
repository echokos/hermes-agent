"""Host-enforced tool budgets for bounded unattended agent runs.

The budget is carried in a ContextVar so a cron run can install it before the
agent hops to its worker thread.  ToolRegistry is the single enforcement
boundary: prompt instructions cannot enlarge or bypass these limits.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import threading
from typing import Any


class RuntimeToolBudgetError(PermissionError):
    """Raised when an unattended run exceeds its host-issued tool budget."""


@dataclass
class RuntimeToolBudget:
    max_calls: int
    max_writes: int
    max_detail_reads: int
    max_list_items: int
    allowed_tools: frozenset[str]
    calls: int = 0
    writes: int = 0
    detail_reads: int = 0
    denied: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "writes": self.writes,
                "detail_reads": self.detail_reads,
                "denied": self.denied,
                "max_calls": self.max_calls,
                "max_writes": self.max_writes,
                "max_detail_reads": self.max_detail_reads,
                "max_list_items": self.max_list_items,
            }


_ACTIVE_BUDGET: ContextVar[RuntimeToolBudget | None] = ContextVar(
    "hermes_runtime_tool_budget", default=None
)

_DETAIL_READ_TOOLS = frozenset({"kanban_show", "kanban_attachments", "runbook_get"})
_WRITE_TOOLS = frozenset(
    {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
        "kanban_heartbeat",
        "kanban_comment",
        "kanban_archive_stale",
        "kanban_attach",
        "kanban_attach_url",
        "kanban_create",
        "kanban_unblock",
        "kanban_link",
        "workforce_signal",
        "workforce_goals:publish",
        "workforce_vision:request",
        "workforce_vision:respond",
        "workforce_handoff",
        "runbook_propose_create",
        "runbook_propose_edit",
    }
)


def activate_runtime_tool_budget(
    config: dict[str, Any] | None,
) -> tuple[Token, RuntimeToolBudget | None]:
    """Install a validated runtime budget and return its reset token/state."""
    if not config:
        return _ACTIVE_BUDGET.set(None), None
    if not isinstance(config, dict):
        raise ValueError("runtime_tool_budget must be a mapping")

    def positive_int(name: str) -> int:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"runtime_tool_budget.{name} must be a positive integer")
        return value

    raw_allowed = config.get("allowed_tools")
    if not isinstance(raw_allowed, list) or not raw_allowed:
        raise ValueError("runtime_tool_budget.allowed_tools must be a non-empty list")
    allowed = frozenset(str(item).strip() for item in raw_allowed if str(item).strip())
    if len(allowed) != len(raw_allowed):
        raise ValueError("runtime_tool_budget.allowed_tools must contain unique names")
    state = RuntimeToolBudget(
        max_calls=positive_int("max_calls"),
        max_writes=positive_int("max_writes"),
        max_detail_reads=positive_int("max_detail_reads"),
        max_list_items=positive_int("max_list_items"),
        allowed_tools=allowed,
    )
    return _ACTIVE_BUDGET.set(state), state


def reset_runtime_tool_budget(token: Token) -> None:
    _ACTIVE_BUDGET.reset(token)


def charge_runtime_tool_attempt(name: str) -> bool:
    """Charge one allowed invocation before tool-specific input preflight."""
    budget = _ACTIVE_BUDGET.get()
    if budget is None:
        return False
    with budget._lock:
        if name not in budget.allowed_tools:
            budget.denied += 1
            raise RuntimeToolBudgetError(f"tool {name!r} is not allowed for this bounded run")
        if budget.calls >= budget.max_calls:
            budget.denied += 1
            raise RuntimeToolBudgetError(
                f"tool-call budget exhausted ({budget.max_calls} calls)"
            )
        budget.calls += 1
    return True


def enforce_runtime_tool_budget(
    name: str,
    args: dict[str, Any],
    *,
    attempt_charged: bool = False,
) -> dict[str, Any]:
    """Reserve post-preflight sub-budgets and clamp bounded list arguments."""
    budget = _ACTIVE_BUDGET.get()
    if budget is None:
        return args
    if not attempt_charged:
        charge_runtime_tool_attempt(name)
    with budget._lock:
        action_name = f"{name}:{str(args.get('action') or '').strip()}"
        is_write = name in _WRITE_TOOLS or action_name in _WRITE_TOOLS
        is_detail = name in _DETAIL_READ_TOOLS
        if is_write and budget.writes >= budget.max_writes:
            budget.denied += 1
            raise RuntimeToolBudgetError(
                f"write budget exhausted ({budget.max_writes} write)"
            )
        if is_detail and budget.detail_reads >= budget.max_detail_reads:
            budget.denied += 1
            raise RuntimeToolBudgetError(
                f"detail-read budget exhausted ({budget.max_detail_reads} reads)"
            )
        if is_write:
            budget.writes += 1
        if is_detail:
            budget.detail_reads += 1

    bounded_args = dict(args)
    if name in {"kanban_list", "workforce_vision", "workforce_observe_buzz"}:
        requested = bounded_args.get("limit")
        try:
            requested_limit = int(requested) if requested is not None else budget.max_list_items
        except (TypeError, ValueError):
            requested_limit = budget.max_list_items
        bounded_args["limit"] = min(max(requested_limit, 1), budget.max_list_items)
    return bounded_args
