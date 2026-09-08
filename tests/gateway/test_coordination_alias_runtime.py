"""Runtime coverage for canonical workforce aliases on final-return wakes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.wake import (
    FINAL_RETURN_CONTEXT_METADATA_KEY,
    FINAL_RETURN_DELIVERY_STATE_METADATA_KEY,
    FinalReturnDeliveryState,
)


class _PassedFinalReturnGuard(Exception):
    """Sentinel raised at the first session-dispatch boundary."""


@pytest.fixture
def workforce_alias_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    main_profile = home / "profiles" / "main"
    foo_profile = home / "profiles" / "foo"
    aurora_profile = home / "profiles" / "aurora"
    main_profile.mkdir(parents=True)
    foo_profile.mkdir(parents=True)
    aurora_profile.mkdir(parents=True)

    organization = tmp_path / "organization.yaml"
    organization.write_text(
        f"""
schema_version: 1
agents:
  - agent: elliott
    display_name: Elliott
    status: artifact
    operational: false
    department: null
    function: Owner
    manager: null
    direct_reports: [root, main, aurora]
    mission: Retain final authority
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: null
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: null
  - agent: root
    display_name: Root
    status: active
    operational: true
    department: Operations
    function: Infrastructure
    manager: elliott
    direct_reports: []
    mission: Own infrastructure
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: elliott
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: {main_profile}
  - agent: main
    display_name: Canonical Main
    status: active
    operational: true
    department: Operations
    function: Separate canonical agent
    manager: elliott
    direct_reports: []
    mission: Exercise runtime-name collisions
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: elliott
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: {foo_profile}
  - agent: aurora
    display_name: Aurora
    status: active
    operational: true
    department: Operations
    function: Coordination
    manager: elliott
    direct_reports: []
    mission: Coordinate delivery
    owned_outcomes: []
    authority: []
    prohibited_actions: []
    escalation_target: elliott
    cross_team_request_path: null
    buzz_rooms: []
    profile_path: {aurora_profile}
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(organization))
    return tmp_path


def _final_return_event(
    tmp_path: Path,
    *,
    profile: str,
    loop: asyncio.AbstractEventLoop,
) -> tuple[MessageEvent, FinalReturnDeliveryState]:
    board = tmp_path / "kanban.db"
    board.touch(exist_ok=True)
    context = {
        "request_root_id": "cr_alias_runtime",
        "task_id": "task_alias_runtime",
        "event_id": "1",
        "responsible_agent": "root",
        "db_path": str(board),
    }
    state = FinalReturnDeliveryState(
        context=context,
        completion=loop.create_future(),
    )
    event = MessageEvent(
        text="Return the verified result",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="coordination-origin",
            profile=profile,
        ),
        internal=True,
        metadata={
            FINAL_RETURN_CONTEXT_METADATA_KEY: context,
            FINAL_RETURN_DELIVERY_STATE_METADATA_KEY: state,
        },
    )
    return event, state


@pytest.mark.asyncio
async def test_root_final_return_runs_on_main_profile(workforce_alias_home):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    downstream = MagicMock(side_effect=_PassedFinalReturnGuard)
    runner._session_key_for_source = downstream
    event, state = _final_return_event(
        workforce_alias_home,
        profile="main",
        loop=asyncio.get_running_loop(),
    )

    with pytest.raises(_PassedFinalReturnGuard):
        await runner._handle_message(event)

    downstream.assert_called_once_with(event.source)
    assert state.completion.done() is False


@pytest.mark.asyncio
async def test_root_final_return_is_denied_on_wrong_profile(workforce_alias_home):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    downstream = MagicMock(side_effect=_PassedFinalReturnGuard)
    runner._session_key_for_source = downstream
    event, state = _final_return_event(
        workforce_alias_home,
        profile="aurora",
        loop=asyncio.get_running_loop(),
    )

    assert await runner._handle_message(event) is None

    downstream.assert_not_called()
    assert state.completion.done() is True
    outcome = state.completion.result()
    assert outcome.state == "pending"
    assert outcome.detail == "final-return wake routed to the wrong profile"
