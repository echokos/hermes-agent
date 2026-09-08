"""Active-turn controls revoke opening-message agent-photo authority."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent.agent_photo_request import (
    bind_agent_photo_request,
    finish_agent_photo_request_run,
    start_agent_photo_request_run,
)
from run_agent import AIAgent


def _agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._hard_interrupt_requested = threading.Event()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


@contextmanager
def _photo_authorization(monkeypatch, agent):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "kourtnie"
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: Path("/profiles/kourtnie")
    )
    run, token = start_agent_photo_request_run(agent)
    authorization = bind_agent_photo_request(
        run,
        "send me a photo",
        session_id="session-1",
        turn_id="turn-1",
        user_message_index=1,
    )
    try:
        yield authorization
    finally:
        finish_agent_photo_request_run(agent, run, token)


def _review_state(authorization) -> str:
    return authorization.begin_review(
        session_id="session-1",
        turn_id="turn-1",
        subject={
            "profile_name": "kourtnie",
            "profile_path": "/profiles/kourtnie",
        },
    )[0]


def test_accepted_steer_revokes_opening_photo_request(monkeypatch):
    agent = _agent()
    with _photo_authorization(monkeypatch, agent) as authorization:
        assert agent.steer("do not generate that") is True
        assert _review_state(authorization) == "expired"


def test_ignored_empty_steer_preserves_opening_photo_request(monkeypatch):
    agent = _agent()
    with _photo_authorization(monkeypatch, agent) as authorization:
        assert agent.steer("  ") is False
        assert _review_state(authorization) == "review"


def test_accepted_redirect_revokes_opening_photo_request(monkeypatch):
    agent = _agent()
    agent._model_request_active.set()
    with _photo_authorization(monkeypatch, agent) as authorization:
        assert agent.redirect("do not generate that") is True
        assert _review_state(authorization) == "expired"


def test_rejected_redirect_preserves_opening_photo_request(monkeypatch):
    agent = _agent()
    with _photo_authorization(monkeypatch, agent) as authorization:
        assert agent.redirect("do not generate that") is False
        assert _review_state(authorization) == "review"


@pytest.mark.parametrize(
    "interrupt",
    [
        pytest.param(lambda agent: agent.interrupt("replacement"), id="message"),
        pytest.param(lambda agent: agent.hard_interrupt(), id="control-stop"),
    ],
)
def test_interrupt_revokes_opening_photo_request(monkeypatch, interrupt):
    agent = _agent()
    with _photo_authorization(monkeypatch, agent) as authorization:
        interrupt(agent)
        assert _review_state(authorization) == "expired"
