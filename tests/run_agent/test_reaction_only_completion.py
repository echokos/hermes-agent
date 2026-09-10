"""Real conversation-loop coverage for reaction-only Telegram turns."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from run_agent import AIAgent


def _tool_call():
    return SimpleNamespace(
        id="reaction-1",
        type="function",
        function=SimpleNamespace(
            name="react_to_message", arguments='{"emoji":"👍"}'
        ),
    )


def _response(content="", *, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )],
        model="test/model",
        usage=None,
    )


def _agent(responses):
    from model_tools import get_tool_definitions

    definitions = get_tool_definitions(
        enabled_toolsets=["message_reactions"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    assert {tool["function"]["name"] for tool in definitions} == {
        "react_to_message"
    }
    with (
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="telegram",
            enabled_toolsets=["message_reactions"],
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = responses
    return agent


@pytest.mark.parametrize(
    "accepted,expected_calls,expected_response",
    [(True, 2, ""), (False, 3, "Recovered after failed reaction.")],
)
def test_real_loop_treats_only_accepted_task_local_reaction_as_complete(
    monkeypatch, accepted, expected_calls, expected_response
):
    """The receipt survives the executor hop and never leaks to a later turn."""
    import gateway.run as gateway_run

    adapter = MagicMock()

    async def set_reaction(chat_id, message_id, emoji):
        assert (chat_id, message_id, emoji) == ("-1001", "42", "👍")
        return accepted

    adapter._set_reaction = set_reaction
    runner = MagicMock()
    runner._adapter_for_source.return_value = adapter
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        message_id="42",
        profile="main",
    )
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        profile="main",
        origin_source=source,
        cron_session="",
    )
    try:
        agent = _agent([
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls"),
            _response(),
            _response("Recovered after failed reaction."),
        ])
        agent._persist_session = MagicMock()
        agent._save_trajectory = MagicMock()
        agent._cleanup_task_resources = MagicMock()
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                context.run, agent.run_conversation, "acknowledge this"
            ).result(timeout=10)
    finally:
        clear_session_vars(tokens)

    assert result["api_calls"] == expected_calls, result["messages"]
    assert result["final_response"] == expected_response
    assert getattr(source, "_explicit_reaction_committed", False) is accepted
    if accepted:
        assert result["turn_exit_reason"] == "current_turn_reaction_acknowledgement"
        assert not any(message.get("_empty_recovery_synthetic") for message in result["messages"])

    later_source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", message_id="43", profile="main"
    )
    assert not getattr(later_source, "_explicit_reaction_committed", False)
