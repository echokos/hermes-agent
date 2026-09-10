"""Gateway intentional-silence token behavior."""

import json
import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionContext, SessionEntry, SessionSource
from gateway.response_filters import (
    is_current_turn_reaction_acknowledgement,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
        message_id="msg-42",
    )


def _event():
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-silent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_invisible_only_placeholder_is_intentional_silence():
    assert is_intentional_silence_response("\u2063")


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def _reaction_ack_result(*, content=None, history_offset=0, **extra):
    if content is None:
        content = json.dumps({
            "success": True,
            "reaction_acknowledgement": True,
            "operation": "telegram_current_turn_reaction",
            "platform": "telegram",
            "chat_id": "-1001",
            "message_id": "msg-42",
        })
    return {
        "failed": False,
        "history_offset": history_offset,
        "messages": [
            {"role": "user", "content": "acknowledge this"},
            {
                "role": "tool",
                "name": "react_to_message",
                "tool_name": "react_to_message",
                "content": content,
            },
            {"role": "assistant", "content": ""},
        ],
        **extra,
    }


def test_reaction_acknowledgement_is_bound_to_current_telegram_turn():
    result = _reaction_ack_result()
    assert is_current_turn_reaction_acknowledgement(result, "", _source())
    assert not is_current_turn_reaction_acknowledgement(
        _reaction_ack_result(content="{}"), "", _source()
    )
    assert not is_current_turn_reaction_acknowledgement(
        _reaction_ack_result(interrupted=True), "", _source()
    )
    assert not is_current_turn_reaction_acknowledgement(
        _reaction_ack_result(history_offset=2), "", _source()
    )
    other_source = _source()
    other_source.message_id = "other-message"
    assert not is_current_turn_reaction_acknowledgement(result, "", other_source)
    assert not is_current_turn_reaction_acknowledgement(result, "substantive reply", _source())


def test_reaction_acknowledgement_survives_a_later_current_turn_tool_result():
    result = _reaction_ack_result()
    result["messages"].insert(-1, {
        "role": "tool",
        "name": "todo",
        "tool_name": "todo",
        "content": '{"success": true}',
    })

    assert is_current_turn_reaction_acknowledgement(result, "", _source())


def test_reaction_acknowledgement_is_checked_before_empty_normalization():
    result = _reaction_ack_result()
    assert gateway_run._is_reaction_only_acknowledgement_before_normalization(
        result, "", _source(), history_offset=0
    )
    assert not gateway_run._is_reaction_only_acknowledgement_before_normalization(
        result, "", _source(), history_offset=2
    )


@pytest.mark.asyncio
async def test_reaction_acknowledgement_suppresses_only_blank_delivery(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "",
        "tools": [],
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "reaction_only_acknowledgement": True,
        **_reaction_ack_result(),
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_outcome", [True, False, RuntimeError("telegram failed")])
async def test_registered_reaction_receipt_crosses_real_turn_lifecycle(
    monkeypatch, tmp_path, adapter_outcome
):
    """Exercise dispatch, host receipt stamping, normalization, and delivery."""
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.registry import registry

    class ReactionAdapter:
        supports_async_delivery = True

        def __init__(self):
            self.calls = []

        def toolsets_for_source(self, _source):
            return None

        def get_pending_message(self, _session_key):
            return None

        async def _set_reaction(self, chat_id, message_id, emoji):
            self.calls.append((chat_id, message_id, emoji))
            if isinstance(adapter_outcome, Exception):
                raise adapter_outcome
            return adapter_outcome

        async def stop_typing(self, _chat_id):
            return None

    class RegisteredReactionAgent:
        last_tool_message = None
        raw_result = None

        def __init__(self, **kwargs):
            from model_tools import get_tool_definitions

            self.tools = get_tool_definitions(
                enabled_toolsets=kwargs.get("enabled_toolsets"),
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id")
            assert "message_reactions" in kwargs.get("enabled_toolsets", [])
            assert "react_to_message" in {
                tool["function"]["name"] for tool in self.tools
            }

        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            tool_call_id = "call-reaction"
            tool_result = registry.dispatch("react_to_message", {"emoji": "👍"})
            tool_message = make_tool_result_message(
                "react_to_message", tool_result, tool_call_id
            )
            self.__class__.last_tool_message = tool_message
            messages = list(conversation_history or []) + [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "react_to_message",
                            "arguments": json.dumps({"emoji": "👍"}),
                        },
                    }],
                },
                tool_message,
                {"role": "assistant", "content": ""},
            ]
            raw_result = {
                "final_response": "",
                "messages": messages,
                "api_calls": 1,
                "failed": False,
                "partial": False,
                "interrupted": False,
                "completed": True,
            }
            self.__class__.raw_result = raw_result
            return raw_result

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RegisteredReactionAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "platform_toolsets": {
                "telegram": ["hermes-telegram", "message_reactions"]
            },
            "display": {
                "tool_progress": "off",
                "interim_assistant_messages": False,
            },
        },
    )

    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner.config,
        "get_home_channel",
        lambda platform: "-1001" if platform == Platform.TELEGRAM else None,
    )
    adapter = ReactionAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    # Gateway startup owns this weak reference; the handler resolves its
    # transport through that production seam rather than a test-only global.
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    runner._set_session_env = gateway_run.GatewayRunner._set_session_env.__get__(runner)
    runner._clear_session_env = gateway_run.GatewayRunner._clear_session_env.__get__(runner)
    source = _source()
    event = MessageEvent(text="acknowledge this", source=source, message_id="msg-42")

    # The registered handler is session-bound. Install the same task-local
    # provenance the outer gateway path sets before it dispatches the agent.
    tokens = runner._set_session_env(
        SessionContext(
            source=source,
            connected_platforms=[Platform.TELEGRAM],
            home_channels={},
            session_key="agent:main:telegram:group:-1001:12345",
            session_id="sess-silent",
        )
    )
    try:
        response = await runner._handle_message_with_agent(
            event, source, "agent:main:telegram:group:-1001:12345", 1
        )
    finally:
        runner._clear_session_env(tokens)

    assert adapter.calls == [("-1001", "msg-42", "👍")]
    assert RegisteredReactionAgent.last_tool_message["name"] == "react_to_message"
    assert RegisteredReactionAgent.last_tool_message["tool_name"] == "react_to_message"
    assert "reaction_only_acknowledgement" not in RegisteredReactionAgent.raw_result
    if adapter_outcome is True:
        assert response == ""
    else:
        assert "no response was generated" in response


@pytest.mark.asyncio
async def test_silence_token_suppresses_delivery_but_preserves_transcript(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert {"role": "assistant", "content": "[SILENT]"}.items() <= appended[-1].items()
    assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_invisible_only_placeholder_suppresses_delivery(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "\u2063",
        "messages": [
            {"role": "user", "content": "question for another room member"},
            {"role": "assistant", "content": "\u2063"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert appended[-1]["content"] == "\u2063"


@pytest.mark.asyncio
async def test_empty_success_still_gets_empty_response_warning(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "no response was generated" in response


@pytest.mark.asyncio
async def test_prose_mentioning_silence_token_is_delivered(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    text = "Use [SILENT] when no answer is needed."
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == text


@pytest.mark.asyncio
async def test_agent_end_hook_includes_model_and_provider(monkeypatch, tmp_path):
    """Gateway hooks receive the actual model/provider for post-turn routing."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "done",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "model": "gpt-5.6-terra",
        "provider": "openai-codex",
    })

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    end_context = next(
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    )
    assert end_context["model"] == "gpt-5.6-terra"
    assert end_context["provider"] == "openai-codex"
