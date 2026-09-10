"""Regression tests for session-scoped model/provider overrides in gateway agents.

These cover the bug where `/model ...` stored a session override, but fresh
agent constructions still resolved model/provider from global config/runtime.
That let helper agents (and cache-miss main agents) route GPT-5.4 to the wrong
provider, e.g. Nous instead of OpenAI Codex.
"""

import asyncio
import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = None
    runner.config = None
    runner._voice_mode = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_approvals = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    return runner


def _codex_override():
    return {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }


def _explode_runtime_resolution():
    raise AssertionError(
        "global runtime resolution should not run when a complete session override exists"
    )


def test_run_agent_prefers_session_override_over_global_runtime(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _explode_runtime_resolution)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )
    session_key = "agent:main:local:dm"
    runner._session_model_overrides[session_key] = _codex_override()
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key=session_key,
        )
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "gpt-5.4"
    assert _CapturingAgent.last_init["provider"] == "openai-codex"
    assert _CapturingAgent.last_init["api_mode"] == "codex_responses"
    assert _CapturingAgent.last_init["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert _CapturingAgent.last_init["api_key"] == "***"
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_matrix_platform_model_default_overrides_global_model(monkeypatch):
    calls = []

    def fake_runtime_resolution(*, requested_provider=None, target_model=None):
        calls.append(
            {
                "requested_provider": requested_provider,
                "target_model": target_model,
            }
        )
        return {
            "provider": requested_provider,
            "api_key": "***",
            "base_url": "https://ollama.com/v1",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", fake_runtime_resolution)
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        chat_name="Matrix room",
        chat_type="group",
        user_id="@elliott:example.org",
    )
    config = {
        "model": {
            "default": "gpt-5.5",
            "provider": "openai-codex",
        },
        "platform_models": {
            "matrix": {
                "default": "glm-5.2:cloud",
                "provider": "ollama-cloud",
            },
        },
    }

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key="agent:main:matrix:group:!room:example.org:@elliott:example.org",
        user_config=config,
    )

    assert model == "glm-5.2:cloud"
    assert runtime["provider"] == "ollama-cloud"
    assert calls == [
        {
            "requested_provider": "ollama-cloud",
            "target_model": "glm-5.2:cloud",
        }
    ]


def test_voice_platform_reasoning_effort_overrides_global_reasoning():
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.VOICE,
        chat_id="voice:vox-grace",
        chat_name="Voice call",
        chat_type="dm",
        user_id="elliott",
    )
    config = {
        "agent": {"reasoning_effort": "medium"},
        "platform_models": {
            "voice": {
                "default": "gpt-5.4-mini",
                "provider": "openai-codex",
                "reasoning_effort": "low",
            },
        },
    }

    reasoning = runner._resolve_session_reasoning_config(
        source=source,
        user_config=config,
    )

    assert reasoning == {"enabled": True, "effort": "low"}


def test_session_model_override_wins_over_platform_model(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _explode_runtime_resolution)
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        chat_name="Matrix room",
        chat_type="group",
        user_id="@elliott:example.org",
    )
    session_key = "agent:main:matrix:group:!room:example.org:@elliott:example.org"
    runner._session_model_overrides[session_key] = _codex_override()
    config = {
        "model": {
            "default": "gpt-5.5",
            "provider": "openai-codex",
        },
        "platform_models": {
            "matrix": {
                "default": "glm-5.2:cloud",
                "provider": "ollama-cloud",
            },
        },
    }

    model, runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
        user_config=config,
    )

    assert model == "gpt-5.4"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"


@pytest.mark.asyncio
async def test_background_task_prefers_session_override_over_global_runtime(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _explode_runtime_resolution)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    adapter = AsyncMock()
    adapter.send = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "ok"))
    adapter.extract_images = MagicMock(return_value=([], "ok"))
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides[session_key] = _codex_override()
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

    await runner._run_background_task("say hello", source, "bg_test")

    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "gpt-5.4"
    assert _CapturingAgent.last_init["provider"] == "openai-codex"
    assert _CapturingAgent.last_init["api_mode"] == "codex_responses"
    assert _CapturingAgent.last_init["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert _CapturingAgent.last_init["api_key"] == "***"
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_gateway_credential_error_does_not_switch_provider(tmp_path, monkeypatch):
    """A missing Codex credential stays terminal at initial resolution."""
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  default: gpt-5.5
  provider: openai-codex
fallback_providers:
  - provider: openrouter
    model: minimax/minimax-m2.7
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    def fake_resolve_runtime_provider(*, requested=None, explicit_base_url=None, explicit_api_key=None, target_model=None):
        if requested in {None, "", "openai-codex"}:
            from hermes_cli.auth import AuthError
            raise AuthError("No Codex credentials stored. Run `hermes auth` to authenticate.")
        assert requested == "openrouter"
        return {
            "api_key": "sk-openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)

    runner = _make_runner()
    with pytest.raises(RuntimeError, match="No Codex credentials stored"):
        runner._resolve_session_agent_runtime(
            session_key="agent:main:telegram:group:-1003715515980:63",
            user_config={
                "model": {"default": "gpt-5.5", "provider": "openai-codex"},
                "fallback_providers": [{"provider": "openrouter", "model": "minimax/minimax-m2.7"}],
            },
        )


def test_gateway_auth_fallback_resolves_key_env_for_custom_provider(tmp_path, monkeypatch):
    """Auth-failure fallback should honor key_env/api_key_env custom-endpoint hints."""
    config = tmp_path / "config.yaml"
    config.write_text(
        """
fallback_providers:
  - provider: custom
    model: fallback-model
    base_url: https://fallback.example/v1
    key_env: MY_FALLBACK_KEY
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("MY_FALLBACK_KEY", "env-secret")

    def fake_resolve_runtime_provider(
        *, requested=None, explicit_base_url=None, explicit_api_key=None, target_model=None
    ):
        assert requested == "custom"
        assert explicit_base_url == "https://fallback.example/v1"
        assert explicit_api_key == "env-secret"
        return {
            "api_key": explicit_api_key,
            "base_url": explicit_base_url,
            "provider": "custom",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)

    runtime_kwargs = gateway_run._try_resolve_fallback_provider()

    assert runtime_kwargs is not None
    assert runtime_kwargs["provider"] == "custom"
    assert runtime_kwargs["api_key"] == "env-secret"
    assert runtime_kwargs["base_url"] == "https://fallback.example/v1"
    assert runtime_kwargs["model"] == "fallback-model"
