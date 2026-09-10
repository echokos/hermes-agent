"""Tests for eligible initial provider-resolution fallback."""

from unittest.mock import patch

import pytest


class TestResolveRuntimeAgentKwargsFallback:
    """Only service/rate resolution failures may use configured fallback."""

    def test_codex_usage_limit_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """A Codex subscription usage limit with a reset remains eligible."""
        from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: openrouter\n"
            "  model: meta-llama/llama-4-maverick\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path with explicit_api_key + explicit_base_url
            # supplied by gateway from fallback_model config.
            if call_count["n"] == 1:
                raise AuthError(
                    "Codex usage limit reached; retry after 120s",
                    provider="openai-codex",
                    code=CODEX_RATE_LIMITED_CODE,
                )
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        # Should have been called at least twice (primary + fallback)
        assert call_count["n"] >= 2

    def test_credential_auth_error_does_not_try_fallback(self, tmp_path, monkeypatch):
        from hermes_cli.auth import AuthError
        from gateway.run import _resolve_runtime_agent_kwargs

        (tmp_path / "config.yaml").write_text(
            "fallback_model:\n  provider: openrouter\n  model: test-model\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError("Codex credentials missing", relogin_required=True),
        ) as resolve:
            with pytest.raises(RuntimeError, match="credentials missing"):
                _resolve_runtime_agent_kwargs()

        assert resolve.call_count == 1

    def test_server_error_tries_fallback(self, tmp_path, monkeypatch):
        from gateway.run import _resolve_runtime_agent_kwargs

        (tmp_path / "config.yaml").write_text(
            "fallback_model:\n  provider: openrouter\n  model: test-model\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        class _ServiceError(Exception):
            status_code = 503

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=[
                _ServiceError("upstream overloaded"),
                {
                    "api_key": "fallback-key",
                    "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                    "command": None,
                    "args": [],
                    "credential_pool": None,
                },
            ],
        ):
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"

    def test_wrapped_oauth_503_tries_fallback(self, tmp_path, monkeypatch):
        from gateway.run import _resolve_runtime_agent_kwargs
        from hermes_cli.auth import AuthError

        (tmp_path / "config.yaml").write_text(
            "fallback_model:\n  provider: openrouter\n  model: test-model\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=[
                AuthError(
                    "Codex token refresh failed: temporarily unavailable",
                    provider="openai-codex",
                    code="server_error",
                    status_code=503,
                ),
                {
                    "api_key": "fallback-key",
                    "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                    "command": None,
                    "args": [],
                    "credential_pool": None,
                },
            ],
        ) as resolve:
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert resolve.call_count == 2

    def test_relogin_error_with_service_status_does_not_try_fallback(
        self, tmp_path, monkeypatch
    ):
        from gateway.run import _resolve_runtime_agent_kwargs
        from hermes_cli.auth import AuthError

        (tmp_path / "config.yaml").write_text(
            "fallback_model:\n  provider: openrouter\n  model: test-model\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError(
                "invalid_grant",
                provider="openai-codex",
                code="invalid_grant",
                relogin_required=True,
                status_code=503,
            ),
        ) as resolve:
            with pytest.raises(RuntimeError, match="invalid_grant"):
                _resolve_runtime_agent_kwargs()

        assert resolve.call_count == 1
