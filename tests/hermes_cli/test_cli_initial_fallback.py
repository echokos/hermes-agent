"""Initial CLI provider resolution may fail over only for service/rate failures."""

from unittest.mock import patch

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _CLI(CLIAgentSetupMixin):
    def __init__(self):
        self.model = "primary-model"
        self.requested_provider = "openai-codex"
        self.provider = "openai-codex"
        self.api_key = None
        self.base_url = None
        self.api_mode = "codex_responses"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = [
            {"provider": "ollama-cloud", "model": "glm-5.3:cloud"},
        ]
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None

    def _normalize_model_for_provider(self, _provider):
        return False


def _fallback_runtime():
    return {
        "api_key": "test-fallback-key",
        "base_url": "https://ollama.com/v1",
        "provider": "ollama-cloud",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }


def test_cli_initial_rate_limit_uses_configured_fallback():
    from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

    cli = _CLI()
    with (
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=[
                AuthError(
                    "Codex usage limit reached",
                    provider="openai-codex",
                    code=CODEX_RATE_LIMITED_CODE,
                ),
                _fallback_runtime(),
            ],
        ) as resolve,
        patch("hermes_cli.fallback_config.resolve_entry_api_key", return_value=None),
    ):
        assert cli._ensure_runtime_credentials() is True

    assert resolve.call_count == 2
    assert cli.provider == "ollama-cloud"
    assert cli.requested_provider == "ollama-cloud"
    assert cli.model == "glm-5.3:cloud"


def test_cli_initial_server_error_uses_configured_fallback():
    class _ServiceError(Exception):
        status_code = 503

    cli = _CLI()
    with (
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=[_ServiceError("upstream overloaded"), _fallback_runtime()],
        ) as resolve,
        patch("hermes_cli.fallback_config.resolve_entry_api_key", return_value=None),
    ):
        assert cli._ensure_runtime_credentials() is True

    assert resolve.call_count == 2
    assert cli.provider == "ollama-cloud"
    assert cli.model == "glm-5.3:cloud"


def test_cli_initial_wrapped_oauth_503_uses_configured_fallback():
    from hermes_cli.auth import AuthError

    cli = _CLI()
    with (
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=[
                AuthError(
                    "Codex token refresh failed: temporarily unavailable",
                    provider="openai-codex",
                    code="server_error",
                    status_code=503,
                ),
                _fallback_runtime(),
            ],
        ) as resolve,
        patch("hermes_cli.fallback_config.resolve_entry_api_key", return_value=None),
    ):
        assert cli._ensure_runtime_credentials() is True

    assert resolve.call_count == 2
    assert cli.provider == "ollama-cloud"
    assert cli.model == "glm-5.3:cloud"


def test_cli_relogin_error_with_service_status_does_not_use_fallback():
    """Explicit credential-invalid metadata wins over a conflicting 503."""
    from hermes_cli.auth import AuthError

    cli = _CLI()
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
        assert cli._ensure_runtime_credentials() is False

    assert resolve.call_count == 1
    assert cli.provider == "openai-codex"
    assert cli.model == "primary-model"


def test_cli_initial_credential_error_does_not_use_fallback():
    from hermes_cli.auth import AuthError

    cli = _CLI()
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=AuthError("Codex credentials missing", relogin_required=True),
    ) as resolve:
        assert cli._ensure_runtime_credentials() is False

    assert resolve.call_count == 1
    assert cli.provider == "openai-codex"
    assert cli.model == "primary-model"
