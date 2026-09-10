"""Auth failures remain on native credential recovery or terminal handling."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.error_classifier import classify_api_error, FailoverReason


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _auth_error(status=401, msg="Your API key is invalid, blocked or out of funds."):
    err = Exception(f"Error code: {status} - {msg}")
    err.status_code = status
    return err


class TestAuthErrorClassification:
    def test_401_is_auth(self):
        c = classify_api_error(_auth_error(401))
        assert c.reason in {FailoverReason.auth, FailoverReason.auth_permanent}
        assert c.is_auth is True


    def test_500_is_not_auth(self):
        err = Exception("Error code: 500 - internal server error")
        err.status_code = 500
        c = classify_api_error(err)
        assert c.is_auth is False


class TestAuthFallbackPolicy:
    def test_auth_failure_does_not_activate_configured_fallback(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        classified = classify_api_error(_auth_error(401))
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=AssertionError("auth must not resolve a fallback"),
        ) as resolve:
            advanced = agent._try_activate_fallback(reason=classified.reason)
        assert advanced is False
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False
        resolve.assert_not_called()
