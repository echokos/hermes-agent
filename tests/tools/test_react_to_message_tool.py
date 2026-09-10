"""Ownership tests for desktop message reactions."""

import json
import weakref
from unittest.mock import MagicMock

from gateway.session_context import clear_session_vars, set_session_vars
from tools import react_to_message_tool as reactions


def test_reaction_database_closes_when_write_fails(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    monkeypatch.setattr(
        reactions,
        "get_session_env",
        lambda _name, _default="": "session-1",
    )

    result = reactions.react_to_message_tool("👍")

    assert "write failed" in result
    db.close.assert_called_once()


def test_telegram_reaction_uses_only_current_turn_provenance(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource

    adapter = MagicMock()

    async def set_reaction(chat_id, message_id, emoji):
        assert (chat_id, message_id, emoji) == ("-1001", "42", "👍")
        return True

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
        cron_session="",
        origin_source=source,
    )
    try:
        from tools.registry import registry

        entry = registry.get_entry("react_to_message")
        result = json.loads(entry.handler({"emoji": "👍"}))
    finally:
        clear_session_vars(tokens)

    assert result == {
        "success": True,
        "reaction_acknowledgement": True,
        "operation": "telegram_current_turn_reaction",
        "platform": "telegram",
        "chat_id": "-1001",
        "message_id": "42",
    }
    assert runner._adapter_for_source.call_args.args[0] is source


def test_telegram_reaction_rejects_retargeting_before_adapter_lookup(monkeypatch):
    import gateway.run as gateway_run

    runner = MagicMock()
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram", chat_id="-1001", message_id="42", cron_session=""
    )
    try:
        result = reactions.react_to_message_tool("👍", messages_back=1)
    finally:
        clear_session_vars(tokens)

    assert "only an emoji" in result
    runner._adapter_for_source.assert_not_called()


def test_telegram_reaction_rejects_cron_before_adapter_lookup(monkeypatch):
    import gateway.run as gateway_run

    runner = MagicMock()
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram", chat_id="-1001", message_id="42", cron_session="1"
    )
    try:
        result = reactions.react_to_message_tool("👍")
    finally:
        clear_session_vars(tokens)

    assert "cron" in result.lower()
    runner._adapter_for_source.assert_not_called()


def test_telegram_reaction_preserves_default_transport_for_secondary_runtime(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource

    class Adapter:
        async def _set_reaction(self, chat_id, message_id, emoji):
            assert (chat_id, message_id, emoji) == ("-1001", "42", "👍")
            return True

    default_adapter = Adapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        message_id="42",
        profile="secondary",
    )
    source._transport_adapter_ref = weakref.ref(default_adapter)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        profile="secondary",
        cron_session="",
        origin_source=source,
    )
    try:
        assert json.loads(reactions.react_to_message_tool("👍"))["success"] is True
    finally:
        clear_session_vars(tokens)


def test_telegram_reaction_fails_closed_for_unowned_secondary_profile(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: object()}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        message_id="42",
        profile="secondary",
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        profile="secondary",
        cron_session="",
        origin_source=source,
    )
    try:
        result = reactions.react_to_message_tool("👍")
    finally:
        clear_session_vars(tokens)

    assert "cannot set reactions" in result


def test_telegram_reaction_rejects_stale_origin_before_adapter_lookup(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource

    runner = MagicMock()
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    stale_source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-other", message_id="42"
    )
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        cron_session="",
        origin_source=stale_source,
    )
    try:
        result = reactions.react_to_message_tool("👍")
    finally:
        clear_session_vars(tokens)

    assert "provenance" in result
    runner._adapter_for_source.assert_not_called()


def test_reaction_schema_preserves_desktop_retargeting_parameters():
    properties = reactions.REACT_TO_MESSAGE_SCHEMA["parameters"]["properties"]
    assert {"emoji", "message_row_id", "messages_back"} <= set(properties)


def test_desktop_reaction_keeps_explicit_message_target(monkeypatch):
    db = MagicMock()
    db.get_message_role.return_value = "user"
    db.set_message_reaction.return_value = {"agent": "👍"}
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    tokens = set_session_vars(platform="desktop", session_key="desktop-session")
    try:
        result = json.loads(reactions.react_to_message_tool("👍", message_row_id=9))
    finally:
        clear_session_vars(tokens)

    assert result["row_id"] == 9
    db.set_message_reaction.assert_called_once_with(
        "desktop-session", 9, "👍", author="agent"
    )


def test_telegram_reaction_handles_missing_adapter_false_and_error(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource

    class Adapter:
        def __init__(self, outcome):
            self.outcome = outcome

        async def _set_reaction(self, *_args):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", message_id="42"
    )
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        cron_session="",
        origin_source=source,
    )
    try:
        monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)
        assert "No live Telegram gateway" in reactions.react_to_message_tool("👍")
        for outcome, expected in ((False, "did not accept"), (RuntimeError("nope"), "failed")):
            runner = MagicMock()
            runner._adapter_for_source.return_value = Adapter(outcome)
            monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
            assert expected in reactions.react_to_message_tool("👍")
    finally:
        clear_session_vars(tokens)
