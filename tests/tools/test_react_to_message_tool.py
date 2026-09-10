"""Ownership tests for desktop message reactions."""

import json
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

    adapter = MagicMock()

    async def set_reaction(chat_id, message_id, emoji):
        assert (chat_id, message_id, emoji) == ("-1001", "42", "👍")
        return True

    adapter._set_reaction = set_reaction
    runner = MagicMock()
    runner._adapter_for_source.return_value = adapter
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        message_id="42",
        profile="main",
        cron_session="",
    )
    try:
        result = json.loads(reactions.react_to_message_tool("👍"))
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
    source = runner._adapter_for_source.call_args.args[0]
    assert source.chat_id == "-1001"
    assert source.message_id == "42"
    assert source.profile == "main"


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
