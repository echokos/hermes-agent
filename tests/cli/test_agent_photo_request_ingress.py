"""Interactive CLI provenance boundary for direct agent-photo requests."""

from __future__ import annotations

import inspect

import pytest

import cli


@pytest.fixture
def interactive_tty(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)


def test_interactive_tty_returns_only_clean_authored_text(interactive_tty):
    assert (
        cli._interactive_cli_agent_photo_request_text(
            "  Send me an agent photo.  ", has_images=False
        )
        == "Send me an agent photo."
    )


@pytest.mark.parametrize(
    ("text", "has_images"),
    [
        ("/model", False),
        ("!git status", False),
        ("[Pasted text #2: 14 lines -> paste.txt]", False),
        ("Send me an agent photo.", True),
        ("   ", False),
    ],
)
def test_commands_paste_placeholders_images_and_empty_text_are_ineligible(
    interactive_tty, text, has_images
):
    assert (
        cli._interactive_cli_agent_photo_request_text(
            text, has_images=has_images
        )
        is None
    )


@pytest.mark.parametrize("stream_name", ["stdin", "stdout"])
def test_non_tty_stream_is_ineligible(interactive_tty, monkeypatch, stream_name):
    monkeypatch.setattr(getattr(cli.sys, stream_name), "isatty", lambda: False)

    assert (
        cli._interactive_cli_agent_photo_request_text(
            "Send me an agent photo.", has_images=False
        )
        is None
    )


def test_single_query_session_is_ineligible(interactive_tty, monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    assert (
        cli._interactive_cli_agent_photo_request_text(
            "Send me an agent photo.", has_images=False
        )
        is None
    )


def test_one_interrupted_typed_message_preserves_origin_marker():
    message = cli._InteractiveCLIInputMessage(
        "Send me an agent photo.", "Send me an agent photo."
    )

    combined = cli._coalesce_interrupted_cli_inputs([message])

    assert combined is message
    assert combined.agent_photo_request_text == "Send me an agent photo."


def test_merged_interrupted_messages_drop_origin_marker():
    first = cli._InteractiveCLIInputMessage("first", "first")
    second = cli._InteractiveCLIInputMessage("second", "second")

    combined = cli._coalesce_interrupted_cli_inputs([first, second])

    assert combined == "first\nsecond"
    assert type(combined) is str
    assert not hasattr(combined, "agent_photo_request_text")


def test_programmatic_chat_has_no_direct_request_by_default():
    parameter = inspect.signature(cli.HermesCLI.chat).parameters[
        "_direct_agent_photo_request_text"
    ]

    assert parameter.default is None
