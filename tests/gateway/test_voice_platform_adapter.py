"""Tests for the Vox voice gateway platform adapter."""

import asyncio
import json
from inspect import Parameter, signature

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_key


def test_gateway_runner_creates_voice_adapter():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()

    adapter = runner._create_adapter(
        Platform.VOICE,
        PlatformConfig(enabled=True, extra={"vox_url": "ws://localhost:8600/adapter/hermes"}),
    )

    assert adapter is not None
    assert adapter.platform is Platform.VOICE
    assert adapter._vox_url == "ws://localhost:8600/adapter/hermes"
    assert adapter._platform_name == "hermes"


def test_voice_adapter_uses_configured_platform_name():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()

    adapter = runner._create_adapter(
        Platform.VOICE,
        PlatformConfig(
            enabled=True,
            extra={
                "vox_url": "ws://localhost:8600/adapter/xenia",
                "platform_name": "xenia",
            },
        ),
    )

    assert adapter is not None
    assert adapter._vox_url == "ws://localhost:8600/adapter/xenia"
    assert adapter._platform_name == "xenia"


def test_voice_adapter_connect_accepts_gateway_reconnect_kwarg():
    """Voice must honor the same connect signature the gateway calls."""
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.voice import VoiceAdapter

    sig = signature(VoiceAdapter.connect)
    base_sig = signature(BasePlatformAdapter.connect)

    param = sig.parameters["is_reconnect"]
    base_param = base_sig.parameters["is_reconnect"]

    assert param.kind is Parameter.KEYWORD_ONLY
    assert param.kind is base_param.kind
    assert param.default is False


def _make_voice_adapter():
    from gateway.platforms.voice import VoiceAdapter

    return VoiceAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "vox_url": "ws://localhost:8600/adapter/grace",
                "platform_name": "grace",
            },
        )
    )


async def _drain_background_tasks(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*tuple(adapter._background_tasks))


@pytest.mark.asyncio
async def test_phone_transcript_carries_verified_owner_context_once(monkeypatch):
    adapter = _make_voice_adapter()
    events = []

    async def handle_message(event):
        events.append(event)
        return "I hear you."

    sent = []

    async def send_to_vox(message):
        sent.append(message)

    adapter.set_message_handler(handle_message)
    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-grace",
            "agent": "grace",
            "source": "phone",
            "callerID": "+15555550123",
            "callerName": "Elliott",
            "isElliott": True,
            "direction": "inbound",
        }
    )
    await adapter._handle_vox_message(
        {"type": "text", "callId": "vox-grace", "content": "Can you hear me?"}
    )
    await adapter._handle_vox_message(
        {"type": "text", "callId": "vox-grace", "content": "Good."}
    )
    await _drain_background_tasks(adapter)

    assert len(events) == 2
    assert events[0].source.user_id == "elliott"
    assert events[0].text.startswith(
        "[Phone call context: incoming FaceTime Audio call from Elliott."
    )
    assert "Can you hear me?" in events[0].text
    assert events[1].text == "Good."
    assert events[0].agent_photo_request_text == "Can you hear me?"
    assert events[1].agent_photo_request_text == "Good."
    assert sent == [
        {
            "type": "text",
            "content": "I hear you.",
            "callId": "vox-grace",
        },
        {
            "type": "text",
            "content": "I hear you.",
            "callId": "vox-grace",
        },
    ]


@pytest.mark.asyncio
async def test_unverified_inbound_phone_call_is_hung_up(monkeypatch):
    adapter = _make_voice_adapter()
    events = []
    sent = []

    async def handle_message(event):
        events.append(event)
        return None

    async def send_to_vox(message):
        sent.append(message)

    adapter.set_message_handler(handle_message)
    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-grace",
            "agent": "grace",
            "source": "phone",
            "callerID": "+15555550999",
            "callerName": "Unknown",
            "isElliott": False,
            "direction": "inbound",
        }
    )
    await adapter._handle_vox_message(
        {"type": "text", "callId": "vox-grace", "content": "Queued speech."}
    )

    assert "vox-grace" not in adapter._active_calls
    assert events == []
    assert sent == [{"type": "hangup", "callId": "vox-grace"}]


@pytest.mark.asyncio
async def test_string_false_does_not_verify_inbound_phone_caller(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-malformed-owner",
            "agent": "grace",
            "source": "phone",
            "isElliott": "false",
            "direction": "inbound",
        }
    )

    assert "vox-malformed-owner" not in adapter._active_calls
    assert sent == [{"type": "hangup", "callId": "vox-malformed-owner"}]


@pytest.mark.asyncio
async def test_regular_voice_transcript_remains_plain_text(monkeypatch):
    adapter = _make_voice_adapter()
    events = []

    async def handle_message(event):
        events.append(event)
        return None

    adapter.set_message_handler(handle_message)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-grace",
            "agent": "grace",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {"type": "text", "callId": "vox-grace", "content": "Hello."}
    )
    await _drain_background_tasks(adapter)

    assert len(events) == 1
    assert events[0].source.user_id == "elliott"
    assert events[0].text.startswith("[Voice call context: This is live speech.")
    assert events[0].text.endswith("\n\nHello.")
    assert events[0].agent_photo_request_text == "Hello."


@pytest.mark.asyncio
async def test_unknown_voice_source_has_no_direct_photo_authorization():
    adapter = _make_voice_adapter()
    events = []

    async def handle_message(event):
        events.append(event)
        return None

    adapter.set_message_handler(handle_message)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-system",
            "agent": "grace",
            "source": "system",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "text",
            "callId": "vox-system",
            "content": "Generate an agent photo.",
        }
    )
    await _drain_background_tasks(adapter)

    assert len(events) == 1
    assert events[0].agent_photo_request_text is None


@pytest.mark.asyncio
async def test_transport_call_id_can_use_a_stable_session_id(monkeypatch):
    adapter = _make_voice_adapter()
    events = []
    sent = []

    async def handle_message(event):
        events.append(event)
        return "I remember."

    async def send_to_vox(message):
        sent.append(message)

    adapter.set_message_handler(handle_message)
    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "elevenagents-kenzie-turn-1",
            "sessionId": "vox-kenzie-1",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "text",
            "callId": "elevenagents-kenzie-turn-1",
            "content": "Do you remember?",
        }
    )
    await _drain_background_tasks(adapter)

    assert events[0].source.chat_id == "voice:vox-kenzie-1"
    assert sent == [
        {
            "type": "text",
            "content": "I remember.",
            "callId": "elevenagents-kenzie-turn-1",
        }
    ]


@pytest.mark.asyncio
async def test_streaming_response_emits_appended_deltas_and_one_final(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-1",
            "sessionId": "vox-kenzie-1",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    result = await adapter.send(
        "voice:vox-kenzie-1",
        "Fin \u2589",
        metadata={"expect_edits": True, "stream_cursor": " \u2589"},
    )
    await adapter.edit_message(
        "voice:vox-kenzie-1",
        result.message_id,
        "Finished \u2589",
        finalize=False,
    )
    await adapter.edit_message(
        "voice:vox-kenzie-1",
        result.message_id,
        "Finished answer \u2589",
        finalize=False,
    )
    await adapter.edit_message(
        "voice:vox-kenzie-1",
        result.message_id,
        "Finished answer",
        finalize=True,
    )
    await adapter.edit_message(
        "voice:vox-kenzie-1",
        result.message_id,
        "Finished answer",
        finalize=True,
    )

    assert sent == [
        {
            "type": "text",
            "content": "Fin",
            "callId": "speech-1",
            "stream": True,
            "isFinal": False,
            "streamId": result.message_id,
        },
        {
            "type": "text",
            "content": "ished",
            "callId": "speech-1",
            "stream": True,
            "isFinal": False,
            "streamId": result.message_id,
        },
        {
            "type": "text",
            "content": " answer",
            "callId": "speech-1",
            "stream": True,
            "isFinal": False,
            "streamId": result.message_id,
        },
        {
            "type": "text",
            "content": "",
            "callId": "speech-1",
            "stream": True,
            "isFinal": True,
            "streamId": result.message_id,
        },
    ]


@pytest.mark.asyncio
async def test_stream_finalization_retries_after_transport_failure(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []
    final_attempts = 0

    async def send_to_vox(message):
        nonlocal final_attempts
        if message.get("isFinal") is True:
            final_attempts += 1
            if final_attempts == 1:
                raise OSError("temporary transport failure")
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-retry",
            "sessionId": "vox-kenzie-retry",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    result = await adapter.send(
        "voice:vox-kenzie-retry",
        "Complete answer",
        metadata={"expect_edits": True},
    )
    with pytest.raises(OSError, match="temporary transport failure"):
        await adapter.edit_message(
            "voice:vox-kenzie-retry",
            result.message_id,
            "Complete answer",
            finalize=True,
        )

    await adapter.edit_message(
        "voice:vox-kenzie-retry",
        result.message_id,
        "Complete answer",
        finalize=True,
    )

    assert final_attempts == 2
    assert sum(frame["isFinal"] is True for frame in sent) == 1


@pytest.mark.asyncio
async def test_unanchored_send_falls_back_from_stale_context_to_replacement_call(
    monkeypatch,
):
    from gateway.platforms.voice import _ACTIVE_TRANSPORT_CALL

    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-old",
            "sessionId": "vox-kenzie-replaced",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    token = _ACTIVE_TRANSPORT_CALL.set((id(adapter), "speech-old"))
    try:
        await adapter._handle_vox_message(
            {
                "type": "call_start",
                "callId": "speech-new",
                "sessionId": "vox-kenzie-replaced",
                "agent": "kenzie",
                "source": "voice",
            }
        )
        await adapter.send("voice:vox-kenzie-replaced", "Current response")
    finally:
        _ACTIVE_TRANSPORT_CALL.reset(token)

    assert sent == [
        {
            "type": "text",
            "content": "Current response",
            "callId": "speech-new",
        }
    ]


@pytest.mark.asyncio
async def test_concurrent_stream_finalizers_emit_exactly_one_final(monkeypatch):
    adapter = _make_voice_adapter()
    final_started = asyncio.Event()
    release_final = asyncio.Event()
    sent = []

    async def send_to_vox(message):
        sent.append(message)
        if message.get("isFinal") is True:
            final_started.set()
            await release_final.wait()

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-concurrent",
            "sessionId": "vox-kenzie-concurrent",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    result = await adapter.send(
        "voice:vox-kenzie-concurrent",
        "Complete answer",
        metadata={"expect_edits": True, "stream_cursor": ""},
    )

    first = asyncio.create_task(
        adapter.edit_message(
            "voice:vox-kenzie-concurrent",
            result.message_id,
            "Complete answer",
            finalize=True,
        )
    )
    await asyncio.wait_for(final_started.wait(), timeout=1)
    second = asyncio.create_task(
        adapter.edit_message(
            "voice:vox-kenzie-concurrent",
            result.message_id,
            "Complete answer",
            finalize=True,
        )
    )
    await asyncio.sleep(0)
    release_final.set()
    await asyncio.gather(first, second)

    assert sum(frame.get("isFinal") is True for frame in sent) == 1


@pytest.mark.asyncio
async def test_non_default_stream_cursor_is_removed_from_spoken_deltas(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-cursor",
            "sessionId": "vox-kenzie-cursor",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    result = await adapter.send(
        "voice:vox-kenzie-cursor",
        "Hel <WAIT>",
        metadata={"expect_edits": True, "stream_cursor": " <WAIT>"},
    )
    await adapter.edit_message(
        "voice:vox-kenzie-cursor",
        result.message_id,
        "Hello <WAIT>",
    )
    await adapter.edit_message(
        "voice:vox-kenzie-cursor",
        result.message_id,
        "Hello",
        finalize=True,
    )

    assert [frame["content"] for frame in sent] == ["Hel", "lo", ""]


@pytest.mark.asyncio
async def test_completed_and_deactivated_streams_do_not_accumulate(monkeypatch):
    adapter = _make_voice_adapter()

    async def send_to_vox(_message):
        return None

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-cleanup",
            "sessionId": "vox-kenzie-cleanup",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    for index in range(5):
        result = await adapter.send(
            "voice:vox-kenzie-cleanup",
            f"Answer {index}",
            metadata={"expect_edits": True, "stream_cursor": ""},
        )
        await adapter.edit_message(
            "voice:vox-kenzie-cleanup",
            result.message_id,
            f"Answer {index}",
            finalize=True,
        )
        assert adapter._streams == {}

    await adapter.send(
        "voice:vox-kenzie-cleanup",
        "Pending answer",
        metadata={"expect_edits": True, "stream_cursor": ""},
    )
    assert len(adapter._streams) == 1

    await adapter._handle_vox_message(
        {"type": "call_end", "callId": "speech-cleanup"}
    )

    assert adapter._streams == {}


@pytest.mark.asyncio
async def test_non_extension_edit_uses_safe_suffix_then_closes_stream(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-edit",
            "sessionId": "vox-kenzie-edit",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    result = await adapter.send(
        "voice:vox-kenzie-edit",
        "Original phrase",
        metadata={"expect_edits": True},
    )
    await adapter.edit_message(
        "voice:vox-kenzie-edit",
        result.message_id,
        "Preface: Original phrase plus suffix",
        finalize=False,
    )
    await adapter.edit_message(
        "voice:vox-kenzie-edit",
        result.message_id,
        "Completely rewritten response",
        finalize=False,
    )
    await adapter.edit_message(
        "voice:vox-kenzie-edit",
        result.message_id,
        "Completely rewritten response with more text",
        finalize=True,
    )

    assert [frame["content"] for frame in sent] == [
        "Original phrase",
        " plus suffix",
        "",
    ]
    assert sent[-1]["isFinal"] is True
    assert sum(frame["isFinal"] is True for frame in sent) == 1


@pytest.mark.asyncio
async def test_non_streaming_response_keeps_legacy_text_shape(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-plain",
            "sessionId": "vox-kenzie-plain",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    await adapter.send("voice:vox-kenzie-plain", "Complete answer")
    await adapter.edit_message(
        "voice:vox-kenzie-plain",
        "legacy-message-id",
        "Corrected answer",
    )

    assert sent == [
        {
            "type": "text",
            "content": "Complete answer",
            "callId": "speech-plain",
        },
        {
            "type": "text",
            "content": "Corrected answer",
            "callId": "speech-plain",
        },
    ]


@pytest.mark.asyncio
async def test_voice_suppresses_non_conversational_metadata(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-status",
            "sessionId": "vox-kenzie-status",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    result = await adapter.send(
        "voice:vox-kenzie-status",
        "⏳ Compressing context, your message is queued.",
        metadata={"non_conversational": True},
    )
    await adapter.edit_message(
        "voice:vox-kenzie-status",
        result.message_id,
        "Tool memory returned error: details",
    )

    assert sent == []


@pytest.mark.asyncio
async def test_voice_suppresses_long_file_or_log_dump(monkeypatch):
    adapter = _make_voice_adapter()
    sent = []

    async def send_to_vox(message):
        sent.append(message)

    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-file",
            "sessionId": "vox-kenzie-file",
            "agent": "kenzie",
            "source": "voice",
        }
    )

    long_markdown = "\n".join(f"- line {index}: configuration value" for index in range(40))
    await adapter.send("voice:vox-kenzie-file", long_markdown)

    assert sent == []


@pytest.mark.asyncio
async def test_listener_processes_call_end_while_generation_is_running(monkeypatch):
    adapter = _make_voice_adapter()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()
    sent = []

    async def handle_message(_event):
        handler_started.set()
        await release_handler.wait()
        return "Late answer"

    async def send_to_vox(message):
        sent.append(message)

    class ScriptedWebSocket:
        async def __aiter__(self):
            yield json.dumps(
                {
                    "type": "call_start",
                    "callId": "speech-ended",
                    "sessionId": "vox-kenzie-ended",
                    "agent": "kenzie",
                    "source": "voice",
                }
            )
            yield json.dumps(
                {
                    "type": "text",
                    "callId": "speech-ended",
                    "content": "Still there?",
                }
            )
            await asyncio.wait_for(handler_started.wait(), timeout=1)
            yield json.dumps({"type": "call_end", "callId": "speech-ended"})

    adapter.set_message_handler(handle_message)
    adapter._ws = ScriptedWebSocket()
    monkeypatch.setattr(adapter, "_send_to_vox", send_to_vox)

    await asyncio.wait_for(adapter._listen(), timeout=1)

    assert "speech-ended" not in adapter._active_calls
    release_handler.set()
    await _drain_background_tasks(adapter)
    assert sent == []


@pytest.mark.asyncio
async def test_turn_end_only_deactivates_call_without_cancelling_session_task():
    adapter = _make_voice_adapter()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    events = []
    task = None

    async def handle_message(event):
        events.append(event)
        handler_started.set()
        try:
            await release_handler.wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    adapter.set_message_handler(handle_message)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-turn-end",
            "sessionId": "vox-kenzie-turn-end",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "text",
            "callId": "speech-turn-end",
            "content": "Keep this session alive.",
        }
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1)

    session_key = build_session_key(
        events[0].source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    task = adapter._session_tasks[session_key]

    try:
        await adapter._handle_vox_message(
            {"type": "turn_end", "callId": "speech-turn-end"}
        )
        await asyncio.sleep(0)

        assert handler_cancelled.is_set() is False
        assert task.done() is False
        assert session_key in adapter._session_tasks
        assert "speech-turn-end" not in adapter._active_calls
    finally:
        release_handler.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_current_call_end_cancels_and_drains_stable_session_task():
    adapter = _make_voice_adapter()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    events = []
    task = None

    async def handle_message(event):
        events.append(event)
        handler_started.set()
        try:
            await release_handler.wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    adapter.set_message_handler(handle_message)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-current-end",
            "sessionId": "vox-kenzie-current-end",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "text",
            "callId": "speech-current-end",
            "content": "Please keep working",
        }
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1)

    session_key = build_session_key(
        events[0].source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    task = adapter._session_tasks[session_key]

    try:
        assert adapter._active_calls["speech-current-end"]["session_key"] == session_key

        await adapter._handle_vox_message(
            {"type": "call_end", "callId": "speech-current-end"}
        )
        await asyncio.sleep(0)

        assert handler_cancelled.is_set()
        assert task.done()
        assert session_key not in adapter._session_tasks
        assert task not in adapter._background_tasks
    finally:
        release_handler.set()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_superseded_call_end_does_not_cancel_replacement_session_task():
    adapter = _make_voice_adapter()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    events = []
    task = None

    async def handle_message(event):
        events.append(event)
        handler_started.set()
        try:
            await release_handler.wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    adapter.set_message_handler(handle_message)
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-superseded",
            "sessionId": "vox-kenzie-shared",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "speech-replacement",
            "sessionId": "vox-kenzie-shared",
            "agent": "kenzie",
            "source": "voice",
        }
    )
    await adapter._handle_vox_message(
        {
            "type": "text",
            "callId": "speech-replacement",
            "content": "Replacement request",
        }
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1)

    session_key = build_session_key(
        events[0].source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    task = adapter._session_tasks[session_key]

    try:
        await adapter._handle_vox_message(
            {"type": "call_end", "callId": "speech-superseded"}
        )
        await asyncio.sleep(0)

        assert handler_cancelled.is_set() is False
        assert task.done() is False
        assert adapter._session_tasks[session_key] is task
        assert "speech-replacement" in adapter._active_calls
        assert adapter._session_calls["vox-kenzie-shared"] == "speech-replacement"
    finally:
        release_handler.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_outbound_phone_call_labels_the_recipient(monkeypatch):
    adapter = _make_voice_adapter()
    events = []

    async def handle_message(event):
        events.append(event)
        return None

    adapter.set_message_handler(handle_message)

    await adapter._handle_vox_message(
        {
            "type": "call_start",
            "callId": "vox-grace",
            "agent": "grace",
            "source": "phone",
            "callerName": "Dr. Smith",
            "isElliott": False,
            "direction": "outbound",
        }
    )
    await adapter._handle_vox_message(
        {"type": "text", "callId": "vox-grace", "content": "Hello?"}
    )
    await _drain_background_tasks(adapter)

    assert len(events) == 1
    assert events[0].source.user_id == "phone-outbound"
    assert events[0].text.startswith(
        "[Phone call context: outgoing FaceTime Audio call to Dr. Smith."
    )
    assert events[0].agent_photo_request_text is None
