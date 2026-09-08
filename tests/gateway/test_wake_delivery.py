"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import (
    FINAL_RETURN_CONTEXT_HEADER,
    FINAL_RETURN_CONTEXT_METADATA_KEY,
    INTERNAL_WAKE_HEADER,
    FINAL_RETURN_CLAIM_HEADER,
    WakeDeliveryOutcome,
    adapter_supports_push,
    complete_final_return_delivery,
    deliver_wake,
    final_return_context_from_event,
    final_return_context_matches_profile,
)


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)
        if final_return_context_from_event(event) is not None:
            complete_final_return_delivery(
                event,
                WakeDeliveryOutcome(
                    "acknowledged", returned_message_id="session-message:sid:1"
                ),
            )


class HangingPushAdapter(PushAdapter):
    async def handle_message(self, event):
        self.handled.append(event)


class FailingPushAdapter(PushAdapter):
    async def handle_message(self, event):
        self.handled.append(event)
        raise RuntimeError("adapter wake failed")


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


def test_ordinary_push_wake_preserves_adapter_failure():
    adapter = FailingPushAdapter()

    with pytest.raises(RuntimeError, match="adapter wake failed"):
        asyncio.run(deliver_wake(adapter, text="ordinary wake", source=_source()))

    assert len(adapter.handled) == 1


def _final_return_context(tmp_path):
    board = tmp_path / "board.db"
    board.touch()
    return {
        "request_root_id": "cr_final_1",
        "task_id": "task_final_1",
        "event_id": "42",
        "responsible_agent": "aurora",
        "db_path": str(board),
    }


def _admitted_final_return(*_args, **_kwargs):
    return SimpleNamespace(
        state="sending", send_claimed=True, claim_token="a" * 32,
        returned_message_id="",
    )


def test_deliver_push_wake_preserves_only_valid_final_return_context(tmp_path):
    adapter = PushAdapter()

    with patch(
        "gateway.delivery_ledger.admit_coordination_final_return_turn",
        side_effect=_admitted_final_return,
    ):
        outcome = asyncio.run(
            deliver_wake(
                adapter,
                text="final return",
                source=_source(),
                coordination_context=_final_return_context(tmp_path),
            )
        )

    assert len(adapter.handled) == 1
    event = adapter.handled[0]
    assert event.internal is True
    assert final_return_context_from_event(event) == _final_return_context(tmp_path)
    assert FINAL_RETURN_CONTEXT_METADATA_KEY in event.metadata
    assert outcome is not None and outcome.state == "acknowledged"

    event.internal = False
    assert final_return_context_from_event(event) is None


def test_deliver_push_wake_rejects_partial_or_extra_final_return_context(tmp_path):
    adapter = PushAdapter()
    context = _final_return_context(tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(
            deliver_wake(
                adapter,
                text="bad",
                source=_source(),
                coordination_context={"request_root_id": context["request_root_id"]},
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            deliver_wake(
                adapter,
                text="bad",
                source=_source(),
                coordination_context={**context, "purpose": "final_return"},
            )
        )


def test_final_return_profile_match_uses_canonical_workforce_identity(
    tmp_path, monkeypatch,
):
    context = _final_return_context(tmp_path)
    context["responsible_agent"] = "root"
    monkeypatch.setenv(
        "HERMES_WORKFORCE_ORG",
        str(Path(__file__).parents[2] / "workforce" / "organization.yaml"),
    )

    assert final_return_context_matches_profile(context, "main") is True
    assert final_return_context_matches_profile(context, "aurora") is False
    assert final_return_context_matches_profile(context, "missing-profile") is False

    context["responsible_agent"] = "amy"
    assert final_return_context_matches_profile(context, "amy") is False


def test_final_return_profile_match_preserves_legacy_only_when_org_absent(
    tmp_path, monkeypatch,
):
    context = _final_return_context(tmp_path)
    context["responsible_agent"] = "legacy"
    organization = tmp_path / "missing-organization.yaml"
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(organization))

    assert final_return_context_matches_profile(context, "legacy") is True
    assert final_return_context_matches_profile(context, "other") is False

    organization.write_text("not: [valid", encoding="utf-8")
    assert final_return_context_matches_profile(context, "legacy") is False

    read_error = tmp_path / "organization-directory"
    read_error.mkdir()
    monkeypatch.setenv("HERMES_WORKFORCE_ORG", str(read_error))
    assert final_return_context_matches_profile(context, "legacy") is False


def test_repeated_final_return_after_ambiguous_turn_never_enqueues_second_model(tmp_path, monkeypatch):
    """A timed-out worker keeps its ticket; a watcher replay is inert."""
    import gateway.delivery_ledger as ledger
    import gateway.wake as wake_mod

    adapter = HangingPushAdapter()
    calls = [
        _admitted_final_return(),
        SimpleNamespace(
            state="uncertain", send_claimed=False, claim_token="", returned_message_id=""
        ),
    ]
    monkeypatch.setattr(wake_mod, "WAKE_TURN_TIMEOUT_SECONDS", 0.01)
    with (
        patch.object(ledger, "admit_coordination_final_return_turn", side_effect=calls),
        patch.object(ledger, "mark_coordination_final_return_uncertain"),
    ):
        first = asyncio.run(
            deliver_wake(
                adapter, text="final", source=_source(),
                coordination_context=_final_return_context(tmp_path),
            )
        )
        second = asyncio.run(
            deliver_wake(
                adapter, text="final", source=_source(),
                coordination_context=_final_return_context(tmp_path),
            )
        )
    assert first is not None and first.state == "uncertain"
    assert second is not None and second.state == "uncertain"
    assert len(adapter.handled) == 1


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_final_return_self_post_carries_only_authenticated_header_context(tmp_path):
    from aiohttp import web

    import gateway.delivery_ledger as ledger

    seen = {}

    async def handler(request):
        seen["marker"] = request.headers.get(INTERNAL_WAKE_HEADER)
        seen["context"] = request.headers.get(FINAL_RETURN_CONTEXT_HEADER)
        seen["claim"] = request.headers.get(FINAL_RETURN_CLAIM_HEADER)
        seen["body"] = await request.json()
        return web.json_response(
            {"choices": []},
            headers={"X-Hermes-Final-Return-Receipt": "session-message:sid:7"},
        )

    async def run():
        runner, port = await _serve(handler)
        try:
            with patch.object(
                ledger,
                "mark_coordination_final_return_acknowledged",
                return_value=SimpleNamespace(returned_message_id="session-message:sid:7"),
            ) as mark_ack, patch.object(
                ledger,
                "admit_coordination_final_return_turn",
                side_effect=_admitted_final_return,
            ):
                return await deliver_wake(
                    ApiServerLikeAdapter(port=port),
                    text="final return",
                    session_id="sid",
                    coordination_context=_final_return_context(tmp_path),
                )
        finally:
            await runner.cleanup()

    outcome = asyncio.run(run())
    assert seen["marker"] == "final-return-v1"
    assert isinstance(seen["context"], str) and seen["context"]
    assert seen["claim"] == "a" * 32
    assert FINAL_RETURN_CONTEXT_METADATA_KEY not in seen["body"]
    assert outcome is not None and outcome.state == "acknowledged"


def test_final_return_api_timeout_is_uncertain_without_an_automatic_replay(tmp_path, monkeypatch):
    """A lost HTTP response might already have displayed the answer."""
    from aiohttp import web

    import gateway.delivery_ledger as ledger
    import gateway.wake as wake_mod

    calls = {"count": 0}

    async def handler(_request):
        calls["count"] += 1
        await asyncio.sleep(0.2)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            with (
                patch.object(
                    ledger,
                    "admit_coordination_final_return_turn",
                    side_effect=_admitted_final_return,
                ),
                patch.object(ledger, "mark_coordination_final_return_uncertain") as uncertain,
            ):
                outcome = await deliver_wake(
                    ApiServerLikeAdapter(port=port),
                    text="final return",
                    session_id="sid",
                    coordination_context=_final_return_context(tmp_path),
                )
                uncertain.assert_called_once()
                return outcome
        finally:
            await runner.cleanup()

    monkeypatch.setattr(wake_mod, "WAKE_TURN_TIMEOUT_SECONDS", 0.01)
    outcome = asyncio.run(run())
    assert outcome is not None and outcome.state == "uncertain"
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_final_return_push_attempts_one_ambiguous_send_without_generic_retry(tmp_path):
    """A timeout-like SendResult cannot trigger retry or fallback delivery."""
    from gateway.config import PlatformConfig
    from gateway.platforms.base import (
        BasePlatformAdapter,
        MessageEvent,
        MessageType,
        SendResult,
    )
    from gateway.session import build_session_key
    from gateway.wake import (
        FINAL_RETURN_DELIVERY_STATE_METADATA_KEY,
        FinalReturnDeliveryState,
    )

    class OneSendAdapter(BasePlatformAdapter):
        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            return None

        async def send(self, *_args, **_kwargs):
            return SendResult(success=False, error="timed out", retryable=True)

        async def get_chat_info(self, _chat_id):
            return {}

    context = _final_return_context(tmp_path)
    adapter = OneSendAdapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    adapter._message_handler = AsyncMock(return_value="finished")
    adapter._send_with_retry = AsyncMock(side_effect=AssertionError("generic retry used"))
    adapter.send = AsyncMock(return_value=SendResult(
        success=False, error="timed out", retryable=True,
    ))
    state = FinalReturnDeliveryState(
        context=context,
        completion=asyncio.get_running_loop().create_future(),
        claim_token="a" * 32,
    )
    event = MessageEvent(
        text="wake", message_type=MessageType.TEXT, source=_source(), internal=True,
        metadata={
            FINAL_RETURN_CONTEXT_METADATA_KEY: context,
            FINAL_RETURN_DELIVERY_STATE_METADATA_KEY: state,
        },
    )
    with (
        patch(
            "gateway.delivery_ledger.claim_coordination_final_return_delivery",
            return_value=SimpleNamespace(state="sending", send_claimed=True),
        ),
        patch("gateway.delivery_ledger.mark_coordination_final_return_uncertain") as uncertain,
    ):
        await adapter._process_message_background(event, build_session_key(event.source))
    assert adapter.send.await_count == 1
    assert adapter._send_with_retry.await_count == 0
    assert uncertain.call_count == 1
    assert (await state.completion).state == "uncertain"


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2
