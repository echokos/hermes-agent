"""Wake an existing agent session from a background completion event.

Two delivery strategies, selected by the target adapter's
``supports_async_delivery`` capability flag:

* Push-capable adapters (telegram, discord, plugin platforms, ...): inject a
  synthetic ``MessageEvent(internal=True)`` through ``adapter.handle_message``
  — the pre-existing wake path, preserved exactly.

* Stateless request/response adapters (the API server,
  ``supports_async_delivery = False``): ``handle_message`` would run the wake
  turn under a ``build_session_key()``-derived key
  (``agent:main:api_server:group:<sid>``) that NEVER matches the raw
  ``X-Hermes-Session-Id`` key real gateway/HQ turns run under
  (``_bind_api_server_session``), so the wake lands in a parallel, invisible
  session. Instead we self-POST ``/v1/chat/completions`` on the in-pod API
  server with the raw session id in the ``X-Hermes-Session-Id`` header — the
  exact entry point real turns use — so the wake turn resumes the REAL
  session, with full history, and its result is visible the next time the
  client polls/reopens the conversation.

Failures RAISE (after bounded retries on transient errors) so callers can
rewind cursors / retry instead of silently losing the event.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# A wake self-post runs the entire agent turn synchronously (stream=false);
# generous ceiling so long tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff delays between retries on transient failures (429 concurrency cap,
# connection errors). The API server has no per-session lock — concurrent
# turns on one session are last-writer-wins — but it DOES enforce a global
# max_concurrent_runs cap via HTTP 429, which is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)

# This is deliberately an event-local, all-or-nothing envelope.  A final
# return receives a separate reserve from its originating request only when
# the watcher injected every field through ``deliver_wake``.  It is not a
# caller-facing API surface.
FINAL_RETURN_CONTEXT_METADATA_KEY = "hermes_final_return_coordination"
FINAL_RETURN_DELIVERY_STATE_METADATA_KEY = "hermes_final_return_delivery_state"
FINAL_RETURN_ISOLATED_FOLLOWUP_METADATA_KEY = "hermes_final_return_isolated_followup"
FINAL_RETURN_CONTEXT_HEADER = "X-Hermes-Internal-Final-Return"
FINAL_RETURN_CLAIM_HEADER = "X-Hermes-Final-Return-Claim"
FINAL_RETURN_RECEIPT_HEADER = "X-Hermes-Final-Return-Receipt"
INTERNAL_WAKE_HEADER = "X-Hermes-Internal-Wake"
_INTERNAL_WAKE_HEADER_VALUE = "final-return-v1"
_FINAL_RETURN_CONTEXT_KEYS = frozenset({
    "request_root_id",
    "task_id",
    "event_id",
    "responsible_agent",
    "db_path",
})


@dataclass(frozen=True)
class WakeDeliveryOutcome:
    """Result of a final-return wake, independent of watcher cursor policy."""

    state: str
    returned_message_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.state not in {"pending", "acknowledged", "uncertain"}:
            raise ValueError("invalid final-return wake outcome")
        if self.state == "acknowledged" and not self.returned_message_id:
            raise ValueError("acknowledged final-return wake requires a receipt")


@dataclass
class FinalReturnDeliveryState:
    """Private event-local bridge from adapter delivery back to ``deliver_wake``."""

    context: dict[str, str]
    completion: "asyncio.Future[WakeDeliveryOutcome]"
    claim_token: str = ""
    returned_message_id: str = ""

    def record_receipt(self, returned_message_id: str) -> None:
        receipt = str(returned_message_id or "").strip()
        if not (
            receipt.startswith("session-message:")
            or receipt.startswith("platform-message:")
        ) or any(char in receipt for char in "\x00\r\n"):
            raise ValueError("invalid final-return session receipt")
        self.returned_message_id = receipt

    def complete(self, outcome: WakeDeliveryOutcome) -> None:
        if not self.completion.done():
            self.completion.set_result(outcome)


def validate_final_return_context(context: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact final-return envelope or reject it fail-closed."""
    if not isinstance(context, Mapping) or set(context) != _FINAL_RETURN_CONTEXT_KEYS:
        raise ValueError(
            "final-return coordination context must have exactly root, task, event, responsible agent, and DB"
        )
    normalized: dict[str, str] = {}
    for key in _FINAL_RETURN_CONTEXT_KEYS:
        value = context.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("final-return coordination context has an invalid field")
        if "\x00" in value or "\r" in value or "\n" in value:
            raise ValueError("final-return coordination context has an unsafe field")
        normalized[key] = value
    if not normalized["event_id"].isdigit() or int(normalized["event_id"]) <= 0:
        raise ValueError("final-return coordination event id is invalid")
    if not Path(normalized["db_path"]).is_file():
        raise ValueError("final-return coordination board is unavailable")
    return normalized


def final_return_context_from_event(event: Any) -> Optional[dict[str, str]]:
    """Read a trusted final-return envelope from one internal wake event.

    Callers deliberately treat every malformed/external envelope as absent;
    it must never turn an ordinary turn into a reserved final return.
    """
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, Mapping) or FINAL_RETURN_CONTEXT_METADATA_KEY not in metadata:
        return None
    if not bool(getattr(event, "internal", False)):
        return None
    try:
        return validate_final_return_context(metadata[FINAL_RETURN_CONTEXT_METADATA_KEY])
    except (TypeError, ValueError):
        return None


def final_return_delivery_state_from_event(
    event: Any,
) -> Optional[FinalReturnDeliveryState]:
    """Return the event-local result bridge only for the sealed envelope."""
    context = final_return_context_from_event(event)
    metadata = getattr(event, "metadata", None)
    if context is None or not isinstance(metadata, Mapping):
        return None
    state = metadata.get(FINAL_RETURN_DELIVERY_STATE_METADATA_KEY)
    if not isinstance(state, FinalReturnDeliveryState) or state.context != context:
        return None
    return state


def is_final_return_isolated_followup(event: Any) -> bool:
    """Whether Base marked a normal event behind an active final return."""
    metadata = getattr(event, "metadata", None)
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get(FINAL_RETURN_ISOLATED_FOLLOWUP_METADATA_KEY) is True
    )


def final_return_context_matches_profile(
    context: Mapping[str, Any], actual_profile: str,
) -> bool:
    """Require the executing profile, not only the envelope, to match."""
    try:
        from hermes_cli.profiles import normalize_profile_name

        expected = normalize_profile_name(
            validate_final_return_context(context)["responsible_agent"]
        )
        actual = normalize_profile_name(str(actual_profile or ""))
    except Exception:
        return False
    if not expected or not actual:
        return False
    try:
        from hermes_cli.workforce_org import (
            WorkforceOrganizationAbsentError,
            load_organization,
        )
    except Exception:
        return False

    try:
        organization = load_organization()
    except WorkforceOrganizationAbsentError:
        return expected == actual
    except Exception:
        return False
    try:
        expected_agent = organization.validate_execution_profile(expected).agent
        declared = organization.from_profile_path(actual)
        actual_agent = organization.validate_execution_profile(declared.agent).agent
    except Exception:
        return False
    return expected_agent == actual_agent


def complete_final_return_delivery(
    event: Any, outcome: WakeDeliveryOutcome,
) -> None:
    """Resolve the private wake bridge exactly once, if this is our event."""
    state = final_return_delivery_state_from_event(event)
    if state is not None:
        state.complete(outcome)


def encode_final_return_context(context: Mapping[str, Any]) -> str:
    """Encode the sealed context for the authenticated loopback API bridge."""
    normalized = validate_final_return_context(context)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_final_return_context(encoded: str) -> dict[str, str]:
    """Decode and validate a final-return context received over loopback HTTP."""
    if not isinstance(encoded, str) or not encoded or len(encoded) > 4096:
        raise ValueError("invalid final-return coordination header")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid final-return coordination header") from exc
    return validate_final_return_context(decoded)


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends.

    Mirrors ``gateway.session_context.async_delivery_supported`` but reads the
    capability off the adapter class (``supports_async_delivery``) instead of
    the request-scoped contextvar — background watchers run outside any bound
    session context. Adapters that don't declare the flag are push-capable.
    """
    return bool(getattr(adapter, "supports_async_delivery", True))


async def deliver_wake(
    adapter: Any,
    *,
    text: str,
    session_id: str = "",
    source: Any = None,
    coordination_context: Optional[Mapping[str, Any]] = None,
) -> Optional[WakeDeliveryOutcome]:
    """Deliver a wake turn to the session behind ``adapter``.

    ``session_id`` is the RAW session id (the ``X-Hermes-Session-Id`` value /
    ``state.db`` key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` used to build the synthetic event — required for
    push-capable adapters.

    Raises on failure (bad arguments, exhausted retries, HTTP error) so the
    caller can rewind/retry instead of treating the wake as delivered.
    """
    final_return_context = (
        validate_final_return_context(coordination_context)
        if coordination_context is not None
        else None
    )

    admission = None
    if final_return_context is not None:
        # Reserve the event before queueing a provider turn.  This is the
        # idempotency boundary for a watcher timeout: a duplicate observation
        # sees `sending`/`uncertain` and must never buy another model turn.
        from gateway.delivery_ledger import admit_coordination_final_return_turn

        if source is not None:
            admission_platform = str(getattr(getattr(source, "platform", ""), "value", getattr(source, "platform", "")))
            admission_chat_id = str(getattr(source, "chat_id", ""))
            admission_thread_id = getattr(source, "thread_id", None)
            admission_session = f"final-return:{final_return_context['request_root_id']}:{final_return_context['event_id']}"
        else:
            admission_platform = "api_server"
            admission_chat_id = str(session_id)
            admission_thread_id = None
            admission_session = str(session_id)
        admission = await asyncio.to_thread(
            admit_coordination_final_return_turn,
            request_root_id=final_return_context["request_root_id"],
            task_id=final_return_context["task_id"],
            event_id=int(final_return_context["event_id"]),
            responsible_agent=final_return_context["responsible_agent"],
            board_path=final_return_context["db_path"],
            session_key=admission_session,
            platform=admission_platform,
            chat_id=admission_chat_id,
            thread_id=admission_thread_id,
        )
        if admission.state == "acknowledged":
            return WakeDeliveryOutcome(
                "acknowledged", returned_message_id=admission.returned_message_id
            )
        if not admission.send_claimed:
            return WakeDeliveryOutcome(
                "uncertain",
                detail="final-return delivery is already admitted elsewhere",
            )

    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError(
                "deliver_wake: push-capable adapter requires a SessionSource"
            )
        from gateway.platforms.base import MessageEvent, MessageType

        metadata: dict[str, Any] = (
            {FINAL_RETURN_CONTEXT_METADATA_KEY: final_return_context}
            if final_return_context is not None
            else {}
        )
        delivery_state: Optional[FinalReturnDeliveryState] = None
        if final_return_context is not None:
            delivery_state = FinalReturnDeliveryState(
                context=final_return_context,
                completion=asyncio.get_running_loop().create_future(),
                claim_token=admission.claim_token,
            )
            metadata[FINAL_RETURN_DELIVERY_STATE_METADATA_KEY] = delivery_state
        synth_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            metadata=metadata,
        )
        try:
            await adapter.handle_message(synth_event)
        except Exception:
            if final_return_context is not None:
                from gateway.delivery_ledger import mark_coordination_final_return_pending

                await asyncio.to_thread(
                    mark_coordination_final_return_pending,
                    final_return_context["request_root_id"],
                    int(final_return_context["event_id"]),
                    error="wake_enqueue_failed",
                )
            raise
        if delivery_state is None:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.shield(delivery_state.completion),
                timeout=WAKE_TURN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            from gateway.delivery_ledger import mark_coordination_final_return_uncertain

            await asyncio.to_thread(
                mark_coordination_final_return_uncertain,
                final_return_context["request_root_id"],
                int(final_return_context["event_id"]),
                error="wake_turn_timeout",
            )
            return WakeDeliveryOutcome(
                "uncertain", detail="wake processing did not report a durable delivery outcome"
            )

    if not session_id:
        raise ValueError(
            "deliver_wake: non-push adapter (supports_async_delivery=False) "
            "requires the raw session id to self-post the wake turn"
        )
    if final_return_context is None:
        # Preserve the established helper seam for ordinary wake tests and
        # downstream adapters which replace the self-post implementation.
        await _self_post_chat_completion(adapter, text=text, session_id=session_id)
        return None
    else:
        return await _self_post_chat_completion(
            adapter,
            text=text,
            session_id=session_id,
            coordination_context=final_return_context,
            claim_token=admission.claim_token,
        )


async def _self_post_chat_completion(
    adapter: Any,
    *,
    text: str,
    session_id: str,
    coordination_context: Optional[Mapping[str, Any]] = None,
    claim_token: str = "",
) -> Optional[WakeDeliveryOutcome]:
    """POST the wake text to the in-pod API server as a normal session turn.

    Uses the adapter's own bind host/port/key (``ApiServerAdapter.__init__``).
    Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY`` being configured, so a missing key is a hard error —
    raise loudly rather than run the wake in a fresh fingerprint-derived
    session nobody is looking at.
    """
    import aiohttp

    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        # Wildcard bind address — connect over loopback.
        host = "127.0.0.1"
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError(
            "wake self-post requires API_SERVER_KEY: session continuation via "
            "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
            "server, so the wake cannot reach the target session"
        )

    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Hermes-Session-Id": session_id,
    }
    if coordination_context is not None:
        headers[INTERNAL_WAKE_HEADER] = _INTERNAL_WAKE_HEADER_VALUE
        headers[FINAL_RETURN_CONTEXT_HEADER] = encode_final_return_context(
            coordination_context
        )
        headers[FINAL_RETURN_CLAIM_HEADER] = _coordination_claim_header(claim_token)
    payload = {
        "model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
        "messages": [{"role": "user", "content": text}],
        "stream": False,
    }

    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:
                        # Global concurrency cap (max_concurrent_runs) —
                        # transient; back off and retry.
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) "
                            f"for session {session_id}"
                        )
                        logger.warning(
                            "%s; attempt %d/%d", last_err, attempt + 1, attempts
                        )
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        # Non-transient (auth/validation) — fail immediately.
                        if coordination_context is not None:
                            from gateway.delivery_ledger import mark_coordination_final_return_pending

                            await asyncio.to_thread(
                                mark_coordination_final_return_pending,
                                coordination_context["request_root_id"],
                                int(coordination_context["event_id"]),
                                error=f"api_http_{resp.status}",
                            )
                            return WakeDeliveryOutcome(
                                "pending", detail=f"API wake was rejected before delivery: HTTP {resp.status}"
                            )
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: "
                            f"HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    if coordination_context is not None:
                        receipt = str(
                            resp.headers.get(FINAL_RETURN_RECEIPT_HEADER, "")
                        ).strip()
                        if receipt:
                            # The HTTP response was the visible final send.
                            # Only this caller has observed its receipt, so
                            # complete the durable outbox acknowledgement
                            # here.  A post-response write failure is
                            # intentionally uncertain: do not replay it.
                            try:
                                from gateway.delivery_ledger import (
                                    mark_coordination_final_return_acknowledged,
                                )

                                acknowledged = await asyncio.to_thread(
                                    mark_coordination_final_return_acknowledged,
                                    coordination_context["request_root_id"],
                                    int(coordination_context["event_id"]),
                                    returned_message_id=receipt,
                                )
                            except Exception:
                                logger.warning(
                                    "final-return API receipt could not be acknowledged",
                                    exc_info=True,
                                )
                                return WakeDeliveryOutcome(
                                    "uncertain",
                                    detail="API response receipt could not be durably acknowledged",
                                )
                            return WakeDeliveryOutcome(
                                "acknowledged",
                                returned_message_id=acknowledged.returned_message_id,
                            )
                        return WakeDeliveryOutcome(
                            "uncertain",
                            detail="API wake returned without a persisted final-return receipt",
                        )
                    logger.info(
                        "wake self-post delivered for session %s (attempt %d)",
                        session_id,
                        attempt + 1,
                    )
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            if coordination_context is not None:
                from gateway.delivery_ledger import mark_coordination_final_return_uncertain

                await asyncio.to_thread(
                    mark_coordination_final_return_uncertain,
                    coordination_context["request_root_id"],
                    int(coordination_context["event_id"]),
                    error=type(exc).__name__,
                )
                return WakeDeliveryOutcome(
                    "uncertain", detail="final-return API transport outcome is ambiguous"
                )
            logger.warning(
                "wake self-post transient failure for session %s "
                "(attempt %d/%d): %s",
                session_id,
                attempt + 1,
                attempts,
                exc,
            )
            continue
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after "
        f"{attempts} attempts: {last_err}"
    ) from last_err


def _coordination_claim_header(claim_token: str) -> str:
    token = str(claim_token or "").strip()
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("invalid final-return delivery claim")
    return token
