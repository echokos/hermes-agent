"""Voice platform adapter for the local Vox gateway."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import json
import logging
import re
from typing import Any, Dict

try:
    import websockets
except ImportError:  # pragma: no cover - exercised through check fn
    websockets = None

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key

logger = logging.getLogger(__name__)

VOICE_CONTEXT_PREFIX = (
    "[Voice call context: This is live speech. Reply conversationally in short "
    "spoken turns. Do not read system messages, context compaction, tool output, "
    "file contents, logs, markdown, or long lists aloud. If tools or files are "
    "needed, give only a brief spoken summary and ask before continuing.]\n\n"
)

SUPPRESSED_TEXT_PREFIXES = (
    "[This response was interrupted",
    "ℹ Codex",
    "⚡ Interrupting current task",
    "⏳ Queued for the next turn",
    "⏩ Steered into current run",
    "⏳ Subagent working",
    "⏳ Compressing context",
    "⚕ Hermes Gateway Starting",
    "Hermes Gateway Starting",
    "Context compaction",
    "Compaction",
)

SUPPRESSED_TEXT_SNIPPETS = (
    "AUTH_REFRESH_FAILED",
    "auto-compaction was raised",
    "Claude Code returned an error result",
    "Codex error",
    "Codex app-server exited",
    "Codex streaming attempt superseded",
    "Error: Codex",
    "No conversation found with session ID",
    "Tool ",
    " returned error",
    "Traceback (most recent call last)",
    "execute_code",
    "mcp__",
    "read_file(",
    "read_file ",
)

SUPPRESSED_TEXT_REGEXES = (
    re.compile(r"^Conversation compact(?:ed|ion)\b", re.IGNORECASE),
    re.compile(r"^Context compact(?:ed|ion)\b", re.IGNORECASE),
    re.compile(r"^(?:We|Need) need to continue from (?:the )?summary\b", re.IGNORECASE),
    re.compile(r"^Summary from previous conversation\b", re.IGNORECASE),
    re.compile(r"^System (?:message|note|event):", re.IGNORECASE),
    re.compile(r"^(?:WARNING|ERROR|INFO|DEBUG)\s+[\w.:-]+:", re.IGNORECASE),
    re.compile(r"^File \"[^\"]+\", line \d+", re.IGNORECASE),
    re.compile(r"^<[/]?(?:analysis|commentary|tool|function|message|system)[\s>]", re.IGNORECASE),
)

LONG_MARKDOWN_LINE_LIMIT = 18

_ACTIVE_TRANSPORT_CALL: ContextVar[tuple[int, str] | None] = ContextVar(
    "voice_active_transport_call",
    default=None,
)


class VoiceAdapter(BasePlatformAdapter):
    """Bridge Vox WebSocket speech events into the Hermes gateway pipeline."""

    MAX_MESSAGE_LENGTH = 100_000
    REQUIRES_EDIT_FINALIZE = True
    # A complete response can reach GatewayStreamConsumer before its first
    # preview tick. Vox still needs the non-final checkpoint and stable stream
    # id so a transient final-frame failure can retry on the same stream.
    EXPECT_EDITS_ON_FINAL_SEND = True

    def __init__(self, config):
        super().__init__(config, Platform.VOICE)
        self._ws = None
        self._ws_task: asyncio.Task | None = None
        extra = getattr(config, "extra", {}) or {}
        self._vox_url = extra.get("vox_url", "ws://localhost:8600/adapter/hermes")
        self._platform_name = str(extra.get("platform_name") or "hermes").strip() or "hermes"
        self._active_calls: Dict[str, Dict[str, Any]] = {}
        self._session_calls: Dict[str, str] = {}
        self._message_calls: Dict[str, str] = {}
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._message_sequence = 0
        self._rejected_calls: set[str] = set()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not websockets:
            logger.error("websockets package not installed")
            return False
        try:
            self._ws = await websockets.connect(self._vox_url)
            logger.info("Connected to Vox at %s", self._vox_url)
            await self._ws.send(json.dumps({"type": "ready", "platform": self._platform_name}))
            self._ws_task = asyncio.create_task(self._listen())
            self._mark_connected()
            return True
        except Exception as exc:
            logger.error("Failed to connect to Vox: %s", exc)
            return False

    async def _listen(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_vox_message(msg)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from Vox")
                except Exception as exc:
                    logger.error("Error handling Vox message: %s", exc, exc_info=True)
        except Exception:
            logger.warning("Vox connection closed, will attempt reconnect")
            self._mark_disconnected()
            await self._reconnect()

    async def _reconnect(self) -> None:
        if not websockets:
            return
        while True:
            await asyncio.sleep(5)
            try:
                self._ws = await websockets.connect(self._vox_url)
                await self._ws.send(json.dumps({"type": "ready", "platform": self._platform_name}))
                self._ws_task = asyncio.create_task(self._listen())
                self._mark_connected()
                logger.info("Reconnected to Vox")
                return
            except Exception as exc:
                logger.warning("Reconnect failed: %s, retrying in 5s", exc)

    async def _handle_vox_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")

        if msg_type == "call_start":
            call_id = str(msg["callId"])
            session_id = str(msg.get("sessionId") or call_id).strip()[:160] or call_id
            agent = str(msg.get("agent", "default"))
            source = str(msg.get("source") or "voice").lower()
            direction = str(msg.get("direction") or "inbound").lower()
            is_elliott = msg.get("isElliott") is True

            # Phone Bridge is owner-only for inbound calls. Enforce the same
            # boundary here so a bridge regression cannot present an unknown
            # caller to Hermes as Elliott.
            if source == "phone" and direction != "outbound" and not is_elliott:
                self._active_calls.pop(call_id, None)
                self._rejected_calls.add(call_id)
                logger.warning(
                    "Rejected unverified inbound phone call: %s (agent: %s)",
                    call_id,
                    agent,
                )
                await self._send_to_vox({"type": "hangup", "callId": call_id})
                return

            self._rejected_calls.discard(call_id)
            previous_call_id = self._session_calls.get(session_id)
            if previous_call_id and previous_call_id != call_id:
                self._deactivate_call(previous_call_id)
            caller_name = str(msg.get("callerName") or "").strip()[:120]
            session_source = SessionSource(
                platform=Platform.VOICE,
                chat_type="dm",
                chat_id=f"voice:{session_id}",
            )
            self._active_calls[call_id] = {
                "agent": agent,
                "session_id": session_id,
                "source": source,
                "direction": direction,
                "is_elliott": is_elliott,
                "caller_name": caller_name,
                "context_sent": False,
                "message_ids": set(),
                "session_key": self._session_key_for_source(session_source),
            }
            self._session_calls[session_id] = call_id
            logger.info(
                "Voice call started: %s (agent: %s, source: %s, direction: %s, owner: %s)",
                call_id,
                agent,
                source,
                direction,
                is_elliott,
            )
            return

        if msg_type == "text":
            call_id = str(msg.get("callId", ""))
            if call_id in self._rejected_calls:
                logger.warning("Dropped transcript for rejected phone call: %s", call_id)
                return
            content = str(msg.get("content", ""))
            if not content.strip():
                return
            authored_content = content.strip()

            call = self._active_calls.get(call_id)
            if not call:
                logger.info("Dropped transcript for inactive voice call: %s", call_id)
                return
            user_id = "elliott"
            if call.get("source") == "phone":
                is_elliott = bool(call.get("is_elliott"))
                user_id = "elliott" if is_elliott else "phone-outbound"
                if not call.get("context_sent"):
                    direction = str(call.get("direction") or "inbound")
                    direction_label = "outgoing" if direction == "outbound" else "incoming"
                    direction_preposition = "to" if direction == "outbound" else "from"
                    caller_name = str(call.get("caller_name") or "").strip()
                    party = "Elliott" if is_elliott else caller_name or "the other party"
                    content = (
                        f"[Phone call context: {direction_label} FaceTime Audio call "
                        f"{direction_preposition} {party}. This is live speech, so respond naturally and "
                        "without markdown. To end the call, put [HANGUP] at the end of "
                        f"your response.]\n\n{content}"
                    )
                    call["context_sent"] = True
            elif not call.get("context_sent"):
                content = VOICE_CONTEXT_PREFIX + content
                call["context_sent"] = True

            agent_photo_request_text = None
            if call.get("source") == "voice" or (
                call.get("source") == "phone"
                and call.get("is_elliott") is True
            ):
                agent_photo_request_text = authored_content

            message_id = f"voice-{call_id}-{id(msg)}"
            event = MessageEvent(
                text=content,
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.VOICE,
                    user_id=user_id,
                    chat_id=f"voice:{call.get('session_id') or call_id}",
                ),
                message_id=message_id,
                agent_photo_request_text=agent_photo_request_text,
            )
            call["session_key"] = self._session_key_for_source(event.source)
            call["message_ids"].add(message_id)
            self._message_calls[message_id] = call_id

            token = _ACTIVE_TRANSPORT_CALL.set((id(self), call_id))
            try:
                await self.handle_message(event)
            finally:
                _ACTIVE_TRANSPORT_CALL.reset(token)
            return

        if msg_type == "turn_end":
            call_id = str(msg.get("callId", ""))
            self._deactivate_call(call_id)
            self._rejected_calls.discard(call_id)
            logger.info("Voice turn ended: %s", call_id)
            return

        if msg_type == "call_end":
            call_id = str(msg.get("callId", ""))
            call = self._active_calls.get(call_id)
            session_key = None
            if call:
                session_id = str(call.get("session_id") or call_id)
                if self._session_calls.get(session_id) == call_id:
                    session_key = call.get("session_key")
            self._deactivate_call(call_id)
            if session_key:
                await self.cancel_session_processing(str(session_key))
            self._rejected_calls.discard(call_id)
            logger.info("Voice call ended: %s", call_id)

    async def disconnect(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
        if self._ws:
            await self._ws.close()
        self._ws = None
        self._mark_disconnected()
        logger.info("Disconnected from Vox")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to=None,
        metadata=None,
    ) -> SendResult:
        call_id = self._resolve_call_id(chat_id, reply_to=reply_to)
        is_stream = (
            isinstance(metadata, dict) and metadata.get("expect_edits") is True
        )
        message_id = self._next_message_id(
            call_id or "inactive",
            stream=is_stream,
        )
        if call_id is None:
            return SendResult(success=True, message_id=message_id)

        if self._should_suppress_voice_output(content, metadata=metadata):
            return SendResult(success=True, message_id=message_id)

        if is_stream:
            cursor = metadata.get("stream_cursor")
            if not isinstance(cursor, str):
                cursor = ""
            stream = {
                "call_id": call_id,
                "cursor": cursor,
                "last_content": self._clean_stream_content(content, cursor),
                "final_sent": False,
                "lock": asyncio.Lock(),
            }
            self._streams[message_id] = stream
            if stream["last_content"]:
                await self._send_stream_frame(
                    call_id,
                    stream["last_content"],
                    is_final=False,
                    stream_id=message_id,
                )
            if metadata.get("notify") is True:
                try:
                    await self._finalize_stream(message_id, stream)
                except Exception as exc:
                    logger.warning(
                        "Voice stream final frame failed; preserving retry state: %s",
                        exc,
                    )
            return SendResult(success=True, message_id=message_id)

        await self._send_to_vox({"type": "text", "content": content, "callId": call_id})
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        stream = self._streams.get(message_id)
        if stream is None:
            if str(message_id).startswith("voice-stream-"):
                return SendResult(success=True, message_id=message_id)
            call_id = self._resolve_call_id(chat_id)
            if call_id is not None and not self._should_suppress_voice_output(content):
                await self._send_to_vox(
                    {"type": "text", "content": content, "callId": call_id}
                )
            return SendResult(success=True, message_id=message_id)
        async with stream["lock"]:
            if stream["final_sent"]:
                return SendResult(success=True, message_id=message_id)

            call_id = stream["call_id"]
            if not self._call_is_current(call_id, chat_id):
                stream["final_sent"] = True
                self._streams.pop(message_id, None)
                return SendResult(success=True, message_id=message_id)

            current = self._clean_stream_content(content, stream["cursor"])
            if self._should_suppress_voice_output(current):
                await self._finalize_stream_locked(message_id, stream)
                return SendResult(success=True, message_id=message_id)
            suffix, safe = self._safe_stream_suffix(stream["last_content"], current)
            if not safe:
                await self._finalize_stream_locked(message_id, stream)
                return SendResult(success=True, message_id=message_id)

            if suffix:
                await self._send_stream_frame(
                    call_id,
                    suffix,
                    is_final=False,
                    stream_id=message_id,
                )
            if not stream["last_content"].startswith(current):
                stream["last_content"] = current
            if finalize:
                await self._finalize_stream_locked(message_id, stream)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id: str, is_typing: bool = True) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        call_id = self._resolve_call_id(chat_id)
        call = self._active_calls.get(call_id) or {}
        return {
            "name": f"Voice call {call_id}",
            "type": "dm",
            "chat_id": chat_id,
            "agent": call.get("agent", "unknown"),
            "source": call.get("source", "voice"),
            "direction": call.get("direction"),
        }

    def _deactivate_call(self, call_id: str) -> None:
        call = self._active_calls.pop(call_id, None)
        if not call:
            return
        session_id = str(call.get("session_id") or call_id)
        if self._session_calls.get(session_id) == call_id:
            self._session_calls.pop(session_id, None)
        for message_id in call.get("message_ids", ()):
            self._message_calls.pop(message_id, None)
        for message_id, stream in tuple(self._streams.items()):
            if stream["call_id"] == call_id:
                self._streams.pop(message_id, None)

    def _session_key_for_source(self, source: SessionSource) -> str:
        return build_session_key(
            source,
            group_sessions_per_user=self.config.extra.get(
                "group_sessions_per_user", True
            ),
            thread_sessions_per_user=self.config.extra.get(
                "thread_sessions_per_user", False
            ),
        )

    def _resolve_call_id(self, chat_id: str, *, reply_to=None) -> str | None:
        session_id = str(chat_id).removeprefix("voice:")
        if reply_to:
            anchored_call_id = self._message_calls.get(str(reply_to))
            if anchored_call_id and self._call_is_current(anchored_call_id, chat_id):
                return anchored_call_id
            return None

        active_context = _ACTIVE_TRANSPORT_CALL.get()
        if active_context and active_context[0] == id(self):
            context_call_id = active_context[1]
            if self._call_is_current(context_call_id, chat_id):
                return context_call_id

        session_call_id = self._session_calls.get(session_id)
        if session_call_id and self._call_is_current(session_call_id, chat_id):
            return session_call_id
        return None

    def _call_is_current(self, call_id: str, chat_id: str) -> bool:
        call = self._active_calls.get(call_id)
        if not call:
            return False
        session_id = str(chat_id).removeprefix("voice:")
        call_session_id = str(call.get("session_id") or call_id)
        return (
            call_session_id == session_id
            and self._session_calls.get(session_id) == call_id
        )

    def _next_message_id(self, call_id: str, *, stream: bool = False) -> str:
        self._message_sequence += 1
        kind = "stream" if stream else "response"
        return f"voice-{kind}-{call_id}-{self._message_sequence}"

    @staticmethod
    def _clean_stream_content(content: str, cursor: str) -> str:
        text = str(content or "")
        return text[:-len(cursor)] if cursor and text.endswith(cursor) else text

    @classmethod
    def _should_suppress_voice_output(
        cls,
        content: str,
        *,
        metadata: dict | None = None,
    ) -> bool:
        if isinstance(metadata, dict) and metadata.get("non_conversational") is True:
            return True
        text = str(content or "").strip()
        if not text:
            return False
        if text.startswith(SUPPRESSED_TEXT_PREFIXES):
            return True
        if any(pattern.search(text) for pattern in SUPPRESSED_TEXT_REGEXES):
            return True
        if any(snippet in text for snippet in SUPPRESSED_TEXT_SNIPPETS):
            return True
        if cls._looks_like_file_or_log_dump(text):
            logger.info("Suppressed long file/log-style output on voice channel")
            return True
        return False

    @staticmethod
    def _looks_like_file_or_log_dump(text: str) -> bool:
        if len(text) < 900:
            return False
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) < LONG_MARKDOWN_LINE_LIMIT:
            return False
        markdown_or_log_lines = 0
        for line in lines:
            stripped = line.lstrip()
            if (
                stripped.startswith(("#", "-", "*", "```", ">", "|"))
                or re.match(r"^\d+[\.\)]\s+", stripped)
                or re.match(r"^[A-Z][A-Za-z0-9_.-]+Error:", stripped)
                or re.match(r"^[A-Z][A-Za-z0-9_.-]+\s+(WARNING|ERROR|INFO):", stripped)
            ):
                markdown_or_log_lines += 1
        return markdown_or_log_lines >= 8

    @staticmethod
    def _safe_stream_suffix(previous: str, current: str) -> tuple[str, bool]:
        if current.startswith(previous):
            return current[len(previous):], True
        if previous.startswith(current):
            return "", True
        if previous and current.count(previous) == 1:
            start = current.index(previous) + len(previous)
            return current[start:], True
        return "", False

    async def _send_stream_frame(
        self,
        call_id: str,
        content: str,
        *,
        is_final: bool,
        stream_id: str,
    ) -> None:
        call = self._active_calls.get(call_id)
        if not call:
            return
        session_id = str(call.get("session_id") or call_id)
        if self._session_calls.get(session_id) != call_id:
            return
        await self._send_to_vox(
            {
                "type": "text",
                "content": content,
                "callId": call_id,
                "stream": True,
                "isFinal": is_final,
                "streamId": stream_id,
            }
        )

    async def _finalize_stream(
        self,
        message_id: str,
        stream: dict[str, Any],
    ) -> None:
        async with stream["lock"]:
            await self._finalize_stream_locked(message_id, stream)

    async def _finalize_stream_locked(
        self,
        message_id: str,
        stream: dict[str, Any],
    ) -> None:
        if stream["final_sent"]:
            return
        await self._send_stream_frame(
            stream["call_id"],
            "",
            is_final=True,
            stream_id=message_id,
        )
        stream["final_sent"] = True
        if self._streams.get(message_id) is stream:
            self._streams.pop(message_id, None)

    async def _send_to_vox(self, msg: dict[str, Any]) -> None:
        if self._ws and not getattr(self._ws, "closed", False):
            await self._ws.send(json.dumps(msg))


def check_voice_requirements() -> bool:
    return websockets is not None
