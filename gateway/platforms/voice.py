"""
Voice platform adapter for Vox gateway.

Connects to Vox over WebSocket. Receives transcribed user speech,
routes through Hermes pipeline. Streams agent responses back as
text chunks for TTS.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any

try:
    import websockets
except ImportError:
    websockets = None

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
    MessageType,
)
from gateway.session import SessionSource
from gateway.config import Platform

logger = logging.getLogger(__name__)


class VoiceAdapter(BasePlatformAdapter):
    """Adapter that bridges Vox voice gateway to Hermes agent pipeline."""

    MAX_MESSAGE_LENGTH = 100_000

    def __init__(self, config):
        super().__init__(config, Platform.VOICE)
        self._ws = None
        self._ws_task = None
        self._vox_url = (
            config.extra.get("vox_url", "ws://localhost:8600/adapter/hermes")
            if hasattr(config, "extra") and config.extra
            else "ws://localhost:8600/adapter/hermes"
        )
        self._active_calls: Dict[str, str] = {}

    async def connect(self) -> bool:
        if not websockets:
            logger.error("websockets package not installed")
            return False
        try:
            self._ws = await websockets.connect(self._vox_url)
            logger.info("Connected to Vox at %s", self._vox_url)
            await self._ws.send(json.dumps({"type": "ready", "platform": "hermes"}))
            self._ws_task = asyncio.create_task(self._listen())
            self._mark_connected()
            return True
        except Exception as e:
            logger.error("Failed to connect to Vox: %s", e)
            return False

    async def _listen(self):
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_vox_message(msg)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from Vox")
                except Exception as e:
                    logger.error("Error handling Vox message: %s", e, exc_info=True)
        except Exception:
            logger.warning("Vox connection closed, will attempt reconnect")
            self._mark_disconnected()
            await self._reconnect()

    async def _reconnect(self):
        while True:
            await asyncio.sleep(5)
            try:
                self._ws = await websockets.connect(self._vox_url)
                await self._ws.send(json.dumps({"type": "ready", "platform": "hermes"}))
                self._ws_task = asyncio.create_task(self._listen())
                self._mark_connected()
                logger.info("Reconnected to Vox")
                return
            except Exception as e:
                logger.warning("Reconnect failed: %s, retrying in 5s", e)

    async def _handle_vox_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "call_start":
            call_id = msg["callId"]
            agent = msg.get("agent", "default")
            self._active_calls[call_id] = agent
            logger.info("Voice call started: %s (agent: %s)", call_id, agent)

        elif msg_type == "text":
            call_id = msg.get("callId", "")
            content = msg.get("content", "")
            if not content.strip():
                return

            event = MessageEvent(
                text=content,
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.VOICE,
                    user_id="elliott",
                    chat_id=f"voice:{call_id}",
                ),
                message_id=f"voice-{call_id}-{id(msg)}",
            )

            if self._message_handler:
                response = await self._message_handler(event)
                if response:
                    await self._send_to_vox({
                        "type": "text",
                        "content": response,
                        "callId": call_id,
                    })

        elif msg_type == "call_end":
            call_id = msg.get("callId", "")
            self._active_calls.pop(call_id, None)
            logger.info("Voice call ended: %s", call_id)

    async def disconnect(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
        if self._ws:
            await self._ws.close()
        self._ws = None
        self._mark_disconnected()
        logger.info("Disconnected from Vox")

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        call_id = chat_id.replace("voice:", "")
        await self._send_to_vox({"type": "text", "content": content, "callId": call_id})
        return SendResult(success=True, message_id=f"voice-{call_id}-{id(content)}")

    async def edit_message(self, chat_id: str, message_id: str, content: str) -> bool:
        call_id = chat_id.replace("voice:", "")
        await self._send_to_vox({"type": "text", "content": content, "callId": call_id})
        return True

    async def send_typing(self, chat_id: str, is_typing: bool = True) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        call_id = chat_id.replace("voice:", "")
        agent = self._active_calls.get(call_id, "unknown")
        return {"name": f"Voice call {call_id}", "type": "dm", "chat_id": chat_id, "agent": agent}


    async def _send_to_vox(self, msg: dict) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send(json.dumps(msg))


def check_voice_requirements() -> bool:
    return websockets is not None
