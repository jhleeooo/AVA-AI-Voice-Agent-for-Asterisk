#!/usr/bin/env python3
"""
Protocol-level tests for Local AI Server against the current contract in
local_ai_server/main.py and docs/local-ai-server/PROTOCOL.md.

These tests assume the server is reachable at ws://127.0.0.1:8765.
"""

import asyncio
import base64
import json
import os
import sys
import logging
from typing import Optional

import websockets
import pytest
import socket

WS_URL = os.getenv("LOCAL_WS_URL", "ws://127.0.0.1:8765")


def _server_available(url: str) -> bool:
    try:
        host, port = url.replace("ws://", "").replace("wss://", "").split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False

# Mark as integration: requires a running local-ai-server WebSocket
pytestmark = pytest.mark.skipif(
    not _server_available(WS_URL),
    reason="Requires local AI server at 127.0.0.1:8765. Start local_ai_server to enable.",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _synthesize_speech_pcm16(text: str = "testing one two three", *, sample_rate_hz: int = 16000) -> bytes:
    """Round-trip through the server's own TTS to get real speech audio.

    STT cannot produce a non-empty final transcript from silence (the idle
    finalizer explicitly suppresses empty-transcript finals), so tests that
    exercise STT need actual speech. Using the server's TTS keeps the tests
    self-contained without shipping a binary audio fixture.
    """
    async with websockets.connect(WS_URL, max_size=None) as ws:
        req = {
            "type": "tts_request",
            "text": text,
            "call_id": "synth-source",
            "request_id": "synth1",
            "output_encoding": "linear16",
            "output_sample_rate_hz": sample_rate_hz,
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert resp["type"] == "tts_response"
        pcm = base64.b64decode(resp["audio_data"])
        assert len(pcm) > 0
        return pcm


async def test_tts_roundtrip() -> None:
    async with websockets.connect(WS_URL, max_size=None) as ws:
        req = {
            "type": "tts_request",
            "text": "Hello from protocol test.",
            "call_id": "test-call",
            "request_id": "t1",
        }
        await ws.send(json.dumps(req))
        # Per docs/local-ai-server/PROTOCOL.md, a `tts_request` gets a single
        # `tts_response` JSON message with base64 audio (no separate binary
        # frame) — the `tts_audio` + binary pairing is only for `mode=full`.
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert resp["type"] == "tts_response"
        assert resp["encoding"] == "mulaw"
        pcm = base64.b64decode(resp["audio_data"])
        logger.info("Received TTS audio bytes: %s", len(pcm))
        assert len(pcm) > 0


async def test_stt_binary_flow() -> None:
    speech_pcm = await _synthesize_speech_pcm16("testing one two three")
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "set_mode", "mode": "stt", "call_id": "demo"}))
        # mode_ready is optional; server may not echo. Continue regardless.
        try:
            _ = await asyncio.wait_for(ws.recv(), timeout=2.0)
        except Exception:
            pass
        await ws.send(speech_pcm)
        # Idle finalizer fires ~5s after the last audio frame; give it room.
        for _ in range(5):
            msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
            if isinstance(msg, (bytes, bytearray)):
                # ignore any unexpected binary frames
                continue
            evt = json.loads(msg)
            if evt.get("type") == "stt_result" and evt.get("is_final"):
                logger.info("Final STT text: '%s'", evt.get("text", ""))
                return
        raise AssertionError("Did not receive a final stt_result for synthesized speech")


async def test_full_audio_frame() -> None:
    speech_pcm = await _synthesize_speech_pcm16("what time is it")
    async with websockets.connect(WS_URL, max_size=None) as ws:
        req = {
            "type": "audio",
            "mode": "full",
            "rate": 16000,
            "call_id": "full-test",
            "request_id": "r1",
            "data": base64.b64encode(speech_pcm).decode("utf-8"),
        }
        await ws.send(json.dumps(req))
        # Expect stt_result partials/final then llm_response then tts_audio + binary
        saw_final = False
        saw_llm = False
        saw_tts_meta = False
        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
            if isinstance(msg, (bytes, bytearray)):
                if saw_tts_meta:
                    logger.info("Received TTS audio bytes: %s", len(msg))
                    assert saw_final and saw_llm
                    return
                continue
            evt = json.loads(msg)
            if evt.get("type") == "stt_result" and evt.get("is_final"):
                saw_final = True
            elif evt.get("type") == "llm_response":
                saw_llm = True
            elif evt.get("type") == "tts_audio":
                saw_tts_meta = True
        raise AssertionError(
            f"Full-mode pipeline incomplete: saw_final={saw_final} saw_llm={saw_llm} saw_tts_meta={saw_tts_meta}"
        )


async def main() -> None:
    results = {}
    for name, coro in (
        ("tts_roundtrip", test_tts_roundtrip),
        ("stt_binary_flow", test_stt_binary_flow),
        ("full_audio_frame", test_full_audio_frame),
    ):
        try:
            await coro()
            results[name] = True
        except Exception:
            logger.exception("Test failed: %s", name)
            results[name] = False
    total = sum(results.values())
    print(f"Local AI Server protocol tests passed: {total}/3")
    sys.exit(0 if total == 3 else 1)


if __name__ == "__main__":
    asyncio.run(main())
