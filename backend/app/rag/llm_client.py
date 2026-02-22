"""
Groq LLM client with streaming support, latency tracking, and token capture.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncGroq] = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


@dataclass
class LLMResponse:
    content: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model: str


async def call_streaming(
    system_prompt: str,
    messages: list[dict],
    model: str,
    timeout_seconds: float = 30.0,
) -> AsyncIterator[str]:
    """
    Yields token chunks as they arrive from the Groq streaming API.
    Caller is responsible for accumulating the full response.
    """
    client = _get_client()
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            stream=True,
            timeout=timeout_seconds,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        logger.error("Groq streaming error: %s", e)
        raise


async def call_with_metrics(
    system_prompt: str,
    messages: list[dict],
    model: str,
    timeout_seconds: float = 30.0,
) -> LLMResponse:
    """
    Non-streaming call that returns full response with token usage.
    Used for evaluation and testing.
    """
    client = _get_client()
    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            stream=False,
            timeout=timeout_seconds,
        )
    except Exception as e:
        logger.error("Groq call error: %s", e)
        raise
    latency_ms = (time.monotonic() - start) * 1000

    usage = response.usage
    return LLMResponse(
        content=response.choices[0].message.content or "",
        tokens_input=usage.prompt_tokens if usage else 0,
        tokens_output=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
        model=response.model,
    )


async def stream_and_measure(
    system_prompt: str,
    messages: list[dict],
    model: str,
    timeout_seconds: float = 30.0,
) -> tuple[AsyncIterator[str], "LatencyTracker"]:
    """
    Returns a stream iterator plus a tracker that records timing on first and last token.
    """
    tracker = LatencyTracker()
    client = _get_client()

    async def _stream() -> AsyncIterator[str]:
        tracker.start()
        first = True
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                stream=True,
                timeout=timeout_seconds,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first:
                        tracker.mark_first_token()
                        first = False
                    yield delta.content
        except Exception as e:
            logger.error("Groq streaming error: %s", e)
            raise
        finally:
            tracker.end()

    return _stream(), tracker


class LatencyTracker:
    def __init__(self) -> None:
        self._start: float = 0.0
        self._first_token: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> None:
        self._start = time.monotonic()

    def mark_first_token(self) -> None:
        self._first_token = time.monotonic()

    def end(self) -> None:
        self._end = time.monotonic()

    @property
    def total_ms(self) -> float:
        if self._end is None:
            return 0.0
        return (self._end - self._start) * 1000

    @property
    def ttfb_ms(self) -> float:
        if self._first_token is None:
            return 0.0
        return (self._first_token - self._start) * 1000