"""
Sliding-window conversation memory with token-budget awareness.
Prioritizes grounded turns when trimming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken

from app.config import settings
from app.evaluator.evaluator import Confidence

logger = logging.getLogger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class MemoryTurn:
    role: Role
    text: str
    source: str
    grounded: bool

    def token_count(self) -> int:
        return len(_TOKENIZER.encode(self.text))

    def to_message(self) -> dict:
        return {"role": self.role.value, "content": self.text}


class ConversationMemory:
    def __init__(self, token_budget: int = settings.memory_token_budget) -> None:
        self._turns: list[MemoryTurn] = []
        self._token_budget = token_budget

    def add_turn(
        self,
        role: Role,
        text: str,
        source: str = "",
        confidence: Optional[Confidence] = None,
    ) -> None:
        grounded = confidence in (Confidence.HIGH, Confidence.MEDIUM) if confidence else False
        turn = MemoryTurn(role=role, text=text, source=source, grounded=grounded)
        self._turns.append(turn)
        self._trim()

    def _total_tokens(self) -> int:
        return sum(t.token_count() for t in self._turns)

    def _trim(self) -> None:
        while self._total_tokens() > self._token_budget and len(self._turns) > 1:
            removed = self._evict_one()
            if not removed:
                break

    def _evict_one(self) -> bool:
        # Priority: remove ungrounded assistant turns first, then oldest
        for i, turn in enumerate(self._turns):
            if turn.role == Role.ASSISTANT and not turn.grounded:
                logger.debug("Evicting ungrounded assistant turn at index %d", i)
                self._turns.pop(i)
                return True
        # Evict oldest
        if self._turns:
            logger.debug("Evicting oldest turn")
            self._turns.pop(0)
            return True
        return False

    def to_messages(self) -> list[dict]:
        return [t.to_message() for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)