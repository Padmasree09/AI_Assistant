from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class MemoryTurn:
    query: str
    answer: str


class SessionMemory:
    """In-memory short-term memory by session."""

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._store: dict[str, Deque[MemoryTurn]] = defaultdict(lambda: deque(maxlen=self.max_turns))

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        self._store[session_id].append(MemoryTurn(query=query, answer=answer))

    def get_history(self, session_id: str, limit: int = 3) -> list[MemoryTurn]:
        turns = list(self._store.get(session_id, []))
        return turns[-limit:]

    def format_history(self, session_id: str, limit: int = 3) -> str:
        turns = self.get_history(session_id, limit=limit)
        if not turns:
            return ""

        lines = []
        for idx, turn in enumerate(turns, start=1):
            lines.append(f"Turn {idx} Question: {turn.query}")
            lines.append(f"Turn {idx} Answer: {turn.answer}")
        return "\n".join(lines)