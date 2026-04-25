from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryTurn:
    query: str
    answer: str


class SessionMemory:
    """SQLite-backed short-term memory by session.

    Survives server restarts while staying lightweight (no extra deps).
    """

    def __init__(self, db_path: str = "data/memory.db", max_turns: int = 8) -> None:
        self.max_turns = max_turns
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id TEXT NOT NULL,"
            "  query TEXT NOT NULL,"
            "  answer TEXT NOT NULL,"
            "  ts REAL NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, ts)"
        )
        self._conn.commit()

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        self._conn.execute(
            "INSERT INTO turns (session_id, query, answer, ts) VALUES (?, ?, ?, ?)",
            (session_id, query, answer, time.time()),
        )
        self._conn.commit()
        # Trim old turns beyond max_turns for this session
        self._conn.execute(
            "DELETE FROM turns WHERE id NOT IN ("
            "  SELECT id FROM turns WHERE session_id = ? ORDER BY ts DESC LIMIT ?"
            ") AND session_id = ?",
            (session_id, self.max_turns, session_id),
        )
        self._conn.commit()

    def get_history(self, session_id: str, limit: int = 3) -> list[MemoryTurn]:
        rows = self._conn.execute(
            "SELECT query, answer FROM turns WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        # Return in chronological order (oldest first)
        return [MemoryTurn(query=r[0], answer=r[1]) for r in reversed(rows)]

    def format_history(self, session_id: str, limit: int = 3) -> str:
        turns = self.get_history(session_id, limit=limit)
        if not turns:
            return ""

        lines = []
        for idx, turn in enumerate(turns, start=1):
            lines.append(f"Turn {idx} Question: {turn.query}")
            lines.append(f"Turn {idx} Answer: {turn.answer}")
        return "\n".join(lines)