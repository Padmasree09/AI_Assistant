from __future__ import annotations

import hashlib


class ResponseCache:
    """Simple in-memory query cache."""

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def build_key(self, query: str, top_k: int) -> str:
        payload = f"{query.strip().lower()}::{top_k}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self._cache[key] = value