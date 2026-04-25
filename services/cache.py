from __future__ import annotations

import hashlib
from collections import OrderedDict


class ResponseCache:
    """LRU-evicting in-memory query cache.

    Prevents unbounded memory growth while still caching recent responses.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size

    def build_key(self, query: str, top_k: int) -> str:
        payload = f"{query.strip().lower()}::{top_k}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> dict | None:
        if key in self._cache:
            self._cache.move_to_end(key)  # mark as recently used
            return self._cache[key]
        return None

    def set(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        # Evict least-recently-used entries when over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)