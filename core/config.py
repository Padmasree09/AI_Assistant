from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    llama_url: str
    llama_timeout_seconds: int
    agent_max_tokens: int
    planner_max_tokens: int
    synthesis_max_tokens: int
    critique_max_tokens: int
    use_llm_planner: bool
    enable_critic: bool
    retriever_mode: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    qdrant_local_path: str
    embedding_backend: str
    embedding_model_name: str
    embedding_size: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()

    def _env_bool(name: str, default: str) -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        llama_url=os.getenv("LLAMA_URL", "http://localhost:8080/completion"),
        llama_timeout_seconds=int(os.getenv("LLAMA_TIMEOUT_SECONDS", "120")),
        agent_max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "192")),
        planner_max_tokens=int(os.getenv("PLANNER_MAX_TOKENS", "96")),
        synthesis_max_tokens=int(os.getenv("SYNTHESIS_MAX_TOKENS", "256")),
        critique_max_tokens=int(os.getenv("CRITIQUE_MAX_TOKENS", "96")),
        use_llm_planner=_env_bool("USE_LLM_PLANNER", "false"),
        enable_critic=_env_bool("ENABLE_CRITIC", "true"),
        retriever_mode=os.getenv("RETRIEVER_MODE", "server").lower(),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
        qdrant_local_path=os.getenv("QDRANT_LOCAL_PATH", "data/qdrant"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "hash").lower(),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        embedding_size=int(os.getenv("EMBEDDING_SIZE", "384")),
    )
