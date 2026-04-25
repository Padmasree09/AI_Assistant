import requests

from core.config import get_settings


def _clean_response_text(text: str) -> str:
    cleaned = text.strip()

    removable_prefixes = [
        "Answer:",
        "Final Answer:",
        "Final answer:",
        "Response:",
        "Analysis:",
        "Summary:",
    ]
    for prefix in removable_prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    return cleaned.strip()


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    settings = get_settings()
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["</s>", "User: ", "Human: "],
    }

    response = requests.post(settings.llama_url, json=payload, timeout=settings.llama_timeout_seconds)
    response.raise_for_status()
    return _clean_response_text(response.json()["content"])

