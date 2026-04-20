import requests

LLAMA_URL = "http://localhost:8080/completion"

def call_llm(prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["</s>", "User: ", "Human: "]
    }

    response = requests.post(LLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["content"].strip()

