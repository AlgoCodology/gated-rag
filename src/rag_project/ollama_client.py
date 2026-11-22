"""Thin wrapper around the two Ollama HTTP endpoints we use.

Ollama is a program running on your computer (port 11434) that hosts AI models.
We talk to it with plain HTTP requests:

    POST /api/embed  ->  text in, list of numbers (a "vector") out
    POST /api/chat   ->  question in, written answer out

The original code assumed every request succeeds. Here we check for the
common failures (Ollama not running, model not downloaded, slow response)
and raise ONE clear error type, `OllamaError`, with advice on how to fix it.
"""

import base64

import numpy as np
import requests

from . import config


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


def _post(endpoint: str, payload: dict) -> dict:
    """Send a POST request to Ollama and return the parsed JSON reply."""
    url = f"{config.OLLAMA_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS)
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {config.OLLAMA_URL}. Is it running? (try: `ollama serve`)"
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(
            f"Ollama did not answer within {config.REQUEST_TIMEOUT_SECONDS}s."
        ) from exc

    if not response.ok:
        # Ollama puts a human-readable reason in {"error": "..."}, e.g. a missing model.
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        model = payload.get("model")
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}: {detail}\n"
            f"If the model is missing, download it with: ollama pull {model}"
        )
    return response.json()


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into vectors. Returns an array of shape (len(texts), dimensions).

    We send the texts in batches: one giant request with thousands of chunks
    can time out or exhaust memory, while many small ones are safe.
    """
    batches = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": batch})
        if "embeddings" not in data or len(data["embeddings"]) != len(batch):
            raise OllamaError(f"Unexpected reply from the embed model: {str(data)[:200]}")
        batches.append(np.array(data["embeddings"], dtype=np.float32))
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(batches)


def ask_llm(prompt: str) -> str:
    """Send one prompt to the chat model and return its answer text."""
    data = _post(
        "/api/chat",
        {
            "model": config.CHAT_MODEL,
            "stream": False,  # wait for the full answer instead of word-by-word
            "options": {"num_ctx": config.LLM_NUM_CTX},
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    try:
        return data["message"]["content"]
    except KeyError as exc:
        raise OllamaError(f"Unexpected reply from the chat model: {str(data)[:200]}") from exc


def describe_image(image_bytes: bytes, prompt: str, model: str) -> str:
    """Ask a vision model to describe an image (used for image chunks)."""
    data = _post(
        "/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
        },
    )
    try:
        return data["message"]["content"].strip()
    except KeyError as exc:
        raise OllamaError(f"Unexpected reply from the vision model: {str(data)[:200]}") from exc
