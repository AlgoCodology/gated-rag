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


def _post(endpoint: str, payload: dict, timeout: int | None = None) -> dict:
    """Send a POST request to Ollama and return the parsed JSON reply."""
    url = f"{config.OLLAMA_URL}{endpoint}"
    timeout = timeout or config.REQUEST_TIMEOUT_SECONDS
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {config.OLLAMA_URL}. Is it running? (try: `ollama serve`)"
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(
            f"Ollama did not answer within {timeout}s."
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


def ask_llm(prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
    """Send one prompt to the chat model and return its answer text.

    `max_tokens` caps the length of the reply and `temperature=0` makes it
    repeatable; the relevance judge (rerank.py) uses both.
    """
    options: dict = {"num_ctx": config.LLM_NUM_CTX}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    data = _post(
        "/api/chat",
        {
            "model": config.CHAT_MODEL,
            "stream": False,  # wait for the full answer instead of word-by-word
            "options": options,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    try:
        return data["message"]["content"]
    except KeyError as exc:
        raise OllamaError(f"Unexpected reply from the chat model: {str(data)[:200]}") from exc


def shrink_image(image_bytes: bytes, max_side: int) -> bytes:
    """Downscale big images: fewer pixels = fewer image tokens = much faster on a CPU."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    if max(image.size) <= max_side:
        return image_bytes
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue()


def describe_image(image_bytes: bytes, prompt: str, model: str) -> str:
    """Ask a vision model to describe an image (used for image chunks)."""
    image_bytes = shrink_image(image_bytes, config.VISION_MAX_IMAGE_SIDE)
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
            "options": {"num_predict": 400},  # a caption never needs more than this
        },
        timeout=config.VISION_TIMEOUT_SECONDS,
    )
    try:
        return data["message"]["content"].strip()
    except KeyError as exc:
        raise OllamaError(f"Unexpected reply from the vision model: {str(data)[:200]}") from exc
