# What changed from the original script

The original was one file, `rag.py`, that ran everything at import time. It now has one module per pipeline step.

| # | Problem in the original | Fix | Where |
|---|---|---|---|
| 1 | `r.json()["embeddings"]` crashed with a cryptic `KeyError` or `ConnectionError` when Ollama was down or a model missing; no timeout | Timeouts, HTTP status checks, one `OllamaError` with advice (`ollama pull ...`) | [ollama_client.py](../src/rag_project/ollama_client.py) |
| 2 | Re-embedded every chunk on every start | Vectors cached in `.cache/index.npz`, invalidated by a content fingerprint | [index.py](../src/rag_project/index.py) |
| 3 | All chunks sent in one request | Batches of 64 (`EMBED_BATCH_SIZE`) | [ollama_client.py](../src/rag_project/ollama_client.py) |
| 4 | Norms recomputed on every query | Vectors normalized once; scoring is a plain dot product | [index.py](../src/rag_project/index.py), [retrieval.py](../src/rag_project/retrieval.py) |
| 5 | Everything ran on import, so nothing was testable | Small functions, `main()`, `if __name__ == "__main__"` guard, offline tests | [rag.py](../src/rag_project/rag.py), [tests/](../tests/test_rag.py) |
| 6 | Prompt rule 5 asked the model to guess whether the user is a customer | Per-paragraph `access` tags; forbidden chunks are filtered out before the model sees them; `--role` flag | [chunking.py](../src/rag_project/chunking.py), [retrieval.py](../src/rag_project/retrieval.py) |
| 7 | `Path("docs")` depended on the launch directory | Paths resolved from the project root | [config.py](../src/rag_project/config.py) |
| 8 | `rag_project` script entry printed "Hello" | Entry point now runs the chat | [\_\_init\_\_.py](../src/rag_project/__init__.py) |

Other details:

- The handbook's paragraphs are now tagged. The alarm code, Wi-Fi password, repair contact, PTO and shift rules are `internal`. The hours, the drink and the dog policy are `public`.
- Project guides live in `guide/`, not `docs/`, because everything in `docs/` is indexed as knowledge.
- Behavior difference: the default role is `public` (least privilege).
- Not changed: `expense.py`, which is unrelated to RAG.
