"""Central settings. Change a value here and the whole project follows.

Everything that is a "knob" lives in this one file so you can experiment
(different models, chunk sizes, number of results) without hunting through code.
"""

from pathlib import Path

# --- Where things live -------------------------------------------------------
# This file is  <project>/src/rag_project/config.py, so parents[2] is <project>.
# Using the file's own location (instead of the current working directory)
# means the script works no matter which folder you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"  # the knowledge base: Markdown files we search
CACHE_FILE = PROJECT_ROOT / ".cache" / "index.npz"  # saved embeddings (see index.py)

# --- Ollama (the local server that runs the AI models) -----------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"  # turns text into vectors
CHAT_MODEL = "qwen3:4b-instruct"  # writes the final answer
REQUEST_TIMEOUT_SECONDS = 120  # give up on Ollama after this long

# --- RAG behaviour -----------------------------------------------------------
EMBED_BATCH_SIZE = 64  # chunks sent to the embed model per HTTP request
TOP_K = 5  # how many chunks to retrieve per question

# --- Access control ----------------------------------------------------------
# Every chunk is tagged with one of these levels (see chunking.py).
# A user "role" is allowed to see the levels listed for it.
PUBLIC = "public"
INTERNAL = "internal"
ROLE_PERMISSIONS = {
    "public": {PUBLIC},  # a customer: public information only
    "internal": {PUBLIC, INTERNAL},  # an employee: everything
}
# Chunks with no tag are treated as INTERNAL: if someone forgets to label a
# paragraph, it stays hidden instead of leaking. ("Fail closed".)
DEFAULT_ACCESS = INTERNAL
