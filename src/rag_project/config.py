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
CAPTION_CACHE_FILE = PROJECT_ROOT / ".cache" / "captions.json"  # saved image descriptions

# --- Ollama (the local server that runs the AI models) -----------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"  # turns text into vectors
CHAT_MODEL = "qwen3:4b-instruct"  # writes the final answer
REQUEST_TIMEOUT_SECONDS = 120  # give up on Ollama after this long
LLM_NUM_CTX = 8192  # reading space (tokens) for the chat model; Ollama's default is smaller

# Optional: a vision model that describes images that have no `<name>.alt.txt`
# sidecar file (e.g. "moondream" or "qwen2.5vl:3b"). None = images need a sidecar.
VISION_MODEL = None

# --- RAG behaviour -----------------------------------------------------------
EMBED_BATCH_SIZE = 64  # chunks sent to the embed model per HTTP request

# nomic-embed-text was trained with task prefixes; using them improves matching.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

# Chunk sizes (characters). See chunking.py for children vs parents.
CHILD_MAX_CHARS = 800  # a child longer than this is split (sentences / rows / items)
PARENT_MAX_CHARS = 2000  # the section text handed to the chat model per hit

CANDIDATES = 12  # how many small child chunks to fetch per question
TOP_K = 5  # how many parent sections (after de-duplication) go into the prompt
# If even the best child scores below this, we skip the chat model and say
# "I don't know". Calibrate it with `python -m rag_project.evaluate`.
# Measured on eval/questions.jsonl: the weakest answerable question scored 0.618,
# the best off-topic one 0.733 (the ranges overlap), so 0.60 is a cautious choice
# that never rejects a question we can answer.
SCORE_THRESHOLD = 0.60

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
# A file's default level comes from its top-level folder inside docs/.
FOLDER_ACCESS = {"public": PUBLIC, "internal": INTERNAL}
