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
JUDGE_CACHE_FILE = PROJECT_ROOT / ".cache" / "judge.json"  # saved relevance verdicts (rerank.py)

# --- Ollama (the local server that runs the AI models) -----------------------
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"  # turns text into vectors
CHAT_MODEL = "qwen3:4b-instruct"  # writes the final answer
REQUEST_TIMEOUT_SECONDS = 120  # give up on Ollama after this long
LLM_NUM_CTX = 8192  # reading space (tokens) for the chat model; Ollama's default is smaller

# Optional: a vision model that describes images that have no `<name>.alt.txt`
# sidecar file (e.g. "moondream" or "qwen2.5vl:3b"). None = images need a sidecar.
VISION_MODEL = "qwen2.5vl:3b"
# Which description wins when an image has both a hand-written `<name>.alt.txt`
# and a vision-model caption: "sidecar", "vision" or "both" (hand-written text first, then the model's). Compare with `evaluate --ablate`.
IMAGE_DESCRIPTIONS = "both"
VISION_TIMEOUT_SECONDS = 900  # captioning runs once per image and is cached; a CPU is slow
VISION_MAX_IMAGE_SIDE = 896  # images are shrunk to this many pixels on the long side

# --- RAG behaviour -----------------------------------------------------------
EMBED_BATCH_SIZE = 64  # chunks sent to the embed model per HTTP request

# nomic-embed-text was trained with task prefixes; using them improves matching.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

# Chunk sizes (characters). See chunking.py for children vs parents.
CHILD_MAX_CHARS = 800  # a child longer than this is split (sentences / rows / items)
PARENT_MAX_CHARS = 2000  # the section text handed to the chat model per hit

# The vector store is chosen AUTOMATICALLY from the corpus size (number of chunks):
#   fewer than QDRANT_MIN_CHUNKS -> "numpy" (in-memory, exact, fastest for small corpora)
#   at least QDRANT_MIN_CHUNKS   -> "qdrant" (on-disk vector database, scales to millions)
# This is the one number to tune. It is kept low for now (the corpus has ~400 chunks) so
# Qdrant gets exercised; raise it (e.g. to 20000) once Qdrant has proven itself.
QDRANT_MIN_CHUNKS = 300
QDRANT_URL = None  # None = embedded local mode (a folder under .cache/); or e.g. "http://localhost:6333"

CANDIDATES = 12 # how many small child chunks to fetch per question
TOP_K = 5  # how many parent sections (after de-duplication) go into the prompt
# If even the best child scores below this, we skip the chat model and say
# "I don't know". Calibrate it with `python -m rag_project.evaluate`.
# Measured on the 60 answerable questions: the weakest scored 0.583, while near-topic
# questions with no answer score up to ~0.73 (the ranges overlap), so no cutoff
# separates them. 0.55 only rejects clearly unrelated questions and never an answerable one.
# The judge model (see rerank.py) is the better filter for the near-topic cases.
SCORE_THRESHOLD = 0.55

# Optional re-ranking (rerank.py): read the best few chunks together with the question
# and re-order them. Slower than plain search, but sharper. Backends:
#   "none"   plain embedding search (default)
#   "bge"    a small cross-encoder, fast (~0.1 s per chunk). Needs: uv sync --extra rerank
#   "bge-m3" a bigger cross-encoder (568M parameters, ~2.3 GB of memory), multilingual
#   "judge"  the chat model rates each chunk 0-9, slow (~4 s per chunk), no extra install
#   "hybrid" bge reads 20 candidates, the judge reads only the best HYBRID_JUDGE_TOP of them
# Compare them with `python -m rag_project.evaluate --rerank bge judge`.
RERANKER = "hybrid"  # measured best (see evaluate --rerank); "none" = fastest
BGE_MODELS = {  # cross-encoder weights per backend, downloaded on first use
    "bge": "BAAI/bge-reranker-base",  # 278M parameters, MIT licence
    "hybrid": "BAAI/bge-reranker-base",
    "bge-m3": "BAAI/bge-reranker-v2-m3",  # Apache-2.0
}
RERANK_CANDIDATES = {"bge": 20, "bge-m3": 20, "hybrid": 20, "judge": 8}  # child chunks each backend reads
HYBRID_JUDGE_TOP = 6  # how many of the 20 the judge model reads in "hybrid"
# Every backend reports relevance on a 0-1 scale. If even the best chunk scores below
# this, the question is treated as unanswerable ("I don't know"). Calibrate with evaluate.
# Measured for "hybrid": 0.30 turns away 4/11 unanswerable and wrongly rejects 0/60 answerable.
RERANK_THRESHOLD = 0.30

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
