"""A small, heavily documented RAG (Retrieval-Augmented Generation) project.

Start with README.md, then read the modules in pipeline order:
config -> chunking -> ollama_client -> index -> retrieval -> rag
"""


def main() -> None:
    """Console-script entry point (`uv run rag_project`)."""
    # Imported here, not at the top, so `python -m rag_project.rag` doesn't
    # load rag.py twice.
    from .rag import main as run

    run()
