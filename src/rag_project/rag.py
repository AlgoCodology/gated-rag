"""Step 4 of RAG (Generation) and the command-line chat loop.

The whole pipeline, in the order it runs:

    docs/*.md --chunking--> chunks --embedding--> vectors      (index.py, cached)
    question  --embedding--> vector --similarity--> top chunks (retrieval.py)
    top chunks + question --prompt--> chat model --> answer    (this file)

Run it:
    uv run python -m rag_project.rag --role public      # customer view
    uv run python -m rag_project.rag --role internal     # employee view
"""

import argparse

from . import config
from .index import Index, build_index
from .ollama_client import OllamaError, ask_llm
from .retrieval import retrieve

# The prompt is the "instructions + evidence" we hand to the chat model.
# Notice what is NOT here: any rule about guessing who the user is. Access
# control is enforced in retrieval.py, so secrets never reach this prompt.
PROMPT_TEMPLATE = """Answer the question using ONLY the context below. Follow these rules:
1. If the answer isn't in the context, say "I don't know".
2. If the context contains the answer, say it in your own words without making anything up.
3. Be terse, concise and clear.
4. Do not entertain obscene questions or provide vulgar or inappropriate information.

Context:
{context}

Question: {question}"""


def answer(index: Index, question: str, role: str, show_context: bool = True) -> str:
    """Retrieve context for `question` and ask the chat model to answer from it."""
    results = retrieve(index, question, role)
    context = "\n\n".join(chunk.text for chunk, _score in results)

    if show_context:
        print("\nRetrieved context (score | access | text):")
        for chunk, score in results:
            print(f"  {score:.3f} | {chunk.access:8} | {chunk.text[:80]}")

    return ask_llm(PROMPT_TEMPLATE.format(context=context, question=question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the documents in docs/.")
    parser.add_argument(
        "--role",
        choices=sorted(config.ROLE_PERMISSIONS),
        default="public",  # least privilege: you must ask for internal access
        help="who is asking: 'public' (customer) or 'internal' (employee)",
    )
    parser.add_argument("--rebuild", action="store_true", help="ignore the embedding cache")
    parser.add_argument("--hide-context", action="store_true", help="don't print retrieved chunks")
    args = parser.parse_args()

    try:
        index = build_index(use_cache=not args.rebuild)
        print(f"Indexed {len(index.chunks)} chunks. Role: {args.role}. Press Ctrl+C to quit.\n")
        while True:
            question = input("Ask: ").strip()
            if question:
                print("\n" + answer(index, question, args.role, not args.hide_context) + "\n")
    except OllamaError as exc:
        raise SystemExit(f"Error: {exc}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
