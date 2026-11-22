"""Step 4 of RAG (Generation) and the command-line chat loop.

The whole pipeline, in the order it runs:

    docs/**/* --loaders--> chunks --embedding--> vectors       (loaders.py, index.py, cached)
    question  --embedding--> vector --similarity--> children --> parent sections (retrieval.py)
    top chunks + question --prompt--> chat model --> answer    (this file)

Run it:
    uv run python -m rag_project.rag --role public      # customer view
    uv run python -m rag_project.rag --role internal     # employee view
"""

import argparse

from dataclasses import dataclass

from . import config
from .index import Index, build_index
from .ollama_client import OllamaError, ask_llm
from .retrieval import Context, retrieve, u_shaped

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

NO_ANSWER = "I don't know."


@dataclass
class Answer:
    text: str
    contexts: list[Context]  # best first: these are the sources


def build_prompt(contexts: list[Context], question: str) -> str:
    """Assemble the prompt. Contexts arrive best-first and are re-ordered U-shaped."""
    blocks = [f"[Source: {c.label}]\n{c.text}" for c in u_shaped(contexts)]
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


def answer(index: Index, question: str, role: str, show_context: bool = True) -> Answer:
    """Retrieve context for `question` and ask the chat model to answer from it."""
    contexts = retrieve(index, question, role)

    if show_context:
        print("\nRetrieved context (score | access | kind | source):")
        for c in contexts:
            print(f"  {c.score:.3f} | {c.access:8} | {c.kind:5} | {c.label}")

    if not contexts:  # nothing similar enough: don't let the model improvise
        return Answer(NO_ANSWER, [])
    return Answer(ask_llm(build_prompt(contexts, question)), contexts)


def format_sources(contexts: list[Context]) -> str:
    """List the sources of an answer, e.g. for printing under it."""
    return "\n".join(f"  [{i}] {c.label}" for i, c in enumerate(contexts, 1))


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
                result = answer(index, question, args.role, not args.hide_context)
                print("\n" + result.text)
                if result.contexts:
                    print("\nSources:\n" + format_sources(result.contexts))
                print()
    except OllamaError as exc:
        raise SystemExit(f"Error: {exc}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
