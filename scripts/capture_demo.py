"""Run the real pipeline on a few questions and save the answers for scripts/render_demo.py."""

import json
import time

from rag_project.index import build_index
from rag_project.rag import answer, format_sources

QUESTIONS = [
    ("public", "What is the server room access method?"),
    ("internal", "What is the server room access method?"),
    ("public", "What is the pass mark for an exam?"),
]

index = build_index()
out = []
for role, question in QUESTIONS:
    start = time.time()
    result = answer(index, question, role, show_context=False)
    out.append({"role": role, "q": question, "a": result.text.strip(), "sources": format_sources(result.contexts[:3]),
                "seconds": round(time.time() - start, 1)})
    print(out[-1], flush=True)
json.dump(out, open("assets/demo_capture.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
