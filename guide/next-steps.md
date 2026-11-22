# Ideas to extend the project

Roughly in order of difficulty.

1. **Show sources.** Chunks already store `source`. Print "(from handbook.md)" after each answer.
2. **Score threshold.** If the best score is below, say, 0.4, skip the model and answer "I don't know". It's cheaper and safer. Find a good threshold by experimenting.
3. **Embedding prefixes.** `nomic-embed-text` works better when texts start with `search_document: ` and questions with `search_query: `. Add them in `index.py` and `retrieval.py`, then compare results.
4. **Better chunking.** Split by Markdown headings, or use fixed windows of about 200 words with overlap. Compare retrieval quality.
5. **Per-file access.** Allow a front-matter line at the top of a file that sets the default tag for the whole file.
6. **Evaluation.** Write 20 questions with known answers and measure how often the right chunk is in the top 5. That's the standard way to tell whether a change helped.
7. **Real authentication.** Replace `--role` with a login, so the role comes from a trusted source.
8. **Vector database.** Past a few thousand chunks, switch from NumPy to something like Chroma, FAISS or sqlite-vec.
9. **Conversation memory.** Include earlier turns in the prompt so follow-up questions ("and on weekends?") work.
