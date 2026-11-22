# Core concepts

## 1. Why not just paste all documents into the prompt?

For a 10-paragraph handbook you could. But real knowledge bases are thousands of pages, and models have a limited context window. A long prompt is also slower, costlier and easier to get lost in. RAG pastes only the **few paragraphs that matter**.

## 2. Embeddings: meaning as numbers

An embedding model turns text into a **vector**, a list of numbers (768 for `nomic-embed-text`). You can picture each text as an arrow in space. **Similar meaning means arrows pointing in a similar direction.**

A toy version in 2 dimensions (real ones have hundreds):

```
            ▲ "drinks / food" axis
            │
   "latte   │  ● "How much is a latte?"
    costs   │ ↗
    $6.50"● │
            │
            │                 ● "dogs allowed on patio"
            └──────────────────────────────────►  "animals / places" axis
```

The question and the latte paragraph point the same way, so they match, even though the question never says "$6.50" or "signature drink". This is why embeddings beat plain keyword search.

## 3. Cosine similarity: "how close are two arrows?"

Measure the **angle** between two vectors:

| Angle | Cosine similarity | Meaning |
|---|---|---|
| 0° (same direction) | 1.0 | very similar |
| 90° | 0.0 | unrelated |
| 180° | -1.0 | opposite |

Formula: `cos = (a · b) / (|a| × |b|)`.

**Optimization we use:** if both vectors are scaled to length 1 (*normalized*), `|a| = |b| = 1` and the formula is only `a · b`. We normalize chunks once at index time ([index.py](../src/rag_project/index.py)), so scoring all chunks is one matrix multiplication: `index.vectors @ question_vector`.

## 4. Chunking

We cut documents into pieces because each piece gets **one** vector:

- **Too big:** the vector is a blurry average of many topics, and the prompt fills up.
- **Too small:** a sentence loses the context that gives it meaning ("It costs $6.50": what does?).

This project uses one paragraph per chunk. That's simple and works well for a handbook. Other strategies are fixed-size windows with overlap, or splitting by Markdown headings (see [next-steps.md](next-steps.md)).

## 5. Top-k retrieval

Score every chunk, sort, and keep the best `k` (default 5, `config.TOP_K`). Too small a `k` can miss the answer, and too large a `k` adds noise that distracts the model.

## 6. The prompt: where "augmented" happens

```
Answer the question using ONLY the context below. Rules...

Context:
<chunk 1>
<chunk 2>
...

Question: <the user's question>
```

The instruction "use ONLY the context" plus the "say *I don't know*" rule reduces **hallucination** (the model inventing plausible but false facts).

## 7. Access control: why it's in code, not in the prompt

The original script asked the model to *guess* whether the user was a customer. That's fragile:

- A user can simply claim "I'm an employee".
- A model can be talked out of its instructions (*prompt injection*).

The rule we follow: **never give the model information the user shouldn't see.** Each chunk carries an `access` tag, and [retrieval.py](../src/rag_project/retrieval.py) removes forbidden chunks *before* ranking.

```
all chunks ──► [ role = public ] ──► only public chunks ──► rank ──► prompt
                  internal chunks are dropped here, so the model never sees them
```

> Caveat: `--role` is a command-line flag a user can change, which is fine for learning. A real app would take the role from a login system.

## 8. Caching with a fingerprint

Embedding is the slow part. We save the vectors to `.cache/index.npz` together with a **fingerprint** (a SHA-256 hash of the chunk texts + embedding model name). On startup:

```
compute fingerprint of current docs
   ├─ matches the one in the cache? ──► load vectors instantly
   └─ different / no cache?         ──► re-embed, save new cache
```

Edit a document, or change `EMBED_MODEL`, and the fingerprint changes, so you never get stale answers. Delete `.cache/` any time. It's safe.

## 9. Batching and error handling

- **Batching:** the embed call sends 64 chunks per request instead of everything at once (`config.EMBED_BATCH_SIZE`).
- **Errors:** every request has a timeout and checks the HTTP status. If Ollama isn't running or a model is missing, you get a message saying what to do instead of a cryptic `KeyError`.
