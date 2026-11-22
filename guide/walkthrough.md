# Walkthrough: following one question through the system

Role: `public` (a customer). Question: **"How much is the signature drink?"**

## Startup (once)

| Step | Code | What happens |
|---|---|---|
| 1 | `load_chunks()` in [chunking.py](../src/rag_project/chunking.py) | Reads `docs/handbook.md`, splits on blank lines → 11 chunks, each with an `access` tag. |
| 2 | `fingerprint()` in [index.py](../src/rag_project/index.py) | Hashes the chunk texts + model name. |
| 3 | `_load()` | Cache missing → first run, so go on. Cache matches → skip to step 6. |
| 4 | `embed()` in [ollama_client.py](../src/rag_project/ollama_client.py) | Sends chunks to Ollama in batches of 64, gets 11 vectors back. |
| 5 | `normalize()` | Scales each vector to length 1. |
| 6 | `_save()` | Writes `.cache/index.npz`. |

## The question

| Step | Code | What happens |
|---|---|---|
| 7 | `retrieve()` in [retrieval.py](../src/rag_project/retrieval.py) | Embeds the question into one vector. |
| 8 | `top_chunks()`: mask | `ROLE_PERMISSIONS["public"] = {"public"}`. The 6 internal chunks (alarm code, Wi-Fi...) are marked invisible. |
| 9 | `top_chunks()`: score | `index.vectors @ question_vector` gives 11 scores. Hidden ones are set to `-inf`. |
| 10 | `top_chunks()`: rank | Sort, keep the best 5 visible chunks. The Stack Overflow Latte paragraph ranks first. |
| 11 | `answer()` in [rag.py](../src/rag_project/rag.py) | Fills `PROMPT_TEMPLATE` with the chunks and the question. |
| 12 | `ask_llm()` | The chat model replies, e.g. *"$6.50."* |

## Something to try

Ask `What is the alarm code?` as `--role public`:

- Step 8 hides the alarm paragraph, so it can't be retrieved.
- The prompt contains only public chunks, so the model answers *"I don't know"*.
- Note the "Retrieved context" printout: you can see exactly what the model was shown. When an answer is wrong, **always look at the retrieved chunks first.** Most RAG bugs are retrieval bugs, not model bugs.

Then run with `--role internal` and compare.

## Experiments for learning

- Set `TOP_K = 1` in [config.py](../src/rag_project/config.py). Which questions break?
- Swap `CHAT_MODEL` for another Ollama model and compare the style of the answers.
- Add a new tagged paragraph to `docs/handbook.md` and ask about it. Note the cache rebuilding itself.
- Ask a question the documents can't answer, such as "Who won the World Cup?". The prompt rules should produce "I don't know".
