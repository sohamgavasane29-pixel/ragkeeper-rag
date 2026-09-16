# RAGKeeper

A freshness-aware, retrieval-tuned RAG chatbot over the LangChain documentation.
Docs are cloned, chunked, embedded locally, and stored in Qdrant; answers are generated
via Groq with source citations. Phases 1–5 are complete, plus scheduled auto-sync and a
containerized deployment.

## Architecture

```
LangChain docs repo --(git sync)--> chunk (MD headers) --> contextual embed --> Qdrant
                                                                                    |
question --> hybrid retrieval (dense + BM25, RRF) --> cross-encoder rerank --> top-k --> Groq LLM --> answer + sources
```

## Phase 1 — Baseline RAG
- Clone/sync the LangChain docs repo, chunk by Markdown headers, embed locally
  (`sentence-transformers/all-MiniLM-L6-v2`), upsert into Qdrant.
- Retrieve + generate via Groq, with source file paths returned alongside each answer.
- Eval harness (`python main.py eval`) scores a fixed 15-question golden set on retrieval
  (precision/recall/MRR/nDCG@k), generation (LLM-judge: faithfulness, answer relevance,
  correctness), and end-to-end latency/tokens/cost.

## Phase 2 — Freshness-aware incremental re-indexing
- A local SQLite state store (`ragkeeper/state.py`) tracks each file's content hash and
  chunk count across runs.
- `ingest` diffs discovered files against that state: only added/changed files are
  re-chunked and re-embedded; deleted files have their vectors removed; unchanged files
  are skipped entirely.
- Every run is recorded in a `sync_runs` history table (added/updated/deleted/unchanged
  counts, duration, status). `python main.py status` reports the latest run plus the live
  Qdrant point count.
- `ingest --recreate` drops and rebuilds the collection and state from scratch — required
  whenever the embedding model changes.

## Phase 3 — Retrieval quality
Implemented and evaluated in order, each step kept only if it beat the running baseline
on the golden set:

1. **Contextual chunk embedding** — each chunk is embedded with a
   `doc title > heading > subheading` breadcrumb prefixed to it; the raw chunk text is kept
   in metadata for citation/display (`splitter.build_contextual_text`).
2. **BM25 hybrid retrieval** — a keyword index (`rank_bm25`) runs alongside dense vector
   search; results are merged via weighted Reciprocal Rank Fusion
   (`dense_weight=1.0`, `bm25_weight=0.5`).
3. **Cross-encoder reranking** — hybrid retrieval pulls a wider candidate pool (20 docs),
   then `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks down to the final top-k.
4. **Embedding model A/B test** — `BAAI/bge-small-en-v1.5` was tried and rejected (better
   retrieval metrics, worse generation correctness); `all-MiniLM-L6-v2` remains in use.
5. **Query expansion / confidence-based fallback** — implemented (`retrieval.make_llm_query_expander`,
   a reranker-score threshold in `RagKeeperChat.answer`) but **disabled by default**
   (`ENABLE_QUERY_EXPANSION=false`); the golden-set eval didn't show it was necessary.

**Latest confirmed eval** (Groq + MiniLM + contextual chunks + hybrid RRF + rerank, 15 questions):

| Retrieval | | Generation (1–5) | | End-to-end |  |
|---|---|---|---|---|---|
| precision@k | 0.383 | faithfulness | 4.53 | avg latency | 8.6s |
| recall@k | 0.867 | answer relevance | 4.67 | total tokens | 18,774 |
| MRR | 0.722 | correctness | 4.47 | total cost | $0.00574 |
| nDCG@k | 1.123 | | | | |



## Deployment — Docker / Podman
`Dockerfile` + `docker-compose.yml` run the whole stack as three containers, all driven by
the same `.env`:

- `qdrant` — vector store (unchanged from local dev), with a health check.
- `app` — the Streamlit dashboard (`streamlit run dashboard/app.py`), port `8501`.
- `scheduler` — `python main.py schedule --interval-hours 24`, the automation above running
  continuously.

`app` and `scheduler` share the same image, share a `./data` bind mount (state DB + repo
clone), and both reach Qdrant via the container network name (`QDRANT_URL=http://qdrant:6333`,
overridden in compose — no need to edit `.env` for this).

```
docker compose build
docker compose up -d
```

Podman works as a drop-in replacement for the same files: `podman compose build` /
`podman compose up -d`. `torch` is pinned to the CPU-only wheel
(`--index-url https://download.pytorch.org/whl/cpu`) before `requirements.txt` is installed —
`sentence-transformers` otherwise pulls multi-GB CUDA packages that aren't needed for
CPU-only embedding/reranking and can time out the build.

**Windows/WSL2 caveat**: on some machines, WSL2's `localhost` port-forwarding relay
(`wslrelay.exe`) gets stuck and never forwards `http://localhost:8501` (even
`podman machine stop/start` or a full `wsl --shutdown` may not fix it). If the dashboard
won't load, open the WSL VM's real IP instead — the same address already used for
`QDRANT_URL` locally (e.g. `http://172.27.122.123:8501`). Get the current IP with:
```
wsl -d podman-machine-default -- ip addr show eth0
```
It can change after the VM restarts.

**Not yet built**: TLS/reverse proxy for a real public deployment (needs a domain), and
auto-restarting `app` after each `scheduler` sync to refresh its BM25 index.

## Setup

1. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Embeddings run locally, so no
   embedding API key is needed.
2. Install dependencies (into your existing `venv`):
   ```
   pip install -r requirements.txt
   ```
3. Start Qdrant:
   ```
   docker compose up -d
   ```
   Dashboard: open `<QDRANT_URL from your .env>/dashboard` in a browser.

   `QDRANT_URL` must point at wherever Qdrant is actually reachable from this machine —
   it is **not always** `http://localhost:6333`. If Docker is running via WSL2 (common on
   Windows), `localhost` from Windows may not resolve to the container; use the WSL VM's
   IP instead (`wsl hostname -I`) or the address you're already using in `.env`, e.g.
   `http://172.27.122.123:6333/dashboard#/collections`. That IP is assigned by WSL2 and can change after a
   reboot — if the dashboard stops loading, re-check `wsl hostname -I` and update
   `QDRANT_URL` accordingly.

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for generation + LLM-judge eval |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model used for chat and eval |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model — changing this requires `ingest --recreate` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint — set to your actual instance address |
| `QDRANT_COLLECTION` | `langchain_docs` | Qdrant collection name |
| `LANGCHAIN_REPO_URL` | LangChain docs repo | Source repo to sync |
| `LANGCHAIN_REPO_PATH` | `./data/langchain_repo` | Local clone path |
| `DOCS_ROOT_SUBDIR` | `src/oss` | Subdirectory within the repo to index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Markdown chunking parameters |
| `TOP_K` | `4` | Final number of chunks passed to the LLM |
| `ENABLE_QUERY_EXPANSION` | `false` | LLM-generated query paraphrases merged into hybrid retrieval |
| `CONFIDENCE_THRESHOLD` | `0.0` | Minimum top reranker score before answering; below it, a fallback "I don't know" response is returned instead of calling the LLM |

## Usage

Ingest (sync repo + chunk + embed + upsert, incrementally):
```
python main.py ingest
```
Use `--recreate` to drop and fully rebuild the collection (required after an embedding model change).

Index health / last sync summary:
```
python main.py status
```

Chat:
```
python main.py chat
```
Ask questions like "How do I create an agent in LangChain?" — answers are grounded in the
retrieved doc chunks, with source file paths printed below each answer.

Evaluate against the golden set:
```
python main.py eval [--k N]
```
Writes a full report (per-question + aggregate) to `eval_results/eval_<timestamp>.json`.

Run the MCP server (`search_docs`, `get_index_health`):
```
python main.py mcp
```

Launch the dashboard:
```
python main.py dashboard
```

Run the freshness-aware auto-sync loop:
```
python main.py schedule [--interval-hours N]
```

## Verification

- Check the ingestion log for added/updated/deleted/unchanged file counts and the final
  Qdrant point count.
- Re-run `python main.py ingest` with no doc changes and confirm it reports all files
  `unchanged` (no re-embedding, fast run).
- Run `python main.py status` to confirm the state DB matches the live collection.
- Run `python main.py eval` and compare against the latest numbers above before accepting
  any further retrieval change.
