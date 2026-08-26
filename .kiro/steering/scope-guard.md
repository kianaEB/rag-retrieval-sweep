---
inclusion: always
---

# Scope guard

## Prohibited — explicitly out of scope

From `docs/PROJECT_BRIEF.md`, plus items that change the research
question rather than answering it:

- No chat UI.
- No agent framework.
- No fine-tuning.
- No Kubernetes.
- No cloud vector database.
- No serving endpoint.
- No hybrid retrieval / score fusion (RRF or otherwise) combining
  lexical and dense results. Legitimate future work, not this repo — it
  answers "does combining help" instead of "when does dense beat
  lexical."
- No cross-encoder or LLM reranking of retrieved results. Legitimate
  future work, not this repo — it measures reranking, not the
  retrievers being swept.
- No retrievers beyond the three named in the brief (BM25,
  `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`). Legitimate future work,
  not this repo — the grid is fixed so the comparison stays controlled.
- No query expansion, rewriting, or generated pseudo-queries.
  Legitimate future work, not this repo — it changes what's being
  retrieved against, not how retrieval performs.
- No approximate nearest neighbour indexes (FAISS, HNSW, ScaNN).
  Legitimate future work, not this repo — brute-force exact search over
  ~5k documents is fast enough, and ANN adds a recall approximation
  error that contaminates the metric being measured.

Scope creep into a chatbot is the failure mode for this repo.

## If asked for one of these

If the user asks for anything on the prohibited list above, refuse and
name the specific item that's prohibited.

The only way to bring an item into scope is for the user to edit
`docs/PROJECT_BRIEF.md` so the item is no longer on the out-of-scope
list. Verbal confirmation in chat ("yes, build it anyway") is not
sufficient and must not be accepted — the brief is the source of truth,
not the chat turn.

## Not yet — belongs to a later session

These are real parts of the project, but they are not session-1 work.
Do not build them while session 1 is in progress; flag it if a request
would pull one of these in early.

**Session 2:**
- Full grid: all retrievers x all top-k values x all chunking strategies
  (session 1 is BM25 + one dense retriever, whole-document chunking
  only).
- Failure bucketing.
- `ANALYSIS.md`.
- pytest over the metric functions and the data layer.
- GitHub Actions CI.
- `README.md` with the results table and the honest headline.
- `SPEC.md` with design and threats to validity.

**Session 3 (stretch):**
- Groundedness gate over generated answers.
- Quarantine rate.
- Hand-checked sample.

## Session-1 release gate

Session 1 is complete only when ALL of the following hold:

- `results/sweep.csv` exists with 8 rows (2 retrievers x 4 top-k x
  whole-document chunking), each row carrying recall@k, nDCG@10,
  MRR@10, index time, and query latency.
- One entry point runs the sweep end to end from a config file in
  `configs/` on a clean checkout.
- `requirements.txt` lists pinned exact versions.

Until every one of these holds, session-2 and session-3 items (see
above) are refused, not just deferred. When refusing, state which
specific condition is unmet.
