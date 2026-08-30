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
- pytest over the data layer (tests that load or exercise the real
  Corpus_Loader against BEIR SciFact).
- End-to-end tests over the real corpus (tests that run the
  Sweep_Runner against the real BEIR SciFact data rather than an
  in-memory stub corpus).
- `README.md` with the results table and the honest headline.
- `SPEC.md` with design and threats to validity.

GitHub Actions CI has shipped (`.github/workflows/ci.yml`: `push` and
`pull_request`, `ubuntu-latest`, Python 3.13, installs
`requirements.txt`, runs `pytest` only) and is no longer pending work.

Session 1 does, however, include a call-counting test of the
Sweep_Runner orchestration loop itself: a stub `Retriever` implementing
the base protocol, which records its own calls, is run over a small
(no more than 5 documents) in-memory corpus, with no network call and
no model loaded. That test verifies exactly one `build_index` call
and one `retrieve_all` call per retriever, and that every cutoff's
ranked list is a prefix slice of the single deepest-cutoff list. This
is distinct from — and does not pull forward — the data-layer tests
and real-corpus end-to-end tests listed above, which remain session 2.

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
- `tests/` contains passing pytest coverage of the recall@k, nDCG@10,
  and MRR@10 functions, including the pytrec_eval cross-check against
  the same fixtures.
- `tests/` contains a passing call-counting test of the Sweep_Runner
  orchestration loop, using a stub `Retriever` and an in-memory corpus
  of no more than 5 documents, with no network call and no model
  loaded, verifying exactly one `build_index` call and one
  `retrieve_all` call per retriever and that every cutoff's ranked
  list is a prefix slice of the single deepest-cutoff list.

Until every one of these holds, session-2 and session-3 items (see
above) are refused, not just deferred. When refusing, state which
specific condition is unmet.
