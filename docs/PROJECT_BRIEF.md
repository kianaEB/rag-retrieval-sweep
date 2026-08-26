# rag-retrieval-sweep — project brief

A controlled study of when dense retrieval actually beats a lexical
baseline: retriever x top-k x chunking, swept over a corpus with real
human relevance judgments, scored on recall@k / nDCG / MRR, with the
failures bucketed and explained.

## Corpus
SciFact via BEIR (~5k abstracts, ~300 test claims). Ships human relevance
judgments, so recall@k and nDCG are honest numbers rather than
self-graded ones. Free, no API, CPU-runnable. The loading code reports
the exact counts it loads; counts are never hard-coded.

## The grid
Retrievers: BM25 (lexical baseline — the most important row in the repo),
all-MiniLM-L6-v2 (384-dim dense), BAAI/bge-small-en-v1.5.
top-k: 1, 5, 10, 20.
Chunking: whole document, fixed-window with overlap, sentence-window.

## Metrics
recall@k, nDCG@10, MRR@10, plus wall-clock index time and query latency
per run. One row per configuration in results/sweep.csv.

## Expected finding
BM25 is a strong baseline on SciFact and may beat both dense models.
That is the finding, not a failure. It gets reported plainly.

## Mechanism pass
Not "dense scored 0.61" but which queries failed and what they share.
Buckets: low vocabulary overlap with the gold passage; queries hinging on
a number or entity name; very short queries; negated claims. The bucket
is a column in results/sweep.csv so the analysis is reproducible from the
artifact.

## Explicitly OUT of scope
No chat UI. No agent framework. No fine-tuning. No Kubernetes. No cloud
vector database. No serving endpoint. Scope creep into a chatbot is the
failure mode for this repo.

No hybrid retrieval / score fusion (RRF or otherwise) combining lexical
and dense results — legitimate future work, not this repo, because it
answers "does combining help" instead of "when does dense beat lexical."

No cross-encoder or LLM reranking of retrieved results — legitimate
future work, not this repo, because it measures reranking, not the
retrievers being swept.

No retrievers beyond the three named above (BM25, all-MiniLM-L6-v2,
BAAI/bge-small-en-v1.5) — legitimate future work, not this repo, because
the grid is fixed so the comparison stays controlled.

No query expansion, rewriting, or generated pseudo-queries — legitimate
future work, not this repo, because it changes what's being retrieved
against, not how retrieval performs.

No approximate nearest neighbour indexes (FAISS, HNSW, ScaNN) —
legitimate future work, not this repo, because brute-force exact search
over ~5k documents is fast enough, and ANN adds a recall approximation
error that contaminates the metric being measured.

## Sessions
1. Load SciFact + qrels; BM25 and one dense retriever; whole-document
   chunking only; recall@k / nDCG / MRR; results/sweep.csv; seeded;
   requirements.txt; one entry point. Config-driven grid from day one.
2. Full grid; failure bucketing; ANALYSIS.md; pytest over the metric
   functions and the data layer; GitHub Actions CI; README.md with the
   results table and the honest headline; SPEC.md with design and threats
   to validity.
3. Stretch: groundedness gate over generated answers, quarantine rate,
   hand-checked sample.

## Definition of done
README.md, SPEC.md, ANALYSIS.md, tests/ and a passing CI workflow.
