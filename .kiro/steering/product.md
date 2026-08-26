---
inclusion: always
---

# Product

## What this is

`rag-retrieval-sweep` is a controlled empirical study, not a product. It
answers one research question: **when does dense retrieval actually beat
a lexical baseline?**

The study sweeps retriever x top-k x chunking strategy over a corpus with
real human relevance judgments (not self-graded ones), and reports
recall@k, nDCG@10, and MRR@10 for every configuration in
`results/sweep.csv`.

- Corpus: SciFact via BEIR (~5k abstracts, ~300 test claims). Free, no
  API key, CPU-runnable. Ships human relevance judgments (qrels).
- Retrievers: BM25 (lexical baseline), `all-MiniLM-L6-v2` (384-dim
  dense), `BAAI/bge-small-en-v1.5`.
- top-k: 1, 5, 10, 20.
- Chunking: whole document, fixed-window with overlap, sentence-window.

## Expected finding — and why it's not a failure

**BM25 is a strong baseline on SciFact and may beat both dense models.**
That is the expected, honest outcome of this sweep, not a bug to chase
or a result to massage until dense retrieval "wins." If BM25 comes out
on top, the repo reports that plainly, with numbers, as the headline
finding. A result showing dense retrieval underperforming a 40-year-old
lexical algorithm on a scientific-claim corpus is a legitimate,
publishable finding about when semantic embeddings help and when they
don't — it is exactly what this project exists to measure.

Do not:
- Tune the sweep, cherry-pick configurations, or add retrievers/tricks
  after the fact to make dense retrieval look better than the numbers
  show.
- Treat "BM25 won" as a signal that something is broken in the dense
  pipeline. Investigate correctness bugs on their own merits (e.g. wrong
  embedding normalization), but a correctly-implemented dense retriever
  losing to BM25 is a valid, expected result.

## Mechanism pass (why, not just what)

Beyond the top-line metrics, the project buckets *why* queries fail:
low vocabulary overlap with the gold passage, queries hinging on a
number or entity name, very short queries, negated claims. The bucket
is a column in `results/sweep.csv` so the failure analysis is
reproducible straight from the artifact, not from ad hoc notes.

## Explicitly out of scope

See `.kiro/steering/scope-guard.md` for the authoritative prohibited
list and the refusal/override rule. Do not duplicate that list here —
it's kept in one place so the two files can't drift apart.
