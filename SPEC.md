# SPEC

Design summary and threats to validity for `rag-retrieval-sweep`. This
is a plain-language design document for human readers; it is unrelated
to the `.kiro/specs/` folder mechanism used to build this repository.

## Design summary

### Sweep grid

The sweep is config-driven (`configs/sweep.yaml`), and the grid
actually applied to the reported run — read from
`results/run_config.json`'s `sweep_config` object — is: two
retrievers, `bm25` (`k1 = 1.50`, `b = 0.75`) and `all-MiniLM-L6-v2`;
evaluation cutoffs `1, 5, 10, 20`; and a single chunking strategy,
`whole_document`.

### Index once, retrieve once, slice four ways

Each retriever is indexed exactly once over the whole corpus, and
queried exactly once, at the deepest declared cutoff (`20`). Every
cutoff's recall@k value is computed by slicing that single returned
ranked list to the first `k` documents, rather than by issuing a
separate retrieval run per cutoff. nDCG@10 and MRR@10 are each
computed once per retriever, at a fixed cutoff of 10, and copied
unchanged into every one of that retriever's rows in
`results/sweep.csv`.

### Relevance ground truth

The BEIR SciFact qrels (human relevance judgments) are the sole source
of relevance ground truth used to compute every reported metric — no
model judgment and no manual override changes a qrels-derived
relevance determination anywhere in this pipeline.

### Primary and secondary metrics

nDCG@10 was designated the primary metric before any sweep result
existed. recall@k (for k = 1, 5, 10, 20) and MRR@10 are reported for
every configuration as secondary metrics.

### Statistical scheme

Every non-BM25 run is compared against the BM25 reference run with a
paired-bootstrap confidence interval and a paired two-sided permutation
p-value, over `resample_count = 10000` bootstrap resamples and
`permutation_count = 10000` permutations, drawn from a single
`numpy` random generator seeded with `bootstrap_seed = 20240`. The
nDCG@10 comparison family's raw p-values are then adjusted with a
Holm-Bonferroni step-down correction before the primary-metric verdict
is decided.

## nDCG@10 convention

nDCG@10 is computed, for each test query, as DCG@10 divided by IDCG@10,
defined as 0 when IDCG@10 is 0, where:

- **DCG@10** is the sum, over ranks i = 1 to 10 of the top 10 documents
  of the ranked list, of `rel_i / log2(i + 1)`, with `rel_i` equal to
  the graded relevance score recorded in the qrels for the document at
  rank i, or 0 if that document is unjudged.
- **IDCG@10** is the same sum computed over that query's qrels-judged
  relevant documents, sorted by descending relevance score and
  truncated to the first 10 documents.

The nDCG@10 value reported for a run is the arithmetic mean of the
per-query nDCG@10 value across all test queries that have at least one
qrels-judged relevant document.

This convention (the `log2(i + 1)` discount, graded relevance taken
directly from the qrels, and the 0-when-IDCG@10-is-0 rule) was fixed
before any sweep result existed and has not been altered after seeing
results.

## Threats to validity

### Sparse qrels

BEIR SciFact judges an average of **1.13** relevant documents per
query (339 judged query-document pairs over 300 test queries). A
document that is actually relevant to a query but was never judged is
scored as a miss regardless of whether a retriever surfaced it — this
penalizes a retriever that surfaces documents unlike the ones that
happened to get annotated, not necessarily a retriever that is
performing worse.

### BM25's sensitivity to preprocessing

BM25's score is sensitive to its tokenizer, stopword list, stemming
choice, and case handling, none of which are inherent to the BM25
algorithm itself. For this run, those choices were fixed once, before
any result existed, and never adjusted after seeing scores: tokenizer
`regex_word`, stopwords `none` (no stopword list applied), stemming
`none` (no stemming applied), lowercasing enabled, `k1 = 1.50`, and
`b = 0.75`. A different, equally defensible set of preprocessing
choices could move BM25's measured score in either direction; the
sensitivity itself, not just this run's fixed choices, is a threat to
validity worth naming.

### Single-corpus generalization

Every number in this document and in `README.md` describes BEIR
SciFact — a corpus of scientific claims — only. None of it is evidence
about how these retrievers would compare on another domain or corpus.

### Statistical power

The nDCG@10 comparison's 95% confidence interval has a half-width of
**0.0384**. A true difference between the two runs smaller than that
half-width could exist without this study's paired bootstrap being
able to detect it at this sample size — the "indistinguishable"
verdict reported in `README.md` and above reflects an inability to
distinguish the comparison from noise at this study's statistical
power, not proof that no difference exists between the two runs.

### BM25 query latency is an implementation artifact, not a property of lexical retrieval

The measured BM25 query latency in this run was **7.50** seconds
across all 300 test queries. That figure is a property of `rank_bm25`,
a pure-Python scoring implementation, not an inherent property of
lexical retrieval — a compiled or indexed lexical search engine (e.g.
Lucene-based) would be expected to score substantially faster. This
measurement should not be generalized to "lexical retrieval is slow."

### Token-length truncation under whole-document chunking

Measured directly against the cached `all-MiniLM-L6-v2` tokenizer over
every document in the loaded corpus (`results/token_length_report.json`):
**71.02%** of corpus documents exceed the model's 256-token maximum
sequence length under whole-document chunking. Because that fraction
is not a small, negligible one, this is stated as a concrete confound
in the headline comparison, not a minor caveat: the BM25 run
scores each document's full text, while the `all-MiniLM-L6-v2` run
scores only the first 256 tokens of that same document. The
"indistinguishable" nDCG@10 verdict reported above and in `README.md`
should be read with this asymmetry in mind — it is a confound in the
headline comparison, not evidence about either retriever's underlying
ranking quality.
