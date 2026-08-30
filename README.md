# rag-retrieval-sweep

[![CI](https://github.com/kianaEB/rag-retrieval-sweep/actions/workflows/ci.yml/badge.svg)](https://github.com/kianaEB/rag-retrieval-sweep/actions/workflows/ci.yml)

A controlled empirical study of when dense retrieval beats a lexical
baseline, on BEIR SciFact. See `SPEC.md` for the full design and
threats to validity.

## Headline finding

On BEIR SciFact, `all-MiniLM-L6-v2` (dense retrieval) scores **-0.0068**
nDCG@10 relative to BM25 (95% CI: **-0.0451** to **0.0318**;
Holm-Bonferroni-adjusted p = **0.7275**). That difference is
**indistinguishable from noise** — this is not a win for BM25 and not
a win for the dense retriever. Whichever way the pre-declared primary
metric fell, this is the headline: dense retrieval does not
demonstrably beat, or lose to, BM25 on this corpus under whole-document
chunking.

That comparison is itself confounded: **71.02%** of corpus documents
exceed `all-MiniLM-L6-v2`'s 256-token limit, so under whole-document
chunking the dense retriever scored only the first 256 tokens of most
documents while BM25 scored each document's full text. See `SPEC.md`'s
"Threats to validity" section for the full discussion.

## Engineering cost

Indexing the corpus with BM25 (`rank_bm25`) took **0.66** seconds;
indexing the same corpus with `all-MiniLM-L6-v2` took **182.79**
seconds — dense indexing took roughly **275.21x** as long as BM25
indexing, for a comparison that this study cannot distinguish from
noise (nDCG@10 verdict: indistinguishable). On this corpus, the
accuracy case for paying that indexing cost is not established.

## Results

BM25 is the reference row; every `all-MiniLM-L6-v2` value is reported
as a delta against it, per this repository's evaluation-integrity
rule. Only nDCG@10 (the pre-declared primary metric) carries an
adjusted p-value and verdict — every other row's correction does not
apply and is marked `n/a`, matching `results/significance.csv`.

| Metric | BM25 | `all-MiniLM-L6-v2` (Δ) | p (adj.) | Verdict |
|---|---|---|---|---|
| recall@1 | 0.5075 | -0.0252 | n/a | n/a |
| recall@5 | 0.7284 | 0.0095 | n/a | n/a |
| recall@10 | 0.7740 | 0.0093 | n/a | n/a |
| recall@20 | 0.8323 | 0.0051 | n/a | n/a |
| nDCG@10 | 0.6519 | -0.0068 | 0.7275 | indistinguishable |
| MRR@10 | 0.6186 | -0.0139 | n/a | n/a |

## Corpus

The sweep runs over BEIR SciFact: **5183** corpus documents, **300**
test queries, and **339** judged query-document pairs, all read from
the corpus loader's own recorded output at run time.

## Reproducing this sweep

Run the sweep entry point:

```
python -m src.sweep_runner --config configs/sweep.yaml
```

Then run the significance analysis:

```
python -m src.significance --config configs/significance.yaml
```

The combined indexing and retrieval time (the sum of both runs'
`index_time` and `query_latency` from `results/sweep.csv`) is
**194.14** seconds. This is not the total wall-clock runtime of the
sweep — the actual runtime is longer, because it also includes corpus
loading, model loading, metric computation, and report writing, plus,
on a first invocation, a one-time download.

The first invocation of the sweep entry point downloads the BEIR
SciFact corpus and the `all-MiniLM-L6-v2` model weights once, to a
path under `data/`. A later invocation reuses those cached copies and
makes no network call. The significance-analysis entry point makes no
network call and downloads nothing, on either its first or any later
invocation — it only reads `results/per_query.csv`.

Installed package versions for this run: `beir==2.2.0`,
`rank_bm25==0.2.2`, `sentence-transformers==6.0.0`, `torch==2.13.0`,
`numpy==2.5.2`.

## What this does not claim

- **Scope.** This finding is scoped to one corpus, BEIR SciFact
  (scientific claims). It does not generalize to another domain or
  corpus.
- **Chunking.** The only chunking strategy evaluated is
  `whole_document`. This finding does not generalize to fixed-window
  or sentence-window chunking.
- **Retrievers.** The only retrievers compared are `bm25` and
  `all-MiniLM-L6-v2`. This finding does not generalize to another
  retriever, model size, or model family.
- **Production guidance.** This is not a recommendation for or against
  dense retrieval in a production retrieval-augmented-generation
  system.
