# rag-retrieval-sweep

[![CI](https://github.com/kianaEB/rag-retrieval-sweep/actions/workflows/ci.yml/badge.svg)](https://github.com/kianaEB/rag-retrieval-sweep/actions/workflows/ci.yml)

A controlled empirical study of when dense retrieval beats a lexical
baseline, on BEIR SciFact. See `SPEC.md` for the full design and
threats to validity.

## Headline finding

The full grid crosses 3 retrievers (`bm25`, `all-MiniLM-L6-v2`,
`bge-small-en-v1.5`) with 3 chunking strategies (`whole_document`,
`fixed_window`, `sentence_window`) — 9 runs. The headline below reports
the **`whole_document`** slice specifically, since that is the slice
that keeps the same document-level granularity BM25's own scoring
uses; the full 9-run comparison, including `fixed_window` and
`sentence_window`, is in the results table.

Under `whole_document` chunking, on BEIR SciFact:

- `all-MiniLM-L6-v2` (dense retrieval) scores **-0.0068** nDCG@10
  relative to BM25 (95% CI: **-0.0449** to **0.0307**;
  Holm-Bonferroni-adjusted p = **1.0000**). That difference is
  **indistinguishable from noise**.
- `bge-small-en-v1.5` (dense retrieval) scores **+0.0681** nDCG@10
  relative to BM25 (95% CI: **0.0292** to **0.1069**;
  Holm-Bonferroni-adjusted p = **0.0042**). That difference is
  **statistically significant** — `bge-small-en-v1.5` beats BM25 on
  this corpus, under this chunking strategy.

The two dense models do not tell the same story: whichever way the
pre-declared primary metric fell for `all-MiniLM-L6-v2`, it fell the
other way, and significantly, for `bge-small-en-v1.5`. Dense retrieval
does not uniformly beat, or lose to, BM25 on this corpus — which dense
model is used changes the answer.

That `all-MiniLM-L6-v2` comparison is itself confounded: **71.02%** of
corpus documents exceed `all-MiniLM-L6-v2`'s 256-token limit, so under
whole-document chunking the dense retriever scored only the first 256
tokens of most documents while BM25 scored each document's full text.
`bge-small-en-v1.5`'s 512-token limit is exceeded by only 8.78% of
documents, which is part of why its whole-document result should not
be read as directly comparable in kind to `all-MiniLM-L6-v2`'s. See
`SPEC.md`'s "Threats to validity" section for the full discussion,
including the query-prefix omission for `bge-small-en-v1.5` and the
6-cell token-length measurement across every chunking strategy.

## Engineering cost

Indexing the corpus with BM25 (`rank_bm25`) took **1.05** seconds;
indexing the same corpus with `all-MiniLM-L6-v2` took **234.67**
seconds (**224.46x** as long as BM25); indexing it with
`bge-small-en-v1.5` took **764.80** seconds (**731.55x** as long as
BM25). `bge-small-en-v1.5` is the only comparison this study found
statistically significant, but it also carries the largest indexing
cost of the three.

## Results

BM25 is the reference row; every dense-retriever value is reported as
a delta against it, per this repository's evaluation-integrity rule.
Only nDCG@10 (the pre-declared primary metric) carries an adjusted
p-value and verdict — every other row's correction does not apply and
is marked `n/a`, matching `results/significance.csv`. Rows are grouped
by chunking strategy; BM25's own absolute values are shown once per
strategy as the reference.

### `whole_document`

Absolute values (`all-MiniLM-L6-v2`: recall@1 **0.4823**, recall@5
**0.7379**, recall@10 **0.7833**, recall@20 **0.8373**, nDCG@10
**0.6451**, MRR@10 **0.6047**. `bge-small-en-v1.5`: recall@1
**0.5787**, recall@5 **0.7653**, recall@10 **0.8452**, recall@20
**0.8753**, nDCG@10 **0.7200**, MRR@10 **0.6845**.) are shown as
deltas against BM25 below, per this repository's evaluation-integrity
rule.

| Metric | BM25 | `all-MiniLM-L6-v2` (Δ) | p (adj.) | Verdict | `bge-small-en-v1.5` (Δ) | p (adj.) | Verdict |
|---|---|---|---|---|---|---|---|
| recall@1 | 0.5075 | -0.0252 | n/a | n/a | 0.0712 | n/a | n/a |
| recall@5 | 0.7284 | 0.0095 | n/a | n/a | 0.0368 | n/a | n/a |
| recall@10 | 0.7740 | 0.0093 | n/a | n/a | 0.0712 | n/a | n/a |
| recall@20 | 0.8323 | 0.0051 | n/a | n/a | 0.0431 | n/a | n/a |
| nDCG@10 | 0.6519 | -0.0068 | 1.0000 | indistinguishable | 0.0681 | 0.0042 | significant |
| MRR@10 | 0.6186 | -0.0139 | n/a | n/a | 0.0659 | n/a | n/a |

### `fixed_window` (Δ against `bm25__whole_document`)

| Metric | `bm25` (Δ) | p (adj.) | Verdict | `all-MiniLM-L6-v2` (Δ) | p (adj.) | Verdict | `bge-small-en-v1.5` (Δ) | p (adj.) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| recall@1 | -0.0133 | n/a | n/a | 0.0104 | n/a | n/a | 0.0590 | n/a | n/a |
| recall@5 | -0.0204 | n/a | n/a | 0.0177 | n/a | n/a | 0.0527 | n/a | n/a |
| recall@10 | -0.0050 | n/a | n/a | 0.0482 | n/a | n/a | 0.0562 | n/a | n/a |
| recall@20 | -0.0158 | n/a | n/a | 0.0217 | n/a | n/a | 0.0481 | n/a | n/a |
| nDCG@10 | -0.0131 | 0.4792 | indistinguishable | 0.0225 | 0.7082 | indistinguishable | 0.0587 | 0.0130 | significant |
| MRR@10 | -0.0151 | n/a | n/a | 0.0139 | n/a | n/a | 0.0589 | n/a | n/a |

### `sentence_window` (Δ against `bm25__whole_document`)

| Metric | `bm25` (Δ) | p (adj.) | Verdict | `all-MiniLM-L6-v2` (Δ) | p (adj.) | Verdict | `bge-small-en-v1.5` (Δ) | p (adj.) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| recall@1 | -0.0467 | n/a | n/a | 0.0008 | n/a | n/a | 0.0623 | n/a | n/a |
| recall@5 | -0.0484 | n/a | n/a | 0.0053 | n/a | n/a | 0.0426 | n/a | n/a |
| recall@10 | -0.0177 | n/a | n/a | 0.0257 | n/a | n/a | 0.0689 | n/a | n/a |
| recall@20 | -0.0246 | n/a | n/a | 0.0274 | n/a | n/a | 0.0474 | n/a | n/a |
| nDCG@10 | -0.0417 | 0.0032 | significant | 0.0108 | 1.0000 | indistinguishable | 0.0624 | 0.0078 | significant |
| MRR@10 | -0.0476 | n/a | n/a | 0.0061 | n/a | n/a | 0.0582 | n/a | n/a |

Every `bm25`/`fixed_window` and `bm25`/`sentence_window` row above is a
comparison of BM25 under a different chunking strategy against the
same `bm25__whole_document` reference row — the deltas measure
chunking's own effect on BM25, not a different retriever.

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

The combined indexing and retrieval time for the 3 `whole_document`
runs (the sum of `index_time` and `query_latency` for `bm25`,
`all-MiniLM-L6-v2`, and `bge-small-en-v1.5` from `results/sweep.csv`)
is **1014.07** seconds. This is not the total wall-clock runtime of
the sweep — the full 9-run grid (3 retrievers x 3 chunking strategies)
takes substantially longer, since `fixed_window` and `sentence_window`
each produce several times as many chunks as there are documents, and
the actual runtime also includes corpus loading, model loading, metric
computation, and report writing, plus, on a first invocation, a
one-time download.

The first invocation of the sweep entry point downloads the BEIR
SciFact corpus and the `all-MiniLM-L6-v2` and `bge-small-en-v1.5` model
weights once, to a path under `data/`. A later invocation reuses those
cached copies and makes no network call. The significance-analysis
entry point makes no network call and downloads nothing, on either its
first or any later invocation — it only reads `results/per_query.csv`.

Installed package versions for this run: `beir==2.2.0`,
`rank_bm25==0.2.2`, `sentence-transformers==6.0.0`, `torch==2.13.0`,
`numpy==2.5.2`.

## What this does not claim

- **Scope.** This finding is scoped to one corpus, BEIR SciFact
  (scientific claims). It does not generalize to another domain or
  corpus.
- **Chunking.** All 3 declared chunking strategies (`whole_document`,
  `fixed_window`, `sentence_window`) are evaluated above, but only
  with the fixed numeric parameters declared once in
  `configs/sweep.yaml` before any result existed. This finding does
  not generalize to a different window size, stride, sentence count,
  or token cap.
- **Retrievers.** All 3 retrievers declared in this study's grid
  (`bm25`, `all-MiniLM-L6-v2`, `bge-small-en-v1.5`) are compared above.
  This finding does not generalize to another retriever, model size,
  or model family.
- **Production guidance.** This is not a recommendation for or against
  dense retrieval in a production retrieval-augmented-generation
  system.
