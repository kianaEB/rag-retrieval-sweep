# ANALYSIS

Mechanism and failure-bucket analysis for `rag-retrieval-sweep`.

## 1. What this document is

This document is the mechanism / failure-bucket analysis named in
`.kiro/steering/structure.md`'s definition of done. It runs no new
retrieval, computes no new p-value, and re-derives no metric already
committed elsewhere in this repository. It introduces exactly two new
artifacts, both produced by one entry point,
`python -m src.failure_buckets`:

- `results/failure_buckets.csv` — one row per (`run_id`, `query_id`)
  pair (2700 rows), carrying that pair's failure bucket plus six
  per-query token-length covariate columns.
- `results/failure_bucket_counts.csv` — one row per (`run_id`, bucket)
  pair (84 rows: 36 per-run + 48 per-contrast), carrying that bucket's
  count and fraction.

The per-query failure-bucket label lives in `results/failure_buckets.csv`
rather than as a column of `results/sweep.csv`, because `results/sweep.csv`
is keyed by (`run_id`, `k`) and has no per-query dimension — there is no
row in it that a per-query label could attach to without collapsing 300
labels into one cell or multiplying the file by the query count.

## 2. Scope of what this study can and cannot infer

The complete set of inferential results this study supports is the
Pre_Declared_Family: the 8 nDCG@10 comparisons against the reference
run `bm25__whole_document`, under Holm-Bonferroni correction, recorded
in `results/significance.csv` (`is_primary = True`, `metric =
ndcg_at_10`). nDCG@10 was designated the single primary metric before
any sweep result existed; recall@k and MRR@10 are secondary.

Every bucket count, every bucket fraction, every pairwise disagreement
figure, and every covariate description in this document is a
**descriptive contrast**: it is outside the Pre_Declared_Family and
carries no inferential claim. None of them is a hypothesis test, a
p-value, a confidence interval, or evidence of a distributional
difference beyond the 300 queries measured. Where this document
proposes a mechanism, it says so explicitly and grounds it in a named
bucket or covariate figure; where it does not, it says plainly that no
mechanism was identified.

## 3. What the grid showed

`bm25__whole_document` is the reference row. Every other run's nDCG@10
result is reported as a delta (`mean_diff`) against it, alongside the
Holm-Bonferroni-adjusted p-value (`p_value_adjusted`) and `verdict`,
all read from that run's row of `results/significance.csv`.

| Run_Id | mean_diff (nDCG@10) | p (adj.) | Verdict |
|---|---|---|---|
| `bm25__whole_document` (reference) | — | — | — |
| `bm25__fixed_window` | -0.0131 | 0.4792 | indistinguishable |
| `bm25__sentence_window` | -0.0417 | 0.0032 | significant |
| `all-MiniLM-L6-v2__whole_document` | -0.0068 | 1.0000 | indistinguishable |
| `all-MiniLM-L6-v2__fixed_window` | +0.0225 | 0.7082 | indistinguishable |
| `all-MiniLM-L6-v2__sentence_window` | +0.0108 | 1.0000 | indistinguishable |
| `bge-small-en-v1.5__whole_document` | +0.0681 | 0.0042 | significant |
| `bge-small-en-v1.5__fixed_window` | +0.0587 | 0.0130 | significant |
| `bge-small-en-v1.5__sentence_window` | +0.0624 | 0.0078 | significant |

Every verdict above is reported exactly as `results/significance.csv`
records it — none is omitted, softened, or relegated. Three results
stand out and are the subject of Sections 5–7: `bge-small-en-v1.5`
beats the reference row, significantly, under all three chunking
strategies (§5); `bm25__sentence_window` falls below the reference row,
significantly (§6); and four comparisons — the three `all-MiniLM-L6-v2`
runs and `bm25__fixed_window` — are indistinguishable from noise (§7).

## 4. The failure-bucket taxonomy

`src/failure_buckets.py`'s Bucket_Assignment_Stage reads
`results/per_query.csv` alone — no corpus, no Qrels, no tokenizer, no
model — and assigns exactly one of four failure buckets to every
(`run_id`, `query_id`) pair, by evaluating these predicates in order and
taking the first that holds:

1. **`total_miss`** — `recall_at_20` is exactly 0.
2. **`mis_ranked`** — `recall_at_20` is strictly greater than 0 and
   `recall_at_1` is exactly 0.
3. **`partial_recall`** — `num_judged_relevant` is strictly greater
   than 1 and `recall_at_20` is strictly greater than 0 and strictly
   less than 1.
4. **`full_success`** — every pair not assigned by predicates 1–3.

This taxonomy was fixed before any label was assigned; it is declared
as constants in `src/failure_buckets.py`, never read from a config
file, a CLI argument, or an environment variable.

`src/failure_buckets.py` also assigns exactly one of four contrast
buckets to every (Pair_Contrast, `query_id`) combination whose
`query_id` is shared by both runs of the pair, based on whether each
run's `ndcg_at_10` for that query is strictly greater than 0 (an
Answered_Query) or exactly 0 (a Missed_Query):

- **`a_only`** — Run_A answered, Run_B missed.
- **`b_only`** — Run_B answered, Run_A missed.
- **`both_miss`** — both missed.
- **`both_answer`** — both answered.

The Declared_Contrast_Set is 12 pairs, fixed before any count existed:
8 family-aligned pairs (the reference run against each of the 8
non-reference runs — one per Pre_Declared_Family row, including the two
BM25 cross-strategy pairs), plus 4 dense cross-strategy pairs (each
dense retriever's `whole_document` run against its own `fixed_window`
and `sentence_window` runs).

**A note on two figures that sound alike but are not.** Answered_Query
and Missed_Query are defined on `ndcg_at_10`, a **top-10** notion.
`total_miss` is defined on `recall_at_20`, a **top-20** notion. A query
in the `both_miss` bucket of a contrast is therefore not the same
condition as "both runs recorded a `total_miss`" — a query can be a
Missed_Query (no judged-relevant document in the top 10) while still
having one appear between ranks 11 and 20, in which case it is
`mis_ranked`, not `total_miss`. Wherever both figures are reported below,
this document keeps them separate.

Per-run failure-bucket counts and fractions, from
`results/failure_bucket_counts.csv`:

| Run_Id | total_miss | mis_ranked | partial_recall | full_success |
|---|---|---|---|---|
| `bm25__whole_document` | 45 (0.1500) | 98 (0.3267) | 4 (0.0133) | 153 (0.5100) |
| `bm25__fixed_window` | 49 (0.1633) | 98 (0.3267) | 4 (0.0133) | 149 (0.4967) |
| `bm25__sentence_window` | 52 (0.1733) | 105 (0.3500) | 3 (0.0100) | 140 (0.4667) |
| `all-MiniLM-L6-v2__whole_document` | 46 (0.1533) | 103 (0.3433) | 1 (0.0033) | 150 (0.5000) |
| `all-MiniLM-L6-v2__fixed_window` | 41 (0.1367) | 97 (0.3233) | 1 (0.0033) | 161 (0.5367) |
| `all-MiniLM-L6-v2__sentence_window` | 40 (0.1333) | 102 (0.3400) | 0 (0.0000) | 158 (0.5267) |
| `bge-small-en-v1.5__whole_document` | 36 (0.1200) | 83 (0.2767) | 0 (0.0000) | 181 (0.6033) |
| `bge-small-en-v1.5__fixed_window` | 34 (0.1133) | 88 (0.2933) | 0 (0.0000) | 178 (0.5933) |
| `bge-small-en-v1.5__sentence_window` | 35 (0.1167) | 88 (0.2933) | 1 (0.0033) | 176 (0.5867) |

Every run's four counts sum to 300, the number of `query_id` values
`results/per_query.csv` holds for every run.

All 12 Pair_Contrasts, none omitted for being unremarkable, from
`results/failure_bucket_counts.csv`:

| Pair_Contrast (Run_A \| vs \| Run_B) | a_only | b_only | both_miss | both_answer |
|---|---|---|---|---|
| `bm25__whole_document` vs `bm25__fixed_window` | 4 | 3 | 58 | 235 |
| `bm25__whole_document` vs `bm25__sentence_window` | 8 | 4 | 57 | 231 |
| `bm25__whole_document` vs `all-MiniLM-L6-v2__whole_document` | 24 | 23 | 38 | 215 |
| `bm25__whole_document` vs `all-MiniLM-L6-v2__fixed_window` | 17 | 28 | 33 | 222 |
| `bm25__whole_document` vs `all-MiniLM-L6-v2__sentence_window` | 21 | 25 | 36 | 218 |
| `bm25__whole_document` vs `bge-small-en-v1.5__whole_document` | 15 | 33 | 28 | 224 |
| `bm25__whole_document` vs `bge-small-en-v1.5__fixed_window` | 15 | 30 | 31 | 224 |
| `bm25__whole_document` vs `bge-small-en-v1.5__sentence_window` | 14 | 32 | 29 | 225 |
| `all-MiniLM-L6-v2__whole_document` vs `all-MiniLM-L6-v2__fixed_window` | 2 | 14 | 48 | 236 |
| `all-MiniLM-L6-v2__whole_document` vs `all-MiniLM-L6-v2__sentence_window` | 8 | 13 | 49 | 230 |
| `bge-small-en-v1.5__whole_document` vs `bge-small-en-v1.5__fixed_window` | 4 | 1 | 42 | 253 |
| `bge-small-en-v1.5__whole_document` vs `bge-small-en-v1.5__sentence_window` | 7 | 7 | 36 | 250 |

## 5. Mechanism: `bge-small-en-v1.5` above the reference row under all three chunking strategies

`bge-small-en-v1.5` beats `bm25__whole_document` on the primary metric
under every chunking strategy, and every one of the three comparisons
is `significant` after Holm-Bonferroni correction (§3):
`whole_document` +0.0681 (p = 0.0042), `fixed_window` +0.0587
(p = 0.0130), `sentence_window` +0.0624 (p = 0.0078).

Against `bm25__whole_document` (`total_miss` = 45/300, `mis_ranked` =
98/300), every `bge-small-en-v1.5` run has a smaller `total_miss` count
(36, 34, 35) and a smaller `mis_ranked` count (83, 88, 88). The
`bm25__whole_document` vs `bge-small-en-v1.5__whole_document` contrast
resolves this at the per-query level: `b_only` (bge answered, BM25
missed) = 33 queries, versus `a_only` (BM25 answered, bge missed) = 15
queries — bge wins the disagreement roughly 2-to-1, consistent with the
aggregate direction. The `fixed_window` and `sentence_window` contrasts
against BM25 show the same asymmetry (`b_only` 30 vs `a_only` 15;
`b_only` 32 vs `a_only` 14).

Because all three verdicts here are `significant`, a covariate-grounded
mechanism is permitted. Reading `results/failure_buckets.csv`'s
`any_relevant_doc_exceeds_limit__bge-small-en-v1_5` column within each
run's `total_miss` bucket and, separately, within that same run's
`full_success` bucket — the two buckets at opposite ends of the same
run's own outcome — shows the same split in both places:

| Run_Id | `total_miss`: over limit | `total_miss`: within limit | `full_success`: over limit | `full_success`: within limit |
|---|---|---|---|---|
| `bm25__whole_document` | 9 | 36 | 28 | 125 |
| `bge-small-en-v1.5__whole_document` | 6 | 30 | 36 | 145 |
| `bge-small-en-v1.5__fixed_window` | 6 | 28 | 34 | 144 |
| `bge-small-en-v1.5__sentence_window` | 7 | 28 | 32 | 144 |

In every row, "within limit" outnumbers "over limit" by roughly the
same margin whether the bucket is `total_miss` or `full_success` — a
`bge-small-en-v1.5` `total_miss` query is not visibly more likely to
have a judged-relevant document over `bge-small-en-v1.5`'s own
512-token limit than a `full_success` query from the same run is.
**No length-truncation mechanism was identified for this result.** The
bucket figures earlier in this section — smaller `total_miss`/
`mis_ranked` counts and a `b_only`-favoring disagreement pattern against
BM25 — describe *where* `bge-small-en-v1.5` wins, but this document does
not have a covariate account of *why* beyond that description.

## 6. Mechanism: `bm25__sentence_window` below the reference row

`bm25__sentence_window` falls below `bm25__whole_document` on the
primary metric, significantly: `mean_diff` = -0.0417,
`p_value_adjusted` = 0.0032, `verdict` = `significant` (§3).

`bm25__sentence_window`'s own bucket counts are worse than
`bm25__whole_document`'s across the board: `total_miss` 52 vs 45,
`mis_ranked` 105 vs 98, `full_success` 140 vs 153. The
`bm25__whole_document` vs `bm25__sentence_window` contrast shows the
disagreement is one-sided: `b_only` (sentence_window answered,
whole_document missed) = 4 queries, versus `a_only` (whole_document
answered, sentence_window missed) = 8 queries — twice as many queries
lost to chunking as gained.

A covariate-grounded mechanism is not available for this comparison,
and not because a measurement is missing: both compared runs are BM25,
and BM25 does not truncate at any token length, so a token-length
covariate measured against a *dense model's* limit has no bearing on
why sentence-window chunking hurt a lexical retriever. Separately,
Section 11 notes that a covariate measures the source document's full
`title` + `text`, never the three-sentence window a `sentence_window`
run actually indexed, which is the wrong unit for this comparison even
setting the truncation point aside. **No mechanism was identified for
this result.** The bucket figures above describe the size and direction
of the loss; this document does not have an account of its cause.

## 7. The four comparisons the study could not distinguish from noise

Four comparisons against the reference run are `indistinguishable` from
noise, per `results/significance.csv`'s `verdict` column: the three
`all-MiniLM-L6-v2` runs (`whole_document`, `fixed_window`,
`sentence_window`) and `bm25__fixed_window`. Each is described here as
indistinguishable, and as a win for neither side. The indistinguishable
verdict is never treated as proof that the two compared runs perform
identically — a wide confidence interval containing zero is a failure
to distinguish a difference from noise, not a demonstration that the
runs are the same.

No mechanism, cause, explanation, or failure-bucket account is offered
for the *direction* of any of these four `mean_diff` values, and the
covariate columns change nothing about that: Requirement 12.9 of this
project's own build plan states the rule plainly, and it is restated
here because it is the sharpest constraint in this document — **a
covariate column licenses a description of a set of queries, and never
a mechanism, cause, or explanation for a comparison this study could
not distinguish from noise.** The line is drawn at the verb, not at the
evidence: a well-sourced story about noise is still a story about
noise, and would be more persuasive to a reader for being well-sourced,
which makes it worse rather than better.

What can be stated, as description only, for each of the four:

- **`all-MiniLM-L6-v2__whole_document` vs `bm25__whole_document`**: the
  runs disagreed on 47 of 300 queries (`a_only` 24 + `b_only` 23);
  `all-MiniLM-L6-v2__whole_document`'s own `total_miss` bucket has 46
  queries, of which 33 have a judged-relevant document over
  `all-MiniLM-L6-v2`'s 256-token limit
  (`any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2`), against 115 of
  that same run's 150 `full_success` queries. The aggregate nDCG@10
  difference between these two runs is indistinguishable from noise.
- **`all-MiniLM-L6-v2__fixed_window` vs `bm25__whole_document`**: the
  runs disagreed on 45 of 300 queries (`a_only` 17 + `b_only` 28);
  `all-MiniLM-L6-v2__fixed_window`'s own `total_miss` bucket has 41
  queries, of which 28 have a judged-relevant document over the same
  limit, against 130 of that same run's 161 `full_success` queries. The
  aggregate nDCG@10 difference is indistinguishable from noise.
- **`all-MiniLM-L6-v2__sentence_window` vs `bm25__whole_document`**: the
  runs disagreed on 46 of 300 queries (`a_only` 21 + `b_only` 25);
  `all-MiniLM-L6-v2__sentence_window`'s own `total_miss` bucket has 40
  queries, of which 30 have a judged-relevant document over the same
  limit, against 124 of that same run's 158 `full_success` queries. The
  aggregate nDCG@10 difference is indistinguishable from noise.
- **`bm25__fixed_window` vs `bm25__whole_document`**: the runs disagreed
  on only 7 of 300 queries (`a_only` 4 + `b_only` 3) — far fewer than any
  of the other eleven Pair_Contrasts in §4, consistent with a small,
  noise-level effect. `bm25__fixed_window`'s own `total_miss` bucket has
  49 queries. `bm25` does not truncate, so no token-length covariate
  applies to this pair — the same reasoning as §6.

None of the four bullets above is stated as, or should be read as, an
explanation for why the corresponding `mean_diff` sits where it does.

## 8. Chunking and truncation: the corpus level

`results/token_length_report.json` measures what fraction of units
(whole documents, or Chunks under `fixed_window`/`sentence_window`)
exceed each dense model's own Effective_Max_Sequence_Length, per
(Chunking_Strategy, dense model) cell. No token count is computed in
this document; every figure below is read directly from that file's
`cells` entries.

| Chunking_Strategy | Model | Effective_Max_Sequence_Length | Units total | Units exceeding | Fraction exceeding |
|---|---|---|---|---|---|
| `whole_document` | `all-MiniLM-L6-v2` | 256 | 5183 | 3681 | 0.7102 (71.02%) |
| `whole_document` | `bge-small-en-v1.5` | 512 | 5183 | 455 | 0.0878 (8.78%) |
| `fixed_window` | `all-MiniLM-L6-v2` | 256 | 22033 | 0 | 0.0000 |
| `fixed_window` | `bge-small-en-v1.5` | 512 | 22033 | 0 | 0.0000 |
| `sentence_window` | `all-MiniLM-L6-v2` | 256 | 18908 | 12 | 0.0006 |
| `sentence_window` | `bge-small-en-v1.5` | 512 | 18908 | 0 | 0.0000 |

Under `whole_document` chunking, `all-MiniLM-L6-v2`'s 256-token limit
is exceeded by nearly three-quarters of the corpus; `bge-small-en-v1.5`'s
512-token limit is exceeded by less than a tenth of it. Under
`fixed_window` and `sentence_window` chunking, truncation is
essentially eliminated for both models, because each Chunk is itself
short enough to fit under either limit almost always.

This table describes the corpus. It says nothing about which
*queries'* judged-relevant documents were affected — that is Section
8a's subject, and the reason Section 8a exists as a separate section:
a corpus-level fraction cannot be pushed down to an individual query or
an individual failure bucket without a different, query-level
measurement.

## 8a. Truncation at the query level

Section 8's corpus-level fractions describe the corpus; they do not say
which queries were affected. `results/failure_buckets.csv`'s six
per-query covariate columns make that query-level statement possible:
for each (`query_id`, dense model) pair, whether at least one of that
query's judged-relevant documents (measured over the full source
document's `title` + `text`, per `src/token_length_analysis.py`'s
`format_document_text`) exceeds that model's own
Effective_Max_Sequence_Length.

Reading `results/failure_buckets.csv`'s `any_relevant_doc_exceeds_limit__*`
columns for `bm25__whole_document` (a run every query belongs to, so
its own `total_miss` and `full_success` buckets between them cover the
whole comparison this section needs): under `all-MiniLM-L6-v2`'s
256-token limit, `bm25__whole_document`'s `total_miss` bucket has 29
over-limit and 16 within-limit queries, while its `full_success` bucket
has 117 over-limit and 36 within-limit queries. Under
`bge-small-en-v1.5`'s 512-token limit, the same run's `total_miss`
bucket has 9 over-limit and 36 within-limit queries.

Restricting to `all-MiniLM-L6-v2__whole_document`'s own `total_miss`
bucket specifically: 33 of that bucket's 46 queries have a
judged-relevant document over the limit, against 115 of that same run's
150 `full_success` queries — the two buckets carry a similar mix of
over-limit and within-limit queries rather than a clearly different
one. Section 7 already states why this cannot be read as a mechanism
for `all-MiniLM-L6-v2__whole_document`: its `verdict` against the
reference run is `indistinguishable`, and Requirement 12.9 (§7) forbids
a covariate-grounded mechanism for any of the four indistinguishable
comparisons regardless of what the covariate shows. What can be said,
as description only: 33 of `all-MiniLM-L6-v2__whole_document`'s 46
missed queries have a judged-relevant document that would not fit
whole inside the model's own 256-token window — a fact about which
queries were at risk of truncation, not an explanation of why the
aggregate metric came out where it did.

For the runs where a mechanism *is* permitted (§5, all three
`bge-small-en-v1.5` comparisons, all `significant`), §5's table shows
the over-limit-versus-within-limit split inside `total_miss` and inside
`full_success` landing in a similar ratio for both buckets — so even
where the constraint permits a truncation mechanism, this document does
not have one to offer, and says so plainly rather than reading a
non-separating pair of splits as support for one.

The comparison unit throughout this section is the source document, not
the Chunk a `fixed_window` or `sentence_window` run actually indexed —
see Section 11 for why that limits how this section's figures should be
read for the two chunked strategies.

## 9. Per-query covariates behind the buckets

Two kinds of per-query covariate are available in
`results/failure_buckets.csv`, both described here against their
corpus-wide distribution rather than tested as a hypothesis.

**`num_judged_relevant`** (inherited unchanged from
`results/per_query.csv`) ranges from 1 to 5 across the 300 queries: 277
queries have exactly 1 judged-relevant document, 14 have 2, 4 have 3, 3
have 4, and 2 have 5. Summing the four counts above 1, only 23 of the
300 queries have more than one judged-relevant document — the
precondition for the `partial_recall` bucket. This is why
`partial_recall` is small in every run's bucket counts in §4 (0–4
queries out of 300): the covariate that would let a query land there is
itself rare in this corpus, and reporting that plainly is the correct
account rather than a gap to explain away.

**The six token-length covariate columns.** `query_token_len__*`
(measured from the query text itself) is short: query `1`'s own
`query_token_len__all-MiniLM-L6-v2` is 15 tokens, well under either
model's limit — a representative case, since SciFact claims are short
sentences and the query side of the covariate is not where this
document's truncation discussion (§8/§8a) finds its mass. The document
side carries essentially all of that risk: `max_relevant_doc_token_len__*`
is what varies enough to matter, because a judged-relevant abstract can
run well past 256 or even 512 tokens while a claim stays short.

The one figure that makes `all-MiniLM-L6-v2`'s and `bge-small-en-v1.5`'s
covariate columns non-interchangeable: the two models do not share an
Effective_Max_Sequence_Length (256 vs 512, both read from
`results/token_length_report.json`'s `cells[].max_sequence_length`,
never typed in). The same document can be over the limit for
`all-MiniLM-L6-v2` and under it for `bge-small-en-v1.5` at once — this
is exactly why, on `bm25__whole_document` (§8a), the `total_miss` bucket
has far more over-limit queries under `all-MiniLM-L6-v2`'s 256-token
limit (29) than under `bge-small-en-v1.5`'s 512-token limit (9), even
though both counts are read from the same underlying document lengths.

Where a bucket's covariate distribution does not separate that bucket
from the rest of a run's queries, §5 and §8a already say so rather than
substitute a narrative: this section is descriptive infrastructure for
those two, not an independent source of a mechanism claim.

## 10. Where the generated-answer gate fits

The groundedness gate (`results/groundedness.csv`,
`results/hand_checked_joined.csv`, `results/generated_answers.csv`) is a
property of a 30-claim Generation_Subset, not a retrieval result, and is
noted here only for cross-reference. Of the 30 claims produced, 11 were
quarantined (Quarantine_Rate = 0.3667). Against a 30-claim
Hand_Checked_Sample, the pooled Agreement_Rate between the Judge_Model's
verdict and the human hand label is 0.2667 — well below chance on a
binary label, and, per `SPEC.md`'s own discussion, concentrated in
claims the human rater treated as non-assertions. That gate is unrelated
to the sweep's failure buckets: it never touches `results/per_query.csv`,
`results/failure_buckets.csv`, or the Qrels-based recall/nDCG/MRR
metrics this document discusses above.

## 11. What this analysis cannot establish

- **The taxonomy was fixed before assignment.** The four failure-bucket
  predicates and the four contrast-bucket rules (§4) were declared as
  constants in `src/failure_buckets.py` before any label was computed.
  Every count and fraction in this document describes that fixed
  partition; none of them tests a hypothesis about it.
- **Sparse Qrels mean a miss is a miss of a *judged* document, not of
  any useful one.** A `total_miss` bucket assignment records that no
  document with a positive Qrels relevance score appeared in a run's
  top 20 — it says nothing about whether an unjudged document the run
  did retrieve was, in fact, useful. An unjudged document is always
  scored as non-relevant in this repository (per
  `.kiro/steering/evaluation-integrity.md`), and this document inherits
  that convention without re-examining it.
- **Every number here describes BEIR SciFact only.** Scientific claims
  against scientific abstracts is one domain with one length
  distribution; none of the counts, fractions, or covariate figures
  above is claimed to transfer to another corpus or domain.
- **Each Token_Length_Covariate measures the source document, not the
  Chunk a run actually indexed.** `format_document_text` composes a
  covariate from a corpus document's full `title` + `text` — the same
  text a `whole_document` run encodes as a single unit, but *not* the
  same text a `fixed_window` run (overlapping 200-token windows) or a
  `sentence_window` run (three-sentence groups) actually encoded. A
  document long enough to exceed `all-MiniLM-L6-v2`'s 256-token limit
  as a whole can still be under that limit as almost any one of its own
  windows or sentence groups, once it is split up — this is a mismatch
  of *unit*, not an absence of measurement, and it is why "this run's
  misses are the over-limit queries" is a sharper claim for a
  `whole_document` run than for `fixed_window` or `sentence_window` —
  §8a's figures should be read with that asymmetry in mind, most of all
  for the two chunked strategies.
- **Any mechanism this document offers is an account consistent with
  the bucket figures and the covariate columns, not a causal result
  this study established.** Sections 5, 6, 7, and 8a each say plainly,
  at the point the figures are presented, whether a mechanism was
  identified or not; none of those statements is a claim that the
  buckets or covariates *caused* the metric result they accompany.

None of the following appears anywhere in this document as work to
build: hybrid retrieval or score fusion (including reciprocal-rank
fusion), cross-encoder or language-model reranking of retrieved results,
a fourth retriever beyond `bm25`/`all-MiniLM-L6-v2`/`bge-small-en-v1.5`,
query expansion or query rewriting, generated pseudo-queries, an
approximate nearest neighbour index, or fine-tuning of any model. Where
this document names a question its data cannot answer — for example,
why `bge-small-en-v1.5` wins or why sentence-window chunking hurts
BM25 — that question is stated only as a limit of this study (see the
bullets above), never as an item on that list.

## 12. Reproducing the figures

Two commands reproduce every artifact this document cites:

```
python -m src.failure_buckets
```

regenerates `results/failure_buckets.csv` and
`results/failure_bucket_counts.csv` from the already-committed
`results/per_query.csv` and the already-cached BEIR SciFact corpus and
Dense_Model tokenizers under `data/`.

```
python -m src.verify_writeup_numbers --repo-root .
```

checks every number in this document (and in `README.md`/`SPEC.md`)
against the committed artifact it cites, and exits non-zero if any
number no longer matches.

The one asymmetry between these two commands is worth stating plainly:
the verifier needs only committed files and runs on a clean checkout
with no cache and no network access. The assigner's
Covariate_Enrichment_Stage needs a populated `data/` — the BEIR SciFact
dataset and both Dense_Model tokenizer snapshots — and **fails rather
than downloading them** if that cache is empty. A clean checkout can
therefore verify every number in this document, but cannot regenerate
the six covariate columns without first populating `data/` (which
running the sweep or the token-length analysis already does as a side
effect).
