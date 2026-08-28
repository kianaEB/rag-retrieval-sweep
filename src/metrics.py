"""Metrics_Calculator: recall@k, nDCG@10, and MRR@10, computed strictly
against the BEIR SciFact qrels (Requirement 6).

Every function in this module is pure and operates on a single query's
`Ranked_List` (a list of document IDs, never scores -- see
`src/retrievers/base.py`'s `RetrievalRun.ranked_lists` typing) and that
query's qrels row. Qrels are the sole source of relevance ground truth:
a query-document pair absent from qrels, or present with a relevance
score <= 0, is always treated as not relevant, regardless of any
similarity or ranking score a retriever produced (Requirement 6.4).

These functions are the sole unit-under-test surface for Requirement 11
(`tests/test_metrics.py`). `mean_over_qualifying_queries` and
`scored_query_count` are thin aggregation helpers used by the
Sweep_Runner (`src/sweep_runner.py`) and are not part of that test
surface, per `design.md`'s `src/metrics.py` section.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Set, Tuple

from src.errors import ZeroQualifyingQueriesError

# doc_id -> graded relevance score from qrels for a single query.
QueryRelevance = Dict[str, int]


def judged_relevant_docs(qrels_for_query: QueryRelevance) -> Set[str]:
    """Doc IDs with relevance score > 0.

    Qrels are the only source of truth (Requirement 6.4): an entry
    absent from `qrels_for_query`, or present with a score of 0 or
    less, is never relevant.
    """
    return {doc_id for doc_id, score in qrels_for_query.items() if score > 0}


def recall_at_k(ranked_list: List[str], qrels_for_query: QueryRelevance, k: int) -> float:
    """Fraction of this query's judged-relevant docs appearing in the
    top-k of `ranked_list`.

    Returns 0.0 if the query has zero judged-relevant docs, or if
    `ranked_list` is empty -- both are defined edge cases for this
    function, not errors. Exclusion of zero-judged-relevant queries
    from the *mean* across queries is applied by the caller
    (`mean_over_qualifying_queries`), not here.
    """
    relevant = judged_relevant_docs(qrels_for_query)
    if not relevant:
        return 0.0
    top_k = set(ranked_list[:k])
    return len(relevant & top_k) / len(relevant)


def _dcg(relevances: List[int]) -> float:
    """DCG over a list of graded relevances, one per rank position.

    `relevances[i]` is the graded relevance of the document at rank
    `i + 1`. `log2(i + 2)` for a 0-indexed `i` is exactly `log2(rank +
    1)` for the 1-indexed rank, matching Requirement 6.2's
    `rel_i / log2(i + 1)` convention.
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_10(ranked_list: List[str], qrels_for_query: QueryRelevance) -> float:
    """DCG@10 / IDCG@10, at a fixed cutoff of 10 regardless of any row's k.

    Graded relevance for ranks 1-10 of `ranked_list` is taken directly
    from `qrels_for_query` (0 if the document is unjudged or absent).
    IDCG@10 is the same sum computed over this query's judged-relevant
    scores, sorted descending and truncated to the first 10. Returns
    0.0 when IDCG@10 is 0 (Requirement 6.2).
    """
    top10 = ranked_list[:10]
    gains = [qrels_for_query.get(doc_id, 0) for doc_id in top10]
    dcg = _dcg(gains)
    ideal_gains = sorted(
        (score for score in qrels_for_query.values() if score > 0),
        reverse=True,
    )[:10]
    idcg = _dcg(ideal_gains)
    return 0.0 if idcg == 0.0 else dcg / idcg


def mrr_at_10(ranked_list: List[str], qrels_for_query: QueryRelevance) -> float:
    """Reciprocal rank of the first judged-relevant doc within the top
    10 of `ranked_list`, or 0.0 if none appears there.

    Independent of any row's evaluation cutoff k (Requirement 6.3).
    """
    relevant = judged_relevant_docs(qrels_for_query)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(ranked_list[:10], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_over_qualifying_queries(
    per_query_values: Dict[str, float], qrels: Dict[str, QueryRelevance]
) -> float:
    """Arithmetic mean of `per_query_values`, restricted to queries that
    have at least one Qrels-judged relevant document (Requirement 6.1,
    6.2, 6.3).

    A query present in `per_query_values` but absent from `qrels`
    entirely (no judged documents at all, not even a zero-relevance
    entry) is treated the same as a query with an empty judged set --
    `qrels.get(qid, {})` -- and is excluded from the mean by the same
    `judged_relevant_docs` condition.
    """
    qualifying = [
        v
        for qid, v in per_query_values.items()
        if judged_relevant_docs(qrels.get(qid, {}))
    ]
    if not qualifying:
        # Defensive only: Sweep_Runner step 5 already halts the whole
        # run before the retriever loop when num_queries_scored == 0,
        # filtered by this same judged_relevant_docs condition over
        # the same qrels -- so this branch should be unreachable once
        # step 5 has run. Retained as a belt-and-suspenders invariant
        # check, not the primary enforcement point for this condition.
        raise ZeroQualifyingQueriesError(
            "no loaded query has a Qrels-judged relevant document; "
            "Sweep_Runner step 5 should have halted before this "
            "function was ever called"
        )
    return statistics.fmean(qualifying)


def scored_query_count(
    all_query_ids: Iterable[str], qrels: Dict[str, QueryRelevance]
) -> Tuple[int, int]:
    """Returns `(num_queries_total, num_queries_scored)`.

    `num_queries_total` is every query ID passed in -- every query the
    Corpus_Loader loaded, not merely the queries a particular retriever
    happened to score (Requirement 7.3). `num_queries_scored` is the
    subset with at least one Qrels-judged relevant document -- the
    actual denominator consumed by `mean_over_qualifying_queries`.
    """
    all_query_ids = list(all_query_ids)
    num_queries_total = len(all_query_ids)
    num_queries_scored = sum(
        1 for qid in all_query_ids if judged_relevant_docs(qrels.get(qid, {}))
    )
    return num_queries_total, num_queries_scored
