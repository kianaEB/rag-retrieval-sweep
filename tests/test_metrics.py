"""Requirement 11 test suite for the Metrics_Calculator (`src/metrics.py`).

Covers, for each of `recall_at_k`, `ndcg_at_10`, `mrr_at_10`: no
judged-relevant document, a relevant doc outside the top-k cutoff,
perfect ranking, and an empty ranked list -- all fixtures are Python
literals with <=10 documents, expected values are hand-computed
independently of the function under test, and float comparisons use
`pytest.approx(..., abs=1e-6)` (Requirement 11.1, 11.2).

This module imports only `src.metrics` and `pytrec_eval`. It does not
import `src.corpus_loader`, `src.sweep_runner`, or any retriever module,
and makes no network call and downloads nothing (Requirement 11.3,
11.4) -- every fixture below is a literal defined in this file.

The `pytrec_eval` differential cross-check (Requirement 11.5) is
included for all three metrics. For MRR@10, the ranked list handed to
`pytrec_eval` is truncated to its top 10 documents before comparison,
because `pytrec_eval`'s `recip_rank` measure scans the full submitted
ranking with no built-in cutoff; the `recall.<k>` and `ndcg_cut.10`
comparisons pass the full ranked list unmodified, since those
`pytrec_eval` measures already apply their own cutoff internally.
"""

from __future__ import annotations

import pytest
import pytrec_eval

from src.metrics import mrr_at_10, ndcg_at_10, recall_at_k

QID = "q1"


def _recall_via_pytrec_eval(ranked_list, qrels_for_query, k):
    qrels = {QID: qrels_for_query}
    run = {
        QID: {doc_id: float(len(ranked_list) - i) for i, doc_id in enumerate(ranked_list)}
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"recall.{k}"})
    results = evaluator.evaluate(run)
    if QID not in results:
        # pytrec_eval omits a query from the results dict entirely when
        # its run is empty (e.g. an empty ranked_list) -- that is its
        # equivalent of a 0.0 recall, not a missing measurement.
        return 0.0
    return results[QID][f"recall_{k}"]


def _ndcg10_via_pytrec_eval(ranked_list, qrels_for_query):
    qrels = {QID: qrels_for_query}
    run = {
        QID: {doc_id: float(len(ranked_list) - i) for i, doc_id in enumerate(ranked_list)}
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    results = evaluator.evaluate(run)
    if QID not in results:
        return 0.0
    return results[QID]["ndcg_cut_10"]


def _mrr10_via_pytrec_eval(ranked_list, qrels_for_query):
    truncated = ranked_list[:10]
    qrels = {QID: qrels_for_query}
    run = {
        QID: {doc_id: float(len(truncated) - i) for i, doc_id in enumerate(truncated)}
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    results = evaluator.evaluate(run)
    if QID not in results:
        return 0.0
    return results[QID]["recip_rank"]


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_no_judged_relevant_document():
    ranked_list = ["d1", "d2", "d3"]
    qrels_for_query = {}  # no judged-relevant docs at all
    assert recall_at_k(ranked_list, qrels_for_query, k=10) == pytest.approx(0.0, abs=1e-6)


def test_recall_at_k_relevant_doc_outside_top_k_cutoff():
    # 10 docs, ranks 1..10; the only two judged-relevant docs are at
    # rank 10 ("d10") and never retrieved at all ("d_missing"). Neither
    # is within the top-5 cutoff.
    ranked_list = [f"d{i}" for i in range(1, 11)]
    qrels_for_query = {"d10": 1, "d_missing": 1}
    # 0 of 2 judged-relevant docs appear in top 5 -> 0/2 = 0.0
    assert recall_at_k(ranked_list, qrels_for_query, k=5) == pytest.approx(0.0, abs=1e-6)


def test_recall_at_k_perfect_ranking():
    ranked_list = ["d1", "d2", "d3", "d4", "d5"]
    qrels_for_query = {"d1": 1, "d2": 2}
    # both judged-relevant docs occupy the top positions -> 2/2 = 1.0
    assert recall_at_k(ranked_list, qrels_for_query, k=5) == pytest.approx(1.0, abs=1e-6)


def test_recall_at_k_empty_ranked_list():
    ranked_list: list[str] = []
    qrels_for_query = {"d1": 1}
    assert recall_at_k(ranked_list, qrels_for_query, k=10) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("ranked_list", "qrels_for_query", "k"),
    [
        (["d1", "d2", "d3"], {}, 10),
        ([f"d{i}" for i in range(1, 11)], {"d10": 1, "d_missing": 1}, 5),
        (["d1", "d2", "d3", "d4", "d5"], {"d1": 1, "d2": 2}, 5),
        ([], {"d1": 1}, 10),
    ],
)
def test_recall_at_k_matches_pytrec_eval(ranked_list, qrels_for_query, k):
    ours = recall_at_k(ranked_list, qrels_for_query, k)
    theirs = _recall_via_pytrec_eval(ranked_list, qrels_for_query, k)
    assert ours == pytest.approx(theirs, abs=1e-6)


# ---------------------------------------------------------------------------
# ndcg_at_10
# ---------------------------------------------------------------------------


def test_ndcg_at_10_no_judged_relevant_document():
    ranked_list = [f"d{i}" for i in range(1, 11)]
    qrels_for_query = {}  # IDCG@10 == 0
    assert ndcg_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


def test_ndcg_at_10_relevant_doc_outside_top_10_cutoff():
    # 11 docs; the only judged-relevant doc ("d11") sits at rank 11,
    # outside the fixed nDCG@10 cutoff, so DCG@10 == 0 while IDCG@10 > 0.
    ranked_list = [f"d{i}" for i in range(1, 12)]
    qrels_for_query = {"d11": 1}
    assert ndcg_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


def test_ndcg_at_10_perfect_ranking():
    # Relevant docs occupy ranks 1-2 in descending-relevance order, so
    # DCG@10 == IDCG@10 by construction.
    ranked_list = ["d1", "d2"] + [f"filler{i}" for i in range(8)]
    qrels_for_query = {"d1": 2, "d2": 1}
    assert ndcg_at_10(ranked_list, qrels_for_query) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_at_10_empty_ranked_list():
    ranked_list: list[str] = []
    qrels_for_query = {"d1": 1}
    assert ndcg_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("ranked_list", "qrels_for_query"),
    [
        ([f"d{i}" for i in range(1, 11)], {}),
        ([f"d{i}" for i in range(1, 12)], {"d11": 1}),
        (["d1", "d2"] + [f"filler{i}" for i in range(8)], {"d1": 2, "d2": 1}),
        ([], {"d1": 1}),
    ],
)
def test_ndcg_at_10_matches_pytrec_eval(ranked_list, qrels_for_query):
    ours = ndcg_at_10(ranked_list, qrels_for_query)
    theirs = _ndcg10_via_pytrec_eval(ranked_list, qrels_for_query)
    assert ours == pytest.approx(theirs, abs=1e-6)


# ---------------------------------------------------------------------------
# mrr_at_10
# ---------------------------------------------------------------------------


def test_mrr_at_10_no_judged_relevant_document():
    ranked_list = [f"d{i}" for i in range(1, 11)]
    qrels_for_query = {}
    assert mrr_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


def test_mrr_at_10_relevant_doc_outside_top_10_cutoff():
    # The only judged-relevant doc ("d11") is at rank 11, beyond the
    # fixed MRR@10 cutoff, so no relevant doc appears within top 10.
    ranked_list = [f"d{i}" for i in range(1, 12)]
    qrels_for_query = {"d11": 1}
    assert mrr_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


def test_mrr_at_10_perfect_ranking():
    ranked_list = ["d1", "d2", "d3"]
    qrels_for_query = {"d1": 1}
    # first (and only) judged-relevant doc is at rank 1 -> reciprocal rank = 1.0
    assert mrr_at_10(ranked_list, qrels_for_query) == pytest.approx(1.0, abs=1e-6)


def test_mrr_at_10_empty_ranked_list():
    ranked_list: list[str] = []
    qrels_for_query = {"d1": 1}
    assert mrr_at_10(ranked_list, qrels_for_query) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("ranked_list", "qrels_for_query"),
    [
        ([f"d{i}" for i in range(1, 11)], {}),
        ([f"d{i}" for i in range(1, 12)], {"d11": 1}),
        (["d1", "d2", "d3"], {"d1": 1}),
        ([], {"d1": 1}),
    ],
)
def test_mrr_at_10_matches_pytrec_eval(ranked_list, qrels_for_query):
    ours = mrr_at_10(ranked_list, qrels_for_query)
    theirs = _mrr10_via_pytrec_eval(ranked_list, qrels_for_query)
    assert ours == pytest.approx(theirs, abs=1e-6)
