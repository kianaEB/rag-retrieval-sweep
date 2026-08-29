"""Requirement 12 test suite: the Sweep_Runner orchestration loop
(`run_sweep`, `src/sweep_runner.py`), driven entirely by a stub
retriever and an in-memory corpus.

This module defines its own `StubRetriever` and `In_Memory_Test_Corpus`
literals (Requirement 12.1, 12.2, 12.3) instead of importing a real
`BM25Retriever`/`DenseRetriever`. It imports only `src.config`,
`src.corpus_loader`, `src.metrics`, and `src.sweep_runner` -- never
`beir`, `sentence-transformers`, or `huggingface_hub` (Requirement
12.7) -- so nothing here downloads a dataset or a model, or makes a
network call.

The "index once, retrieve once, slice four ways" property (Requirement
5, verified here per Requirement 12) is checked two ways:

1. Call counting: each `StubRetriever`'s own `build_index_calls` and
   `retrieve_all_calls` lists (populated by the stub itself) are
   asserted to have length exactly 1, and the single recorded
   `retrieve_all` call's `top_k` argument is asserted to equal
   `max(cutoffs)` -- proving no separate `retrieve_all` call was ever
   issued for any other declared cutoff (Requirement 12.4, 12.5).
2. Prefix-slice correctness: `src.metrics.recall_at_k` is monkeypatched
   with a spy (still delegating to the real function) so every one of
   its calls made from inside `run_sweep` can be inspected. For every
   declared cutoff `k` and every query, the spy's recorded `ranked_list`
   argument is asserted to be exactly the query's entry in the single
   list a `StubRetriever` returned from its one `retrieve_all` call --
   never a different or separately retrieved list -- which is exactly
   Requirement 12.6's "prefix slice of the single deepest-cutoff list"
   property, checked per query and per cutoff rather than only in
   aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import src.sweep_runner as sweep_runner_module
from src.config import BM25RetrieverConfig, DenseRetrieverConfig, SweepConfig
from src.corpus_loader import CorpusBundle
from src.metrics import mean_over_qualifying_queries, recall_at_k
from src.sweep_runner import run_sweep

# ---------------------------------------------------------------------------
# In_Memory_Test_Corpus (Requirement 12.3): <=5 documents, defined as
# literals in this module -- never the real BEIR SciFact corpus,
# queries, or qrels.
# ---------------------------------------------------------------------------

CORPUS: Dict[str, Dict[str, str]] = {
    "1": {"title": "Doc One", "text": "alpha beta"},
    "2": {"title": "Doc Two", "text": "gamma delta"},
    "3": {"title": "Doc Three", "text": "epsilon zeta"},
    "4": {"title": "Doc Four", "text": "eta theta"},
}

QUERIES: Dict[str, str] = {
    "q1": "alpha query",
    "q2": "gamma query",
}

# doc_id -> graded relevance score. Both queries below have >=1 judged
# relevant doc, so both qualify and run_sweep's zero-qualifying-queries
# halt (Requirement 6.7) never triggers here.
QRELS: Dict[str, Dict[str, int]] = {
    "q1": {"2": 1, "4": 1},
    "q2": {"3": 1},
}

BUNDLE = CorpusBundle(corpus=CORPUS, queries=QUERIES, qrels=QRELS)

CUTOFFS: Tuple[int, ...] = (1, 2, 3)
DEEPEST_CUTOFF = max(CUTOFFS)

# Hand-specified, full (Deepest_Cutoff-length) Ranked_Lists per stub --
# literals, not computed from any similarity or lexical score
# (Requirement 12.2). Deliberately different per stub and per query so
# recall_at_k differs across k and across retrievers, making the
# assertions below non-trivial (never all 0.0 or all 1.0).
STUB_A_RANKED_LISTS: Dict[str, List[str]] = {
    "q1": ["1", "2", "3"],
    "q2": ["3", "1", "2"],
}
STUB_B_RANKED_LISTS: Dict[str, List[str]] = {
    "q1": ["4", "2", "1"],
    "q2": ["1", "3", "2"],
}


@dataclass
class StubRetriever:
    """Test-only implementation of the `Retriever` Protocol
    (`src/retrievers/base.py`) used solely to verify `run_sweep`'s
    orchestration behavior (Requirement 12.1). Records every
    `build_index`/`retrieve_all` call and its arguments; performs no
    real indexing or retrieval computation, loads no model, and makes
    no network call. `retrieve_all` always returns the same
    hand-specified, in-memory `full_ranked_lists` literal regardless of
    its arguments (Requirement 12.2).
    """

    name: str
    full_ranked_lists: Dict[str, List[str]]
    index_time_value: float
    query_latency_value: float
    build_index_calls: List[Dict[str, Dict[str, str]]] = field(default_factory=list)
    retrieve_all_calls: List[Tuple[Dict[str, str], int]] = field(default_factory=list)

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        self.build_index_calls.append(corpus)
        return self.index_time_value

    def retrieve_all(
        self, queries: Dict[str, str], top_k: int
    ) -> Tuple[Dict[str, List[str]], float]:
        self.retrieve_all_calls.append((queries, top_k))
        return self.full_ranked_lists, self.query_latency_value


def _make_config(stub_a_name: str, stub_b_name: str) -> SweepConfig:
    """A 2-retriever, 3-cutoff SweepConfig pointing at the two stub
    names below -- never a real retriever type, and never parsed from
    `configs/sweep.yaml` (this test drives `run_sweep` directly, not
    through `load_sweep_config`)."""
    bm25_cfg = BM25RetrieverConfig(
        name=stub_a_name,
        type="bm25",
        k1=0.9,
        b=0.4,
        tokenizer="regex_word",
        lowercase=True,
        stopwords="none",
        stemming="none",
    )
    dense_cfg = DenseRetrieverConfig(
        name=stub_b_name,
        type="dense",
        model_name="stub-model",
        batch_size=8,
    )
    return SweepConfig(
        seed=42,
        chunking_strategy="whole_document",
        cutoffs=CUTOFFS,
        retrievers=(bm25_cfg, dense_cfg),
        data_dir=Path("data"),
        output_path=Path("results/sweep.csv"),
    )


def test_run_sweep_orchestration_call_counts_and_prefix_slicing(monkeypatch):
    stub_a = StubRetriever(
        name="stub_bm25",
        full_ranked_lists=STUB_A_RANKED_LISTS,
        index_time_value=0.1,
        query_latency_value=0.2,
    )
    stub_b = StubRetriever(
        name="stub_dense",
        full_ranked_lists=STUB_B_RANKED_LISTS,
        index_time_value=0.3,
        query_latency_value=0.4,
    )
    stubs_by_name = {stub_a.name: stub_a, stub_b.name: stub_b}
    config = _make_config(stub_a.name, stub_b.name)

    def retriever_factory(retriever_config):
        return stubs_by_name[retriever_config.name]

    # Spy on src.metrics.recall_at_k as seen through src.sweep_runner's
    # module namespace (where it was bound by `from src.metrics import
    # recall_at_k`) so every call run_sweep makes to it can be inspected,
    # while still delegating to the real implementation so run_sweep's
    # returned rows are unaffected.
    recorded_recall_calls: List[Tuple[List[str], Dict[str, int], int]] = []
    real_recall_at_k = recall_at_k

    def spy_recall_at_k(ranked_list, qrels_for_query, k):
        recorded_recall_calls.append((list(ranked_list), dict(qrels_for_query), k))
        return real_recall_at_k(ranked_list, qrels_for_query, k)

    monkeypatch.setattr(sweep_runner_module, "recall_at_k", spy_recall_at_k)

    rows, per_query_rows, all_succeeded = run_sweep(config, BUNDLE, retriever_factory)

    # -- Row count and overall success (Requirement 7.1, 7.2) --------------
    assert all_succeeded is True
    assert len(rows) == len(config.retrievers) * len(config.cutoffs) == 6

    # -- Call counting: exactly one build_index + one retrieve_all per
    #    retriever, over the course of the run (Requirement 12.4) ---------
    for stub in (stub_a, stub_b):
        assert len(stub.build_index_calls) == 1
        assert len(stub.retrieve_all_calls) == 1
        assert stub.build_index_calls[0] == CORPUS

        retrieve_all_queries, retrieve_all_top_k = stub.retrieve_all_calls[0]
        assert retrieve_all_queries == QUERIES
        # The single retrieve_all call requested Deepest_Cutoff; no
        # separate call was recorded for any other declared cutoff
        # (Requirement 12.5).
        assert retrieve_all_top_k == DEEPEST_CUTOFF

    # -- Prefix-slice correctness, per query and per cutoff (Requirement
    #    12.6): recall_at_k is called, for every declared cutoff k and
    #    every query, with exactly that query's entry in the single list
    #    the stub returned from its one retrieve_all call -- never a
    #    different or separately retrieved list. Expected call order
    #    mirrors run_sweep's own loop nesting: retriever, then cutoff k
    #    (in config.cutoffs order), then query (in ranked_lists' -- i.e.
    #    the stub's own dict's -- insertion order).
    expected_recall_calls: List[Tuple[List[str], Dict[str, int], int]] = []
    for stub, full_ranked_lists in (
        (stub_a, STUB_A_RANKED_LISTS),
        (stub_b, STUB_B_RANKED_LISTS),
    ):
        for k in config.cutoffs:
            for qid in full_ranked_lists:
                expected_recall_calls.append((full_ranked_lists[qid], QRELS.get(qid, {}), k))

    assert recorded_recall_calls == expected_recall_calls

    # -- End-to-end confirmation: each row's recall_at_k equals the mean,
    #    over qualifying queries, of recall_at_k computed on the first k
    #    elements of the exact list retrieve_all returned -- tying the
    #    call-level check above to the actual reported metric value.
    rows_by_key = {(row.retriever, row.k): row for row in rows}
    assert len(rows_by_key) == 6

    full_lists_by_name = {stub_a.name: STUB_A_RANKED_LISTS, stub_b.name: STUB_B_RANKED_LISTS}
    for retriever_name, full_ranked_lists in full_lists_by_name.items():
        for k in config.cutoffs:
            per_query_expected = {
                qid: recall_at_k(full_ranked_lists[qid][:k], QRELS.get(qid, {}), k)
                for qid in QUERIES
            }
            expected_recall_at_k = mean_over_qualifying_queries(per_query_expected, QRELS)
            row = rows_by_key[(retriever_name, k)]
            assert row.recall_at_k == pytest.approx(expected_recall_at_k, abs=1e-6)

    # Sanity: the expected values above are not all identical across k or
    # across retrievers, so the equality assertions above are not
    # trivially satisfied by e.g. every recall being 0.0 or 1.0.
    distinct_values = {rows_by_key[(stub_a.name, k)].recall_at_k for k in config.cutoffs}
    distinct_values |= {rows_by_key[(stub_b.name, k)].recall_at_k for k in config.cutoffs}
    assert len(distinct_values) > 1

    # -- index_time / query_latency copied unchanged into every row sharing
    #    a run_id (Requirement 5.6, 5.7) -------------------------------------
    for k in config.cutoffs:
        assert rows_by_key[(stub_a.name, k)].index_time == stub_a.index_time_value
        assert rows_by_key[(stub_a.name, k)].query_latency == stub_a.query_latency_value
        assert rows_by_key[(stub_b.name, k)].index_time == stub_b.index_time_value
        assert rows_by_key[(stub_b.name, k)].query_latency == stub_b.query_latency_value
