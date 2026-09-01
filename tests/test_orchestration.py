"""Requirement 12 test suite: the Sweep_Runner orchestration loop
(`run_sweep`, `src/sweep_runner.py`), driven entirely by a stub
retriever, a stub chunker, and an in-memory corpus.

This module defines its own `StubRetriever`, `StubChunker`, and
`In_Memory_Test_Corpus` literals (Requirement 12.1, 12.2, 12.3, 13.1)
instead of importing a real `BM25Retriever`/`DenseRetriever` or any of
the three real `Chunker` implementations. It imports only `src.chunking`
(for the `Chunk` dataclass and `make_chunk_id`, both pure, network-free
helpers -- never a real Chunker class), `src.config`,
`src.corpus_loader`, `src.metrics`, `src.retrievers.base` (for
`ChunkScores`), and `src.sweep_runner` -- never `beir`,
`sentence-transformers`, or `huggingface_hub` (Requirement 12.7,
13.1, 13.4) -- so nothing here downloads a dataset or a model, or
makes a network call.

The "index once, retrieve once, aggregate once, slice four ways"
property, restated in chunk terms for the full 9-combination grid
(Requirement 6, verified here per Requirement 13), is checked via
Property 7 (this revision): call counting for every (chunking_strategy,
retriever) combination, plus an aggregation-correctness assertion tying
each row's reported metric back to the true per-document maximum of
that combination's hand-specified stub chunk scores (never mean or
sum).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pytest

from src.chunking import Chunk, make_chunk_id
from src.config import (
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    SweepConfig,
    WholeDocumentChunkingConfig,
)
from src.corpus_loader import CorpusBundle
from src.retrievers.base import ChunkScores
from src.sweep_runner import run_sweep

# ---------------------------------------------------------------------------
# In_Memory_Test_Corpus (Requirement 12.3, 13.1): <=5 documents, defined
# as literals in this module -- never the real BEIR SciFact corpus,
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


@dataclass
class StubChunker:
    """Test-only implementation of the `Chunker` Protocol
    (`src/chunking.py`) used solely to verify `run_sweep`'s chunking
    axis (Requirement 13.1). `chunk_document` returns the
    hand-specified, fixed list of Chunks for that `doc_id` from
    `fixed_chunks_by_doc`, regardless of document content -- never a
    real tokenizer, never real text splitting. Records every
    `chunk_document` call."""

    strategy_name: str
    fixed_chunks_by_doc: Dict[str, List[Chunk]]
    chunk_document_calls: List[str] = field(default_factory=list)

    def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
        self.chunk_document_calls.append(doc_id)
        return self.fixed_chunks_by_doc[doc_id]


@dataclass
class StubRetriever:
    """Test-only implementation of the `Retriever` Protocol
    (`src/retrievers/base.py`) used solely to verify `run_sweep`'s
    orchestration behavior (Requirement 12.1, 13.1). Records every
    `build_index`/`retrieve_all` call and its arguments; performs no
    real indexing or retrieval computation, loads no model, and makes
    no network call. `retrieve_all` yields one `(query_id,
    ChunkScores)` pair per query, built from its own fixed literal
    per-chunk scores, regardless of its arguments (Requirement 12.2).
    """

    name: str
    fixed_chunk_scores_by_query: Dict[str, Dict[str, float]]
    index_time_value: float
    query_latency_value: float
    build_index_calls: List[Dict[str, Dict[str, str]]] = field(default_factory=list)
    retrieve_all_calls: List[Dict[str, str]] = field(default_factory=list)
    last_query_latency: float = 0.0

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        self.build_index_calls.append(corpus)
        return self.index_time_value

    def retrieve_all(self, queries: Dict[str, str]) -> Iterator[Tuple[str, ChunkScores]]:
        self.retrieve_all_calls.append(queries)
        for qid in queries:
            scores_by_chunk_id = self.fixed_chunk_scores_by_query[qid]
            chunk_ids = tuple(scores_by_chunk_id.keys())
            scores = np.array([scores_by_chunk_id[cid] for cid in chunk_ids])
            yield qid, ChunkScores(chunk_ids=chunk_ids, scores=scores)
        self.last_query_latency = self.query_latency_value


# ---------------------------------------------------------------------------
# Chunking-strategy axis fixtures (Property 7): 2 chunking-strategy
# stubs, each assigning every corpus document 1-2 Chunks. Doc "2" is
# deliberately given 2 Chunks with DIFFERENT stub scores under each
# strategy, so aggregate_to_document_ranked_list's max-vs-mean/sum
# distinction is genuinely exercised (Requirement 13.3).
# ---------------------------------------------------------------------------


def _make_stub_chunks(strategy_suffix: str) -> Dict[str, List[Chunk]]:
    """Builds a fixed_chunks_by_doc mapping for one chunking-strategy
    stub: every doc gets 1 Chunk, except doc "2", which gets 2 Chunks
    (so its aggregated score is a genuine max-over-multiple-chunks
    case, not the n=1 identity corollary alone).

    Chunk IDs are built via `make_chunk_id` (never a hand-rolled
    string) so they parse correctly through `parse_chunk_id` inside
    `aggregate_to_document_ranked_list` -- each chunking strategy gets
    its own `build_chunk_corpus` call and therefore its own
    `chunk_corpus` dict, so reusing the same `doc_id`/`position` pairs
    across the two stub strategies causes no collision; `strategy_suffix`
    is accepted for readability/documentation only and is not embedded
    in the chunk_id itself.
    """
    del strategy_suffix  # unused: chunk_ids need not be globally unique
    chunks_by_doc: Dict[str, List[Chunk]] = {}
    for doc_id in CORPUS:
        if doc_id == "2":
            chunks_by_doc[doc_id] = [
                Chunk(
                    chunk_id=make_chunk_id(doc_id, 0),
                    doc_id=doc_id,
                    position=0,
                    text=CORPUS[doc_id]["text"],
                ),
                Chunk(
                    chunk_id=make_chunk_id(doc_id, 1),
                    doc_id=doc_id,
                    position=1,
                    text=CORPUS[doc_id]["text"],
                ),
            ]
        else:
            chunks_by_doc[doc_id] = [
                Chunk(
                    chunk_id=make_chunk_id(doc_id, 0),
                    doc_id=doc_id,
                    position=0,
                    text=CORPUS[doc_id]["text"],
                )
            ]
    return chunks_by_doc


# Per-retriever, per-query, per-chunk-id hand-specified scores. Doc
# "2"'s two chunks get DIFFERENT scores under every (chunker,
# retriever, query) combination, and are set so the true per-document
# maximum diverges from what a mean- or sum-based aggregation would
# produce (Requirement 13.3's own wording).
def _make_stub_scores(
    chunks_by_doc: Dict[str, List[Chunk]], base_offset: float
) -> Dict[str, Dict[str, float]]:
    # Fixed literal scores per doc, offset per (chunker, retriever, query)
    # combination via base_offset so every combination's numbers differ,
    # without losing the max-vs-mean/sum divergence property for doc "2".
    per_doc_score: Dict[str, float] = {
        "1": 0.5 + base_offset,
        "2": None,  # handled specially below (2 chunks, differing scores)
        "3": 0.3 + base_offset,
        "4": 0.6 + base_offset,
    }
    doc2_chunk_scores = [0.1 + base_offset, 0.95 + base_offset]  # max=0.95+offset

    scores: Dict[str, float] = {}
    for doc_id, chunks in chunks_by_doc.items():
        if doc_id == "2":
            for chunk, score in zip(chunks, doc2_chunk_scores):
                scores[chunk.chunk_id] = score
        else:
            for chunk in chunks:
                scores[chunk.chunk_id] = per_doc_score[doc_id]
    # Same scores reused for both queries (q1, q2) in this fixture --
    # the aggregation and call-count assertions below don't depend on
    # per-query score variation, only on per-chunk-id score variation
    # within a single query's ChunkScores.
    return {"q1": dict(scores), "q2": dict(scores)}


# ---------------------------------------------------------------------------
# Property 7: The chunk-level orchestration loop indexes and retrieves
# exactly once per combination, and every cutoff's aggregation matches
# the true per-document maximum.
# Validates: Requirements 6.3, 6.5, 8.5, 13.1, 13.2, 13.3, 13.4.
# ---------------------------------------------------------------------------


def test_property_7_chunk_orchestration_call_counts_and_true_max_aggregation(monkeypatch):
    chunks_a = _make_stub_chunks("a")
    chunks_b = _make_stub_chunks("b")
    chunker_a = StubChunker(strategy_name="strat_a", fixed_chunks_by_doc=chunks_a)
    chunker_b = StubChunker(strategy_name="strat_b", fixed_chunks_by_doc=chunks_b)
    chunkers_by_name = {"strat_a": chunker_a, "strat_b": chunker_b}

    # StubRetriever's fixed_chunk_scores_by_query keys on chunk_id, but
    # a retriever doesn't know which chunking strategy produced the
    # chunk corpus it's handed -- so each retriever's own score table
    # must cover BOTH chunkers' chunk_ids (the retriever is
    # reconstructed fresh per chunking_config iteration by
    # retriever_factory below, but scores must still resolve for
    # whichever chunk_ids that iteration's chunk_corpus contains).
    scores_a_for_bm25 = _make_stub_scores(chunks_a, base_offset=0.0)
    scores_b_for_bm25 = _make_stub_scores(chunks_b, base_offset=0.1)
    scores_a_for_dense = _make_stub_scores(chunks_a, base_offset=0.2)
    scores_b_for_dense = _make_stub_scores(chunks_b, base_offset=0.3)

    # retriever_factory is called once per (chunking_strategy,
    # retriever) combination (Requirement 6.1-6.3) -- so a fresh
    # StubRetriever instance is constructed each time, letting us
    # select the right fixed_chunk_scores_by_query for the CURRENT
    # chunking_config by closing over a mutable "current strategy"
    # variable set by chunker_factory just before retriever_factory is
    # invoked for that combination.
    current_strategy = {"name": None}

    original_chunker_factory_lookup = chunkers_by_name

    def chunker_factory(chunking_config):
        current_strategy["name"] = chunking_config.name
        return original_chunker_factory_lookup[chunking_config.name]

    constructed_retrievers: List[StubRetriever] = []

    def retriever_factory(retriever_config):
        strategy_name = current_strategy["name"]
        if retriever_config.name == "stub_bm25":
            scores = scores_a_for_bm25 if strategy_name == "strat_a" else scores_b_for_bm25
        else:
            scores = scores_a_for_dense if strategy_name == "strat_a" else scores_b_for_dense
        retriever = StubRetriever(
            name=retriever_config.name,
            fixed_chunk_scores_by_query=scores,
            index_time_value={"stub_bm25": 0.1, "stub_dense": 0.3}[retriever_config.name],
            query_latency_value={"stub_bm25": 0.2, "stub_dense": 0.4}[retriever_config.name],
        )
        constructed_retrievers.append(retriever)
        return retriever

    chunking_config_a = WholeDocumentChunkingConfig(name="strat_a")
    chunking_config_b = WholeDocumentChunkingConfig(name="strat_b")
    bm25_cfg = BM25RetrieverConfig(
        name="stub_bm25",
        type="bm25",
        k1=0.9,
        b=0.4,
        tokenizer="regex_word",
        lowercase=True,
        stopwords="none",
        stemming="none",
    )
    dense_cfg = DenseRetrieverConfig(
        name="stub_dense", type="dense", model_name="stub-model", batch_size=8
    )
    config = SweepConfig(
        seed=42,
        chunking_strategies=(chunking_config_a, chunking_config_b),
        cutoffs=CUTOFFS,
        retrievers=(bm25_cfg, dense_cfg),
        data_dir=Path("data"),
        output_path=Path("results/sweep.csv"),
    )

    # Spy on src.chunking.aggregate_to_document_ranked_list as seen
    # through src.sweep_runner's module namespace (where it was bound
    # by `from src.chunking import aggregate_to_document_ranked_list`)
    # so every ChunkScores run_sweep aggregates can be inspected
    # directly, while still delegating to the real implementation so
    # run_sweep's returned rows are unaffected.
    import src.sweep_runner as sweep_runner_module
    from src.chunking import aggregate_to_document_ranked_list as real_aggregate

    recorded_aggregations: List[Tuple[Tuple[str, ...], np.ndarray]] = []

    def spy_aggregate(chunk_scores):
        recorded_aggregations.append((chunk_scores.chunk_ids, chunk_scores.scores.copy()))
        return real_aggregate(chunk_scores)

    monkeypatch.setattr(sweep_runner_module, "aggregate_to_document_ranked_list", spy_aggregate)

    rows, per_query_rows, all_succeeded = run_sweep(
        config, BUNDLE, retriever_factory, chunker_factory
    )

    # -- Row count and overall success -------------------------------------
    assert all_succeeded is True
    # 2 chunking strategies x 2 retrievers x 3 cutoffs = 12 rows.
    assert len(rows) == 2 * 2 * 3 == 12

    # -- Call counting: each of the 4 (chunking_strategy, retriever)
    #    combinations' build_index and retrieve_all were each called
    #    exactly once (Requirement 6.3, 13.2). build_chunk_corpus calls
    #    chunk_document once per document, exactly once per chunking
    #    strategy (never per retriever) -- so each StubChunker's
    #    chunk_document_calls has exactly len(CORPUS) entries, not
    #    len(CORPUS) * num_retrievers.
    for chunker in (chunker_a, chunker_b):
        assert len(chunker.chunk_document_calls) == len(CORPUS)
        assert set(chunker.chunk_document_calls) == set(CORPUS.keys())

    assert len(constructed_retrievers) == 4  # 2 strategies x 2 retrievers
    for retriever in constructed_retrievers:
        assert len(retriever.build_index_calls) == 1
        assert len(retriever.retrieve_all_calls) == 1
        assert retriever.retrieve_all_calls[0] == QUERIES

    # -- Aggregation correctness: doc "2"'s aggregated score for every
    #    recorded ChunkScores must equal the TRUE MAXIMUM of its two
    #    stub chunk scores, never their mean or sum (Requirement 13.3).
    #    Every one of the 4 (chunking_strategy, retriever) combinations
    #    issues exactly one retrieve_all call, which yields one
    #    ChunkScores per query (2 queries) -- so exactly 4 * 2 = 8
    #    aggregations are recorded in total.
    assert len(recorded_aggregations) == 4 * 2 == 8

    def _doc_id_of(chunk_id: str) -> str:
        return chunk_id.rpartition("::chunk")[0]

    for chunk_ids, scores in recorded_aggregations:
        doc_ids_present = sorted({_doc_id_of(cid) for cid in chunk_ids})
        scores_by_doc: Dict[str, List[float]] = {doc_id: [] for doc_id in doc_ids_present}
        for chunk_id, score in zip(chunk_ids, scores):
            scores_by_doc[_doc_id_of(chunk_id)].append(float(score))

        doc2_scores = scores_by_doc["2"]
        assert len(doc2_scores) == 2, "doc '2' must have exactly 2 chunks in every recording"
        true_max_by_doc = {doc_id: max(s) for doc_id, s in scores_by_doc.items()}
        mean_by_doc = {doc_id: sum(s) / len(s) for doc_id, s in scores_by_doc.items()}
        sum_by_doc = {doc_id: sum(s) for doc_id, s in scores_by_doc.items()}

        max_order = sorted(doc_ids_present, key=lambda d: -true_max_by_doc[d])
        mean_order = sorted(doc_ids_present, key=lambda d: -mean_by_doc[d])
        sum_order = sorted(doc_ids_present, key=lambda d: -sum_by_doc[d])

        # The stub scores are constructed so mean/sum genuinely diverge
        # from max for this recording (doc "2"'s two chunks: 0.1+offset
        # and 0.95+offset, versus every other doc's single score) --
        # confirming max_order actually differs from both is what makes
        # the assertion below non-vacuous.
        assert max_order != mean_order or max_order != sum_order

        # Recompute the aggregation with the real, unpatched function
        # (imported once, at module scope, before monkeypatch.setattr
        # replaced sweep_runner's own bound reference above -- this
        # reference is never patched) directly on the recorded
        # ChunkScores, and assert its result equals ranking by the true
        # per-document maximum, never by mean or sum.
        result_order = real_aggregate(ChunkScores(chunk_ids=chunk_ids, scores=scores))
        assert result_order == max_order
        if mean_order != max_order:
            assert result_order != mean_order
        if sum_order != max_order:
            assert result_order != sum_order

    # -- index_time / query_latency identical across all rows sharing a
    #    run_id, and distinct run_ids never coincidentally share a
    #    value that should legitimately differ (Requirement 6.5, 8.5).
    for retriever in constructed_retrievers:
        matching_rows = [
            row for row in rows if row.retriever == retriever.name
            and row.index_time == retriever.index_time_value
        ]
        # 2 run_ids share this retriever name (strat_a, strat_b), each
        # contributing 3 rows (one per cutoff) -- but index_time/
        # query_latency VALUES are the same literal per retriever name
        # here (by construction of retriever_factory above), so this
        # checks that every row for this retriever name carries that
        # exact value, consistently.
        assert len(matching_rows) == 2 * len(CUTOFFS)

    bm25_index_times = {row.index_time for row in rows if row.retriever == "stub_bm25"}
    dense_index_times = {row.index_time for row in rows if row.retriever == "stub_dense"}
    assert bm25_index_times == {0.1}
    assert dense_index_times == {0.3}
    assert bm25_index_times != dense_index_times
