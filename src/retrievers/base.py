"""Shared Retriever interface: the `ChunkScores` dataclass, the
streaming `Retriever` Protocol, and the `doc_id_sort_key` tie-break
helper.

Both `BM25Retriever` (`src/retrievers/bm25_retriever.py`) and
`DenseRetriever` (`src/retrievers/dense_retriever.py`) implement the
`Retriever` protocol defined here, so the Sweep_Runner's orchestration
loop (`src/sweep_runner.py`) is retriever-agnostic: it never issues
more than one `build_index` call and one `retrieve_all` call per
retriever, per `design.md`'s "index once, retrieve once, aggregate
once, slice four ways" data flow.

`retrieve_all` returns a generator yielding one `(query_id,
ChunkScores)` pair at a time, at Full_Chunk_Depth (every chunk in the
index, scored, for every query -- Requirement 6.1, 6.4). There is no
`top_k` parameter: retrieval depth is a slicing decision made later,
downstream of `aggregate_to_document_ranked_list`
(`src/chunking.py`), not something a retriever truncates to.

`doc_id_sort_key` is defined once, here, and shared by both retrievers
so the ascending-document-ID tie-break (Requirements 3.4, 4.8, 5.5)
uses a uniform comparison type in both implementations -- a numeric ID
is never compared against a non-numeric one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Protocol, Tuple, Union

import numpy


@dataclass(frozen=True)
class ChunkScores:
    """One query's score against every chunk in a retriever's index.
    Never sorted: `aggregate_to_document_ranked_list`
    (`src/chunking.py`) selects by score value, not rank position, so
    no chunk-level order is produced or needed.

    `chunk_ids` is fixed once at `build_index` time and the SAME tuple
    object is reused, unmodified, for every query a single
    `retrieve_all` call yields -- it is never rebuilt or copied per
    query."""

    chunk_ids: Tuple[str, ...]  # stable order; identical object across every yield in one run
    scores: numpy.ndarray  # shape (len(chunk_ids),); scores[i] is chunk_ids[i]'s score


class Retriever(Protocol):
    """The interface the Sweep_Runner's orchestration loop calls
    against. `BM25Retriever` and `DenseRetriever` both implement this
    protocol, as does the test-only `StubRetriever`
    (`tests/test_orchestration.py`, Requirement 12)."""

    name: str
    last_query_latency: float  # populated once retrieve_all's generator is exhausted

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        """Builds the index once. Returns index_time in seconds."""
        ...

    def retrieve_all(self, queries: Dict[str, str]) -> Iterator[Tuple[str, ChunkScores]]:
        """A single call to retrieve_all(...) returns ONE generator
        object -- this is what "exactly one retrieval call"
        (Requirement 6.3) means: the property counts how many times
        this method is invoked and how many times the retriever scores
        its corpus, not how many items the resulting generator
        produces when consumed. Fully iterating one generator over
        every query is still the single call session-1's "index once,
        retrieve once" property already counted, restated in chunk
        terms -- not one call per query.

        Yields exactly one (query_id, ChunkScores) pair per query in
        `queries`, in `queries`' own iteration order, lazily: each
        ChunkScores is computed against every chunk in the index
        (Full_Chunk_Depth, Requirement 6.1) one query at a time, so
        peak memory for consuming this call is one query's score
        vector, never every query's score vector held at once.

        There is no top_k parameter (Requirement 6.4): every chunk is
        scored for every query regardless; nothing is truncated or
        dropped here. Depth-slicing happens downstream, after
        Max_Aggregation.

        After the returned generator is fully exhausted,
        `last_query_latency` holds the summed wall-clock time actually
        spent scoring queries (excluding whatever time the caller
        spends between yields, e.g. in aggregation) -- the
        generator-based restatement of session-1's query_latency
        semantics (Requirement 5.7)."""
        ...


def doc_id_sort_key(doc_id: str) -> Tuple[int, Union[int, str]]:
    """Sort key for the ascending-document-ID tie-break (Requirements
    3.4, 4.8). Numeric IDs (SciFact's normal case) sort before
    non-numeric IDs, and within each group the comparison is uniform
    -- int-to-int or str-to-str, never int-to-str -- so mixing a
    numeric and a non-numeric ID in the same corpus can never raise
    TypeError."""
    try:
        return (0, int(doc_id))
    except ValueError:
        return (1, doc_id)
