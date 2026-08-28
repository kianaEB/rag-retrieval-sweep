"""Shared Retriever interface: the `Retriever` Protocol, the
`RetrievalRun` dataclass, and the `doc_id_sort_key` tie-break helper.

Both `BM25Retriever` (`src/retrievers/bm25_retriever.py`) and
`DenseRetriever` (`src/retrievers/dense_retriever.py`) implement the
`Retriever` protocol defined here, so the Sweep_Runner's orchestration
loop (`src/sweep_runner.py`) is retriever-agnostic: it never issues
more than one `build_index` call and one `retrieve_all` call per
retriever, per `design.md`'s "index once, retrieve once, slice four
ways" data flow.

`doc_id_sort_key` is defined once, here, and shared by both retrievers
so the ascending-document-ID tie-break (Requirements 3.4, 4.8) uses a
uniform comparison type in both implementations -- a numeric ID is
never compared against a non-numeric one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple, Union


@dataclass(frozen=True)
class RetrievalRun:
    """The result of one retriever's single index-build-and-retrieval
    run: one Ranked_List per query, plus the two timing measurements
    the Sweep_Runner copies unchanged into every Sweep_Report row
    sharing this run's run_id (Requirements 5.6, 5.7)."""

    run_id: str
    retriever_name: str
    ranked_lists: Dict[str, List[str]]  # query_id -> ordered doc IDs, len == top_k
    index_time: float  # seconds
    query_latency: float  # seconds


class Retriever(Protocol):
    """The interface the Sweep_Runner's orchestration loop calls
    against. `BM25Retriever` and `DenseRetriever` both implement this
    protocol, as does the test-only `StubRetriever`
    (`tests/test_orchestration.py`, Requirement 12)."""

    name: str

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        """Builds the index once. Returns index_time in seconds."""
        ...

    def retrieve_all(
        self, queries: Dict[str, str], top_k: int
    ) -> Tuple[Dict[str, List[str]], float]:
        """Runs retrieval once for all queries at top_k. Returns
        (ranked_lists, query_latency_seconds); each Ranked_List
        contains min(top_k, corpus size) document IDs, so a corpus
        smaller than top_k yields all corpus documents rather than
        raising or padding (Requirement 5.2)."""
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
