"""BM25_Retriever: the lexical baseline built on `rank_bm25.BM25Okapi`
(Requirement 3).

Builds exactly one `BM25Okapi` index over the whole-document corpus in
`build_index` (Requirement 3.1) and produces exactly one Ranked_List
per query in a single `retrieve_all` call (Requirement 3.4). Documents
and queries are tokenized identically by the shared `_tokenize` method
(Requirement 3.2, 3.3), using the tokenizer and case-handling settings
declared in the Sweep_Config -- `stopwords` and `stemming` are already
validated to be `"none"` at config-load time (`src/config.py`), so no
filtering/stemming step exists in this implementation, per
`design.md`'s `src/retrievers/bm25_retriever.py` section.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.config import BM25RetrieverConfig
from src.retrievers.base import doc_id_sort_key

# The only supported tokenizer for session 1 (Requirement 2.3, 3.7,
# enforced at config-load time by src/config.py): split on runs of
# word characters.
_WORD_PATTERN = re.compile(r"\w+")


class BM25Retriever:
    """Lexical baseline retriever implementing the `Retriever` protocol
    (`src/retrievers/base.py`)."""

    def __init__(self, config: BM25RetrieverConfig) -> None:
        self.name = config.name
        self._config = config
        self._bm25: Optional[BM25Okapi] = None
        # Doc IDs in the exact order passed to BM25Okapi, so a score at
        # position i in get_scores()'s returned array can be matched
        # back to the doc_id it belongs to.
        self._doc_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Applies the config's `tokenizer` (only `"regex_word"` is
        supported: split on runs of word characters via `re.findall`),
        then lowercases every token if `config.lowercase` is true.
        Applied identically to every document and every query
        (Requirement 3.2, 3.3) -- this is the single normalization
        pipeline for both.
        """
        tokens = _WORD_PATTERN.findall(text)
        if self._config.lowercase:
            tokens = [token.lower() for token in tokens]
        return tokens

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        """Tokenizes the whole corpus once (`title + " " + text` per
        document) and constructs exactly one `BM25Okapi` index
        (Requirement 3.1). Returns the wall-clock duration of this
        build, in seconds, as index_time (Requirement 3.5).
        """
        start = time.perf_counter()
        self._doc_ids = list(corpus.keys())
        tokenized_corpus = [
            self._tokenize(
                f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
            )
            for doc_id in self._doc_ids
        ]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self._config.k1, b=self._config.b)
        return time.perf_counter() - start

    def retrieve_all(
        self, queries: Dict[str, str], top_k: int
    ) -> Tuple[Dict[str, List[str]], float]:
        """Runs one BM25 scoring pass per query, across all queries, in
        a single retrieval run (Requirement 3.4). For each query,
        ranks documents by `(-score, doc_id_sort_key(doc_id))` --
        descending BM25 score, ties broken by ascending numeric
        document ID -- so the Ranked_List order is identical across
        repeated runs on the same corpus and query. Takes the top
        `min(top_k, corpus size)` document IDs (Requirement 5.2).
        Returns `(ranked_lists, query_latency_seconds)`, timing the
        entire per-query loop once (Requirement 3.6).
        """
        if self._bm25 is None:
            raise RuntimeError("build_index must be called before retrieve_all")

        effective_top_k = min(top_k, len(self._doc_ids))
        start = time.perf_counter()
        ranked_lists: Dict[str, List[str]] = {}
        for qid, query_text in queries.items():
            tokenized_query = self._tokenize(query_text)
            scores = self._bm25.get_scores(tokenized_query)
            scored_doc_ids = sorted(
                zip(self._doc_ids, scores),
                key=lambda pair: (-pair[1], doc_id_sort_key(pair[0])),
            )
            ranked_lists[qid] = [doc_id for doc_id, _score in scored_doc_ids[:effective_top_k]]
        query_latency = time.perf_counter() - start
        return ranked_lists, query_latency
