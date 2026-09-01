"""BM25_Retriever: the lexical baseline built on `rank_bm25.BM25Okapi`
(Requirement 3).

Builds exactly one `BM25Okapi` index over the corpus it is handed in
`build_index` (Requirement 3.1) -- a whole-document corpus or a chunk
corpus, indistinguishably, since this class has no awareness that
chunking occurred (`src/chunking.py`'s `build_chunk_corpus` produces
the same `Dict[str, Dict[str, str]]` shape either way). Documents and
queries are tokenized identically by the shared `_tokenize` method
(Requirement 3.2, 3.3), using the tokenizer and case-handling settings
declared in the Sweep_Config -- `stopwords` and `stemming` are already
validated to be `"none"` at config-load time (`src/config.py`), so no
filtering/stemming step exists in this implementation, per
`design.md`'s `src/retrievers/bm25_retriever.py` section.

`retrieve_all` streams: it yields one `(query_id, ChunkScores)` pair
per query, at Full_Chunk_Depth (every indexed chunk scored, for every
query -- Requirement 6.1, 6.4), never sorted (Requirement 5.7) --
`aggregate_to_document_ranked_list` (`src/chunking.py`) is the only
sort in the whole chunk-to-cutoff path.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Iterator, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.config import BM25RetrieverConfig
from src.retrievers.base import ChunkScores

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

    def retrieve_all(self, queries: Dict[str, str]) -> Iterator[Tuple[str, ChunkScores]]:
        """Yields one `(query_id, ChunkScores)` pair per query, in
        `queries`' own iteration order, at Full_Chunk_Depth -- every
        indexed chunk scored, for every query (Requirement 6.1, 6.4).
        No top_k, no slice, no sort: `rank_bm25.BM25Okapi.get_scores()`
        already returns a numpy array aligned to `chunk_id_index`,
        scoring every indexed chunk, so nothing is truncated or
        reordered here (Requirement 5.7). The same `chunk_id_index`
        tuple object, fixed once at `build_index` time, is reused
        unmodified across every yield. `self.last_query_latency` is
        set only after the generator is fully exhausted, to the summed
        wall-clock time actually spent scoring queries (Requirement
        6.3).
        """
        if self._bm25 is None:
            raise RuntimeError("build_index must be called before retrieve_all")

        chunk_id_index = tuple(self._doc_ids)
        total_latency = 0.0
        for qid, query_text in queries.items():
            start = time.perf_counter()
            tokenized_query = self._tokenize(query_text)
            scores = self._bm25.get_scores(tokenized_query)
            total_latency += time.perf_counter() - start
            yield qid, ChunkScores(chunk_ids=chunk_id_index, scores=scores)
        self.last_query_latency = total_latency
