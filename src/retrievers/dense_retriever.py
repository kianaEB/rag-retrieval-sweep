"""Dense_Retriever: the dense baseline built on `sentence-transformers`
(Requirement 4).

Loads the configured model (`all-MiniLM-L6-v2` for session 1) on
`device="cpu"` only (Requirement 4.2), with weights cached under a
path under `data/` (Requirement 4.3). Builds exactly one embedding
index over the whole-document corpus in `build_index` (Requirement
4.1, 4.5) into L2-normalized embeddings, so cosine similarity reduces
to a dot product.

`retrieve_all` streams one `ChunkScores` per query -- every indexed
chunk, scored, at Full_Chunk_Depth, via a single call
(Requirements 5.7, 6.1, 6.3, 6.4). There is no `top_k` parameter and no
sort: ranking and depth-slicing happen downstream, after
Max_Aggregation (`src/chunking.py`), not here. Ranking still uses
brute-force exact dot product over every corpus embedding -- no ANN
index -- and the `doc_id_sort_key` ascending-document-ID tie-break
`BM25Retriever` uses is applied later, by the aggregation step, not by
this retriever.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy
from sentence_transformers import SentenceTransformer

from src.config import DenseRetrieverConfig
from src.errors import ModelLoadError
from src.retrievers.base import ChunkScores


def format_document_text(doc: Dict[str, str]) -> str:
    """Formats a single corpus document as `title + " " + text`, the
    same `title`/`text` concatenation used to build the embedding index
    in `DenseRetriever.build_index`. Extracted so
    `src/token_length_analysis.py` can tokenize the exact same text the
    dense retriever encodes, rather than re-deriving it -- there is
    exactly one implementation, called from both places."""
    return f"{doc.get('title', '')} {doc.get('text', '')}"


class DenseRetriever:
    """Dense baseline retriever implementing the `Retriever` protocol
    (`src/retrievers/base.py`)."""

    def __init__(self, config: DenseRetrieverConfig, cache_folder: Path) -> None:
        self.name = config.name
        self._config = config
        # Populated by build_index; retrieve_all raises RuntimeError if
        # called first, the same contract BM25Retriever enforces.
        self._doc_ids: List[str] = []
        self._doc_embeddings: Optional[numpy.ndarray] = None

        cache_folder = Path(cache_folder)
        try:
            self._model = SentenceTransformer(
                config.model_name,
                cache_folder=str(cache_folder),
                # Hard-coded, never conditional on CUDA availability
                # (Requirement 4.2) -- the CPU-only constraint holds even
                # on a machine where a GPU happens to be present.
                device="cpu",
            )
        except Exception as exc:
            raise ModelLoadError(
                f"failed to load model {config.model_name!r} with cache_folder "
                f"{cache_folder}: {exc}"
            ) from exc

        # Second, independent layer of defense in depth for Requirement
        # 10.5. The first layer is the deferred-import ordering enforced
        # by src/sweep_runner.py's configure_caches() call (Task 13/14),
        # which this class cannot see or enforce on its own --
        # huggingface_hub resolves HF_HUB_CACHE from the environment once,
        # at its own import time. If the resolved value doesn't match the
        # cache_folder this retriever was constructed with, cache
        # configuration either ran too late (after huggingface_hub's own
        # import) or pointed somewhere else, and the model weights just
        # downloaded/loaded from an unintended location under -- or
        # outside -- data/.
        import huggingface_hub.constants as huggingface_hub_constants

        resolved_cache = Path(huggingface_hub_constants.HF_HUB_CACHE).resolve()
        expected_cache = cache_folder.resolve()
        if resolved_cache != expected_cache:
            raise ModelLoadError(
                f"huggingface_hub resolved its cache to {resolved_cache}, "
                f"expected {expected_cache}; configure_caches() either ran "
                f"too late (after huggingface_hub's own import) or was "
                f"called with a different data_dir (Requirement 10.5)"
            )

    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> float:
        """Encodes `title + " " + text` for every corpus document in one
        batched `model.encode(...)` call, into L2-normalized embeddings
        so cosine similarity reduces to a dot product in `retrieve_all`
        (Requirement 4.1). Returns the wall-clock duration of this
        encode call, in seconds, as index_time -- excluding any query
        encoding time, which happens later in `retrieve_all`
        (Requirement 4.5).
        """
        self._doc_ids = list(corpus.keys())
        texts = [format_document_text(corpus[doc_id]) for doc_id in self._doc_ids]
        start = time.perf_counter()
        self._doc_embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return time.perf_counter() - start

    def retrieve_all(self, queries: Dict[str, str]) -> Iterator[Tuple[str, ChunkScores]]:
        """Encodes all queries in one batched `model.encode(...)` call
        up front -- cheap, `query_count x embedding_dim`, not the memory
        concern -- then, for each query in turn, computes its full
        chunk-score vector via a single mat-vec (`self._doc_embeddings @
        query_embeddings[row_idx]`), never materializing the full query
        x chunk similarity matrix. Yields `(qid, ChunkScores(chunk_ids,
        scores))` per query, lazily, at Full_Chunk_Depth (Requirement
        6.1) -- every corpus embedding, scored, with no ANN index
        (Requirement 4.4) and no truncation (Requirement 6.4). The same
        `chunk_id_index` tuple object is reused, unmodified, across every
        yield (Requirement 5.7). Times query encoding plus the per-query
        mat-vec loop together as `total_latency`, only setting
        `self.last_query_latency` once the loop -- and therefore the
        returned generator -- has been fully exhausted (Requirement 5.7).
        """
        if self._doc_embeddings is None:
            raise RuntimeError("build_index must be called before retrieve_all")

        chunk_id_index: Tuple[str, ...] = tuple(self._doc_ids)
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        start = time.perf_counter()
        query_embeddings = self._model.encode(
            query_texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        total_latency = time.perf_counter() - start

        for row_idx, qid in enumerate(query_ids):
            mat_vec_start = time.perf_counter()
            # Brute-force exact cosine similarity for this one query,
            # against every corpus chunk: both sides are L2-normalized,
            # so the dot product IS the cosine similarity -- no
            # approximation, no ANN index (Requirement 4.4). A single
            # mat-vec, never the full query x chunk matrix at once.
            scores = self._doc_embeddings @ query_embeddings[row_idx]
            total_latency += time.perf_counter() - mat_vec_start
            yield qid, ChunkScores(chunk_ids=chunk_id_index, scores=scores)

        self.last_query_latency = total_latency
