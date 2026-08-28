"""Dense_Retriever: the dense baseline built on `sentence-transformers`
(Requirement 4).

Loads the configured model (`all-MiniLM-L6-v2` for session 1) on
`device="cpu"` only (Requirement 4.2), with weights cached under a
path under `data/` (Requirement 4.3). Builds exactly one embedding
index over the whole-document corpus in `build_index` (Requirement
4.1, 4.5) into L2-normalized embeddings, so cosine similarity reduces
to a dot product, and produces exactly one Ranked_List per query in a
single `retrieve_all` call (Requirement 4.4), ranking by brute-force
exact dot product over every corpus embedding -- no ANN index -- tied
broken by the same `doc_id_sort_key` ascending-document-ID rule
`BM25Retriever` uses (Requirement 4.8), per `design.md`'s
`src/retrievers/dense_retriever.py` section.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
from sentence_transformers import SentenceTransformer

from src.config import DenseRetrieverConfig
from src.errors import ModelLoadError
from src.retrievers.base import doc_id_sort_key


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
        texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
            for doc_id in self._doc_ids
        ]
        start = time.perf_counter()
        self._doc_embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return time.perf_counter() - start

    def retrieve_all(
        self, queries: Dict[str, str], top_k: int
    ) -> Tuple[Dict[str, List[str]], float]:
        """Encodes all queries in one batched `model.encode(...)` call,
        then ranks every corpus document per query by brute-force, exact
        cosine similarity -- a dot product, since both query and corpus
        embeddings are L2-normalized -- computed with `numpy` over the
        full corpus embedding matrix, with no ANN index (Requirement
        4.4). For each query, ranks by
        `(-similarity, doc_id_sort_key(doc_id))`, using the same shared
        `doc_id_sort_key` helper `BM25Retriever` uses, so ties are broken
        by ascending numeric document ID identically across both
        retrievers (Requirement 4.8). Takes the top
        `min(top_k, corpus size)` document IDs (Requirement 5.2). Times
        query encoding plus similarity computation together as
        query_latency (Requirement 4.6).
        """
        if self._doc_embeddings is None:
            raise RuntimeError("build_index must be called before retrieve_all")

        effective_top_k = min(top_k, len(self._doc_ids))
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
        # Brute-force exact cosine similarity over every corpus document:
        # both sides are L2-normalized, so the dot product IS the cosine
        # similarity -- no approximation, no ANN index (Requirement 4.4).
        similarity_matrix = query_embeddings @ self._doc_embeddings.T

        ranked_lists: Dict[str, List[str]] = {}
        for row_idx, qid in enumerate(query_ids):
            scores = similarity_matrix[row_idx]
            scored_doc_ids = sorted(
                zip(self._doc_ids, scores),
                key=lambda pair: (-pair[1], doc_id_sort_key(pair[0])),
            )
            ranked_lists[qid] = [
                doc_id for doc_id, _score in scored_doc_ids[:effective_top_k]
            ]
        query_latency = time.perf_counter() - start
        return ranked_lists, query_latency
