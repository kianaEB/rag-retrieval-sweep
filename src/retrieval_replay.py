"""Retrieval_Replay: read-only reuse of session 1's frozen retriever,
config loader, and corpus loader to obtain Retrieved_Context for the
Generation_Subset (Requirement 3).

This module adds no new retrieval logic. It imports
`src.config.load_sweep_config`, `src.corpus_loader.configure_caches`/
`load_scifact`, `src.retrievers.base.Retriever`, and both concrete
retriever classes, but constructs only the ONE retriever type that
matches the Replayed_Run's name -- never both, the same "exactly one
matched type" discipline `make_default_retriever_factory` uses
per-retriever-config in session 1, narrowed here to a single lookup
instead of a loop over two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

from src.config import BM25RetrieverConfig, DenseRetrieverConfig, SweepConfig, load_sweep_config
from src.corpus_loader import CorpusBundle, configure_caches, load_scifact
from src.errors import (
    ConfigError,
    CorpusLoadError,
    CorpusValidationError,
    FrozenRetrieverConfigError,
    RetrievalReplayError,
)
from src.retrievers.base import Retriever
from src.retrievers.dense_retriever import format_document_text


def _retriever_name_from_run_id(run_id: str) -> str:
    """Extracts the retriever name prefix from a run_id formed as
    f"{retriever_config.name}__{chunking_strategy}" (the exact
    convention src/sweep_runner.py already uses). Splits on the first
    "__" only, since a retriever name itself never contains "__" in
    this repo's fixed grid."""
    return run_id.split("__", 1)[0]


def load_frozen_retriever_config(
    sweep_config_path: Path, replayed_run_id: str
) -> Tuple[SweepConfig, Union[BM25RetrieverConfig, DenseRetrieverConfig]]:
    """Loads configs/sweep.yaml via load_sweep_config, then returns the
    one retriever config entry whose `name` matches `replayed_run_id`'s
    prefix before "__" (Requirement 3.3: reused exactly as declared, no
    field modified).

    Raises `FrozenRetrieverConfigError` if configs/sweep.yaml cannot be
    loaded (wrapping the underlying `ConfigError`), or if no retriever
    config entry's `name` matches (Requirement 3.6).
    """
    try:
        sweep_config = load_sweep_config(sweep_config_path)
    except ConfigError as exc:
        raise FrozenRetrieverConfigError(
            f"failed to load the Frozen_Retriever_Config's source file "
            f"{sweep_config_path}: {exc}"
        ) from exc
    target_name = _retriever_name_from_run_id(replayed_run_id)
    for retriever_config in sweep_config.retrievers:
        if retriever_config.name == target_name:
            return sweep_config, retriever_config
    raise FrozenRetrieverConfigError(
        f"no retriever named {target_name!r} (parsed from replayed_run_id "
        f"{replayed_run_id!r}) is declared in {sweep_config_path}"
    )


def build_frozen_retriever(
    sweep_config: SweepConfig,
    retriever_config: Union[BM25RetrieverConfig, DenseRetrieverConfig],
) -> Tuple[Retriever, CorpusBundle]:
    """Constructs the ONE matched retriever type, loads the real corpus
    via configure_caches + load_scifact (a genuine, non-mocked corpus
    load -- the Frozen_Retriever_Config needs a real index to replay
    against), and builds that retriever's index exactly once over the
    full loaded corpus (Requirement 3.1).

    Raises `FrozenRetrieverConfigError` if the corpus fails to load
    (wrapping CorpusLoadError/CorpusValidationError). Raises
    `RetrievalReplayError` if index construction fails.
    """
    configure_caches(sweep_config.data_dir)
    try:
        bundle, _report = load_scifact(sweep_config.data_dir)
    except (CorpusLoadError, CorpusValidationError) as exc:
        raise FrozenRetrieverConfigError(
            f"Retrieval_Replay could not load the corpus needed to "
            f"rebuild the Frozen_Retriever_Config's index: {exc}"
        ) from exc

    if isinstance(retriever_config, BM25RetrieverConfig):
        from src.retrievers.bm25_retriever import BM25Retriever

        retriever: Retriever = BM25Retriever(retriever_config)
    else:
        from src.retrievers.dense_retriever import DenseRetriever

        retriever = DenseRetriever(
            retriever_config, cache_folder=sweep_config.data_dir / "hf_cache"
        )

    try:
        retriever.build_index(bundle.corpus)
    except Exception as exc:
        raise RetrievalReplayError(
            f"index construction failed for {retriever.name}: {exc}"
        ) from exc

    return retriever, bundle


def replay_retrieval(
    retriever: Retriever,
    bundle: CorpusBundle,
    subset_query_ids: List[str],
    queries: Dict[str, str],
    replay_top_k: int,
) -> Dict[str, List[str]]:
    """Issues exactly ONE `retrieve_all` call, with `queries` limited to
    `subset_query_ids` only (Requirement 3.5), at `top_k=replay_top_k`,
    and returns `{query_id: Retrieved_Context}` -- each Retrieved_Context
    is the ordered list of document texts (via `format_document_text`,
    preserving retrieval-rank order) corresponding to that query's
    ranked document IDs (Requirement 3.2).

    Calling this once for the whole subset, rather than once per
    query_id, is what satisfies both Requirement 3.1 ("rather than
    rebuilding ... separately for each Generation_Subset query") and
    Requirement 3.5 ("SHALL NOT issue a retrieval call for any query
    outside the Generation_Subset") simultaneously: there is exactly
    one call, and its `queries` argument is pre-filtered to the subset
    before the call is made, not filtered from a larger result
    afterward.

    Raises `RetrievalReplayError` naming the query count and
    replay_top_k (best effort; the underlying retriever's
    `retrieve_all` does not fail per-query, so a raised exception here
    fails the entire subset) if retrieval fails.
    """
    subset_queries = {qid: queries[qid] for qid in subset_query_ids}
    try:
        ranked_lists, _query_latency = retriever.retrieve_all(
            subset_queries, top_k=replay_top_k
        )
    except Exception as exc:
        raise RetrievalReplayError(
            f"retrieval failed for the Generation_Subset "
            f"({len(subset_query_ids)} queries) at "
            f"replay_top_k={replay_top_k}: {exc}"
        ) from exc

    retrieved_context: Dict[str, List[str]] = {}
    for qid, doc_ids in ranked_lists.items():
        retrieved_context[qid] = [
            format_document_text(bundle.corpus[doc_id]) for doc_id in doc_ids
        ]
    return retrieved_context
