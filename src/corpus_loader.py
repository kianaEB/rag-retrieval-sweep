"""Corpus_Loader: BEIR SciFact loading, cache configuration, and the
`CorpusBundle`/`CorpusLoadReport` containers (Requirement 1).

`configure_caches(data_dir)` sets `HF_HOME`/`HF_HUB_CACHE` to a path
under `data_dir` *before* any `huggingface_hub`-importing module is
ever imported in the process -- `huggingface_hub` resolves these
environment variables once, at its own import time, so setting them
afterward has no effect (see `src/retrievers/dense_retriever.py`'s
cache-path assertion, which independently verifies this ordering held
at run time).

This module's own module-level imports are limited to the standard
library and `src.errors` -- `beir` is imported lazily, inside
`load_scifact` itself, rather than at module top. This means merely
importing `src.corpus_loader` (e.g. for `CorpusBundle`,
`CorpusLoadReport`, or `configure_caches`) never imports `beir`, and
`beir`'s own module tree (`beir.util`, `beir.datasets.data_loader`)
only ever gets imported at the moment `load_scifact` is actually
called -- which is always after a caller has already run
`configure_caches()` in the intended orchestration order
(`src/sweep_runner.py`). This mirrors the same deferred-import
discipline `DenseRetriever.__init__` uses for `huggingface_hub.constants`
(`src/retrievers/dense_retriever.py`): keep order-sensitive imports out
of module top, so import ordering is enforced by *when a function
runs*, not by the physical position of an `import` statement relative
to some other module's top-level code.

`load_scifact(data_dir)` downloads (if not already cached) and loads
the BEIR SciFact corpus, test queries, and qrels in the standard BEIR
format, derives all three counts directly from the loaded data
structures, verifies every qrels-referenced document ID and query ID
resolves against the loaded corpus/query set, and returns
`(CorpusBundle, CorpusLoadReport)`. Every count in `CorpusLoadReport`
is derived from the loaded structures at load time, never a literal
written in source code (Requirement 1.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from src.errors import CorpusLoadError, CorpusValidationError

if TYPE_CHECKING:  # pragma: no cover - import only needed for type hints
    from beir.datasets.data_loader import GenericDataLoader

# The single BEIR dataset this Corpus_Loader loads for session 1
# (Requirement 1.1). Fixed here, not user-configurable -- SciFact is
# named directly in docs/PROJECT_BRIEF.md and product.md, not
# something the Sweep_Config chooses.
_SCIFACT_DATASET_NAME = "scifact"
_BEIR_DOWNLOAD_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)


@dataclass(frozen=True)
class CorpusBundle:
    """The corpus documents, test queries, and qrels loaded for a run.

    `corpus` maps doc_id -> {"title": ..., "text": ...}; `queries` maps
    query_id -> query text; `qrels` maps query_id -> {doc_id: graded
    relevance score}. This shape is identical whether `bundle` was
    produced by the real `load_scifact` (this module) or hand-built as
    in-memory literals for a test (Requirement 12.3).
    """

    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class CorpusLoadReport:
    """The exact counts `load_scifact` loaded, derived from the loaded
    data structures at load time (Requirement 1.3) -- never a literal
    written in source code. `as_log_line()` is the single
    deterministic, machine-parsable report line (Requirement 1.2) a
    caller (the Sweep_Runner's `main()`, or a verification script)
    prints to stdout after a successful `load_scifact` call.
    """

    num_documents: int
    num_queries: int
    num_qrel_pairs: int

    def as_log_line(self) -> str:
        return (
            f"CORPUS_LOAD_REPORT documents={self.num_documents} "
            f"queries={self.num_queries} qrel_pairs={self.num_qrel_pairs}"
        )


def configure_caches(data_dir: Path) -> None:
    """Sets `HF_HOME` and `HF_HUB_CACHE` to `data_dir / "hf_cache"`.

    Must be called before `src.retrievers.dense_retriever` -- or any
    other module that imports `huggingface_hub`, directly or
    transitively -- is imported anywhere in the process; see this
    module's docstring. `TRANSFORMERS_CACHE` is deprecated by
    `huggingface_hub` in favor of `HF_HUB_CACHE` and is intentionally
    not set (Requirement 4.3, 10.5).
    """
    cache_dir = Path(data_dir) / "hf_cache"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)


def _describe_partial_load(loader: "GenericDataLoader") -> str:
    """Best-effort identification of which of corpus/queries/qrels was
    still empty at the point `loader.load()` raised, for inclusion in
    a `CorpusLoadError` message (Requirement 1.4). These are
    diagnostic-only partial counts attached to an *error* -- never the
    Criterion 2 count report, which is only ever produced on success
    (Requirement 1.7).
    """
    empty_parts = []
    if not loader.corpus:
        empty_parts.append("corpus documents")
    if not loader.queries:
        empty_parts.append("test queries")
    if not loader.qrels:
        empty_parts.append("qrels")
    partial_counts = (
        f"partial counts at failure: corpus={len(loader.corpus)}, "
        f"queries={len(loader.queries)}, qrels={len(loader.qrels)}"
    )
    if not empty_parts:
        return f"failed after corpus/queries/qrels each had non-zero partial counts ({partial_counts})"
    return f"failed to load: {', '.join(empty_parts)} ({partial_counts})"


def _validate_referential_integrity(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
) -> None:
    """Raises `CorpusValidationError` if any qrels-referenced document
    ID is absent from `corpus`, or any qrels query ID is absent from
    `queries` (Requirement 1.6) -- catching a partial or truncated
    download that loaded successfully (no exception, all three
    non-empty) but is nonetheless missing entries that qrels reference.
    """
    unresolved_doc_refs = 0
    unresolved_query_refs = 0
    for query_id, judged_docs in qrels.items():
        if query_id not in queries:
            unresolved_query_refs += 1
        for doc_id in judged_docs:
            if doc_id not in corpus:
                unresolved_doc_refs += 1
    if unresolved_doc_refs or unresolved_query_refs:
        raise CorpusValidationError(
            f"qrels reference {unresolved_doc_refs} document ID(s) not present "
            f"in the loaded corpus and {unresolved_query_refs} query ID(s) not "
            "present in the loaded query set"
        )


def load_scifact(data_dir: Path) -> Tuple[CorpusBundle, CorpusLoadReport]:
    """Downloads (if not already cached under `data_dir`) and loads
    BEIR SciFact's test-split corpus, queries, and qrels, then returns
    `(CorpusBundle, CorpusLoadReport)`.

    Raises `CorpusLoadError` if the download/unzip fails, if loading
    any of corpus/queries/qrels raises, or if any of the three loads
    empty (Requirement 1.4, 1.5). Raises `CorpusValidationError` if any
    qrels-referenced document ID or query ID does not resolve against
    the loaded corpus/query set (Requirement 1.6). Never returns a
    `CorpusLoadReport` unless every one of these checks has passed
    (Requirement 1.7, 1.8) -- every count in the returned report is
    derived from the loaded dict structures, never a literal
    (Requirement 1.3).

    `beir` is imported here, lazily, rather than at module top (see
    this module's docstring) -- `data_dir` is always the caller's
    resolved `SweepConfig.data_dir`, never a hard-coded path, so every
    download this function triggers lands under `data_dir`
    (Requirement 10.5).
    """
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    data_dir = Path(data_dir)
    url = _BEIR_DOWNLOAD_URL_TEMPLATE.format(dataset=_SCIFACT_DATASET_NAME)

    try:
        data_path = beir_util.download_and_unzip(url, str(data_dir))
    except Exception as exc:
        raise CorpusLoadError(
            f"failed to download/unzip BEIR {_SCIFACT_DATASET_NAME!r} from "
            f"{url} to {data_dir}: none of corpus documents, test queries, "
            f"or qrels were loaded: {exc}"
        ) from exc

    loader = GenericDataLoader(data_folder=data_path)
    try:
        corpus, queries, qrels = loader.load(split="test")
    except Exception as exc:
        raise CorpusLoadError(
            f"failed to load BEIR {_SCIFACT_DATASET_NAME!r} from {data_path}: "
            f"{_describe_partial_load(loader)}; underlying error: {exc}"
        ) from exc

    if not corpus:
        raise CorpusLoadError(
            f"BEIR {_SCIFACT_DATASET_NAME!r} corpus documents loaded empty from {data_path}"
        )
    if not queries:
        raise CorpusLoadError(
            f"BEIR {_SCIFACT_DATASET_NAME!r} test queries loaded empty from {data_path}"
        )
    if not qrels:
        raise CorpusLoadError(
            f"BEIR {_SCIFACT_DATASET_NAME!r} qrels loaded empty from {data_path}"
        )

    _validate_referential_integrity(corpus, queries, qrels)

    try:
        report = CorpusLoadReport(
            num_documents=len(corpus),
            num_queries=len(queries),
            num_qrel_pairs=sum(len(judged_docs) for judged_docs in qrels.values()),
        )
    except Exception as exc:
        # The count report is the only detector of a silent truncation
        # or partial download (Requirement 1.8): if deriving it fails
        # for any reason, that failure is fatal, the same as any other
        # CorpusLoadError, even though loading itself already succeeded.
        raise CorpusLoadError(
            f"failed to derive the count report for {data_path}: {exc}"
        ) from exc

    bundle = CorpusBundle(corpus=corpus, queries=queries, qrels=qrels)
    return bundle, report
