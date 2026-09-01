"""Sweep_Runner: the `run_sweep` orchestration seam and the `main()`
CLI entry point.

`run_sweep` (Task 8) is the per-retriever loop, metric computation, and
row assembly described in `design.md`'s "Components and Interfaces"
section, steps 5-7. It is the injection seam for Requirement 12: it
never constructs a retriever directly, only via the injected
`retriever_factory`, so `tests/test_orchestration.py` (Task 9) can
drive the exact same loop with a `StubRetriever` and an in-memory
`CorpusBundle`, with no config file, no cache setup, no seed, and no
network access anywhere on the call path.

`main()` (Task 14) is the CLI entry point that wires everything else
around that seam: config parsing, `configure_caches`, `apply_seed`,
`load_scifact`, writing `results/run_config.json` (which embeds the
`CorpusLoadReport` counts `load_scifact` just returned, so the record
requires a successful corpus load to exist), and finally `run_sweep`
itself, driven by `make_default_retriever_factory` (which builds real
`BM25Retriever`/`DenseRetriever` instances -- the only place in
production code that constructs either). `main()` never constructs a
retriever directly either; it only ever passes a factory into
`run_sweep`.

This module's own top-level imports are limited to `src.config`,
`src.corpus_loader`, `src.errors`, `src.metrics`, `src.report`,
`src.retrievers.base`, and `src.seeding` -- never `beir`,
`sentence-transformers`, or `huggingface_hub` directly, and never
`src.retrievers.bm25_retriever` or `src.retrievers.dense_retriever`.
`src.corpus_loader` imports `beir` lazily, inside `load_scifact`
itself, rather than at its own module top (see that module's
docstring), so importing `configure_caches`/`load_scifact` by name at
this module's top -- as this module does -- never triggers `beir`;
only *calling* `load_scifact(...)` does, and `main()` only does that
in step 4, after `configure_caches(...)` has already run in step 2.

`src.retrievers.dense_retriever`, by contrast, imports
`sentence_transformers` (and transitively `huggingface_hub`) at its
*own* module top -- so that module's import must itself be deferred
until after `configure_caches(...)` has run, not merely the call to a
function inside it. `make_default_retriever_factory` below defers that
import into its inner `factory` closure for exactly this reason: the
closure is only ever invoked from `run_sweep`, which `main()` calls
after step 2, so `src.retrievers.dense_retriever` -- and therefore
`huggingface_hub` -- is never imported anywhere in this process before
`configure_caches` has pointed `HF_HOME`/`HF_HUB_CACHE` at
`config.data_dir` (Requirement 10.5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from src.chunking import (
    Chunker,
    FixedWindowChunker,
    SentenceWindowChunker,
    WholeDocumentChunker,
    aggregate_to_document_ranked_list,
    build_chunk_corpus,
    load_chunking_tokenizer,
)
from src.config import (
    BM25RetrieverConfig,
    ChunkingStrategyConfig,
    DenseRetrieverConfig,
    FixedWindowChunkingConfig,
    SentenceWindowChunkingConfig,
    SUPPORTED_CUTOFFS,
    SweepConfig,
    WholeDocumentChunkingConfig,
    load_sweep_config,
)
from src.corpus_loader import CorpusBundle, configure_caches, load_scifact
from src.errors import (
    ChunkingError,
    ConfigError,
    CorpusLoadError,
    CorpusValidationError,
    PerQueryReportError,
    ReportWriteError,
    SeedApplicationError,
    ZeroQualifyingQueriesError,
)
from src.metrics import (
    judged_relevant_docs,
    mean_over_qualifying_queries,
    mrr_at_10,
    ndcg_at_10,
    recall_at_k,
    scored_query_count,
)
from src.per_query_report import PerQueryReportRow, write_per_query_report
from src.report import MISSING, SweepReportRow, write_run_config_record, write_sweep_report
from src.retrievers.base import Retriever
from src.seeding import apply_seed

RetrieverConfig = Union[BM25RetrieverConfig, DenseRetrieverConfig]

# The only construction path for a Retriever inside `run_sweep`. `main()`
# supplies `make_default_retriever_factory(...)`, which builds real
# `BM25Retriever`/`DenseRetriever` instances; `test_orchestration.py`
# (Task 9) supplies a factory that returns `StubRetriever` instances.
RetrieverFactory = Callable[[RetrieverConfig], Retriever]

# The only construction path for a Chunker inside `run_sweep`, mirroring
# `RetrieverFactory` above (design.md's "chunker-factory seam" section).
# `main()` supplies `make_default_chunker_factory(...)`;
# `tests/test_orchestration.py` (Task 17) supplies a factory that
# returns `StubChunker` instances.
ChunkerFactory = Callable[[ChunkingStrategyConfig], Chunker]

# Default Sweep_Config path (Requirement 10.1): applied when `--config`
# is not passed on the command line.
DEFAULT_CONFIG_PATH = Path("configs/sweep.yaml")

def run_sweep(
    config: SweepConfig,
    bundle: CorpusBundle,
    retriever_factory: RetrieverFactory,
    chunker_factory: ChunkerFactory,
) -> Tuple[List[SweepReportRow], List[PerQueryReportRow], bool]:
    """Runs the chunk-once/retrieve-once/aggregate-once loop, metric
    computation, and row assembly, over the full 9-combination grid
    (3 `config.chunking_strategies` x 3 `config.retrievers`).

    **Outer loop over `config.chunking_strategies`** (3 iterations):
    for each declared Chunking_Strategy, this calls
    `chunker_factory(chunking_config)` exactly once and
    `build_chunk_corpus(chunker, bundle.corpus)` exactly once, caching
    the resulting chunk corpus in a local variable for the duration of
    that iteration -- never rebuilt per retriever. If
    `build_chunk_corpus` raises `ChunkingError` (a Chunker produced
    zero Chunks for some document), every retriever's 4 rows for this
    Chunking_Strategy (12 rows total) get the `MISSING` sentinel, and
    the run proceeds to the next Chunking_Strategy -- the other two
    strategies' 24 rows are unaffected (Requirement 2.6).

    **Inner loop over `config.retrievers`** (3 iterations each, 9
    combinations total): for each retriever, this calls
    `retriever_factory(retriever_config)` exactly once, then
    `build_index(chunk_corpus)` exactly once, then `retrieve_all(
    bundle.queries)` exactly once -- never more than one `build_index`
    call and one `retrieve_all` call per (retriever, Chunking_Strategy)
    combination, regardless of how many cutoffs are declared
    (Requirement 6.1, 6.2, 6.3, 6.5). `retrieve_all` returns a single
    generator yielding one `(query_id, ChunkScores)` pair at a time, at
    Full_Chunk_Depth; this loop consumes that one generator, applying
    `aggregate_to_document_ranked_list` per query to reduce each
    query's `ChunkScores` to a `Document_Ranked_List` -- Max_Aggregation
    (Requirement 5.1, 5.6), never a second retrieval call. Reading
    `retriever.last_query_latency` only after the generator is fully
    exhausted matches the streaming contract's own timing guarantee.

    Everything from there on -- `ndcg_at_10`/`mrr_at_10` computed once
    per run_id at a fixed cutoff of 10, `recall_at_k` sliced per
    declared cutoff, the `MetricComputationError`-equivalent recovery
    tiers, and the `PerQueryReportRow` emission gate -- is unchanged
    from session-1/significance-testing, now operating on
    `document_ranked_lists` (this function's own local dict, built from
    `aggregate_to_document_ranked_list`'s output) instead of a
    retriever's own direct `ranked_lists`. `run_id =
    f"{retriever_config.name}__{chunking_config.name}"` for every row.

    `num_queries_total`/`num_queries_scored` are computed once, from
    `bundle.queries`/`bundle.qrels` alone, before either loop runs
    (Requirement 6.7 restated) -- if zero queries qualify, this raises
    `ZeroQualifyingQueriesError` before any chunk corpus is built.

    Returns `(rows, per_query_rows, all_succeeded)`: `rows` always has
    exactly `len(config.chunking_strategies) * len(config.retrievers) *
    len(config.cutoffs)` entries, regardless of any individual failure
    (Requirement 7.1 restated for the 9-combination grid);
    `all_succeeded` is `True` iff no row carries a `MISSING` marker in
    any column.
    """
    deepest_cutoff = max(config.cutoffs)
    emit_per_query_rows = set(config.cutoffs) == set(SUPPORTED_CUTOFFS)

    # Computed once, from corpus/qrels data alone -- never per-combination.
    num_queries_total, num_queries_scored = scored_query_count(
        bundle.queries.keys(), bundle.qrels
    )
    if num_queries_scored == 0:
        raise ZeroQualifyingQueriesError(
            "no loaded query has a Qrels-judged relevant document; every "
            "retriever would hit the same zero-denominator condition, so "
            "the run halts here, before any chunk corpus is built "
            "(Requirements 6.1, 6.2, 6.3, 10.4)"
        )

    rows: List[SweepReportRow] = []
    per_query_rows: List[PerQueryReportRow] = []
    all_succeeded = True

    for chunking_config in config.chunking_strategies:
        try:
            chunker = chunker_factory(chunking_config)
            chunk_corpus = build_chunk_corpus(chunker, bundle.corpus)
        except ChunkingError:
            # New failure tier (full-grid-chunking-sweep spec): a
            # chunk-corpus build failure is scoped to every run_id
            # sharing this chunking_config -- all 3 retrievers' 4 rows
            # each (12 rows) get MISSING, the same smallest-affected-
            # unit principle session-1 already applies to a single
            # retriever's build_index/retrieve_all failure, generalized
            # to this new per-strategy failure boundary (Requirement
            # 2.6). The other 2 chunking strategies' 24 rows are
            # unaffected and the run proceeds.
            all_succeeded = False
            for retriever_config in config.retrievers:
                run_id = f"{retriever_config.name}__{chunking_config.name}"
                for k in config.cutoffs:
                    rows.append(
                        SweepReportRow(
                            run_id=run_id,
                            retriever=retriever_config.name,
                            chunking_strategy=chunking_config.name,
                            k=k,
                            recall_at_k=MISSING,
                            ndcg_at_10=MISSING,
                            mrr_at_10=MISSING,
                            index_time=MISSING,
                            query_latency=MISSING,
                            num_queries_total=num_queries_total,
                            num_queries_scored=num_queries_scored,
                        )
                    )
            continue

        for retriever_config in config.retrievers:
            # Same run_id iff same retriever + Chunking_Strategy
            # (Requirement 7.4-equivalent). Computed before the try
            # block: run_id itself never fails, and every row of this
            # iteration -- including an all-MISSING row on total
            # failure -- needs it.
            run_id = f"{retriever_config.name}__{chunking_config.name}"

            try:
                # The only construction path for a retriever
                # (Requirement 12.1, 12.4): never
                # `BM25Retriever(...)`/`DenseRetriever(...)` called
                # directly here. Construction is inside this try block
                # so a retriever's own __init__ failure (e.g.
                # ModelLoadError) degrades only this run_id's rows, the
                # same as a build_index/retrieve_all failure.
                retriever = retriever_factory(retriever_config)
                index_time = retriever.build_index(chunk_corpus)
                document_ranked_lists: Dict[str, List[str]] = {}
                for qid, chunk_scores in retriever.retrieve_all(bundle.queries):
                    document_ranked_lists[qid] = aggregate_to_document_ranked_list(
                        chunk_scores
                    )
                query_latency = retriever.last_query_latency
            except Exception:
                # RetrievalError-equivalent: this run_id's rows all get
                # MISSING for every metric/timing cell; the run
                # proceeds to the next retriever rather than halting
                # (Requirement 7.6, 7.8, 10.4).
                all_succeeded = False
                for k in config.cutoffs:
                    rows.append(
                        SweepReportRow(
                            run_id=run_id,
                            retriever=retriever_config.name,
                            chunking_strategy=chunking_config.name,
                            k=k,
                            recall_at_k=MISSING,
                            ndcg_at_10=MISSING,
                            mrr_at_10=MISSING,
                            index_time=MISSING,
                            query_latency=MISSING,
                            num_queries_total=num_queries_total,
                            num_queries_scored=num_queries_scored,
                        )
                    )
                continue

            # ndcg_at_10 / mrr_at_10: computed once per run_id, at a
            # fixed cutoff of 10, from the full document_ranked_lists
            # -- never inside the k-loop below (Requirement 6.6).
            ndcg_mrr_succeeded = True
            try:
                per_query_ndcg: Dict[str, float] = {
                    qid: ndcg_at_10(ranked_list, bundle.qrels.get(qid, {}))
                    for qid, ranked_list in document_ranked_lists.items()
                }
                per_query_mrr: Dict[str, float] = {
                    qid: mrr_at_10(ranked_list, bundle.qrels.get(qid, {}))
                    for qid, ranked_list in document_ranked_lists.items()
                }
                run_ndcg_at_10: Union[float, str] = mean_over_qualifying_queries(
                    per_query_ndcg, bundle.qrels
                )
                run_mrr_at_10: Union[float, str] = mean_over_qualifying_queries(
                    per_query_mrr, bundle.qrels
                )
            except Exception:
                # MetricComputationError-equivalent, scoped to this
                # run_id's ndcg/mrr cells only -- recall_at_k and
                # timing are unaffected.
                all_succeeded = False
                ndcg_mrr_succeeded = False
                per_query_ndcg = {}
                per_query_mrr = {}
                run_ndcg_at_10 = MISSING
                run_mrr_at_10 = MISSING

            # Per-cutoff per-query recall dicts, collected across the
            # k-loop below (significance-testing spec, Requirement
            # 1.1, 1.4) -- the exact same per_query_recall dict each
            # iteration already builds on its way to row_recall_at_k's
            # mean, kept keyed by cutoff so the Per_Query_Report row
            # assembly after the loop can read all four cutoffs'
            # per-query values without recomputing any of them.
            per_query_recall_by_cutoff: Dict[int, Dict[str, float]] = {}
            recall_succeeded_by_cutoff: Dict[int, bool] = {}

            for k in config.cutoffs:
                try:
                    # recall_at_k is recomputed per cutoff by slicing
                    # the same document_ranked_lists object built from
                    # the single retrieve_all call above -- no
                    # additional retrieval call is issued here
                    # (Requirement 6.1, 6.2).
                    per_query_recall: Dict[str, float] = {
                        qid: recall_at_k(ranked_list, bundle.qrels.get(qid, {}), k)
                        for qid, ranked_list in document_ranked_lists.items()
                    }
                    row_recall_at_k: Union[float, str] = mean_over_qualifying_queries(
                        per_query_recall, bundle.qrels
                    )
                    per_query_recall_by_cutoff[k] = per_query_recall
                    recall_succeeded_by_cutoff[k] = True
                except Exception:
                    # MetricComputationError-equivalent, scoped to only
                    # this row's recall_at_k cell.
                    all_succeeded = False
                    row_recall_at_k = MISSING
                    recall_succeeded_by_cutoff[k] = False

                rows.append(
                    SweepReportRow(
                        run_id=run_id,
                        retriever=retriever_config.name,
                        chunking_strategy=chunking_config.name,
                        k=k,
                        recall_at_k=row_recall_at_k,
                        ndcg_at_10=run_ndcg_at_10,
                        mrr_at_10=run_mrr_at_10,
                        index_time=index_time,
                        query_latency=query_latency,
                        num_queries_total=num_queries_total,
                        num_queries_scored=num_queries_scored,
                    )
                )

            # Per_Query_Report row assembly (significance-testing spec,
            # Requirement 1): only when every metric this run_id needs
            # -- ndcg_at_10, mrr_at_10, and recall_at_k for all four
            # fixed cutoffs -- computed successfully, and the declared
            # cutoff set is exactly {1, 5, 10, 20} (PerQueryReportRow's
            # wide-on-cutoff schema). PerQueryReportRow carries no
            # missing-value sentinel, so a run_id with any failure
            # above contributes zero per-query rows rather than a row
            # with a fabricated cell.
            if emit_per_query_rows and ndcg_mrr_succeeded and all(
                recall_succeeded_by_cutoff.get(cutoff, False) for cutoff in SUPPORTED_CUTOFFS
            ):
                for qid, ranked_list in document_ranked_lists.items():
                    qrels_for_query = bundle.qrels.get(qid, {})
                    per_query_rows.append(
                        PerQueryReportRow(
                            run_id=run_id,
                            retriever=retriever_config.name,
                            chunking_strategy=chunking_config.name,
                            query_id=qid,
                            recall_at_1=per_query_recall_by_cutoff[1][qid],
                            recall_at_5=per_query_recall_by_cutoff[5][qid],
                            recall_at_10=per_query_recall_by_cutoff[10][qid],
                            recall_at_20=per_query_recall_by_cutoff[20][qid],
                            ndcg_at_10=per_query_ndcg[qid],
                            mrr_at_10=per_query_mrr[qid],
                            num_judged_relevant=len(judged_relevant_docs(qrels_for_query)),
                        )
                    )

    return rows, per_query_rows, all_succeeded


def make_default_chunker_factory(cache_folder: Path) -> ChunkerFactory:
    """Production `ChunkerFactory` used by `main()`, mirroring
    `make_default_retriever_factory` below.

    For `WholeDocumentChunkingConfig`, returns `WholeDocumentChunker()`
    immediately -- no tokenizer load, ever. For `fixed_window`/
    `sentence_window`, loads the all-MiniLM-L6-v2 tokenizer via
    `load_chunking_tokenizer` at most once (memoized in the
    `tokenizer_cache` closure variable), only the first time either is
    actually requested -- never eagerly at factory-construction time,
    so a `SweepConfig` declaring only `whole_document` (as in a
    stub-based test) never triggers a tokenizer load or network
    access.
    """
    tokenizer_cache: Dict[str, object] = {}

    def factory(chunking_config: ChunkingStrategyConfig) -> Chunker:
        if isinstance(chunking_config, WholeDocumentChunkingConfig):
            return WholeDocumentChunker()
        if "tokenizer" not in tokenizer_cache:
            tokenizer_cache["tokenizer"] = load_chunking_tokenizer(cache_folder)
        tokenizer = tokenizer_cache["tokenizer"]
        if isinstance(chunking_config, FixedWindowChunkingConfig):
            return FixedWindowChunker(
                tokenizer, chunking_config.window_size, chunking_config.stride
            )
        if isinstance(chunking_config, SentenceWindowChunkingConfig):
            return SentenceWindowChunker(
                tokenizer,
                chunking_config.sentences_per_chunk,
                chunking_config.max_chunk_tokens,
            )
        raise ConfigError(f"unsupported chunking config type: {type(chunking_config)!r}")

    return factory


def make_default_retriever_factory(cache_folder: Path) -> RetrieverFactory:
    """Production `RetrieverFactory` used by `main()`.

    Closes over `cache_folder` (always `config.data_dir / "hf_cache"` --
    the same path `configure_caches()` pointed `HF_HOME`/`HF_HUB_CACHE`
    at) so `DenseRetriever` always receives that same cache path.
    `BM25Retriever` and `DenseRetriever` are both imported inside the
    inner `factory` closure below, not at this function's own top and
    never at `sweep_runner.py`'s module top -- so merely calling
    `make_default_retriever_factory(...)` to build the closure does not
    yet import either retriever module; only actually *invoking* the
    returned `factory(retriever_config)` -- which `run_sweep` does,
    once per retriever, after `main()`'s `configure_caches` call has
    already run -- does.
    """

    def factory(retriever_config: RetrieverConfig) -> Retriever:
        if isinstance(retriever_config, BM25RetrieverConfig):
            from src.retrievers.bm25_retriever import BM25Retriever

            return BM25Retriever(retriever_config)
        if isinstance(retriever_config, DenseRetrieverConfig):
            from src.retrievers.dense_retriever import DenseRetriever

            return DenseRetriever(retriever_config, cache_folder=cache_folder)
        raise ConfigError(
            f"unsupported retriever config type: {type(retriever_config)!r}"
        )

    return factory


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.sweep_runner",
        description=(
            "Runs the session-1 baseline sweep end-to-end: loads the "
            "Sweep_Config, loads BEIR SciFact, builds and queries every "
            "declared retriever, computes recall@k/nDCG@10/MRR@10 "
            "against the qrels, and writes results/sweep.csv."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the Sweep_Config YAML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: `python -m src.sweep_runner [--config PATH]`.

    Orchestration, in order (`design.md`'s `src/sweep_runner.py`
    section):

    1. Parse `--config` (default `configs/sweep.yaml`) and load it via
       `load_sweep_config`. On `ConfigError` (including its
       `UnsupportedPreprocessingError` subclass): print the error,
       return non-zero, write nothing (Requirement 2.6, 10.4).
    2. `configure_caches(config.data_dir)` -- before any module that
       imports `huggingface_hub` (transitively) is imported anywhere
       in this process (Requirement 10.5).
    3. `apply_seed(config.seed)`. On `SeedApplicationError`: print the
       error, return non-zero, write nothing (Requirement 8.5).
    4. `load_scifact(config.data_dir)`. On `CorpusLoadError` /
       `CorpusValidationError`: print the error, return non-zero, write
       nothing (Requirement 1.4, 1.5, 1.6, 1.8). On success, print
       `report.as_log_line()` (Requirement 1.2), then write
       `results/run_config.json` (alongside `config.output_path`) via
       `write_run_config_record(config, load_report, run_config_path)`
       -- embedding this run's actual `CorpusLoadReport` counts
       (`num_documents`, `num_queries`, `num_qrel_pairs`) in the record
       alongside the seed, the resolved config, and installed package
       versions (Requirement 8.3), so the dataset statistics a reader
       needs to cite live in a committed artifact rather than only in
       transient stdout. On `ReportWriteError`: print the error,
       return non-zero (Requirement 10.4).
    5-7. `run_sweep(config, bundle, retriever_factory=
       make_default_retriever_factory(config.data_dir / "hf_cache"))`.
       On `ZeroQualifyingQueriesError`: print the error, return
       non-zero, write nothing (Requirement 6.1, 6.2, 6.3, 10.4).
    8. `write_sweep_report(rows, config.output_path)`, unconditionally
       once `run_sweep` returns rows -- even if some rows carry a
       `MISSING` marker, per Requirement 7's row-count guarantee. On
       `ReportWriteError`: print the error, return non-zero
       (Requirement 10.4). Then
       `write_per_query_report(per_query_rows, config.output_path.parent
       / "per_query.csv")` (significance-testing spec, Requirement 1.1)
       -- in this same run, no second retrieval, no separate entry
       point. On `PerQueryReportError`: print the error, return
       non-zero, leaving `results/per_query.csv` absent or in its
       pre-run state (Requirement 1.8) -- even though
       `results/sweep.csv` was already written successfully; the two
       files' write outcomes are independent, but a per_query.csv
       failure still fails the run.
    9. Return 0 only if every row `run_sweep` produced has no
       `MISSING` cell (`all_succeeded is True`); otherwise return
       non-zero, even though `results/sweep.csv` (and
       `results/per_query.csv`, if it was written) was written in full
       (Requirement 10.3, 10.4).
    """
    args = _parse_args(argv)

    try:
        config = load_sweep_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: failed to load Sweep_Config from {args.config}: {exc}", file=sys.stderr)
        return 1

    # Must run before any module that imports huggingface_hub
    # (transitively) is imported anywhere in this process -- see this
    # module's docstring and src/corpus_loader.py's.
    configure_caches(config.data_dir)

    try:
        apply_seed(config.seed)
    except SeedApplicationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        bundle, load_report = load_scifact(config.data_dir)
    except (CorpusLoadError, CorpusValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(load_report.as_log_line())

    run_config_path = config.output_path.parent / "run_config.json"
    try:
        write_run_config_record(config, load_report, run_config_path)
    except ReportWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    retriever_factory = make_default_retriever_factory(config.data_dir / "hf_cache")
    chunker_factory = make_default_chunker_factory(config.data_dir / "hf_cache")
    try:
        rows, per_query_rows, all_succeeded = run_sweep(
            config, bundle, retriever_factory, chunker_factory
        )
    except ZeroQualifyingQueriesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        write_sweep_report(rows, config.output_path)
    except ReportWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    per_query_path = config.output_path.parent / "per_query.csv"
    try:
        write_per_query_report(per_query_rows, per_query_path)
    except PerQueryReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not all_succeeded:
        print(
            f"WARNING: one or more retriever/cutoff combinations failed; "
            f"see 'NA' markers in {config.output_path}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Sweep complete: wrote {len(rows)} rows to {config.output_path} and "
        f"{len(per_query_rows)} rows to {per_query_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
