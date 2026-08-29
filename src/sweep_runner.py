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

from src.config import (
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    SweepConfig,
    load_sweep_config,
)
from src.corpus_loader import CorpusBundle, configure_caches, load_scifact
from src.errors import (
    ConfigError,
    CorpusLoadError,
    CorpusValidationError,
    ReportWriteError,
    SeedApplicationError,
    ZeroQualifyingQueriesError,
)
from src.metrics import (
    mean_over_qualifying_queries,
    mrr_at_10,
    ndcg_at_10,
    recall_at_k,
    scored_query_count,
)
from src.report import MISSING, SweepReportRow, write_run_config_record, write_sweep_report
from src.retrievers.base import Retriever
from src.seeding import apply_seed

RetrieverConfig = Union[BM25RetrieverConfig, DenseRetrieverConfig]

# The only construction path for a Retriever inside `run_sweep`. `main()`
# supplies `make_default_retriever_factory(...)`, which builds real
# `BM25Retriever`/`DenseRetriever` instances; `test_orchestration.py`
# (Task 9) supplies a factory that returns `StubRetriever` instances.
RetrieverFactory = Callable[[RetrieverConfig], Retriever]

# Default Sweep_Config path (Requirement 10.1): applied when `--config`
# is not passed on the command line.
DEFAULT_CONFIG_PATH = Path("configs/sweep.yaml")


def run_sweep(
    config: SweepConfig,
    bundle: CorpusBundle,
    retriever_factory: RetrieverFactory,
) -> Tuple[List[SweepReportRow], bool]:
    """Runs the retriever loop, metric computation, and row assembly.

    For each retriever declared in `config.retrievers`, this calls
    `retriever_factory(retriever_config)` exactly once, then
    `build_index` exactly once, then `retrieve_all` exactly once, at
    `top_k = max(config.cutoffs)` (Deepest_Cutoff) -- never more than
    one `build_index` call and one `retrieve_all` call per retriever,
    regardless of how many cutoffs are declared (Requirement 5.1,
    5.2, 5.5). The single `Ranked_List` per query that `retrieve_all`
    returns is sliced to each declared cutoff `k` to compute that
    row's `recall_at_k`; `ndcg_at_10` and `mrr_at_10` are each computed
    once per retriever, at a fixed cutoff of 10, and copied unchanged
    into all of that retriever's rows (Requirement 6.6).

    `num_queries_total`/`num_queries_scored` are computed once, from
    `bundle.queries`/`bundle.qrels` alone, before the retriever loop
    runs (Requirement 6.7) -- if zero queries qualify (no loaded query
    has a Qrels-judged relevant document), this raises
    `ZeroQualifyingQueriesError` before any index build, since every
    retriever would hit the identical zero-denominator condition.

    A retriever whose `build_index`/`retrieve_all` call raises does
    not halt the run: that retriever's `run_id` gets all of its rows
    written with the `MISSING` ("NA") sentinel for `index_time`,
    `query_latency`, `recall_at_k`, `ndcg_at_10`, and `mrr_at_10`, and
    the loop proceeds to the next retriever (Requirement 7.6, 7.8,
    10.4). Similarly, a failure computing `ndcg_at_10`/`mrr_at_10`
    marks only those two cells `MISSING` for that retriever's rows
    without affecting `recall_at_k` or timing, and a failure computing
    `recall_at_k` for one cutoff marks only that row's `recall_at_k`
    cell `MISSING` without affecting the other cutoffs.

    Returns `(rows, all_succeeded)`: `rows` always has exactly
    `len(config.retrievers) * len(config.cutoffs)` entries, regardless
    of any individual failure (Requirement 7.1, 7.2); `all_succeeded`
    is `True` iff no row carries a `MISSING` marker in any column.
    """
    deepest_cutoff = max(config.cutoffs)

    # Computed once, from corpus/qrels data alone -- never per-retriever
    # (Requirement 6.7).
    num_queries_total, num_queries_scored = scored_query_count(
        bundle.queries.keys(), bundle.qrels
    )
    if num_queries_scored == 0:
        raise ZeroQualifyingQueriesError(
            "no loaded query has a Qrels-judged relevant document; every "
            "retriever would hit the same zero-denominator condition, so "
            "the run halts here, before any index build (Requirements "
            "6.1, 6.2, 6.3, 10.4)"
        )

    rows: List[SweepReportRow] = []
    all_succeeded = True

    for retriever_config in config.retrievers:
        # Same run_id iff same retriever + chunking strategy (Requirement
        # 7.4); with exactly one chunking strategy declared in session 1,
        # this is equivalent to retriever identity. Computed before the
        # try block: run_id itself never fails, and every row of this
        # iteration -- including an all-MISSING row on total failure --
        # needs it.
        run_id = f"{retriever_config.name}__{config.chunking_strategy}"

        try:
            # The only construction path for a retriever (Requirement
            # 12.1, 12.4): never `BM25Retriever(...)`/`DenseRetriever(...)`
            # called directly here. Construction is inside this try block
            # so a retriever's own __init__ failure (e.g. ModelLoadError)
            # degrades only this run_id's rows, the same as a
            # build_index/retrieve_all failure.
            retriever = retriever_factory(retriever_config)
            index_time = retriever.build_index(bundle.corpus)
            ranked_lists, query_latency = retriever.retrieve_all(
                bundle.queries, top_k=deepest_cutoff
            )
        except Exception:
            # RetrievalError-equivalent: this run_id's rows all get
            # MISSING for every metric/timing cell; the run proceeds to
            # the next retriever rather than halting (Requirement 7.6,
            # 7.8, 10.4).
            all_succeeded = False
            for k in config.cutoffs:
                rows.append(
                    SweepReportRow(
                        run_id=run_id,
                        retriever=retriever_config.name,
                        chunking_strategy=config.chunking_strategy,
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

        # ndcg_at_10 / mrr_at_10: computed once per run_id, at a fixed
        # cutoff of 10, from the full ranked_lists -- never inside the
        # k-loop below (Requirement 6.6).
        try:
            per_query_ndcg: Dict[str, float] = {
                qid: ndcg_at_10(ranked_list, bundle.qrels.get(qid, {}))
                for qid, ranked_list in ranked_lists.items()
            }
            per_query_mrr: Dict[str, float] = {
                qid: mrr_at_10(ranked_list, bundle.qrels.get(qid, {}))
                for qid, ranked_list in ranked_lists.items()
            }
            run_ndcg_at_10: Union[float, str] = mean_over_qualifying_queries(
                per_query_ndcg, bundle.qrels
            )
            run_mrr_at_10: Union[float, str] = mean_over_qualifying_queries(
                per_query_mrr, bundle.qrels
            )
        except Exception:
            # MetricComputationError-equivalent, scoped to this run_id's
            # ndcg/mrr cells only -- recall_at_k and timing are unaffected.
            all_succeeded = False
            run_ndcg_at_10 = MISSING
            run_mrr_at_10 = MISSING

        for k in config.cutoffs:
            try:
                # recall_at_k is recomputed per cutoff by slicing the same
                # ranked_lists object returned by the single retrieve_all
                # call above -- no additional retrieval call is issued
                # here (Requirement 5.3, 5.4).
                per_query_recall: Dict[str, float] = {
                    qid: recall_at_k(ranked_list, bundle.qrels.get(qid, {}), k)
                    for qid, ranked_list in ranked_lists.items()
                }
                row_recall_at_k: Union[float, str] = mean_over_qualifying_queries(
                    per_query_recall, bundle.qrels
                )
            except Exception:
                # MetricComputationError-equivalent, scoped to only this
                # row's recall_at_k cell.
                all_succeeded = False
                row_recall_at_k = MISSING

            rows.append(
                SweepReportRow(
                    run_id=run_id,
                    retriever=retriever_config.name,
                    chunking_strategy=config.chunking_strategy,
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

    return rows, all_succeeded


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
       (Requirement 10.4).
    9. Return 0 only if every row `run_sweep` produced has no
       `MISSING` cell (`all_succeeded is True`); otherwise return
       non-zero, even though `results/sweep.csv` was written in full
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
    try:
        rows, all_succeeded = run_sweep(config, bundle, retriever_factory)
    except ZeroQualifyingQueriesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        write_sweep_report(rows, config.output_path)
    except ReportWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not all_succeeded:
        print(
            f"WARNING: one or more retriever/cutoff combinations failed; "
            f"see 'NA' markers in {config.output_path}",
            file=sys.stderr,
        )
        return 1

    print(f"Sweep complete: wrote {len(rows)} rows to {config.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
