"""Significance_Analyzer: paired bootstrap, paired permutation test,
Holm-Bonferroni multiple-comparison adjustment, and the CLI entry point
that wires them together over the committed Per_Query_Report
(`significance-testing` spec, `design.md`'s `src/significance.py`
section).

The three pure functions -- `paired_bootstrap`, `permutation_test`, and
`holm_bonferroni` -- are the sole unit-under-test surface for
Requirement 7 (`tests/test_significance.py`). All three are pure: they
take numpy arrays (or, for `holm_bonferroni`, a plain list of floats)
and, where randomness is used, an injected `np.random.Generator`, and
return floats. None of them reads a file, touches global RNG state (no
`np.random.seed(...)`), or imports any retrieval/model code.

`main()` is the `Significance_Analyzer` CLI entry point
(`python -m src.significance [--config PATH]`) that reads
`results/per_query.csv`, runs the bootstrap/permutation/Holm-Bonferroni
functions above over it, and writes `results/significance.csv` plus a
merged `results/run_config.json`. Per Requirement 2.2, this module
performs no corpus loading, index building, retrieval, query encoding,
or model loading, and makes no network call: its top-level imports are
limited to `numpy`, `pandas`, `src.significance_config`, `src.errors`,
and (for the atomic writer and JSON serialization helper) `src.report`
-- never `src.corpus_loader`, `src.retrievers.*`, `src.seeding`, `beir`,
or `sentence-transformers`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas

from src.errors import (
    BootstrapConfigError,
    MissingReferenceRunError,
    ReportWriteError,
    RunConfigMergeError,
    SignificanceInputError,
    SignificanceWriteError,
)
from src.report import MISSING, _atomic_write_text, _json_default
from src.significance_config import SignificanceConfig, load_significance_config


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    resample_count: int,
    generator: np.random.Generator,
) -> Tuple[float, float, float]:
    """Paired bootstrap of `(a - b)` over the shared queries.

    Computes the per-query difference vector `d = a - b` exactly once
    (shape `(n,)`), then draws a single
    `generator.integers(0, n, size=(resample_count, n))` index matrix
    and resamples `d` directly: `resampled_means = d[idx].mean(axis=1)`.
    Resampling `d` directly is mathematically identical to
    `(a[idx] - b[idx]).mean(axis=1)` because indexing is elementwise,
    but it uses one temporary array instead of two and makes it
    structurally impossible to accidentally unpair `a` and `b` in a
    later edit -- there is only one array to resample, so every
    resample stays paired by construction (Requirement 3.2).

    Returns `(observed_mean_diff, ci_lower, ci_upper)`, where
    `observed_mean_diff = float(d.mean())` and `ci_lower` / `ci_upper`
    are the 2.5th / 97.5th percentiles of the resampled mean
    differences (Requirement 3.3). No Python-level per-resample loop is
    used: the full `(resample_count, n)` index matrix is drawn in one
    call and reduced with `.mean(axis=1)`.

    All randomness is drawn from the injected `generator` only; this
    function never touches `np.random`'s global state, so identical
    seeds passed via `generator` reproduce identical CI bounds exactly
    (Requirement 3.5, 3.7).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[0]
    idx = generator.integers(0, n, size=(resample_count, n))
    resampled_means = d[idx].mean(axis=1)
    ci_lower, ci_upper = np.percentile(resampled_means, [2.5, 97.5])
    observed_mean_diff = float(d.mean())
    return observed_mean_diff, float(ci_lower), float(ci_upper)


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    permutation_count: int,
    generator: np.random.Generator,
) -> float:
    """Paired permutation test for the two-sided p-value of `(a - b)`.

    Let `d = a - b` (the per-query difference vector, shape `(n,)`) and
    `observed = float(d.mean())`. Draws a single sign matrix of shape
    `(permutation_count, n)` via
    `generator.random((permutation_count, n)) < 0.5`, mapped to
    `{-1.0, +1.0}`, so that each paired query is independently
    sign-flipped with probability 0.5 in each permutation. Computes the
    permuted mean differences as `(signs * d).mean(axis=1)` -- a
    length-`permutation_count` vector -- then applies the add-one
    correction:

        count = int((np.abs(permuted) >= abs(observed)).sum())
        p = (count + 1) / (permutation_count + 1)

    where `count` is the number of permuted mean differences whose
    absolute value is `>=` the absolute observed mean difference
    (Requirement 3.4). The add-one correction gives the p-value a floor
    of `1 / (permutation_count + 1)`, so it is never exactly zero: a
    finite permutation sample cannot justify an exact zero, and the
    observed sign assignment is itself a valid draw under the null,
    counted in both the numerator and the denominator. No Python-level
    per-permutation loop is used.

    All randomness is drawn from the injected `generator` only; this
    function never touches `np.random`'s global state.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[0]
    observed = float(d.mean())
    signs = np.where(generator.random((permutation_count, n)) < 0.5, -1.0, 1.0)
    permuted = (signs * d).mean(axis=1)
    count = int((np.abs(permuted) >= abs(observed)).sum())
    return (count + 1) / (permutation_count + 1)


def holm_bonferroni(raw_p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment over a Comparison_Family.

    A pure function of the family's raw p-values and family size
    `m = len(raw_p_values)` alone (Requirement 5.4): sorts the raw
    p-values ascending; multiplies the p-value at ascending rank index
    `i` (from 0) by `(m - i)`; enforces monotonic non-decrease across
    the ascending order by taking the running maximum from the lowest
    rank upward (each adjusted value := max of itself and every
    lower-ranked adjusted value); clamps every adjusted value to
    `[0.0, 1.0]`; and maps the adjusted values back to the family's
    original input order.

    Ties in the raw p-values receive equal adjusted values, so the
    result does not depend on the input order of tied comparisons
    (Requirement 5.5). For `m == 1` the sole multiplier is `(m - 0) ==
    1`, so the adjusted value equals the raw value clamped to `[0, 1]`
    -- the identity (Requirement 5.6).
    """
    p = np.asarray(raw_p_values, dtype=float)
    m = p.shape[0]
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)
    adjusted_sorted = np.maximum.accumulate(sorted_p * multipliers)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Significance_Analyzer entry point (design.md's "src/significance.py"
# `main()` section). Everything below this line is the orchestration that
# consumes the three pure functions above; none of it is part of the
# Requirement 7 test surface.
# ---------------------------------------------------------------------------

# Requirement 1.3's Per_Query_Report columns -- the analyzer's sole input
# contract. Declared here (rather than imported from
# src.per_query_report) so this module's own import list stays exactly
# {numpy, pandas, src.significance_config, src.errors, src.report} per
# design.md, with no dependency on the sweep-side writer module.
REQUIRED_PER_QUERY_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "retriever",
    "chunking_strategy",
    "query_id",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "ndcg_at_10",
    "mrr_at_10",
    "num_judged_relevant",
)

# Fixed metric processing order (RNG discipline: bootstrap-then-
# permutation, per comparison, per metric, in exactly this order --
# see design.md's "RNG discipline" section). nDCG@10 is the
# Primary_Metric (Requirement 6.1) and is always first.
PRIMARY_METRIC = "ndcg_at_10"
METRIC_ORDER: Tuple[str, ...] = (
    "ndcg_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "mrr_at_10",
)

# NOT_APPLICABLE ("n/a"): the correction/verdict legitimately does not
# apply to this row (every secondary-metric row's p_value_adjusted and
# verdict, Requirement 5.1/6.3). Distinct from MISSING ("NA", imported
# from src.report): a value that could not be computed at all (the
# zero-shared-queries case, Requirement 3.8). The two sentinels are
# never conflated -- see design.md's "two distinct sentinels" section.
NOT_APPLICABLE = "n/a"

# Default Bootstrap_Config path (Requirement 4.1/4.7): applied when
# `--config` is not passed on the command line.
DEFAULT_CONFIG_PATH = Path("configs/significance.yaml")


@dataclass(frozen=True)
class SignificanceReportRow:
    """One row of `results/significance.csv`: one (comparison, metric)
    pair (design.md's `results/significance.csv` row schema). `metric`
    ranges over `METRIC_ORDER`; `is_primary` is `True` only for
    `ndcg_at_10`. `mean_diff`/`ci_lower`/`ci_upper`/`p_value_raw` are
    either a real float or the `MISSING` ("NA") sentinel (a
    zero-shared-queries comparison, Requirement 3.8). `p_value_adjusted`
    and `verdict` carry three possible states: a real float / verdict
    string for a computable nDCG@10 row, `MISSING` ("NA") for a
    zero-shared-queries nDCG@10 row, or `NOT_APPLICABLE` ("n/a") for
    every secondary-metric row (the Holm-Bonferroni correction and
    headline verdict apply to nDCG@10 alone, Requirement 5.1/6.3).
    `n_shared_queries` is never missing: `0` is a legitimate value and
    the row is still retained (Requirement 3.8, 6.5).
    """

    run_id: str
    retriever: str
    reference_run_id: str
    metric: str
    is_primary: bool
    mean_diff: Union[float, str]
    ci_lower: Union[float, str]
    ci_upper: Union[float, str]
    p_value_raw: Union[float, str]
    p_value_adjusted: Union[float, str]
    n_shared_queries: int
    verdict: str


def _read_run_config(path: Path) -> dict:
    """Reads and parses the run configuration record at `path` (default
    `results/run_config.json`).

    Raises `RunConfigMergeError` if `path` is absent, unreadable, not
    valid JSON, or not a JSON object -- the analyzer never creates a
    fresh record in place of a missing/unparsable one (Requirement 4.6).
    """
    path = Path(path)
    if not path.is_file():
        raise RunConfigMergeError(
            f"the sweep's run configuration record is missing: {path}"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigMergeError(
            f"the sweep's run configuration record is unreadable ({path}): {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise RunConfigMergeError(
            f"the sweep's run configuration record at {path} is not a JSON object"
        )
    return record


def _read_per_query_report(path: Path) -> pandas.DataFrame:
    """Reads `path` (default `results/per_query.csv`) via pandas.

    Raises `SignificanceInputError` naming the file, the parse failure,
    or the missing column(s) if `path` is absent, cannot be parsed, or
    lacks any of `REQUIRED_PER_QUERY_COLUMNS` (Requirement 1.3, 2.4).
    """
    path = Path(path)
    if not path.is_file():
        raise SignificanceInputError(f"per-query report file not found: {path}")
    try:
        frame = pandas.read_csv(path)
    except Exception as exc:
        raise SignificanceInputError(
            f"failed to parse per-query report {path}: {exc}"
        ) from exc
    missing_columns = [c for c in REQUIRED_PER_QUERY_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise SignificanceInputError(
            f"per-query report {path} is missing required column(s): {missing_columns}"
        )
    return frame


def _find_reference_run_id(frame: pandas.DataFrame, reference_run_id: str) -> str:
    """Returns `reference_run_id` after confirming it is present in
    `frame['run_id']`.

    Performs an EXACT match of `reference_run_id` against
    `frame['run_id'].unique()` and nothing else -- no filtering by
    retriever name, no sorting, no "take the first" rule. With three
    Chunking_Strategy entries now present, an implicit rule could
    silently select `bm25__fixed_window` instead of the intended
    `bm25__whole_document` (Requirement 9.3). Raises
    `MissingReferenceRunError` naming `reference_run_id` exactly if it
    is not present in `frame['run_id']` (Requirement 9.4).
    """
    if reference_run_id not in set(frame["run_id"].unique()):
        raise MissingReferenceRunError(
            f"the pinned Reference_Run '{reference_run_id}' is not present in "
            f"the per-query report; every comparison is defined relative to "
            f"the Reference_Run, and it is never inferred by sorting run_ids"
        )
    return reference_run_id


def _run_comparisons(
    frame: pandas.DataFrame,
    reference_run_id: str,
    resample_count: int,
    permutation_count: int,
    generator: np.random.Generator,
) -> List[SignificanceReportRow]:
    """Computes one `SignificanceReportRow` per (comparison, metric).

    Comparisons are processed in the fixed order (non-BM25 `run_id`s
    sorted ascending); within each comparison, metrics are processed in
    `METRIC_ORDER` (nDCG@10 first) -- the RNG discipline that makes
    Requirement 3.7's bit-identical reruns hold (design.md's "RNG
    discipline" section). Within each (comparison, metric),
    `paired_bootstrap` is called before `permutation_test` (bootstrap
    draw first, then permutation draw), consuming the single injected
    `generator` in that fixed order.

    A comparison with zero query IDs shared between its run and the
    Reference_Run gets the `MISSING` marker for every one of
    `mean_diff`/`ci_lower`/`ci_upper`/`p_value_raw` (and, for the
    primary metric only, `p_value_adjusted`/`verdict` too -- secondary
    rows always carry `NOT_APPLICABLE` there regardless of
    computability), and the row is retained rather than omitted
    (Requirement 3.8). `p_value_adjusted`/`verdict` for a *computable*
    primary-metric row are left as `None` here; `_apply_holm_bonferroni`
    fills them in afterward, once every comparison's raw p-value is
    known (the Comparison_Family, Requirement 5.1).
    """
    run_frames: Dict[str, pandas.DataFrame] = {
        run_id: group.set_index("query_id") for run_id, group in frame.groupby("run_id")
    }
    reference_frame = run_frames[reference_run_id]
    comparison_run_ids = sorted(rid for rid in run_frames if rid != reference_run_id)

    rows: List[SignificanceReportRow] = []

    for run_id in comparison_run_ids:
        comparison_frame = run_frames[run_id]
        shared_ids = sorted(set(reference_frame.index) & set(comparison_frame.index))
        n_shared = len(shared_ids)
        retriever_name = str(comparison_frame["retriever"].iloc[0])

        for metric in METRIC_ORDER:
            is_primary = metric == PRIMARY_METRIC

            if n_shared == 0:
                # Requirement 3.8: retain the row, mark every computed
                # cell MISSING. Secondary rows keep NOT_APPLICABLE for
                # p_value_adjusted/verdict regardless (Requirement 6.3).
                mean_diff: Union[float, str] = MISSING
                ci_lower: Union[float, str] = MISSING
                ci_upper: Union[float, str] = MISSING
                p_value_raw: Union[float, str] = MISSING
                p_value_adjusted: Union[float, str, None] = (
                    MISSING if is_primary else NOT_APPLICABLE
                )
                verdict: Union[str, None] = MISSING if is_primary else NOT_APPLICABLE
            else:
                a = comparison_frame.loc[shared_ids, metric].to_numpy(dtype=float)
                b = reference_frame.loc[shared_ids, metric].to_numpy(dtype=float)
                # Bootstrap draw first, then permutation draw -- fixed
                # order, single shared generator (RNG discipline).
                mean_diff, ci_lower, ci_upper = paired_bootstrap(
                    a, b, resample_count, generator
                )
                p_value_raw = permutation_test(a, b, permutation_count, generator)
                if is_primary:
                    # Filled in by _apply_holm_bonferroni once the whole
                    # nDCG@10 Comparison_Family's raw p-values are known.
                    p_value_adjusted = None
                    verdict = None
                else:
                    p_value_adjusted = NOT_APPLICABLE
                    verdict = NOT_APPLICABLE

            rows.append(
                SignificanceReportRow(
                    run_id=run_id,
                    retriever=retriever_name,
                    reference_run_id=reference_run_id,
                    metric=metric,
                    is_primary=is_primary,
                    mean_diff=mean_diff,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    p_value_raw=p_value_raw,
                    p_value_adjusted=p_value_adjusted,
                    n_shared_queries=n_shared,
                    verdict=verdict,
                )
            )

    return rows


def _apply_holm_bonferroni(
    rows: List[SignificanceReportRow], alpha: float
) -> List[SignificanceReportRow]:
    """Fills in `p_value_adjusted`/`verdict` for every computable
    nDCG@10 (primary-metric) row, via `holm_bonferroni` over the
    Comparison_Family's raw p-values (Requirement 5.1, 5.2).

    A zero-shared-queries nDCG@10 row (`p_value_raw == MISSING`) is
    excluded from the family passed to `holm_bonferroni` -- there is no
    real p-value to adjust -- and keeps the `MISSING` marker already set
    on it by `_run_comparisons`. Every other primary-metric row's
    `verdict` becomes `"significant"` if its adjusted p-value is `<
    alpha`, else `"indistinguishable"` (Requirement 6.4); never decided
    from whether the confidence interval includes zero.
    """
    ndcg_indices = [
        i
        for i, row in enumerate(rows)
        if row.is_primary and not isinstance(row.p_value_raw, str)
    ]
    raw_p_values = [rows[i].p_value_raw for i in ndcg_indices]
    adjusted_values = holm_bonferroni(raw_p_values)
    for i, adjusted in zip(ndcg_indices, adjusted_values):
        verdict = "significant" if adjusted < alpha else "indistinguishable"
        rows[i] = dataclasses.replace(rows[i], p_value_adjusted=adjusted, verdict=verdict)
    return rows


def _write_significance_report(rows: List[SignificanceReportRow], output_path: Path) -> None:
    """Writes `rows` to `output_path` (e.g. `results/significance.csv`)
    as a CSV, atomically, via `src.report._atomic_write_text`.

    Columns are fixed to `SignificanceReportRow`'s field order. Raises
    `SignificanceWriteError` on any failure, leaving `output_path`
    either absent or in its pre-run state (Requirement 2.7).
    """
    output_path = Path(output_path)
    fieldnames = [f.name for f in dataclasses.fields(SignificanceReportRow)]
    try:
        frame = pandas.DataFrame(
            [dataclasses.asdict(row) for row in rows], columns=fieldnames
        )
        csv_text = frame.to_csv(index=False)
    except Exception as exc:
        raise SignificanceWriteError(
            f"failed to build significance report for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(output_path, csv_text, failure_context="significance report")
    except ReportWriteError as exc:
        raise SignificanceWriteError(str(exc)) from exc


def _merge_significance_into_run_config(
    run_config_record: dict, config: SignificanceConfig, run_config_path: Path
) -> None:
    """Merges the `"significance"` sub-object into `run_config_record`
    and re-writes `run_config_path` atomically, preserving every
    existing key (`seed`, `sweep_config`, `corpus_load_report`,
    `installed_versions`, ...) unchanged (Requirement 4.3).

    The recorded `bootstrap_seed`/`resample_count`/`permutation_count`/
    `alpha` are the values actually applied for this run (Requirement
    4.4). Raises `RunConfigMergeError` if serialization or the atomic
    write fails (Requirement 4.6); the original `run_config_path` is
    left untouched on failure.
    """
    record = dict(run_config_record)
    record["significance"] = {
        "bootstrap_seed": config.bootstrap_seed,
        "resample_count": config.resample_count,
        "permutation_count": config.permutation_count,
        "alpha": config.alpha,
    }
    try:
        json_text = json.dumps(record, indent=2, default=_json_default)
    except Exception as exc:
        raise RunConfigMergeError(
            f"failed to serialize merged run config record for {run_config_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(
            run_config_path,
            json_text,
            failure_context="run config record (significance merge)",
        )
    except ReportWriteError as exc:
        raise RunConfigMergeError(str(exc)) from exc


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.significance",
        description=(
            "Reads results/per_query.csv and writes results/significance.csv: "
            "a paired bootstrap and permutation test of each non-BM25 run "
            "against the BM25 Reference_Run, with a Holm-Bonferroni-adjusted "
            "nDCG@10 headline verdict. Performs no corpus loading, index "
            "building, retrieval, query encoding, or model loading, and "
            "makes no network call."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the Bootstrap_Config YAML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: `python -m src.significance [--config PATH]`.

    Orchestration, in order (`design.md`'s `src/significance.py`
    `main()` section):

    1. Parse `--config` (default `configs/significance.yaml`) and load
       it via `load_significance_config`. On `BootstrapConfigError`:
       print the error, return non-zero, write nothing (Requirement
       4.5).
    2. Read `config.run_config_path` (default `results/run_config.json`).
       On `RunConfigMergeError`: print the error, return non-zero,
       never create a fresh file (Requirement 4.6).
    3. Read `config.per_query_path` via pandas. On
       `SignificanceInputError`: print the error, return non-zero,
       write no `results/significance.csv` (Requirement 2.4).
    4. Find the BM25 Reference_Run. On `MissingReferenceRunError`:
       print the error, return non-zero, write nothing (Requirement
       2.5).
    5. Build the fixed comparison order, construct the single
       `generator = np.random.default_rng(config.bootstrap_seed)`, and
       run `paired_bootstrap` + `permutation_test` per (comparison,
       metric) in the fixed RNG-discipline order (Requirement 3.1-3.5,
       3.8).
    6. Apply `holm_bonferroni` over the nDCG@10 Comparison_Family
       (Requirement 5).
    7. Determine each nDCG@10 row's verdict from `p_value_adjusted` vs
       `alpha` (Requirement 6.4).
    8. Write `results/significance.csv` atomically. On
       `SignificanceWriteError`: print the error, return non-zero
       (Requirement 2.7).
    9. Merge the `"significance"` sub-object into
       `config.run_config_path` atomically, preserving every existing
       key. On `RunConfigMergeError`: print the error, return non-zero
       (Requirement 4.6). Otherwise return 0 (Requirement 2.6).
    """
    args = _parse_args(argv)

    try:
        config = load_significance_config(args.config)
    except BootstrapConfigError as exc:
        print(
            f"ERROR: failed to load Bootstrap_Config from {args.config}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        run_config_record = _read_run_config(config.run_config_path)
    except RunConfigMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        per_query_frame = _read_per_query_report(config.per_query_path)
    except SignificanceInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    reference_run_id = f"{config.reference_retriever}__{config.reference_chunking_strategy}"
    try:
        reference_run_id = _find_reference_run_id(per_query_frame, reference_run_id)
    except MissingReferenceRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    generator = np.random.default_rng(config.bootstrap_seed)
    rows = _run_comparisons(
        per_query_frame,
        reference_run_id,
        config.resample_count,
        config.permutation_count,
        generator,
    )
    rows = _apply_holm_bonferroni(rows, config.alpha)

    try:
        _write_significance_report(rows, config.output_path)
    except SignificanceWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        _merge_significance_into_run_config(run_config_record, config, config.run_config_path)
    except RunConfigMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Significance analysis complete: wrote {len(rows)} rows to {config.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
