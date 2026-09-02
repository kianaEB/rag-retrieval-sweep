"""Bucket_Assigner: one entry point, two stages.

Bucket_Assignment_Stage -- reads results/per_query.csv, applies the
fixed Failure_Bucket and Contrast_Bucket taxonomies, and asserts both
partitions are total (Requirements 2-7). This stage loads no corpus, no
qrels, no tokenizer, no embedding model, and no generative model, so no
bucket label can depend on a token count, a corpus document, or a model
(Requirement 2.1).

Covariate_Enrichment_Stage -- loads the already-cached BEIR SciFact
corpus, its Qrels, and each Dense_Model's tokenizer from data/, and
computes the six per-query Token_Length_Covariate columns of
results/failure_buckets.csv (Requirements 2.8, 16). This stage is the
only part of this module, and of this spec, that reads a corpus, a
tokenizer, or a model. It reads data/ read-only, offline, CPU-only, and
after a pre-flight presence check -- it never downloads and never
writes there.

Both stages complete, and both CSV texts are fully serialized, before
the first byte of either report is written, so every failure tier leaves
both reports untouched (Requirements 2.5, 4.6, 5.5, 7.6, 16.13).

Applies no random sampling, no shuffling, and no time-dependent value,
and relies on tokenization being deterministic for a fixed tokenizer
revision, so no seed is required for its output to be reproducible
(Requirements 2.7, 16.16).

Invoked as `python -m src.failure_buckets [--per-query PATH] [--config
PATH] [--buckets-out PATH] [--counts-out PATH]`. `--config` is read only
for its `data_dir` field; no argument names a threshold, a bucket name,
a taxonomy switch, or a token-length limit, so Requirement 3.4's and
Requirement 16.6's "not read from a command-line argument" hold by the
parser's own shape. `main` runs the Bucket_Assignment_Stage to
completion -- including every Totality_Assertion -- before the
Covariate_Enrichment_Stage ever loads anything, and writes neither
report until every assertion in both stages has already passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas

from src.config import load_sweep_config
from src.corpus_loader import configure_caches, load_scifact
from src.errors import (
    ConfigError,
    ContrastQuerySetError,
    CorpusLoadError,
    CorpusValidationError,
    CovariateInputError,
    FailureBucketAssertionError,
    FailureBucketInputError,
    FailureBucketWriteError,
    TokenizerLoadError,
)
from src.metrics import judged_relevant_docs
from src.report import MISSING, _atomic_write_text

# --- Fixed taxonomy constants (Requirement 3.4, 7.5, 7.8) ---

# The four Failure_Bucket names, in the exact first-match evaluation
# order of Requirement 3 Criterion 1 -- which is also the within-run_id
# row order of the Failure_Bucket_Counts_Report (Requirement 7.8).
FAILURE_BUCKET_ORDER: Tuple[str, ...] = (
    "total_miss",
    "mis_ranked",
    "partial_recall",
    "full_success",
)

# The four Contrast_Bucket names, in the within-Composite_Run_Id row
# order of the Failure_Bucket_Counts_Report (Requirement 7.8). Disjoint
# from FAILURE_BUCKET_ORDER by construction, so a mistyped selector
# resolves to zero rows and fails loudly (Requirement 7.5).
CONTRAST_BUCKET_ORDER: Tuple[str, ...] = (
    "a_only",
    "b_only",
    "both_miss",
    "both_answer",
)

# The Composite_Run_Id separator (Requirement 7.3). Chosen because none
# of ";", ".", ",", "=" -- the four delimiters the Verifier's
# source_fields/selector grammar uses -- appears in it.
COMPOSITE_SEPARATOR: str = "|vs|"

# Reference_Run and the cross-strategy contrast rule's parameters
# (Requirement 4.3). Names, not counts: Requirement 2.6 forbids a
# literal Run_Id count / query count / row count, and none appears
# anywhere in this module.
REFERENCE_RUN_ID: str = "bm25__whole_document"
CROSS_STRATEGY_BASE: str = "whole_document"
CROSS_STRATEGY_VARIANTS: Tuple[str, ...] = ("fixed_window", "sentence_window")
DENSE_RETRIEVERS: Tuple[str, ...] = ("all-MiniLM-L6-v2", "bge-small-en-v1.5")

# Columns the Bucket_Assigner requires of the Per_Query_Report
# (Requirement 2.5).
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "retriever",
    "chunking_strategy",
    "query_id",
    "recall_at_1",
    "recall_at_20",
    "ndcg_at_10",
    "num_judged_relevant",
)

# Columns that MUST be parsed as text, never coerced to a numeric
# dtype. query_id is the load-bearing one: SciFact query ids are
# numeric-looking strings ("1", "13", "1012"), so pandas' default
# inference makes the column int64 -- which would (a) change the text
# written back for any id with a leading zero, violating Requirement
# 6.3's "copied unchanged", and (b) make Requirement 6.4's "ascending
# in lexicographic order of the column's text" silently become numeric
# order instead (they differ: lexicographic gives 1, 100, 1012, ...;
# numeric gives 1, 3, 5, 13, ...).
TEXT_COLUMNS: Tuple[str, ...] = ("run_id", "retriever", "chunking_strategy", "query_id")

# fraction is written as a fixed-point decimal with exactly this many
# digits after the point (Requirement 7.7).
FRACTION_DECIMALS: int = 6

# Requirement 5.4's tolerance: applied to the UNROUNDED float fractions
# CountRow carries, before any rendering.
FRACTION_SUM_TOLERANCE: float = 1e-9

# Requirement 5.7's tolerance: applied to the four fractions AS RENDERED
# to FRACTION_DECIMALS places and re-parsed. Rounding to 6 decimal
# places moves each of the four values by at most 5e-7 and therefore
# their sum by at most 4 x 5e-7 = 2e-6, so this constant is that
# arithmetic written out, not a tuned number. Both assertions run;
# Requirement 5.7 requires this one IN ADDITION TO Requirement 5.4's,
# never in place of it.
RENDERED_FRACTION_TOLERANCE: float = 2e-6

# --- Covariate column naming (Requirement 6.1, 6.6, 6.7) ---
# Declared here, alongside the Bucket_Assignment_Stage's own constants,
# because the twelve-column Failure_Bucket_Report schema is one
# declaration and must not be split across two tasks. The
# Covariate_Enrichment_Stage that actually POPULATES these columns is
# added by a later task.

# Retriever name (as it appears in the Per_Query_Report's `retriever`
# column, and therefore in a covariate column name) -> the Hugging Face
# repo id whose tokenizer measures it. Literal model IDENTITIES, never a
# literal LIMIT -- the limit is resolved from each model's own cached
# configuration by the Covariate_Enrichment_Stage. Requirement 6.1 pins
# the covariate column names, which embed these retriever names, so
# they are fixed either way. Dict order is declaration order, which is
# the model-major ordering Requirement 6.1 requires of
# FAILURE_BUCKET_COLUMNS below.
DENSE_MODEL_NAMES: Mapping[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

# The three covariate names, in the per-model column order of
# Requirement 6.1.
COVARIATE_NAMES: Tuple[str, ...] = (
    "query_token_len",
    "max_relevant_doc_token_len",
    "any_relevant_doc_exceeds_limit",
)


def model_tag(retriever_name: str) -> str:
    """Requirement 6.6's tag rule: the Dense_Model's retriever name as
    it appears in the Per_Query_Report's `retriever` column, with every
    "." replaced by "_". So "all-MiniLM-L6-v2" is unchanged and
    "bge-small-en-v1.5" becomes "bge-small-en-v1_5"."""
    return retriever_name.replace(".", "_")


def covariate_column(covariate: str, retriever_name: str) -> str:
    """Requirement 6.6's column name: f"{covariate}__{model_tag(...)}".

    No column name produced by this function may contain a "." -- see
    Requirement 6.7: the Verifier separates a ledger reference's row
    selector from its field name at the LAST "." of the reference text
    (`str.rsplit(".", 1)`), so a "." inside a column NAME would make
    that split resolve the wrong field. A "." inside a column VALUE is
    unaffected and remains permitted -- the `run_id`/`retriever`
    columns keep their unsubstituted "bge-small-en-v1.5" text."""
    return f"{covariate}__{model_tag(retriever_name)}"


# Default paths, relative to the repository root (Requirement 2.3).
DEFAULT_CONFIG_PATH = Path("configs/sweep.yaml")
DEFAULT_PER_QUERY_PATH = Path("results/per_query.csv")
DEFAULT_BUCKETS_PATH = Path("results/failure_buckets.csv")
DEFAULT_COUNTS_PATH = Path("results/failure_bucket_counts.csv")


# --- Row schemas (Requirement 6.1, 6.2, 7.1) ---


@dataclass(frozen=True)
class FailureBucketRow:
    """One row of results/failure_buckets.csv: exactly one (Run_Id,
    query_id) pair (Requirement 6.2). Field order IS the committed
    twelve-column order (Requirement 6.1), as built by
    FAILURE_BUCKET_COLUMNS below. retriever, chunking_strategy,
    query_id and num_judged_relevant are copied unchanged from the
    corresponding Per_Query_Report row (Requirement 6.3); bucket is the
    one Failure_Bucket assign_failure_bucket returned; the six
    covariate fields come from the Covariate_Enrichment_Stage, joined
    on query_id alone (added by a later task).

    The three token-length fields are typed Union[int, str] and the
    three boolean fields Union[bool, str], because either may hold the
    Missing_Value_Sentinel "NA" instead of a value (Requirements 6.8,
    16.8) -- the same Union[float, str] shape SweepReportRow already
    uses for its own MISSING-capable cells.

    Field names cannot be written literally in Python (a "." and a "-"
    are not identifiers), so this dataclass declares them with safe
    identifiers (`minilm`/`bge`); FAILURE_BUCKET_COLUMNS supplies the
    committed header names through covariate_column(...) instead, and a
    module-level assertion pins the two against each other."""

    run_id: str
    retriever: str
    chunking_strategy: str
    query_id: str
    bucket: str
    num_judged_relevant: int
    query_token_len__minilm: Union[int, str]
    max_relevant_doc_token_len__minilm: Union[int, str]
    any_relevant_doc_exceeds_limit__minilm: Union[bool, str]
    query_token_len__bge: Union[int, str]
    max_relevant_doc_token_len__bge: Union[int, str]
    any_relevant_doc_exceeds_limit__bge: Union[bool, str]


@dataclass(frozen=True)
class CountRow:
    """One row of results/failure_bucket_counts.csv (Requirement 7.1).
    `run_id` is either a Run_Id (for a per-run Failure_Bucket row) or a
    Composite_Run_Id (for a Pair_Contrast Contrast_Bucket row).
    `fraction` is held here as an unrounded float and rendered to
    exactly FRACTION_DECIMALS places by the writer (Requirement 7.7).
    Requirement 5.4's 1e-9 assertion runs against this float;
    Requirement 5.7's 2e-6 assertion runs against its 6-decimal
    rendering, re-parsed. Both run (Requirement 5.7)."""

    run_id: str
    bucket: str
    count: int
    fraction: float


# The committed twelve-column order (Requirement 6.1), built ONCE from
# the same rules that name the columns -- so the header, the dataclass,
# and the ledger's field names cannot drift apart. Nesting is
# model-major, covariate-minor: all three covariates for
# all-MiniLM-L6-v2 first, then all three for bge-small-en-v1.5.
FAILURE_BUCKET_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "retriever",
    "chunking_strategy",
    "query_id",
    "bucket",
    "num_judged_relevant",
) + tuple(
    covariate_column(covariate, retriever_name)
    for retriever_name in DENSE_MODEL_NAMES
    for covariate in COVARIATE_NAMES
)

# A future edit to the tag rule or to FailureBucketRow that
# desynchronizes them must fail at import time, not produce a
# mislabelled artifact.
assert len(FAILURE_BUCKET_COLUMNS) == len(dataclasses.fields(FailureBucketRow)), (
    "FAILURE_BUCKET_COLUMNS and FailureBucketRow have drifted apart: "
    f"{len(FAILURE_BUCKET_COLUMNS)} columns vs "
    f"{len(dataclasses.fields(FailureBucketRow))} dataclass fields"
)
assert FAILURE_BUCKET_COLUMNS[-6:] == (
    "query_token_len__all-MiniLM-L6-v2",
    "max_relevant_doc_token_len__all-MiniLM-L6-v2",
    "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2",
    "query_token_len__bge-small-en-v1_5",
    "max_relevant_doc_token_len__bge-small-en-v1_5",
    "any_relevant_doc_exceeds_limit__bge-small-en-v1_5",
), (
    "Requirement 6.1's six covariate column names have drifted from "
    f"FAILURE_BUCKET_COLUMNS: {FAILURE_BUCKET_COLUMNS[-6:]}"
)


# --- The pure predicate functions (Requirement 3, 4.1, 4.2) ---


def assign_failure_bucket(
    recall_at_1: float, recall_at_20: float, num_judged_relevant: int
) -> str:
    """Returns the one Failure_Bucket for a (Run_Id, query_id) pair, by
    evaluating Requirement 3 Criterion 1's four predicates in order and
    returning the first that holds.

    Comparisons against 0 and 1 are exact, never within a tolerance
    (Requirement 3.3): each recall value is a ratio of integer counts,
    so 0 and 1 are exactly representable.

    Total by construction: the fourth branch is an unconditional
    fallthrough, so every input returns exactly one member of
    FAILURE_BUCKET_ORDER (Requirement 3.2). The `partial_recall` branch
    drops the redundant `recall_at_20 > 0` clause Criterion 1(3) states
    explicitly -- branch 1 already returned for every input with
    `recall_at_20 == 0`, and recall is never negative, so this is a
    simplification of the code, not of the taxonomy.
    """
    if recall_at_20 == 0:
        return "total_miss"
    if recall_at_1 == 0:
        return "mis_ranked"
    if num_judged_relevant > 1 and recall_at_20 < 1:
        return "partial_recall"
    return "full_success"


def is_answered(ndcg_at_10: float) -> bool:
    """True for an Answered_Query (ndcg_at_10 strictly greater than 0),
    False for a Missed_Query (exactly 0) -- Requirement 4.1."""
    return ndcg_at_10 > 0


def assign_contrast_bucket(ndcg_a: float, ndcg_b: float) -> str:
    """Returns the one Contrast_Bucket for a (Pair_Contrast, query_id)
    combination, from Run_A's and Run_B's ndcg_at_10 values
    (Requirement 4.2). Exhaustive and mutually exclusive: the two
    booleans have four combinations and each maps to exactly one
    name."""
    a, b = is_answered(ndcg_a), is_answered(ndcg_b)
    if a and not b:
        return "a_only"
    if b and not a:
        return "b_only"
    if not a and not b:
        return "both_miss"
    return "both_answer"


# --- The Declared_Contrast_Set (Requirement 4.3, 4.4, 4.5) ---


def make_composite_run_id(run_a: str, run_b: str) -> str:
    """Returns f"{run_a}{COMPOSITE_SEPARATOR}{run_b}" -- the identifier
    a Pair_Contrast occupies in the Failure_Bucket_Counts_Report's
    run_id column (Requirement 7.3)."""
    return f"{run_a}{COMPOSITE_SEPARATOR}{run_b}"


def build_declared_contrast_set(run_ids: Iterable[str]) -> Tuple[Tuple[str, str], ...]:
    """Builds the Declared_Contrast_Set from the Run_Ids the
    Per_Query_Report actually contains (Requirement 2.6, 4.3).

    Group (a), family-aligned: (REFERENCE_RUN_ID, other) for every
    observed Run_Id other than the Reference_Run, in ascending
    lexicographic order of `other`. One pair per row of the
    Pre_Declared_Family, so a Contrast_Bucket figure and a
    Pre_Declared_Family verdict are always about the same two runs.
    This group also supplies the two BM25 cross-strategy contrasts
    (Requirement 4.4), so group (b) below must not re-emit them.

    Group (b), dense cross-strategy: for each retriever in
    DENSE_RETRIEVERS and each variant in CROSS_STRATEGY_VARIANTS,
    (f"{retriever}__{CROSS_STRATEGY_BASE}", f"{retriever}__{variant}").

    Returns group (a) followed by group (b). Raises
    ContrastQuerySetError, naming the absent Run_Id, if the
    Reference_Run or any Run_Id group (b) needs is absent from
    `run_ids` -- a declared contrast over a run that was never swept
    has no partition. Ends with a `len(set(pairs)) == len(pairs)` guard
    raising FailureBucketAssertionError, belt to the structural braces
    that already make the two groups disjoint and each group's own
    pairs distinct (Requirement 4.5).
    """
    observed = set(run_ids)
    if REFERENCE_RUN_ID not in observed:
        raise ContrastQuerySetError(
            f"Reference_Run {REFERENCE_RUN_ID!r} required by the "
            "Declared_Contrast_Set is absent from the Per_Query_Report's "
            "run_id column"
        )
    group_a: Tuple[Tuple[str, str], ...] = tuple(
        (REFERENCE_RUN_ID, other) for other in sorted(observed - {REFERENCE_RUN_ID})
    )

    group_b: List[Tuple[str, str]] = []
    for retriever in DENSE_RETRIEVERS:
        base_run = f"{retriever}__{CROSS_STRATEGY_BASE}"
        if base_run not in observed:
            raise ContrastQuerySetError(
                f"Pair_Contrast base Run_Id {base_run!r} is absent from "
                "the Per_Query_Report's run_id column"
            )
        for variant in CROSS_STRATEGY_VARIANTS:
            variant_run = f"{retriever}__{variant}"
            if variant_run not in observed:
                raise ContrastQuerySetError(
                    f"Pair_Contrast Run_Id {variant_run!r} is absent from "
                    "the Per_Query_Report's run_id column"
                )
            group_b.append((base_run, variant_run))

    pairs = group_a + tuple(group_b)
    if len(set(pairs)) != len(pairs):
        raise FailureBucketAssertionError(
            f"Declared_Contrast_Set contains a duplicate Pair_Contrast: {pairs}"
        )
    return pairs


# --- Loading (Requirement 2.5) ---


def load_per_query(path: Path) -> pandas.DataFrame:
    """Reads the Per_Query_Report, with TEXT_COLUMNS forced to str and
    the "NA"/"n/a" sentinel strings preserved as literal text
    (keep_default_na=False, na_values=[]) exactly as
    src/verify_writeup_numbers.py's _read_csv_artifact already does.

    Raises FailureBucketInputError, naming the path, if the file is
    absent or cannot be parsed; and naming every missing column at once
    if any of REQUIRED_COLUMNS is absent, so a caller fixing a
    malformed input sees the full list rather than one column per run.
    """
    path = Path(path)
    if not path.is_file():
        raise FailureBucketInputError(f"Per_Query_Report not found: {path}")
    try:
        frame = pandas.read_csv(
            path,
            dtype={column: str for column in TEXT_COLUMNS},
            keep_default_na=False,
            na_values=[],
        )
    except Exception as exc:
        raise FailureBucketInputError(
            f"failed to parse Per_Query_Report {path}: {exc}"
        ) from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FailureBucketInputError(
            f"Per_Query_Report {path} is missing required column(s): "
            f"{', '.join(missing)}"
        )
    return frame


# --- Building the two frames ---


def build_failure_buckets(per_query: pandas.DataFrame) -> pandas.DataFrame:
    """Assigns one Failure_Bucket per (Run_Id, query_id) row and returns
    the Failure_Bucket_Report frame: `run_id`, `retriever`,
    `chunking_strategy`, `query_id`, `bucket`, `num_judged_relevant`
    columns, one data row per input row, sorted by run_id text
    ascending then query_id text ascending, with a reset positional
    index the writer never emits (Requirements 6.1-6.4). The six
    covariate columns are added later by attach_covariates.
    """
    rows = []
    for row in per_query.itertuples(index=False):
        bucket = assign_failure_bucket(
            float(getattr(row, "recall_at_1")),
            float(getattr(row, "recall_at_20")),
            int(getattr(row, "num_judged_relevant")),
        )
        rows.append(
            {
                "run_id": getattr(row, "run_id"),
                "retriever": getattr(row, "retriever"),
                "chunking_strategy": getattr(row, "chunking_strategy"),
                "query_id": getattr(row, "query_id"),
                "bucket": bucket,
                "num_judged_relevant": int(getattr(row, "num_judged_relevant")),
            }
        )
    frame = pandas.DataFrame(
        rows,
        columns=[
            "run_id",
            "retriever",
            "chunking_strategy",
            "query_id",
            "bucket",
            "num_judged_relevant",
        ],
    )
    frame = frame.sort_values(
        by=["run_id", "query_id"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def build_run_counts(failure_buckets: pandas.DataFrame) -> pandas.DataFrame:
    """Aggregates the per-query frame into four CountRows per Run_Id
    (Requirement 7.2): count is the number of query_ids in that bucket,
    fraction is that count divided by the number of distinct query_ids
    the Per_Query_Report holds for that Run_Id.

    Emits all four declared buckets for every Run_Id, including a
    bucket with a count of 0 -- a zero-count row is a real result, and
    omitting it would make the selector run_id=X,bucket=partial_recall
    resolve to zero rows and fail the Verifier.

    Calls assert_partition_total per Run_Id before returning.
    """
    rows: List[CountRow] = []
    for run_id, group in failure_buckets.groupby("run_id", sort=False):
        total = len(group)
        bucket_counts = {
            bucket: int((group["bucket"] == bucket).sum())
            for bucket in FAILURE_BUCKET_ORDER
        }
        assert_partition_total(run_id, bucket_counts, total, FAILURE_BUCKET_ORDER)
        for bucket in FAILURE_BUCKET_ORDER:
            count = bucket_counts[bucket]
            rows.append(
                CountRow(run_id=run_id, bucket=bucket, count=count, fraction=count / total)
            )
    return pandas.DataFrame(
        [dataclasses.asdict(row) for row in rows],
        columns=[field.name for field in dataclasses.fields(CountRow)],
    )


def build_contrast_counts(
    per_query: pandas.DataFrame, contrast_set: Sequence[Tuple[str, str]]
) -> pandas.DataFrame:
    """Assigns one Contrast_Bucket per (Pair_Contrast, query_id) and
    aggregates into four CountRows per Pair_Contrast (Requirement 7.3),
    with run_id set to make_composite_run_id(run_a, run_b) and fraction
    computed over the query_ids shared by Run_A and Run_B.

    Raises ContrastQuerySetError, naming the Pair_Contrast and the
    lexicographically smallest offending query_id, if the two Run_Ids'
    query_id sets differ in either direction (Requirement 4.6).

    Emits all four declared buckets per Pair_Contrast, zero counts
    included, for the same reason build_run_counts does. Calls
    assert_partition_total per Pair_Contrast before returning.
    """
    ndcg_lookup: Dict[Tuple[str, str], float] = {
        (run_id, query_id): float(ndcg)
        for run_id, query_id, ndcg in zip(
            per_query["run_id"], per_query["query_id"], per_query["ndcg_at_10"]
        )
    }
    query_ids_by_run: Dict[str, Set[str]] = {}
    for run_id, query_id in zip(per_query["run_id"], per_query["query_id"]):
        query_ids_by_run.setdefault(run_id, set()).add(query_id)

    rows: List[CountRow] = []
    for run_a, run_b in contrast_set:
        qids_a = query_ids_by_run.get(run_a, set())
        qids_b = query_ids_by_run.get(run_b, set())
        if qids_a != qids_b:
            offending = min(qids_a ^ qids_b)
            raise ContrastQuerySetError(
                f"Pair_Contrast ({run_a}, {run_b}) has asymmetric "
                f"query_id coverage: query_id {offending!r} is present "
                "for one run and absent for the other"
            )
        shared = qids_a
        composite = make_composite_run_id(run_a, run_b)
        bucket_counts = {bucket: 0 for bucket in CONTRAST_BUCKET_ORDER}
        for query_id in shared:
            bucket = assign_contrast_bucket(
                ndcg_lookup[(run_a, query_id)], ndcg_lookup[(run_b, query_id)]
            )
            bucket_counts[bucket] += 1
        total = len(shared)
        assert_partition_total(composite, bucket_counts, total, CONTRAST_BUCKET_ORDER)
        for bucket in CONTRAST_BUCKET_ORDER:
            count = bucket_counts[bucket]
            rows.append(
                CountRow(run_id=composite, bucket=bucket, count=count, fraction=count / total)
            )
    return pandas.DataFrame(
        [dataclasses.asdict(row) for row in rows],
        columns=[field.name for field in dataclasses.fields(CountRow)],
    )


def build_failure_bucket_counts(
    run_counts: pandas.DataFrame, contrast_counts: pandas.DataFrame
) -> pandas.DataFrame:
    """Concatenates the two count frames and applies Requirement 7.8's
    total order: every Run_Id row before every Composite_Run_Id row,
    within each group by run_id text ascending, within each run_id by
    declared bucket order. Sorts on an explicit key column triple that
    is dropped before returning, so the resulting row order does not
    depend on pandas' sort stability or on the concatenation order.
    """
    combined = pandas.concat([run_counts, contrast_counts], ignore_index=True)

    def _bucket_rank(row: pandas.Series) -> int:
        if COMPOSITE_SEPARATOR in row["run_id"]:
            return CONTRAST_BUCKET_ORDER.index(row["bucket"])
        return FAILURE_BUCKET_ORDER.index(row["bucket"])

    combined["_group"] = combined["run_id"].map(
        lambda run_id: 1 if COMPOSITE_SEPARATOR in run_id else 0
    )
    combined["_bucket_rank"] = combined.apply(_bucket_rank, axis=1)
    combined = combined.sort_values(
        by=["_group", "run_id", "_bucket_rank"], kind="mergesort"
    )
    combined = combined.drop(columns=["_group", "_bucket_rank"]).reset_index(drop=True)
    return combined


# --- The assertions (Requirement 5) ---


def assert_no_separator_collision(run_ids: Iterable[str]) -> None:
    """Raises FailureBucketAssertionError naming the offending Run_Id(s)
    if COMPOSITE_SEPARATOR occurs in any of them (Requirement 7.6).
    Called before any bucket is assigned, so a colliding Run_Id halts
    the run at the earliest possible point."""
    offending = sorted({run_id for run_id in run_ids if COMPOSITE_SEPARATOR in run_id})
    if offending:
        raise FailureBucketAssertionError(
            f"Run_Id(s) contain the Composite_Run_Id separator "
            f"{COMPOSITE_SEPARATOR!r}, which could collide with a "
            f"Composite_Run_Id in the counts report: {offending}"
        )


def assert_unique_pairs(failure_buckets: pandas.DataFrame) -> None:
    """Raises FailureBucketAssertionError if any (run_id, query_id) pair
    occurs in more than one row, naming the duplicated pairs
    (Requirement 5.3)."""
    duplicated = failure_buckets.duplicated(subset=["run_id", "query_id"], keep=False)
    if duplicated.any():
        pairs = sorted(
            set(
                zip(
                    failure_buckets.loc[duplicated, "run_id"],
                    failure_buckets.loc[duplicated, "query_id"],
                )
            )
        )
        raise FailureBucketAssertionError(
            f"duplicated (run_id, query_id) pair(s) in the Failure_Bucket_Report: {pairs}"
        )


def assert_partition_total(
    partition_label: str,
    bucket_counts: Mapping[str, int],
    expected_total: int,
    declared_buckets: Sequence[str],
) -> None:
    """The shared Totality_Assertion for both partitions (Requirements
    5.1, 5.2). Checks that bucket_counts' keys are exactly
    declared_buckets and that their values sum to expected_total;
    raises FailureBucketAssertionError naming partition_label (a Run_Id
    or a Composite_Run_Id), the observed sum, and the expected total
    (Requirement 5.5). One helper, called once per Run_Id and once per
    Pair_Contrast, so the two partitions cannot drift into two
    different notions of "total".
    """
    observed_buckets = set(bucket_counts.keys())
    expected_buckets = set(declared_buckets)
    if observed_buckets != expected_buckets:
        raise FailureBucketAssertionError(
            f"{partition_label}: bucket_counts keys {sorted(observed_buckets)} "
            f"do not match the declared bucket set {sorted(expected_buckets)}"
        )
    observed_total = sum(bucket_counts.values())
    if observed_total != expected_total:
        raise FailureBucketAssertionError(
            f"{partition_label}: bucket counts sum to {observed_total}, "
            f"expected {expected_total}"
        )


def assert_fraction_sums(counts: pandas.DataFrame) -> None:
    """Raises FailureBucketAssertionError if, for any run_id value, the
    four unrounded `fraction` values differ from 1 by more than
    FRACTION_SUM_TOLERANCE (Requirement 5.4), naming the run_id and the
    observed sum.

    Also checks the four values as they will be *written* -- each
    rendered to FRACTION_DECIMALS places and re-parsed -- against
    RENDERED_FRACTION_TOLERANCE (Requirement 5.7). Both values are
    always computed for every run_id before either raise, so neither
    check is skipped and neither replaces the other.
    """
    for run_id, group in counts.groupby("run_id", sort=False):
        unrounded_sum = float(group["fraction"].sum())
        rendered_sum = sum(
            float(f"{value:.{FRACTION_DECIMALS}f}") for value in group["fraction"]
        )
        unrounded_ok = abs(unrounded_sum - 1.0) <= FRACTION_SUM_TOLERANCE
        rendered_ok = abs(rendered_sum - 1.0) <= RENDERED_FRACTION_TOLERANCE
        if not unrounded_ok:
            raise FailureBucketAssertionError(
                f"{run_id}: unrounded fraction sum {unrounded_sum!r} differs "
                f"from 1 by more than {FRACTION_SUM_TOLERANCE}"
            )
        if not rendered_ok:
            raise FailureBucketAssertionError(
                f"{run_id}: rendered fraction sum {rendered_sum!r} differs "
                f"from 1 by more than {RENDERED_FRACTION_TOLERANCE}"
            )


# --- Writers ---


def write_failure_buckets(frame: pandas.DataFrame, output_path: Path) -> None:
    """Writes `frame` to `output_path` (e.g. results/failure_buckets.csv)
    as a CSV, atomically. Columns are fixed to FAILURE_BUCKET_COLUMNS
    regardless of `frame`'s own column order; rows are written in the
    order given (already fixed by build_failure_buckets /
    attach_covariates). Every covariate cell reaching this writer is
    already rendered text, so this writer formats nothing.

    Raises FailureBucketWriteError naming `output_path` on any failure
    -- the only tier reached after every assertion has already passed.
    """
    output_path = Path(output_path)
    try:
        ordered = frame[list(FAILURE_BUCKET_COLUMNS)]
        csv_text = ordered.to_csv(index=False)
    except Exception as exc:
        raise FailureBucketWriteError(
            f"failed to build Failure_Bucket_Report for {output_path}: {exc}"
        ) from exc
    _atomic_write_text(
        output_path, csv_text, failure_context="failure bucket report", newline=""
    )


def write_failure_bucket_counts(frame: pandas.DataFrame, output_path: Path) -> None:
    """Writes `frame` to `output_path` (e.g.
    results/failure_bucket_counts.csv) as a CSV, atomically. Columns
    are fixed to CountRow's field order (`run_id`, `bucket`, `count`,
    `fraction`); `count` is written as a base-ten integer, and
    `fraction` is pre-formatted to exactly FRACTION_DECIMALS places
    before `to_csv` is called, so the text asserted by
    assert_fraction_sums and the text written are the same
    (Requirement 7.7).

    Raises FailureBucketWriteError naming `output_path` on any failure.
    Called after write_failure_buckets in main, so a failure here means
    the per-query report already landed and this one did not.
    """
    output_path = Path(output_path)
    fieldnames = [field.name for field in dataclasses.fields(CountRow)]
    try:
        ordered = frame[fieldnames].copy()
        ordered["count"] = ordered["count"].astype(int)
        ordered["fraction"] = ordered["fraction"].map(
            lambda value: f"{value:.{FRACTION_DECIMALS}f}"
        )
        csv_text = ordered.to_csv(index=False)
    except Exception as exc:
        raise FailureBucketWriteError(
            f"failed to build Failure_Bucket_Counts_Report for {output_path}: {exc}"
        ) from exc
    _atomic_write_text(
        output_path, csv_text, failure_context="failure bucket counts report", newline=""
    )


# --- Covariate_Enrichment_Stage (Requirement 16) ---

# Requirement 6.8's rendering of a boolean covariate: literal text that
# is deliberately NOT "true"/"false" (or Python's "True"/"False" repr).
# "exceeds"/"within" are not coercible to a boolean or numeric dtype by
# pandas' CSV type inference, so a column holding only these two values
# is read back as `object` dtype regardless of whether the
# Missing_Value_Sentinel is present elsewhere in that column.
EXCEEDS_TEXT: str = "exceeds"
WITHIN_TEXT: str = "within"

# Local_Cache subpaths the pre-flight check requires (Requirement
# 16.12). The BEIR dataset directory name matches
# src/corpus_loader.py's own SciFact dataset name; the tokenizer
# snapshot directory name follows huggingface_hub's own
# "models--{org}--{name}" convention.
SCIFACT_CACHE_SUBDIR: str = "scifact"
HF_CACHE_SUBDIR: str = "hf_cache"


def _import_tokenizer_helpers():
    """Deferred import of the four reused primitives.

    src/token_length_analysis.py imports `transformers` at module top,
    and src/retrievers/dense_retriever.py imports
    `sentence_transformers` and `numpy` at module top. Importing either
    at THIS module's top would make `import src.failure_buckets` --
    and therefore `import tests/test_failure_buckets.py` -- pull the
    whole transformers/torch stack, breaking the reviewable-import-
    surface property and slowing every test run for a code path the
    tests never take.

    Deferring is the discipline this repository already uses for
    order-sensitive and heavy imports: src/corpus_loader.py defers
    `beir` inside load_scifact, and DenseRetriever.__init__ defers
    `huggingface_hub.constants`. It also gets the ordering right for
    free: this function is only ever called AFTER configure_caches has
    set HF_HOME/HF_HUB_CACHE, which is exactly the ordering
    src/corpus_loader.py's docstring requires, since huggingface_hub
    resolves those variables once at its own import time.
    """
    from src.retrievers.dense_retriever import format_document_text
    from src.token_length_analysis import (
        count_tokens,
        load_tokenizer_offline,
        resolve_effective_max_sequence_length,
    )
    return (
        format_document_text,
        count_tokens,
        load_tokenizer_offline,
        resolve_effective_max_sequence_length,
    )


@dataclass(frozen=True)
class CovariateInputs:
    """Everything the covariate computation needs, already loaded. The
    boundary between the impure and the pure half of this stage: built
    only by load_covariate_inputs, consumed only by
    compute_token_length_covariates, and constructible from Python
    literals in a test.

    `tokenizers` and `limits` are keyed by RETRIEVER name (the
    Per_Query_Report's `retriever` value, e.g. "bge-small-en-v1.5"),
    not by Hugging Face repo id, so the covariate column names follow
    from the keys without a second lookup."""

    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    tokenizers: Mapping[str, Any]
    limits: Mapping[str, int]


def assert_local_cache_present(data_dir: Path) -> None:
    """Requirement 16.12's pre-flight check. Confirms that
    `data_dir / SCIFACT_CACHE_SUBDIR` is a directory and that, for
    every model in DENSE_MODEL_NAMES, the snapshot directory
    `data_dir / HF_CACHE_SUBDIR / f"models--{org}--{name}"` is a
    directory -- the same paths tests/test_data_layer.py's own
    Local_Cache_Availability gate already checks, and the same
    "models--{org}--{name}" convention huggingface_hub writes.

    Raises CovariateInputError naming EVERY absent path, BEFORE
    load_scifact is called. This ordering is the requirement: the
    committed load_scifact begins with
    `beir_util.download_and_unzip(url, str(data_dir))`, so calling it
    against an empty cache would reach the network and populate
    data/, which Requirements 16.12 and 16.14 both forbid.

    Deliberately a directory-existence check, not a load: it must be
    cheap, and it must not itself import transformers or beir."""
    data_dir = Path(data_dir)
    missing: List[str] = []

    scifact_dir = data_dir / SCIFACT_CACHE_SUBDIR
    if not scifact_dir.is_dir():
        missing.append(str(scifact_dir))

    for repo_id in DENSE_MODEL_NAMES.values():
        org, name = repo_id.split("/", 1)
        snapshot_dir = data_dir / HF_CACHE_SUBDIR / f"models--{org}--{name}"
        if not snapshot_dir.is_dir():
            missing.append(str(snapshot_dir))

    if missing:
        raise CovariateInputError(
            "Local_Cache is missing the following required path(s) under "
            f"{data_dir}: {missing}. Populate data/ (e.g. by running the "
            "sweep or the token-length analysis) before running the "
            "Covariate_Enrichment_Stage; this stage never downloads."
        )


def resolve_model_limits(
    tokenizers: Mapping[str, Any], data_dir: Path
) -> Dict[str, int]:
    """Returns each retriever name's Effective_Max_Sequence_Length, via
    the committed resolve_effective_max_sequence_length(model_name,
    tokenizer, data_dir / HF_CACHE_SUBDIR) -- the model's own cached
    configuration and nothing else (Requirement 16.6).

    Never reads a limit from a literal in this module, from
    configs/sweep.yaml, from a CLI argument, or from an environment
    variable. In particular it does NOT import the fixed
    all-MiniLM-L6-v2 sequence-length constant
    src/token_length_analysis.py declares for its own single-model
    measurement, which would be wrong for bge-small-en-v1.5 -- the two
    Dense_Models do not share this value.

    Impure only in that it may read the cached sentence_bert_config.json;
    kept separate from load_covariate_inputs so a test can supply
    `limits` as a plain dict of ints and never call it."""
    _, _, _, resolve_effective_max_sequence_length = _import_tokenizer_helpers()
    hf_cache_dir = Path(data_dir) / HF_CACHE_SUBDIR
    return {
        retriever_name: resolve_effective_max_sequence_length(
            DENSE_MODEL_NAMES[retriever_name], tokenizer, hf_cache_dir
        )
        for retriever_name, tokenizer in tokenizers.items()
    }


def load_covariate_inputs(
    data_dir: Path, retriever_names: Sequence[str]
) -> CovariateInputs:
    """The only impure function in this stage besides
    resolve_model_limits. In this exact order:

    1. assert_local_cache_present(data_dir).
    2. configure_caches(data_dir) -- sets HF_HOME/HF_HUB_CACHE to
       data_dir / "hf_cache" (Requirement 16.10). Called BEFORE the
       deferred tokenizer imports, because huggingface_hub resolves
       those variables once at its own import time.
    3. _import_tokenizer_helpers() -- the deferred import.
    4. bundle, load_report = load_scifact(data_dir). Wraps
       CorpusLoadError and CorpusValidationError as CovariateInputError.
       Prints load_report.as_log_line() so the corpus counts this
       stage actually loaded appear in the run's own output.
    5. For each retriever_name: load_tokenizer_offline(
       DENSE_MODEL_NAMES[retriever_name], data_dir / HF_CACHE_SUBDIR),
       wrapping TokenizerLoadError as CovariateInputError.
    6. limits = resolve_model_limits(...).

    Returns CovariateInputs. Raises CovariateInputError, naming which
    of Requirement 16.13's conditions failed, on any failure; never
    downloads, never substitutes a default, and never writes under
    data/ (Requirement 16.14)."""
    data_dir = Path(data_dir)

    # Step 1: pre-flight, BEFORE load_scifact (which downloads).
    assert_local_cache_present(data_dir)

    # Step 2: configure caches BEFORE the deferred tokenizer import.
    configure_caches(data_dir)

    # Step 3: deferred import.
    (
        _format_document_text,
        _count_tokens,
        load_tokenizer_offline,
        _resolve_effective_max_sequence_length,
    ) = _import_tokenizer_helpers()

    # Step 4: load the corpus.
    try:
        bundle, load_report = load_scifact(data_dir)
    except (CorpusLoadError, CorpusValidationError) as exc:
        raise CovariateInputError(
            f"failed to load the BEIR SciFact corpus/Qrels from {data_dir}: {exc}"
        ) from exc
    print(load_report.as_log_line())

    # Step 5: load each retriever's tokenizer, offline.
    tokenizers: Dict[str, Any] = {}
    for retriever_name in retriever_names:
        model_name = DENSE_MODEL_NAMES[retriever_name]
        try:
            tokenizers[retriever_name] = load_tokenizer_offline(
                model_name, data_dir / HF_CACHE_SUBDIR
            )
        except TokenizerLoadError as exc:
            raise CovariateInputError(
                f"failed to load the {retriever_name!r} tokenizer "
                f"({model_name!r}) from the local cache under "
                f"{data_dir / HF_CACHE_SUBDIR}: {exc}"
            ) from exc

    # Step 6: resolve each model's own Effective_Max_Sequence_Length.
    limits = resolve_model_limits(tokenizers, data_dir)

    return CovariateInputs(
        corpus=bundle.corpus,
        queries=bundle.queries,
        qrels=bundle.qrels,
        tokenizers=tokenizers,
        limits=limits,
    )


def max_relevant_doc_token_len(
    doc_token_lens: Mapping[str, int], relevant_doc_ids: Iterable[str]
) -> Optional[int]:
    """FULLY PURE. Returns max(doc_token_lens[d] for d in
    relevant_doc_ids), or None when relevant_doc_ids is empty
    (Requirement 16.2, 16.8).

    Takes ALREADY-TOKENIZED lengths, so the unit tests for this
    function need no tokenizer at all. None, not 0, is the empty
    answer: the caller renders it as MISSING, keeping "no judged
    relevant document" distinguishable from "a judged relevant
    document of length 0".

    A KeyError for an id absent from doc_token_lens is left to
    propagate: load_scifact already validates that every
    qrels-referenced document id resolves against the loaded corpus,
    so this cannot happen for real inputs, and silently skipping the
    id would understate the maximum."""
    relevant_doc_ids = list(relevant_doc_ids)
    if not relevant_doc_ids:
        return None
    return max(doc_token_lens[doc_id] for doc_id in relevant_doc_ids)


def compute_token_length_covariates(
    query_ids: Sequence[str], inputs: CovariateInputs
) -> pandas.DataFrame:
    """Computes the six Token_Length_Covariate values per query_id and
    returns a frame with columns ["query_id"] + the six
    covariate_column(...) names (Requirement 16.1-16.3).

    Takes already-loaded objects and paths nothing, so a stub corpus, a
    stub qrels mapping, and a hand-written stub tokenizer are a
    complete fixture. Per (query_id, retriever_name):

      relevant = judged_relevant_docs(inputs.qrels.get(query_id, {}))
          -- src/metrics.py's own > 0 condition, the only source of
             relevance (Requirement 16.7). No retrieval result, no
             model score, no heuristic is consulted.
      query_token_len = count_tokens(tokenizer, inputs.queries[query_id])
          -- untruncated, special tokens included (Requirement 16.4).
      doc_token_lens = {doc_id: count_tokens(tokenizer,
          format_document_text(inputs.corpus[doc_id])) for doc_id in
          relevant} -- format_document_text is title + " " + text,
          measured over the SOURCE document, never a Chunk
          (Requirement 16.5).
      max_len = max_relevant_doc_token_len(doc_token_lens, relevant)
      exceeds = None if max_len is None else max_len > inputs.limits[retriever_name]
          -- STRICTLY greater than (Requirement 16.3).

    Raises CovariateInputError naming the query_id if it is absent from
    inputs.queries (Requirement 16.13's fourth condition).

    Each document's token count is computed once per (retriever,
    doc_id) and memoized across queries -- SciFact's judged-relevant
    document sets overlap, and tokenizing the same abstract twice would
    only cost time, never change an answer. Memoization does not affect
    determinism: count_tokens is a pure function of (tokenizer, text).

    The returned frame has ONE row per query_id -- never per (run_id,
    query_id) -- which is what makes Requirement 16.9's
    run-independence structural rather than asserted."""
    format_document_text, count_tokens, _load_tokenizer_offline, _resolve = (
        _import_tokenizer_helpers()
    )

    retriever_names = list(inputs.tokenizers.keys())
    doc_token_len_cache: Dict[Tuple[str, str], int] = {}

    def _doc_token_len(retriever_name: str, doc_id: str) -> int:
        key = (retriever_name, doc_id)
        if key not in doc_token_len_cache:
            tokenizer = inputs.tokenizers[retriever_name]
            text = format_document_text(inputs.corpus[doc_id])
            doc_token_len_cache[key] = count_tokens(tokenizer, text)
        return doc_token_len_cache[key]

    rows: List[Dict[str, Any]] = []
    for query_id in query_ids:
        if query_id not in inputs.queries:
            raise CovariateInputError(
                f"query_id {query_id!r} present in the Per_Query_Report is "
                "absent from the loaded BEIR SciFact query set"
            )
        relevant = judged_relevant_docs(inputs.qrels.get(query_id, {}))
        row: Dict[str, Any] = {"query_id": query_id}
        for retriever_name in retriever_names:
            tokenizer = inputs.tokenizers[retriever_name]
            query_token_len = count_tokens(tokenizer, inputs.queries[query_id])
            doc_token_lens = {
                doc_id: _doc_token_len(retriever_name, doc_id) for doc_id in relevant
            }
            max_len = max_relevant_doc_token_len(doc_token_lens, relevant)
            exceeds = (
                None if max_len is None else max_len > inputs.limits[retriever_name]
            )
            row[covariate_column("query_token_len", retriever_name)] = query_token_len
            row[covariate_column("max_relevant_doc_token_len", retriever_name)] = max_len
            row[covariate_column("any_relevant_doc_exceeds_limit", retriever_name)] = exceeds
        rows.append(row)

    columns = ["query_id"] + [
        covariate_column(covariate, retriever_name)
        for retriever_name in retriever_names
        for covariate in COVARIATE_NAMES
    ]
    return pandas.DataFrame(rows, columns=columns)


def assert_covariates_run_independent(
    covariates: pandas.DataFrame, query_ids: Iterable[str]
) -> None:
    """Raises FailureBucketAssertionError if the covariate frame holds
    more than one row for any query_id, if any query_id in query_ids
    is absent from it, or if a max_relevant_doc_token_len__* /
    any_relevant_doc_exceeds_limit__* pair of cells disagrees about
    whether the Missing_Value_Sentinel is required for that (query_id,
    retriever) (Requirements 16.8, 16.9).

    The first two checks are belt to the structural braces of the
    one-row-per-query_id frame compute_token_length_covariates already
    builds. The third checks the pairing compute_token_length_covariates
    guarantees by construction -- `exceeds = None if max_len is None
    else ...` -- so the two columns' None-ness must always agree: a
    max_relevant_doc_token_len__* cell holding a numeric 0 is a
    legitimate value (a real judged-relevant document of length 0), so
    it is the DISAGREEMENT between the two paired columns that is the
    invariant violation, not a bare numeric 0."""
    duplicated_ids = covariates["query_id"][covariates["query_id"].duplicated(keep=False)]
    if not duplicated_ids.empty:
        raise FailureBucketAssertionError(
            "Covariate frame holds more than one row for query_id(s): "
            f"{sorted(set(duplicated_ids))}"
        )

    covariate_query_ids = set(covariates["query_id"])
    missing_ids = sorted(set(query_ids) - covariate_query_ids)
    if missing_ids:
        raise FailureBucketAssertionError(
            f"query_id(s) absent from the covariate frame: {missing_ids}"
        )

    max_len_columns = [
        column for column in covariates.columns if column.startswith("max_relevant_doc_token_len__")
    ]
    for max_len_column in max_len_columns:
        retriever_tag = max_len_column[len("max_relevant_doc_token_len__"):]
        exceeds_column = f"any_relevant_doc_exceeds_limit__{retriever_tag}"
        for query_id, max_len, exceeds in zip(
            covariates["query_id"], covariates[max_len_column], covariates[exceeds_column]
        ):
            max_len_missing = pandas.isna(max_len)
            exceeds_missing = exceeds is None or (
                isinstance(exceeds, float) and pandas.isna(exceeds)
            )
            if max_len_missing != exceeds_missing:
                raise FailureBucketAssertionError(
                    f"query_id {query_id!r}: {max_len_column!r} and "
                    f"{exceeds_column!r} disagree about whether the "
                    f"Missing_Value_Sentinel is required (max_len={max_len!r}, "
                    f"exceeds={exceeds!r}) -- they must be None together or "
                    "not at all"
                )


def attach_covariates(
    failure_buckets: pandas.DataFrame, covariates: pandas.DataFrame
) -> pandas.DataFrame:
    """Left-joins the covariate frame onto the per-query bucket frame
    ON `query_id` ALONE, never on (run_id, query_id), and returns the
    frame with columns exactly FAILURE_BUCKET_COLUMNS in that order.

    Joining on query_id is what makes Requirement 6.9 -- "the same six
    covariate values in every one of that query_id's rows" -- true by
    construction: there is one covariate row per query_id and nine
    bucket rows per query_id, so all nine receive the same values from
    the same source row. A join on (run_id, query_id) would require the
    covariate frame to carry a run_id it has no business knowing, and
    would make a per-run covariate value REPRESENTABLE, which is
    exactly what Requirement 16.9 forbids.

    Renders every covariate cell here, at the boundary (Requirement
    6.8): an int as a base-ten integer with no decimal point, a bool as
    the literal EXCEEDS_TEXT / WITHIN_TEXT non-coercible text, and a
    None as MISSING.

    Calls assert_unique_pairs on the result: a left join whose right
    side had a duplicated query_id would silently fan the frame out,
    and a row count that grew during a join is exactly the kind of
    failure a totality assertion exists to catch."""
    covariate_columns = [c for c in covariates.columns if c != "query_id"]
    rendered = covariates.copy()
    for column in covariate_columns:
        if column.startswith("any_relevant_doc_exceeds_limit__"):
            rendered[column] = rendered[column].map(
                lambda v: MISSING if pandas.isna(v) else (EXCEEDS_TEXT if v else WITHIN_TEXT)
            )
        else:
            rendered[column] = rendered[column].map(
                lambda v: MISSING if pandas.isna(v) else str(int(v))
            )

    merged = failure_buckets.merge(rendered, on="query_id", how="left")
    merged = merged.reindex(columns=list(FAILURE_BUCKET_COLUMNS))
    assert_unique_pairs(merged)
    return merged


# --- main orchestration ---


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.failure_buckets",
        description=(
            "Assigns the fixed Failure_Bucket and Contrast_Bucket taxonomies "
            "to results/per_query.csv, enriches the per-query report with "
            "per-query token-length covariates loaded from the already-"
            "cached BEIR SciFact corpus and Dense_Model tokenizers under "
            "data/, and writes results/failure_buckets.csv and "
            "results/failure_bucket_counts.csv."
        ),
    )
    parser.add_argument(
        "--per-query",
        type=Path,
        default=DEFAULT_PER_QUERY_PATH,
        help=f"Path to the Per_Query_Report (default: {DEFAULT_PER_QUERY_PATH}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the Sweep_Config YAML, read only for its data_dir field "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--buckets-out",
        type=Path,
        default=DEFAULT_BUCKETS_PATH,
        help=f"Output path for the Failure_Bucket_Report (default: {DEFAULT_BUCKETS_PATH}).",
    )
    parser.add_argument(
        "--counts-out",
        type=Path,
        default=DEFAULT_COUNTS_PATH,
        help=f"Output path for the Failure_Bucket_Counts_Report (default: {DEFAULT_COUNTS_PATH}).",
    )
    return parser.parse_args(argv)


def main(
    argv: Optional[List[str]] = None,
    *,
    covariate_inputs: Optional[CovariateInputs] = None,
) -> int:
    """CLI entry point: `python -m src.failure_buckets [--per-query PATH]
    [--config PATH] [--buckets-out PATH] [--counts-out PATH]`.

    Runs the Bucket_Assignment_Stage to completion (every
    Totality_Assertion included) before the Covariate_Enrichment_Stage
    loads anything, and writes neither report until every assertion in
    both stages has already passed (Requirements 2.5, 4.6, 5.5, 7.6,
    16.13, 16.17).

    `covariate_inputs` is a keyword-only Python-API testing seam, never
    exposed as a CLI argument: when provided, it is used in place of
    `load_sweep_config` + `load_covariate_inputs`, so a caller (a test)
    can exercise the full successful pipeline -- the printed summary
    included -- against a hand-written `CovariateInputs` built from
    literals, without a mock, a monkeypatch, or a real corpus/tokenizer
    (Requirement 15.10). `None` (the default, and the only path
    `python -m src.failure_buckets` itself ever takes) means "load for
    real", exactly as before.
    """
    args = _parse_args(argv)

    # --- Bucket_Assignment_Stage ---
    try:
        per_query = load_per_query(args.per_query)
    except FailureBucketInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        assert_no_separator_collision(per_query["run_id"].unique())
        failure_buckets = build_failure_buckets(per_query)
        assert_unique_pairs(failure_buckets)
        run_counts = build_run_counts(failure_buckets)
        contrast_set = build_declared_contrast_set(per_query["run_id"].unique())
        contrast_counts = build_contrast_counts(per_query, contrast_set)
        counts = build_failure_bucket_counts(run_counts, contrast_counts)
        assert_fraction_sums(counts)
    except (FailureBucketAssertionError, ContrastQuerySetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # The counts frame is FINAL here: the covariate stage never
    # touches it (Requirement 16.17).

    # --- Covariate_Enrichment_Stage ---
    if covariate_inputs is None:
        try:
            config = load_sweep_config(args.config)
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        try:
            inputs = load_covariate_inputs(config.data_dir, list(DENSE_MODEL_NAMES))
        except CovariateInputError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        inputs = covariate_inputs

    try:
        covariates = compute_token_length_covariates(
            sorted(per_query["query_id"].unique()), inputs
        )
        assert_covariates_run_independent(covariates, per_query["query_id"])
        failure_buckets = attach_covariates(failure_buckets, covariates)
    except (CovariateInputError, FailureBucketAssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Serialize, report, write ---

    num_run_ids = per_query["run_id"].nunique()
    print(f"per_query.csv: {len(per_query)} rows, {num_run_ids} run_id(s)")
    for run_id, group in per_query.groupby("run_id", sort=False):
        print(f"  {run_id}: {group['query_id'].nunique()} query_id(s)")
    num_family_aligned = sum(1 for a, _ in contrast_set if a == REFERENCE_RUN_ID)
    num_cross_strategy = len(contrast_set) - num_family_aligned
    print(
        f"Pair_Contrasts: {len(contrast_set)} "
        f"({num_family_aligned} family-aligned + {num_cross_strategy} dense cross-strategy)"
    )
    print(f"failure_buckets.csv: {len(failure_buckets)} data row(s)")
    print(
        f"failure_bucket_counts.csv: {len(counts)} data row(s) "
        f"({len(run_counts)} per-run + {len(contrast_counts)} per-contrast)"
    )

    limits_text = " ".join(
        f"{retriever_name}={limit}" for retriever_name, limit in inputs.limits.items()
    )
    print(f"effective_max_sequence_length: {limits_text}")
    num_sentinel_query_ids = 0
    for _query_id, group in covariates.groupby("query_id"):
        max_len_columns = [
            column for column in covariates.columns
            if column.startswith("max_relevant_doc_token_len__")
        ]
        if any(pandas.isna(group[column].iloc[0]) for column in max_len_columns):
            num_sentinel_query_ids += 1
    print(f"covariates computed: {len(covariates)} query_id(s)")
    print(f'covariates recorded as "NA": {num_sentinel_query_ids} query_id(s)')

    # Step 14: build BOTH CSV texts in memory, from the two finished
    # frames, before either write call -- so a serialization failure
    # on the SECOND frame cannot leave the first report written while
    # the second is not. write_failure_buckets/write_failure_bucket_counts
    # (below) re-derive the same text deterministically from the same
    # frames, so this is a pre-flight validation, not wasted work that
    # could disagree with what actually gets written.
    try:
        _build_csv_texts(failure_buckets, counts)
    except Exception as exc:
        print(f"ERROR: failed to serialize one of the two reports: {exc}", file=sys.stderr)
        return 1

    # Step 16: write buckets, then counts -- the only tier that can
    # raise FailureBucketWriteError, reached after every assertion in
    # both stages has already passed.
    try:
        write_failure_buckets(failure_buckets, Path(args.buckets_out))
        write_failure_bucket_counts(counts, Path(args.counts_out))
    except FailureBucketWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


def _build_csv_texts(
    failure_buckets: pandas.DataFrame, counts: pandas.DataFrame
) -> Tuple[str, str]:
    """Builds both output CSV texts in memory, without writing either,
    so `main` can serialize both before writing either (step 14 of the
    design's ordering)."""
    buckets_text = failure_buckets[list(FAILURE_BUCKET_COLUMNS)].to_csv(index=False)
    counts_ordered = counts[[field.name for field in dataclasses.fields(CountRow)]].copy()
    counts_ordered["count"] = counts_ordered["count"].astype(int)
    counts_ordered["fraction"] = counts_ordered["fraction"].map(
        lambda value: f"{value:.{FRACTION_DECIMALS}f}"
    )
    counts_text = counts_ordered.to_csv(index=False)
    return buckets_text, counts_text


if __name__ == "__main__":
    sys.exit(main())
