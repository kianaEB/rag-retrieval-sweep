"""Verification_Pass: checks every Numeric_Claim in
`README.md`/`SPEC.md`/`ANALYSIS.md`
against the committed traceability ledger `docs/numeric_traceability.csv`
and the artifact it cites (Requirement 12,
`.kiro/specs/repo-writeup/design.md`'s `src/verify_writeup_numbers.py`
section).

For each ledger row, `verify_row` performs two independent checks, in
order:

1. **Document-presence check.** Confirms `row.stated_value` still
   occurs, as a literal substring, in the full text of the document
   (`README.md`, `SPEC.md`, or `ANALYSIS.md`) the row cites. A miss
   returns immediately with `failure_mode="value_not_in_document"`,
   without ever touching the cited artifact.
2. **Ledger-to-artifact comparison.** Only reached once (1) passes.
   Resolves the cited artifact value(s) via `load_artifact_values`,
   applies the row's `computation` via `apply_computation`, rounds
   both the stated and the freshly computed value with
   `round_half_up` (round-half-up, not Python's round-half-to-even)
   at the row's declared `stated_precision`, and compares the two
   rounded strings for exact equality. A disagreement returns
   `failure_mode="artifact_mismatch"`.

A row whose `source_fields` column is the literal sentinel string
`"n/a"` or `"NA"` (the two non-numeric sentinels
`results/significance.csv` itself defines) is special-cased: it is not
a Numeric_Claim under the glossary's definition, so no artifact I/O or
arithmetic is attempted for it -- `row.stated_value` is compared
directly against that sentinel text as a plain string, and the row is
exempt from `load_ledger`'s `stated_value_matches_precision` format
check (a non-numeric sentinel has no decimal-place format to check).

`main` is invoked manually (`python -m src.verify_writeup_numbers
--repo-root .`), never from CI -- see `design.md`'s "What the
Verification_Pass does not automate" section.
"""

from __future__ import annotations

import argparse
import dataclasses
import decimal
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import pandas

from src.errors import TraceabilityFileError, VerificationSourceError

# The fixed enum of arithmetic operations a ledger row's `computation`
# column may name (Requirement 12.1's "using an arithmetic operation
# already specified elsewhere in this spec"). Adding a computation this
# enum doesn't already support requires editing apply_computation, not
# just adding a value to a ledger row -- a deliberate friction point.
_ALLOWED_COMPUTATIONS: Sequence[str] = (
    "copy",
    "ratio",
    "delta",
    "mean",
    "percentage",
    "sum",
    "half_ci_width",
    "complement_percentage",
    "wilson_ci_lower",
    "wilson_ci_upper",
)

# z-score for a 95% Wilson score interval -- the same 95% confidence
# level already used throughout this project (the nDCG@10 bootstrap CI,
# configs/significance.yaml's alpha). Fixed here rather than threaded
# through as a ledger-row parameter, since every confidence interval
# in this repo uses the same 95% level.
_WILSON_Z_95 = 1.959963984540054

# The two non-numeric sentinels a CSV artifact cell (or a ledger row's
# own source_fields column) may hold, mirroring src.report.MISSING
# ("NA": could not be computed) and src.significance.NOT_APPLICABLE
# ("n/a": correction/verdict does not apply to this row). Declared
# locally, rather than imported, so this module's dependency on the
# sweep/significance modules stays limited to the sentinel *values*
# both already fix, not their implementations.
_MISSING_SENTINEL = "NA"
_NOT_APPLICABLE_SENTINEL = "n/a"
_SENTINEL_VALUES = (_MISSING_SENTINEL, _NOT_APPLICABLE_SENTINEL)

# source_artifact values resolved as a row-selected CSV column.
# groundedness.csv, hand_checked_joined.csv, and generated_answers.csv
# (groundedness-gate spec) are resolved by the same generic
# row_selector.field logic as the session-1/significance-testing
# artifacts already listed here, plus the additive "all"/"__count__"
# and "col_a==col_b" extensions below.
_CSV_ARTIFACTS = (
    "sweep.csv",
    "significance.csv",
    "per_query.csv",
    "groundedness.csv",
    "hand_checked_joined.csv",
    "generated_answers.csv",
    "failure_buckets.csv",
    "failure_bucket_counts.csv",
)

_NDP_PATTERN = re.compile(r"(\d+)dp")
_PERCENTAGE_PATTERN = re.compile(r"percentage:(\d+)dp")

# Default paths, relative to --repo-root (Requirement 12).
DEFAULT_LEDGER_PATH = Path("docs/numeric_traceability.csv")
DEFAULT_ARTIFACTS_DIR = Path("results")


@dataclass(frozen=True)
class TraceabilityRow:
    """One row of `docs/numeric_traceability.csv` (Requirement 12.1,
    12.4). Field order matches the ledger's committed CSV column order
    exactly."""

    claim_id: str
    document: str
    location: str
    stated_value: str
    stated_precision: str
    source_artifact: str
    source_fields: str
    computation: str


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of `verify_row` for one `TraceabilityRow`.
    `failure_mode` is `None` on a match, `"value_not_in_document"` if
    the document-presence check failed, or `"artifact_mismatch"` if the
    ledger-to-artifact comparison disagreed."""

    claim_id: str
    matched: bool
    failure_mode: Optional[str]
    stated_rounded: str
    computed_rounded: str
    detail: str


def _strip_ratio_suffix(text: str) -> str:
    """Strips a single trailing `x` (a ratio suffix, e.g. `"268x"`),
    if present. Never strips a trailing `%` -- that suffix is handled
    only by the `percentage:Ndp` branches of `stated_value_matches_precision`
    and `_stated_value_as_float`, never generically."""
    return text[:-1] if text.endswith("x") else text


def stated_value_matches_precision(stated_value: str, stated_precision: str) -> bool:
    """Checks `stated_value`'s literal text against the format
    `stated_precision` declares -- a format check against text, never a
    numeric derivation (Requirement 12.2).

    Recognized `stated_precision` shapes:

    - `"integer"`: no `.` in `stated_value` (after stripping an
      optional trailing ratio `x` suffix, e.g. `"268x"`).
    - `"Ndp"` (e.g. `"4dp"`): exactly `N` digits after a single `.`
      (after stripping an optional trailing ratio `x` suffix).
    - `"percentage:Ndp"` (e.g. `"percentage:1dp"`): a trailing `%`
      with exactly `N` decimal digits before it.
    - `"exact"`: a non-numeric Numeric_Claim compared byte-for-byte
      against the string the cited artifact holds (e.g. a package
      version string like `"2.2.0"`, which is not shaped like any of
      the decimal-place formats above but is still a recorded
      configuration value under the glossary's definition). Any text
      matches `"exact"` -- the format check that matters for this
      shape happens later, in `verify_row`'s direct string comparison,
      not here.

    Any other `stated_precision` value is malformed and this returns
    `False` -- `load_ledger` raises `TraceabilityFileError` naming the
    offending row rather than treating a malformed row as a
    verification mismatch.
    """
    if stated_precision == "exact":
        return True

    percentage_match = _PERCENTAGE_PATTERN.fullmatch(stated_precision)
    if percentage_match:
        n = int(percentage_match.group(1))
        if not stated_value.endswith("%"):
            return False
        body = stated_value[:-1]
        if n == 0:
            return re.fullmatch(r"-?\d+", body) is not None
        decimal_match = re.fullmatch(r"-?\d+\.(\d+)", body)
        return bool(decimal_match) and len(decimal_match.group(1)) == n

    text = _strip_ratio_suffix(stated_value)

    if stated_precision == "integer":
        return "." not in text

    ndp_match = _NDP_PATTERN.fullmatch(stated_precision)
    if ndp_match:
        n = int(ndp_match.group(1))
        decimal_match = re.fullmatch(r"-?\d+\.(\d+)", text)
        return bool(decimal_match) and len(decimal_match.group(1)) == n

    return False


def _base_precision(stated_precision: str) -> str:
    """Strips the `"percentage:"` prefix from `stated_precision`
    (e.g. `"percentage:1dp"` -> `"1dp"`), for use with `round_half_up`,
    which only ever rounds to a plain decimal-place target -- the `%`
    suffix is a display concern handled by
    `stated_value_matches_precision` and `_stated_value_as_float`, not
    by `round_half_up` itself."""
    if stated_precision.startswith("percentage:"):
        return stated_precision.split(":", 1)[1]
    return stated_precision


def _stated_value_as_float(stated_value: str, stated_precision: str) -> float:
    """Parses `stated_value`'s numeric core into a `float`, stripping
    whichever display suffix `stated_precision` implies (`%` for
    `percentage:Ndp`, an optional ratio `x` otherwise) before calling
    `float(...)`."""
    text = stated_value[:-1] if stated_precision.startswith("percentage:") else _strip_ratio_suffix(
        stated_value
    )
    try:
        return float(text)
    except ValueError as exc:
        raise VerificationSourceError(
            f"stated_value {stated_value!r} is not numeric after stripping its "
            f"display suffix for stated_precision {stated_precision!r}: {exc}"
        ) from exc


def round_half_up(value: float, precision_spec: str) -> str:
    """Rounds `value` to the decimal-place target `precision_spec`
    names, using round-half-up (never Python's built-in `round()`,
    which performs round-half-to-even) -- Requirement 12.2.

    `precision_spec` is `"integer"` (round to the nearest whole
    number) or `"Ndp"` (round to `N` decimal places); a
    `"percentage:Ndp"` value is accepted as an alias for `"Ndp"` (the
    `%` suffix is a stated-text display concern handled elsewhere, not
    a rounding-target concern). Converts through `str(value)` first
    (never `decimal.Decimal(value)` directly), so the digits rounded
    are the same digits a human reading the float's usual string
    representation would see, avoiding a spurious mismatch caused by
    binary floating-point representation noise several digits past
    what either document actually states.
    """
    resolved_spec = _base_precision(precision_spec)
    if resolved_spec == "integer":
        target = decimal.Decimal("1")
    else:
        ndp_match = _NDP_PATTERN.fullmatch(resolved_spec)
        if not ndp_match:
            raise VerificationSourceError(f"unrecognized precision_spec: {precision_spec!r}")
        n = int(ndp_match.group(1))
        target = decimal.Decimal("1") if n == 0 else decimal.Decimal(f"1e-{n}")
    quantized = decimal.Decimal(str(value)).quantize(target, rounding=decimal.ROUND_HALF_UP)
    return str(quantized)


def apply_computation(computation: str, values: List[Union[float, str]]) -> Union[float, str]:
    """Applies the named arithmetic operation over `values`, resolved
    from a single artifact via `load_artifact_values` (Requirement
    12.1). `computation` must be one of `_ALLOWED_COMPUTATIONS`; any
    other value raises `VerificationSourceError` -- a row naming an
    unrecognized computation is malformed, not merely wrong.

    Value-count and ordering conventions, fixed by this function (a
    ledger row's `source_fields` must list its references in the order
    the computation expects):

    - `"copy"`: exactly 1 value, returned unchanged (numeric or not).
    - `"ratio"`: exactly 2 numeric values, `[numerator, denominator]`.
    - `"delta"`: exactly 2 numeric values, `[minuend, subtrahend]`
      (matching evaluation-integrity's "dense minus BM25" convention).
    - `"mean"`: 1 or more numeric values; their arithmetic mean.
    - `"percentage"`: exactly 1 numeric value, multiplied by 100.
    - `"sum"`: 1 or more numeric values; their sum.
    - `"half_ci_width"`: exactly 2 numeric values, `[ci_upper,
      ci_lower]`; returns `(ci_upper - ci_lower) / 2`.
    - `"complement_percentage"`: exactly 1 numeric value `x` expressed
      as a proportion in `[0, 1]`; returns `(1 - x) * 100`. Distinct
      from `"percentage"` (which is a plain `x * 100`, no complement) --
      kept as its own enum member, not a generalization of
      `"percentage"`, so a ledger row cannot silently flip which one
      applies. Used for a stated confidence level (e.g. "95%" from a
      recorded `alpha` of `0.05`), never for a raw fraction that should
      be reported as-is.
    - `"wilson_ci_lower"` / `"wilson_ci_upper"`: exactly 2 numeric
      values, `[successes, total]`; returns the lower/upper bound of
      the 95% Wilson score interval for a binomial proportion
      `successes / total` (Agreement_Rate's assertion-partition
      confidence interval -- SPEC.md's "Agreement_Rate, partitioned by
      declarative assertion vs. non-assertion" section). The Wilson
      interval, not a Wald/normal-approximation interval, because the
      Wald interval is a poor approximation at small `total` and can
      extend outside `[0, 1]`; Wilson stays well-behaved at n as small
      as 11 or 19. Two separate enum members, not one computation
      returning a pair, because `apply_computation`'s contract (like
      every other computation here) is one row -> one scalar -> one
      `round_half_up` comparison; the lower and upper bounds are two
      separate Numeric_Claims, each with its own ledger row.
    """
    if computation not in _ALLOWED_COMPUTATIONS:
        raise VerificationSourceError(f"unrecognized computation: {computation!r}")

    if computation == "copy":
        if len(values) != 1:
            raise VerificationSourceError(
                f"'copy' requires exactly 1 value, got {len(values)}"
            )
        return values[0]

    try:
        numeric_values = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise VerificationSourceError(
            f"computation {computation!r} requires numeric values, got {values!r}: {exc}"
        ) from exc

    if computation == "ratio":
        if len(numeric_values) != 2:
            raise VerificationSourceError(
                "'ratio' requires exactly 2 values: [numerator, denominator]"
            )
        numerator, denominator = numeric_values
        return numerator / denominator

    if computation == "delta":
        if len(numeric_values) != 2:
            raise VerificationSourceError(
                "'delta' requires exactly 2 values: [minuend, subtrahend]"
            )
        minuend, subtrahend = numeric_values
        return minuend - subtrahend

    if computation == "mean":
        if not numeric_values:
            raise VerificationSourceError("'mean' requires at least 1 value")
        return sum(numeric_values) / len(numeric_values)

    if computation == "percentage":
        if len(numeric_values) != 1:
            raise VerificationSourceError("'percentage' requires exactly 1 value")
        return numeric_values[0] * 100.0

    if computation == "sum":
        if not numeric_values:
            raise VerificationSourceError("'sum' requires at least 1 value")
        return sum(numeric_values)

    if computation == "half_ci_width":
        if len(numeric_values) != 2:
            raise VerificationSourceError(
                "'half_ci_width' requires exactly 2 values: [ci_upper, ci_lower]"
            )
        ci_upper, ci_lower = numeric_values
        return (ci_upper - ci_lower) / 2.0

    if computation == "complement_percentage":
        if len(numeric_values) != 1:
            raise VerificationSourceError("'complement_percentage' requires exactly 1 value")
        return (1.0 - numeric_values[0]) * 100.0

    # computation in ("wilson_ci_lower", "wilson_ci_upper")
    if len(numeric_values) != 2:
        raise VerificationSourceError(f"{computation!r} requires exactly 2 values: [successes, total]")
    successes, total = numeric_values
    if total <= 0:
        raise VerificationSourceError(f"{computation!r}: total must be positive, got {total!r}")
    if not (0.0 <= successes <= total):
        raise VerificationSourceError(
            f"{computation!r}: successes {successes!r} must be within [0, total={total!r}]"
        )
    z = _WILSON_Z_95
    p_hat = successes / total
    denominator = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denominator
    half_width = (
        z * ((p_hat * (1.0 - p_hat) / total) + (z * z) / (4.0 * total * total)) ** 0.5
    ) / denominator
    if computation == "wilson_ci_lower":
        return center - half_width
    return center + half_width


def load_ledger(path: Path) -> List[TraceabilityRow]:
    """Parses `docs/numeric_traceability.csv` into `TraceabilityRow`
    instances (Requirement 12.1, 12.4).

    Immediately after parsing each row -- before any artifact I/O,
    before `verify_row` is ever called -- calls
    `stated_value_matches_precision(row.stated_value,
    row.stated_precision)` on every row whose `source_fields` is not
    the sentinel-comparison special case (a sentinel row's
    `stated_value` is non-numeric text by definition and is exempt
    from this numeric-format check). A row that fails this check is a
    malformed ledger row -- its own two columns disagree about how the
    value is formatted -- so this raises `TraceabilityFileError` naming
    the offending `claim_id`, halting before verifying *any* row.

    Raises `TraceabilityFileError` if `path` is missing, cannot be
    parsed, or lacks a required column.
    """
    path = Path(path)
    if not path.is_file():
        raise TraceabilityFileError(f"traceability ledger not found: {path}")

    try:
        frame = pandas.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except Exception as exc:
        raise TraceabilityFileError(
            f"failed to parse traceability ledger {path}: {exc}"
        ) from exc

    required_columns = [f.name for f in dataclasses.fields(TraceabilityRow)]
    missing_columns = [c for c in required_columns if c not in frame.columns]
    if missing_columns:
        raise TraceabilityFileError(
            f"traceability ledger {path} is missing required column(s): {missing_columns}"
        )

    rows: List[TraceabilityRow] = []
    for _, record in frame.iterrows():
        row = TraceabilityRow(
            **{column: str(record[column]) for column in required_columns}
        )
        if row.source_fields.strip() not in _SENTINEL_VALUES:
            if not stated_value_matches_precision(row.stated_value, row.stated_precision):
                raise TraceabilityFileError(
                    f"claim_id {row.claim_id!r}: stated_value {row.stated_value!r} "
                    f"does not match its declared stated_precision {row.stated_precision!r}"
                )
        rows.append(row)
    return rows


def _read_csv_artifact(path: Path) -> pandas.DataFrame:
    if not path.is_file():
        raise VerificationSourceError(f"artifact file not found: {path}")
    try:
        # keep_default_na=False / na_values=[] preserve the "NA" /
        # "n/a" sentinel cells as literal text rather than converting
        # them to NaN -- pandas' default NA-value list includes both,
        # which would otherwise make the two sentinels indistinguishable
        # from a genuinely missing cell.
        return pandas.read_csv(path, keep_default_na=False, na_values=[])
    except Exception as exc:
        raise VerificationSourceError(f"failed to parse artifact {path}: {exc}") from exc


def _read_json_artifact(path: Path) -> dict:
    if not path.is_file():
        raise VerificationSourceError(f"artifact file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationSourceError(f"failed to parse artifact {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise VerificationSourceError(f"artifact {path} is not a JSON object")
    return data


def _coerce_artifact_cell(cell: Any) -> Union[float, str]:
    """Coerces one resolved CSV cell to a `float`, unless it is exactly
    one of the two non-numeric sentinels (`"NA"`/`"n/a"`) or otherwise
    not numeric (e.g. a `retriever` name or a `verdict` string), in
    which case it is returned as the literal string it already is."""
    text = str(cell)
    if text in _SENTINEL_VALUES:
        return text
    try:
        return float(text)
    except ValueError:
        return text


def _resolve_csv_reference(
    frame: pandas.DataFrame, reference: str, source_artifact: str
) -> Union[float, str]:
    """Resolves one `row_selector.field` reference (e.g.
    `"metric=ndcg_at_10.mean_diff"` or
    `"run_id=bm25__whole_document,k=1.index_time"`) against `frame`.

    `row_selector` is a comma-separated list of `key=value` exact-match
    filters; the field name is separated from the row selector by the
    *last* `.` in `reference` (no field name or filter value in this
    repository's artifacts contains a `.`, so this split is
    unambiguous). Raises `VerificationSourceError` if the reference is
    malformed, the selector matches zero or more than one row, or the
    named field/column is absent.

    Two additive extensions, needed for Quarantine_Rate
    (groundedness-gate spec), compose with the `row_selector.field`
    dot-split unchanged:

    - If `row_selector` is the literal string `"all"`, per-`key=value`
      filtering is skipped entirely and every row of `frame` matches --
      used when no filtering is needed (e.g. the denominator of
      Quarantine_Rate, the total row count).
    - If `field` is the literal sentinel `"__count__"`, the "must match
      exactly 1 row" check is skipped and `float(len(matched))` (the
      count of matching rows) is returned instead of reading a column
      value from a single row.
    """
    if "." not in reference:
        raise VerificationSourceError(
            f"malformed field reference {reference!r} for {source_artifact}: "
            f"expected 'row_selector.field'"
        )
    row_selector_str, field = reference.rsplit(".", 1)

    if row_selector_str == "all":
        matched = frame
    else:
        mask = pandas.Series(True, index=frame.index)
        for pair in row_selector_str.split(","):
            if "=" not in pair:
                raise VerificationSourceError(
                    f"malformed row selector {row_selector_str!r} for {source_artifact}"
                )
            key, value = pair.split("=", 1)
            if key not in frame.columns:
                raise VerificationSourceError(
                    f"row selector column {key!r} not found in {source_artifact}"
                )
            mask &= frame[key].astype(str) == value
        matched = frame[mask]

    if field == "__count__":
        return float(len(matched))

    if len(matched) != 1:
        raise VerificationSourceError(
            f"row selector {row_selector_str!r} matched {len(matched)} row(s) "
            f"in {source_artifact}, expected exactly 1"
        )
    if field not in matched.columns:
        raise VerificationSourceError(f"field {field!r} not found in {source_artifact}")
    return _coerce_artifact_cell(matched.iloc[0][field])


_JSON_SEGMENT_PATTERN = re.compile(r"([^.\[\]]+)((?:\[\d+\])*)")


def _resolve_json_path(data: dict, reference: str, source_artifact: str) -> Union[float, str]:
    """Walks a dotted JSON path (e.g. `"corpus_load_report.num_documents"`,
    `"sweep_config.retrievers[0].k1"`) through `data`. Raises
    `VerificationSourceError` if any segment's key is absent, an index
    is out of range, or the final value is not a scalar.
    """
    current: Any = data
    for segment in reference.split("."):
        match = _JSON_SEGMENT_PATTERN.fullmatch(segment)
        if not match:
            raise VerificationSourceError(
                f"malformed JSON path segment {segment!r} in {reference!r} for {source_artifact}"
            )
        key = match.group(1)
        if not isinstance(current, dict) or key not in current:
            raise VerificationSourceError(
                f"key {key!r} not found while resolving {reference!r} in {source_artifact}"
            )
        current = current[key]
        for index_str in re.findall(r"\[(\d+)\]", match.group(2)):
            index = int(index_str)
            if not isinstance(current, list) or index >= len(current):
                raise VerificationSourceError(
                    f"index [{index}] out of range while resolving {reference!r} "
                    f"in {source_artifact}"
                )
            current = current[index]
    if isinstance(current, (dict, list)):
        raise VerificationSourceError(
            f"path {reference!r} in {source_artifact} resolves to a non-scalar value"
        )
    return current


def _resolve_top_level_key(data: dict, reference: str, source_artifact: str) -> Union[float, str]:
    """Resolves `reference` against `data` (`token_length_report.json`).

    Delegates to `_resolve_json_path` -- the same dotted-path-plus-
    `[index]` resolver `run_config.json` already uses -- rather than a
    bespoke top-level-only lookup: for a plain key with no `.` or `[]`
    (e.g. `"fraction_exceeding"`, the pre-existing top-level fields
    that describe the `whole_document` x `all-MiniLM-L6-v2` cell
    specifically), this resolves identically to a direct dict lookup.
    For a nested reference into the full-grid-chunking-sweep spec's
    6-cell `"cells"` list (e.g. `"cells[1].fraction_exceeding"`), the
    same call now also reaches any of the 6 per-`(Chunking_Strategy,
    dense model)` cells -- needed to make every cell's own number
    traceable, not only the two cells whose values happen to be
    duplicated at the top level."""
    return _resolve_json_path(data, reference, source_artifact)


def _resolve_column_equality_reference(
    frame: pandas.DataFrame, reference: str, source_artifact: str
) -> float:
    """Resolves a `col_a==col_b` reference (e.g.
    `"judge_verdict==hand_label"`, Agreement_Rate's resolution against
    `hand_checked_joined.csv`) to the fraction of rows where the two
    named columns agree, as a single float over the whole file --
    genuinely a two-column aggregate, not a single-cell
    `row_selector.field` lookup, so this is resolved directly from
    `frame` rather than composing with `_resolve_csv_reference`.

    Raises `VerificationSourceError` if either column is absent from
    `frame`, or if `frame` has zero rows (a mean over zero rows is
    undefined).
    """
    col_a, col_b = reference.split("==", 1)
    if col_a not in frame.columns:
        raise VerificationSourceError(
            f"column {col_a!r} not found in {source_artifact} (from {reference!r})"
        )
    if col_b not in frame.columns:
        raise VerificationSourceError(
            f"column {col_b!r} not found in {source_artifact} (from {reference!r})"
        )
    if len(frame) == 0:
        raise VerificationSourceError(
            f"{source_artifact} has zero rows; cannot resolve {reference!r}"
        )
    return float((frame[col_a].astype(str) == frame[col_b].astype(str)).mean())


def load_artifact_values(
    source_artifact: str, source_fields: str, artifacts_dir: Path
) -> List[Union[float, str]]:
    """Resolves every `;`-separated reference in `source_fields` against
    `source_artifact` under `artifacts_dir`, returning one value per
    reference, in order (Requirement 12.1).

    Dispatches on `source_artifact`: for `sweep.csv`/`significance.csv`/
    `per_query.csv`/`groundedness.csv`/`hand_checked_joined.csv`, each
    reference is resolved against the parsed CSV -- first checking
    whether it is a `col_a==col_b` column-equality reference
    (`_resolve_column_equality_reference`, e.g. Agreement_Rate's
    `judge_verdict==hand_label`), and only if not, falling back to the
    `row_selector.field` resolution (`_resolve_csv_reference`, which
    itself also supports the `all` row selector and the `__count__`
    field sentinel for Quarantine_Rate's count-aggregate references);
    for `run_config.json`, each reference is a dotted JSON path
    (`_resolve_json_path`); for `token_length_report.json`, each
    reference is a top-level key (`_resolve_top_level_key`). Raises
    `VerificationSourceError` if the artifact file is absent, a row
    selector matches zero or more than one row, or a named field/key is
    absent.
    """
    artifacts_dir = Path(artifacts_dir)
    references = [ref.strip() for ref in source_fields.split(";")]

    if source_artifact in _CSV_ARTIFACTS:
        frame = _read_csv_artifact(artifacts_dir / source_artifact)
        values: List[Union[float, str]] = []
        for ref in references:
            if "==" in ref:
                values.append(_resolve_column_equality_reference(frame, ref, source_artifact))
            else:
                values.append(_resolve_csv_reference(frame, ref, source_artifact))
        return values

    if source_artifact == "run_config.json":
        data = _read_json_artifact(artifacts_dir / source_artifact)
        return [_resolve_json_path(data, ref, source_artifact) for ref in references]

    if source_artifact == "token_length_report.json":
        data = _read_json_artifact(artifacts_dir / source_artifact)
        return [_resolve_top_level_key(data, ref, source_artifact) for ref in references]

    raise VerificationSourceError(f"unsupported source_artifact: {source_artifact!r}")


def verify_row(row: TraceabilityRow, artifacts_dir: Path, repo_root: Path) -> VerificationResult:
    """Verifies one `TraceabilityRow` (Requirement 12.2, 12.3):

    1. Document-presence check: `row.stated_value` must occur, as a
       literal substring, in `(repo_root / row.document)`'s full text.
       On a miss, returns immediately with
       `failure_mode="value_not_in_document"` -- the artifact is never
       consulted.
    2. Sentinel short-circuit: if `row.source_fields` is exactly `"n/a"`
       or `"NA"`, compares `row.stated_value` against that sentinel
       text directly (no rounding, no artifact I/O) -- Requirement
       12.1's tracing condition does not apply to a non-numeric
       sentinel.
    3. Ledger-to-artifact comparison: resolves the cited value(s) via
       `load_artifact_values`, applies `row.computation`, rounds both
       the stated and computed values with `round_half_up` at
       `row.stated_precision`, and compares the rounded strings for
       exact equality.

    Raises `VerificationSourceError` if step 3's artifact resolution or
    computation fails -- a row that cannot be resolved is a hard
    failure, not a mismatch.
    """
    document_path = Path(repo_root) / row.document
    try:
        document_text = document_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationSourceError(f"failed to read document {document_path}: {exc}") from exc

    if row.stated_value not in document_text:
        return VerificationResult(
            claim_id=row.claim_id,
            matched=False,
            failure_mode="value_not_in_document",
            stated_rounded=row.stated_value,
            computed_rounded="",
            detail=f"{row.stated_value!r} not found in {row.document}",
        )

    if row.source_fields.strip() in _SENTINEL_VALUES:
        sentinel = row.source_fields.strip()
        matched = row.stated_value == sentinel
        return VerificationResult(
            claim_id=row.claim_id,
            matched=matched,
            failure_mode=None if matched else "artifact_mismatch",
            stated_rounded=row.stated_value,
            computed_rounded=sentinel,
            detail=(
                "sentinel-to-sentinel match"
                if matched
                else f"stated {row.stated_value!r} != sentinel {sentinel!r}"
            ),
        )

    if row.stated_precision == "exact":
        # A non-numeric Numeric_Claim (e.g. a package version string):
        # compared verbatim against the cited artifact's value, never
        # parsed as a float or rounded -- round_half_up has no defined
        # target for text that is not a decimal number.
        values = load_artifact_values(row.source_artifact, row.source_fields, artifacts_dir)
        computed_value = apply_computation(row.computation, values)
        computed_text = str(computed_value)
        matched = row.stated_value == computed_text
        return VerificationResult(
            claim_id=row.claim_id,
            matched=matched,
            failure_mode=None if matched else "artifact_mismatch",
            stated_rounded=row.stated_value,
            computed_rounded=computed_text,
            detail="" if matched else f"stated {row.stated_value!r} != computed {computed_text!r}",
        )

    base_precision = _base_precision(row.stated_precision)
    stated_float = _stated_value_as_float(row.stated_value, row.stated_precision)
    stated_rounded = round_half_up(stated_float, base_precision)

    values = load_artifact_values(row.source_artifact, row.source_fields, artifacts_dir)
    computed_value = apply_computation(row.computation, values)
    try:
        computed_float = float(computed_value)
    except (TypeError, ValueError) as exc:
        raise VerificationSourceError(
            f"claim_id {row.claim_id!r}: computed value {computed_value!r} is not "
            f"numeric, cannot be rounded and compared: {exc}"
        ) from exc
    computed_rounded = round_half_up(computed_float, base_precision)

    matched = stated_rounded == computed_rounded
    return VerificationResult(
        claim_id=row.claim_id,
        matched=matched,
        failure_mode=None if matched else "artifact_mismatch",
        stated_rounded=stated_rounded,
        computed_rounded=computed_rounded,
        detail="" if matched else f"stated {stated_rounded} != computed {computed_rounded}",
    )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.verify_writeup_numbers",
        description=(
            "Checks every Numeric_Claim row in docs/numeric_traceability.csv "
            "against its cited document (README.md/SPEC.md/ANALYSIS.md) and "
            "its cited artifact under results/. Prints MATCH/MISMATCH per row "
            "and a summary; exits 0 only if every row matched."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root all other paths are resolved against (default: '.').",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help=f"Path to the traceability ledger, relative to --repo-root (default: {DEFAULT_LEDGER_PATH}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing the cited artifacts, relative to --repo-root (default: {DEFAULT_ARTIFACTS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: `python -m src.verify_writeup_numbers [--repo-root
    PATH] [--ledger PATH] [--artifacts-dir PATH]`.

    1. Loads `docs/numeric_traceability.csv` via `load_ledger`. On
       `TraceabilityFileError`, prints and returns non-zero -- no
       partial verification is reported as a pass.
    2. For each row (file order), calls `verify_row`. A raised
       `VerificationSourceError` is reported as a hard failure for that
       row (printed, counted as unmatched) without halting the
       remaining rows -- every ledgered row gets checked in a single
       run.
    3. Prints one line per row (`MATCH`, or `MISMATCH`/`ERROR` with
       detail) plus a summary count.
    4. Returns `0` only if every row matched; otherwise `1`.
    """
    args = _parse_args(argv)
    repo_root = Path(args.repo_root)
    ledger_path = repo_root / args.ledger
    artifacts_dir = repo_root / args.artifacts_dir

    try:
        rows = load_ledger(ledger_path)
    except TraceabilityFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    match_count = 0
    mismatch_count = 0
    for row in rows:
        try:
            result = verify_row(row, artifacts_dir, repo_root)
        except VerificationSourceError as exc:
            print(f"{row.claim_id}: ERROR {exc}")
            mismatch_count += 1
            continue

        if result.matched:
            print(f"{row.claim_id}: MATCH")
            match_count += 1
        else:
            print(
                f"{row.claim_id}: MISMATCH failure_mode={result.failure_mode} "
                f"stated={result.stated_rounded!r} computed={result.computed_rounded!r}"
            )
            mismatch_count += 1

    print(f"SUMMARY: {match_count} matched, {mismatch_count} mismatched, {len(rows)} total")
    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
