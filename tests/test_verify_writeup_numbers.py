"""Test suite for `src/verify_writeup_numbers.py` (repo-writeup spec,
Requirement 12).

Covers `round_half_up`'s tie-breaking behavior against Python's own
`round()` on the same input (Property 5); each member of the fixed
computation enum (`copy`, `ratio`, `delta`, `mean`, `percentage`,
`sum`, `half_ci_width`, `complement_percentage`) against a small
literal input with an independently hand-computed expected output,
plus an unrecognized `computation` value raising
`VerificationSourceError`;
`stated_value_matches_precision` against one matching case per
recognized `stated_precision` shape and one deliberately inconsistent
case, plus `load_ledger` raising `TraceabilityFileError` naming the
offending `claim_id` for a fixture row that fails that check;
`verify_row` (Property 6) on a matching pair and on a pair that would
falsely match under naive tolerance-based float comparison but
correctly mismatches once both sides are rounded to the stated
precision; and `verify_row`'s document-presence check (Property 7),
succeeding when the stated value is present in a fixture document and
failing with `failure_mode="value_not_in_document"` when the fixture
document has been doctored to no longer contain it, even though the
cited artifact fixture's value would otherwise match.

This module imports only `src.verify_writeup_numbers`, plus `json` and
`pytest` for building fixtures. It does not import any retriever or
corpus-loading module, and reads no file under the real `results/` or
`docs/` directories -- every "document" and "artifact" a test needs is
written into `tmp_path` (pytest's built-in temporary-directory
fixture), standing in for `repo_root / row.document` and
`artifacts_dir / row.source_artifact` respectively.
"""

from __future__ import annotations

import json

import pytest

from src.verify_writeup_numbers import (
    TraceabilityFileError,
    TraceabilityRow,
    VerificationSourceError,
    apply_computation,
    load_ledger,
    round_half_up,
    stated_value_matches_precision,
    verify_row,
)

# ---------------------------------------------------------------------------
# round_half_up -- Property 5 (round-half-up vs. Python's round-half-to-even)
# ---------------------------------------------------------------------------


def test_round_half_up_ties_away_from_zero_unlike_python_round():
    # Property 5: the digit immediately past the stated precision is
    # exactly 5 -- round_half_up rounds away from zero at that digit,
    # where Python's built-in round() rounds to the nearest even digit
    # instead. Both assertions document why a bespoke function is
    # needed, not just what it returns.
    assert round_half_up(0.125, "2dp") == "0.13"
    assert round(0.125, 2) == 0.12


# ---------------------------------------------------------------------------
# apply_computation -- one case per _ALLOWED_COMPUTATIONS member, plus the
# unrecognized-computation failure
# ---------------------------------------------------------------------------


def test_apply_computation_copy_returns_the_single_value_unchanged():
    assert apply_computation("copy", [0.5]) == 0.5


def test_apply_computation_copy_passes_through_non_numeric_values():
    assert apply_computation("copy", ["bm25"]) == "bm25"


def test_apply_computation_ratio():
    # 10.0 / 4.0 = 2.5, hand-computed.
    assert apply_computation("ratio", [10.0, 4.0]) == pytest.approx(2.5, abs=1e-9)


def test_apply_computation_delta():
    # 0.55 - 0.60 = -0.05, hand-computed.
    assert apply_computation("delta", [0.55, 0.60]) == pytest.approx(-0.05, abs=1e-9)


def test_apply_computation_mean():
    # (0.2 + 0.4 + 0.6) / 3 = 0.4, hand-computed.
    assert apply_computation("mean", [0.2, 0.4, 0.6]) == pytest.approx(0.4, abs=1e-9)


def test_apply_computation_percentage():
    # 0.05 * 100 = 5.0, hand-computed.
    assert apply_computation("percentage", [0.05]) == pytest.approx(5.0, abs=1e-9)


def test_apply_computation_sum():
    # 1.0 + 2.0 + 3.0 + 4.0 = 10.0, hand-computed.
    assert apply_computation("sum", [1.0, 2.0, 3.0, 4.0]) == pytest.approx(10.0, abs=1e-9)


def test_apply_computation_half_ci_width():
    # (0.5 - 0.1) / 2 = 0.2, hand-computed; [ci_upper, ci_lower] order.
    assert apply_computation("half_ci_width", [0.5, 0.1]) == pytest.approx(0.2, abs=1e-9)


def test_apply_computation_complement_percentage():
    # (1 - 0.05) * 100 = 95.0, hand-computed -- a stated 95% confidence
    # level derived from a recorded alpha of 0.05.
    assert apply_computation("complement_percentage", [0.05]) == pytest.approx(95.0, abs=1e-9)


def test_apply_computation_unrecognized_computation_raises():
    with pytest.raises(VerificationSourceError):
        apply_computation("logarithm", [1.0])


# ---------------------------------------------------------------------------
# stated_value_matches_precision -- one matching case per recognized shape,
# plus a deliberately inconsistent case
# ---------------------------------------------------------------------------


def test_stated_value_matches_precision_integer_shape():
    assert stated_value_matches_precision("5183", "integer")


def test_stated_value_matches_precision_ndp_shape():
    assert stated_value_matches_precision("0.1234", "4dp")


def test_stated_value_matches_precision_percentage_ndp_shape():
    assert stated_value_matches_precision("12.5%", "percentage:1dp")


def test_stated_value_matches_precision_ratio_x_suffix_shape():
    # The ratio "Nx" suffix is treated the same as integer/Ndp on the
    # digits preceding the "x".
    assert stated_value_matches_precision("268x", "integer")


def test_stated_value_matches_precision_deliberately_inconsistent_case():
    # "-0.007" shows 3 decimal digits, not the 4 its declared
    # stated_precision claims -- must fail.
    assert not stated_value_matches_precision("-0.007", "4dp")


def test_load_ledger_raises_traceability_file_error_naming_the_claim_id(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    ledger_path.write_text(
        "claim_id,document,location,stated_value,stated_precision,"
        "source_artifact,source_fields,computation\n"
        "bad-precision-claim,README.md,para1,-0.007,4dp,"
        "run_config.json,a.b,copy\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceabilityFileError) as exc_info:
        load_ledger(ledger_path)
    assert "bad-precision-claim" in str(exc_info.value)


# ---------------------------------------------------------------------------
# verify_row -- Property 6 (rounded-string comparison, not raw tolerance)
# ---------------------------------------------------------------------------


def test_verify_row_reports_match_for_a_correctly_rounded_pair(tmp_path):
    (tmp_path / "README.md").write_text(
        "The measured value is 0.55 in this run.\n", encoding="utf-8"
    )
    artifacts_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    (artifacts_dir / "token_length_report.json").write_text(
        json.dumps({"fraction_exceeding": 0.55}), encoding="utf-8"
    )
    row = TraceabilityRow(
        claim_id="match-1",
        document="README.md",
        location="para1",
        stated_value="0.55",
        stated_precision="2dp",
        source_artifact="token_length_report.json",
        source_fields="fraction_exceeding",
        computation="copy",
    )
    result = verify_row(row, artifacts_dir, tmp_path)
    assert result.matched is True
    assert result.failure_mode is None


def test_verify_row_rounding_catches_a_mismatch_naive_tolerance_would_hide(tmp_path):
    # Naive abs(a - b) < epsilon comparison (e.g. epsilon = 0.01) would
    # call stated "0.12" and computed 0.125 a match, since their raw
    # difference is only 0.005. round_half_up at the stated 2dp
    # precision instead rounds 0.125 up to "0.13" (Property 5), which
    # disagrees with the stated "0.12" once both are compared as
    # rounded strings -- exactly the false-match a tolerance-based
    # comparison would have let through.
    assert abs(0.12 - 0.125) < 0.01  # what a naive comparison would see

    (tmp_path / "README.md").write_text(
        "The measured value is 0.12 in this run.\n", encoding="utf-8"
    )
    artifacts_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    (artifacts_dir / "token_length_report.json").write_text(
        json.dumps({"fraction_exceeding": 0.125}), encoding="utf-8"
    )
    row = TraceabilityRow(
        claim_id="mismatch-1",
        document="README.md",
        location="para1",
        stated_value="0.12",
        stated_precision="2dp",
        source_artifact="token_length_report.json",
        source_fields="fraction_exceeding",
        computation="copy",
    )
    result = verify_row(row, artifacts_dir, tmp_path)
    assert result.matched is False
    assert result.failure_mode == "artifact_mismatch"


# ---------------------------------------------------------------------------
# verify_row's document-presence check -- Property 7
# ---------------------------------------------------------------------------


def test_verify_row_document_presence_succeeds_when_value_is_present(tmp_path):
    (tmp_path / "SPEC.md").write_text(
        "Corpus documents total 5183 in this run.\n", encoding="utf-8"
    )
    artifacts_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    (artifacts_dir / "token_length_report.json").write_text(
        json.dumps({"num_documents_total": 5183}), encoding="utf-8"
    )
    row = TraceabilityRow(
        claim_id="presence-ok",
        document="SPEC.md",
        location="threats",
        stated_value="5183",
        stated_precision="integer",
        source_artifact="token_length_report.json",
        source_fields="num_documents_total",
        computation="copy",
    )
    result = verify_row(row, artifacts_dir, tmp_path)
    assert result.failure_mode is None
    assert result.matched is True


def test_verify_row_document_presence_failure_even_when_artifact_matches(tmp_path):
    # The fixture document has been doctored to no longer contain the
    # stated value anywhere, even though the cited artifact fixture's
    # value would otherwise match exactly -- this must be reported as
    # a document-presence failure, never masked as a MATCH by the
    # artifact agreeing.
    (tmp_path / "SPEC.md").write_text(
        "Corpus documents total nine thousand in this run.\n", encoding="utf-8"
    )
    artifacts_dir = tmp_path / "results"
    artifacts_dir.mkdir()
    (artifacts_dir / "token_length_report.json").write_text(
        json.dumps({"num_documents_total": 5183}), encoding="utf-8"
    )
    row = TraceabilityRow(
        claim_id="presence-stale",
        document="SPEC.md",
        location="threats",
        stated_value="5183",
        stated_precision="integer",
        source_artifact="token_length_report.json",
        source_fields="num_documents_total",
        computation="copy",
    )
    result = verify_row(row, artifacts_dir, tmp_path)
    assert result.matched is False
    assert result.failure_mode == "value_not_in_document"
