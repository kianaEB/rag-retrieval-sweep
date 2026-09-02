"""Groundedness_Report row schema and writer (Requirement 8).

Mirrors `src/per_query_report.py`'s shape exactly: a frozen dataclass
schema plus a writer reusing `src.report._atomic_write_text`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas

from src.errors import GroundednessReportWriteError
from src.report import _atomic_write_text


@dataclass(frozen=True)
class GroundednessReportRow:
    """One row of results/groundedness.csv: exactly one (query_id,
    claim_index) pair (Requirement 8.1). The first six columns'
    names and order match Requirement 8 Criterion 2 exactly. None of
    these columns carries a missing-value marker -- a row is either
    fully computed or the whole run halts before this writer is ever
    called (Requirement 8.5).

    `matched_sentence` is a live correctness-fix extension beyond
    Requirement 8.2's original schema: the Judge_Model now scores a
    Claim against each sentence of the Retrieved_Context individually
    (`nli-deberta-v3-xsmall` was trained on single-sentence premises,
    not multi-hundred-token concatenated abstracts) and `judge_score`
    is the maximum entailment probability across those per-sentence
    comparisons. `matched_sentence` records which retrieved sentence
    produced that maximum, so a hand-labeller reading this file can
    see exactly what the Judge_Model matched the Claim against,
    without re-running the gate."""

    query_id: str
    claim_index: int
    claim_text: str
    groundedness_verdict: str  # "SUPPORTED" or "NOT_SUPPORTED"
    judge_score: float  # entailment probability, [0.0, 1.0]
    quarantine_decision: bool
    matched_sentence: str  # the Retrieved_Context sentence that produced judge_score


def write_groundedness_report(rows: List[GroundednessReportRow], output_path: Path) -> None:
    """Writes `rows` to `output_path` (default results/groundedness.csv)
    as a CSV, atomically, via src.report._atomic_write_text (temp file
    + os.replace, temp removed on failure). Columns fixed to
    GroundednessReportRow's field order; rows written in the order
    given (Requirement 8.1). Every produced Claim's row is retained
    regardless of its verdict or quarantine_decision -- this function
    never filters `rows` (Requirement 8.4). Raises
    GroundednessReportWriteError on any failure, leaving output_path
    either absent or byte-for-byte in its pre-run state (Requirement
    8.5).
    """
    output_path = Path(output_path)
    fieldnames = [f.name for f in dataclasses.fields(GroundednessReportRow)]
    try:
        frame = pandas.DataFrame(
            [dataclasses.asdict(row) for row in rows], columns=fieldnames
        )
        csv_text = frame.to_csv(index=False)
    except Exception as exc:
        raise GroundednessReportWriteError(
            f"failed to build groundedness report for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(
            output_path, csv_text, failure_context="groundedness report", newline=""
        )
    except Exception as exc:
        raise GroundednessReportWriteError(str(exc)) from exc
