"""Per_Query_Report row schema and writer (sweep side).

`PerQueryReportRow` and `write_per_query_report` are the sweep-side
counterpart to `src/report.py`'s `SweepReportRow` / `write_sweep_report`:
this is the schema the extended `run_sweep` (`src/sweep_runner.py`)
constructs its per-query rows from, and the writer that persists them to
`results/per_query.csv` atomically, per `design.md`'s
`src/per_query_report.py` section (Requirement 1).

Every column in `PerQueryReportRow` is wide on cutoff -- the four recall
cutoffs are four separate fields on a single row, so no per-query metric
value is duplicated across multiple rows (Requirement 1.4) -- and none
of its columns carries a missing-value marker: a per-query metric value
is always computable from a scored query's ranked list and qrels, so
there is no analogue of `src/report.py`'s `MISSING` sentinel here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas

from src.errors import PerQueryReportError
from src.report import _atomic_write_text


@dataclass(frozen=True)
class PerQueryReportRow:
    """One row of `results/per_query.csv`: exactly one (run_id,
    query_id) pair (Requirement 1.2). Wide on cutoff -- the four recall
    cutoffs are four separate columns on a single row, so no per-query
    value is duplicated across rows (Requirement 1.4). Every metric
    column (`recall_at_1`, `recall_at_5`, `recall_at_10`,
    `recall_at_20`, `ndcg_at_10`, `mrr_at_10`) is a real float in
    `[0.0, 1.0]` (Requirement 1.6); `num_judged_relevant` is a
    non-negative int, derived only from the loaded qrels (Requirement
    1.5). None of these columns carries a missing marker: a per-query
    value is always computable from the ranked list and qrels for a
    scored query.
    """

    run_id: str
    retriever: str
    chunking_strategy: str
    query_id: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    mrr_at_10: float
    num_judged_relevant: int


def write_per_query_report(rows: List[PerQueryReportRow], output_path: Path) -> None:
    """Writes `rows` to `output_path` (e.g. `results/per_query.csv`) as
    a CSV, atomically.

    Columns are fixed to `PerQueryReportRow`'s field order --
    `run_id, retriever, chunking_strategy, query_id, recall_at_1,
    recall_at_5, recall_at_10, recall_at_20, ndcg_at_10, mrr_at_10,
    num_judged_relevant` -- regardless of `rows`' contents, and rows
    are written in the order given.

    Reuses `src.report._atomic_write_text` (temp file + `os.replace`,
    temp removed on any failure), so `output_path` is left either
    absent or byte-for-byte in its pre-run state, never partially
    written (Requirement 1.8). Raises `PerQueryReportError` -- the
    sweep-side analogue of `ReportWriteError` -- if building the
    DataFrame or writing the file fails for any reason.
    """
    output_path = Path(output_path)
    fieldnames = [f.name for f in dataclasses.fields(PerQueryReportRow)]
    try:
        frame = pandas.DataFrame(
            [dataclasses.asdict(row) for row in rows], columns=fieldnames
        )
        csv_text = frame.to_csv(index=False)
    except Exception as exc:
        raise PerQueryReportError(
            f"failed to build per-query report for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(output_path, csv_text, failure_context="per-query report")
    except Exception as exc:
        raise PerQueryReportError(
            f"failed to write per-query report to {output_path}: {exc}"
        ) from exc
