"""Sweep_Report row schema, the MISSING sentinel, and the report/run
config writers (Requirement 7, Requirement 8.3).

`SweepReportRow` and `MISSING` are the schema `run_sweep`
(`src/sweep_runner.py`) constructs its return value from.
`write_sweep_report` writes that schema to `results/sweep.csv`
atomically (temp file + `os.replace`), and `write_run_config_record`
writes the accompanying run configuration record (e.g.
`results/run_config.json`) -- the fixed seed, the fully resolved
`SweepConfig`, the `CorpusLoadReport` counts (`num_documents`,
`num_queries`, `num_qrel_pairs`) derived from the same corpus load that
fed the run, and the installed version of each of `beir`, `rank_bm25`,
`sentence-transformers`, `torch`, and `numpy`, read via
`importlib.metadata.version(...)` at run time -- per `design.md`'s
`src/report.py` section. Every `pathlib.Path` value in the record
(currently `sweep_config.data_dir` and `sweep_config.output_path`) is
rendered in POSIX form (forward slashes) rather than the host
platform's native separator, so the record is byte-for-byte portable
across machines regardless of which OS produced it -- the
reproducibility bar (tech.md) applies to the record describing the
run, not only to the metric columns it lets a reader cross-check.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas

from src.config import SweepConfig
from src.corpus_loader import CorpusLoadReport
from src.errors import ReportWriteError

# Sentinel written for any metric/timing value that could not be
# computed. Never emitted for a legitimately computed 0.0 -- the two
# are distinguishable both visually and by str-vs-float type when the
# CSV is re-parsed (Requirement 7.8).
MISSING = "NA"


@dataclass(frozen=True)
class SweepReportRow:
    """One row of `results/sweep.csv`: one retriever x cutoff
    combination (Requirement 7.3). `recall_at_k`, `ndcg_at_10`,
    `mrr_at_10`, `index_time`, and `query_latency` are each either a
    real float or the `MISSING` sentinel string; `run_id`, `retriever`,
    `chunking_strategy`, `k`, `num_queries_total`, and
    `num_queries_scored` are never missing.
    """

    run_id: str
    retriever: str
    chunking_strategy: str
    k: int
    recall_at_k: Union[float, str]
    ndcg_at_10: Union[float, str]
    mrr_at_10: Union[float, str]
    index_time: Union[float, str]
    query_latency: Union[float, str]
    num_queries_total: int
    num_queries_scored: int


# Packages whose installed version is recorded in every run config
# record (Requirement 8.3). Read via importlib.metadata.version(...)
# at run time -- never hard-coded -- so the record always reflects
# what's actually installed in the environment that produced the run,
# not a value copied from `requirements.txt` by hand.
_VERSION_TRACKED_PACKAGES: Tuple[str, ...] = (
    "beir",
    "rank_bm25",
    "sentence-transformers",
    "torch",
    "numpy",
)


def _atomic_write_text(
    output_path: Path,
    text: str,
    *,
    failure_context: str,
    newline: Union[str, None] = None,
) -> None:
    """Writes `text` to `output_path` atomically.

    Writes to a temp file (`output_path` with an added `.tmp` suffix)
    in the same directory, then swaps it into place with
    `os.replace()`. On any failure -- including creating the parent
    directory, writing the temp file, or the replace itself -- the
    temp file is removed (if it exists) and `ReportWriteError` is
    raised naming `output_path` and `failure_context`, so
    `output_path` is never left partially written or corrupted
    (Requirement 10.4, `design.md` Property 7).

    `newline` is forwarded to `Path.write_text` unchanged (default
    `None`, i.e. Python's own universal-newline translation of every
    `"\\n"` in `text` to `os.linesep`). CSV callers -- whose `text` was
    already produced by `pandas.DataFrame.to_csv()`, and so already
    contains an explicit `"\\r\\n"` on every line -- MUST pass
    `newline=""` so that embedded `"\\r\\n"` is written byte-for-byte
    instead of being run through a second, redundant `"\\n"` ->
    `os.linesep` translation that would double it into `"\\r\\r\\n"`
    on Windows. JSON/Markdown callers, whose `text` contains only bare
    `"\\n"`, are unaffected either way and keep the default.
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8", newline=newline)
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ReportWriteError(
            f"failed to write {failure_context} to {output_path}: {exc}"
        ) from exc


def write_sweep_report(rows: List[SweepReportRow], output_path: Path) -> None:
    """Writes `rows` to `output_path` (e.g. `results/sweep.csv`) as a
    CSV, atomically.

    Columns are fixed to `SweepReportRow`'s field order -- `run_id,
    retriever, chunking_strategy, k, recall_at_k, ndcg_at_10,
    mrr_at_10, index_time, query_latency, num_queries_total,
    num_queries_scored` -- regardless of `rows`' contents, and rows are
    written in the order given, i.e. the declared-grid order
    `run_sweep` (`src/sweep_runner.py`) produced them in. A row's
    `MISSING` (`"NA"`) cells are written as the literal string `"NA"`;
    a legitimately computed `0.0` is written as the float `0.0` -- the
    two remain distinguishable in the written file (Requirement 7.8).

    Raises `ReportWriteError` if building the DataFrame or writing the
    file fails for any reason. `output_path` is left untouched (either
    absent, or at whatever it previously contained) rather than
    partially written (Requirement 10.4).
    """
    output_path = Path(output_path)
    fieldnames = [f.name for f in dataclasses.fields(SweepReportRow)]
    try:
        frame = pandas.DataFrame(
            [dataclasses.asdict(row) for row in rows], columns=fieldnames
        )
        csv_text = frame.to_csv(index=False)
    except Exception as exc:
        raise ReportWriteError(
            f"failed to build sweep report for {output_path}: {exc}"
        ) from exc
    _atomic_write_text(output_path, csv_text, failure_context="sweep report", newline="")


def _json_default(value: object) -> str:
    """`default=` handler for the `json.dumps` call in
    `write_run_config_record`.

    Renders any `pathlib.Path` value via `Path.as_posix()` (forward
    slashes) instead of `str(value)`'s host-platform-native separator
    -- e.g. `Path("results/sweep.csv")` always serializes as
    `"results/sweep.csv"`, never `"results\\sweep.csv"` on Windows --
    so the written run config record is textually identical for a
    given logical path regardless of which OS produced it (the
    reproducibility record must be portable across machines, not just
    the metric values it accompanies). Any other value reaching this
    handler (i.e. anything `json` cannot serialize natively) falls
    back to `str(value)`, matching this module's previous behavior for
    non-`Path` values.
    """
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def write_run_config_record(
    config: SweepConfig,
    corpus_report: CorpusLoadReport,
    output_path: Path,
) -> None:
    """Writes the Requirement 8.3 run configuration record to
    `output_path` (e.g. `results/run_config.json`), atomically.

    The record has exactly four top-level keys: `seed` (the fixed
    random seed applied for the run), `sweep_config` (the fully
    resolved `SweepConfig` contents, as loaded and validated by
    `load_sweep_config`), `corpus_load_report` (the `num_documents`,
    `num_queries`, and `num_qrel_pairs` counts from the
    `CorpusLoadReport` that `load_scifact` returned for *this* run's
    corpus load -- the same counts `report.as_log_line()` prints to
    stdout, now also living in a committed artifact instead of only
    transient stdout), and `installed_versions` (the installed version
    of each of `beir`, `rank_bm25`, `sentence-transformers`, `torch`,
    and `numpy`, read via `importlib.metadata.version(...)` at run time
    -- never hard-coded). Used, alongside `results/sweep.csv`, for the
    manual rerun-identity diff described in Requirement 8.4.

    Every `pathlib.Path` value nested in the record (currently
    `sweep_config.data_dir` and `sweep_config.output_path`) is written
    via `_json_default` in POSIX form (forward slashes), never the
    host platform's native separator, so the record is portable across
    machines.

    Raises `ReportWriteError` if building or writing the record fails
    for any reason. `output_path` is left untouched rather than
    partially written (Requirement 10.4).
    """
    output_path = Path(output_path)
    try:
        record = {
            "seed": config.seed,
            "sweep_config": dataclasses.asdict(config),
            "corpus_load_report": dataclasses.asdict(corpus_report),
            "installed_versions": {
                package: importlib.metadata.version(package)
                for package in _VERSION_TRACKED_PACKAGES
            },
        }
        json_text = json.dumps(record, indent=2, default=_json_default)
    except Exception as exc:
        raise ReportWriteError(
            f"failed to build run config record for {output_path}: {exc}"
        ) from exc
    _atomic_write_text(output_path, json_text, failure_context="run config record")
