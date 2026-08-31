"""Generated_Answers_Report row schema and writer.

Persists the raw Generated_Answer for every Generation_Subset query,
alongside its prompt's untruncated token count, so a Claim can be
traced back to the text it was segmented from, and a
generation/truncation bug is visible from a committed artifact without
a re-run. Mirrors `src/groundedness_report.py`'s shape exactly: a
frozen dataclass schema plus a writer reusing
`src.report._atomic_write_text`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas

from src.errors import GeneratedAnswersWriteError
from src.report import _atomic_write_text


@dataclass(frozen=True)
class GeneratedAnswerRow:
    """One row of results/generated_answers.csv: exactly one
    `query_id` per Generation_Subset query. `prompt_token_count` is
    the *untruncated* token count of that query's prompt under the
    Generator_Model's own tokenizer -- i.e. the true input length
    before `GeneratorModel.generate()`'s `truncation=True` silently
    clamps it to the tokenizer's `model_max_length`. Comparing
    `prompt_token_count` against the Generator_Model's
    `max_input_tokens` (recorded once in `results/run_config.json`'s
    `"groundedness"` sub-object, not per row) shows exactly how much of
    each prompt was dropped, rather than leaving that truncation
    silent."""

    query_id: str
    prompt_token_count: int
    answer_text: str


def write_generated_answers_report(
    rows: List[GeneratedAnswerRow], output_path: Path
) -> None:
    """Writes `rows` to `output_path` (default
    results/generated_answers.csv) as a CSV, atomically, via
    src.report._atomic_write_text (temp file + os.replace, temp
    removed on failure). Columns fixed to GeneratedAnswerRow's field
    order; rows written in the order given. Raises
    GeneratedAnswersWriteError on any failure, leaving output_path
    either absent or byte-for-byte in its pre-run state.
    """
    output_path = Path(output_path)
    fieldnames = [f.name for f in dataclasses.fields(GeneratedAnswerRow)]
    try:
        frame = pandas.DataFrame(
            [dataclasses.asdict(row) for row in rows], columns=fieldnames
        )
        csv_text = frame.to_csv(index=False)
    except Exception as exc:
        raise GeneratedAnswersWriteError(
            f"failed to build generated-answers report for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(output_path, csv_text, failure_context="generated-answers report")
    except Exception as exc:
        raise GeneratedAnswersWriteError(str(exc)) from exc
