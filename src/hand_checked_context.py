"""Hand_Checked_Sample labelling aid: `results/hand_checked_sample_context.md`.

A read-only reading aid for the human doing the hand-labelling, *not*
an artifact any Numeric_Claim is ever cited against (no
`source_artifact` entry is added to `src.verify_writeup_numbers` for
it) and not a file the Groundedness_Runner or any code ever reads
back. It exists so a human labelling
`results/hand_checked_sample.csv` can see the same query text and
Retrieved_Context the Generator_Model and Judge_Model saw for that
Claim, without re-running anything.

Deliberately excludes `groundedness_verdict`, `judge_score`,
`matched_sentence`, `quarantine_decision`, and the Generated_Answer
itself -- the same independence
`export_hand_checked_sample`/`HandCheckedSampleRow` already enforce for
`results/hand_checked_sample.csv` (Requirement 10.8): a human assigning
a Hand_Label must not be anchored by the Judge_Model's own
determination, or shown text (the Generated_Answer) that isn't the
Claim itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.errors import HandCheckedContextWriteError
from src.hand_checked_sample import HandCheckedSampleRow
from src.report import _atomic_write_text


def render_hand_checked_context_markdown(
    rows: List[HandCheckedSampleRow],
    query_text_by_id: Dict[str, str],
    retrieved_context_by_query: Dict[str, List[str]],
) -> str:
    """Renders one Markdown section per `rows` entry, in the given
    order (the same order `results/hand_checked_sample.csv` is stored
    in -- see `read_hand_checked_sample_rows`), each showing that row's
    `query_id`, the full query text (`query_text_by_id[query_id]`),
    the claim text, and the ordered Retrieved_Context documents
    (`retrieved_context_by_query[query_id]`) -- the same top-k
    documents the Generator_Model and Judge_Model saw for that query.

    Never includes `groundedness_verdict`, `judge_score`,
    `matched_sentence`, `quarantine_decision`, or the Generated_Answer
    -- none of those five values is a parameter of this function, so
    there is no argument position through which any of them could
    reach the rendered text even by mistake.
    """
    lines: List[str] = [
        "# Hand-checked sample: labelling context",
        "",
        "Read-only reading aid for hand-labelling "
        "`results/hand_checked_sample.csv`. Shows, for each sampled "
        "Claim, the query text and the Retrieved_Context the "
        "Generator_Model and Judge_Model both saw. Does not include the "
        "Judge_Model's verdict, score, matched sentence, quarantine "
        "decision, or the Generated_Answer -- a Hand_Label is assigned "
        "from the Claim and its Retrieved_Context alone, independent of "
        "the Judge_Model's own determination. Fill in `hand_label` in "
        "`results/hand_checked_sample.csv` itself, not here.",
        "",
    ]
    for row in rows:
        lines.append(f"## query_id={row.query_id}, claim_index={row.claim_index}")
        lines.append("")
        lines.append("**Query:**")
        lines.append("")
        lines.append(query_text_by_id[row.query_id])
        lines.append("")
        lines.append("**Claim:**")
        lines.append("")
        lines.append(row.claim_text)
        lines.append("")
        lines.append("**Retrieved context** (rank order, top-k as seen by the generator/judge):")
        lines.append("")
        for rank, document_text in enumerate(retrieved_context_by_query[row.query_id], start=1):
            lines.append(f"{rank}. {document_text}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def write_hand_checked_context(
    rows: List[HandCheckedSampleRow],
    query_text_by_id: Dict[str, str],
    retrieved_context_by_query: Dict[str, List[str]],
    output_path: Path,
) -> None:
    """Renders and writes the labelling aid to `output_path` (default
    `results/hand_checked_sample_context.md`), atomically, via
    `src.report._atomic_write_text`. Always overwrites unconditionally
    -- unlike `results/hand_checked_sample.csv`, this file is never
    hand-edited (it carries no `hand_label` column at all), so there is
    no existing-label state to protect. Raises
    `HandCheckedContextWriteError` on any failure, leaving `output_path`
    either absent or byte-for-byte in its pre-run state.
    """
    try:
        markdown_text = render_hand_checked_context_markdown(
            rows, query_text_by_id, retrieved_context_by_query
        )
    except Exception as exc:
        raise HandCheckedContextWriteError(
            f"failed to build hand-checked sample labelling context for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(
            Path(output_path), markdown_text, failure_context="hand-checked sample labelling context"
        )
    except Exception as exc:
        raise HandCheckedContextWriteError(str(exc)) from exc
