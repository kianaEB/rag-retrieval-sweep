"""Hand_Checked_Sample selection, export, import, agreement, and the
joined-artifact writer (Requirement 10).

Selection is computed from Claim identity alone (`query_id`,
`claim_index`), a canonical sort order, and the
`Hand_Checked_Sample_Seed` -- never from any `Groundedness_Verdict`,
`judge_score`, or `quarantine_decision` (Requirement 10.2). This is
enforced structurally by the function's argument list, not by
convention: `select_hand_checked_sample` below takes only `claim_ids`
(a list of `(query_id, claim_index)` tuples) and `seed`, so there is
no parameter through which a verdict, score, or quarantine decision
could reach the selection even by mistake.

The export file and the re-import file are the *same path*
(`results/hand_checked_sample.csv`), edited in place by a human between
runs (design decision noted in the requirements' Requirement 10
Criterion 7: "SHALL NOT overwrite that file ... leave the existing
Hand_Label values unmodified" only makes sense against one artifact
that is exported once and then re-read on every subsequent run).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas

from src.claim_assertion_classification import (
    DEFAULT_CLASSIFICATION_PATH,
    load_claim_assertion_classification,
    lookup_classification,
)
from src.errors import HandCheckedJoinedWriteError, HandCheckedSampleWriteError
from src.report import _atomic_write_text

ClaimId = Tuple[str, int]  # (query_id, claim_index)


def select_hand_checked_sample(
    claim_ids: List[ClaimId], sample_size: int, seed: int
) -> List[ClaimId]:
    """Draws `min(sample_size, len(claim_ids))` claim IDs uniformly at
    random, without replacement, from `claim_ids` sorted into
    canonical order (by query_id, then claim_index) before sampling,
    seeded with `seed` (Requirement 10.1, 10.3).

    Takes only `claim_ids` and `seed` as input -- no verdict, score, or
    quarantine_decision parameter exists on this function's signature,
    so Requirement 10.2's independence is a structural property of the
    call site, not a convention a future edit could quietly violate.
    """
    canonical_order = sorted(claim_ids)
    rng = random.Random(seed)
    k = min(sample_size, len(canonical_order))
    return rng.sample(canonical_order, k)


@dataclass(frozen=True)
class HandCheckedSampleRow:
    query_id: str
    claim_index: int
    claim_text: str
    hand_label: str  # blank ("") until a human fills it in


def read_hand_checked_sample_rows(path: Path) -> List[HandCheckedSampleRow]:
    """Reads `path` (default results/hand_checked_sample.csv) back into
    `HandCheckedSampleRow`s, in the exact row order the file is stored
    in -- never re-sorted or re-derived from a fresh
    `select_hand_checked_sample` draw. This matters specifically
    because `export_hand_checked_sample` is a no-op once the file
    already carries a non-blank `hand_label` (Requirement 10.7): the
    file on disk can differ from whatever this run's own in-memory
    selection computed, so a caller that needs "the rows of
    `results/hand_checked_sample.csv`, in that same order" (e.g. the
    labelling-context aid in `src.hand_checked_context`) must read them
    back from the file itself, not from the in-memory list.
    """
    frame = pandas.read_csv(Path(path), dtype=str, keep_default_na=False)
    return [
        HandCheckedSampleRow(
            query_id=str(record["query_id"]),
            claim_index=int(record["claim_index"]),
            claim_text=str(record["claim_text"]),
            hand_label=str(record["hand_label"]),
        )
        for _, record in frame.iterrows()
    ]


def export_hand_checked_sample(
    rows: List[HandCheckedSampleRow], output_path: Path
) -> None:
    """Writes `rows` to `output_path` (default
    results/hand_checked_sample.csv), atomically, with a blank
    `hand_label` field (Requirement 10.4). This export intentionally
    excludes the Claim's Groundedness_Verdict, Judge_Model score, and
    Quarantine_Decision (Requirement 10.8) -- a human labelling from
    this file must not see, or be anchored by, the judge's own
    determination for that Claim. This is also why Agreement_Rate
    cannot be resolved from this file alone; see
    `join_hand_labels_with_verdicts` / `write_hand_checked_joined`
    below for the derived artifact that carries both columns.

    If `output_path` already exists AND contains a non-blank
    `hand_label` for one or more of its rows, this function does
    NOT overwrite it -- it returns without writing, leaving the
    existing file and its Hand_Label values unmodified (Requirement
    10.7). "Non-blank" means neither an empty string nor a string
    containing only whitespace (matching Requirement 10.5/10.6's
    definition). Raises HandCheckedSampleWriteError if the write
    itself fails.
    """
    output_path = Path(output_path)
    if output_path.is_file():
        existing = pandas.read_csv(output_path, dtype=str, keep_default_na=False)
        if "hand_label" in existing.columns and existing["hand_label"].str.strip().ne("").any():
            return
    try:
        frame = pandas.DataFrame([dataclasses.asdict(r) for r in rows])
        _atomic_write_text(
            output_path, frame.to_csv(index=False), failure_context="hand-checked sample export"
        )
    except Exception as exc:
        raise HandCheckedSampleWriteError(str(exc)) from exc


def read_hand_label_import(
    path: Path, expected_claim_ids: List[ClaimId]
) -> Optional[Dict[ClaimId, str]]:
    """Reads `path` (the same results/hand_checked_sample.csv path) and
    returns `{(query_id, claim_index): hand_label}` ONLY if `path`
    exists, contains a row for every one of `expected_claim_ids`, and
    every one of those rows carries a non-blank `hand_label`
    (Requirement 10.5). Otherwise returns `None` (Requirement 10.6) --
    the caller leaves the file available for manual labelling and does
    not compute Agreement_Rate.
    """
    path = Path(path)
    if not path.is_file():
        return None

    try:
        frame = pandas.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None

    required_columns = {"query_id", "claim_index", "hand_label"}
    if not required_columns.issubset(frame.columns):
        return None

    labels: Dict[ClaimId, str] = {}
    for _, record in frame.iterrows():
        try:
            claim_id: ClaimId = (str(record["query_id"]), int(record["claim_index"]))
        except (TypeError, ValueError):
            continue
        labels[claim_id] = record["hand_label"]

    result: Dict[ClaimId, str] = {}
    for claim_id in expected_claim_ids:
        if claim_id not in labels:
            return None
        hand_label = labels[claim_id]
        if hand_label.strip() == "":
            return None
        result[claim_id] = hand_label

    return result


def compute_agreement_rate(
    judge_verdicts: Dict[ClaimId, str], hand_labels: Dict[ClaimId, str]
) -> float:
    """Fraction of `hand_labels`' keys whose judge_verdicts[key] ==
    hand_labels[key] (Requirement 10.5's Agreement_Rate definition).
    Pure function of the two dicts; assumes read_hand_label_import
    already verified full coverage and non-blank labels.
    """
    matches = sum(1 for cid, label in hand_labels.items() if judge_verdicts.get(cid) == label)
    return matches / len(hand_labels) if hand_labels else 0.0


@dataclass(frozen=True)
class HandCheckedJoinedRow:
    """One row of results/hand_checked_joined.csv -- written ONLY
    after read_hand_label_import returns non-None (every Hand_Label
    present and non-blank), joining that Claim's already-computed
    Groundedness_Verdict (from the same run's judging step, never
    re-derived) to its Hand_Label. This is the ONE artifact that
    carries both columns side by side, and it exists specifically so
    Agreement_Rate can be resolved as a single-artifact aggregate at
    verification time without ever exposing the judge's verdict in the
    file a human actually labels from (results/hand_checked_sample.csv,
    Requirement 10.8).

    `claim_text` and `is_declarative_assertion` are a later addition,
    beyond the original Requirement 10 schema: SPEC.md's Agreement_Rate
    discussion partitions the Hand_Checked_Sample into declarative
    assertions and non-assertions because a single pooled 0.2667
    conflates two different disagreement mechanisms -- the Judge_Model
    finding textual entailment on a non-assertion Claim (a copied
    title, a bare noun phrase) the human reviewer additionally declined
    to call SUPPORTED because it isn't a declarative assertion at all,
    versus a genuine miss on a real assertion.

    `is_declarative_assertion` is read from the committed, hand-audited
    `docs/claim_assertion_classification.csv` (see
    `src.claim_assertion_classification`), not computed by any
    mechanical heuristic -- an earlier marker-word heuristic
    (`src/claim_classifier.py`, since removed) matched a fixed word
    list (is/are/was/has/...) rather than actually detecting a finite
    main verb, and misclassified 7 of the 30 Claims as non-assertions
    (e.g. "Sildenafil improves ...", "CD11b+ monocytes abrogate ...").
    The committed file is a grammatical fact about each Claim's
    committed `claim_text` -- anyone can audit any row against
    `results/groundedness.csv` without a human reviewer or a re-run,
    which is what distinguishes it from `hand_label` even though
    neither one is mechanically re-derived.

    Recording `is_declarative_assertion` per row here, in the one
    artifact that already carries both `judge_verdict` and
    `hand_label`, is what lets the partitioned Agreement_Rate numbers in
    SPEC.md be checked against a committed artifact via
    `src.verify_writeup_numbers`'s existing `row_selector.field`
    mechanism, rather than only being reproducible by re-running a
    separate analysis script.

    `agrees` (`judge_verdict == hand_label`, computed once here) exists
    so a per-partition Agreement_Rate can be resolved as a single
    `ratio` computation over two `__count__` references within one
    ledger row (`agrees=True.__count__` over
    `is_declarative_assertion=<True|False>` filtered rows, divided by
    that partition's `__count__` total) -- `src.verify_writeup_numbers`
    already supports arbitrary `key=value` filters and the `__count__`
    field sentinel, but its `ratio` computation takes exactly the two
    values a `source_fields` list resolves, not a sum of sub-counts
    computed inside the same row. Pre-computing `agrees` here keeps
    that resolution a single `row_selector.field` lookup instead of a
    multi-step derivation the traceability mechanism cannot express in
    one row."""

    query_id: str
    claim_index: int
    judge_verdict: str
    hand_label: str
    claim_text: str
    is_declarative_assertion: bool
    agrees: bool


def join_hand_labels_with_verdicts(
    judge_verdicts: Dict[ClaimId, str],
    hand_labels: Dict[ClaimId, str],
    claim_text_by_id: Dict[ClaimId, str],
    classification_path: Path = DEFAULT_CLASSIFICATION_PATH,
) -> List[HandCheckedJoinedRow]:
    """Builds one HandCheckedJoinedRow per key of `hand_labels` (every
    Hand_Checked_Sample Claim with a non-blank Hand_Label), pairing it
    with that Claim's judge_verdicts[key] and claim_text_by_id[key] --
    assumes read_hand_label_import already verified full coverage, so
    every key of hand_labels is guaranteed present in both
    judge_verdicts and claim_text_by_id.

    `is_declarative_assertion` is read from the committed
    `docs/claim_assertion_classification.csv` (default
    `classification_path`) via
    `src.claim_assertion_classification.load_claim_assertion_classification`,
    never computed by a mechanical heuristic -- see
    `HandCheckedJoinedRow`'s docstring for why. Raises
    `ClaimClassificationError` (propagated from
    `load_claim_assertion_classification`/`lookup_classification`) if
    the file is missing, malformed, or has no row for one of
    `hand_labels`' keys.
    """
    classification = load_claim_assertion_classification(classification_path)
    return [
        HandCheckedJoinedRow(
            query_id=qid,
            claim_index=idx,
            judge_verdict=judge_verdicts[(qid, idx)],
            hand_label=label,
            claim_text=claim_text_by_id[(qid, idx)],
            is_declarative_assertion=lookup_classification(
                classification, (qid, idx), classification_path
            ),
            agrees=(judge_verdicts[(qid, idx)] == label),
        )
        for (qid, idx), label in sorted(hand_labels.items())
    ]


def write_hand_checked_joined(rows: List[HandCheckedJoinedRow], output_path: Path) -> None:
    """Writes `rows` to `output_path` (default
    results/hand_checked_joined.csv), atomically, via
    src.report._atomic_write_text. Unlike
    export_hand_checked_sample's `results/hand_checked_sample.csv`
    (which a human hand-edits and must never be overwritten once
    labelled -- Requirement 10.7), this file is fully derived and never
    hand-edited, so it is safe to overwrite unconditionally on every
    run that successfully reads back a complete Hand_Label_Import.
    Raises HandCheckedJoinedWriteError on any failure, leaving
    output_path either absent or byte-for-byte in its pre-run state.
    """
    try:
        frame = pandas.DataFrame([dataclasses.asdict(r) for r in rows])
        _atomic_write_text(
            output_path, frame.to_csv(index=False), failure_context="hand-checked joined export"
        )
    except Exception as exc:
        raise HandCheckedJoinedWriteError(str(exc)) from exc
