"""Loader for `docs/claim_assertion_classification.csv` -- the
committed, per-claim declarative-assertion classification used only
for the Agreement_Rate partition analysis in SPEC.md's "Groundedness
gate" section.

This replaces an earlier mechanical heuristic (`src/claim_classifier.py`,
now removed) that matched a fixed marker word list (is/are/was/has/...)
rather than actually detecting a finite main verb -- it caught copular
sentences ("X is a Y") but missed ordinary present-tense assertions
whose only verb is a bare third-person-singular main verb ("Sildenafil
*improves* ...", "Aspirin *inhibits* ...", "CD11b+ monocytes *abrogate*
..."), misclassifying seven of the thirty Claims as non-assertions.

The rule applied to build the committed file is: **a Claim is a
declarative assertion if its text contains a finite main verb, making
it a declarative assertion rather than a title, noun phrase, or
fragment.** Unlike `hand_label` (a human judgment call about support
that cannot be re-derived from the text alone), this is a grammatical
fact about the committed `claim_text` -- anyone can audit any row of
`docs/claim_assertion_classification.csv` against
`results/groundedness.csv`'s `claim_text` column without needing to
re-run anything or consult a human reviewer. It was applied by
inspecting each of the 30 committed Claim texts individually (not
mechanically re-derived from a marker list), after the hand labels
already existed.

This module never influences a Groundedness_Verdict, `judge_score`, or
Quarantine_Decision -- it is read only when building
`results/hand_checked_joined.csv`'s `is_declarative_assertion` column,
purely to explain where the Judge_Model and the human reviewer
disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas

from src.errors import ClaimClassificationError

ClaimId = Tuple[str, int]

DEFAULT_CLASSIFICATION_PATH = Path("docs/claim_assertion_classification.csv")

_REQUIRED_COLUMNS = ("query_id", "claim_index", "claim_text", "is_declarative_assertion")


def load_claim_assertion_classification(
    path: Path = DEFAULT_CLASSIFICATION_PATH,
) -> Dict[ClaimId, bool]:
    """Reads `path` (default `docs/claim_assertion_classification.csv`)
    into `{(query_id, claim_index): is_declarative_assertion}`.

    Raises `ClaimClassificationError` if `path` is absent, cannot be
    parsed as a CSV, lacks any of `_REQUIRED_COLUMNS`, or contains an
    `is_declarative_assertion` cell that is not literally `"True"` or
    `"False"`.
    """
    path = Path(path)
    if not path.is_file():
        raise ClaimClassificationError(
            f"claim assertion classification file not found: {path}"
        )
    try:
        frame = pandas.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ClaimClassificationError(
            f"failed to parse claim assertion classification file {path}: {exc}"
        ) from exc

    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ClaimClassificationError(
            f"claim assertion classification file {path} is missing required "
            f"column(s): {missing_columns}"
        )

    classification: Dict[ClaimId, bool] = {}
    for _, record in frame.iterrows():
        raw_value = record["is_declarative_assertion"]
        if raw_value not in ("True", "False"):
            raise ClaimClassificationError(
                f"claim assertion classification file {path}: "
                f"is_declarative_assertion value {raw_value!r} is not "
                f"'True' or 'False' for query_id={record['query_id']!r}, "
                f"claim_index={record['claim_index']!r}"
            )
        claim_id: ClaimId = (str(record["query_id"]), int(record["claim_index"]))
        classification[claim_id] = raw_value == "True"
    return classification


def lookup_classification(
    classification: Dict[ClaimId, bool], claim_id: ClaimId, path: Path
) -> bool:
    """Returns `classification[claim_id]`, raising
    `ClaimClassificationError` naming `claim_id` and `path` if absent --
    a Claim in the Hand_Checked_Sample with no row in the committed
    classification file is a hard failure, not a silent default.
    """
    if claim_id not in classification:
        raise ClaimClassificationError(
            f"no declarative-assertion classification for "
            f"query_id={claim_id[0]!r}, claim_index={claim_id[1]!r} in {path}"
        )
    return classification[claim_id]
