"""Claim_Segmenter: splits a Generated_Answer into Claims at sentence
boundaries (Requirement 5).

Pure function of a `str` -- no model load, no file I/O, no network
call. Standard library only (`re`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Any occurrence of . ! or ? immediately followed by one or more
# whitespace characters, or by the end of the text (Requirement 5.1).
# The trailing punctuation is captured so it stays attached to the
# preceding segment (Requirement 5.2's "including that segment's
# terminating sentence-boundary punctuation character when present").
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s+|$)")


@dataclass(frozen=True)
class Claim:
    """One sentence-boundary-delimited segment of a Generated_Answer
    (Requirement 5.2). `claim_index` is 0-based position within its
    answer's ordered list of Claims; `text` has leading/trailing
    whitespace removed and never carries a claim_index gap, since
    empty segments are dropped before indices are assigned."""

    claim_index: int
    text: str


def segment_claims(generated_answer: str) -> List[Claim]:
    """Splits `generated_answer` into an ordered list of `Claim`s at
    sentence boundaries (Requirement 5.1).

    Splitting is a crude heuristic, not a solved natural-language-
    processing problem (Requirement 5.4): `_SENTENCE_BOUNDARY` matches
    any of `.`/`!`/`?` followed by whitespace or end-of-string, so an
    abbreviation, a decimal number, or a quotation mark placed after
    the terminator can all mis-split a single intended sentence into
    more than one Claim, or fail to split where a human reader would.
    A mis-split sentence is a source of measurement error in what
    counts as one Claim, not a correctness bug in this function's
    contract -- the contract is exactly the boundary rule in
    Requirement 5.1, not "linguistically correct sentence
    segmentation."

    Segments are trimmed of leading/trailing whitespace; a segment
    that is empty after trimming is dropped and never receives a
    `claim_index` (Requirement 5.2) -- so `claim_index` values are
    always a contiguous 0..n-1 range with no gaps. If, after trimming
    the whole `generated_answer`, no sentence boundary is found
    (including when the trimmed text is the empty string), the entire
    trimmed text becomes a single Claim at `claim_index` 0, rather
    than raising (Requirement 5.5).
    """
    trimmed = generated_answer.strip()
    if not _SENTENCE_BOUNDARY.search(trimmed):
        return [Claim(claim_index=0, text=trimmed)]

    # re.split on a pattern with no capture group would drop the
    # matched boundary text itself, so segments are instead sliced out
    # by scanning match end-positions directly with finditer -- this
    # keeps each segment's terminating punctuation attached, per
    # Requirement 5.2.
    claims: List[Claim] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(trimmed):
        segment = trimmed[start:match.end()].strip()
        if segment:
            claims.append(Claim(claim_index=len(claims), text=segment))
        start = match.end()
    tail = trimmed[start:].strip()
    if tail:
        claims.append(Claim(claim_index=len(claims), text=tail))
    return claims
