"""Quarantine_Rule: maps (Groundedness_Verdict, score, threshold) to a
Quarantine_Decision (Requirement 7).

A pure, deterministic function of exactly three inputs (Requirement
7.4): no corpus, no model, no file I/O, no other parameter. Standard
library only.
"""

from __future__ import annotations

from src.groundedness_labels import Verdict


def decide_quarantine(verdict: Verdict, score: float, threshold: float) -> bool:
    """Maps (Groundedness_Verdict, score, threshold) to a
    Quarantine_Decision (Requirement 7).

    Three-branch decision table, exhaustive over the two possible
    values of `verdict`:

    - `verdict == "NOT_SUPPORTED"` -> True, regardless of `score`
      (Requirement 7.1).
    - `verdict == "SUPPORTED"` and `score < threshold` -> True
      (Requirement 7.2).
    - `verdict == "SUPPORTED"` and `score >= threshold` -> False
      (Requirement 7.3).

    The `score < threshold` boundary is strict: a score numerically
    equal to `threshold` falls into the third branch (`quarantine ==
    False`), never the second -- this is the exact tie-break
    Requirement 12.4's test case (a) exercises. Same
    (verdict, score, threshold) tuple always returns the same result;
    no randomness, no hidden state.
    """
    if verdict == "NOT_SUPPORTED":
        return True
    return score < threshold
