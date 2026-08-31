"""Requirement 12 test suite (part 2 of 2): the Quarantine_Rule's
threshold-based decision function.

Imports only `src.quarantine_rule` (`decide_quarantine`) and
`src.groundedness_labels` (`Verdict`, for constructing literal
`"SUPPORTED"`/`"NOT_SUPPORTED"` inputs) -- no orchestrating entry
point, no NLI judging model wrapper, and no model- or corpus-loading
module (Requirement 12.3). Makes no network call and loads no model or
corpus (Requirement 12.2).

Together with `tests/test_claim_segmenter.py`, this is the entire
automated-test surface of the groundedness-gate spec (Requirement 12).
"""

from src.groundedness_labels import Verdict
from src.quarantine_rule import decide_quarantine

THRESHOLD = 0.5


def test_supported_score_equal_to_threshold_is_not_quarantined():
    # Req 12.4(a): SUPPORTED at score == threshold -> not quarantined
    # (strict `<` comparison, so equality lands in the "not
    # quarantined" branch).
    verdict: Verdict = "SUPPORTED"
    assert decide_quarantine(verdict, 0.5, THRESHOLD) is False


def test_supported_score_above_threshold_is_not_quarantined():
    # Req 12.4(b): SUPPORTED above threshold -> not quarantined.
    verdict: Verdict = "SUPPORTED"
    assert decide_quarantine(verdict, 0.9, THRESHOLD) is False


def test_supported_score_below_threshold_is_quarantined():
    # Req 12.4(c): SUPPORTED below threshold -> quarantined.
    verdict: Verdict = "SUPPORTED"
    assert decide_quarantine(verdict, 0.1, THRESHOLD) is True


def test_not_supported_is_quarantined_regardless_of_score():
    # Req 12.4(d): NOT_SUPPORTED at two distinct scores, one above and
    # one below threshold, both quarantined.
    verdict: Verdict = "NOT_SUPPORTED"
    above = decide_quarantine(verdict, 0.9, THRESHOLD)
    below = decide_quarantine(verdict, 0.1, THRESHOLD)
    assert above is True
    assert below is True


def test_decision_is_deterministic_for_repeated_calls():
    # Req 7.4: same (verdict, score, threshold) tuple always produces
    # the same Quarantine_Decision.
    first = decide_quarantine("SUPPORTED", 0.42, THRESHOLD)
    second = decide_quarantine("SUPPORTED", 0.42, THRESHOLD)
    assert first == second
