"""Requirement 12 test suite (part 1 of 2): the Claim_Segmenter's
sentence-boundary splitting function.

Imports only `src.claim_segmenter` (`segment_claims`, `Claim`) --
no orchestrating entry point, no retrieval component, no generative or
judging model wrapper, and no data-loading module (Requirement 12.3).
Makes no network call and loads no model or corpus -- every input is a
Python string literal defined in this file (Requirement 12.2).

Together with `tests/test_quarantine_rule.py`, this is the entire
automated-test surface of the groundedness-gate spec (Requirement 12).
"""

from src.claim_segmenter import segment_claims


def test_multi_sentence_answer_yields_one_claim_per_sentence():
    # Req 12.5(a): multi-sentence answer -> one Claim per sentence,
    # claim_index equal to 0-based position.
    answer = "BM25 is a strong baseline. Dense retrieval underperforms here."
    claims = segment_claims(answer)
    assert [c.text for c in claims] == [
        "BM25 is a strong baseline.",
        "Dense retrieval underperforms here.",
    ]
    assert [c.claim_index for c in claims] == list(range(len(claims)))


def test_single_sentence_answer_yields_exactly_one_claim():
    # Req 12.5(b): single-sentence answer -> exactly one Claim.
    answer = "BM25 is a strong baseline on SciFact."
    claims = segment_claims(answer)
    assert len(claims) == 1
    assert claims[0].claim_index == 0
    assert claims[0].text == answer


def test_no_terminal_punctuation_yields_one_claim_of_full_text():
    # Req 12.5(c): no sentence-ending punctuation -> exactly one Claim
    # whose text is the entire Generated_Answer.
    answer = "BM25 is a strong baseline on SciFact"  # no . ! or ?
    claims = segment_claims(answer)
    assert len(claims) == 1
    assert claims[0].claim_index == 0
    assert claims[0].text == answer


def test_claim_indices_are_contiguous_with_no_gaps():
    # Property 1: claim_index values form a contiguous 0..n-1 range,
    # even across three sentences.
    answer = "First claim. Second claim! Third claim?"
    claims = segment_claims(answer)
    assert [c.claim_index for c in claims] == [0, 1, 2]
    assert [c.text for c in claims] == [
        "First claim.",
        "Second claim!",
        "Third claim?",
    ]


def test_empty_string_yields_one_claim_with_empty_text():
    # Req 5.5: whitespace-trimmed empty input is a single Claim at
    # claim_index 0 with empty text, not an error.
    claims = segment_claims("   ")
    assert len(claims) == 1
    assert claims[0].claim_index == 0
    assert claims[0].text == ""
