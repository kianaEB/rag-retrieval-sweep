"""Test suite for `src/token_length_analysis.py` (repo-writeup spec,
Requirement 11).

Covers `compute_exceedance_stats` (Property 1: an empty list, an
all-under-threshold list, an all-over-threshold list, a mixed list with
an independently hand-computed expected fraction, and the
`256`-vs-`257` boundary pair), order independence (Property 2: the same
mixed list shuffled), and `load_tokenizer_offline` raising
`TokenizerLoadError` against an empty cache directory with no network
call attempted (Property 4).

This module imports only `src.token_length_analysis` -- which
transitively imports `src.retrievers.dense_retriever` for
`format_document_text`, the direct, intended consequence of Task 1's
extract-and-import refactor, not a sign this test reaches into
unrelated retriever internals (`format_document_text` itself is not
re-tested here; see design.md's Property 3 note). It does not import
the orchestration entry point module or the significance analyzer
module. It makes no network call anywhere, including in the "missing
cache" test:
`local_files_only=True` guarantees a local error rather than a request
when the cache directory is empty, so asserting `TokenizerLoadError`
there is safe under "no network access inside tests." It does not
load the real cached model or tokenize the real ~5k-document corpus --
that real-corpus run is a manual step (Task 6), not an automated test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.errors import TokenizerLoadError
from src.token_length_analysis import TokenLengthStats, compute_exceedance_stats, load_tokenizer_offline

# A real model name is required for load_tokenizer_offline's "missing
# cache" test (Property 4) -- local_files_only=True still needs a real
# repo id to look up locally and fail to find, rather than short-
# circuiting on a malformed id before ever touching the cache lookup
# path. This is the same model configs/sweep.yaml declares.
_REAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# compute_exceedance_stats -- Property 1 (boundary cases)
# ---------------------------------------------------------------------------


def test_compute_exceedance_stats_empty_list():
    stats = compute_exceedance_stats([], max_sequence_length=256)
    assert stats == TokenLengthStats(
        num_documents_total=0, num_documents_exceeding=0, fraction_exceeding=0.0
    )


def test_compute_exceedance_stats_all_under_threshold():
    token_counts = [10, 50, 100, 255]
    stats = compute_exceedance_stats(token_counts, max_sequence_length=256)
    assert stats == TokenLengthStats(
        num_documents_total=4, num_documents_exceeding=0, fraction_exceeding=0.0
    )


def test_compute_exceedance_stats_all_over_threshold():
    token_counts = [257, 300, 500, 1000]
    stats = compute_exceedance_stats(token_counts, max_sequence_length=256)
    assert stats == TokenLengthStats(
        num_documents_total=4, num_documents_exceeding=4, fraction_exceeding=1.0
    )


def test_compute_exceedance_stats_mixed_list_hand_computed_fraction():
    # 7 documents; independently hand-counted: 100, 200, 256 do not
    # exceed; 257, 300, 400, 1000 do -> 4 of 7 exceed -> 4/7.
    token_counts = [100, 257, 200, 300, 256, 400, 1000]
    stats = compute_exceedance_stats(token_counts, max_sequence_length=256)
    assert stats.num_documents_total == 7
    assert stats.num_documents_exceeding == 4
    assert stats.fraction_exceeding == pytest.approx(4 / 7, abs=1e-9)


def test_compute_exceedance_stats_boundary_256_vs_257():
    # Requirement 11.2: "strictly greater than" -- exactly 256 is not
    # exceeding; 257 is.
    stats = compute_exceedance_stats([256, 257], max_sequence_length=256)
    assert stats.num_documents_exceeding == 1
    assert stats.num_documents_total == 2
    assert stats.fraction_exceeding == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_exceedance_stats -- Property 2 (order independence)
# ---------------------------------------------------------------------------


def test_compute_exceedance_stats_order_independent():
    token_counts = [100, 257, 200, 300, 256, 400, 1000]
    shuffled = [1000, 256, 300, 100, 257, 400, 200]
    original_stats = compute_exceedance_stats(token_counts, max_sequence_length=256)
    shuffled_stats = compute_exceedance_stats(shuffled, max_sequence_length=256)
    assert shuffled_stats.num_documents_exceeding == original_stats.num_documents_exceeding
    assert shuffled_stats.num_documents_total == original_stats.num_documents_total
    assert shuffled_stats.fraction_exceeding == pytest.approx(
        original_stats.fraction_exceeding, abs=1e-9
    )


# ---------------------------------------------------------------------------
# load_tokenizer_offline -- Property 4 (no network call, ever)
# ---------------------------------------------------------------------------


def test_load_tokenizer_offline_raises_on_empty_cache_no_network_call():
    # local_files_only=True (plus the HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
    # environment variables load_tokenizer_offline sets) makes
    # transformers raise a local error immediately on a cache miss,
    # rather than falling back to a network request -- safe to run
    # under "no network access inside tests" for exactly this reason.
    with tempfile.TemporaryDirectory() as empty_cache_dir:
        with pytest.raises(TokenizerLoadError):
            load_tokenizer_offline(_REAL_MODEL_NAME, Path(empty_cache_dir))
