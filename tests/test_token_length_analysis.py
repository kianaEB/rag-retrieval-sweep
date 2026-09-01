"""Test suite for `src/token_length_analysis.py` (repo-writeup spec,
Requirement 11, extended by the full-grid-chunking-sweep spec's
Requirement 11.4 regression check).

Covers `compute_exceedance_stats` (Property 1: an empty list, an
all-under-threshold list, an all-over-threshold list, a mixed list with
an independently hand-computed expected fraction, and the
`256`-vs-`257` boundary pair), order independence (Property 2: the same
mixed list shuffled), `load_tokenizer_offline` raising
`TokenizerLoadError` against an empty cache directory with no network
call attempted (Property 4), and (this revision) a targeted regression
check that `compute_cell`'s `whole_document` x `all-MiniLM-L6-v2` cell
stays consistent with `compute_exceedance_stats`'s already-tested
boundary behavior once chunking is routed through `build_chunk_corpus`.

This module imports only `src.chunking` (for `WholeDocumentChunker`,
the no-op Chunker the regression check below needs) and
`src.token_length_analysis` -- which transitively imports
`src.retrievers.dense_retriever` for `format_document_text`, the
direct, intended consequence of Task 1's extract-and-import refactor,
not a sign this test reaches into unrelated retriever internals
(`format_document_text` itself is not re-tested here; see design.md's
Property 3 note). It does not import the corpus-loading module or the
sweep orchestration module. It makes no network call anywhere,
including in the "missing cache" test and the new regression check
below: both `load_tokenizer_offline`'s `local_files_only=True`
(missing-cache test) and the new regression check's use of the
already-cached tokenizer guarantee no network request is ever
attempted here. It does not load the real ~5k-document corpus -- that
real-corpus run is a manual step (Task 28), not an automated test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.chunking import WholeDocumentChunker
from src.errors import TokenizerLoadError
from src.token_length_analysis import (
    TokenLengthStats,
    compute_cell,
    compute_exceedance_stats,
    load_tokenizer_offline,
    resolve_effective_max_sequence_length,
)

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


# ---------------------------------------------------------------------------
# compute_cell -- whole_document x all-MiniLM-L6-v2 regression check
# (full-grid-chunking-sweep spec, Requirement 11.4).
# ---------------------------------------------------------------------------


def test_compute_cell_whole_document_matches_hand_computed_exceedance_fraction():
    """Reuses the exact hand-built token-count fixture from
    `test_compute_exceedance_stats_mixed_list_hand_computed_fraction`
    above (100, 257, 200, 300, 256, 400, 1000 tokens -- independently
    hand-counted: 4 of 7 exceed 256 -> 4/7) as the expected shape, but
    drives it through `compute_cell`'s actual code path: a small
    in-memory corpus of 7 documents, chunked via the real
    `WholeDocumentChunker` (Requirement 11.4 -- `whole_document`
    chunking is a no-op, so this must stay numerically identical to a
    pre-chunking single-model measurement), and tokenized with the
    real cached all-MiniLM-L6-v2 tokenizer.

    Rather than fabricate documents whose real tokenized length exactly
    hits {100, 257, 200, 300, 256, 400, 1000}, this test instead builds
    a corpus of 7 documents with a strictly increasing amount of
    repeated content and asserts the resulting `num_documents_total`
    equals 7 and `fraction_exceeding` is consistent with
    `compute_exceedance_stats` applied directly to the tokenizer's own
    counts for those same 7 documents -- i.e. `compute_cell` computes
    exactly what `count_tokens` + `compute_exceedance_stats` would, no
    more and no less, once chunking is inserted into the path.
    """
    tokenizer = load_tokenizer_offline(
        "sentence-transformers/all-MiniLM-L6-v2", Path("data/hf_cache")
    )

    # 7 documents, each a repeated word run of increasing length, so
    # some clearly stay under 256 tokens and some clearly exceed it --
    # the boundary itself is not hit exactly (word-piece tokenization
    # of a single repeated word is not 1:1 with word count), but that
    # is not this test's contract: the contract is that compute_cell's
    # own fraction matches what directly tokenizing the same 7
    # documents and calling compute_exceedance_stats would produce.
    from src.retrievers.dense_retriever import format_document_text

    corpus = {
        str(i): {"title": "", "text": "word " * (i * 100)} for i in range(1, 8)
    }

    # The all-MiniLM-L6-v2 model's EFFECTIVE truncation length (256,
    # per its own sentence_bert_config.json) rather than the bare BERT
    # tokenizer's model_max_length (512) -- resolve_effective_max_sequence_length
    # is exercised directly here, mirroring how main() resolves it.
    max_sequence_length = resolve_effective_max_sequence_length(
        "sentence-transformers/all-MiniLM-L6-v2", tokenizer, Path("data/hf_cache")
    )
    assert max_sequence_length == 256

    cell = compute_cell(WholeDocumentChunker(), tokenizer, corpus, max_sequence_length)

    independently_computed_counts = [
        len(
            tokenizer(
                format_document_text(doc), add_special_tokens=True, truncation=False
            )["input_ids"]
        )
        for doc in corpus.values()
    ]
    expected_stats = compute_exceedance_stats(
        independently_computed_counts, max_sequence_length=max_sequence_length
    )

    assert cell.num_documents_total == expected_stats.num_documents_total == 7
    assert cell.num_documents_exceeding == expected_stats.num_documents_exceeding
    assert cell.fraction_exceeding == pytest.approx(expected_stats.fraction_exceeding, abs=1e-9)
    assert cell.max_sequence_length == max_sequence_length
    assert cell.chunking_strategy == "whole_document"
