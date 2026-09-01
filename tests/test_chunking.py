"""full-grid-chunking-sweep spec: Properties 1-3 (network-free) and
Properties 4-5 (tokenizer-gated, this revision).

Covers `src.chunking`'s network-free surface: `WholeDocumentChunker`
(Property 1), `aggregate_to_document_ranked_list`'s Max_Aggregation
rule (Property 2), and `build_chunk_corpus`'s well-formedness /
coverage / non-duplication guarantees over `WholeDocumentChunker`
(Property 3). No test covering Properties 1-3 loads a tokenizer or
makes a network call -- `WholeDocumentChunker` needs none, and
`aggregate_to_document_ranked_list` operates on a hand-built
`ChunkScores` alone.

Properties 4 and 5 (this revision) exercise `FixedWindowChunker` and
`SentenceWindowChunker` against the real, already-cached
all-MiniLM-L6-v2 tokenizer, via the module-scoped `windowing_tokenizer`
fixture below. That fixture -- and every test depending on it -- is
gated by a `Local_Cache_Availability` check
(`data/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2`) via
`pytest.mark.skipif`, so on a clean checkout with no cached tokenizer,
Properties 4/5 are skipped, not failed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from src.chunking import (
    Chunk,
    FixedWindowChunker,
    SentenceWindowChunker,
    WholeDocumentChunker,
    _split_into_sentences,
    aggregate_to_document_ranked_list,
    build_chunk_corpus,
    make_chunk_id,
    parse_chunk_id,
)
from src.config import BM25RetrieverConfig
from src.errors import ChunkingError
from src.retrievers.base import ChunkScores, doc_id_sort_key
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import format_document_text

# Local_Cache_Availability (Requirement 9): whether the all-MiniLM-L6-v2
# tokenizer snapshot is already present under data/hf_cache. Checked
# once at module import time, so the skip-gate below is evaluated
# without needing to actually load the tokenizer.
_TOKENIZER_CACHE_DIR = Path("data/hf_cache")
_TOKENIZER_SNAPSHOT_MARKER = _TOKENIZER_CACHE_DIR / "models--sentence-transformers--all-MiniLM-L6-v2"


def _local_tokenizer_cache_available() -> bool:
    return _TOKENIZER_SNAPSHOT_MARKER.exists()

# ---------------------------------------------------------------------------
# Property 1: Whole-document chunking preserves content exactly.
# Validates: Requirement 2.2.
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(title=st.text(), text=st.text())
def test_property_1_whole_document_chunking_preserves_content_exactly(title, text):
    document = {"title": title, "text": text}
    chunks = WholeDocumentChunker().chunk_document("doc-1", document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.position == 0
    assert chunk.doc_id == "doc-1"
    assert chunk.text == format_document_text(document)


# ---------------------------------------------------------------------------
# Property 2: Max_Aggregation is the true maximum, tie-broken by
# ascending document ID, and reduces to the identity for single-chunk
# documents.
# Validates: Requirements 2.5, 5.1, 5.2, 5.5, 10.2.
# ---------------------------------------------------------------------------

# A small, fixed alphabet mixing numeric-looking IDs ("1", "2", "10")
# and non-numeric IDs ("A", "B", "C") so doc_id_sort_key's two branches
# (numeric vs. string comparison) are both exercised, and so that a
# purely lexicographic tie-break ("10" < "2") would visibly diverge
# from the correct numeric one (2 < 10).
_DOC_ID_ALPHABET = ["1", "2", "10", "A", "B", "C"]


@st.composite
def _chunk_scores_with_varying_chunk_counts(draw):
    """Generates a small set of documents, each assigned 1-5 chunks
    with independently drawn float scores, assembled into a
    `ChunkScores` via `make_chunk_id`, with `chunk_ids`/`scores` order
    shuffled (since `ChunkScores` carries no meaningful order)."""
    doc_ids = draw(
        st.lists(st.sampled_from(_DOC_ID_ALPHABET), min_size=1, max_size=6, unique=True)
    )
    per_doc_scores: Dict[str, List[float]] = {}
    chunk_ids: List[str] = []
    scores: List[float] = []
    for doc_id in doc_ids:
        n_chunks = draw(st.integers(min_value=1, max_value=5))
        doc_scores = draw(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False),
                min_size=n_chunks,
                max_size=n_chunks,
            )
        )
        per_doc_scores[doc_id] = doc_scores
        for position, score in enumerate(doc_scores):
            chunk_ids.append(make_chunk_id(doc_id, position))
            scores.append(score)

    order = draw(st.permutations(range(len(chunk_ids))))
    chunk_ids = [chunk_ids[i] for i in order]
    scores = [scores[i] for i in order]
    return per_doc_scores, chunk_ids, scores


def _order_by(scores_by_doc: Dict[str, float]) -> List[str]:
    return sorted(scores_by_doc.keys(), key=lambda d: (-scores_by_doc[d], doc_id_sort_key(d)))


@settings(max_examples=100)
# A hand-specified example deliberately tying two documents at the
# same maximum score, guaranteeing the ascending-doc_id_sort_key
# tie-break path is exercised on every run, not left to chance.
@example(
    (
        {"10": [0.9], "2": [0.9], "A": [0.1, 0.9, 0.2]},
        [make_chunk_id("10", 0), make_chunk_id("2", 0), make_chunk_id("A", 0), make_chunk_id("A", 1), make_chunk_id("A", 2)],
        [0.9, 0.9, 0.1, 0.9, 0.2],
    )
)
@given(_chunk_scores_with_varying_chunk_counts())
def test_property_2_max_aggregation_true_maximum_tie_broken_ascending(data):
    per_doc_scores, chunk_ids, scores = data
    chunk_scores = ChunkScores(chunk_ids=tuple(chunk_ids), scores=np.array(scores))
    result = aggregate_to_document_ranked_list(chunk_scores)

    true_max = {doc_id: max(s) for doc_id, s in per_doc_scores.items()}
    expected_order = _order_by(true_max)
    assert result == expected_order

    # Diverges from mean/sum whenever they would produce a different
    # order than the true maximum -- max is never silently equivalent
    # to mean or sum here.
    mean_scores = {doc_id: sum(s) / len(s) for doc_id, s in per_doc_scores.items()}
    sum_scores = {doc_id: sum(s) for doc_id, s in per_doc_scores.items()}
    mean_order = _order_by(mean_scores)
    sum_order = _order_by(sum_scores)
    if mean_order != expected_order:
        assert result != mean_order
    if sum_order != expected_order:
        assert result != sum_order


@st.composite
def _chunk_scores_single_chunk_per_doc(draw):
    """The n=1 corollary's dedicated strategy: every document is
    assigned exactly one chunk, so Max_Aggregation must reduce to
    ranking documents directly by their own single score."""
    doc_ids = draw(
        st.lists(st.sampled_from(_DOC_ID_ALPHABET), min_size=1, max_size=6, unique=True)
    )
    scores_by_doc: Dict[str, float] = {}
    for doc_id in doc_ids:
        scores_by_doc[doc_id] = draw(st.floats(allow_nan=False, allow_infinity=False))

    chunk_ids = [make_chunk_id(doc_id, 0) for doc_id in doc_ids]
    scores = [scores_by_doc[doc_id] for doc_id in doc_ids]
    order = draw(st.permutations(range(len(chunk_ids))))
    chunk_ids = [chunk_ids[i] for i in order]
    scores = [scores[i] for i in order]
    return scores_by_doc, chunk_ids, scores


@settings(max_examples=100)
@given(_chunk_scores_single_chunk_per_doc())
def test_property_2_n1_corollary_matches_direct_ranking(data):
    scores_by_doc, chunk_ids, scores = data
    chunk_scores = ChunkScores(chunk_ids=tuple(chunk_ids), scores=np.array(scores))
    result = aggregate_to_document_ranked_list(chunk_scores)

    direct_order = _order_by(scores_by_doc)
    assert result == direct_order


# ---------------------------------------------------------------------------
# Property 3: Every Chunker produces a well-formed, fully-covering,
# non-duplicating chunk corpus.
# Validates: Requirements 2.3, 2.4, 2.6.
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=12),
        values=st.fixed_dictionaries({"title": st.text(), "text": st.text()}),
        min_size=1,
        max_size=8,
    )
)
def test_property_3_whole_document_chunk_corpus_is_wellformed(corpus):
    chunker = WholeDocumentChunker()
    chunk_corpus = build_chunk_corpus(chunker, corpus)

    # Every produced chunk_id round-trips through parse_chunk_id to its
    # exact (doc_id, position).
    for chunk_id in chunk_corpus:
        doc_id, position = parse_chunk_id(chunk_id)
        assert doc_id in corpus
        assert position == 0  # WholeDocumentChunker always emits position 0

    # All produced chunk_ids are pairwise distinct (guaranteed by dict
    # keys, restated explicitly via the count check below).
    assert len(chunk_corpus) == len(corpus)

    # No chunk_id equals any input doc_id.
    for chunk_id in chunk_corpus:
        assert chunk_id not in corpus

    # The set of doc_ids recovered by parsing every produced chunk_id
    # equals exactly the input corpus's doc_id set.
    recovered_doc_ids = {parse_chunk_id(chunk_id)[0] for chunk_id in chunk_corpus}
    assert recovered_doc_ids == set(corpus.keys())


def test_property_3_zero_chunk_chunker_raises_chunking_error_naming_document():
    class _ZeroChunkStubChunker:
        """Hand-written stub Chunker (not one of the three real
        strategies) that deliberately returns an empty chunk list for
        one designated doc_id, to exercise build_chunk_corpus's guard
        (Requirement 2.6)."""

        strategy_name = "stub_zero_chunk"

        def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
            if doc_id == "bad-doc":
                return []
            return [
                Chunk(
                    chunk_id=make_chunk_id(doc_id, 0),
                    doc_id=doc_id,
                    position=0,
                    text=document.get("text", ""),
                )
            ]

    corpus = {
        "good-doc": {"title": "", "text": "hello"},
        "bad-doc": {"title": "", "text": "world"},
    }
    with pytest.raises(ChunkingError) as excinfo:
        build_chunk_corpus(_ZeroChunkStubChunker(), corpus)
    assert "bad-doc" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Property 4: FixedWindowChunker produces a token-budgeted,
# fully-covering, order-preserving, text-faithful partition.
# Property 5: SentenceWindowChunker produces a token-budgeted partition
# that covers every sentence exactly once.
#
# Both gated by Local_Cache_Availability -- skipped, not failed, if
# data/hf_cache's all-MiniLM-L6-v2 snapshot is absent.
# Validates: Requirements 3.3, 3.4, 3.5, 3.6, 3.7, 4.3, 4.4, 4.5, 4.6, 4.7.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def windowing_tokenizer():
    if not _local_tokenizer_cache_available():
        pytest.skip(
            "requires a local all-MiniLM-L6-v2 tokenizer cache under "
            "data/hf_cache; skipped on a clean checkout"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(_TOKENIZER_CACHE_DIR)
    )


_FIXED_WINDOW_WINDOW_SIZE = 200
_FIXED_WINDOW_STRIDE = 50
_SENTENCE_WINDOW_SENTENCES_PER_CHUNK = 3
_SENTENCE_WINDOW_MAX_CHUNK_TOKENS = 256

# Multi-paragraph document text, spanning both under- and
# over-window_size lengths: a list of 1-50 non-empty text fragments
# joined with spaces.
_document_text_strategy = st.lists(
    st.text(min_size=1, max_size=40), min_size=1, max_size=50
).map(" ".join)


@settings(max_examples=50, deadline=None)
@given(text=_document_text_strategy)
def test_property_4_fixed_window_chunker_token_budgeted_covering_ordered(
    text, windowing_tokenizer
):
    chunker = FixedWindowChunker(
        windowing_tokenizer, window_size=_FIXED_WINDOW_WINDOW_SIZE, stride=_FIXED_WINDOW_STRIDE
    )
    document = {"title": "", "text": text}
    source_text = format_document_text(document)
    chunks = chunker.chunk_document("doc-1", document)

    # Every Chunk's independently re-tokenized length <= window_size.
    for chunk in chunks:
        encoding = windowing_tokenizer(
            chunk.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        assert len(encoding["offset_mapping"]) <= _FIXED_WINDOW_WINDOW_SIZE

    # A document at or under window_size tokens yields exactly one Chunk.
    full_encoding = windowing_tokenizer(
        source_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    if len(full_encoding["offset_mapping"]) <= _FIXED_WINDOW_WINDOW_SIZE:
        assert len(chunks) == 1
        assert chunks[0].text == source_text

    # Every Chunk's text is an exact substring of the source, located
    # in non-decreasing order (Chunks are produced with non-decreasing
    # start offsets) -- coverage is checked by locating each Chunk from
    # the rightmost point already covered, never re-searching from 0
    # (which could falsely match an earlier, unrelated occurrence of a
    # short/repeated fragment).
    covered_until = 0
    for chunk in chunks:
        idx = source_text.find(chunk.text, max(0, covered_until - len(chunk.text)))
        assert idx != -1, "chunk text must be an exact substring of the source"
        covered_until = max(covered_until, idx + len(chunk.text))

    # The union of Chunks' spans covers every character (and therefore
    # every token) of the source at least once.
    assert covered_until >= len(source_text)


# A single, deliberately oversized "sentence" with no embedded
# sentence-boundary punctuation (so _split_into_sentences treats it as
# one sentence), forcing SentenceWindowChunker's token-level fallback
# (Criterion 5).
_oversized_sentence_strategy = st.text(
    alphabet=st.characters(blacklist_characters=".!?", blacklist_categories=("Cs",)),
    min_size=2000,
    max_size=4000,
)

# A document with a varying number of short, well-formed sentences.
_multi_sentence_document_strategy = st.lists(
    st.text(
        alphabet=st.characters(blacklist_characters=".!?", blacklist_categories=("Cs",)),
        min_size=1,
        max_size=30,
    ).map(lambda s: s.strip() + "."),
    min_size=1,
    max_size=10,
).map(" ".join)


@settings(max_examples=50, deadline=None)
@given(text=_multi_sentence_document_strategy)
def test_property_5_sentence_window_chunker_multi_sentence(text, windowing_tokenizer):
    chunker = SentenceWindowChunker(
        windowing_tokenizer,
        sentences_per_chunk=_SENTENCE_WINDOW_SENTENCES_PER_CHUNK,
        max_chunk_tokens=_SENTENCE_WINDOW_MAX_CHUNK_TOKENS,
    )
    document = {"title": "", "text": text}
    source_text = format_document_text(document)
    sentences = _split_into_sentences(source_text)
    chunks = chunker.chunk_document("doc-1", document)

    # Every multi-sentence Chunk's token length <= max_chunk_tokens.
    for chunk in chunks:
        encoding = windowing_tokenizer(chunk.text, add_special_tokens=False, truncation=False)
        assert len(encoding["input_ids"]) <= _SENTENCE_WINDOW_MAX_CHUNK_TOKENS

    # The concatenation of all Chunks covers every sentence of the
    # source exactly once, in order (checked by locating each sentence
    # from the point after the previous sentence was found).
    concatenated = "".join(chunk.text for chunk in chunks)
    search_from = 0
    for sentence in sentences:
        idx = concatenated.find(sentence, search_from)
        assert idx != -1, "every sentence must appear, in order, in the concatenated Chunks"
        search_from = idx + len(sentence)

    # A document with <= 3 sentences under 256 tokens yields exactly
    # one Chunk.
    full_encoding = windowing_tokenizer(source_text, add_special_tokens=False, truncation=False)
    if (
        len(sentences) <= _SENTENCE_WINDOW_SENTENCES_PER_CHUNK
        and len(full_encoding["input_ids"]) <= _SENTENCE_WINDOW_MAX_CHUNK_TOKENS
    ):
        assert len(chunks) == 1


@settings(max_examples=20, deadline=None)
@given(oversized_sentence=_oversized_sentence_strategy)
def test_property_5_sentence_window_chunker_oversized_sentence_fallback(
    oversized_sentence, windowing_tokenizer
):
    chunker = SentenceWindowChunker(
        windowing_tokenizer,
        sentences_per_chunk=_SENTENCE_WINDOW_SENTENCES_PER_CHUNK,
        max_chunk_tokens=_SENTENCE_WINDOW_MAX_CHUNK_TOKENS,
    )
    document = {"title": "", "text": oversized_sentence}
    source_text = format_document_text(document)

    full_encoding = windowing_tokenizer(source_text, add_special_tokens=False, truncation=False)
    if len(full_encoding["input_ids"]) <= _SENTENCE_WINDOW_MAX_CHUNK_TOKENS:
        # The generated text happened to tokenize under budget (rare,
        # e.g. an empty/whitespace-only draw) -- not the scenario this
        # test targets; skip rather than assert a fallback split that
        # wasn't actually needed.
        return

    chunks = chunker.chunk_document("doc-1", document)

    # Split into 2+ consecutive, non-overlapping, <= 256-token pieces.
    assert len(chunks) >= 2
    for chunk in chunks:
        encoding = windowing_tokenizer(chunk.text, add_special_tokens=False, truncation=False)
        assert len(encoding["input_ids"]) <= _SENTENCE_WINDOW_MAX_CHUNK_TOKENS

    # Concatenation covers the sentence's own token sequence exactly
    # once, in order (checked at the character level, since a
    # token-level fallback split never overlaps or drops text).
    # Compared against the single stripped sentence _split_into_sentences
    # actually produced (not the raw source_text): the whole document's
    # leading/trailing whitespace is intentionally stripped once by
    # _split_into_sentences (mirroring claim_segmenter's own slicing
    # convention, per that function's docstring), so a leading/trailing
    # space introduced by format_document_text's "title + ' ' + text"
    # concatenation (e.g. when title is empty) is never itself part of
    # any Chunk's text.
    (expected_sentence,) = _split_into_sentences(source_text)
    concatenated = "".join(chunk.text for chunk in chunks)
    assert concatenated == expected_sentence


# ---------------------------------------------------------------------------
# Property 6: BM25Retriever.retrieve_all yields every indexed chunk,
# scored, at Full_Chunk_Depth, unordered.
#
# Network-free: constructs a real BM25Retriever (no model, no network,
# matching session-1's own precedent) over a small hand-generated
# in-memory chunk corpus.
# Validates: Requirements 5.7, 6.1.
# ---------------------------------------------------------------------------


def _make_bm25_retriever() -> BM25Retriever:
    config = BM25RetrieverConfig(
        name="bm25",
        type="bm25",
        k1=1.5,
        b=0.75,
        tokenizer="regex_word",
        lowercase=True,
        stopwords="none",
        stemming="none",
    )
    return BM25Retriever(config)


# BM25Retriever's own tokenizer (regex_word, `\w+`): reused here only to
# filter out the one corpus shape rank_bm25.BM25Okapi cannot handle at
# all -- a corpus whose combined vocabulary across every document is
# empty (e.g. every document's text is pure punctuation/whitespace, so
# `\w+` matches nothing anywhere). BM25Okapi's own `average_idf =
# idf_sum / len(self.idf)` divides by zero in that case -- a genuine
# limitation of the third-party rank_bm25 library, unrelated to the
# Full_Chunk_Depth streaming contract this property actually validates
# (Requirements 5.7, 6.1). Real BEIR SciFact documents always contain
# real words, so this shape can never occur in production; it is only
# reachable here because Hypothesis's text() strategy can generate
# all-punctuation text by chance.
_WORD_PATTERN = re.compile(r"\w+")


def _has_nonempty_vocabulary(chunk_corpus: Dict[str, Dict[str, str]]) -> bool:
    return any(
        _WORD_PATTERN.search(f"{doc.get('title', '')} {doc.get('text', '')}")
        for doc in chunk_corpus.values()
    )


@settings(max_examples=50, deadline=None)
@given(
    chunk_corpus=st.dictionaries(
        keys=st.text(min_size=1, max_size=12).filter(lambda s: s.strip() != ""),
        values=st.fixed_dictionaries({"title": st.text(), "text": st.text(min_size=1)}),
        min_size=1,
        max_size=10,
    ).filter(_has_nonempty_vocabulary),
    queries=st.dictionaries(
        keys=st.text(min_size=1, max_size=8).filter(lambda s: s.strip() != ""),
        values=st.text(min_size=1, max_size=30),
        min_size=1,
        max_size=5,
    ),
)
def test_property_6_bm25_retrieve_all_yields_full_chunk_depth_unordered(chunk_corpus, queries):
    retriever = _make_bm25_retriever()
    retriever.build_index(chunk_corpus)

    expected_chunk_ids = set(chunk_corpus.keys())
    yielded_query_ids = []
    chunk_ids_objects = []

    for qid, chunk_scores in retriever.retrieve_all(queries):
        yielded_query_ids.append(qid)
        chunk_ids_objects.append(chunk_scores.chunk_ids)

        # Every yielded ChunkScores.chunk_ids is a set-equal match to
        # the indexed corpus's chunk IDs -- none truncated, none
        # duplicated, in no particular required order (Full_Chunk_Depth).
        assert set(chunk_scores.chunk_ids) == expected_chunk_ids
        assert len(chunk_scores.chunk_ids) == len(expected_chunk_ids)
        assert len(chunk_scores.scores) == len(chunk_scores.chunk_ids)

    # Every query in `queries` was yielded exactly once, in queries'
    # own iteration order.
    assert yielded_query_ids == list(queries.keys())

    # The same chunk_ids tuple OBJECT (is, not ==) is reused,
    # unmodified, across every yield within this one retrieve_all call.
    first_chunk_ids = chunk_ids_objects[0]
    for chunk_ids in chunk_ids_objects[1:]:
        assert chunk_ids is first_chunk_ids

    # last_query_latency is set only once the generator is exhausted,
    # and is a non-negative float.
    assert isinstance(retriever.last_query_latency, float)
    assert retriever.last_query_latency >= 0.0
