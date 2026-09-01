"""Chunking abstraction: the `Chunker` protocol, chunk-ID codec, and
`build_chunk_corpus`, plus (in later parts of this module) the three
declared `Chunking_Strategy` implementations and `Max_Aggregation`.

This module has no dependency on any retriever class; it depends only
on `src.retrievers.dense_retriever.format_document_text` (reused, not
re-derived, for `WholeDocumentChunker`) and, for the two token-aware
strategies added later in this module, a loaded Hugging Face
tokenizer.

Part 1 (this revision): `CHUNK_ID_SEPARATOR`, `Chunk`, `make_chunk_id`,
`parse_chunk_id`, the `Chunker` Protocol, `build_chunk_corpus`, and
`WholeDocumentChunker` (Requirements 2.2, 2.3, 2.4, 2.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

from src.claim_segmenter import _SENTENCE_BOUNDARY
from src.errors import ChunkingError
from src.retrievers.base import ChunkScores, doc_id_sort_key

# `format_document_text` is imported lazily, inside each function that
# needs it (below), rather than at this module's own top. `src.chunking`
# has no dependency on any retriever class other than this one function,
# but `src.retrievers.dense_retriever` itself imports
# `sentence_transformers` (and transitively `huggingface_hub`) at ITS
# OWN module top -- so a top-level `from src.retrievers.dense_retriever
# import format_document_text` here would mean merely importing
# `src.chunking` (e.g. from `src/sweep_runner.py`'s own module top, to
# get `build_chunk_corpus`/`aggregate_to_document_ranked_list`) always
# imports `huggingface_hub` too, resolving its cache path before
# `configure_caches()` has a chance to run (Requirement 10.5) --
# exactly the ordering hazard `src/sweep_runner.py`'s and
# `src/corpus_loader.py`'s own docstrings already defend against for
# every other module in this codebase. Deferring the import here keeps
# that guarantee intact for `src.chunking` too.

# Never occurs in a BEIR SciFact doc_id, so a chunk_id can never be
# mistaken for a bare document ID.
CHUNK_ID_SEPARATOR = "::chunk"


@dataclass(frozen=True)
class Chunk:
    """One Chunker's output unit: a contiguous span of a single source
    document's `title` + `text` content."""

    chunk_id: str  # make_chunk_id(doc_id, position)
    doc_id: str  # source document ID
    position: int  # 0-based position within doc_id's ordered chunk list
    text: str  # this Chunk's own text content


def make_chunk_id(doc_id: str, position: int) -> str:
    """`doc_id + CHUNK_ID_SEPARATOR + position` (Requirement 2.3): never
    equal to `doc_id` itself (the separator is always appended), and
    always parseable back via `parse_chunk_id`."""
    return f"{doc_id}{CHUNK_ID_SEPARATOR}{position}"


def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """Inverse of `make_chunk_id`. Splits on the LAST occurrence of
    `CHUNK_ID_SEPARATOR` (`str.rpartition`) rather than the first, so a
    `doc_id` that happens to contain the separator substring still
    parses correctly -- the separator we appended is always the
    rightmost occurrence in a string this module produced.

    Raises `ValueError` if `chunk_id` contains no separator or the
    trailing segment is not an integer (Requirement 2.3's "recoverable
    by parsing the identifier alone")."""
    doc_id, sep, position_str = chunk_id.rpartition(CHUNK_ID_SEPARATOR)
    if not sep:
        raise ValueError(f"not a well-formed chunk_id: {chunk_id!r}")
    try:
        position = int(position_str)
    except ValueError:
        raise ValueError(f"not a well-formed chunk_id: {chunk_id!r}") from None
    return doc_id, position


class Chunker(Protocol):
    """The pure component that maps one corpus document to an ordered
    list of one or more Chunks, according to one declared
    Chunking_Strategy."""

    strategy_name: str

    def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
        """Splits one corpus document into an ordered, 0-indexed list
        of Chunks. Never returns an empty list for a Chunker
        implementation this spec defines (Requirement 2.6's guard
        exists for a hypothetically misbehaving Chunker, e.g. a test
        stub, not for any of the three below)."""
        ...


def build_chunk_corpus(
    chunker: Chunker, corpus: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    """Applies `chunker` to every document in `corpus` (Requirement 2.4
    -- no document skipped), and returns a chunk corpus: `chunk_id ->
    {"title": "", "text": chunk.text}`. Every Chunk's own text is
    stored entirely in `"text"` with an empty `"title"`, so
    `BM25Retriever`'s tokenization and `DenseRetriever`'s
    `format_document_text` -- both of which concatenate
    `title + " " + text` -- consume a Chunk's content unchanged, with
    no retriever-side awareness that chunking happened at all.

    Raises `ChunkingError`, naming the offending `doc_id` and
    `chunker.strategy_name`, and does not proceed to build any index,
    if `chunker.chunk_document` returns an empty list for any document
    (Requirement 2.6)."""
    chunk_corpus: Dict[str, Dict[str, str]] = {}
    for doc_id, document in corpus.items():
        chunks = chunker.chunk_document(doc_id, document)
        if not chunks:
            raise ChunkingError(
                f"{chunker.strategy_name!r} produced zero Chunks for "
                f"document {doc_id!r}; halting before any index build"
            )
        for chunk in chunks:
            chunk_corpus[chunk.chunk_id] = {"title": "", "text": chunk.text}
    return chunk_corpus


class WholeDocumentChunker:
    """The Chunker for the `whole_document` Chunking_Strategy: a no-op
    wrapper of session-1's existing whole-document behavior. Produces
    exactly one Chunk per corpus document, containing that document's
    full, unmodified content (Requirement 2.2)."""

    strategy_name = "whole_document"

    def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
        from src.retrievers.dense_retriever import format_document_text

        text = format_document_text(document)
        return [
            Chunk(chunk_id=make_chunk_id(doc_id, 0), doc_id=doc_id, position=0, text=text)
        ]


# ---------------------------------------------------------------------------
# Part 2: FixedWindowChunker (Requirements 3.1-3.7).
#
# Token-length measurement, window slicing, and the isolated-re-
# tokenization verify/trim step are all performed via a Hugging Face
# `PreTrainedTokenizerBase`-compatible tokenizer instance passed in by
# the caller (the already-cached all-MiniLM-L6-v2 tokenizer, in every
# production and test use of this class) -- this module never loads a
# tokenizer itself.
# ---------------------------------------------------------------------------


def _slice_by_offset(tokenizer, text: str, max_tokens: int) -> str:
    """Tokenizes `text` and slices out the character span covering at
    most the first `max_tokens` tokens (or the whole text, if it
    already fits), via the tokenizer's own offset mapping (Requirement
    3.7 -- an exact, reconstructable text span, never a token-id
    array)."""
    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    offsets = encoding["offset_mapping"]
    if len(offsets) <= max_tokens:
        return text
    char_end = offsets[max_tokens - 1][1]
    return text[:char_end]


def _verify_and_trim(tokenizer, candidate: str, max_tokens: int) -> str:
    """Re-tokenizes `candidate` in isolation; if its own re-tokenized
    length still exceeds `max_tokens` -- a window boundary landing
    mid-word can change subword segmentation once the fragment is
    tokenized alone (Requirement 3.3) -- repeatedly trims the last
    re-tokenized token's own character span off the end and re-checks,
    until the length is at or under `max_tokens`. Only ever trims,
    never grows, so it cannot violate coverage or ordering."""
    while candidate:
        encoding = tokenizer(
            candidate, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        offsets = encoding["offset_mapping"]
        if len(offsets) <= max_tokens:
            return candidate
        last_start = offsets[-1][0]
        trimmed = candidate[:last_start] if last_start < len(candidate) else candidate[:-1]
        if trimmed == candidate:
            # No progress from offset-based trimming alone (e.g. a
            # degenerate zero-width trailing token) -- forcibly drop
            # one character so the loop always makes progress.
            trimmed = candidate[:-1]
        candidate = trimmed
    return candidate


def _split_text_by_token_budget(tokenizer, text: str, max_tokens: int) -> List[str]:
    """Splits `text` into consecutive, non-overlapping pieces, each of
    which re-tokenizes -- in isolation -- to at most `max_tokens`
    tokens, covering `text` exactly once in left-to-right order with
    nothing dropped or duplicated.

    Slices a token-offset span (`_slice_by_offset`), then verifies and,
    if needed, trims that slice's own isolated re-tokenization
    (`_verify_and_trim`) -- the shared trim-and-verify core reused both
    by `FixedWindowChunker._shrink_to_token_budget` (this module) and
    by `SentenceWindowChunker`'s oversized-sentence fallback (Task 6),
    so neither re-implements the same logic."""
    if text == "":
        return [text]
    pieces: List[str] = []
    remaining = text
    while remaining:
        candidate = _slice_by_offset(tokenizer, remaining, max_tokens)
        piece = _verify_and_trim(tokenizer, candidate, max_tokens)
        if not piece:
            # Degenerate case (e.g. the leading token re-tokenizes to
            # nothing under isolation) -- take at least one character
            # so the loop always makes progress and text is never
            # silently dropped.
            piece = remaining[0]
        pieces.append(piece)
        remaining = remaining[len(piece):]
    return pieces


class FixedWindowChunker:
    """The Chunker for the `fixed_window` Chunking_Strategy
    (Requirement 3). Splits a document into consecutive, possibly
    overlapping Chunks of `window_size` all-MiniLM-L6-v2 subword
    tokens, advancing the chunk start position by `stride` tokens
    between consecutive Chunks."""

    strategy_name = "fixed_window"

    def __init__(self, tokenizer, window_size: int, stride: int) -> None:
        self._tokenizer = tokenizer
        self._window_size = window_size
        self._stride = stride

    def _shrink_to_token_budget(self, candidate_text: str) -> str:
        """Fixes up one already offset-sliced window candidate whose
        own isolated re-tokenization may exceed `window_size`
        (Requirement 3.3). Delegates to the shared
        `_split_text_by_token_budget` core and keeps only its first
        piece -- the trimmed prefix -- since whatever overflow this
        discards is exactly the text the next, stride-advanced window
        covers instead. Only ever trims, never grows."""
        return _split_text_by_token_budget(
            self._tokenizer, candidate_text, self._window_size
        )[0]

    def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
        from src.retrievers.dense_retriever import format_document_text

        text = format_document_text(document)
        encoding = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        offsets = encoding["offset_mapping"]
        total_tokens = len(offsets)

        if total_tokens <= self._window_size:
            # Requirement 3.4: a document at or under window_size
            # tokens yields exactly one Chunk containing its full
            # content.
            return [
                Chunk(chunk_id=make_chunk_id(doc_id, 0), doc_id=doc_id, position=0, text=text)
            ]

        chunks: List[Chunk] = []
        position = 0
        start = 0
        # The rightmost character position covered by any Chunk
        # appended so far. `_shrink_to_token_budget` only ever trims
        # from the end of a window's candidate text (Requirement 3.3's
        # isolated-re-tokenization fix-up), so a gap can only ever open
        # up between one window's actual (post-trim) coverage and the
        # next window's start -- never anywhere else. Tracking this
        # explicitly, and filling any such gap with one or more
        # additional Chunks before moving on, guarantees every
        # character (and therefore every token) of the document is
        # covered by at least one Chunk (Requirement 3.5) regardless of
        # how much any individual window happened to be trimmed.
        covered_char_end = 0
        while True:
            end = min(start + self._window_size, total_tokens)
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            candidate_text = text[char_start:char_end]
            chunk_text = self._shrink_to_token_budget(candidate_text)

            if char_start > covered_char_end:
                # A prior window's trim left a gap before this
                # window's start -- fill it now, in order, before
                # appending this window's own Chunk.
                gap_text = text[covered_char_end:char_start]
                for piece in _split_text_by_token_budget(
                    self._tokenizer, gap_text, self._window_size
                ):
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(doc_id, position),
                            doc_id=doc_id,
                            position=position,
                            text=piece,
                        )
                    )
                    position += 1

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, position),
                    doc_id=doc_id,
                    position=position,
                    text=chunk_text,
                )
            )
            position += 1
            covered_char_end = max(covered_char_end, char_start + len(chunk_text))

            if end >= total_tokens:
                if covered_char_end < len(text):
                    # The final window's own trim left the document's
                    # tail uncovered -- there is no later window to
                    # absorb it, so cover it explicitly (Requirement
                    # 3.5).
                    tail_text = text[covered_char_end:]
                    for piece in _split_text_by_token_budget(
                        self._tokenizer, tail_text, self._window_size
                    ):
                        chunks.append(
                            Chunk(
                                chunk_id=make_chunk_id(doc_id, position),
                                doc_id=doc_id,
                                position=position,
                                text=piece,
                            )
                        )
                        position += 1
                break
            # Requirement 3.6: left-to-right, non-decreasing start
            # offsets between consecutive Chunks.
            start += self._stride
        return chunks


# ---------------------------------------------------------------------------
# Part 3: SentenceWindowChunker (Requirements 4.1-4.7).
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> List[str]:
    """Splits `text` into an ordered list of sentence strings, reusing
    `src.claim_segmenter._SENTENCE_BOUNDARY` directly (Requirement 4.1)
    -- the one sentence-boundary heuristic in this repository, applied
    here to document text rather than Generated_Answer text.

    Mirrors `claim_segmenter.segment_claims`'s own slicing convention:
    each segment keeps its terminating boundary punctuation, is
    stripped of leading/trailing whitespace, and an empty segment is
    dropped. If no sentence boundary is found in the trimmed text
    (including when the trimmed text is the empty string), the entire
    trimmed text is returned as the sole sentence -- so this function
    always returns at least one element, never an empty list, mirroring
    `WholeDocumentChunker`'s guarantee that no Chunker this spec defines
    ever produces zero output for a document."""
    trimmed = text.strip()
    if not _SENTENCE_BOUNDARY.search(trimmed):
        return [trimmed]

    sentences: List[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(trimmed):
        segment = trimmed[start:match.end()].strip()
        if segment:
            sentences.append(segment)
        start = match.end()
    tail = trimmed[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [trimmed]


def _token_length(tokenizer, text: str) -> int:
    """Returns `text`'s all-MiniLM-L6-v2 tokenizer token count,
    excluding any special tokens the tokenizer would add."""
    encoding = tokenizer(text, add_special_tokens=False, truncation=False)
    return len(encoding["input_ids"])


def _split_group_to_budget(
    tokenizer, sentences: List[str], max_chunk_tokens: int
) -> List[str]:
    """Greedily packs `sentences` (an ordered, non-overlapping group of
    up to `sentences_per_chunk` consecutive sentences) into the maximum
    number of consecutive whole sentences whose combined token length
    is `<= max_chunk_tokens`, repeating for the remainder (Requirement
    4.4). If a single sentence's own token length exceeds
    `max_chunk_tokens`, splits that one sentence via the shared
    `_split_text_by_token_budget` helper into consecutive,
    non-overlapping, `<= max_chunk_tokens`-sized pieces covering it
    exactly once (Requirement 4.5) -- the shared trim-and-verify core
    Task 5's `FixedWindowChunker` already defines, reused rather than
    re-implemented.

    Every sentence in `sentences` is covered exactly once across the
    returned pieces, in original order, with nothing dropped or
    duplicated (Requirement 4.6)."""
    if not sentences:
        return []

    combined = " ".join(sentences)
    if _token_length(tokenizer, combined) <= max_chunk_tokens:
        return [combined]

    pieces: List[str] = []
    i = 0
    n = len(sentences)
    while i < n:
        current = sentences[i]
        current_len = _token_length(tokenizer, current)
        if current_len > max_chunk_tokens:
            # Requirement 4.5: a single sentence's own token length
            # exceeds the budget -- no whole-sentence packing can help;
            # split its token sequence directly.
            pieces.extend(
                _split_text_by_token_budget(tokenizer, current, max_chunk_tokens)
            )
            i += 1
            continue

        # Requirement 4.4: greedily extend the packed group with as
        # many further consecutive whole sentences as fit under the
        # budget.
        j = i + 1
        while j < n:
            candidate = current + " " + sentences[j]
            if _token_length(tokenizer, candidate) > max_chunk_tokens:
                break
            current = candidate
            j += 1
        pieces.append(current)
        i = j
    return pieces


class SentenceWindowChunker:
    """The Chunker for the `sentence_window` Chunking_Strategy
    (Requirement 4). Splits a document into sentences via
    `_split_into_sentences`, groups consecutive sentences into Chunks
    of up to `sentences_per_chunk` sentences each, and further splits
    any group exceeding `max_chunk_tokens` at sentence (or, as a last
    resort, token) boundaries."""

    strategy_name = "sentence_window"

    def __init__(self, tokenizer, sentences_per_chunk: int, max_chunk_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._sentences_per_chunk = sentences_per_chunk
        self._max_chunk_tokens = max_chunk_tokens

    def chunk_document(self, doc_id: str, document: Dict[str, str]) -> List[Chunk]:
        from src.retrievers.dense_retriever import format_document_text

        text = format_document_text(document)
        sentences = _split_into_sentences(text)

        chunks: List[Chunk] = []
        position = 0
        # Requirement 4.3: partition sentences, in order, without
        # overlap, into groups of up to sentences_per_chunk consecutive
        # sentences, before any further splitting.
        for group_start in range(0, len(sentences), self._sentences_per_chunk):
            group = sentences[group_start : group_start + self._sentences_per_chunk]
            for piece in _split_group_to_budget(
                self._tokenizer, group, self._max_chunk_tokens
            ):
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc_id, position),
                        doc_id=doc_id,
                        position=position,
                        text=piece,
                    )
                )
                position += 1
        return chunks


# ---------------------------------------------------------------------------
# Part 4: aggregate_to_document_ranked_list, load_chunking_tokenizer
# (Requirements 2.5, 5.1, 5.2, 5.5, 5.6, 10.2).
# ---------------------------------------------------------------------------


def aggregate_to_document_ranked_list(chunk_scores: ChunkScores) -> List[str]:
    """Reduces one query's `ChunkScores` (every indexed Chunk, scored)
    to a `Document_Ranked_List`: an ordered list of source document
    IDs, via Max_Aggregation (Requirement 5.1 -- the maximum numeric
    score among each document's own Chunks, never the mean or the
    sum).

    Groups `chunk_scores.chunk_ids`/`scores` by parsed source `doc_id`
    (via `parse_chunk_id`), computes each document's maximum chunk
    score, and returns document IDs ordered by `(-max_score,
    doc_id_sort_key(doc_id))` -- descending score, ties broken by
    ascending document ID (Requirement 5.5, the same tie-break
    `BM25Retriever`/`DenseRetriever` used at the per-chunk level in
    session 1, now applied at the document level after aggregation).
    This is the only sort in the whole chunk-to-cutoff path, since
    `ChunkScores` itself is never sorted (Requirement 5.2).

    Pure: no I/O, no retriever call, operates only on the single
    `ChunkScores` already passed in (Requirement 5.6). For
    `whole_document` chunking, every document has exactly one Chunk, so
    this reduces to the identity ranking by that Chunk's own score
    (Requirement 2.5's `n = 1` corollary)."""
    max_score_by_doc: Dict[str, float] = {}
    for chunk_id, score in zip(chunk_scores.chunk_ids, chunk_scores.scores):
        doc_id, _position = parse_chunk_id(chunk_id)
        current = max_score_by_doc.get(doc_id)
        if current is None or score > current:
            max_score_by_doc[doc_id] = score

    ordered_doc_ids = sorted(
        max_score_by_doc.keys(),
        key=lambda doc_id: (-max_score_by_doc[doc_id], doc_id_sort_key(doc_id)),
    )
    return ordered_doc_ids


def load_chunking_tokenizer(cache_folder: Path):
    """Loads the all-MiniLM-L6-v2 tokenizer used by `FixedWindowChunker`
    and `SentenceWindowChunker`, from (or, on first invocation,
    downloaded into) `cache_folder`.

    Unlike `src.token_length_analysis`'s forced-offline loader, this
    function allows a network download on a first, uncached invocation
    -- Task 5's and Task 6's done checks already exercise this same
    tokenizer from the already-cached `data/hf_cache`, so in every
    production and test use in this spec the weights are already local
    and no network call actually occurs."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(cache_folder)
    )
