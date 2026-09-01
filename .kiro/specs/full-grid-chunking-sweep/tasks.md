# Implementation Plan

## Overview

Ordering rule: foundational, network-free pieces come first — the new
exception types, the `hypothesis` pin, the `Retriever` protocol's
streaming `ChunkScores` contract, and `src/chunking.py` itself (the one
genuinely new module, with no dependency on any retriever or config
class) — before anything that depends on them. Within `src/chunking.py`,
each of `WholeDocumentChunker` (Task 4), `FixedWindowChunker` (Task 5),
`SentenceWindowChunker` (Task 6), and `aggregate_to_document_ranked_list`/
`load_chunking_tokenizer` (Task 7) is its own task, in that order, because
`SentenceWindowChunker`'s oversized-sentence fallback shares its
token-splitting helper with `FixedWindowChunker`'s shrink mechanism —
Task 6 depends on Task 5's helper existing first. `BM25Retriever`/
`DenseRetriever`'s streaming-contract update (Tasks 10, 11) depends only
on Task 3's `ChunkScores`, not on `src/chunking.py`, so they proceed in
parallel with the chunking module. Task 12 (the BM25 Full_Chunk_Depth
property test) comes after both the chunking property tests (Tasks 8, 9)
and the retriever update (Task 10), since it extends the same
`tests/test_chunking.py` file. Task 13 is a checkpoint: the full
`tests/test_chunking.py` suite must be green before `src/config.py`
(Task 14) and `src/sweep_runner.py` (Task 16) are restructured around
the chunking abstraction. Task 18 is a correctness gate — the chunking
abstraction must reproduce the already-published whole-document
baseline before any new-strategy result can be trusted — and Task 19 is
an informational runtime measurement; both are numbered immediately
after Task 17, but both are scheduled to run once the full pytest suite
(the renumbered checkpoint, Task 27) is green, since they need the
complete, tested chunking/retriever/sweep_runner implementation but not
the significance/retrieval_replay/token_length_analysis work (Tasks
20–26) — so they execute as their own wave immediately before Task 28,
not immediately after Task 17 in actual execution order.
`src/significance_config.py`/`src/significance.py`
(Tasks 20, 22), `src/retrieval_replay.py` (Task 23), and
`src/token_length_analysis.py` (Task 24) are independent of each other
and of `src/sweep_runner.py`'s own internals — they depend only on
`src/chunking.py` and/or `src/retrievers/base.py` being finished — so
they proceed as parallel siblings once those are done. Task 27 is a
second checkpoint: the entire network-free `pytest` suite (including the
new, skip-gated `tests/test_data_layer.py`) must be green before any
task that touches the network. Task 28 (the real end-to-end sweep) is
the first task in this spec's sequence that touches the network — a
one-time `bge-small-en-v1.5` weight download, since `bm25` and
`all-MiniLM-L6-v2` are already cached from session 1. Tasks 29–31
(the real significance re-run, the real 6-cell token-length report, and
the retrieval-replay equivalence rerun-and-diff) all depend on Task 28's
artifacts and proceed in parallel with each other. Task 32 (the
README/SPEC/traceability updates) depends on all three, and Task 33 (the
real Verification_Pass) is last, mirroring `repo-writeup`'s own
Task 9/Task 10 closing sequence.

Only `tests/test_chunking.py` (new), `tests/test_orchestration.py`
(extended), `tests/test_data_layer.py` (new), and
`tests/test_token_length_analysis.py` (extended) are touched by this
spec's test tasks. `tests/test_metrics.py`, `tests/test_significance.py`,
`tests/test_claim_segmenter.py`, `tests/test_quarantine_rule.py`, and
`tests/test_verify_writeup_numbers.py` are never touched (Requirement
12.6). All done checks are Git Bash / POSIX shell or Python one-liners,
per shell-conventions.md.

## Tasks

- [x] 1. Extend `src/errors.py` with `ChunkingError` and `ChunkingConfigError`
  - Add `ChunkingError(Exception)`: raised by `build_chunk_corpus` when a
    `Chunker` produces zero Chunks for a document (Requirement 2.6), and
    reused, wrapped as `FrozenRetrieverConfigError`, if
    `retrieval_replay.build_frozen_retriever`'s `WholeDocumentChunker`
    application ever fails.
  - Add `ChunkingConfigError(ConfigError)`: raised by `src/config.py`'s
    per-Chunking_Strategy-entry field validation (`fixed_window`'s
    `window_size`/`stride`, `sentence_window`'s `sentences_per_chunk`/
    `max_chunk_tokens` must each be positive integers, Requirement 7.6) —
    a `ConfigError` subclass, mirroring `UnsupportedPreprocessingError`'s
    relationship to `ConfigError`. Do NOT redefine `TokenizerLoadError`;
    it already exists (repo-writeup spec) and is reused verbatim by
    `load_chunking_tokenizer`.
  - Done check:
    `python -c "from src.errors import ConfigError, ChunkingError, ChunkingConfigError, TokenizerLoadError; assert issubclass(ChunkingConfigError, ConfigError); [cls('x') for cls in (ChunkingError, ChunkingConfigError)]; print('ok')"`
    prints `ok` with exit code 0.
  - _Requirements: 2.6, 7.6_

- [x] 2. Pin `hypothesis` in `requirements.txt`
  - Add `hypothesis==6.167.1` (or whatever version `pip install
    hypothesis` actually resolves in this environment — resolve it the
    same way Task 1 of `session-1-baseline-sweep` resolved its own
    versions: install unpinned, confirm the import, then copy the
    resolved version from `pip freeze`) as a new line in
    `requirements.txt`, in alphabetical position. This is a test-only
    dependency: no `src/` module imports `hypothesis` anywhere in this
    spec.
  - Done check: `pip install -r requirements.txt` exits 0, then
    `python -c "import hypothesis; print('ok')"` prints `ok`. Additionally,
    `! grep -rl 'import hypothesis' src/ && echo ok` prints `ok` (the `!`
    asserts no `src/` file imports it).
  - _Requirements: (Testing Strategy — hypothesis is a new, pinned, test-only dependency)_

- [x] 3. Extend `src/retrievers/base.py`: `ChunkScores`, and the streaming `Retriever` protocol
  - Remove the `RetrievalRun` dataclass entirely — no retriever bundles
    every query's result into one object anymore, so there is nothing
    left for it to hold.
  - Add the frozen `ChunkScores` dataclass: `chunk_ids: Tuple[str, ...]`
    (the stable order fixed once at `build_index` time; the same tuple
    object is reused, unmodified, across every query a single
    `retrieve_all` call yields) and `scores: numpy.ndarray` (shape
    `(len(chunk_ids),)`). Never sorted.
  - Change the `Retriever` Protocol: drop `top_k` from `retrieve_all`'s
    signature entirely (Full_Chunk_Depth always, Requirement 6.1, 6.4);
    change `retrieve_all`'s return type from
    `Tuple[Dict[str, List[str]], float]` to
    `Iterator[Tuple[str, ChunkScores]]` — a generator, yielding one
    `(query_id, ChunkScores)` pair at a time in `queries`' own iteration
    order; add `last_query_latency: float`, populated only once the
    returned generator has been fully exhausted (Requirement 5.7).
    `doc_id_sort_key` is unchanged.
  - Done check:
    `python -c "import inspect, src.retrievers.base as b; assert not hasattr(b, 'RetrievalRun'); import dataclasses; assert {f.name for f in dataclasses.fields(b.ChunkScores)} == {'chunk_ids', 'scores'}; sig = inspect.signature(b.Retriever.retrieve_all); assert 'top_k' not in sig.parameters; print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 5.7, 6.1, 6.4_

- [x] 4. Write `src/chunking.py` (part 1): chunk-ID codec, `Chunker` protocol, `build_chunk_corpus`, `WholeDocumentChunker`
  - Implement `CHUNK_ID_SEPARATOR = "::chunk"`, the frozen `Chunk`
    dataclass (`chunk_id`, `doc_id`, `position`, `text`),
    `make_chunk_id(doc_id, position) -> str` (`f"{doc_id}{CHUNK_ID_SEPARATOR}{position}"`),
    and `parse_chunk_id(chunk_id) -> Tuple[str, int]` (splits on the
    *last* occurrence of `CHUNK_ID_SEPARATOR` via `str.rpartition`;
    raises `ValueError` if `chunk_id` contains no separator or the
    trailing segment is not an integer) — Requirement 2.3.
  - Implement the `Chunker` `Protocol` (`strategy_name: str`,
    `chunk_document(doc_id, document) -> List[Chunk]`) and
    `build_chunk_corpus(chunker, corpus) -> Dict[str, Dict[str, str]]`:
    applies `chunker` to every document in `corpus` (Requirement 2.4),
    storing each Chunk's text under `{"title": "", "text": chunk.text}`
    (so `BM25Retriever`/`DenseRetriever` consume it unchanged, with no
    awareness that chunking occurred); raises `ChunkingError` naming the
    offending `doc_id` and `chunker.strategy_name`, and does not proceed
    to build any index, if `chunk_document` returns an empty list for
    any document (Requirement 2.6).
  - Implement `WholeDocumentChunker` (`strategy_name = "whole_document"`):
    returns exactly one `Chunk` at position `0`, whose `text` equals
    `format_document_text(document)` (imported from
    `src.retrievers.dense_retriever` — reused, not re-derived) exactly
    (Requirement 2.2).
  - Done check:
    `python -c "from src.chunking import make_chunk_id, parse_chunk_id, build_chunk_corpus, WholeDocumentChunker, Chunk; assert parse_chunk_id(make_chunk_id('37', 2)) == ('37', 2); assert make_chunk_id('37', 0) != '37'; c = build_chunk_corpus(WholeDocumentChunker(), {'1': {'title': 'T', 'text': 'X'}}); assert list(c.keys()) == ['1::chunk0']; assert c['1::chunk0']['text'] == 'T X'; print('ok')"`
    prints `ok`. Additionally, a script constructing a hand-written stub
    `Chunker` whose `chunk_document` returns `[]` for one designated
    `doc_id` and asserting `build_chunk_corpus` raises `ChunkingError`
    naming that `doc_id` prints `ok` and exits 0.
  - _Requirements: 2.2, 2.3, 2.4, 2.6_

- [x] 5. Extend `src/chunking.py` (part 2): `FixedWindowChunker`
  - Implement `FixedWindowChunker(tokenizer, window_size, stride)`
    (`strategy_name = "fixed_window"`): tokenizes
    `format_document_text(document)` with `add_special_tokens=False,
    return_offsets_mapping=True, truncation=False`; if the total token
    count is `<= window_size`, returns exactly one Chunk containing the
    full text (Requirement 3.4); otherwise slides a window of
    `window_size` tokens forward by `stride` tokens between consecutive
    starts (Requirement 3.1), using the tokenizer's own offset mapping to
    slice each window's exact character span (Requirement 3.7), until the
    last window reaches the end of the document (Requirement 3.5, 3.6 —
    left-to-right, non-decreasing start offsets).
  - Implement the private `_shrink_to_token_budget(candidate_text) -> str`
    helper: re-tokenizes `candidate_text` in isolation; if its
    independently re-tokenized length exceeds `window_size`, repeatedly
    trims the last re-tokenized token's own character span off the end
    (using that re-tokenization's own offset mapping) and re-checks,
    until the length is at or under `window_size` (Requirement 3.3 — a
    window boundary landing mid-word can change subword segmentation
    once the fragment is tokenized in isolation). Only ever trims,
    never grows, so it cannot violate coverage or ordering.
  - Refactor the token-slicing core of `_shrink_to_token_budget` into a
    private module-level helper,
    `_split_text_by_token_budget(tokenizer, text, max_tokens) -> List[str]`
    (slice a token-offset span, then verify), so Task 6's
    `SentenceWindowChunker` oversized-sentence fallback can reuse it
    rather than re-implementing the same trim-and-verify logic.
  - Done check: a script that loads the already-cached all-MiniLM-L6-v2
    tokenizer (`AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="data/hf_cache")`), constructs
    `FixedWindowChunker(tokenizer, window_size=200, stride=50)`, and
    asserts: a short hand-written document (well under 200 tokens)
    yields exactly one Chunk containing its full text; a long
    hand-written document (several thousand characters, spanning many
    windows) yields 2+ Chunks, each of whose text, when independently
    re-tokenized, has a token count `<= 200`; the Chunks' concatenated
    character spans cover every character of the source text at least
    once; and each Chunk's text is an exact substring of the source
    document's formatted text. Prints `ok` and exits 0.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 6. Extend `src/chunking.py` (part 3): `SentenceWindowChunker`
  - Implement the private `_split_into_sentences(text) -> List[str]`
    helper, importing and reusing `src.claim_segmenter._SENTENCE_BOUNDARY`
    directly (never re-implemented) — Requirement 4.1.
  - Implement `SentenceWindowChunker(tokenizer, sentences_per_chunk,
    max_chunk_tokens)` (`strategy_name = "sentence_window"`): splits
    `format_document_text(document)` into sentences via
    `_split_into_sentences`, partitions them in order, without overlap,
    into groups of up to `sentences_per_chunk` consecutive sentences
    (Requirement 4.3), then calls the private
    `_split_group_to_budget(sentences) -> List[str]` helper on each group.
  - Implement `_split_group_to_budget`: greedily packs the group's
    sentences into the maximum number of consecutive whole sentences
    whose combined token length is `<= max_chunk_tokens`, repeating for
    the remainder (Requirement 4.4); if a single sentence's own token
    length still exceeds `max_chunk_tokens`, splits that one sentence via
    Task 5's shared `_split_text_by_token_budget` helper into consecutive,
    non-overlapping, `<= max_chunk_tokens`-sized pieces covering it
    exactly once (Requirement 4.5). A document with `<=
    sentences_per_chunk` sentences that also fits under
    `max_chunk_tokens` yields exactly one Chunk (Requirement 4.7); every
    sentence (or, for a token-split sentence, its full token sequence) is
    covered exactly once across the returned Chunks, with nothing dropped
    or duplicated (Requirement 4.6).
  - Done check: a script using the same cached tokenizer as Task 5,
    constructing `SentenceWindowChunker(tokenizer, sentences_per_chunk=3,
    max_chunk_tokens=256)`, and asserting: a 2-sentence hand-written
    document yields exactly one Chunk; a hand-written document of 7 short
    sentences yields Chunks each containing `<= 3` sentences in original
    order; a hand-written single sentence of 2000+ non-punctuated
    characters (deliberately exceeding 256 tokens) is split into 2+
    Chunks, each `<= 256` tokens, whose concatenated text reconstructs the
    original sentence with no character dropped or duplicated. Prints
    `ok` and exits 0.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 7. Extend `src/chunking.py` (part 4): `aggregate_to_document_ranked_list`, `load_chunking_tokenizer`
  - Implement `aggregate_to_document_ranked_list(chunk_scores:
    ChunkScores) -> List[str]`: groups `chunk_scores.chunk_ids`/`scores`
    by parsed source `doc_id` (via `parse_chunk_id`), computes each
    document's maximum chunk score (Requirement 5.1 — never mean or
    sum), and returns document IDs ordered by `(-max_score,
    doc_id_sort_key(doc_id))` (Requirement 5.5's tie-break, imported from
    `src.retrievers.base`) — the only sort in the whole chunk-to-cutoff
    path, since `ChunkScores` itself is never sorted. Pure: no I/O, no
    retriever call, operates only on the single `ChunkScores` already
    passed in (Requirement 5.6).
  - Implement `load_chunking_tokenizer(cache_folder: Path) ->
    PreTrainedTokenizerBase`: loads the all-MiniLM-L6-v2 tokenizer via
    `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=str(cache_folder))`, allowing a network download on a
    first, uncached invocation (unlike `token_length_analysis.py`'s
    forced-offline loader) — Task 5's/6's done checks already exercise
    this same tokenizer from the already-cached `data/hf_cache`.
  - Done check: a script building a hand-specified `ChunkScores` for two
    documents — one with 3 chunks of differing scores (max clearly
    distinct from mean/sum) and two documents tied at the same maximum
    score — and asserting `aggregate_to_document_ranked_list` returns the
    document with the true maximum first, that the result differs from
    what a mean- or sum-based aggregation would produce, and that the two
    tied documents appear in ascending `doc_id_sort_key` order. Prints
    `ok` and exits 0. No network call (uses a hand-built `ChunkScores`,
    not a real tokenizer).
  - _Requirements: 2.5, 5.1, 5.2, 5.5, 5.6, 10.2_

- [x] 8. Write `tests/test_chunking.py` — Properties 1, 2, 3
  - **Property 1: Whole-document chunking preserves content exactly.**
    `hypothesis.strategies.text()` (including empty string and non-ASCII)
    for `title`/`text`, composed into a document dict; asserts
    `WholeDocumentChunker.chunk_document` returns exactly one Chunk at
    position 0 whose text equals `format_document_text(document)`
    exactly. **Validates: Requirement 2.2.**
  - **Property 2: Max_Aggregation is the true maximum, tie-broken by
    ascending document ID, and reduces to the identity for single-chunk
    documents.** A custom strategy generating documents each assigned
    1–5 chunks with `hypothesis.strategies.floats(allow_nan=False)`
    scores — some documents deliberately given 2+ chunks with distinct
    scores (so mean/sum are guaranteed to diverge from max whenever the
    generated scores differ), some documents deliberately tied at equal
    maximum scores, and the generated `chunk_ids`/`scores` order shuffled
    between runs of the same logical input (since `ChunkScores` carries
    no meaningful order) — assembled into a `ChunkScores` via
    `make_chunk_id`. Asserts the result matches the true maximum
    (diverging from mean/sum whenever they would differ), ties broken by
    ascending `doc_id_sort_key`, and — as the `n = 1` corollary, exercised
    with documents restricted to exactly one chunk each — that the
    result is identical to ranking those documents directly by their own
    single score with the same tie-break. **Validates: Requirements 2.5,
    5.1, 5.2, 5.5, 10.2.**
  - **Property 3: Every Chunker produces a well-formed, fully-covering,
    non-duplicating chunk corpus.** A strategy generating a small corpus
    (1–8 documents via `st.dictionaries`), run against `WholeDocumentChunker`
    (no tokenizer needed): asserts every produced `chunk_id` round-trips
    through `parse_chunk_id` to its exact `(doc_id, position)`; all
    produced `chunk_id`s are pairwise distinct; no `chunk_id` equals any
    input `doc_id`; and the set of `doc_id`s recovered by parsing every
    produced `chunk_id` equals exactly the input corpus's `doc_id` set.
    The zero-chunk edge case is exercised once (not under `@given`) with
    a hand-written stub `Chunker` returning `[]` for one designated
    `doc_id`, asserting `ChunkingError` is raised naming that document.
    **Validates: Requirements 2.3, 2.4, 2.6.**
  - Every `@given(...)` test uses `@settings(max_examples=100)` (or
    Hypothesis's default, which already meets this) and is tagged with a
    comment referencing its Property number, per the design's Testing
    Strategy.
  - Done check: `pytest tests/test_chunking.py -v` reports all Property
    1/2/3 tests passed. No network call is made by any test in this task
    (Property 3's `WholeDocumentChunker` run needs no tokenizer).
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 5.1, 5.2, 5.5, 5.6, 10.2_

- [x] 9. Extend `tests/test_chunking.py` — Properties 4, 5 (tokenizer-gated)
  - Add a `session`-scoped pytest fixture that loads the all-MiniLM-L6-v2
    tokenizer once from `data/hf_cache`, itself gated by a
    `Local_Cache_Availability` check (does
    `data/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2` exist
    under the repo's `data/` directory?) via `pytest.mark.skipif` — on a
    clean checkout with no cached tokenizer, every test depending on this
    fixture is skipped, not failed.
  - **Property 4: `Fixed_Window_Chunker` produces a token-budgeted,
    fully-covering, order-preserving, text-faithful partition.**
    `hypothesis.strategies.text()` composed into multi-paragraph document
    text (e.g. `st.lists(st.text(min_size=1), min_size=1,
    max_size=50).map(" ".join)`, spanning both under- and
    over-`window_size` lengths), run against `FixedWindowChunker`
    constructed with the session-scoped tokenizer and `window_size=200,
    stride=50`. Asserts, all at once: every Chunk's independently
    re-tokenized length `<= 200`; a document at or under 200 tokens
    yields exactly one Chunk; the union of Chunks' token spans covers
    every token of the source at least once; Chunks are produced with
    non-decreasing start offsets; and every Chunk's text is an exact
    substring of the source. **Validates: Requirements 3.3, 3.4, 3.5,
    3.6, 3.7.**
  - **Property 5: `Sentence_Window_Chunker` produces a token-budgeted
    partition that covers every sentence exactly once.** A strategy
    generating documents with a varying number of sentences of varying
    length, plus a dedicated strategy for a deliberately oversized single
    sentence (`st.text(min_size=2000)` with no embedded sentence-boundary
    punctuation), run against `SentenceWindowChunker` constructed with the
    same tokenizer and `sentences_per_chunk=3, max_chunk_tokens=256`.
    Asserts, all at once: every multi-sentence Chunk's token length `<=
    256`; an oversized single sentence is split into consecutive,
    non-overlapping, `<= 256`-token pieces covering its own token
    sequence exactly once; the concatenation of all Chunks covers every
    sentence of the source exactly once (or, for a token-split sentence,
    its full token sequence exactly once); and a document with `<= 3`
    sentences under 256 tokens yields exactly one Chunk. **Validates:
    Requirements 4.3, 4.4, 4.5, 4.6, 4.7.**
  - Done check: `pytest tests/test_chunking.py -v` reports all tests
    passed when `data/hf_cache`'s all-MiniLM-L6-v2 snapshot is present
    (it already is, from session 1), and
    `pytest tests/test_chunking.py -v -k "Property4 or Property5"`
    reports `skipped` (not `failed`) when run against a copy of the repo
    with `data/hf_cache` renamed/absent — confirming the skip-gate
    actually gates, not just exists.
  - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 10. Update `src/retrievers/bm25_retriever.py` to the Full_Chunk_Depth streaming contract
  - Remove `retrieve_all`'s `top_k` parameter, the
    `effective_top_k = min(top_k, len(self._doc_ids))` line, the
    `[:effective_top_k]` slice, and the `(-score,
    doc_id_sort_key(doc_id))` sort entirely — no chunk-level order is
    produced or needed (Requirement 5.7, 6.1, 6.4).
  - Change `retrieve_all` into a generator: fix
    `chunk_id_index = tuple(self._doc_ids)` once at the top (the same
    tuple object reused, unmodified, across every yield); for each query,
    tokenize, call `self._bm25.get_scores(tokenized_query)` (already a
    numpy array aligned to `chunk_id_index`, already scoring every
    indexed chunk), and `yield qid, ChunkScores(chunk_ids=chunk_id_index,
    scores=scores)`; accumulate `total_latency` across the loop and set
    `self.last_query_latency = total_latency` only after the loop
    completes (i.e. once the generator is exhausted).
  - Done check: a script that builds a `BM25Retriever` over a 4-chunk
    in-memory corpus, calls `build_index` once, then fully iterates
    `retrieve_all(queries)`'s returned generator; asserts every yielded
    `ChunkScores.chunk_ids` is a set-equal (not necessarily order-equal)
    match to the indexed chunk IDs, that no chunk is missing or
    duplicated, that the exact same `chunk_ids` tuple object (`is`, not
    `==`) is yielded for every query, and that `self.last_query_latency`
    is a non-negative float only after the generator is exhausted (not
    before). Prints `ok` and exits 0. No network call.
  - _Requirements: 5.7, 6.1, 6.3, 6.4_

- [x] 11. Update `src/retrievers/dense_retriever.py` to the same streaming contract
  - Apply the identical `top_k`/slice/sort removal described in Task 10.
  - Change `retrieve_all` into a generator: batch-encode all query texts
    once via `self._model.encode(...)` (cheap — `query_count x
    embedding_dim`, not the memory concern); then, for each query in
    turn, compute `scores = self._doc_embeddings @
    query_embeddings[row_idx]` (one query's full chunk-score vector via
    a single mat-vec — never materializing the full query x chunk
    similarity matrix) and `yield qid, ChunkScores(chunk_ids=chunk_id_index,
    scores=scores)`; time query encoding plus the per-query mat-vec loop
    together as `total_latency`, setting `self.last_query_latency` only
    after the loop completes.
  - Done check: a script that builds a `DenseRetriever` (real
    `all-MiniLM-L6-v2`, already cached under `data/hf_cache`) over a
    3-chunk in-memory corpus, calls `build_index` once, fully iterates
    `retrieve_all(queries)`'s generator, and asserts every yielded
    `ChunkScores.chunk_ids` matches the indexed chunk IDs exactly (as a
    set), every yielded `scores` array has the same length as
    `chunk_ids`, and `self.last_query_latency` is a non-negative float
    only after the generator is exhausted. Prints `ok` and exits 0.
  - _Requirements: 5.7, 6.1, 6.3, 6.4_

- [x] 12. Extend `tests/test_chunking.py` — Property 6
  - **Property 6: `BM25Retriever.retrieve_all` yields every indexed
    chunk, scored, at Full_Chunk_Depth, unordered.** A strategy
    generating a small in-memory chunk corpus (`st.dictionaries`,
    1–10 entries) and a small query set (1–5 queries), run against a
    real `BM25Retriever` (no model, no network, matching session-1's own
    precedent). Asserts every yielded `ChunkScores.chunk_ids` is a
    set-equal match to the indexed corpus's chunk IDs — none truncated,
    none duplicated, in no particular required order — and that the same
    `chunk_ids` tuple object (`is`, not `==`) is reused across every
    query's yield within one `retrieve_all` call. **Validates:
    Requirements 5.7, 6.1.**
  - Done check: `pytest tests/test_chunking.py -v` reports all tests
    passed (Properties 1–6 combined), and
    `! grep -Eq 'beir|sentence_transformers|huggingface' tests/test_chunking.py`
    ... is NOT required here (Properties 3/4/5's tokenizer fixture is an
    expected, documented exception per Task 9 — only Property 6 itself
    must avoid any model/network dependency, which the real
    `BM25Retriever` constructor already satisfies).
  - _Requirements: 5.7, 6.1_

- [x] 13. Checkpoint — ensure `tests/test_chunking.py` passes
  - Run `pytest tests/test_chunking.py -v` and confirm every Property
    1–6 test passes (or, for Properties 4/5, is skipped only if
    `data/hf_cache`'s tokenizer snapshot is genuinely absent). Ensure all
    tests pass, ask the user if questions arise, before restructuring
    `src/config.py` and `src/sweep_runner.py` around the chunking
    abstraction.

- [x] 14. Extend `src/config.py`: `chunking_strategies` schema, 3-retriever validation, cross-product derivation
  - Add `WholeDocumentChunkingConfig` (`name`), `FixedWindowChunkingConfig`
    (`name`, `window_size: int`, `stride: int`), and
    `SentenceWindowChunkingConfig` (`name`, `sentences_per_chunk: int`,
    `max_chunk_tokens: int`) frozen dataclasses, plus the
    `ChunkingStrategyConfig` union. Change `SweepConfig.chunking_strategy:
    str` to `chunking_strategies: Tuple[ChunkingStrategyConfig, ...]`.
  - Update `load_sweep_config`: read `chunking_strategies` (a list)
    instead of the old singular `chunking_strategy`; if the YAML declares
    the old singular field, raise `ConfigError` naming it explicitly
    (Requirement 7.4) rather than silently ignoring it. Validate
    `chunking_strategies` has exactly 3 entries, one each named
    `whole_document`, `fixed_window`, `sentence_window` (no duplicates, no
    omissions, no unsupported name) — any other count/value set raises
    `ConfigError` naming the invalid declaration (Requirement 7.3).
    Validate each entry's own required fields per its `name` using
    `ChunkingConfigError` for an invalid `window_size`/`stride`/
    `sentences_per_chunk`/`max_chunk_tokens` (Requirement 7.6).
  - Update the `retrievers` validation: exactly 3 entries — exactly one
    `type: bm25`, and exactly two `type: dense` entries whose `name`
    fields are the exact, case-sensitive strings `all-MiniLM-L6-v2` and
    `bge-small-en-v1.5` (both required, in either order) — with no other
    count, type combination, or `name` value accepted (Requirement 1.1,
    1.5). The `bge-small-en-v1.5` entry's `model_name` must equal
    `BAAI/bge-small-en-v1.5` (Requirement 1.2); it uses the identical
    `DenseRetrieverConfig`/`_load_dense_config` path already used for
    `all-MiniLM-L6-v2` — no new field, no query-prefix field (Requirement
    1.3).
  - The 9-run cross product (`itertools.product(config.retrievers,
    config.chunking_strategies)`) is computed inside `Sweep_Runner` at
    orchestration time (Task 16) — `SweepConfig` itself keeps the two
    length-3 tuples as data and does not materialize the cross product
    (Requirement 7.2).
  - Done check: a script that (a) loads `configs/sweep.yaml` (once Task
    15 has updated it) and asserts `len(config.chunking_strategies) ==
    3` and `len(config.retrievers) == 3`; (b) asserts a copy of the YAML
    with the old singular `chunking_strategy: whole_document` field
    instead of `chunking_strategies` raises `ConfigError`; (c) asserts a
    copy with only 2 `chunking_strategies` entries raises `ConfigError`;
    (d) asserts a copy with `fixed_window.window_size` set to a string
    raises `ChunkingConfigError`; (e) asserts a copy with only 1 dense
    retriever (missing `bge-small-en-v1.5`) raises `ConfigError`. Prints
    `ok` and exits 0. (This task's done check depends on Task 15's YAML
    update; run them together if Task 15 is not yet applied.)
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 7.1, 7.2, 7.3, 7.4, 7.6_

- [x] 15. Update `configs/sweep.yaml`: third retriever and `chunking_strategies` list
  - Replace the single `chunking_strategy: whole_document` line with a
    `chunking_strategies` list of exactly 3 entries: `whole_document`
    (name only); `fixed_window` with `window_size: 200` and `stride: 50`
    (Requirement 3.2); `sentence_window` with `sentences_per_chunk: 3`
    and `max_chunk_tokens: 256` (Requirement 4.2) — all declared as
    explicit YAML data fields, never literals hard-coded in `src/`
    (Requirement 7.6).
  - Add a third `retrievers` entry: `name: bge-small-en-v1.5`, `type:
    dense`, `model_name: BAAI/bge-small-en-v1.5`, `batch_size: 32`
    (matching the existing `all-MiniLM-L6-v2` entry's shape exactly).
  - Done check: `python -c "from src.config import load_sweep_config; from pathlib import Path; c = load_sweep_config(Path('configs/sweep.yaml')); assert len(c.chunking_strategies) == 3; assert {s.name for s in c.chunking_strategies} == {'whole_document', 'fixed_window', 'sentence_window'}; assert len(c.retrievers) == 3; assert {r.name for r in c.retrievers} == {'bm25', 'all-MiniLM-L6-v2', 'bge-small-en-v1.5'}; print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 1.1, 1.2, 3.2, 4.2, 7.1, 7.6_

- [x] 16. Restructure `src/sweep_runner.py`: chunker-factory seam and the chunk-once/retrieve-once/aggregate-once loop
  - Add `ChunkerFactory = Callable[[ChunkingStrategyConfig], Chunker]` and
    `make_default_chunker_factory(cache_folder) -> ChunkerFactory`: for
    `WholeDocumentChunkingConfig`, returns `WholeDocumentChunker()`
    immediately (no tokenizer load, ever); for `fixed_window`/
    `sentence_window`, loads the all-MiniLM-L6-v2 tokenizer via
    `load_chunking_tokenizer` at most once (memoized in a closure
    variable), only the first time either is actually requested — never
    eagerly, never for a `SweepConfig` declaring only `whole_document`.
  - Change `run_sweep`'s signature to accept `chunker_factory:
    ChunkerFactory` alongside the existing `retriever_factory`. Restructure
    the loop to an **outer loop over `config.chunking_strategies`** (3
    iterations — one `build_chunk_corpus` call each, via
    `chunker_factory(chunking_config)`, cached in a local variable for
    that iteration's duration) and an **inner loop over
    `config.retrievers`** (3 iterations each, 9 combinations total).
    `run_id = f"{retriever_config.name}__{chunking_config.name}"`.
  - Add the new per-strategy failure tier: if `build_chunk_corpus` raises
    `ChunkingError` for a chunking_config, append 4 all-`MISSING` rows for
    each of that strategy's 3 `run_id`s (12 rows total), skip that
    strategy's retriever loop entirely, and continue to the next
    strategy — the other 2 strategies' 24 rows are unaffected
    (Requirement 2.6, new failure tier described in the design's Error
    Handling table).
  - Inside the retriever loop: `retriever.build_index(chunk_corpus)` once;
    then iterate the single `retriever.retrieve_all(bundle.queries)`
    generator once, calling `aggregate_to_document_ranked_list(chunk_scores)`
    per query to build `document_ranked_lists: Dict[str, List[str]]`; read
    `retriever.last_query_latency` only after the generator is exhausted.
    Everything after that (the `ndcg_at_10`/`mrr_at_10`-once,
    `recall_at_k`-per-cutoff logic, the existing per-retriever
    `try/except` `RetrievalError`-equivalent tier, the per-query-row
    emission gate) is unchanged from session-1/significance-testing,
    operating on `document_ranked_lists` instead of a retriever's own
    direct `ranked_lists`.
  - Update `make_default_retriever_factory` (unchanged in shape) and
    `main()`: add `chunker_factory = make_default_chunker_factory(config.data_dir
    / "hf_cache")`, passed into `run_sweep` alongside `retriever_factory`.
    `write_run_config_record` needs no code change — it already serializes
    `dataclasses.asdict(config)`, which now includes the 3-entry
    `chunking_strategies` list automatically.
  - Done check: a script driving `run_sweep` with 2 `StubChunker`
    instances (from Task 17, or a minimal hand-written equivalent), 2
    `StubRetriever` instances, and cutoffs `(1, 5, 10, 20)`; asserts
    `len(rows) == 2 * 2 * 4 == 16` and that each of the 4
    (chunking_strategy, retriever) combinations' stub `build_index`/
    `retrieve_all` were each called exactly once. Prints `ok` and exits 0.
    No network call, no real retriever/chunker imported. (Task 17's own
    pytest suite is the authoritative verification of this behavior;
    this is a quick sanity check before that suite exists.)
  - _Requirements: 2.6, 5.1, 5.6, 6.1, 6.2, 6.3, 6.5, 7.2, 7.5, 8.1, 8.5_

- [x] 17. Extend `tests/test_orchestration.py`: `StubChunker` and the chunking-strategy axis (Property 7)
  - Update `StubRetriever` to the new generator-based `retrieve_all`
    contract: no `top_k` parameter; yields `(query_id, ChunkScores)` pairs
    one at a time (built from its fixed literal scores) instead of
    returning a single `(Dict[str, List[str]], float)` tuple; still
    records every `build_index`/`retrieve_all` call, still performs no
    real computation.
  - Add `StubChunker` (`strategy_name`, `fixed_chunks_by_doc: Dict[str,
    List[Chunk]]`): `chunk_document` returns the hand-specified,
    fixed list of Chunks for that `doc_id`, regardless of document
    content — never a real tokenizer, never real text splitting.
  - Build a hand-specified `In_Memory_Test_Corpus` in which at least one
    document has 2+ Chunks with **different** stub scores (so
    `aggregate_to_document_ranked_list`'s max-vs-mean/sum distinction is
    genuinely exercised, Requirement 13.3's own wording); construct a
    `SweepConfig` with 2 chunking-strategy stubs x 2 retriever stubs (4
    combinations, generalizing session-1's 2-retriever case).
  - **Property 7: The chunk-level orchestration loop indexes and
    retrieves exactly once per combination, and every cutoff's
    aggregation matches the true per-document maximum.** Assert: each of
    the 4 combinations' `build_index` and `retrieve_all` were each called
    exactly once; each row's aggregated score matches the hand-computed
    true maximum of the relevant document's stub chunk scores for that
    combination (never mean or sum); and `index_time`/`query_latency` are
    identical across all rows sharing a `run_id`, while distinct `run_id`s
    never coincidentally share a value that should legitimately differ.
    **Validates: Requirements 6.3, 6.5, 8.5, 13.2, 13.3.**
  - Done check: `pytest tests/test_orchestration.py -v` passes, reporting
    the call-count and aggregation assertions above explicitly (e.g. via
    inline `assert` statements inside the test, following this file's
    existing style). No network call, no real retriever/chunker imported
    (Requirement 13.1, 13.4 — this test is not treated as satisfying
    Requirement 12's real-corpus tests).
  - _Requirements: 6.3, 6.5, 8.5, 13.1, 13.2, 13.3, 13.4_

- [x] 18. Verify the chunking abstraction reproduces the committed whole-document baseline (Requirement 2.5)
  - Capture the exact committed baseline via `git show HEAD:results/sweep.csv`,
    written to a temp file — never relying on the working tree's
    `results/sweep.csv`, so the comparison is anchored to the version
    committed before Task 28 (the full grid run) overwrites it.
  - Construct a hand-built `SweepConfig` in a standalone script,
    bypassing `load_sweep_config`'s Task 14 validation (which now
    mandates exactly 3 `chunking_strategies` entries) — the same
    direct-construction approach `tests/test_orchestration.py`'s
    stub-based tests already use — declaring `chunking_strategies =
    (WholeDocumentChunkingConfig(name="whole_document"),)` (only this one
    entry) and the same 2 retrievers (`bm25` with `k1=1.5, b=0.75,
    tokenizer=regex_word, lowercase=True, stopwords=none, stemming=none`,
    and `all-MiniLM-L6-v2`) whose settings match the committed
    `results/run_config.json`'s `sweep_config` exactly. Point
    `output_path` at a `mktemp`-created temp file, never at the real
    `results/sweep.csv`.
  - Run this config through the real, restructured `run_sweep` (Task 16),
    using the real `load_scifact`-loaded corpus, the real
    `make_default_retriever_factory`, and `WholeDocumentChunker()`
    directly (no factory needed, since only one strategy is declared) —
    exercising the actual production chunk-once/retrieve-once/aggregate-once
    path, not a stub corpus.
  - Compare every `recall_at_k` (k in 1, 5, 10, 20), `ndcg_at_10`, and
    `mrr_at_10` value for `run_id in {bm25__whole_document,
    all-MiniLM-L6-v2__whole_document}` against the corresponding row in
    the git-HEAD-captured baseline, within a tolerance of `1e-9`. Exclude
    `index_time` and `query_latency` from the comparison entirely (timing
    columns vary run to run, per `tech.md`, and are never compared across
    runs).
  - This is the first point in this spec's task sequence where the
    chunking refactor is checked against a real, previously-published
    number rather than a hand-built stub corpus. It isolates the
    chunking abstraction itself from the two new strategies: if it
    fails, the abstraction broke the already-published whole-document
    baseline, and nothing produced by any later task (the full grid, the
    significance re-run, the token-length report) can be trusted until
    this is fixed. Must pass before Task 28 runs.
  - Done check: a script performing the steps above end to end, which
    must additionally assert, before it is allowed to report success:
    (1) the git-HEAD baseline, captured via `git show
    HEAD:results/sweep.csv` and filtered to `run_id in
    {bm25__whole_document, all-MiniLM-L6-v2__whole_document}`, contains
    exactly 8 rows (2 retrievers x 4 k values); (2) the newly produced
    temp sweep CSV, filtered the same way, also contains exactly 8 rows,
    and its set of `(run_id, k)` pairs is exactly equal — as sets, not
    merely the same count — to the baseline's set of `(run_id, k)`
    pairs; and (3) exactly 24 individual value comparisons are performed
    in total (8 rows x 3 metrics: `recall_at_k`, `ndcg_at_10`,
    `mrr_at_10`), with the script explicitly counting how many
    comparisons it actually performs and failing (non-zero exit, no
    `ok` printed) if that count is anything other than exactly 24 — so a
    mismatched or missing `run_id`/`k` pair that would otherwise
    silently reduce the comparison count below 24 causes a hard failure
    instead of a vacuous pass. Only after all three of these checks hold
    does the script proceed to assert every compared value matches
    within `1e-9`, printing `ok` with exit code 0 only if every
    comparison passes, and deleting its temp output file on exit
    regardless of outcome.
  - _Requirements: 2.5_

- [x] 19. Measure fixed_window runtime and chunk count, and extrapolate to the full 9-run grid before committing to Task 28
  - Load the real, already-cached SciFact corpus (`load_scifact`) and
    chunk it once under `fixed_window` (`window_size=200, stride=50`,
    Task 5's `FixedWindowChunker`, using the already-cached
    all-MiniLM-L6-v2 tokenizer) — record the resulting chunk count
    `C_fixed = len(chunk_corpus)`, read from the actual built chunk
    corpus, never assumed or hard-coded.
  - Build one dense index over that chunk corpus using `DenseRetriever`
    with `all-MiniLM-L6-v2` (already cached from session 1 — no new
    network download in this task) and record the wall-clock
    `build_index` time.
  - Run `retrieve_all` for every real test query `load_scifact` loads
    (the full query set, never a subset) and record the total wall-clock
    time to fully exhaust the generator.
  - Extrapolate an approximate total wall-clock time for the full 9-run
    grid: use the already-committed `results/sweep.csv`'s real
    `index_time`/`query_latency` for the 2 `whole_document` combinations
    (`bm25`, `all-MiniLM-L6-v2`) as ground truth; use this task's own
    measured `fixed_window` x `all-MiniLM-L6-v2` numbers directly; and
    approximate every remaining cell (`sentence_window` for any
    retriever, and every `bge-small-en-v1.5` cell) by analogy — scaling
    `bm25`'s time roughly by chunk-count ratio, and treating
    `bge-small-en-v1.5` as comparable to `all-MiniLM-L6-v2` at the same
    chunk count, since neither `sentence_window`'s real chunk count nor
    any `bge-small-en-v1.5` timing is measured before this task runs.
    State this extrapolation's approximate nature explicitly in the
    printed output — it is a planning estimate to inform when to start
    Task 28, not a committed number, and carries no
    `docs/numeric_traceability.csv` obligation.
  - Print, in the task's own output: the measured `C_fixed`, the
    measured `fixed_window` x `all-MiniLM-L6-v2` `index_time` and total
    query-retrieval time, and the extrapolated total wall-clock estimate
    for all 9 runs (e.g. in minutes) — so a human can decide whether to
    run Task 28 now or schedule it for later. The full grid is 3
    retrievers x 3 strategies over a corpus that may be several times
    larger in chunks than in documents; this task exists because knowing
    whether the full run is on the order of minutes or hours changes when
    to start it, and it is cheap to learn now rather than discovering it
    mid-way through Task 28.
  - Done check: a script performing the measurement and extrapolation
    above, printing all four figures (`C_fixed`, measured index time,
    measured query time, extrapolated total) in a human-readable form,
    and exiting 0. There is no numeric pass/fail assertion in this
    task's done check — the deliverable is the printed measurement
    itself.
  - Do not add a `_Requirements:` line to this task (it is an
    operational, decision-support step, not derived from a specific
    acceptance criterion — the same convention already used for the
    checkpoint tasks, which also omit a `_Requirements:` line).

- [x] 20. Extend `src/significance_config.py`: pinned `reference_chunking_strategy`
  - Add `reference_chunking_strategy: str` to `SignificanceConfig`, as a
    **required** field with no default (unlike `run_config_path`) — an
    implicit default here would reintroduce exactly the "silently pick a
    strategy" risk Requirement 9.3 rules out.
  - Update `load_significance_config` to require
    `reference_chunking_strategy` via `_require_field`, raising
    `BootstrapConfigError` naming it if absent.
  - Done check: a script that calls `load_significance_config` against a
    copy of `configs/significance.yaml` (before Task 21 updates the real
    file) with `reference_chunking_strategy: whole_document` added, and
    asserts `config.reference_chunking_strategy == "whole_document"`;
    then asserts a copy omitting that field raises `BootstrapConfigError`.
    Prints `ok` and exits 0.
  - _Requirements: 9.3_

- [x] 21. Update `configs/significance.yaml`: pin `reference_chunking_strategy`
  - Add `reference_chunking_strategy: whole_document` alongside the
    existing `reference_retriever: bm25` field (Requirement 9.2) — the
    explicit pin to `bm25__whole_document`, matching the already-published
    README/SPEC baseline.
  - Done check: `python -c "from src.significance_config import load_significance_config; from pathlib import Path; c = load_significance_config(Path('configs/significance.yaml')); assert c.reference_retriever == 'bm25'; assert c.reference_chunking_strategy == 'whole_document'; print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 9.2_

- [x] 22. Extend `src/significance.py`: explicit `reference_run_id` pin, replacing the sorted-first-bm25 rule
  - Replace `_find_reference_run_id(frame, reference_retriever)` with
    `_find_reference_run_id(frame, reference_run_id)`: performs an EXACT
    match of `reference_run_id` against `frame['run_id'].unique()` and
    nothing else — no filtering by retriever name, no sorting, no "take
    the first" rule (Requirement 9.3, since with 3 chunking strategies
    present, an implicit rule could silently select `bm25__fixed_window`
    instead of `bm25__whole_document`). Raises `MissingReferenceRunError`
    naming `reference_run_id` exactly if it is not present in
    `frame['run_id']` (Requirement 9.4).
  - Update `main()`'s call site to compute
    `reference_run_id = f"{config.reference_retriever}__{config.reference_chunking_strategy}"`
    before calling `_find_reference_run_id`. Do NOT modify
    `_run_comparisons`, `_apply_holm_bonferroni`, `paired_bootstrap`,
    `permutation_test`, or `holm_bonferroni` — `_run_comparisons` already
    derives `comparison_run_ids = sorted(rid for rid in run_frames if rid
    != reference_run_id)`, which naturally yields 8 entries once
    `results/per_query.csv` contains 9 run_ids, with no further code
    change needed (Requirement 9.1, 9.5, 9.6, 9.7).
  - Done check: a script (in a `mktemp -d` temp dir, never touching the
    real `results/` directory, mirroring the significance-testing spec's
    own Task 7 done-check pattern) that fabricates a tiny `per_query.csv`
    with 3 distinct `run_id`s (`bm25__whole_document`,
    `bm25__fixed_window`, `all-MiniLM-L6-v2__whole_document`) and asserts:
    calling `_find_reference_run_id(frame, "bm25__whole_document")`
    returns that exact string (never `bm25__fixed_window`, even though it
    would sort first alphabetically); and calling it with
    `"bm25__sentence_window"` (absent from the fabricated frame) raises
    `MissingReferenceRunError` naming that exact string. Prints `ok` and
    exits 0.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [x] 23. Update `src/retrieval_replay.py`: route through `WholeDocumentChunker` + trivial `Max_Aggregation`
  - Update `build_frozen_retriever`: after loading the real corpus,
    construct a `WholeDocumentChunker()`, call
    `build_chunk_corpus(chunker, bundle.corpus)`, and pass the resulting
    chunk corpus (not `bundle.corpus` directly) into
    `retriever.build_index(...)` (Requirement 10.1) — exactly one Chunk
    per document, containing that document's unmodified content, matching
    the Sweep_Runner's own no-op wrapping of session-1's whole-document
    behavior.
  - Update `replay_retrieval`: call `retriever.retrieve_all(subset_queries)`
    (no `top_k` argument — the parameter no longer exists), consume the
    returned generator, apply `aggregate_to_document_ranked_list` to each
    query's `ChunkScores` to reduce it to a `Document_Ranked_List`, THEN
    slice to `replay_top_k` (the slicing `retrieve_all`'s own `top_k`
    parameter used to perform now happens here, after aggregation).
    Everything else (pre-filtering `queries` to `subset_query_ids` before
    the single call, mapping ranked document IDs to `Retrieved_Context`
    via `format_document_text`, the `RetrievalReplayError` wrapping) is
    unchanged.
  - Because `whole_document` chunking followed by 1-chunk
    `Max_Aggregation` is the identity transform (Property 2's `n = 1`
    corollary, already proven in Task 8), the updated path produces, for
    every Generation_Subset query, a `Retrieved_Context` numerically
    identical to what the pre-update code path produced (Requirement
    10.2) — this task's done check verifies the code routes correctly;
    Task 31 verifies the real-corpus, real-model claim end to end.
  - Do NOT change `configs/groundedness.yaml`, `src/groundedness_runner.py`,
    `src/generator_model.py`, `src/judge_model.py`, or
    `src/quarantine_rule.py` (Requirement 10.6). The `run_id`
    `bm25__whole_document` continues to resolve correctly against the
    updated `configs/sweep.yaml` schema, since `whole_document` remains a
    valid declared Chunking_Strategy value (Requirement 10.5).
  - Done check: a two-part script, mirroring `retrieval_replay.py`'s own
    existing done-check style. Part (a), no network/corpus touch:
    `load_frozen_retriever_config` still returns a `BM25RetrieverConfig`
    for `"bm25__whole_document"`. Part (b), against the already-cached
    SciFact corpus: `build_frozen_retriever` builds the BM25 index once
    (now via the chunk corpus internally), then `replay_retrieval` is
    called once with a 2-query subset at `replay_top_k=3`, asserting the
    returned dict has exactly 2 keys and every value is a list of at most
    3 document-text strings — identical in shape to the pre-update
    behavior. Prints `ok` and exits 0.
  - _Requirements: 10.1, 10.2, 10.5, 10.6_

- [x] 24. Extend `src/token_length_analysis.py`: 6-cell report (3 strategies x 2 dense models)
  - Add the frozen `TokenLengthCell` dataclass (`chunking_strategy`,
    `model_name`, `max_sequence_length`, `num_documents_total` — actually
    "num Chunks total" for that strategy, `num_documents_exceeding`,
    `fraction_exceeding`).
  - Implement `compute_cell(chunker, tokenizer, corpus) ->
    TokenLengthCell`: builds a chunk corpus via `build_chunk_corpus(chunker,
    corpus)` (Requirement 11.4 — including the `whole_document` cells,
    which stay numerically consistent with the pre-existing single-model
    measurement because `whole_document` chunking is a no-op); counts
    tokens of every Chunk's text via the cell's own tokenizer; reads
    `max_sequence_length = int(tokenizer.model_max_length)` at run time
    (Requirement 11.2 — never a hard-coded 256 or 512); calls the
    existing, unchanged `compute_exceedance_stats`.
  - Update `main()`: loop over the 6 `(Chunking_Strategy, dense model)`
    cells (`whole_document`/`fixed_window`/`sentence_window` x
    `all-MiniLM-L6-v2`/`bge-small-en-v1.5`), loading each dense model's
    tokenizer via the existing `load_tokenizer_offline` (Requirement
    11.5 — no network call under any circumstance), and loading the
    all-MiniLM-L6-v2 windowing tokenizer used by `fixed_window`/
    `sentence_window` via that same offline-forced path too (not
    `load_chunking_tokenizer`, since this analysis must never make a
    network call). On any single cell's failure (e.g. a tokenizer fails
    to load), halt the whole analysis before writing any partial report
    — no per-cell recovery tier (unlike the sweep). Write all 6 cells to
    `results/token_length_report.json` under a `"cells"` key, extending
    (not replacing) the existing top-level file schema (Requirement
    11.3).
  - Done check: a script asserting `compute_cell` against a small
    in-memory corpus and `WholeDocumentChunker` plus a fake tokenizer
    stub (or the real cached all-MiniLM-L6-v2 tokenizer) returns a
    `TokenLengthCell` whose `num_documents_total` equals the corpus size
    and whose `max_sequence_length` matches `tokenizer.model_max_length`
    exactly (not a hard-coded literal). Prints `ok` and exits 0. No
    network call (uses the already-cached tokenizer or a fixture).
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 25. Extend `tests/test_token_length_analysis.py`: whole-document regression check
  - Add one test asserting that a `compute_cell` call for the
    `whole_document` x `all-MiniLM-L6-v2` cell, against a small
    hand-built in-memory corpus and the real cached tokenizer, produces a
    `fraction_exceeding` consistent with `compute_exceedance_stats`'s
    already-tested boundary behavior (reusing the existing, unmodified
    fixtures from the repo-writeup spec's own test module where
    possible) — a targeted regression check, not a generated property.
    `compute_exceedance_stats` itself needs no new tests (unchanged).
  - Done check: `pytest tests/test_token_length_analysis.py -v` reports
    all tests passed (the existing repo-writeup tests plus this new
    regression check), and
    `! grep -Eq 'sentence_transformers|corpus_loader|sweep_runner' tests/test_token_length_analysis.py && echo ok`
    prints `ok`.
  - _Requirements: 11.4_

- [x] 26. Write `tests/test_data_layer.py`: `Data_Layer_Tests` and `Real_Corpus_End_To_End_Tests`
  - Implement `_local_cache_available() -> bool` (Requirement 12.3):
    checks that `data/scifact` and every `data/hf_cache/models--*`
    directory this spec's grid needs (`all-MiniLM-L6-v2`,
    `bge-small-en-v1.5`; `bm25` needs none) are already present, and set
    `pytestmark = pytest.mark.skipif(not _local_cache_available(),
    reason="requires a local BEIR SciFact + model weight cache; skipped
    on a clean checkout")` at module scope, so every test in the module
    is skipped, not failed, on a clean checkout (Requirement 12.4).
  - Implement `test_load_scifact_against_real_cached_corpus`: calls the
    real `load_scifact(Path("data"))` against the real cached corpus,
    asserting non-zero `num_documents`/`num_queries`/`num_qrel_pairs`
    counts and referential integrity (Requirement 12.1) — exercising the
    real `Corpus_Loader`, not a stub.
  - Implement `test_sweep_runner_end_to_end_one_combination_against_real_corpus`:
    runs the Sweep_Runner (via `run_sweep`, with the production
    `retriever_factory`/`chunker_factory`, not stubs) against the real
    BEIR SciFact corpus for one full retriever x Chunking_Strategy
    combination — `bm25` x `whole_document` (the cheapest combination: no
    dense model load, no tokenizer download, fastest to run locally) —
    asserting the resulting rows have the correct columns and every
    metric value is either a float in `[0.0, 1.0]` or `MISSING`
    (Requirement 12.2).
  - These tests are additive to, and do not replace or modify,
    `tests/test_metrics.py`, `tests/test_orchestration.py`,
    `tests/test_significance.py`, `tests/test_claim_segmenter.py`,
    `tests/test_quarantine_rule.py`, `tests/test_token_length_analysis.py`,
    or `tests/test_verify_writeup_numbers.py` (Requirement 12.6).
  - Done check: `pytest tests/test_data_layer.py -v` reports the two
    tests **skipped** (not failed) if run against a checkout where
    `data/scifact`/`data/hf_cache` are absent, and reports both tests
    **passed** when run locally where the cache from session 1 (and,
    once Task 28 runs, `bge-small-en-v1.5`) already exists.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [x] 27. Checkpoint — ensure the full `pytest` suite passes
  - Run `pytest -v` (the entire suite: `test_chunking.py`,
    `test_orchestration.py`, `test_metrics.py`, `test_significance.py`,
    `test_claim_segmenter.py`, `test_quarantine_rule.py`,
    `test_token_length_analysis.py`, `test_verify_writeup_numbers.py`,
    `test_data_layer.py`) and confirm everything passes or is
    legitimately skipped. This must be green before any task below,
    which are the first tasks in this spec's sequence to touch the
    network. Ensure all tests pass, ask the user if questions arise.

- [x] 28. Run the real sweep end to end and commit the 36-row `results/sweep.csv` / `9 * Q`-row `results/per_query.csv`
  - Run `python -m src.sweep_runner --config configs/sweep.yaml`. This is
    the first task in this spec's sequence that touches the network: a
    one-time `BAAI/bge-small-en-v1.5` weight download to
    `data/hf_cache` (`bm25` and `all-MiniLM-L6-v2` are already cached
    from session 1).
  - Done check: a script that loads `results/sweep.csv` with `pandas` and
    asserts exactly 36 rows, `{run_id}` has exactly 9 distinct values,
    each of the form `{retriever}__{chunking_strategy}` for one of the 3
    retrievers x 3 strategies, and every `k` value per `run_id` is
    exactly `{1, 5, 10, 20}`; a second assertion loads
    `results/per_query.csv` and asserts its row count equals `9 *
    num_queries_scored`, where `num_queries_scored` is read from
    `results/sweep.csv`'s own `num_queries_scored` column (never
    hard-coded), consistent with Requirement 8.3's "never hard-code Q"
    rule. Prints `ok` and exits 0.
  - _Requirements: 1.1, 1.2, 1.3, 6.1, 6.2, 6.3, 6.5, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 29. Run the real significance re-run and commit the 48-row `results/significance.csv`
  - Run `python -m src.significance --config configs/significance.yaml`
    against the `results/per_query.csv` Task 28 just produced. Makes no
    network call.
  - Done check: a script that loads `results/significance.csv` and
    asserts exactly 48 rows (8 comparisons x 6 metrics), that every
    `run_id` other than `bm25__whole_document` appears in exactly 6 rows,
    that `bm25__whole_document` never appears as a `run_id` value (it is
    the Reference_Run, never compared to itself), and that every
    computed comparison is retained regardless of its result value (no
    row filtered for being unflattering, Requirement 9.8). Prints `ok`
    and exits 0.
  - _Requirements: 9.1, 9.2, 9.4, 9.5, 9.6, 9.7, 9.8_

- [x] 30. Run the real Token_Length_Analysis and commit the extended 6-cell `results/token_length_report.json`
  - Run `python -m src.token_length_analysis` (using the already-cached
    corpus and tokenizers under `data/` — no new download expected, since
    Task 28 already downloaded `bge-small-en-v1.5`'s weights, and its
    tokenizer ships alongside them).
  - Done check: a script that loads `results/token_length_report.json`
    and asserts its `"cells"` list has exactly 6 entries, one for each of
    the 3 Chunking_Strategy entries x 2 dense-model names; that the
    `whole_document` x `all-MiniLM-L6-v2` cell's `fraction_exceeding`
    matches the pre-existing single-cell measurement within `1e-9`
    (Requirement 11.4's regression check, restated against the real
    artifact); and that every cell's `max_sequence_length` was read from
    that cell's own tokenizer (256 for `all-MiniLM-L6-v2`, 512 for
    `bge-small-en-v1.5`, cross-checked against each model's own published
    `model_max_length`). Prints `ok` and exits 0.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 31. Verify `retrieval_replay.py`'s output-preserving update via rerun-and-diff of the groundedness gate
  - Before re-running: copy the five existing groundedness-gate artifacts
    (`results/groundedness.csv`, `results/generated_answers.csv`,
    `results/hand_checked_sample.csv`, `results/hand_checked_joined.csv`,
    `results/hand_checked_sample_context.md`) to a temp location for
    comparison.
  - Run `python -m src.groundedness_runner --config
    configs/groundedness.yaml` (identical, unmodified
    `configs/groundedness.yaml`, including `replayed_run_id:
    bm25__whole_document`) against the post-update `src/retrieval_replay.py`
    and the just-regenerated `results/sweep.csv`/`results/per_query.csv`
    from Task 28.
  - Diff the newly-produced files against the temp-saved pre-update
    copies: byte-for-byte for the four CSV/Markdown files, and within
    `1e-9` for any floating-point CSV column (Requirement 10.3) — the
    documented manual rerun-and-diff procedure this spec's design commits
    to as Requirement 10.4's explicit, automatable acceptance check
    (Property 2, already covered by Task 8's Hypothesis test, is the
    other half of that acceptance check).
  - Restore the original committed artifacts if the rerun produced any
    difference beyond the stated tolerance (a genuine regression would
    mean this task is not done, not that the committed artifacts should
    be silently replaced).
  - Done check: a script that diffs each of the five files
    old-vs-regenerated: `cmp -s` (or a byte-for-byte Python comparison)
    for the Markdown/text-shaped files, and a `pandas`-based per-column
    numeric comparison at `abs=1e-9` for the CSV files' float columns and
    exact string equality for their non-float columns. Prints `ok` and
    exits 0 only if every file matches within tolerance.
  - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 32. Author the `README.md`/`SPEC.md`/`docs/numeric_traceability.csv` updates
  - Update `README.md`'s headline finding, results table (now covering 3
    retrievers x 3 Chunking_Strategy entries, with the `whole_document`
    slice called out explicitly as the slice being reported for the
    headline comparison), and "Reproducing this sweep" section, derived
    from the real `results/sweep.csv` (36 rows) and
    `results/significance.csv` (8 comparisons) Tasks 28/29 produced.
  - Update `SPEC.md`'s "Design summary" section to describe the 3x4x3
    grid and 9 runs (replacing the "two retrievers... a single chunking
    strategy" sentence); add a new "Max_Aggregation convention"
    subsection recording the max-over-mean/sum justification (Requirement
    5.4) and the chunk identifier scheme; and state the pinned
    Reference_Run (`bm25__whole_document`) replacing the
    implicit-selection description (Requirement 14.3).
  - Extend SPEC.md's "Threats to validity" section: add the
    `bge-small-en-v1.5` query-prefix-omission limitation (Requirement
    1.4), declared as an accepted limitation expected to reduce
    `bge-small-en-v1.5`'s measured nDCG@10 without asserting a specific
    numeric magnitude; rewrite the existing token-length-truncation
    threat to describe the 6-cell measurement (Requirement 11) instead of
    the prior single-model, whole-document-only number; retain "Sparse
    qrels," "BM25 preprocessing sensitivity," and "Single-corpus
    generalization" verbatim, since nothing about this spec changes their
    content (Requirement 14.4). Do not alter the "What this does not
    claim" bullet list's chunking/retriever generalization caveats beyond
    what is necessary to reflect that all 3 Chunking_Strategy entries and
    all 3 retrievers are now covered (Requirement 14.5).
  - For every new number introduced above, add one corresponding row to
    `docs/numeric_traceability.csv` in the same edit (Requirement 14.2),
    following the existing `claim_id,document,location,stated_value,
    stated_precision,source_artifact,source_fields,computation` schema.
  - Done check:
    `python -c "text = open('README.md', encoding='utf-8').read(); assert 'bge-small-en-v1.5' in text; print('ok')"`
    prints `ok`; a similar check confirms `SPEC.md` contains
    `"Max_Aggregation"` and `"bm25__whole_document"`; and
    `python -c "import pandas as pd; df = pd.read_csv('docs/numeric_traceability.csv'); assert (df['document'] == 'README.md').sum() > 0 and (df['document'] == 'SPEC.md').sum() > 0; print('ok')"`
    prints `ok`, confirming ledger rows were added alongside the new
    prose.
  - _Requirements: 1.4, 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 33. Run the Verification_Pass for real and fix any mismatch
  - Run `python -m src.verify_writeup_numbers --repo-root .` against the
    fully updated `docs/numeric_traceability.csv` (Task 32) and the real
    committed `results/sweep.csv`, `results/per_query.csv`,
    `results/significance.csv`, `results/run_config.json`, and
    `results/token_length_report.json` (Tasks 28–30).
  - For any `MISMATCH`, correct the Numeric_Claim's text in
    `README.md`/`SPEC.md`, or correct the ledger row's `source_fields`/
    `computation`/`stated_precision` — never by editing any of the cited
    artifacts. Re-run until every row reports `MATCH`.
  - Done check: `python -m src.verify_writeup_numbers --repo-root .`
    exits 0 (verify with `echo $?` immediately after, in Git Bash), and
    its printed summary reports zero `MISMATCH` lines across every
    ledgered row, including every row added in Task 32.
  - _Requirements: 14.2_

## Notes

- Tasks 4–7 all write to `src/chunking.py`; each is scheduled in its own
  wave in the dependency graph below to avoid a same-file conflict, even
  where the underlying logic could otherwise proceed in parallel.
- Task 12 extends the same `tests/test_chunking.py` file Tasks 8/9
  already wrote, so it is scheduled after both, once `BM25Retriever`
  (Task 10) exists.
- Tasks 18 and 19, despite being numbered immediately after Task 17, are
  scheduled in the dependency graph to execute right before Task 28 (the
  real full-grid run), not immediately after Task 17 — Task 18 is a hard
  gate ("must pass before Task 28 runs"), while Task 19 is an
  informational measurement with no pass/fail assertion.
- Tasks 20/22 (significance) and Task 23 (retrieval_replay) and Task 24
  (token_length_analysis) are independent siblings — none of the three
  depends on the others, or on `src/sweep_runner.py`'s own restructuring
  (Task 16) — so they proceed in parallel once `src/chunking.py` and
  `src/retrievers/base.py` are finished.
- Tasks 28–31 are real-artifact-producing / real-corpus manual steps, not
  pytest tasks, matching this repo's established pattern (session-1,
  significance-testing, and repo-writeup all defer real-corpus/real-model
  runs and manual reconciliation to non-automated steps). Task 28 is the
  only task in this spec that downloads new model weights
  (`bge-small-en-v1.5`); every other network-touching task reuses the
  cache Task 28 populates.
- No task in this spec creates `ANALYSIS.md`, adds a fourth retriever, a
  fourth chunking strategy, hybrid/fusion retrieval, reranking, query
  expansion, or an ANN index — all out of scope per
  `docs/PROJECT_BRIEF.md` and `.kiro/steering/scope-guard.md`, and none
  of Requirements 1–14 calls for them.
- `src/metrics.py`, `src/report.py`, `src/per_query_report.py`,
  `src/corpus_loader.py`, `configs/groundedness.yaml`,
  `src/groundedness_runner.py`, `src/generator_model.py`,
  `src/judge_model.py`, and `src/quarantine_rule.py` are never modified by
  any task in this spec.
- All done checks are Git Bash / POSIX shell (`grep`, `echo $?`, `cmp`,
  `mktemp -d`) or Python one-liners — no PowerShell.
- Every task references specific acceptance criteria for traceability;
  each builds on the prior ones, and no task is a giant
  "implement everything" step.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2", "3"] },
    { "id": 1, "tasks": ["4", "10", "11", "14", "20"] },
    { "id": 2, "tasks": ["5", "15", "21", "22"] },
    { "id": 3, "tasks": ["6"] },
    { "id": 4, "tasks": ["7"] },
    { "id": 5, "tasks": ["8", "16", "23", "24"] },
    { "id": 6, "tasks": ["9", "17", "25", "26"] },
    { "id": 7, "tasks": ["12"] },
    { "id": 8, "tasks": ["18", "19"] },
    { "id": 9, "tasks": ["28"] },
    { "id": 10, "tasks": ["29", "30", "31"] },
    { "id": 11, "tasks": ["32"] },
    { "id": 12, "tasks": ["33"] }
  ]
}
```
