# Requirements Document

## Introduction

This spec completes the retriever x top-k x chunking grid declared in
`docs/PROJECT_BRIEF.md` and removes the whole-document truncation
confound `results/token_length_report.json` already measured:
**71.02%** of SciFact documents exceed `all-MiniLM-L6-v2`'s 256-token
limit under whole-document chunking. It is session-2 chunking/grid
work, built directly on top of the already-shipped
`session-1-baseline-sweep` and `significance-testing` specs, and it
touches the read-only retrieval-replay path the `groundedness-gate`
spec depends on.

Concretely, this spec adds:

1. A third retriever, `BAAI/bge-small-en-v1.5`, declared purely as a
   third `configs/sweep.yaml` entry using the existing `Dense_Retriever`
   implementation — no new retriever class.
2. Two new chunking strategies, `fixed_window` and `sentence_window`,
   alongside the existing `whole_document` strategy, each declared as
   data with numeric parameters fixed before any result exists.
3. A chunk-to-document score aggregation rule (maximum over a
   document's chunks) applied uniformly across all three retrievers and
   all three chunking strategies.
4. A `configs/sweep.yaml` schema change from a single
   `chunking_strategy` field to a `chunking_strategies` list, crossed
   with the (now three-entry) `retrievers` list to produce the full
   9-run grid.
5. A re-run of the significance analysis over the resulting 8-member
   Comparison_Family, with the Reference_Run pinned explicitly rather
   than inferred.
6. An output-preserving internal update to `src/retrieval_replay.py` so
   the groundedness gate's frozen retrieval path keeps working under
   the new chunking-aware `build_index` contract.
7. Token-length exceedance receipts extended from one model x one
   chunking strategy to two dense models x three chunking strategies.
8. Data-layer tests and end-to-end tests over the real corpus — both
   named as session-2 work in `scope-guard.md` and still outstanding —
   plus an extended stub-based orchestration test that adds a
   chunking-strategy axis and a max-aggregation assertion to session-1's
   call-counting property.
9. Secondary, repo-consistency updates to `README.md`, `SPEC.md`, and
   `docs/numeric_traceability.csv` so every new number introduced by the
   full grid and the re-run significance analysis has a receipt.

**This spec explicitly restates and extends two properties established
by prior specs, rather than introducing new ones from scratch:**

- Session-1's Requirement 5 property, "index once, retrieve once, slice
  four ways," is restated in chunk terms (Requirement 6 below): each
  retriever x chunking-strategy run still builds exactly one index and
  issues exactly one retrieval call — now over chunks rather than whole
  documents, retrieving the full chunk depth rather than a numeric
  cutoff — before that single result is aggregated to the document
  level and sliced to four cutoffs.
- Significance-testing's `Comparison_Family` and `Reference_Run`
  concepts (Requirements 2 and 5 of that spec) are extended, not
  replaced: the family grows from 1 comparison to 8 (9 runs minus 1
  pinned reference), and `Reference_Run` selection changes from an
  implicit "first bm25 run_id in sorted order" rule to an explicit,
  declared pin, because with three chunking strategies now present, an
  implicit rule could silently select a different BM25 variant (e.g.
  `bm25__fixed_window`) as the reference.

### Out of scope

Per `docs/PROJECT_BRIEF.md` and `.kiro/steering/scope-guard.md`, the
following are explicitly out of scope for this spec (no requirements
and no tasks appear below for any of these):

- `ANALYSIS.md` (failure bucketing / mechanism-pass analysis) — remains
  session-2 work not covered by this spec.
- Any change to the groundedness gate's behavior, configuration, or
  output artifacts. `configs/groundedness.yaml`, `src/groundedness_runner.py`,
  `src/generator_model.py`, `src/judge_model.py`, and `src/quarantine_rule.py`
  are unchanged in behavior; the only touch is the internal,
  output-preserving `src/retrieval_replay.py` update (Requirement 10).
- Any change to the metric definitions in `src/metrics.py`
  (`recall_at_k`, `ndcg_at_10`, `mrr_at_10`, `judged_relevant_docs`,
  `mean_over_qualifying_queries`, `scored_query_count`) — reused as-is,
  computed over aggregated document rankings exactly as they were
  computed over whole-document rankings in session 1.
- Any change to the significance scheme itself (`paired_bootstrap`,
  `permutation_test`, `holm_bonferroni` function bodies, resample /
  permutation counts, or Alpha) — only the family size and the
  reference-run selection mechanism change (Requirement 9).
- Any change to BM25's or `all-MiniLM-L6-v2`'s tokenizer/preprocessing
  settings, or to BM25's `k1`/`b` — these remain exactly as fixed in
  session 1.
- Hybrid retrieval / score fusion, cross-encoder or LLM reranking, a
  fourth retriever, query expansion/rewriting, and approximate nearest
  neighbour indexes — all remain out of scope per `scope-guard.md`'s
  prohibited list.

## Glossary

- **Chunk**: A contiguous span of a single corpus document's `title` +
  `text` content, produced by a Chunker, identified unambiguously by
  its source document ID and its position within that document's
  ordered chunk list. A Chunk is the unit a retriever actually indexes
  and scores under this spec, replacing the whole document as that
  unit for `fixed_window` and `sentence_window` chunking.
- **Chunking_Strategy**: One of the three declared strategies —
  `whole_document`, `fixed_window`, or `sentence_window` — that
  determines how a Chunker splits each corpus document into Chunks.
  Two Run_Ids share a Chunking_Strategy if and only if they were
  produced using the same Chunker configuration.
- **Chunker**: The pure component that maps one corpus document to an
  ordered list of one or more Chunks, according to one declared
  Chunking_Strategy. `Whole_Document_Chunker`, `Fixed_Window_Chunker`,
  and `Sentence_Window_Chunker` are the three Chunker implementations
  this spec adds (the first as a no-op wrapper of session-1's existing
  behavior).
- **Whole_Document_Chunker**: The Chunker for the `whole_document`
  Chunking_Strategy. Produces exactly one Chunk per corpus document,
  containing that document's full, unmodified content — a no-op
  wrapping of session-1's existing whole-document behavior, not a
  behavior change.
- **Fixed_Window_Chunker**: The Chunker for the `fixed_window`
  Chunking_Strategy. Splits a document into consecutive, possibly
  overlapping Chunks of a declared `window_size` measured in
  all-MiniLM-L6-v2 subword-token units, advancing by a declared
  `stride` between consecutive chunk start positions.
- **Sentence_Window_Chunker**: The Chunker for the `sentence_window`
  Chunking_Strategy. Splits a document into sentences using the same
  `_SENTENCE_BOUNDARY` heuristic `src/claim_segmenter.py` already
  defines, groups consecutive sentences into Chunks of a declared
  `sentences_per_chunk` count, and further splits any resulting group
  whose token length (all-MiniLM-L6-v2 tokenizer) exceeds a declared
  `max_chunk_tokens` cap.
- **Chunk_Ranked_List**: The ordered list of (Chunk identifier,
  retrieval score) pairs, for one query and one retriever x
  Chunking_Strategy run, ranked by descending retrieval score, produced
  by a single `retrieve_all` call at Full_Chunk_Depth. Carries each
  Chunk's numeric retrieval score alongside its identifier, not an
  ordered ID list alone, so that Max_Aggregation can select a
  document's maximum Chunk score by score value rather than by rank
  position.
- **Full_Chunk_Depth**: The retrieval depth used for every Chunk_Ranked_List:
  every Chunk present in that run's chunk index, never a fixed numeric
  depth (e.g. 200) and never a truncated top-N chunk list. Because
  `BM25Okapi.get_scores()` and the dense brute-force similarity matmul
  both already score every item before ranking, requesting
  Full_Chunk_Depth costs no additional retrieval call beyond the single
  call session-1's Requirement 5 already established. For
  `fixed_window` and `sentence_window` Chunking_Strategy entries, the
  resulting chunk count for a run exceeds the corpus document count.
- **Max_Aggregation**: The rule that a Document_Ranked_List's score for
  one document, for one query and one retriever x Chunking_Strategy
  run, equals the maximum numeric retrieval score among that document's
  Chunks appearing in the single Chunk_Ranked_List for that query,
  selected by score value rather than by rank position. Applied
  identically for BM25 and both dense retrievers, and for all three
  Chunking_Strategy entries, including `whole_document` (where
  aggregation is a no-op because every document has exactly one
  Chunk).
- **Document_Ranked_List**: The ordered list of document IDs, for one
  query and one retriever x Chunking_Strategy run, ranked by descending
  Max_Aggregation score, derived from that run's single Chunk_Ranked_List
  without any additional retrieval call. Sliced to each of the four
  declared cutoffs (1, 5, 10, 20) to compute that cutoff's metrics —
  the direct chunk-terms restatement of session-1's Ranked_List.
- **Run_Id**: Unchanged in definition from session-1/significance-testing:
  `{retriever_name}__{chunking_strategy}`. This spec is the first to
  actually vary `chunking_strategy` across more than one declared
  value, producing 9 distinct Run_Ids from the 3 retrievers x 3
  Chunking_Strategy entries.
- **Reference_Run**: The single run, identified by its Run_Id, against
  which every other run is compared in the Significance_Report. Unlike
  the significance-testing spec (where exactly one BM25 run existed and
  selection was implicit), this spec requires the Reference_Run to be
  pinned explicitly to the Run_Id `bm25__whole_document`, matching the
  already-published README/SPEC baseline, and prohibits inferring it by
  sorting Run_Ids or by any other implicit rule.
- **Comparison_Family**: The fixed, declared-in-advance set of
  comparisons over which the Holm-Bonferroni adjustment is applied:
  every one of the 9 Run_Ids present in `results/per_query.csv` except
  the pinned Reference_Run, compared against the Reference_Run on
  nDCG@10 — 8 comparisons under this spec's full grid.
- **Token_Length_Exceedance_Report**: The committed JSON artifact under
  `results/` (either `results/token_length_report.json`, extended, or a
  clearly-linked companion file) that reports, for each of the 6
  (Chunking_Strategy x dense-model) cells, the count and fraction of
  Chunks whose untruncated token length exceeds that model's own
  tokenizer-reported maximum sequence length.
- **Data_Layer_Test**: A pytest test that exercises the real
  `Corpus_Loader`/`load_scifact` against the real cached BEIR SciFact
  data under `data/scifact`, as opposed to an in-memory stub corpus.
- **Real_Corpus_End_To_End_Test**: A pytest test that runs the
  Sweep_Runner against the real BEIR SciFact data (not a stub/in-memory
  corpus) for at least one retriever x Chunking_Strategy combination.
- **Local_Cache_Availability**: The condition, checked by a
  `pytest.mark.skipif` guard, that `data/scifact` (the corpus) and the
  relevant `data/hf_cache/models--*` directories (the model weights)
  are already present on the machine running pytest.
- **Corpus_Loader**, **Qrels**, **Sweep_Config**, **Sweep_Runner**,
  **BM25_Retriever**, **Dense_Retriever**, **Deepest_Cutoff**,
  **Metrics_Calculator**, **Sweep_Report**, **Index_Time**,
  **Query_Latency**, **Stub_Retriever**, **In_Memory_Test_Corpus**:
  Unchanged from `session-1-baseline-sweep`'s Glossary.
- **Per_Query_Report**, **Significance_Analyzer**,
  **Significance_Report**, **Paired_Bootstrap**, **Permutation_Test**,
  **Bootstrap_Config**, **Bootstrap_Seed**, **Holm_Bonferroni_Adjustment**,
  **Primary_Metric**, **Alpha**: Unchanged from `significance-testing`'s
  Glossary.
- **Retrieval_Replay**, **Frozen_Retriever_Config**, **Replayed_Run**,
  **Retrieved_Context**, **Generation_Subset**: Unchanged from
  `groundedness-gate`'s Glossary; Requirement 10 updates
  `Retrieval_Replay`'s internal implementation only, not its contract.

## Requirements

### Requirement 1: Third Retriever (`bge-small-en-v1.5`) Declared As Config Data Only

**User Story:** As a researcher, I want the third retriever added
purely as a `configs/sweep.yaml` entry reusing the existing
`Dense_Retriever` class, so that the grid gains its third retriever
without a new retriever implementation.

#### Acceptance Criteria

1. THE Sweep_Config SHALL declare exactly three retrievers under
   `configs/sweep.yaml`'s `retrievers` list, and no more: exactly one
   entry of type `bm25`, exactly one `Dense_Retriever` entry whose
   `name` field is the exact, case-sensitive string
   `all-MiniLM-L6-v2`, and exactly one `Dense_Retriever` entry whose
   `name` field is the exact, case-sensitive string
   `bge-small-en-v1.5`.
2. THE `Dense_Retriever` entry whose `name` field is
   `bge-small-en-v1.5` (Criterion 1) SHALL declare a `model_name`
   field whose value is the exact, case-sensitive string
   `BAAI/bge-small-en-v1.5`, and SHALL use the identical
   `DenseRetrieverConfig` schema and the identical `Dense_Retriever`
   implementation class already used for the `all-MiniLM-L6-v2`
   entry, so that adding the third retriever introduces no new
   retriever class.
3. THE Dense_Retriever SHALL encode every Chunk and every query for the
   `bge-small-en-v1.5` retriever through the same `build_index`/
   `retrieve_all` code path already used for `all-MiniLM-L6-v2`,
   without prepending a query-prefix instruction string to query text
   for either retriever, even though `bge-small-en-v1.5`'s model
   documentation recommends a query-prefix instruction string for
   best-case asymmetric retrieval quality.
4. THE SPEC.md threats-to-validity material SHALL document the
   query-prefix omission described in Criterion 3 as a deliberately
   accepted limitation of this run, declared in this spec before any
   full-grid sweep result exists, and SHALL state that this omission
   is expected to reduce `bge-small-en-v1.5`'s measured nDCG@10
   relative to a run that included the vendor-recommended
   query-prefix instruction, without asserting a specific numeric
   magnitude for that reduction.
5. IF `configs/sweep.yaml`'s `retrievers` list contains any count of
   entries, combination of retriever types, retriever `name` field
   values, or `model_name` field values other than exactly the one
   `bm25` entry and the two `Dense_Retriever` entries
   (`all-MiniLM-L6-v2` and `bge-small-en-v1.5`) declared in Criteria 1
   and 2 of this requirement, THEN THE Sweep_Runner SHALL halt before
   producing sweep results and SHALL produce an error message naming
   the invalid count, combination, name, or model_name value found.

### Requirement 2: Chunking Abstraction With `whole_document` As An Identity Transform

**User Story:** As a researcher, I want a single Chunker abstraction
applied before every retriever's index build, with `whole_document`
chunking behaving exactly as it did in session 1, so that adding
chunking strategies never silently changes the already-published
whole-document results.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL apply exactly one Chunker, selected by each
   run's declared Chunking_Strategy, to every corpus document before
   that run's retriever's `build_index` call, and SHALL supply that
   run's retriever with a chunk corpus — a mapping from Chunk
   identifier to that Chunk's `title`/`text` content, replacing the
   document-keyed corpus the retriever previously received — such
   that the retriever's `build_index` and `retrieve_all` calls operate
   on Chunk identifiers rather than raw document IDs, for every
   retriever and every Chunking_Strategy, producing an ordered list of
   one or more Chunks per document.
2. THE Whole_Document_Chunker SHALL produce exactly one Chunk per
   corpus document, containing that document's full, unmodified
   `title`/`text` content, so that `whole_document` chunking is a
   no-op wrapping of session-1's existing behavior rather than a
   behavior change.
3. THE Sweep_Runner SHALL construct every Chunk's identifier as a
   single string deterministically derived from its source document ID
   and its 0-based position within that document's ordered chunk list,
   such that: no two Chunks produced within the same run share the
   same identifier; a Chunk's identifier is never equal to any corpus
   document ID; and the source document ID and position are
   recoverable by parsing the identifier alone — for every
   Chunking_Strategy, so a Chunk's originating document ID is always
   recoverable without ambiguity.
4. THE Sweep_Runner SHALL apply the identical Chunker, for a given
   run, to every corpus document without exception, so no document is
   silently skipped or left unchunked.
5. WHILE the `whole_document` Chunking_Strategy is declared for a run,
   THE Sweep_Runner SHALL produce recall@k, nDCG@10, and MRR@10 values
   for that run that are identical, within a floating-point tolerance
   of 1e-9, to the values that run's retriever would have produced
   under session-1's pre-chunking-abstraction `build_index` contract,
   so that introducing the Chunker abstraction itself changes no
   already-published whole-document number.
6. IF a Chunker produces zero Chunks for any corpus document, THEN THE
   Sweep_Runner SHALL raise an error identifying the affected document
   ID and that run's declared Chunking_Strategy, and SHALL NOT proceed
   to that run's `build_index` call.

### Requirement 3: `fixed_window` Chunking Strategy

**User Story:** As a researcher, I want a fixed-size, overlapping,
token-budgeted chunking strategy declared once as data, so that
`all-MiniLM-L6-v2`'s 256-token truncation confound is removed by
construction rather than merely reduced.

#### Acceptance Criteria

1. THE Fixed_Window_Chunker SHALL split each corpus document's
   `title`/`text` content into consecutive Chunks of `window_size`
   subword tokens, excluding any special tokens the tokenizer would
   add, advancing the chunk start position by `stride` subword tokens
   between consecutive Chunks, both `window_size` and `stride`
   measured using the all-MiniLM-L6-v2 tokenizer already cached under
   `data/hf_cache` — the same tokenizer `results/token_length_report.json`
   already uses — not words or characters.
2. `configs/sweep.yaml` SHALL declare `window_size = 200` and
   `stride = 50` (a 150-token overlap) as explicit integer values for
   the `fixed_window` Chunking_Strategy, declared before any sweep
   result exists and never adjusted after seeing results, mirroring
   `tech.md`'s "declared once, not tuned" discipline for BM25
   preprocessing.
3. THE Fixed_Window_Chunker SHALL produce every Chunk such that,
   when that Chunk's own produced text is independently re-tokenized
   by the all-MiniLM-L6-v2 tokenizer, the resulting token count
   (excluding any special tokens the tokenizer would add) is less
   than or equal to `window_size` — not merely that the chunk was
   sliced at a `window_size`-token boundary within the source
   document's token sequence — because a window boundary landing
   mid-word can change subword segmentation when the resulting text
   fragment is tokenized in isolation, and this criterion is what
   guarantees `fixed_window` chunking removes, rather than merely
   shrinks, the whole-document truncation confound by construction.
4. IF a document's all-MiniLM-L6-v2 tokenizer token length (excluding
   any special tokens the tokenizer would add) is less than or equal
   to `window_size`, THEN THE Fixed_Window_Chunker SHALL produce
   exactly one Chunk for that document, containing its full content.
5. THE Fixed_Window_Chunker SHALL cover every token of a document by at
   least one produced Chunk, so no document content is dropped by the
   windowing.
6. THE Fixed_Window_Chunker SHALL produce Chunks in left-to-right
   document order, such that the Chunk at position i+1 in a document's
   ordered chunk list starts, in terms of all-MiniLM-L6-v2 token
   offset within the source document, no earlier than the Chunk at
   position i, for every document and every run.
7. THE Fixed_Window_Chunker SHALL produce every Chunk as an exact,
   reconstructable text span of the source document — a plain string,
   never a token-id array tied to the all-MiniLM-L6-v2 tokenizer —
   derived from the all-MiniLM-L6-v2 token boundary via the
   tokenizer's offset mapping (or an equivalent decode step), so that
   the resulting Chunk text is usable as input to BM25 tokenization
   and to each dense retriever's own tokenizer, not only to
   all-MiniLM-L6-v2's.

### Requirement 4: `sentence_window` Chunking Strategy

**User Story:** As a researcher, I want an N-sentence chunking strategy
with a hard token cap, reusing the repo's one existing sentence-boundary
heuristic, so that chunk boundaries respect sentence structure while
still guaranteeing every chunk fits under the token budget.

#### Acceptance Criteria

1. THE Sentence_Window_Chunker SHALL split each corpus document's
   `title`/`text` content into sentences using the identical
   sentence-boundary regular expression already defined as
   `_SENTENCE_BOUNDARY` in `src/claim_segmenter.py` (any of `.`/`!`/`?`
   immediately followed by whitespace or end-of-text), applied here to
   document text rather than Generated_Answer text — the one
   sentence-boundary heuristic in this repository, reused rather than
   re-implemented.
2. `configs/sweep.yaml` SHALL declare `sentences_per_chunk = 3` and
   `max_chunk_tokens = 256` as explicit integer values for the
   `sentence_window` Chunking_Strategy, declared before any sweep
   result exists and never adjusted after seeing results.
3. THE Sentence_Window_Chunker SHALL partition each document's
   sentences, in order and without overlap between Chunks, into groups
   of up to `sentences_per_chunk` consecutive sentences each, before
   any further splitting described in Criterion 4.
4. IF an N-sentence group produced by Criterion 3 has an
   all-MiniLM-L6-v2 tokenizer token length exceeding
   `max_chunk_tokens`, THEN THE Sentence_Window_Chunker SHALL further
   split that group at its original sentence boundaries into two or
   more consecutive whole-sentence sub-groups, each sub-group
   containing the maximum number of consecutive sentences whose
   combined token length does not exceed `max_chunk_tokens`, repeating
   this splitting until every resulting Chunk's token length is at or
   below `max_chunk_tokens` or until a Chunk consists of exactly one
   sentence — in which case the further-split behavior for that one
   sentence is defined by Criterion 5.
5. IF a single sentence's own all-MiniLM-L6-v2 tokenizer token length
   exceeds `max_chunk_tokens` (so no whole-sentence split under
   Criterion 4 can bring that sentence's Chunk under the cap), THEN
   THE Sentence_Window_Chunker SHALL split that sentence's token
   sequence into two or more consecutive, non-overlapping Chunks of at
   most `max_chunk_tokens` tokens each, covering the sentence's full
   token sequence in left-to-right order, with no token dropped or
   duplicated across the resulting Chunks.
6. THE Sentence_Window_Chunker SHALL produce Chunks that, concatenated
   in order, cover every sentence of the source document exactly once,
   except that a sentence split under Criterion 5's token-level
   fallback SHALL have its full token sequence covered exactly once
   across the two or more resulting Chunks — so that no document
   content is dropped or duplicated by the grouping (Criterion 3), the
   sentence-boundary further-split (Criterion 4), or the token-level
   fallback split (Criterion 5).
7. IF a document's sentence count is less than or equal to
   `sentences_per_chunk` and that document's all-MiniLM-L6-v2 tokenizer
   token length does not exceed `max_chunk_tokens`, THEN THE
   Sentence_Window_Chunker SHALL produce exactly one Chunk for that
   document.

### Requirement 5: Max-Aggregation From Chunk Scores To Document Scores

**User Story:** As a researcher, I want a single, declared-in-advance
rule for turning a retriever's per-chunk scores into a per-document
score, so that the choice of aggregation cannot be seen as tuned after
seeing results and so multi-chunk documents are not penalized for
their own chunk count.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL compute each document's aggregated score, for
   one query and one retriever x Chunking_Strategy run, as the maximum
   numeric retrieval score among that document's Chunks appearing in
   that run's single Chunk_Ranked_List for that query (Max_Aggregation)
   — selected by the Chunk's actual retrieval score value, never by its
   rank position within the Chunk_Ranked_List, and never the mean or
   the sum of that document's Chunk scores.
2. THE Sweep_Runner SHALL apply the identical Max_Aggregation rule for
   every retriever (BM25, `all-MiniLM-L6-v2`, `bge-small-en-v1.5`) and
   for every Chunking_Strategy, including `whole_document`, where
   aggregation is a no-op because every document has exactly one Chunk.
3. THE Max_Aggregation rule SHALL be declared and recorded in `SPEC.md`
   before any full-grid sweep result exists, and SHALL NOT be adjusted
   after seeing results.
4. `SPEC.md` SHALL record an explicit justification for choosing
   maximum over mean or sum: mean and sum would penalize a document for
   having many Chunks, or for irrelevant Chunks diluting one strongly
   relevant Chunk's score, and would make a document's aggregate score
   mechanically dependent on its own chunk count — itself a confound —
   whereas maximum evaluates "does at least one Chunk of this document
   look relevant," which matches what recall@k, nDCG@10, and MRR@10 are
   already trying to measure at the document level.
5. THE Sweep_Runner SHALL break ties among two or more documents with
   equal aggregated score, for a given query and run, using the same
   ascending-document-ID tie-break rule (`doc_id_sort_key`) session-1
   already defines and applies at the per-chunk retrieval level, now
   applied at the document level after aggregation.
6. THE Sweep_Runner SHALL derive every Document_Ranked_List entirely
   from the single Chunk_Ranked_List a run's one `retrieve_all` call
   already returned — including that call's per-Chunk retrieval scores
   required by Criterion 1 — without issuing any additional retrieval
   call to compute, look up, or refine an aggregated score.
7. THE Retriever protocol's `retrieve_all` return contract SHALL
   expose, for every query, each returned Chunk's identifier paired
   with its numeric retrieval score — extending session-1's
   `retrieve_all` return contract (an ordered Chunk-identifier list
   only, with no accompanying score) so a per-Chunk score is available
   to Max_Aggregation — and THIS extended contract SHALL apply
   identically to the BM25_Retriever and to the Dense_Retriever for
   both dense models.

### Requirement 6: Full Chunk Depth Retrieval — Index Once, Retrieve Once, Aggregate Once, Slice Four Ways

**User Story:** As a researcher, I want chunk-level retrieval restated
in the same "index once, retrieve once" terms session 1 established for
whole documents, so that document-level top-k after aggregation is
exactly correct by construction, not an empirically-tuned approximation.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL, for every retriever x Chunking_Strategy run,
   build exactly one index over that run's full set of Chunks across
   the whole corpus, and SHALL issue exactly one retrieval call, across
   all queries, requesting a Chunk_Ranked_List containing every Chunk
   present in that run's chunk index (Full_Chunk_Depth) for every one
   of that run's test queries — never a truncated top-N chunk list and
   never a second retrieval call for any other purpose. THE Sweep_Runner's
   single retrieval call therefore SHALL hold, at once, a
   query_count x total_chunk_count-sized set of Chunk_Ranked_List
   entries for that run — for `fixed_window` and `sentence_window`
   Chunking_Strategy entries, where a document may split into several
   Chunks, this total chunk count exceeds the corpus document count,
   unlike session-1's fixed top_k=20 depth.
2. THE Sweep_Runner SHALL derive each query's Document_Ranked_List by
   applying Max_Aggregation (Requirement 5) to the scores in that run's
   single Chunk_Ranked_List, grouping Chunks by source document ID, and
   SHALL slice that single Document_Ranked_List to each of the four
   declared cutoffs (1, 5, 10, 20) to compute that cutoff's metrics —
   the chunk-terms restatement of session-1's Requirement 5 "index
   once, retrieve once, slice four ways" property.
3. THE Sweep_Runner SHALL perform, across a full run, exactly 9 index
   builds and exactly 9 retrieval calls in total — exactly one index
   build and exactly one retrieval call per retriever x Chunking_Strategy
   combination — and SHALL NOT issue a second retrieval call for
   aggregation or for any cutoff.
4. THIS requirement's design SHALL state Full_Chunk_Depth explicitly as
   "every Chunk present in that run's chunk index," never a fixed
   numeric depth such as 200, and SHALL explain that because every
   Chunk's score is computed, by both `BM25Okapi.get_scores()` and the
   dense brute-force similarity matmul, before any truncation occurs,
   the document-level top-20 ranking produced after aggregation is
   exactly correct by construction, requiring no empirical "is depth D
   large enough" argument. THIS requirement's design SHALL further
   state that, for `fixed_window` and `sentence_window` Chunking_Strategy
   entries, that run's total chunk count — and therefore the size of
   the query_count x total_chunk_count Chunk_Ranked_List set Criterion 1
   requires — is larger than the `whole_document` run's total chunk
   count (which equals the corpus document count), and SHALL document
   this as an accepted consequence of computing the full similarity
   matrix Criterion 1 already requires, not a defect to remediate.
5. THE Sweep_Runner SHALL assign the same Index_Time value, taken from
   a run's single chunk-index build, and the same Query_Latency value,
   taken from that run's single Full_Chunk_Depth retrieval call, to
   every `results/sweep.csv` row sharing that run's Run_Id, consistent
   with session-1's Requirement 5.6/5.7.

### Requirement 7: Config-Driven Retriever x Chunking-Strategy Cross Product

**User Story:** As a researcher, I want `configs/sweep.yaml` to declare
the chunking strategies as a list crossed with the retriever list, so
that the full 9-run grid is declared as data rather than assembled by a
hard-coded loop.

#### Acceptance Criteria

1. `configs/sweep.yaml`'s top-level schema SHALL replace the single
   `chunking_strategy: str` field with a `chunking_strategies` field
   declaring a list of exactly 3 Chunking_Strategy entries:
   `whole_document`, `fixed_window`, and `sentence_window`.
2. `src/config.py`'s `SweepConfig` dataclass and `load_sweep_config`
   function SHALL be updated to parse and validate `chunking_strategies`
   as a list of exactly 3 supported values, and SHALL derive the
   sweep's run set as the full cross product of the 3 declared
   retrievers (Requirement 1) and the 3 declared Chunking_Strategy
   entries, producing exactly 9 runs, with no run omitted and no run
   added beyond that cross product.
3. IF `configs/sweep.yaml` declares `chunking_strategies` with a count
   other than exactly 3, or declares a value not among `whole_document`,
   `fixed_window`, `sentence_window`, THEN `load_sweep_config` SHALL
   raise a config error naming the invalid declaration, and THE
   Sweep_Runner SHALL halt before producing sweep results.
4. IF `configs/sweep.yaml` declares the old singular `chunking_strategy`
   field instead of `chunking_strategies`, THEN `load_sweep_config`
   SHALL raise a config error, so a stale session-1/2-style config
   cannot silently run only 1 chunking strategy instead of 3.
5. THE Sweep_Runner SHALL derive each of the 9 runs' Run_Id as
   `{retriever_name}__{chunking_strategy}`, consistent with the Run_Id
   convention already established in `session-1-baseline-sweep` and
   `significance-testing`.
6. `configs/sweep.yaml` SHALL declare each of `fixed_window`'s
   `window_size`/`stride` and `sentence_window`'s
   `sentences_per_chunk`/`max_chunk_tokens` as explicit data fields
   nested under that Chunking_Strategy's own declaration, not as
   literals hard-coded in `src/` source code.

### Requirement 8: Full 36-Row Grid And Per-Query Row Count

**User Story:** As a researcher, I want the full grid written to
`results/sweep.csv` and `results/per_query.csv` with a row count
derived from the declared grid and the loader's own reported query
count, so that the artifacts are complete and never rely on a
hard-coded query count.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL write exactly 36 rows to `results/sweep.csv`
   — 3 retrievers x 4 cutoffs x 3 Chunking_Strategy entries — one row
   per declared retriever x cutoff x Chunking_Strategy combination.
2. THE Sweep_Runner SHALL write exactly one `results/per_query.csv` row
   per (run_id, query_id) pair, for every one of the 9 run_ids and
   every test query loaded, so the Per_Query_Report contains `9 * Q`
   rows, where `Q` is the Corpus_Loader's own reported test-query
   count for that run.
3. THIS spec SHALL NOT hard-code `Q` (currently 300) as a literal
   expectation anywhere in requirements, design, or test code; any
   assertion on `results/per_query.csv`'s row count SHALL derive `Q`
   from the Corpus_Loader's own reported count for the test run being
   verified, consistent with `evaluation-integrity.md`'s "dataset stats
   come from the loader's own output" rule.
4. THE Sweep_Runner SHALL retain every one of the 36 declared
   `results/sweep.csv` rows regardless of its computed metric or timing
   values, extending `evaluation-integrity.md`'s "no row gets dropped
   for being unflattering" rule to the grown grid.
5. THE Sweep_Runner SHALL assign the same Index_Time and the same
   Query_Latency value to every `results/sweep.csv` row sharing a
   run_id, and SHALL assign a different run_id to rows produced from a
   different retriever x Chunking_Strategy combination, consistent with
   session-1's Requirement 7.4/7.7.

### Requirement 9: Significance Re-Run Over An 8-Member Comparison Family With A Pinned Reference Run

**User Story:** As a researcher, I want the significance analysis
re-run over all 9 grid runs with the Reference_Run pinned explicitly to
`bm25__whole_document`, so that adding two chunking strategies cannot
silently swap out the already-published baseline comparison.

#### Acceptance Criteria

1. THE Significance_Analyzer's Comparison_Family SHALL contain exactly
   8 comparisons: every one of the 9 Run_Ids present in
   `results/per_query.csv` except exactly 1 pinned Reference_Run,
   compared against that Reference_Run on nDCG@10.
2. THE Reference_Run SHALL be pinned explicitly to the Run_Id
   `bm25__whole_document` — the retriever named `bm25` under the
   `whole_document` Chunking_Strategy — matching the already-published
   README/SPEC baseline.
3. `src/significance.py`'s reference-run selection (currently
   `_find_reference_run_id`, "first bm25 run_id in sorted order") SHALL
   be replaced with logic that requires an explicit reference
   identification — via an explicit reference Chunking_Strategy field
   added alongside `configs/significance.yaml`'s existing
   `reference_retriever` field, or an explicit reference `run_id`
   field — and SHALL NOT select a reference run by sorting Run_Ids and
   taking the first result, or by any other implicit rule, because with
   3 Chunking_Strategy entries now present, an implicit rule could
   silently select a different BM25 variant (e.g. `bm25__fixed_window`)
   as the reference.
4. IF the explicitly pinned Reference_Run's Run_Id is absent from
   `results/per_query.csv`, THEN THE Significance_Analyzer SHALL halt
   before writing `results/significance.csv`, produce an error message
   naming the pinned Run_Id, and terminate with a non-zero exit status
   — the same halt behavior significance-testing's Requirement 2.5
   already defines, now keyed on the explicit pin rather than an
   implicit BM25-only match.
5. THE Holm_Bonferroni_Adjustment's step-down multiplier, ascending-sort
   ordering rule, monotonic non-decrease enforcement, clamping to
   [0.0, 1.0], and tie-handling rule SHALL remain unchanged in behavior
   from significance-testing's Requirement 5, applying identically
   regardless of whether the Comparison_Family has 1 member or 8.
6. THE bootstrap resample count, the permutation iteration count, the
   Bootstrap_Seed, and Alpha (0.05) declared in `configs/significance.yaml`
   SHALL remain unchanged from the significance-testing spec's
   declarations.
7. THIS spec SHALL NOT modify the `paired_bootstrap`, `permutation_test`,
   or `holm_bonferroni` function bodies in `src/significance.py` — only
   the reference-run selection mechanism and the resulting family size
   change.
8. THE Significance_Analyzer SHALL report, for every one of the 8
   comparisons in the Comparison_Family and for every secondary-metric
   comparison it computes, one row in `results/significance.csv`, and
   SHALL NOT drop, exclude, or filter any computed comparison from the
   report on the basis of its result value.

### Requirement 10: `retrieval_replay.py` Update — Output-Preserving Whole-Document No-Op

**User Story:** As a maintainer, I want the groundedness gate's frozen
retrieval path updated to route through the new chunking-aware
`build_index` contract without changing any of its own output, so the
groundedness gate keeps working without a behavior change or a re-audit
of its already-published numbers.

#### Acceptance Criteria

1. `src/retrieval_replay.py`'s `build_frozen_retriever` SHALL apply the
   Whole_Document_Chunker (Requirement 2's no-op, exactly 1 Chunk per
   document) to the loaded corpus before calling the retriever's
   `build_index`, and SHALL apply a trivial Max_Aggregation of exactly
   1 Chunk's score per document before `replay_retrieval` slices the
   result to `replay_top_k`.
2. BECAUSE `whole_document` chunking followed by Max_Aggregation of
   exactly 1 Chunk is mathematically the identity transform on a
   single-chunk document, THE updated `build_frozen_retriever`/
   `replay_retrieval` path SHALL produce, for every Generation_Subset
   query, a Retrieved_Context that is byte-for-byte/numerically
   identical to the Retrieved_Context the pre-update code path
   produced for that same query.
3. WHEN the groundedness gate is re-run after this update with the
   identical `configs/groundedness.yaml` declarations (including
   `replayed_run_id: bm25__whole_document`), THE re-run SHALL produce
   `results/groundedness.csv`, `results/generated_answers.csv`,
   `results/hand_checked_sample.csv`, `results/hand_checked_joined.csv`,
   and `results/hand_checked_sample_context.md` that are byte-for-byte
   identical to the files that existed before this update, or, for any
   floating-point column, numerically identical within a tolerance of
   1e-9.
4. THIS spec SHALL provide an explicit, automatable acceptance check
   for Criterion 3 — either an equivalence test that asserts
   byte-for-byte/numeric equality after a rerun, or a documented
   rerun-and-diff verification step — rather than relying on a prose
   assertion alone.
5. THE Run_Id `bm25__whole_document` referenced by
   `configs/groundedness.yaml`'s `replayed_run_id` field SHALL continue
   to resolve correctly against the updated `configs/sweep.yaml` schema
   (Requirement 7), because `whole_document` remains a valid declared
   Chunking_Strategy value.
6. THIS spec SHALL NOT change `configs/groundedness.yaml`,
   `src/groundedness_runner.py`, `src/generator_model.py`,
   `src/judge_model.py`, or `src/quarantine_rule.py`.

### Requirement 11: Token-Length Exceedance Receipts For Two Dense Models Across Three Chunking Strategies

**User Story:** As a researcher, I want the token-length exceedance
measurement extended to both dense models and to all three chunking
strategies, measured at chunk granularity, so the "confound removed"
claim has a receipt for `bge-small-en-v1.5` as well as
`all-MiniLM-L6-v2`, and for the new chunking strategies as well as
`whole_document`.

#### Acceptance Criteria

1. THE Token_Length_Analysis SHALL report, for each of the 3
   Chunking_Strategy entries crossed with each of the 2 dense-model
   tokenizers (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`) — 6 cells total
   — the count and fraction of Chunks whose untruncated token length,
   measured by that cell's own model's tokenizer, exceeds that same
   tokenizer's own reported maximum sequence length.
2. THE Token_Length_Analysis SHALL read each dense model's maximum
   sequence length from that model's own loaded tokenizer's reported
   value at run time (e.g. `tokenizer.model_max_length`), rather than
   hard-coding 256 for `all-MiniLM-L6-v2` or 512 for `bge-small-en-v1.5`,
   consistent with `evaluation-integrity.md`'s "dataset stats come from
   the loader's own output" rule extended to tokenizer properties.
3. THE Token_Length_Analysis SHALL persist all 6 cells' counts and
   fractions to a single, well-defined, committed JSON artifact under
   `results/` — either the existing `results/token_length_report.json`,
   extended, or a clearly-linked companion artifact — with the specific
   file and internal schema left as a design decision.
4. THE Token_Length_Analysis SHALL measure Chunk-level exceedance (each
   Chunking_Strategy's actual produced Chunks) for every one of the 6
   cells, including the `whole_document` cells, so the `whole_document`
   cells of the extended report remain directly comparable to, and for
   `all-MiniLM-L6-v2` numerically consistent with, the existing
   single-model `results/token_length_report.json` measurement this
   spec extends.
5. THE Token_Length_Analysis SHALL make no network call while measuring
   any of the 6 cells, reusing the same offline tokenizer-loading
   discipline (`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` plus
   `local_files_only=True`) already established for `all-MiniLM-L6-v2`
   in `src/token_length_analysis.py`.

### Requirement 12: Data-Layer Tests And Real-Corpus End-To-End Tests, Skipped Without Local Cache

**User Story:** As a maintainer, I want pytest coverage over the real
`Corpus_Loader` and a real-corpus end-to-end run of the Sweep_Runner,
skipped automatically when the local BEIR/model cache is absent, so
that session-2's outstanding data-layer and end-to-end test coverage
ships without breaking CI on a clean checkout.

#### Acceptance Criteria

1. THE test suite SHALL include one or more Data_Layer_Tests that
   invoke the real `Corpus_Loader`/`load_scifact` against the real
   cached BEIR SciFact data under `data/scifact`.
2. THE test suite SHALL include one or more Real_Corpus_End_To_End_Tests
   that run the Sweep_Runner against the real BEIR SciFact data — not a
   stub/in-memory corpus — for at least one full retriever x
   Chunking_Strategy combination, and SHALL assert that the resulting
   `results/sweep.csv`/`results/per_query.csv` rows for that combination
   are well-formed: correct columns present, and every metric value
   either a float in [0.0, 1.0] or the declared `MISSING` sentinel.
3. THE Data_Layer_Tests and the Real_Corpus_End_To_End_Tests SHALL each
   be decorated with a `pytest.mark.skipif` condition (or an equivalent
   fixture-based check) that inspects Local_Cache_Availability —
   whether `data/scifact` and the relevant `data/hf_cache/models--*`
   directories are already present — and SHALL be automatically
   skipped, not failed and not erroring, when that local cache is
   absent.
4. WHEN these tests run in the GitHub Actions CI environment described
   in `structure.md` (a clean checkout that never downloads a dataset
   or model weights, per `tech.md`), THE test suite SHALL report the
   tests described in Criteria 1 and 2 as skipped, not failed.
5. WHEN these tests run locally where `data/scifact` and the relevant
   model weight caches already exist, THE test suite SHALL execute the
   tests described in Criteria 1 and 2, and those tests SHALL pass.
6. THE tests described in this requirement SHALL be added in addition
   to, and SHALL NOT replace or modify, `tests/test_metrics.py`,
   `tests/test_orchestration.py`, `tests/test_significance.py`,
   `tests/test_claim_segmenter.py`, `tests/test_quarantine_rule.py`,
   `tests/test_token_length_analysis.py`, and
   `tests/test_verify_writeup_numbers.py`.

### Requirement 13: Extended Stub-Based Orchestration Test For Chunk-Level Aggregation

**User Story:** As a maintainer, I want an automated, stub-based test
that proves the chunk-level "index once, retrieve once (full chunk
depth), aggregate once, slice four ways" property across a chunking-
strategy axis, so that Requirement 6's property is verified by a test
rather than resting on code review alone.

#### Acceptance Criteria

1. THE test suite SHALL extend `tests/test_orchestration.py`, or add an
   accompanying module, with a stub-based test that exercises the
   "index once, retrieve once (Full_Chunk_Depth), aggregate once, slice
   four ways" property (Requirement 6) with a Chunking_Strategy axis
   present, using one or more Stub_Retriever instances and an in-memory
   multi-chunk-per-document corpus, with no network call and no model
   loaded.
2. THE extended test SHALL assert, for each of at least two distinct
   (retriever, Chunking_Strategy) run combinations, that `build_index`
   was called exactly once and `retrieve_all` was called exactly once
   for that combination.
3. THE extended test SHALL assert that the aggregated per-document
   score used to produce every cutoff's Document_Ranked_List equals the
   maximum score among that document's Chunks' scores from the single
   `retrieve_all` call for that (retriever, Chunking_Strategy) run,
   using a hand-specified in-memory corpus in which at least one
   document has more than one Chunk with differing scores, so the
   max-versus-mean/sum distinction is actually exercised rather than
   trivially satisfied by single-chunk documents.
4. THIS requirement's test SHALL NOT be treated as satisfying, or as a
   substitute for, the Data_Layer_Tests or Real_Corpus_End_To_End_Tests
   required by Requirement 12.

### Requirement 14: Repo-Consistency Updates To README, SPEC, And The Traceability Ledger

**User Story:** As a maintainer, I want `README.md`, `SPEC.md`, and
`docs/numeric_traceability.csv` updated to describe the completed
36-row grid and 8-comparison significance re-run, so every reader-facing
number still traces to a committed artifact.

#### Acceptance Criteria

1. `README.md`'s headline finding, results table, and "Reproducing this
   sweep" section SHALL be updated to report the 36-row
   `results/sweep.csv` and 8-comparison `results/significance.csv`
   produced by this spec, remaining internally consistent with those
   artifacts.
2. Every new number introduced into `README.md` or `SPEC.md` by this
   spec's updates SHALL trace to a committed artifact via a
   corresponding new row in `docs/numeric_traceability.csv`, following
   that file's existing `claim_id,document,location,stated_value,
   stated_precision,source_artifact,source_fields,computation` schema.
3. `SPEC.md`'s "Design summary" section SHALL be updated to describe
   the 3x4x3 grid, the 9 runs, the Max_Aggregation convention (a new
   subsection alongside the existing nDCG@10 convention section), and
   the pinned Reference_Run.
4. `SPEC.md`'s "Threats to validity" section SHALL gain the
   `bge-small-en-v1.5` query-prefix limitation (Requirement 1) and
   SHALL update the existing token-length-truncation threat to describe
   the new per-model, per-Chunking_Strategy measurement (Requirement 11)
   rather than describing only the prior whole-document-only,
   single-model number.
5. THE `README.md` and `SPEC.md` updates described in this requirement
   SHALL NOT alter the wording of the pre-existing "What this does not
   claim" bullet list's chunking/retriever generalization caveats beyond
   what is necessary to reflect that all 3 Chunking_Strategy entries and
   all 3 retrievers are now covered by the grid.
