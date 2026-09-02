# Requirements Document

## Introduction

This spec covers `ANALYSIS.md` — the mechanism / failure-bucket analysis
named in `.kiro/steering/structure.md`'s definition of done, and the
last remaining deliverable in this repository. It is a write-up of
results that already exist, plus the two committed artifacts that make
its failure-bucket figures reproducible from code rather than from
prose.

Every number in `ANALYSIS.md` derives from an artifact that is already
committed (`results/sweep.csv`, `results/per_query.csv`,
`results/significance.csv`, `results/run_config.json`,
`results/token_length_report.json`, `results/groundedness.csv`,
`results/hand_checked_joined.csv`, `results/generated_answers.csv`) or
from one of the two artifacts this spec introduces
(`results/failure_buckets.csv`, `results/failure_bucket_counts.csv`).

The boundary between what loads a corpus and what does not is precise,
and stating it precisely matters more here than any single number this
spec produces:

- The Failure_Bucket taxonomy and the Contrast_Bucket taxonomy are
  corpus-free. Every predicate reads `results/per_query.csv` and
  nothing else, so no bucket label depends on a token count, on a
  corpus document, or on a model.
- The per-query token-length covariate columns of
  `results/failure_buckets.csv` are not corpus-free. A second,
  separately specified stage of the same entry point loads the
  already-cached BEIR SciFact corpus and Qrels from `data/`, loads each
  Dense_Model's already-cached tokenizer from `data/`, and computes
  those columns. That stage is the only part of this spec that reads a
  corpus, a tokenizer, or a model, it runs CPU-only, and it never
  reaches the network.
- The test suite and the Verification_Pass stay corpus-free,
  model-free and network-free. Both read only committed files; the
  covariate computation is covered with a stub corpus and a stub
  tokenizer rather than with the real ones.

Committing the covariates is what lets the Analysis_Document state a
query-level truncation mechanism at all. The corpus-level truncation
fractions the Token_Length_Report already records cannot be attributed
to an individual query; a committed per-query column can be. The
alternative — forbidding the covariate — would leave every query-level
mechanism claim without a receipt, so the covariate is committed rather
than prohibited.

This spec runs no retrieval, computes no embedding, invokes no
generative model, makes no network call, and re-runs neither the sweep,
nor the significance analysis, nor the groundedness gate, nor the
token-length analysis. It adds no entry to `requirements.txt`:
`transformers==5.16.1` and `sentence-transformers==6.0.0` are already
pinned there, so the covariate stage introduces no new top-level
dependency.

Failure buckets are assigned by committed code, never by hand: the
Bucket_Assigner's Bucket_Assignment_Stage reads
`results/per_query.csv`, applies a fixed taxonomy declared in this
document, and its Covariate_Enrichment_Stage adds the per-query
token-length covariate columns before either artifact is written. The
counts artifact exists so that every failure-bucket figure in
`ANALYSIS.md` resolves through a unique single-row selector — the
existing `src/verify_writeup_numbers.py` resolver already errors when a
selector matches more than one row, so this spec adds both files to that
module's `_CSV_ARTIFACTS` tuple and changes no resolution logic.

**Deviation from `.kiro/steering/structure.md`, recorded deliberately.**
That steering document says "the failure bucket is a column in
`results/sweep.csv`". `results/sweep.csv` is keyed by (`run_id`, `k`)
and holds 36 per-configuration rows; it has no per-query dimension, so a
per-query bucket label has no home in one of its rows without either
collapsing 300 labels into a single cell or multiplying the file by the
query count. This spec therefore carries the bucket label in
`results/failure_buckets.csv` (one row per `run_id`, `query_id`) and its
aggregation in `results/failure_bucket_counts.csv` (one row per
`run_id`, `bucket`), and leaves `results/sweep.csv` byte-for-byte
unchanged. The property that steering document protects — that the
failure analysis is reproducible straight from a committed artifact
rather than from ad hoc notes — holds unchanged, and holds more
directly, because the partition itself is now a committed artifact.

Inferential claims in `ANALYSIS.md` are limited to what
`results/significance.csv` already contains: the pre-declared 8-member
nDCG@10 comparison family under Holm-Bonferroni correction. Every
contrast this spec introduces — every bucket count, every bucket
fraction, every pairwise disagreement figure, every covariate
description — is descriptive only, is labelled in prose as outside that
pre-declared family, and carries no inferential claim.

Out of scope for this spec, per `.kiro/steering/scope-guard.md` (no
requirement and no task below produces any of these, and none SHALL be
described in `ANALYSIS.md` as work to build): hybrid retrieval or score
fusion, cross-encoder or LLM reranking, any retriever beyond the three
already swept, query expansion or query rewriting, approximate nearest
neighbour indexes, and fine-tuning of any model. Also out of scope: any
new retrieval run, any new chunking strategy, any new p-value, any new
confidence interval, any post-hoc hypothesis test, and any edit to
`results/sweep.csv`, `results/per_query.csv`,
`results/significance.csv`, `results/run_config.json`,
`results/token_length_report.json`, or any `results/groundedness*`,
`results/generated_answers.csv`, or `results/hand_checked*` artifact.

## Glossary

- **Analysis_Writeup**: This feature as a whole — the activity of
  producing the Analysis_Document, the Bucket_Assigner, the
  Failure_Bucket_Report, and the Failure_Bucket_Counts_Report, and of
  extending the Traceability_Ledger and the Verifier to cover the
  Analysis_Document.
- **Analysis_Document**: The `ANALYSIS.md` file at the repository root —
  the mechanism / failure-bucket analysis named in
  `.kiro/steering/structure.md`'s definition of done.
- **Readme_Document**: The `README.md` file at the repository root,
  already committed and read-only for this spec.
- **Spec_Document**: The `SPEC.md` file at the repository root, already
  committed and read-only for this spec.
- **Per_Query_Report**: The committed `results/per_query.csv` file: one
  row per (`run_id`, `query_id`) pair, with columns `run_id`,
  `retriever`, `chunking_strategy`, `query_id`, `recall_at_1`,
  `recall_at_5`, `recall_at_10`, `recall_at_20`, `ndcg_at_10`,
  `mrr_at_10`, and `num_judged_relevant`. The sole input to the
  Bucket_Assigner, and read-only for this spec.
- **Sweep_Report**: The committed `results/sweep.csv` file, keyed by
  (`run_id`, `k`), with no per-query dimension. Read-only for this
  spec.
- **Significance_Report**: The committed `results/significance.csv`
  file: one row per (comparison, metric) pair, carrying `mean_diff`,
  `ci_lower`, `ci_upper`, `p_value_raw`, `p_value_adjusted`,
  `n_shared_queries`, `is_primary`, and `verdict`. Read-only for this
  spec, and the only source of an inferential statement in the
  Analysis_Document.
- **Run_Config_Record**: The committed `results/run_config.json` file.
  Read-only for this spec.
- **Token_Length_Report**: The committed
  `results/token_length_report.json` file, carrying `model_name`,
  `max_sequence_length`, `num_documents_total`,
  `num_documents_exceeding`, `fraction_exceeding`, and a `cells` list
  with one entry per (Chunking_Strategy, dense model) pair. Read-only
  for this spec, and the sole source of every truncation figure the
  Analysis_Document states.
- **Run_Id**: The identifier of one swept configuration, formed as
  `{retriever}__{chunking_strategy}`, as it appears in the
  Per_Query_Report's `run_id` column. Nine Run_Ids exist: the three
  retrievers `bm25`, `all-MiniLM-L6-v2`, and `bge-small-en-v1.5`, each
  crossed with the three Chunking_Strategy values `whole_document`,
  `fixed_window`, and `sentence_window`.
- **Chunking_Strategy**: One of the three strategy values recorded in
  the Per_Query_Report's `chunking_strategy` column: `whole_document`,
  `fixed_window`, or `sentence_window`.
- **Reference_Run**: The Run_Id `bm25__whole_document` — the reference
  row every other run is reported against, per
  `.kiro/steering/evaluation-integrity.md`.
- **Primary_Metric**: nDCG@10, designated the single primary metric
  before any sweep result existed, per
  `.kiro/steering/evaluation-integrity.md`, and not revised by this
  spec.
- **Pre_Declared_Family**: The 8 rows of the Significance_Report whose
  `metric` is `ndcg_at_10` and whose `is_primary` is true — one per
  non-reference Run_Id, each compared against the Reference_Run under
  Holm-Bonferroni correction. The complete and only set of inferential
  results available to the Analysis_Document.
- **Bucket_Assigner**: The committed module, with a single command-line
  entry point, that runs the Bucket_Assignment_Stage and the
  Covariate_Enrichment_Stage in one invocation, asserts both partitions
  are total, and writes the Failure_Bucket_Report and the
  Failure_Bucket_Counts_Report.
- **Bucket_Assignment_Stage**: The stage of the Bucket_Assigner that
  reads the Per_Query_Report and applies the Failure_Bucket taxonomy of
  Requirement 3 and the Contrast_Bucket taxonomy of Requirement 4. Reads
  the Per_Query_Report and no other data.
- **Covariate_Enrichment_Stage**: The stage of the Bucket_Assigner that
  loads the Local_Cache's BEIR SciFact corpus, Qrels, and Dense_Model
  tokenizers and computes each query's Token_Length_Covariate values,
  specified in Requirement 16. The only part of the Analysis_Writeup
  that reads a corpus, a tokenizer, or a model.
- **Qrels**: The human relevance judgments shipped with the BEIR SciFact
  test split, as loaded by the committed `src/corpus_loader.py`
  `load_scifact` function: a mapping from `query_id` to a mapping from
  document ID to an integer relevance score. The only source of
  relevance in this repository, per
  `.kiro/steering/evaluation-integrity.md`.
- **Judged_Relevant_Document**: A document whose Qrels relevance score
  for a given `query_id` is strictly greater than 0 — the same condition
  the committed `src/metrics.py` `judged_relevant_docs` function
  applies.
- **Dense_Model**: One of the two dense retrieval models already swept:
  `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`.
- **Effective_Max_Sequence_Length**: The maximum sequence length one
  Dense_Model actually truncates to at encode time, resolved from that
  model's own cached configuration by the committed
  `src/token_length_analysis.py` `resolve_effective_max_sequence_length`
  function. The two Dense_Models do not share this value.
- **Token_Length_Covariate**: One of the three per-query, per-Dense_Model
  values the Covariate_Enrichment_Stage computes: the query's own token
  length, the maximum token length over that query's
  Judged_Relevant_Documents, and whether any of that query's
  Judged_Relevant_Documents exceeds that Dense_Model's
  Effective_Max_Sequence_Length.
- **Local_Cache**: The `data/` directory as configured by the committed
  `src/corpus_loader.py` `configure_caches` function, holding the
  downloaded BEIR SciFact dataset under `data/scifact` and the
  downloaded model and tokenizer snapshots under `data/hf_cache`.
- **Missing_Value_Sentinel**: The literal string `"NA"`, exposed as the
  committed `src/report.py` `MISSING` constant — this repository's
  existing marker for a value that has no defined number, and distinct
  from a numeric `0`.
- **Failure_Bucket**: Exactly one of the four per-run labels
  `total_miss`, `mis_ranked`, `partial_recall`, or `full_success`,
  assigned to one (Run_Id, `query_id`) pair by the Bucket_Assigner.
- **Answered_Query**: A (Run_Id, `query_id`) pair whose `ndcg_at_10`
  value in the Per_Query_Report is strictly greater than 0.
- **Missed_Query**: A (Run_Id, `query_id`) pair whose `ndcg_at_10` value
  in the Per_Query_Report is exactly 0.
- **Pair_Contrast**: An ordered pair of distinct Run_Ids (Run_A,
  Run_B), drawn from the Declared_Contrast_Set, over whose shared
  `query_id` values the Contrast_Bucket taxonomy is applied.
- **Run_A**: The first Run_Id of a Pair_Contrast — the Reference_Run for
  a family-aligned contrast, and the `whole_document` Run_Id for a
  cross-strategy contrast.
- **Run_B**: The second Run_Id of a Pair_Contrast — the non-reference
  Run_Id for a family-aligned contrast, and the `fixed_window` or
  `sentence_window` Run_Id for a cross-strategy contrast.
- **Contrast_Bucket**: Exactly one of the four Pair_Contrast labels
  `a_only`, `b_only`, `both_miss`, or `both_answer`, assigned to one
  (Pair_Contrast, `query_id`) pair by the Bucket_Assigner.
- **Composite_Run_Id**: The identifier a Pair_Contrast occupies in the
  Failure_Bucket_Counts_Report's `run_id` column, formed as
  `{Run_A}|vs|{Run_B}` — for example
  `bm25__whole_document|vs|bge-small-en-v1.5__whole_document`.
- **Declared_Contrast_Set**: The fixed set of 12 Pair_Contrasts this
  spec computes, enumerated in Requirement 4.
- **Failure_Bucket_Report**: The `results/failure_buckets.csv` file this
  spec introduces: one row per (Run_Id, `query_id`) pair, carrying that
  pair's Failure_Bucket and that query's six Token_Length_Covariate
  values.
- **Failure_Bucket_Counts_Report**: The
  `results/failure_bucket_counts.csv` file this spec introduces: one row
  per (`run_id`, `bucket`) pair, carrying that bucket's count and
  fraction, where `run_id` is either a Run_Id (for a per-run bucket) or
  a Composite_Run_Id (for a Pair_Contrast bucket).
- **Totality_Assertion**: The Bucket_Assigner's check that each
  partition is total and mutually exclusive — that a partition's bucket
  counts sum to the number of `query_id` values that partition covers.
- **Judged_Relevant_Count**: The Per_Query_Report's
  `num_judged_relevant` column value for one (Run_Id, `query_id`) pair
  — the one per-query covariate this spec inherits from an existing
  committed artifact rather than computing.
- **Descriptive_Contrast**: Any comparison, count, fraction, or
  covariate description the Analysis_Document states that is not a row
  of the Pre_Declared_Family. Reported without an inferential claim and
  labelled as such in prose.
- **Numeric_Claim**: Any number appearing in the Analysis_Document that
  states a measured, computed, or configured quantity belonging to this
  repository's sweep, significance analysis, token-length analysis,
  groundedness gate, or failure-bucket partition — including a metric
  value, a count, a fraction, a percentage, a ratio, a timing value, or
  a recorded configuration value such as a seed.
- **Traceability_Ledger**: The committed `docs/numeric_traceability.csv`
  file, whose columns are `claim_id`, `document`, `location`,
  `stated_value`, `stated_precision`, `source_artifact`,
  `source_fields`, and `computation`, and which currently holds 167
  rows whose `document` value is either `README.md` or `SPEC.md`.
- **Verifier**: The committed `src/verify_writeup_numbers.py` module,
  invoked as `python -m src.verify_writeup_numbers`, which checks every
  Traceability_Ledger row against the document it cites and the
  artifact it cites, and exits non-zero if any row fails.
- **Verification_Pass**: One invocation of the Verifier over the whole
  Traceability_Ledger, covering every ledgered document.

## Requirements

### Requirement 1: One Document, Two New Artifacts, Nothing Regenerated

**User Story:** As a maintainer, I want this feature to add `ANALYSIS.md`
and exactly two committed data artifacts without touching any existing
result, so that the last deliverable in the repo cannot silently change
a number an earlier session already published.

#### Acceptance Criteria

1. THE Analysis_Writeup SHALL produce exactly one documentation
   deliverable: the Analysis_Document at the repository root.
2. THE Analysis_Writeup SHALL produce exactly two new files under
   `results/`: the Failure_Bucket_Report at
   `results/failure_buckets.csv` and the Failure_Bucket_Counts_Report at
   `results/failure_bucket_counts.csv`, and SHALL carry every
   Token_Length_Covariate as an additional column of the
   Failure_Bucket_Report rather than as a third file.
3. THE Analysis_Writeup SHALL leave the Sweep_Report, the
   Per_Query_Report, the Significance_Report, the Run_Config_Record,
   the Token_Length_Report, `results/groundedness.csv`,
   `results/generated_answers.csv`, `results/hand_checked_sample.csv`,
   `results/hand_checked_joined.csv`, and
   `results/hand_checked_sample_context.md` byte-for-byte identical to
   their pre-existing committed state.
4. THE Analysis_Writeup SHALL leave the Readme_Document and the
   Spec_Document byte-for-byte identical to their pre-existing committed
   state, except for an edit whose sole effect is to reference the
   Analysis_Document by filename.
5. THE Analysis_Writeup SHALL derive every Numeric_Claim in the
   Analysis_Document from a committed artifact, and SHALL NOT execute a
   retrieval run, an embedding computation, a generative-model
   invocation, a significance computation, or any network call in order
   to obtain one.
6. THE Analysis_Document SHALL state that the per-query Failure_Bucket
   label is carried in the Failure_Bucket_Report rather than as a
   column of the Sweep_Report, and SHALL state the reason: the
   Sweep_Report is keyed by (`run_id`, `k`) with no per-query dimension.
7. THE Analysis_Writeup SHALL NOT add a retriever, a Chunking_Strategy,
   an evaluation cutoff, or a metric to `configs/sweep.yaml`,
   `configs/significance.yaml`, or `configs/groundedness.yaml`.
8. THE Analysis_Writeup SHALL perform a tokenization pass only within
   the Covariate_Enrichment_Stage of Requirement 16, whose output is
   committed as columns of the Failure_Bucket_Report, so that the
   Analysis_Document reads a token count from a committed artifact rather
   than computing one.
9. THE Analysis_Writeup SHALL leave `requirements.txt` unchanged,
   because the Covariate_Enrichment_Stage uses only the already-pinned
   `transformers` and `sentence-transformers` entries and therefore
   introduces no new top-level dependency.

### Requirement 2: Bucket_Assigner Inputs, Entry Point, And Exit Behavior

**User Story:** As a researcher, I want the bucket taxonomy computed by
one committed entry point from a single committed artifact, with the
token-length covariates computed by a separate stage of that same entry
point, so that no bucket label can depend on a corpus and every
covariate still lands in a committed file.

#### Acceptance Criteria

1. THE Bucket_Assignment_Stage SHALL read the Per_Query_Report as its
   only data input, and SHALL NOT load the SciFact corpus, the Qrels, a
   tokenizer, an embedding model, or a generative model, so that no
   Failure_Bucket and no Contrast_Bucket label depends on a token count,
   a corpus document, or a model.
2. THE Bucket_Assigner SHALL execute using CPU-only code paths in both
   the Bucket_Assignment_Stage and the Covariate_Enrichment_Stage, and
   SHALL make no network call in either stage.
3. THE Analysis_Writeup SHALL expose the Bucket_Assigner through a
   single command-line entry point, invoked as
   `python -m src.failure_buckets`, that runs the
   Bucket_Assignment_Stage and the Covariate_Enrichment_Stage and writes
   both the Failure_Bucket_Report and the Failure_Bucket_Counts_Report
   in one invocation.
4. WHEN the Bucket_Assigner completes every Totality_Assertion and
   writes both reports, THEN THE Bucket_Assigner SHALL exit with status
   code 0.
5. IF the Per_Query_Report is absent, cannot be parsed as a CSV, or
   lacks any of the columns `run_id`, `retriever`, `chunking_strategy`,
   `query_id`, `recall_at_1`, `recall_at_20`, `ndcg_at_10`, or
   `num_judged_relevant`, THEN THE Bucket_Assigner SHALL raise an error
   naming the missing file or column, SHALL exit with a non-zero status
   code, and SHALL write neither the Failure_Bucket_Report nor the
   Failure_Bucket_Counts_Report.
6. THE Bucket_Assigner SHALL derive the set of Run_Ids, the set of
   `query_id` values, and every count it reports from the rows the
   Per_Query_Report actually contains, and SHALL NOT compare against,
   or substitute, a Run_Id count, a query count, or a row count written
   as a literal in its own source.
7. THE Bucket_Assigner SHALL apply no random sampling, no shuffling,
   and no time-dependent value, and SHALL rely on tokenization being
   deterministic for a fixed tokenizer revision, so that no seed is
   required for its output to be reproducible.
8. THE Covariate_Enrichment_Stage SHALL load the BEIR SciFact corpus,
   the Qrels, and each Dense_Model's tokenizer from the Local_Cache, and
   SHALL compute the Token_Length_Covariate values specified in
   Requirement 16 before either report is written.

### Requirement 3: The Four-Bucket Per-Run Failure Partition

**User Story:** As a researcher, I want each query labelled with exactly
one failure mode per run under a taxonomy fixed in advance, so that the
mechanism analysis rests on a partition nobody can reshape after seeing
which retriever it favors.

#### Acceptance Criteria

1. THE Bucket_Assigner SHALL assign exactly one Failure_Bucket to every
   (Run_Id, `query_id`) pair present in the Per_Query_Report, by
   evaluating the following four predicates in this exact order and
   assigning the first bucket whose predicate holds:
   (1) `total_miss` — `recall_at_20` is exactly 0;
   (2) `mis_ranked` — `recall_at_20` is strictly greater than 0 and
   `recall_at_1` is exactly 0;
   (3) `partial_recall` — `num_judged_relevant` is strictly greater
   than 1 and `recall_at_20` is strictly greater than 0 and strictly
   less than 1;
   (4) `full_success` — every (Run_Id, `query_id`) pair not assigned by
   predicates 1 through 3, which is exactly the set of pairs whose
   `recall_at_1` is strictly greater than 0 and whose `recall_at_20`
   is either exactly 1 or belongs to a query with a
   `num_judged_relevant` of 1.
2. THE Bucket_Assigner SHALL treat the four Failure_Bucket values as
   exhaustive and mutually exclusive by construction of the ordered
   first-match evaluation in Criterion 1: a pair satisfying an earlier
   predicate SHALL NOT be assigned a later bucket, and the
   `full_success` bucket SHALL absorb every pair no earlier predicate
   matched.
3. THE Bucket_Assigner SHALL evaluate the Criterion 1 predicates
   against the `recall_at_1`, `recall_at_20`, and `num_judged_relevant`
   values as parsed from the Per_Query_Report, comparing to 0 and to 1
   exactly rather than within a tolerance, because each recall value is
   a ratio of integer counts and therefore represents 0 and 1 exactly.
4. THE Bucket_Assigner SHALL define the four Failure_Bucket predicates
   as fixed constants in its own source, and SHALL NOT read a
   predicate, a threshold, or a bucket name from a configuration file,
   a command-line argument, or an environment variable, so that the
   taxonomy cannot be adjusted per run or after results are seen.
5. THE Analysis_Document SHALL state each of the four Failure_Bucket
   predicates in the same order and with the same meaning as Criterion
   1, so that a reader can reproduce the partition from the document
   alone.

### Requirement 4: Pair_Contrast Taxonomy And The Declared Contrast Set

**User Story:** As a researcher, I want per-query disagreement between
two runs bucketed under a fixed four-way taxonomy over a fixed set of
run pairs, so that "which queries does dense answer that BM25 misses"
is a committed number rather than an anecdote.

#### Acceptance Criteria

1. THE Bucket_Assigner SHALL classify a (Run_Id, `query_id`) pair as an
   Answered_Query when its `ndcg_at_10` value in the Per_Query_Report is
   strictly greater than 0, and as a Missed_Query when that value is
   exactly 0.
2. THE Bucket_Assigner SHALL assign exactly one Contrast_Bucket to
   every (Pair_Contrast, `query_id`) combination whose `query_id` is
   present for both Run_A and Run_B, by evaluating the following four
   mutually exclusive and exhaustive conditions:
   `a_only` — Run_A is an Answered_Query and Run_B is a Missed_Query;
   `b_only` — Run_B is an Answered_Query and Run_A is a Missed_Query;
   `both_miss` — Run_A and Run_B are both Missed_Query;
   `both_answer` — Run_A and Run_B are both Answered_Query.
3. THE Bucket_Assigner SHALL compute the Declared_Contrast_Set as
   exactly these 12 Pair_Contrasts, with Run_A and Run_B in the stated
   order:
   (a) the 8 family-aligned contrasts, each pairing the Reference_Run as
   Run_A with one of the 8 non-reference Run_Ids as Run_B, so that
   every Pair_Contrast in this group corresponds to exactly one row of
   the Pre_Declared_Family; and
   (b) the 4 dense cross-strategy contrasts, each pairing a dense
   retriever's `whole_document` Run_Id as Run_A with that same
   retriever's `fixed_window` Run_Id and with that same retriever's
   `sentence_window` Run_Id as Run_B — that is,
   `all-MiniLM-L6-v2__whole_document` against
   `all-MiniLM-L6-v2__fixed_window` and against
   `all-MiniLM-L6-v2__sentence_window`, and
   `bge-small-en-v1.5__whole_document` against
   `bge-small-en-v1.5__fixed_window` and against
   `bge-small-en-v1.5__sentence_window`.
4. THE Bucket_Assigner SHALL treat group (a) of Criterion 3 as
   supplying the BM25 cross-strategy contrasts as well, because the
   Reference_Run paired with `bm25__fixed_window` and with
   `bm25__sentence_window` already holds the retriever fixed while
   varying the Chunking_Strategy, and SHALL NOT emit a duplicate
   Pair_Contrast for either of those two pairs.
5. THE Bucket_Assigner SHALL emit each Pair_Contrast in the
   Declared_Contrast_Set exactly once, so that no two rows of the
   Failure_Bucket_Counts_Report share the same Composite_Run_Id and
   Contrast_Bucket.
6. IF a `query_id` is present for Run_A but absent for Run_B in a
   Pair_Contrast, or is present for Run_B but absent for Run_A, THEN
   THE Bucket_Assigner SHALL raise an error naming the Pair_Contrast
   and that `query_id`, SHALL exit with a non-zero status code, and
   SHALL write neither report, because a Pair_Contrast over an
   asymmetric query set has no total partition.

### Requirement 5: Totality Assertions With Hard Failure

**User Story:** As a maintainer, I want the script to prove the
partition is total before it writes anything, so that a bucket count
quoted in `ANALYSIS.md` cannot be silently missing queries.

#### Acceptance Criteria

1. WHEN the Bucket_Assigner has assigned a Failure_Bucket to every
   (Run_Id, `query_id`) pair, THEN THE Bucket_Assigner SHALL assert, for
   each Run_Id independently, that the sum of that Run_Id's four
   Failure_Bucket counts equals the number of distinct `query_id` values
   the Per_Query_Report holds for that Run_Id.
2. WHEN the Bucket_Assigner has assigned a Contrast_Bucket for every
   Pair_Contrast in the Declared_Contrast_Set, THEN THE Bucket_Assigner
   SHALL assert, for each Pair_Contrast independently, that the sum of
   that Pair_Contrast's four Contrast_Bucket counts equals the number of
   `query_id` values shared by Run_A and Run_B.
3. THE Bucket_Assigner SHALL assert that every (Run_Id, `query_id`) pair
   appears in exactly one row of the Failure_Bucket_Report, so that no
   pair is labelled twice and none is omitted.
4. THE Bucket_Assigner SHALL assert, for each `run_id` value in the
   Failure_Bucket_Counts_Report, that the sum of that value's four
   `fraction` entries as computed — the unrounded float values, before
   the 6-decimal rendering Requirement 7.7 applies — differs from 1 by
   no more than 1e-9.
5. IF any assertion required by Criteria 1 through 4 or by Criterion 7
   does not hold, THEN THE Bucket_Assigner SHALL raise an error naming
   the affected Run_Id or Pair_Contrast, the observed sum, and the
   expected value, SHALL exit with a non-zero status code, and SHALL write
   neither the Failure_Bucket_Report nor the
   Failure_Bucket_Counts_Report, leaving any pre-existing copy of either
   file byte-for-byte in its pre-run state.
6. THE Bucket_Assigner SHALL print, on a successful run, the number of
   Run_Ids processed, the number of `query_id` values per Run_Id, the
   number of Pair_Contrasts processed, and the total row count written
   to each report, so that a silent truncation of the Per_Query_Report
   is visible in the run's own output.
7. THE Bucket_Assigner SHALL assert, for each `run_id` value in the
   Failure_Bucket_Counts_Report, that the sum of that value's four
   `fraction` entries as rendered to exactly 6 digits after the decimal
   point per Requirement 7.7 and re-parsed as floats differs from 1 by
   no more than 2e-6, because rounding to 6 decimal places moves each of
   the four values by at most 5e-7 and therefore moves their sum by at
   most 4 x 5e-7 = 2e-6; and SHALL run this assertion in addition to
   Criterion 4's assertion rather than in place of it, so that neither
   the unrounded check nor the rendered check is skipped.

### Requirement 6: Failure_Bucket_Report Schema And Deterministic Serialization

**User Story:** As a maintainer, I want the per-query bucket artifact to
have a fixed schema, including the token-length covariate columns, and a
byte-identical rerun, so that a regenerated file produces an empty diff
rather than a review burden and every covariate figure has a committed
home.

#### Acceptance Criteria

1. THE Failure_Bucket_Report SHALL contain exactly the following twelve
   columns, in this order, with one header row: `run_id`, `retriever`,
   `chunking_strategy`, `query_id`, `bucket`, `num_judged_relevant`,
   `query_token_len__all-MiniLM-L6-v2`,
   `max_relevant_doc_token_len__all-MiniLM-L6-v2`,
   `any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2`,
   `query_token_len__bge-small-en-v1_5`,
   `max_relevant_doc_token_len__bge-small-en-v1_5`, and
   `any_relevant_doc_exceeds_limit__bge-small-en-v1_5`.
2. THE Failure_Bucket_Report SHALL contain exactly one data row per
   (Run_Id, `query_id`) pair present in the Per_Query_Report, and SHALL
   carry in each row's `bucket` column exactly one of the four
   Failure_Bucket values assigned by Requirement 3.
3. THE Failure_Bucket_Report SHALL carry each row's `retriever`,
   `chunking_strategy`, `query_id`, and `num_judged_relevant` values
   copied unchanged from the corresponding Per_Query_Report row.
4. THE Bucket_Assigner SHALL order the Failure_Bucket_Report's data rows
   by `run_id` ascending in lexicographic order of the column's text,
   then by `query_id` ascending in lexicographic order of the column's
   text, and SHALL write no row index column.
5. WHEN the Bucket_Assigner is invoked twice on the same
   Per_Query_Report, the same Local_Cache corpus, and the same
   Dense_Model tokenizer revisions, THEN THE Failure_Bucket_Report
   written by the second invocation SHALL be byte-for-byte identical to
   the one written by the first, including every covariate column.
6. THE Bucket_Assigner SHALL name each covariate column
   `{covariate}__{model_tag}`, where `{covariate}` is one of
   `query_token_len`, `max_relevant_doc_token_len`, or
   `any_relevant_doc_exceeds_limit`, and `{model_tag}` is the
   Dense_Model's retriever name as it appears in the Per_Query_Report's
   `retriever` column with every `.` character replaced by `_`, so that
   `bge-small-en-v1.5` yields the tag `bge-small-en-v1_5`.
7. THE Failure_Bucket_Report SHALL carry no column name containing a `.`
   character, because the Verifier separates a ledger row's field name
   from its row selector at the last `.` of the reference text, so a `.`
   inside a column name would make that split resolve the wrong field.
   A `.` inside a column *value* is unaffected and remains permitted.
8. THE Bucket_Assigner SHALL write each `query_token_len__*` and
   `max_relevant_doc_token_len__*` value as a base-ten integer with no
   decimal point, each `any_relevant_doc_exceeds_limit__*` value as the
   literal text `exceeds` or `within`, and any covariate value with no
   defined number as the Missing_Value_Sentinel. THE Bucket_Assigner
   SHALL NOT write `true`, `false`, or any other token a CSV-reading
   library would infer as a boolean dtype into an
   `any_relevant_doc_exceeds_limit__*` cell, because a column holding
   only `true`/`false` text is inferred as a boolean column when no row
   of that column also holds the Missing_Value_Sentinel, and an
   `astype(str)` read of a boolean column yields `'True'`/`'False'`
   rather than the `'true'`/`'false'` text the file itself contains —
   making the correct ledger filter literal depend on whether some
   other row in the same column happens to be missing, a property no
   filter names. `exceeds` and `within` are not coercible to a boolean
   or numeric dtype, so a column of only those two values is read back
   as text regardless of whether the Missing_Value_Sentinel is present
   elsewhere in that column, and `astype(str)` is the identity on it.
9. THE Bucket_Assigner SHALL write, for a given `query_id`, the same six
   covariate values in every one of that `query_id`'s rows, because each
   Token_Length_Covariate is run-independent as established by
   Requirement 16.

### Requirement 7: Failure_Bucket_Counts_Report Schema And Unique Selectors

**User Story:** As a maintainer, I want every bucket figure in
`ANALYSIS.md` to resolve through a `run_id`-plus-`bucket` selector that
matches exactly one row, so that the existing verifier can check it
without any change to its resolver.

#### Acceptance Criteria

1. THE Failure_Bucket_Counts_Report SHALL contain exactly the columns
   `run_id`, `bucket`, `count`, and `fraction`, in that order, with one
   header row.
2. THE Failure_Bucket_Counts_Report SHALL contain one data row for each
   (Run_Id, Failure_Bucket) combination, carrying that bucket's query
   count and that count divided by the number of distinct `query_id`
   values the Per_Query_Report holds for that Run_Id.
3. THE Failure_Bucket_Counts_Report SHALL contain one data row for each
   (Pair_Contrast, Contrast_Bucket) combination, with the Pair_Contrast
   identified in the `run_id` column by its Composite_Run_Id formed as
   `{Run_A}|vs|{Run_B}`, carrying that bucket's query count and that
   count divided by the number of `query_id` values shared by Run_A and
   Run_B.
4. THE Bucket_Assigner SHALL emit no two rows of the
   Failure_Bucket_Counts_Report sharing the same combination of `run_id`
   and `bucket` value, so that a `run_id={value},bucket={value}`
   selector resolves to exactly one row under the Verifier's existing
   exact-match row-selector semantics.
5. THE Bucket_Assigner SHALL use the four Failure_Bucket names
   (`total_miss`, `mis_ranked`, `partial_recall`, `full_success`) only
   on rows whose `run_id` is a Run_Id, and the four Contrast_Bucket
   names (`a_only`, `b_only`, `both_miss`, `both_answer`) only on rows
   whose `run_id` is a Composite_Run_Id, keeping the two name sets
   disjoint so that a mistyped selector resolves to zero rows and fails
   loudly.
6. THE Bucket_Assigner SHALL assert that the four-character separator
   sequence `|vs|` occurs in no Run_Id read from the Per_Query_Report,
   and IF that sequence does occur in a Run_Id, THEN THE
   Bucket_Assigner SHALL raise an error naming that Run_Id, SHALL exit
   with a non-zero status code, and SHALL write neither report, because
   a Composite_Run_Id could otherwise collide with a Run_Id.
7. THE Bucket_Assigner SHALL write every `count` value as a base-ten
   integer with no decimal point, and every `fraction` value as a
   fixed-point decimal with exactly 6 digits after the decimal point,
   so that the file's float formatting does not vary between
   invocations.
8. THE Bucket_Assigner SHALL order the Failure_Bucket_Counts_Report's
   data rows with every Run_Id row before every Composite_Run_Id row,
   within each of those two groups by `run_id` ascending in
   lexicographic order of the column's text, and within each `run_id`
   by the declared bucket order — `total_miss`, `mis_ranked`,
   `partial_recall`, `full_success` for a Run_Id row, and `a_only`,
   `b_only`, `both_miss`, `both_answer` for a Composite_Run_Id row —
   and SHALL write no row index column.
9. WHEN the Bucket_Assigner is invoked twice on the same
   Per_Query_Report, THEN THE Failure_Bucket_Counts_Report written by
   the second invocation SHALL be byte-for-byte identical to the one
   written by the first.

### Requirement 8: Verifier Registration Without A Resolver Change

**User Story:** As a maintainer, I want the two new artifacts readable by
the existing verifier through the smallest possible edit, so that the
verification machinery gains coverage without gaining behavior.

#### Acceptance Criteria

1. THE Analysis_Writeup SHALL add the two string values
   `failure_buckets.csv` and `failure_bucket_counts.csv` to the
   Verifier's `_CSV_ARTIFACTS` tuple.
2. THE Analysis_Writeup SHALL leave the Verifier's
   `_resolve_csv_reference`, `_resolve_column_equality_reference`,
   `_resolve_json_path`, `_resolve_top_level_key`,
   `load_artifact_values`, `load_ledger`, `verify_row`,
   `apply_computation`, `round_half_up`, and
   `stated_value_matches_precision` functions unchanged in behavior,
   and SHALL NOT add a member to `_ALLOWED_COMPUTATIONS`.
3. THE Analysis_Writeup SHALL update the Verifier's module docstring so
   that it names the Analysis_Document alongside the Readme_Document and
   the Spec_Document as a document whose Numeric_Claims the Verifier
   checks.
4. THE Analysis_Writeup SHALL update the Verifier's argparse
   `description` text, shown by `--help`, so that it names the
   Analysis_Document alongside the Readme_Document and the
   Spec_Document.
5. THE Analysis_Writeup SHALL rely on the Verifier's existing
   resolution of a ledger row's `document` column as a
   repository-root-relative path, and SHALL NOT add a document allowlist,
   a document-name branch, or a per-document code path to the Verifier.

### Requirement 9: Traceability Ledger Coverage For The Analysis_Document

**User Story:** As a maintainer safeguarding the "no number without a
receipt" rule, I want every number in `ANALYSIS.md` carried by its own
ledger row and checked by the verifier, so that the analysis cannot
drift from the artifacts it describes.

#### Acceptance Criteria

1. THE Analysis_Writeup SHALL add, for every Numeric_Claim in the
   Analysis_Document, exactly one Traceability_Ledger row whose
   `document` value is the literal text `ANALYSIS.md`.
2. THE Analysis_Writeup SHALL set each added row's `source_artifact`
   value to one of `failure_bucket_counts.csv`, `failure_buckets.csv`,
   `sweep.csv`, `per_query.csv`, `significance.csv`, `groundedness.csv`,
   `hand_checked_joined.csv`, `generated_answers.csv`,
   `run_config.json`, or `token_length_report.json`.
3. THE Analysis_Writeup SHALL set each added row's `computation` value
   to a member of the Verifier's existing `_ALLOWED_COMPUTATIONS`
   tuple.
4. THE Analysis_Writeup SHALL preserve the 167 Traceability_Ledger rows
   whose `document` value is `README.md` or `SPEC.md` unchanged in
   content and in relative order, adding the Analysis_Document's rows
   after them.
5. WHEN the Analysis_Document and both new artifacts are written, THEN
   THE Analysis_Writeup SHALL run a Verification_Pass over the whole
   Traceability_Ledger and SHALL obtain an exit status of 0, with every
   row reported as a match.
6. IF a Verification_Pass reports a mismatch or an error for any row,
   THEN THE Analysis_Writeup SHALL correct the Analysis_Document's
   stated text, the ledger row's selector, or the ledger row's declared
   precision, and SHALL NOT edit any value inside the Sweep_Report, the
   Per_Query_Report, the Significance_Report, the Run_Config_Record, or
   the Token_Length_Report to resolve the mismatch.
7. IF a number the Analysis_Document would state cannot be resolved
   from a committed artifact through a single ledger row, THEN THE
   Analysis_Writeup SHALL remove that number from the
   Analysis_Document, rather than stating it without a ledger row.

### Requirement 10: Reference Row, Primary Metric, And A Mechanism For Each Grid Result

**User Story:** As a reader of the analysis, I want BM25 as the
reference row, nDCG@10 as the primary metric, and each of the three
things the grid actually showed either explained through the buckets or
explicitly marked unexplained, so that the write-up neither invents a
mechanism nor quietly skips a result.

#### Acceptance Criteria

1. THE Analysis_Document SHALL present the Reference_Run as the
   reference row in every results table it contains, and SHALL report
   every non-reference Run_Id's Primary_Metric result as a delta against
   the Reference_Run, read from the `mean_diff` value of that Run_Id's
   Pre_Declared_Family row.
2. THE Analysis_Document SHALL state that nDCG@10 is the single primary
   metric, designated before any result existed, and that recall@k and
   MRR@10 are secondary.
3. THE Analysis_Document SHALL address the result that
   `bge-small-en-v1.5` exceeds the Reference_Run on the Primary_Metric
   under each of the three Chunking_Strategy values, citing the
   `mean_diff`, `p_value_adjusted`, and `verdict` values of the three
   corresponding Pre_Declared_Family rows, and SHALL accompany that
   result with either a mechanism grounded in named Failure_Bucket or
   Contrast_Bucket figures read from the Failure_Bucket_Counts_Report,
   or the explicit statement that no mechanism was identified.
4. THE Analysis_Document SHALL address the result that
   `bm25__sentence_window` falls below the Reference_Run on the
   Primary_Metric, citing that comparison's `mean_diff`,
   `p_value_adjusted`, and `verdict` values from its
   Pre_Declared_Family row, and SHALL accompany that result with either
   a mechanism grounded in named Failure_Bucket or Contrast_Bucket
   figures read from the Failure_Bucket_Counts_Report, or the explicit
   statement that no mechanism was identified.
5. THE Analysis_Document SHALL address the result that
   `all-MiniLM-L6-v2` is indistinguishable from the Reference_Run on
   the Primary_Metric under each of the three Chunking_Strategy values,
   citing the `verdict` value of the three corresponding
   Pre_Declared_Family rows.
6. WHERE the Analysis_Document states a mechanism for a result, THE
   Analysis_Document SHALL cite at least one Failure_Bucket or
   Contrast_Bucket figure from the Failure_Bucket_Counts_Report in
   support of that mechanism, so that no mechanism rests on unsupported
   narration.
7. THE Analysis_Document SHALL report every Pre_Declared_Family row's
   verdict it discusses as that row records it, and SHALL NOT omit,
   soften, or relegate a result because it is unfavorable to a dense
   retriever, to BM25, or to the project.
8. THE Analysis_Document SHALL report every Pair_Contrast in the
   Declared_Contrast_Set, including each Pair_Contrast whose counts turn
   out to be unremarkable, and SHALL NOT omit a Pair_Contrast because its
   figures are uninteresting or unflattering — the document-side
   application of `.kiro/steering/evaluation-integrity.md`'s "no row gets
   dropped for being unflattering" rule, and the reason the
   Declared_Contrast_Set is declared in Requirement 4 before any count
   exists rather than selected after the counts are known.

### Requirement 11: Indistinguishable Verdicts Narrated As Indistinguishable

**User Story:** As a reader assessing what the study established, I want
every comparison the study could not distinguish from noise described
that way and left unexplained, so that no mechanism is invented for a
difference that may not exist.

#### Acceptance Criteria

1. WHERE the Analysis_Document discusses a Pre_Declared_Family
   comparison, IF that row's `verdict` value is `indistinguishable`,
   THEN THE Analysis_Document SHALL describe that comparison as
   indistinguishable from noise.
2. WHERE the Analysis_Document discusses a Pre_Declared_Family
   comparison, IF that row's `verdict` value is `indistinguishable`,
   THEN THE Analysis_Document SHALL NOT state a mechanism, a cause, an
   explanation, a failure-bucket account, or a Token_Length_Covariate
   account for the direction of that comparison's `mean_diff` — the
   covariate columns Requirement 16 commits license a description of
   those queries, never a mechanism for a comparison the study could not
   distinguish from noise.
3. THE Analysis_Document SHALL describe the four comparisons whose
   Pre_Declared_Family `verdict` is `indistinguishable` — the three
   `all-MiniLM-L6-v2` runs and `bm25__fixed_window`, each against the
   Reference_Run — as indistinguishable, and SHALL NOT describe any of
   them as a win for either compared run.
4. THE Analysis_Document SHALL NOT restate an `indistinguishable`
   verdict as evidence that no difference exists between the compared
   runs.
5. WHERE the Analysis_Document reports a Failure_Bucket or
   Contrast_Bucket figure for a comparison whose Pre_Declared_Family
   `verdict` is `indistinguishable`, THE Analysis_Document SHALL
   present that figure as a description of where the two runs disagreed
   per query, together with the statement that the aggregate
   Primary_Metric difference between those runs is indistinguishable
   from noise.

### Requirement 12: Descriptive-Only Status Of Every Contrast This Spec Introduces

**User Story:** As a researcher, I want every new comparison labelled as
descriptive and outside the pre-declared family, so that the analysis
adds explanation without smuggling in untested inferential claims.

#### Acceptance Criteria

1. THE Analysis_Document SHALL state, in prose, that the
   Pre_Declared_Family — the 8 nDCG@10 comparisons against the
   Reference_Run under Holm-Bonferroni correction recorded in the
   Significance_Report — is the complete set of inferential results the
   study supports.
2. THE Analysis_Document SHALL label every Descriptive_Contrast it
   reports, in prose, as descriptive and outside the Pre_Declared_Family,
   carrying no inferential claim.
3. THE Analysis_Document SHALL NOT state a p-value, a confidence
   interval, a standard error, a test statistic, or a
   statistical-significance determination for any Descriptive_Contrast.
4. THE Analysis_Document SHALL state every p-value, confidence-interval
   bound, and verdict it reports as a value read from a
   Significance_Report row, and SHALL NOT compute a new one.
5. WHERE the Analysis_Document compares a Failure_Bucket's
   Judged_Relevant_Count distribution or Token_Length_Covariate
   distribution against the corresponding corpus-wide distribution, THE
   Analysis_Document SHALL present that comparison as a description of
   the two distributions, and SHALL NOT present it as a hypothesis test
   or as evidence of a distributional difference beyond the queries
   measured.
6. THE Analysis_Document SHALL derive every per-query token-length
   covariate figure it states from a named column of the
   Failure_Bucket_Report, SHALL cite that column, and SHALL NOT state a
   per-query token-length figure that no committed artifact column
   carries.
7. WHERE the Analysis_Document states that one Run_Id's Missed_Queries
   are disproportionately the queries whose Judged_Relevant_Documents
   exceed that Dense_Model's Effective_Max_Sequence_Length, THE
   Analysis_Document SHALL read every figure supporting that statement
   from a named Failure_Bucket_Report covariate column, and SHALL present
   the statement as a Descriptive_Contrast carrying no inferential claim.
8. WHERE a Failure_Bucket's covariate distribution does not separate that
   bucket from the remaining queries, THE Analysis_Document SHALL state
   that no mechanism was identified for that result, and SHALL NOT
   substitute a narrative account for the absent separation.
9. THE Analysis_Document SHALL treat a covariate column as licensing a
   description only, and SHALL NOT state a mechanism, a cause, or a
   covariate-grounded explanation for a comparison whose
   Pre_Declared_Family `verdict` is `indistinguishable`, so that
   Requirement 11's prohibition holds with the covariate columns present
   exactly as it held without them.

### Requirement 13: Truncation Evidence And The Limits Section

**User Story:** As a reader deciding how far to trust the mechanism
account, I want the truncation evidence sourced from the committed
token-length artifact and a limits section that states plainly what the
buckets cannot establish, so that a description is not read as a causal
finding.

#### Acceptance Criteria

1. WHERE the Analysis_Document discusses corpus-level truncation, THE
   Analysis_Document SHALL read every corpus-level truncation figure it
   states from the Token_Length_Report's `max_sequence_length`,
   `num_documents_total`, `num_documents_exceeding`,
   `fraction_exceeding`, and `cells` entries, SHALL read every per-query
   truncation figure it states from a named Failure_Bucket_Report
   covariate column, and SHALL NOT itself compute a token count.
2. THE Analysis_Document SHALL describe every Token_Length_Report figure
   as a corpus-level property measured per (Chunking_Strategy,
   Dense_Model) cell, and SHALL NOT attribute a Token_Length_Report
   figure to an individual query or to an individual Failure_Bucket;
   per-query and per-bucket truncation statements SHALL rest on the
   Failure_Bucket_Report's covariate columns instead.
3. THE Analysis_Document SHALL include a section stating the limits of
   its own analysis.
4. THE Analysis_Document SHALL state, within that section, that the
   Failure_Bucket taxonomy was fixed before assignment and that its
   counts describe the partition rather than test a hypothesis about
   it.
5. THE Analysis_Document SHALL state, within that section, that sparse
   Qrels mean an unjudged document is scored as a miss, so that a
   `total_miss` bucket assignment records the absence of a judged
   relevant document in the ranked list rather than the absence of any
   useful document.
6. THE Analysis_Document SHALL state, within that section, that every
   reported number describes BEIR SciFact only and may not transfer to
   another corpus or domain.
7. THE Analysis_Document SHALL state, within that section, that each
   Token_Length_Covariate is measured over the source corpus document's
   own `title` and `text` content rather than over the Chunk a
   `fixed_window` or `sentence_window` run actually encoded, so a
   covariate value describes the document a query's judged-relevant
   evidence lives in rather than the unit a given run indexed.
8. THE Analysis_Document SHALL state, within that section, that a
   mechanism it offers is an account consistent with the bucket figures
   and the covariate columns rather than a causal result established by
   the study.

### Requirement 14: Prohibited-Scope Content Excluded From The Analysis_Document

**User Story:** As a maintainer guarding against scope creep, I want the
analysis to avoid proposing out-of-scope machinery even as future work,
so that the last document in the repo does not reopen a closed question.

#### Acceptance Criteria

1. THE Analysis_Document SHALL NOT describe hybrid retrieval, score
   fusion, or reciprocal-rank fusion as work to build, propose, or
   recommend.
2. THE Analysis_Document SHALL NOT describe cross-encoder reranking or
   language-model reranking of retrieved results as work to build,
   propose, or recommend.
3. THE Analysis_Document SHALL NOT describe a retriever beyond `bm25`,
   `all-MiniLM-L6-v2`, and `bge-small-en-v1.5` as work to build,
   propose, or recommend.
4. THE Analysis_Document SHALL NOT describe query expansion, query
   rewriting, generated pseudo-queries, an approximate nearest
   neighbour index, or fine-tuning of any model as work to build,
   propose, or recommend.
5. WHERE the Analysis_Document names a question its data cannot answer,
   THE Analysis_Document SHALL state that question as a limit of this
   study rather than as a proposal to build one of the items named in
   Criteria 1 through 4.

### Requirement 15: Network-Free Test Coverage And Unchanged CI

**User Story:** As a maintainer, I want the bucket assignment logic
covered by fast fixture-based tests that make no network call, so that
CI keeps catching regressions without downloading a dataset or a model.

#### Acceptance Criteria

1. THE test suite SHALL include a pytest test that applies the
   Failure_Bucket predicates of Requirement 3 to an in-memory fixture
   frame and asserts the expected bucket for at least one query in each
   of the four Failure_Bucket values.
2. THE test suite SHALL include a pytest test that applies the
   Contrast_Bucket rules of Requirement 4 to an in-memory fixture frame
   containing two Run_Ids and asserts the expected bucket for at least
   one query in each of the four Contrast_Bucket values.
3. THE test suite SHALL include a pytest test that asserts the
   Bucket_Assigner raises an error and writes neither report when a
   Totality_Assertion of Requirement 5 does not hold.
4. THE test suite SHALL include a pytest test that asserts two
   successive Bucket_Assigner invocations over the same fixture input
   produce byte-for-byte identical Failure_Bucket_Report and
   Failure_Bucket_Counts_Report contents.
5. THE test suite SHALL include a pytest test that asserts every
   `run_id`-and-`bucket` combination written to the
   Failure_Bucket_Counts_Report for a fixture input is unique.
6. THE test suite's tests for the Bucket_Assigner SHALL use in-memory
   frames or fixture files of no more than 40 rows, SHALL make no
   network call, SHALL load no model, and SHALL read no file under
   `data/`.
7. THE Analysis_Writeup SHALL leave `.github/workflows/ci.yml`
   unchanged, so that CI continues to install `requirements.txt` and run
   `pytest` only, without running the Bucket_Assigner, the Verifier, the
   sweep, or any download.
8. THE test suite SHALL include a pytest test that exercises the
   Covariate_Enrichment_Stage's covariate computation against a stub
   corpus of no more than 5 documents, a stub Qrels mapping, and a
   hand-written stub tokenizer, following the hand-written-stub pattern
   `tests/test_chunking.py` already establishes, and SHALL assert the
   expected `query_token_len`, `max_relevant_doc_token_len`, and
   `any_relevant_doc_exceeds_limit` value for at least one query whose
   stub Qrels entry names a Judged_Relevant_Document and for at least one
   query whose stub Qrels entry names none.
9. THE test suite SHALL include a pytest test that asserts the
   Covariate_Enrichment_Stage writes the Missing_Value_Sentinel, rather
   than a numeric 0, into `max_relevant_doc_token_len__*` and
   `any_relevant_doc_exceeds_limit__*` for a stub query with no
   Judged_Relevant_Document.
10. THE Analysis_Writeup SHALL add no test that loads a real
    Dense_Model, loads a real tokenizer, reads the real BEIR SciFact
    corpus, reads any file under `data/`, or makes a network call, and
    SHALL add no new skip-gated real-corpus or real-tokenizer test, so
    that the test suite's existing network-free guarantee is unchanged.
11. THE Verification_Pass SHALL resolve every covariate Numeric_Claim
    from the committed Failure_Bucket_Report, and SHALL load no corpus,
    no Qrels, no tokenizer, and no model.
### Requirement 16: Per-Query Token-Length Covariates Committed To The Failure_Bucket_Report

**User Story:** As a researcher, I want each query's own token length and
its judged-relevant documents' maximum token length committed as columns
of a results artifact, so that a query-level truncation mechanism can be
stated with a receipt instead of being ruled out for lack of one.

#### Acceptance Criteria

1. THE Covariate_Enrichment_Stage SHALL compute, for each `query_id`
   present in the Per_Query_Report and for each Dense_Model, that
   query's own token length under that Dense_Model's tokenizer.
2. THE Covariate_Enrichment_Stage SHALL compute, for each `query_id`
   present in the Per_Query_Report and for each Dense_Model, the maximum
   token length over that query's Judged_Relevant_Documents under that
   Dense_Model's tokenizer.
3. THE Covariate_Enrichment_Stage SHALL compute, for each `query_id`
   present in the Per_Query_Report and for each Dense_Model, a boolean
   recording whether at least one of that query's
   Judged_Relevant_Documents has a token length strictly greater than
   that Dense_Model's Effective_Max_Sequence_Length; this Criterion
   fixes only that semantic value, and Requirement 6.8 governs the
   literal text the Bucket_Assigner writes to represent it in the
   Failure_Bucket_Report.
4. THE Covariate_Enrichment_Stage SHALL count tokens without truncation
   and including the special tokens the tokenizer inserts, matching the
   committed `src/token_length_analysis.py` `count_tokens` function, so
   that a document longer than the limit reports its true length rather
   than the limit.
5. THE Covariate_Enrichment_Stage SHALL compose each document's text as
   that document's `title` value, a single space, and its `text` value —
   the same composition the committed `format_document_text` function
   applies — and SHALL measure the source corpus document rather than
   any Chunk.
6. THE Covariate_Enrichment_Stage SHALL resolve each Dense_Model's
   Effective_Max_Sequence_Length from that Dense_Model's own loaded
   tokenizer and cached configuration, through the committed
   `src/token_length_analysis.py`
   `resolve_effective_max_sequence_length` function, and SHALL NOT read
   that limit from a literal in its own source, from a configuration
   file, from a command-line argument, or from an environment variable,
   because `all-MiniLM-L6-v2` and `bge-small-en-v1.5` do not share a
   limit and `.kiro/steering/evaluation-integrity.md` forbids a
   typed-in configuration number.
7. THE Covariate_Enrichment_Stage SHALL determine
   Judged_Relevant_Document membership from the loaded Qrels alone,
   applying the strictly-greater-than-0 relevance score condition the
   committed `src/metrics.py` `judged_relevant_docs` function applies,
   and SHALL NOT consult a retrieval result, a model score, a heuristic,
   or any other source of relevance.
8. IF a `query_id` has no Judged_Relevant_Document in the loaded Qrels,
   THEN THE Covariate_Enrichment_Stage SHALL record the
   Missing_Value_Sentinel as that query's
   `max_relevant_doc_token_len__*` and
   `any_relevant_doc_exceeds_limit__*` values for both Dense_Models,
   rather than a numeric 0, so that an absent judgment stays
   distinguishable from a measured zero.
9. THE Covariate_Enrichment_Stage SHALL compute each
   Token_Length_Covariate from the query text, the Qrels, the corpus
   documents, and the Dense_Model's tokenizer alone, so that all six
   values are run-independent and identical across every
   Failure_Bucket_Report row carrying the same `query_id`, and SHALL NOT
   let a Run_Id, a retriever, or a Chunking_Strategy change a covariate
   value.
10. THE Covariate_Enrichment_Stage SHALL resolve the Local_Cache path
    from `configs/sweep.yaml`'s `data_dir` field through the committed
    `load_sweep_config` function and SHALL call the committed
    `configure_caches` function on that path before loading any
    tokenizer, so that `HF_HOME` and `HF_HUB_CACHE` resolve under
    `data/` per `.kiro/steering/tech.md` and nothing downloads to a
    location outside the repository; this configuration read supplies a
    cache path only and SHALL NOT supply a predicate, a threshold, or a
    bucket name, which Requirement 3.4 keeps fixed in source.
11. THE Covariate_Enrichment_Stage SHALL load each Dense_Model's
    tokenizer through the committed `src/token_length_analysis.py`
    `load_tokenizer_offline` function, which sets `HF_HUB_OFFLINE` and
    `TRANSFORMERS_OFFLINE` and passes `local_files_only=True`, and SHALL
    execute CPU-only code paths and invoke no metered inference
    endpoint.
12. THE Covariate_Enrichment_Stage SHALL confirm that the BEIR SciFact
    dataset directory and each Dense_Model's tokenizer snapshot
    directory are already present under the Local_Cache before it calls
    the corpus loader, because the committed `load_scifact` function
    downloads the dataset when the Local_Cache lacks it.
13. IF the BEIR SciFact corpus is absent from the Local_Cache, the Qrels
    are absent from the Local_Cache, a Dense_Model's tokenizer snapshot
    is absent from the Local_Cache, or a `query_id` present in the
    Per_Query_Report is absent from the loaded query set, THEN THE
    Bucket_Assigner SHALL raise an error naming which of those inputs
    was missing or unresolvable, SHALL exit with a non-zero status code,
    and SHALL write neither the Failure_Bucket_Report nor the
    Failure_Bucket_Counts_Report, leaving any pre-existing copy of
    either file byte-for-byte in its pre-run state.
14. WHILE a Local_Cache input required by Criterion 13 is missing, THE
    Covariate_Enrichment_Stage SHALL write no covariate column as empty
    text, SHALL substitute no default token length, and SHALL download
    no corpus, tokenizer, or model.
15. THE Analysis_Writeup SHALL treat the four conditions of Criterion 13
    as one distinguishable failure condition — an unavailable or
    unresolvable covariate input — served by one error type, so that no
    error type exists for a failure raised from the same place as
    another.
16. WHEN the Covariate_Enrichment_Stage is run twice over the same
    Local_Cache corpus and the same Dense_Model tokenizer revisions,
    THEN THE covariate columns written by the second run SHALL be
    byte-for-byte identical to those written by the first, and THE
    Covariate_Enrichment_Stage SHALL require no seed, because
    tokenization is deterministic for a fixed tokenizer revision.
17. THE Covariate_Enrichment_Stage SHALL leave the
    Failure_Bucket_Counts_Report's four-column schema of Requirement 7.1
    unchanged, so that no Token_Length_Covariate is aggregated into a
    counts row.
18. THE Covariate_Enrichment_Stage SHALL print, on a successful run,
    each Dense_Model's resolved Effective_Max_Sequence_Length, the
    number of `query_id` values whose covariates were computed, and the
    number of `query_id` values recorded with the Missing_Value_Sentinel,
    so that a partially loaded corpus or an empty Qrels set is visible in
    the run's own output.
19. THE Bucket_Assigner SHALL confirm, for every `query_id` present in
    the committed Per_Query_Report, that the number of
    Judged_Relevant_Documents the Covariate_Enrichment_Stage loads for
    that `query_id` equals that `query_id`'s committed
    `num_judged_relevant` value exactly, and SHALL NOT write the
    Failure_Bucket_Report or the Failure_Bucket_Counts_Report if any
    `query_id` disagrees or if fewer than the full set of `query_id`
    values present in the Per_Query_Report is compared, because this is
    the check that proves the Qrels the Covariate_Enrichment_Stage loads
    are the same Qrels the committed Primary_Metric and secondary-metric
    values were scored against, rather than merely a Qrels file that
    happens to leave every query with at least one judged-relevant
    document.
