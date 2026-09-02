"""Exception types shared across `rag-retrieval-sweep` modules.

These exist so the Sweep_Runner can distinguish failures that must halt
the whole pipeline (no `results/sweep.csv` written at all) from failures
that are recoverable at the level of a single retriever's run_id or a
single row's metric cell (recorded as an `"NA"` missing-value marker
instead). See `design.md`'s "Error Handling" table for the full mapping
from each exception type to Sweep_Runner behavior and requirement.
"""


class ConfigError(Exception):
    """Sweep_Config missing, unparsable, or declares something unsupported."""


class UnsupportedPreprocessingError(ConfigError):
    """BM25 preprocessing setting not supported by BM25_Retriever."""


class CorpusLoadError(Exception):
    """Corpus, queries, or qrels failed to load, or loaded empty."""


class CorpusValidationError(Exception):
    """Qrels reference a document ID or query ID not present in the loaded data."""


class SeedApplicationError(Exception):
    """Applying the fixed seed to random/numpy/torch failed."""


class ModelLoadError(Exception):
    """Dense retriever model weights failed to load."""


class RetrievalError(Exception):
    """A specific retriever's index build or retrieval run failed."""


class MetricComputationError(Exception):
    """A specific metric computation failed for a specific row."""


class ZeroQualifyingQueriesError(Exception):
    """No loaded query has at least one Qrels-judged relevant document,
    so a mean over qualifying queries has no defined value anywhere in
    the run. Not a MetricComputationError subclass: this is a
    pre-indexing halt condition, detected once in Sweep_Runner step 5
    from corpus/qrels data alone (the same tier as Requirement 1.5's
    empty-corpus/empty-qrels check), not a per-cell recoverable failure
    -- a run with this condition never reaches the retriever loop, so
    there is no run_id or row to mark "NA".
    """


class ReportWriteError(Exception):
    """results/sweep.csv could not be written."""


# --- significance-testing spec: extends the session-1 hierarchy above ---


class PerQueryReportError(Exception):
    """results/per_query.csv could not be written by the Sweep_Runner.
    The sweep-side analogue of ReportWriteError: a halt condition for
    the sweep run (Requirement 1.8), never a per-cell recovery."""


class BootstrapConfigError(ConfigError):
    """Bootstrap_Config (configs/significance.yaml) missing, unparsable,
    or declaring a missing / non-integer resample_count,
    permutation_count, or bootstrap_seed, or an invalid alpha /
    reference retriever / path (Requirement 4.5). A ConfigError subclass
    so the Significance_Analyzer's config-failure contract matches
    load_sweep_config's."""


class SignificanceInputError(Exception):
    """results/per_query.csv is missing, cannot be parsed, or lacks a
    column required by Requirement 1.3 (Requirement 2.4). Halts the
    analyzer before it writes results/significance.csv."""


class MissingReferenceRunError(Exception):
    """The Per_Query_Report contains no run identified as the BM25
    Reference_Run (Requirement 2.5). Every comparison is defined
    relative to the Reference_Run, so the analyzer halts."""


class RunConfigMergeError(Exception):
    """results/run_config.json is absent, unparsable, or could not be
    re-written after merging the 'significance' sub-object (Requirement
    4.6). The analyzer halts rather than creating a fresh record that
    would lack the Sweep_Runner's own keys."""


class SignificanceWriteError(Exception):
    """results/significance.csv could not be written (Requirement 2.7).
    The analyzer's analogue of ReportWriteError."""


# --- repo-writeup spec: extends the hierarchy above ---


class TokenizerLoadError(Exception):
    """The all-MiniLM-L6-v2 tokenizer could not be loaded from the local
    cache under data/ without making a network call (Requirement 11.7).
    Never retried without the offline flags; no network call is ever
    attempted after this is raised."""


class TokenLengthReportError(Exception):
    """results/token_length_report.json could not be written
    (Requirement 11.4). The Token_Length_Analysis's analogue of
    ReportWriteError."""


class TraceabilityFileError(Exception):
    """docs/numeric_traceability.csv is missing, cannot be parsed,
    lacks a required column, or contains a row whose stated_value text
    is not formatted consistent with its own declared stated_precision
    (Requirement 12.1, 12.2, 12.4). Halts the Verification_Pass before
    verifying any row -- no partial verification is reported as a
    pass."""


class VerificationSourceError(Exception):
    """A traceability row's cited artifact file is absent, its row
    selector matches zero or more than one row, its named field/key is
    absent, or its `computation` value is not one of the fixed enum
    members (Requirement 12.1, 12.3). A row that cannot be resolved is
    a hard failure, not a skipped row."""


# --- groundedness-gate spec: extends the hierarchy above ---


class GroundednessConfigError(ConfigError):
    """Groundedness_Config (configs/groundedness.yaml) missing,
    unparsable, or declaring a missing/invalid field (Requirement 1.5)."""


class LabelMappingMismatchError(GroundednessConfigError):
    """The Groundedness_Config's documented native-label-to-
    Groundedness_Verdict mapping record, or its documented score
    definition, disagrees with the corresponding hard-coded constant in
    src/groundedness_labels.py (Requirement 1.5, 6.3, 6.10). Raised at
    config-load time, before any answer is generated -- this is what
    makes "fixed before any Quarantine_Rate exists ... never revised"
    a structurally enforced property rather than a documentation
    promise alone."""


class GenerationSubsetInputError(Exception):
    """results/per_query.csv is absent, cannot be parsed, or lacks a
    run_id column or a query_id column (Requirement 2.1). Halts before
    the Generation_Subset is sampled."""


class ReplayedRunNotFoundError(Exception):
    """The Replayed_Run's run_id declared in the Groundedness_Config is
    not present in results/per_query.csv's run_id column (Requirement
    2.5). Halts before the Generation_Subset is sampled."""


class FrozenRetrieverConfigError(Exception):
    """The Frozen_Retriever_Config declared for the Replayed_Run could
    not be loaded from configs/sweep.yaml, or no retriever config
    entry's name matches the run_id's retriever-name prefix (Requirement
    3.6). Halts the entire run before any Generation_Subset query is
    processed."""


class RetrievalReplayError(Exception):
    """The replayed retriever failed to build its index, or failed to
    retrieve documents for a Generation_Subset query (Requirement 3.7).
    Halts the entire run; no Generated_Answer is produced for any
    remaining query."""


class GeneratorModelLoadError(Exception):
    """The Generator_Model's weights could not be downloaded to or
    loaded from the path under data/ (Requirement 4.7)."""


class GeneratorGenerationError(Exception):
    """The Generator_Model failed to produce a Generated_Answer for a
    Generation_Subset query after its weights had already loaded
    successfully (Requirement 4.8)."""


class JudgeModelLoadError(Exception):
    """The Judge_Model's weights could not be loaded from the path
    under data/, or the loaded model's id2label does not expose an
    'entailment' class (Requirement 6.12)."""


class JudgeVerdictError(Exception):
    """The Judge_Model failed to produce a Groundedness_Verdict for a
    Claim (Requirement 6.12)."""


class GroundednessReportWriteError(Exception):
    """results/groundedness.csv could not be written (Requirement 8.5).
    The groundedness-gate analogue of ReportWriteError."""


class HandCheckedSampleWriteError(Exception):
    """results/hand_checked_sample.csv could not be exported."""


class HandCheckedJoinedWriteError(Exception):
    """results/hand_checked_joined.csv could not be written after
    Hand_Label_Import succeeded (Requirement 10.5). The
    Groundedness_Runner's analogue of GroundednessReportWriteError for
    this third, fully-derived artifact -- never hand-edited, so a
    write failure here is always safe to retry on a later run."""


class GeneratedAnswersWriteError(Exception):
    """results/generated_answers.csv could not be written. Records the
    raw Generated_Answer, its prompt's token count (Generator_Model
    tokenizer, before truncation), and the paired Judge_Model premise's
    token count (before truncation) for every Generation_Subset query,
    so a Claim can be traced back to the text it was segmented from,
    and prompt/premise truncation is visible without a re-run."""


class HandCheckedContextWriteError(Exception):
    """results/hand_checked_sample_context.md could not be written.
    A read-only labelling aid, never hand-edited -- like
    HandCheckedJoinedWriteError's rationale for
    hand_checked_joined.csv, a write failure here is always safe to
    retry on a later run."""


class ClaimClassificationError(Exception):
    """docs/claim_assertion_classification.csv is missing, cannot be
    parsed, lacks a required column, or has no row for a
    (query_id, claim_index) the Groundedness_Runner needs a
    declarative-assertion classification for (used only for the
    Agreement_Rate partition analysis in SPEC.md; never influences a
    Groundedness_Verdict, judge_score, or Quarantine_Decision)."""


# --- full-grid-chunking-sweep spec: extends the hierarchy above ---


class ChunkingError(Exception):
    """A Chunker produced zero Chunks for a document. Raised by
    build_chunk_corpus, naming the offending doc_id and the chunker's
    strategy_name, before any index is built for that Chunking_Strategy
    (Requirement 2.6). Also reused, wrapped as FrozenRetrieverConfigError,
    if retrieval_replay.build_frozen_retriever's WholeDocumentChunker
    application ever fails."""


class ChunkingConfigError(ConfigError):
    """A Chunking_Strategy entry in configs/sweep.yaml declares an
    invalid field: fixed_window's window_size/stride or
    sentence_window's sentences_per_chunk/max_chunk_tokens must each be
    a positive integer (Requirement 7.6). A ConfigError subclass,
    mirroring UnsupportedPreprocessingError's relationship to
    ConfigError."""


# --- analysis-writeup spec: extends the hierarchy above ---


class FailureBucketInputError(Exception):
    """results/per_query.csv is absent, cannot be parsed as a CSV, or
    lacks one of the columns the Bucket_Assigner requires -- run_id,
    retriever, chunking_strategy, query_id, recall_at_1, recall_at_20,
    ndcg_at_10, num_judged_relevant (Requirement 2.5). The
    analysis-writeup analogue of SignificanceInputError: halts before
    either report is written.

    Raised from exactly one place: load_per_query. Concerns a COMMITTED
    artifact under results/ -- distinct from CovariateInputError, which
    concerns the gitignored data/ cache."""


class CovariateInputError(Exception):
    """A Covariate_Enrichment_Stage input is unavailable or
    unresolvable: the BEIR SciFact dataset directory is absent from the
    Local_Cache, the loaded Qrels are absent or empty, a Dense_Model's
    tokenizer snapshot directory is absent from the Local_Cache, or a
    query_id present in results/per_query.csv is absent from the loaded
    query set (Requirement 16.13).

    Requirement 16.15 requires these four conditions to share ONE error
    type: they are one distinguishable failure -- "the covariate inputs
    are not available" -- with one remedy (populate data/ by running the
    sweep, then re-run), and no caller can usefully branch between them.
    The MESSAGE names which of the four it was. Halts before either
    report is written, leaving any pre-existing copy of either file
    byte-for-byte in its pre-run state (Requirement 16.13), and never
    downloads anything or substitutes a default (Requirement 16.14).

    Raise sites: assert_local_cache_present, load_covariate_inputs
    (wrapping CorpusLoadError / CorpusValidationError from load_scifact
    and TokenizerLoadError from load_tokenizer_offline), and
    compute_token_length_covariates (the query-coverage check)."""


class ContrastQuerySetError(Exception):
    """A Pair_Contrast's two Run_Ids do not cover the same query_id set
    -- a query_id is present for Run_A and absent for Run_B, or the
    reverse -- or one of the Pair_Contrast's Run_Ids is absent from
    results/per_query.csv entirely (Requirement 4.6). A Contrast_Bucket
    partition over an asymmetric query set has no total partition, so
    this halts before either report is written.

    Raise sites: build_declared_contrast_set (an absent Run_Id) and
    build_contrast_counts (an asymmetric query set). Reports a
    data-coverage fault in the INPUT -- distinct from
    FailureBucketAssertionError, which reports a violated invariant in
    this module's own arithmetic."""


class FailureBucketAssertionError(Exception):
    """A pre-write invariant did not hold. Covers every assert_* helper
    in the module, because all of them check the same kind of thing --
    a property this module's own computation is supposed to guarantee --
    and none of them is separately actionable by a caller:

      - a Run_Id's or Pair_Contrast's four bucket counts do not sum to
        the query count that partition covers (Requirement 5.1, 5.2);
      - a (run_id, query_id) pair is labelled more than once, including
        after the covariate join (Requirement 5.3);
      - a run_id's four unrounded fractions do not sum to 1 within
        FRACTION_SUM_TOLERANCE (Requirement 5.4), or their 6-decimal
        renderings do not sum to 1 within RENDERED_FRACTION_TOLERANCE
        (Requirement 5.7);
      - a Run_Id read from results/per_query.csv contains the
        four-character Composite_Run_Id separator "|vs|", so a
        Composite_Run_Id could collide with a Run_Id in the counts
        report's shared run_id column (Requirement 7.6) -- folded in
        here from what would otherwise be a fifth type with a single
        assert_* raise site;
      - a query_id's six covariate values are not identical across
        every row carrying that query_id (Requirement 16.9), or a
        max_relevant_doc_token_len / any_relevant_doc_exceeds_limit
        cell holds a numeric 0 where the Missing_Value_Sentinel was
        required (Requirement 16.8);
      - a duplicate Pair_Contrast would be emitted (Requirement 4.5).

    Names the affected Run_Id, Pair_Contrast, query_id, or run_id, the
    observed value, and the expected value (Requirement 5.5). Halts
    before either report is written, leaving any pre-existing copy of
    either file byte-for-byte in its pre-run state."""


class FailureBucketWriteError(Exception):
    """results/failure_buckets.csv or results/failure_bucket_counts.csv
    could not be written (disk full, permissions). The
    analysis-writeup analogue of ReportWriteError. Distinct from the
    four validation errors above: this is an I/O failure reached only
    after every assertion has already passed, so the error message
    states which of the two paths failed and whether the other was
    already written.

    Raise sites: write_failure_buckets and
    write_failure_bucket_counts, both wrapping ReportWriteError from
    src.report._atomic_write_text."""
