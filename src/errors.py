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
