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
