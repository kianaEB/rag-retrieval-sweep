"""Sweep_Config schema, YAML loading, and validation.

`load_sweep_config` is the single point where the full-grid "supported
set" (exactly 3 retrievers -- one `bm25`, `all-MiniLM-L6-v2`, and
`bge-small-en-v1.5` -- cutoffs == {1, 5, 10, 20}, exactly 3
Chunking_Strategy entries -- `whole_document`, `fixed_window`,
`sentence_window` -- BM25 preprocessing restricted to the fixed values
declared in `configs/sweep.yaml`) is enforced, per `design.md`'s
`src/config.py` section. This module imports only `PyYAML` and the
standard library -- never `beir`, `sentence-transformers`, or
`huggingface_hub` -- so it is always safe to run before
`configure_caches()` (see `src/sweep_runner.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import yaml

from src.errors import ChunkingConfigError, ConfigError, UnsupportedPreprocessingError

# Session-1 supported sets. Fixed here, once; never adjusted after seeing
# results (see tech.md's "Lexical preprocessing is declared once" rule).
SUPPORTED_CUTOFFS: Tuple[int, ...] = (1, 5, 10, 20)
SUPPORTED_BM25_TOKENIZER = "regex_word"
SUPPORTED_BM25_STOPWORDS = "none"
SUPPORTED_BM25_STEMMING = "none"

# full-grid-chunking-sweep spec: the 3 declared Chunking_Strategy names
# and the 2 declared Dense_Retriever names -- fixed here, once, per
# Requirements 1.1, 1.5, 7.1, 7.3.
SUPPORTED_CHUNKING_STRATEGY_NAMES: Tuple[str, ...] = (
    "whole_document",
    "fixed_window",
    "sentence_window",
)
SUPPORTED_DENSE_RETRIEVER_NAMES: Tuple[str, ...] = (
    "all-MiniLM-L6-v2",
    "bge-small-en-v1.5",
)


@dataclass(frozen=True)
class BM25RetrieverConfig:
    name: str
    type: str
    k1: float
    b: float
    tokenizer: str
    lowercase: bool
    stopwords: str
    stemming: str


@dataclass(frozen=True)
class DenseRetrieverConfig:
    name: str
    type: str
    model_name: str
    batch_size: int


RetrieverConfig = Union[BM25RetrieverConfig, DenseRetrieverConfig]


@dataclass(frozen=True)
class WholeDocumentChunkingConfig:
    name: str  # "whole_document"


@dataclass(frozen=True)
class FixedWindowChunkingConfig:
    name: str  # "fixed_window"
    window_size: int  # 200 (Requirement 3.2), read from YAML, never hard-coded
    stride: int  # 50 (Requirement 3.2), read from YAML, never hard-coded


@dataclass(frozen=True)
class SentenceWindowChunkingConfig:
    name: str  # "sentence_window"
    sentences_per_chunk: int  # 3 (Requirement 4.2), read from YAML, never hard-coded
    max_chunk_tokens: int  # 256 (Requirement 4.2), read from YAML, never hard-coded


ChunkingStrategyConfig = Union[
    WholeDocumentChunkingConfig, FixedWindowChunkingConfig, SentenceWindowChunkingConfig
]


@dataclass(frozen=True)
class SweepConfig:
    seed: int
    chunking_strategies: Tuple[ChunkingStrategyConfig, ...]  # exactly 3, one of each
    cutoffs: Tuple[int, ...]
    retrievers: Tuple[RetrieverConfig, ...]  # exactly 3
    data_dir: Path
    output_path: Path


def _require_field(mapping: dict, field: str, context: str) -> Any:
    """Returns mapping[field], raising ConfigError naming the missing
    field if absent. `context` (e.g. "top-level config" or
    "retrievers[0] (bm25)") is included so the error identifies *where*
    the field was expected, per Requirement 2.6."""
    if field not in mapping:
        raise ConfigError(f"{context}: missing required field '{field}'")
    return mapping[field]


def _load_bm25_config(raw: dict, index: int) -> BM25RetrieverConfig:
    context = f"retrievers[{index}] (type: bm25)"
    name = _require_field(raw, "name", context)
    k1 = _require_field(raw, "k1", context)
    b = _require_field(raw, "b", context)
    tokenizer = _require_field(raw, "tokenizer", context)
    lowercase = _require_field(raw, "lowercase", context)
    stopwords = _require_field(raw, "stopwords", context)
    stemming = _require_field(raw, "stemming", context)

    if tokenizer != SUPPORTED_BM25_TOKENIZER:
        raise UnsupportedPreprocessingError(
            f"{context}: unsupported tokenizer {tokenizer!r}; "
            f"only {SUPPORTED_BM25_TOKENIZER!r} is supported"
        )
    if stopwords != SUPPORTED_BM25_STOPWORDS:
        raise UnsupportedPreprocessingError(
            f"{context}: unsupported stopwords setting {stopwords!r}; "
            f"only {SUPPORTED_BM25_STOPWORDS!r} is supported"
        )
    if stemming != SUPPORTED_BM25_STEMMING:
        raise UnsupportedPreprocessingError(
            f"{context}: unsupported stemming setting {stemming!r}; "
            f"only {SUPPORTED_BM25_STEMMING!r} is supported"
        )
    if not isinstance(lowercase, bool):
        raise UnsupportedPreprocessingError(
            f"{context}: 'lowercase' must be a boolean, got {lowercase!r}"
        )

    try:
        k1 = float(k1)
        b = float(b)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: 'k1' and 'b' must be numeric") from exc

    return BM25RetrieverConfig(
        name=str(name),
        type="bm25",
        k1=k1,
        b=b,
        tokenizer=tokenizer,
        lowercase=lowercase,
        stopwords=stopwords,
        stemming=stemming,
    )


def _load_dense_config(raw: dict, index: int) -> DenseRetrieverConfig:
    context = f"retrievers[{index}] (type: dense)"
    name = _require_field(raw, "name", context)
    model_name = _require_field(raw, "model_name", context)
    batch_size = _require_field(raw, "batch_size", context)

    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: 'batch_size' must be an integer") from exc
    if batch_size <= 0:
        raise ConfigError(f"{context}: 'batch_size' must be positive, got {batch_size}")

    return DenseRetrieverConfig(
        name=str(name),
        type="dense",
        model_name=str(model_name),
        batch_size=batch_size,
    )


def _load_retriever_config(raw: Any, index: int) -> RetrieverConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"retrievers[{index}]: expected a mapping, got {type(raw).__name__}")
    retriever_type = _require_field(raw, "type", f"retrievers[{index}]")
    if retriever_type == "bm25":
        return _load_bm25_config(raw, index)
    if retriever_type == "dense":
        return _load_dense_config(raw, index)
    raise ConfigError(
        f"retrievers[{index}]: unsupported retriever type {retriever_type!r}; "
        f"supported types are 'bm25' and 'dense'"
    )


def _require_positive_int(value: Any, field: str, context: str) -> int:
    """Validates `value` is a positive integer, raising `ChunkingConfigError`
    (never a plain `ConfigError`) naming `field` and `context` if not
    (Requirement 7.6). `bool` is rejected explicitly even though it is a
    subclass of `int` in Python -- `True`/`False` are never a valid
    `window_size`/`stride`/`sentences_per_chunk`/`max_chunk_tokens`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChunkingConfigError(
            f"{context}: '{field}' must be a positive integer, got {value!r}"
        )
    if value <= 0:
        raise ChunkingConfigError(
            f"{context}: '{field}' must be a positive integer, got {value!r}"
        )
    return value


def _load_chunking_strategy_config(raw: Any, index: int) -> ChunkingStrategyConfig:
    """Parses and validates one `chunking_strategies[index]` entry,
    dispatching on its `name` field. `fixed_window`'s `window_size`/
    `stride` and `sentence_window`'s `sentences_per_chunk`/
    `max_chunk_tokens` are read from the YAML here, never hard-coded in
    `src/` (Requirement 7.6)."""
    if not isinstance(raw, dict):
        raise ConfigError(
            f"chunking_strategies[{index}]: expected a mapping, got {type(raw).__name__}"
        )
    context = f"chunking_strategies[{index}]"
    name = _require_field(raw, "name", context)

    if name == "whole_document":
        return WholeDocumentChunkingConfig(name=str(name))

    if name == "fixed_window":
        window_size = _require_positive_int(
            _require_field(raw, "window_size", context), "window_size", context
        )
        stride = _require_positive_int(
            _require_field(raw, "stride", context), "stride", context
        )
        return FixedWindowChunkingConfig(name=str(name), window_size=window_size, stride=stride)

    if name == "sentence_window":
        sentences_per_chunk = _require_positive_int(
            _require_field(raw, "sentences_per_chunk", context),
            "sentences_per_chunk",
            context,
        )
        max_chunk_tokens = _require_positive_int(
            _require_field(raw, "max_chunk_tokens", context), "max_chunk_tokens", context
        )
        return SentenceWindowChunkingConfig(
            name=str(name),
            sentences_per_chunk=sentences_per_chunk,
            max_chunk_tokens=max_chunk_tokens,
        )

    raise ConfigError(
        f"{context}: unsupported chunking strategy name {name!r}; supported names "
        f"are {list(SUPPORTED_CHUNKING_STRATEGY_NAMES)}"
    )


def load_sweep_config(path: Path) -> SweepConfig:
    """Reads and validates the Sweep_Config YAML file at `path`.

    Raises `ConfigError` (or its subclasses `UnsupportedPreprocessingError`/
    `ChunkingConfigError`) if the file is missing, is not valid YAML,
    omits a required declaration, or declares a retriever, evaluation
    cutoff, or Chunking_Strategy outside the supported set (Requirement
    2.6, 7.3, 7.4, 7.6). Never partially applies a Sweep_Config: the
    first violation found raises and no `SweepConfig` is returned.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Sweep_Config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"failed to read Sweep_Config file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse {path} as YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML content must be a mapping")

    context = "top-level config"

    # The old singular `chunking_strategy` field is never silently
    # ignored: its presence -- even alongside a correctly-declared
    # `chunking_strategies` list -- raises explicitly, so a stale
    # session-1/2-style config can never be mistaken for "field not
    # present" and cannot silently run only 1 Chunking_Strategy instead
    # of 3 (Requirement 7.4).
    if "chunking_strategy" in data:
        raise ConfigError(
            f"{context}: stale field 'chunking_strategy' found; this spec replaces it "
            f"with 'chunking_strategies' (a list of exactly 3 entries) -- remove the "
            f"old singular field and declare 'chunking_strategies' instead"
        )

    seed = _require_field(data, "seed", context)
    chunking_strategies_raw = _require_field(data, "chunking_strategies", context)
    cutoffs_raw = _require_field(data, "cutoffs", context)
    retrievers_raw = _require_field(data, "retrievers", context)
    data_dir = _require_field(data, "data_dir", context)
    output_path = _require_field(data, "output_path", context)

    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: 'seed' must be an integer, got {seed!r}") from exc

    if not isinstance(chunking_strategies_raw, list):
        raise ConfigError(f"{context}: 'chunking_strategies' must be a list")

    chunking_strategies = tuple(
        _load_chunking_strategy_config(entry, i)
        for i, entry in enumerate(chunking_strategies_raw)
    )
    if len(chunking_strategies) != 3:
        raise ConfigError(
            f"{context}: 'chunking_strategies' must declare exactly 3 entries "
            f"({list(SUPPORTED_CHUNKING_STRATEGY_NAMES)}), got {len(chunking_strategies)}"
        )
    chunking_strategy_names = {cs.name for cs in chunking_strategies}
    if chunking_strategy_names != set(SUPPORTED_CHUNKING_STRATEGY_NAMES):
        raise ConfigError(
            f"{context}: 'chunking_strategies' must declare exactly one entry each "
            f"named {list(SUPPORTED_CHUNKING_STRATEGY_NAMES)} (no duplicates, no "
            f"omissions, no unsupported name); got {sorted(chunking_strategy_names)}"
        )

    if not isinstance(cutoffs_raw, list):
        raise ConfigError(f"{context}: 'cutoffs' must be a list of integers")
    try:
        cutoffs = tuple(int(c) for c in cutoffs_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: 'cutoffs' must contain only integers") from exc
    if tuple(sorted(cutoffs)) != SUPPORTED_CUTOFFS:
        raise ConfigError(
            f"{context}: 'cutoffs' must be exactly {list(SUPPORTED_CUTOFFS)}, "
            f"got {list(cutoffs)}"
        )

    if not isinstance(retrievers_raw, list):
        raise ConfigError(f"{context}: 'retrievers' must be a list")

    retrievers = tuple(
        _load_retriever_config(entry, i) for i, entry in enumerate(retrievers_raw)
    )
    if len(retrievers) != 3:
        raise ConfigError(
            f"{context}: 'retrievers' must declare exactly 3 entries, got {len(retrievers)}"
        )
    bm25_entries = [r for r in retrievers if isinstance(r, BM25RetrieverConfig)]
    dense_entries = [r for r in retrievers if isinstance(r, DenseRetrieverConfig)]
    if len(bm25_entries) != 1 or len(dense_entries) != 2:
        raise ConfigError(
            f"{context}: 'retrievers' must declare exactly one 'bm25' entry and "
            f"exactly two 'dense' entries; got {len(bm25_entries)} bm25 and "
            f"{len(dense_entries)} dense"
        )
    dense_names = {d.name for d in dense_entries}
    if dense_names != set(SUPPORTED_DENSE_RETRIEVER_NAMES):
        raise ConfigError(
            f"{context}: the two 'dense' retriever entries' 'name' fields must be "
            f"exactly {list(SUPPORTED_DENSE_RETRIEVER_NAMES)} (both required, in "
            f"either order); got {sorted(dense_names)}"
        )
    bge_entry = next(d for d in dense_entries if d.name == "bge-small-en-v1.5")
    if bge_entry.model_name != "BAAI/bge-small-en-v1.5":
        raise ConfigError(
            f"{context}: the 'bge-small-en-v1.5' retriever entry's 'model_name' must "
            f"equal 'BAAI/bge-small-en-v1.5', got {bge_entry.model_name!r}"
        )

    return SweepConfig(
        seed=seed,
        chunking_strategies=chunking_strategies,
        cutoffs=cutoffs,
        retrievers=retrievers,
        data_dir=Path(data_dir),
        output_path=Path(output_path),
    )
