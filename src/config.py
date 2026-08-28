"""Sweep_Config schema, YAML loading, and validation.

`load_sweep_config` is the single point where the session-1 "supported
set" (exactly 2 retrievers, cutoffs == {1, 5, 10, 20}, chunking ==
"whole_document", BM25 preprocessing restricted to the fixed values
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

from src.errors import ConfigError, UnsupportedPreprocessingError

# Session-1 supported sets. Fixed here, once; never adjusted after seeing
# results (see tech.md's "Lexical preprocessing is declared once" rule).
SUPPORTED_CUTOFFS: Tuple[int, ...] = (1, 5, 10, 20)
SUPPORTED_CHUNKING_STRATEGY = "whole_document"
SUPPORTED_BM25_TOKENIZER = "regex_word"
SUPPORTED_BM25_STOPWORDS = "none"
SUPPORTED_BM25_STEMMING = "none"


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
class SweepConfig:
    seed: int
    chunking_strategy: str
    cutoffs: Tuple[int, ...]
    retrievers: Tuple[RetrieverConfig, ...]
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


def load_sweep_config(path: Path) -> SweepConfig:
    """Reads and validates the Sweep_Config YAML file at `path`.

    Raises `ConfigError` (or its subclass `UnsupportedPreprocessingError`)
    if the file is missing, is not valid YAML, omits a required
    declaration, or declares a retriever, evaluation cutoff, or
    chunking strategy outside the session-1 supported set (Requirement
    2.6). Never partially applies a Sweep_Config: the first violation
    found raises and no `SweepConfig` is returned.
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
    seed = _require_field(data, "seed", context)
    chunking_strategy = _require_field(data, "chunking_strategy", context)
    cutoffs_raw = _require_field(data, "cutoffs", context)
    retrievers_raw = _require_field(data, "retrievers", context)
    data_dir = _require_field(data, "data_dir", context)
    output_path = _require_field(data, "output_path", context)

    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: 'seed' must be an integer, got {seed!r}") from exc

    if chunking_strategy != SUPPORTED_CHUNKING_STRATEGY:
        raise ConfigError(
            f"{context}: unsupported chunking_strategy {chunking_strategy!r}; "
            f"only {SUPPORTED_CHUNKING_STRATEGY!r} is supported in session 1"
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
    if len(retrievers) != 2:
        raise ConfigError(
            f"{context}: 'retrievers' must declare exactly 2 entries, got {len(retrievers)}"
        )
    bm25_entries = [r for r in retrievers if isinstance(r, BM25RetrieverConfig)]
    dense_entries = [r for r in retrievers if isinstance(r, DenseRetrieverConfig)]
    if len(bm25_entries) != 1 or len(dense_entries) != 1:
        raise ConfigError(
            f"{context}: 'retrievers' must declare exactly one 'bm25' entry and "
            f"exactly one 'dense' entry; got {len(bm25_entries)} bm25 and "
            f"{len(dense_entries)} dense"
        )

    return SweepConfig(
        seed=seed,
        chunking_strategy=chunking_strategy,
        cutoffs=cutoffs,
        retrievers=retrievers,
        data_dir=Path(data_dir),
        output_path=Path(output_path),
    )
