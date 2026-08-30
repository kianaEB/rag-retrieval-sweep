"""Token_Length_Analysis: measures what fraction of the loaded SciFact
corpus exceeds `all-MiniLM-L6-v2`'s 256-token maximum sequence length
under whole-document chunking (Requirement 11,
`.kiro/specs/repo-writeup/design.md`'s `src/token_length_analysis.py`
section).

Reuses session 1's own `load_sweep_config`, `configure_caches`, and
`load_scifact` rather than re-implementing any of them, and tokenizes
each document's `title + " " + text` input via the single
`format_document_text` function imported from
`src.retrievers.dense_retriever` (Task 1's extract-and-import refactor)
-- the exact same string `DenseRetriever.build_index` encodes, so
Requirement 11.1's "exactly as the dense retriever encodes it" holds by
construction rather than by two independently-maintained copies
staying in sync.

`compute_exceedance_stats` is the sole pure aggregation function (no
corpus, no tokenizer, no file I/O) and is this module's unit-under-test
surface, mirroring how `src/metrics.py` and `src/significance.py` each
isolate one pure aggregation core.

`load_tokenizer_offline` loads the tokenizer with two independent
"never touch the network" layers -- the `HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE` environment variables, set before the load call,
and the explicit `local_files_only=True` argument to
`AutoTokenizer.from_pretrained` -- mirroring the defense-in-depth style
`DenseRetriever.__init__` already uses for its own cache-path assertion
(Requirement 11.3, 11.7). Either layer alone would prevent a network
call; both together are deliberate redundancy, not an accident.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

from transformers import AutoTokenizer

from src.config import DenseRetrieverConfig, SweepConfig, load_sweep_config
from src.corpus_loader import configure_caches, load_scifact
from src.errors import (
    ConfigError,
    CorpusLoadError,
    CorpusValidationError,
    ReportWriteError,
    TokenizerLoadError,
    TokenLengthReportError,
)
from src.report import _atomic_write_text
from src.retrievers.dense_retriever import format_document_text

if TYPE_CHECKING:  # pragma: no cover - import only needed for type hints
    from transformers import PreTrainedTokenizerBase

# The all-MiniLM-L6-v2 model's published maximum sequence length
# (Requirement 11.2). A literal, not a config field: this is a fixed
# property of the model's architecture being measured against, not a
# sweep parameter -- see design.md's "Components and Interfaces"
# section for why this is a deliberate exception to "derive constants
# from data, don't hardcode them."
MAX_SEQUENCE_LENGTH = 256

# Defaults applied when `--config`/`--output` are not passed on the
# command line (mirroring src/sweep_runner.py's DEFAULT_CONFIG_PATH
# convention).
DEFAULT_CONFIG_PATH = Path("configs/sweep.yaml")
DEFAULT_OUTPUT_PATH = Path("results/token_length_report.json")


@dataclass(frozen=True)
class TokenLengthStats:
    """The three numbers `compute_exceedance_stats` derives from a list
    of per-document token counts (Requirement 11.2, 11.4)."""

    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float


def compute_exceedance_stats(
    token_counts: Sequence[int], max_sequence_length: int
) -> TokenLengthStats:
    """Computes exceedance statistics over `token_counts` against
    `max_sequence_length`.

    `num_documents_exceeding` counts elements *strictly greater than*
    `max_sequence_length` (Requirement 11.2 -- a count exactly equal to
    the threshold does not count as exceeding). `fraction_exceeding` is
    `num_documents_exceeding / num_documents_total`, defined as `0.0`
    when `num_documents_total == 0` rather than raising
    `ZeroDivisionError`. A pure function: no corpus, no tokenizer, no
    file I/O, and its result does not depend on the order of
    `token_counts` (both `sum` and `len` are order-independent
    reductions).
    """
    num_documents_total = len(token_counts)
    num_documents_exceeding = sum(1 for count in token_counts if count > max_sequence_length)
    fraction_exceeding = (
        num_documents_exceeding / num_documents_total if num_documents_total else 0.0
    )
    return TokenLengthStats(
        num_documents_total=num_documents_total,
        num_documents_exceeding=num_documents_exceeding,
        fraction_exceeding=fraction_exceeding,
    )


@dataclass(frozen=True)
class TokenLengthReport:
    """The committed `results/token_length_report.json` schema
    (Requirement 11.4). Never carries a missing-value sentinel: this
    analysis either fully succeeds and every field is populated, or it
    halts before writing anything at all."""

    model_name: str
    max_sequence_length: int
    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float


def load_tokenizer_offline(model_name: str, cache_folder: Path) -> "PreTrainedTokenizerBase":
    """Loads `model_name`'s tokenizer from the local cache under
    `cache_folder`, making no network call under any circumstance
    (Requirement 11.3, 11.7).

    Sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` before the load call
    (enforced globally by `huggingface_hub`/`transformers` regardless
    of which loading API is used), and passes `local_files_only=True`
    explicitly to `AutoTokenizer.from_pretrained` alongside the same
    `cache_folder` `configure_caches` points `HF_HOME`/`HF_HUB_CACHE`
    at -- two independent layers, either one sufficient on its own,
    together as defense in depth. Any exception from either layer (a
    missing snapshot, a corrupted cache, `huggingface_hub`'s own
    `LocalEntryNotFoundError`, or a generic `OSError`) is caught and
    re-raised as `TokenizerLoadError` -- never retried without the
    offline flags, and never allowed to fall through to a network
    request.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_folder),
            local_files_only=True,
        )
    except Exception as exc:
        raise TokenizerLoadError(
            f"failed to load tokenizer {model_name!r} from local cache "
            f"{cache_folder} without a network call: {exc}"
        ) from exc


def count_tokens(tokenizer: "PreTrainedTokenizerBase", text: str) -> int:
    """Returns the untruncated token count of `text`, including any
    special tokens the tokenizer inserts (Requirement 11.1).

    `truncation=False` is essential: `DenseRetriever` itself truncates
    at encode time, but this analysis exists specifically to measure
    *how much* would be truncated, so it must count the true,
    untruncated length -- truncating first would make every document's
    counted length `<= max_sequence_length` by construction and the
    whole measurement would be vacuous. `add_special_tokens=True`
    matches what `SentenceTransformer.encode` does internally.
    """
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def write_token_length_report(report: TokenLengthReport, output_path: Path) -> None:
    """Writes `report` to `output_path` (e.g.
    `results/token_length_report.json`) as JSON, atomically, reusing
    `src.report._atomic_write_text` (temp file + `os.replace`) so
    `output_path` is left absent or in its pre-run state on any
    failure. Raises `TokenLengthReportError` on any failure.
    """
    output_path = Path(output_path)
    try:
        json_text = json.dumps(dataclasses.asdict(report), indent=2)
    except Exception as exc:
        raise TokenLengthReportError(
            f"failed to build token length report for {output_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(output_path, json_text, failure_context="token length report")
    except ReportWriteError as exc:
        raise TokenLengthReportError(str(exc)) from exc


def _find_dense_retriever_config(config: SweepConfig) -> DenseRetrieverConfig:
    """Returns the one `DenseRetrieverConfig` entry in `config.retrievers`.

    `load_sweep_config` already guarantees exactly one such entry
    exists (`src/config.py`'s own validation), so this never raises in
    practice against a config that already passed that validation; it
    exists only to name the extraction step, per design.md's `main()`
    step 2.
    """
    for retriever_config in config.retrievers:
        if isinstance(retriever_config, DenseRetrieverConfig):
            return retriever_config
    raise ConfigError("Sweep_Config declares no 'dense' retriever entry")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.token_length_analysis",
        description=(
            "Tokenizes every document in the loaded SciFact corpus with "
            "the already-cached all-MiniLM-L6-v2 tokenizer and writes "
            "results/token_length_report.json: the count and fraction of "
            "documents whose untruncated token count exceeds the model's "
            "256-token maximum sequence length. Makes no network call."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the Sweep_Config YAML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the Token_Length_Report JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: `python -m src.token_length_analysis [--config
    PATH] [--output PATH]`.

    Orchestration, in order (`design.md`'s `src/token_length_analysis.py`
    `main()` section):

    1. Parse `--config` (default `configs/sweep.yaml`) and `--output`
       (default `results/token_length_report.json`). Load the config
       via `load_sweep_config`. On `ConfigError`, print and return
       non-zero, write nothing.
    2. Extract the one `DenseRetrieverConfig` entry for `model_name`.
    3. `configure_caches(config.data_dir)` -- before any
       `huggingface_hub`/`transformers`/`beir`-importing *call* runs,
       same ordering discipline `sweep_runner.py` follows for its own
       calls.
    4. `load_scifact(config.data_dir)`. On `CorpusLoadError` /
       `CorpusValidationError`, print and return non-zero, write
       nothing.
    5. `load_tokenizer_offline(model_name, config.data_dir /
       "hf_cache")`. On `TokenizerLoadError`, print and return
       non-zero, write nothing, no network call attempted.
    6. For every document in `bundle.corpus`, count tokens of
       `format_document_text(doc)` (imported from
       `src.retrievers.dense_retriever`).
    7. `compute_exceedance_stats(token_counts, MAX_SEQUENCE_LENGTH)`.
    8. Write the report atomically. On `TokenLengthReportError`, print
       and return non-zero.
    9. Return 0.
    """
    args = _parse_args(argv)

    try:
        config = load_sweep_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: failed to load Sweep_Config from {args.config}: {exc}", file=sys.stderr)
        return 1

    try:
        dense_config = _find_dense_retriever_config(config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Must run before any module that imports huggingface_hub
    # (transitively) is *called* -- see src/corpus_loader.py's and
    # src/sweep_runner.py's docstrings.
    configure_caches(config.data_dir)

    try:
        bundle, _load_report = load_scifact(config.data_dir)
    except (CorpusLoadError, CorpusValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        tokenizer = load_tokenizer_offline(dense_config.model_name, config.data_dir / "hf_cache")
    except TokenizerLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    token_counts = [
        count_tokens(tokenizer, format_document_text(doc)) for doc in bundle.corpus.values()
    ]
    stats = compute_exceedance_stats(token_counts, MAX_SEQUENCE_LENGTH)

    report = TokenLengthReport(
        model_name=dense_config.model_name,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_documents_total=stats.num_documents_total,
        num_documents_exceeding=stats.num_documents_exceeding,
        fraction_exceeding=stats.fraction_exceeding,
    )

    try:
        write_token_length_report(report, args.output)
    except TokenLengthReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"TOKEN_LENGTH_REPORT documents={report.num_documents_total} "
        f"exceeding={report.num_documents_exceeding} "
        f"fraction={report.fraction_exceeding}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
