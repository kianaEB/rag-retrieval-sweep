"""Token_Length_Analysis: measures what fraction of Chunks exceed a
dense model's maximum sequence length, across all 6
`(Chunking_Strategy, dense model)` cells (3 Chunking_Strategy entries x
2 dense models) -- full-grid-chunking-sweep spec's extension of
Requirement 11, `.kiro/specs/repo-writeup/design.md`'s
`src/token_length_analysis.py` section.

Reuses session 1's own `load_sweep_config`, `configure_caches`, and
`load_scifact` rather than re-implementing any of them. Each cell's
Chunker builds a chunk corpus via `build_chunk_corpus` (the same
chunking abstraction the Sweep_Runner itself uses), and `compute_cell`
tokenizes every produced Chunk's own text -- for `whole_document`
chunking (one Chunk per document, containing that document's
unmodified `title + " " + text` content), this stays numerically
identical to the pre-chunking single-model measurement, since
`whole_document` chunking is a no-op.

`compute_exceedance_stats` is the sole pure aggregation function (no
corpus, no tokenizer, no file I/O) and is this module's unit-under-test
surface, mirroring how `src/metrics.py` and `src/significance.py` each
isolate one pure aggregation core. It is unchanged by this spec's
6-cell extension.

`load_tokenizer_offline` loads each tokenizer (the all-MiniLM-L6-v2
windowing tokenizer used by `fixed_window`/`sentence_window` Chunkers,
plus each of the 2 dense models' own tokenizers) with two independent
"never touch the network" layers -- the `HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE` environment variables, set before the load call,
and the explicit `local_files_only=True` argument to
`AutoTokenizer.from_pretrained` -- mirroring the defense-in-depth style
`DenseRetriever.__init__` already uses for its own cache-path assertion
(Requirement 11.3, 11.5, 11.7). Either layer alone would prevent a
network call; both together are deliberate redundancy, not an
accident.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from src.chunking import Chunker, FixedWindowChunker, SentenceWindowChunker, WholeDocumentChunker, build_chunk_corpus
from src.config import load_sweep_config
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

if TYPE_CHECKING:  # pragma: no cover - import only needed for type hints
    from transformers import PreTrainedTokenizerBase

# The all-MiniLM-L6-v2 model's published maximum sequence length
# (Requirement 11.2). A literal, not a config field: this is a fixed
# property of the model's architecture being measured against, not a
# sweep parameter -- see design.md's "Components and Interfaces"
# section for why this is a deliberate exception to "derive constants
# from data, don't hardcode them."
MAX_SEQUENCE_LENGTH = 256

# The 3 Chunking_Strategy entries x 2 dense-model names measured by the
# 6-cell report (Requirement 11.1). The all-MiniLM-L6-v2 windowing
# tokenizer used by fixed_window/sentence_window is a fixed, literal
# model name here (matching configs/sweep.yaml's own fixed choice for
# chunking, independent of which dense model each cell's own tokenizer
# measures) -- never derived from a Sweep_Config's `retrievers` list.
_WINDOWING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DENSE_MODEL_NAMES: Tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
)
_CHUNKING_STRATEGY_NAMES: Tuple[str, ...] = (
    "whole_document",
    "fixed_window",
    "sentence_window",
)

# configs/sweep.yaml's fixed fixed_window/sentence_window numeric
# parameters (Requirement 3.2, 4.2) -- declared once, not tuned, and
# read here as the same literals configs/sweep.yaml declares, since
# this analysis must never make a network call and therefore never
# parses the Sweep_Config's own chunking_strategies list through
# load_sweep_config's normal path in a way that would risk one.
_FIXED_WINDOW_WINDOW_SIZE = 200
_FIXED_WINDOW_STRIDE = 50
_SENTENCE_WINDOW_SENTENCES_PER_CHUNK = 3
_SENTENCE_WINDOW_MAX_CHUNK_TOKENS = 256

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
    halts before writing anything at all.

    Extended (not replaced) by the full-grid-chunking-sweep spec's
    6-cell report: the pre-existing top-level `model_name`/
    `max_sequence_length`/`num_documents_total`/`num_documents_exceeding`/
    `fraction_exceeding` fields remain populated from the
    `whole_document` x `all-MiniLM-L6-v2` cell specifically -- the
    exact pre-existing single-model measurement, numerically unchanged
    (Requirement 11.4's regression check) -- while `cells` adds the
    full list of 6 `TokenLengthCell` records under a new `"cells"` key
    (Requirement 11.1, 11.3)."""

    model_name: str
    max_sequence_length: int
    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float
    cells: List[TokenLengthCell] = dataclasses.field(default_factory=list)


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


@dataclass(frozen=True)
class TokenLengthCell:
    """One of the 6 `(Chunking_Strategy, dense model)` cells
    (Requirement 11.1). `num_documents_total` is actually "num Chunks
    total" for this Chunking_Strategy -- named to match
    `TokenLengthReport`'s pre-existing field name, since the
    `whole_document` cells (one Chunk per document) keep it
    numerically equal to a true document count, but `fixed_window`/
    `sentence_window` cells count Chunks, which exceed the document
    count. `max_sequence_length` is the model's own EFFECTIVE
    truncation length -- read via `resolve_effective_max_sequence_length`
    below, never a hard-coded 256 or 512 (Requirement 11.2)."""

    chunking_strategy: str
    model_name: str
    max_sequence_length: int
    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float


def resolve_effective_max_sequence_length(
    model_name: str, tokenizer: "PreTrainedTokenizerBase", cache_folder: Path
) -> int:
    """Returns `model_name`'s EFFECTIVE maximum sequence length -- the
    length `sentence_transformers.SentenceTransformer.encode()`
    actually truncates to at run time, not necessarily the bare
    tokenizer's own `model_max_length`.

    A `sentence-transformers`-packaged model ships a
    `sentence_bert_config.json` file declaring its own `max_seq_length`,
    which `SentenceTransformer` applies as an override on top of
    whatever the underlying tokenizer's `model_max_length` happens to
    be -- for `all-MiniLM-L6-v2` specifically, the bare BERT tokenizer
    reports `model_max_length=512`, but `SentenceTransformer` truncates
    at 256 (`sentence_bert_config.json`'s own declared value). Reading
    `tokenizer.model_max_length` alone would silently under-count
    truncation for that model -- exactly the measurement error
    Requirement 11.1's "exactly as the dense retriever encodes it"
    exists to prevent.

    This function reads `sentence_bert_config.json` from the local
    cache under `cache_folder`, via `huggingface_hub.hf_hub_download(
    ..., local_files_only=True)` -- never a network call, matching
    `load_tokenizer_offline`'s own no-network-call guarantee: a cache
    miss raises immediately rather than falling back to a request.
    Falls back to `int(tokenizer.model_max_length)` if no such file is
    cached locally for `model_name` (e.g. a plain, non-sentence-
    transformers model)."""
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=model_name,
            filename="sentence_bert_config.json",
            cache_dir=str(cache_folder),
            local_files_only=True,
        )
        with open(config_path, "r", encoding="utf-8") as f:
            sentence_bert_config = json.load(f)
        if "max_seq_length" in sentence_bert_config:
            return int(sentence_bert_config["max_seq_length"])
    except Exception:
        pass
    return int(tokenizer.model_max_length)


def compute_cell(
    chunker: "Chunker",
    tokenizer: "PreTrainedTokenizerBase",
    corpus: Dict[str, Dict[str, str]],
    max_sequence_length: int,
) -> TokenLengthCell:
    """Computes one `(Chunking_Strategy, dense model)` cell's chunk-level
    exceedance stats (Requirement 11.4 -- including the `whole_document`
    cells, which stay numerically consistent with the pre-existing
    single-model measurement because `whole_document` chunking is a
    no-op: one Chunk per document, identical text to the pre-chunking
    measurement).

    Builds a chunk corpus via `build_chunk_corpus(chunker, corpus)` --
    the same chunking abstraction the Sweep_Runner itself uses -- then
    counts tokens of every Chunk's own text (not the raw document
    corpus directly) via `tokenizer`, against the caller-supplied
    `max_sequence_length` (Requirement 11.2 -- resolved by
    `resolve_effective_max_sequence_length`, the model's own EFFECTIVE
    truncation length, not necessarily the bare tokenizer's
    `model_max_length`). Calls the existing, unchanged
    `compute_exceedance_stats`.
    """
    chunk_corpus = build_chunk_corpus(chunker, corpus)
    token_counts = [
        count_tokens(tokenizer, chunk["text"]) for chunk in chunk_corpus.values()
    ]
    stats = compute_exceedance_stats(token_counts, max_sequence_length)
    return TokenLengthCell(
        chunking_strategy=chunker.strategy_name,
        model_name=tokenizer.name_or_path,
        max_sequence_length=max_sequence_length,
        num_documents_total=stats.num_documents_total,
        num_documents_exceeding=stats.num_documents_exceeding,
        fraction_exceeding=stats.fraction_exceeding,
    )


def _make_chunkers(windowing_tokenizer: "PreTrainedTokenizerBase") -> Dict[str, "Chunker"]:
    """Returns the 3 `Chunker` instances used across every cell,
    keyed by Chunking_Strategy name -- built once and reused across
    both dense models' cells for a given Chunking_Strategy, since a
    `Chunker`'s own behavior does not depend on which dense model's
    tokenizer will later count that Chunker's output tokens.
    `fixed_window`/`sentence_window` are built from the same
    all-MiniLM-L6-v2 `windowing_tokenizer` `configs/sweep.yaml`
    declares for chunking itself (Requirement 3.1, 4.1), independent
    of which dense model's tokenizer measures exceedance for a given
    cell."""
    return {
        "whole_document": WholeDocumentChunker(),
        "fixed_window": FixedWindowChunker(
            windowing_tokenizer, _FIXED_WINDOW_WINDOW_SIZE, _FIXED_WINDOW_STRIDE
        ),
        "sentence_window": SentenceWindowChunker(
            windowing_tokenizer,
            _SENTENCE_WINDOW_SENTENCES_PER_CHUNK,
            _SENTENCE_WINDOW_MAX_CHUNK_TOKENS,
        ),
    }


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

    `dataclasses.asdict` recurses into the nested `cells` list of
    `TokenLengthCell` dataclasses automatically, so the written JSON's
    `"cells"` key holds a list of 6 plain objects, extending (not
    replacing) the existing top-level schema (Requirement 11.3).
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
    `main()` section, extended for the full-grid-chunking-sweep spec's
    6-cell report):

    1. Parse `--config` (default `configs/sweep.yaml`) and `--output`
       (default `results/token_length_report.json`). Load the config
       via `load_sweep_config`. On `ConfigError`, print and return
       non-zero, write nothing.
    2. `configure_caches(config.data_dir)` -- before any
       `huggingface_hub`/`transformers`/`beir`-importing *call* runs,
       same ordering discipline `sweep_runner.py` follows for its own
       calls.
    3. `load_scifact(config.data_dir)`. On `CorpusLoadError` /
       `CorpusValidationError`, print and return non-zero, write
       nothing.
    4. Load the all-MiniLM-L6-v2 windowing tokenizer (used by
       `fixed_window`/`sentence_window` Chunkers) and each of the 2
       dense-model tokenizers, all via the offline-forced
       `load_tokenizer_offline` -- never `load_chunking_tokenizer`,
       since this analysis must never make a network call under any
       circumstance (Requirement 11.5). On `TokenizerLoadError` for
       any single tokenizer, print and return non-zero, write nothing
       -- no per-cell recovery tier (unlike the sweep).
    5. Build the 3 `Chunker` instances (`_make_chunkers`) and loop over
       the 6 `(Chunking_Strategy, dense model)` cells, calling
       `compute_cell` for each. On any single cell's failure, halt the
       whole analysis before writing any partial report.
    6. Assemble `TokenLengthReport`: the pre-existing top-level fields
       are populated from the `whole_document` x `all-MiniLM-L6-v2`
       cell specifically (Requirement 11.4's regression check), and
       `cells` holds all 6 records (Requirement 11.1, 11.3).
    7. Write the report atomically. On `TokenLengthReportError`, print
       and return non-zero.
    8. Return 0.
    """
    args = _parse_args(argv)

    try:
        config = load_sweep_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: failed to load Sweep_Config from {args.config}: {exc}", file=sys.stderr)
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

    hf_cache_folder = config.data_dir / "hf_cache"
    try:
        windowing_tokenizer = load_tokenizer_offline(_WINDOWING_MODEL_NAME, hf_cache_folder)
        dense_tokenizers = {
            model_name: load_tokenizer_offline(model_name, hf_cache_folder)
            for model_name in _DENSE_MODEL_NAMES
        }
    except TokenizerLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    chunkers = _make_chunkers(windowing_tokenizer)

    # Resolved once per dense model (not once per cell): each model's
    # EFFECTIVE truncation length is a property of the model itself,
    # independent of which Chunking_Strategy is being measured.
    try:
        effective_max_sequence_length = {
            model_name: resolve_effective_max_sequence_length(
                model_name, dense_tokenizers[model_name], hf_cache_folder
            )
            for model_name in _DENSE_MODEL_NAMES
        }
    except Exception as exc:
        print(f"ERROR: failed to resolve effective max sequence length: {exc}", file=sys.stderr)
        return 1

    cells: List[TokenLengthCell] = []
    for chunking_strategy_name in _CHUNKING_STRATEGY_NAMES:
        chunker = chunkers[chunking_strategy_name]
        for model_name in _DENSE_MODEL_NAMES:
            tokenizer = dense_tokenizers[model_name]
            try:
                cell = compute_cell(
                    chunker,
                    tokenizer,
                    bundle.corpus,
                    max_sequence_length=effective_max_sequence_length[model_name],
                )
            except Exception as exc:
                # No per-cell recovery tier (unlike the sweep): a
                # single cell's failure halts the whole analysis
                # before writing any partial report.
                print(
                    f"ERROR: failed to compute the "
                    f"({chunking_strategy_name!r}, {model_name!r}) cell: {exc}",
                    file=sys.stderr,
                )
                return 1
            cells.append(cell)

    # The pre-existing top-level fields are superseded by, and kept
    # numerically identical to, the whole_document x all-MiniLM-L6-v2
    # cell specifically (Requirement 11.4's regression check).
    regression_cell = next(
        cell
        for cell in cells
        if cell.chunking_strategy == "whole_document"
        and cell.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    )

    report = TokenLengthReport(
        model_name=regression_cell.model_name,
        max_sequence_length=regression_cell.max_sequence_length,
        num_documents_total=regression_cell.num_documents_total,
        num_documents_exceeding=regression_cell.num_documents_exceeding,
        fraction_exceeding=regression_cell.fraction_exceeding,
        cells=cells,
    )

    try:
        write_token_length_report(report, args.output)
    except TokenLengthReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"TOKEN_LENGTH_REPORT cells={len(report.cells)} "
        f"whole_document_all-MiniLM-L6-v2_documents={report.num_documents_total} "
        f"exceeding={report.num_documents_exceeding} "
        f"fraction={report.fraction_exceeding}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
