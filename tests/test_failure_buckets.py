"""analysis-writeup spec: tests over `src.failure_buckets`'s
Bucket_Assignment_Stage and Covariate_Enrichment_Stage (Requirement 15).

Imports only `src.failure_buckets`, the five exception types from
`src.errors`, `_resolve_csv_reference`/`_CSV_ARTIFACTS`/
`_ALLOWED_COMPUTATIONS` from `src.verify_writeup_numbers` (for the one
selector test and the one registration test), plus `pandas`, `pytest`,
and `hypothesis`. No retriever, no corpus loader, no tokenizer, no model
wrapper, and no other `src` module is imported (Requirement 15.6).

Every fixture is an in-memory `pandas.DataFrame` built from Python
literals, a hand-written stub object, or a small CSV written under
`pytest`'s `tmp_path`, and no fixture exceeds 40 rows. The stub corpus
holds no more than 5 documents (Requirement 15.8). No test reads a file
under `results/` or `data/`, loads a real model, loads a real tokenizer,
reads the real BEIR SciFact corpus, or makes a network call. No test
calls `resolve_model_limits`, `load_covariate_inputs`'s success path, or
`resolve_effective_max_sequence_length` -- limits arrive as a plain
`{retriever_name: int}` dict, which is what makes Requirement 15.10 hold
without patching anything or gating a test on the local cache being
present (Requirement 15.10). This module declares no such gate anywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from src.errors import (
    ContrastQuerySetError,
    CovariateInputError,
    FailureBucketAssertionError,
    FailureBucketInputError,
    FailureBucketWriteError,
)
from src.failure_buckets import (
    CONTRAST_BUCKET_ORDER,
    COVARIATE_NAMES,
    DENSE_MODEL_NAMES,
    EXCEEDS_TEXT,
    FAILURE_BUCKET_COLUMNS,
    FAILURE_BUCKET_ORDER,
    REQUIRED_COLUMNS,
    WITHIN_TEXT,
    CovariateInputs,
    assert_covariates_run_independent,
    assert_fraction_sums,
    assert_local_cache_present,
    assert_no_separator_collision,
    assert_partition_total,
    assert_unique_pairs,
    assign_contrast_bucket,
    assign_failure_bucket,
    attach_covariates,
    build_contrast_counts,
    build_declared_contrast_set,
    build_failure_bucket_counts,
    build_failure_buckets,
    build_run_counts,
    compute_token_length_covariates,
    covariate_column,
    is_answered,
    load_per_query,
    main,
    make_composite_run_id,
    max_relevant_doc_token_len,
    model_tag,
    write_failure_buckets,
)

# ---------------------------------------------------------------------------
# Shared fixture shape (matches load_per_query's dtype mapping).
# ---------------------------------------------------------------------------

_COLUMNS = [
    "run_id", "retriever", "chunking_strategy", "query_id",
    "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
    "ndcg_at_10", "mrr_at_10", "num_judged_relevant",
]


def _frame(rows: List[tuple]) -> pandas.DataFrame:
    """Builds a Per_Query_Report-shaped frame from literal tuples, with
    run_id/retriever/chunking_strategy/query_id as str -- matching what
    load_per_query's dtype mapping produces from the real CSV."""
    frame = pandas.DataFrame(rows, columns=_COLUMNS)
    for column in ("run_id", "retriever", "chunking_strategy", "query_id"):
        frame[column] = frame[column].astype(str)
    return frame


def _write_per_query(tmp_path: Path, frame: pandas.DataFrame) -> Path:
    path = tmp_path / "per_query.csv"
    frame.to_csv(path, index=False)
    return path


def _sweep_config_yaml(tmp_path: Path, data_dir: Path) -> Path:
    """Writes a minimal Sweep_Config YAML whose only field main() reads
    is data_dir -- load_sweep_config still validates the whole file, so
    this carries every required field with the real project's fixed
    values."""
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
seed: 42
chunking_strategies:
  - name: whole_document
  - name: fixed_window
    window_size: 200
    stride: 50
  - name: sentence_window
    sentences_per_chunk: 3
    max_chunk_tokens: 256
cutoffs: [1, 5, 10, 20]
retrievers:
  - name: bm25
    type: bm25
    k1: 1.5
    b: 0.75
    tokenizer: regex_word
    lowercase: true
    stopwords: none
    stemming: none
  - name: all-MiniLM-L6-v2
    type: dense
    model_name: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 32
  - name: bge-small-en-v1.5
    type: dense
    model_name: BAAI/bge-small-en-v1.5
    batch_size: 32
data_dir: {data_dir.as_posix()}
output_path: results/sweep.csv
""",
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# Covariate fixtures (Requirement 15.8): hand-written stubs, following the
# _ZeroChunkStubChunker pattern tests/test_chunking.py established.
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Hand-written stub tokenizer (not a real one, and never loaded
    from data/) exercising compute_token_length_covariates' arithmetic.

    Implements exactly the surface count_tokens calls --
    __call__(text, add_special_tokens=..., truncation=...) returning a
    mapping with an "input_ids" key -- so the real, committed
    count_tokens runs against it unmodified rather than being stubbed
    out too. Tokenizes on whitespace and adds two special tokens, so a
    fixture's expected count is `len(text.split()) + 2`, readable by
    eye from the fixture text.

    Deliberately does NOT implement model_max_length: nothing in the
    tested path resolves a limit from a tokenizer -- limits arrive as a
    plain {retriever_name: int} dict -- so an attribute error here would
    mean the pure/impure split had been breached (Requirement 15.10).
    """

    special_tokens_added = 2

    def __call__(self, text, add_special_tokens=True, truncation=False):
        assert add_special_tokens is True  # Requirement 16.4
        assert truncation is False  # Requirement 16.4
        ids = list(range(len(text.split()) + self.special_tokens_added))
        return {"input_ids": ids}


# Stub corpus: 5 documents, well inside Requirement 15.8's ceiling.
# Deliberately includes a long one (over the stub limit) and a short one
# (under it), and documents with an empty title/text so
# format_document_text's `title + " " + text` composition is exercised
# on the shapes it actually meets in SciFact.
_STUB_CORPUS = {
    "d1": {"title": "Short title", "text": "one two three"},
    "d2": {"title": "", "text": " ".join(["w"] * 40)},
    "d3": {"title": "Mid", "text": " ".join(["w"] * 8)},
    "d4": {"title": "Long", "text": " ".join(["w"] * 200)},
    "d5": {"title": "Empty body", "text": ""},
}  # type: Dict[str, Dict[str, str]]

# Stub qrels: q1 has two judged-relevant documents (one long enough to
# exceed the stub limit), q2 has one, q3 has an entry whose only score is
# 0 -- which judged_relevant_docs' `> 0` condition must treat as NO
# judged-relevant document -- and q4 is absent from qrels entirely.
# q3 and q4 are the two shapes Requirement 15.9's sentinel test needs.
_STUB_QRELS = {
    "q1": {"d1": 1, "d4": 2},
    "q2": {"d3": 1},
    "q3": {"d2": 0},
}  # type: Dict[str, Dict[str, int]]

_STUB_QUERIES = {
    "q1": "claim one two",
    "q2": "claim two",
    "q3": "claim three",
    "q4": "claim four",
}  # type: Dict[str, str]

# Limits as plain ints, never resolved from a model (Requirement 15.10).
# Deliberately DIFFERENT per model, so a test would fail if the code
# applied one model's limit to the other model's column.
_STUB_LIMITS = {"all-MiniLM-L6-v2": 20, "bge-small-en-v1.5": 60}  # type: Dict[str, int]


def _stub_covariate_inputs() -> CovariateInputs:
    return CovariateInputs(
        corpus=_STUB_CORPUS,
        queries=_STUB_QUERIES,
        qrels=_STUB_QRELS,
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,
    )


# ---------------------------------------------------------------------------
# The 3-Run_Id x 4-query_id fixture used by several end-to-end tests.
# ---------------------------------------------------------------------------


def _three_run_fixture_rows() -> List[tuple]:
    """7 Run_Ids is the MINIMUM `build_declared_contrast_set` accepts
    without raising `ContrastQuerySetError`: the Reference_Run, plus
    both `DENSE_RETRIEVERS` entries each crossed with
    `whole_document`/`fixed_window`/`sentence_window` (group (b) names
    all six of those Run_Ids unconditionally, regardless of how many
    are actually observed -- "a declared contrast over a run that was
    never swept has no partition"). 7 Run_Ids x 4 query_ids = 28 rows,
    still well under the 40-row ceiling. (Function name kept for the
    existing call sites; it now returns 28 rows over 7 Run_Ids rather
    than a literal "3 runs".)"""
    rows = []
    for run_id, retriever, chunking in [
        ("bm25__whole_document", "bm25", "whole_document"),
        ("all-MiniLM-L6-v2__whole_document", "all-MiniLM-L6-v2", "whole_document"),
        ("all-MiniLM-L6-v2__fixed_window", "all-MiniLM-L6-v2", "fixed_window"),
        ("all-MiniLM-L6-v2__sentence_window", "all-MiniLM-L6-v2", "sentence_window"),
        ("bge-small-en-v1.5__whole_document", "bge-small-en-v1.5", "whole_document"),
        ("bge-small-en-v1.5__fixed_window", "bge-small-en-v1.5", "fixed_window"),
        ("bge-small-en-v1.5__sentence_window", "bge-small-en-v1.5", "sentence_window"),
    ]:
        for query_id in ["1", "2", "3", "4"]:
            rows.append(
                (
                    run_id, retriever, chunking, query_id,
                    1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 1. test_failure_bucket_predicates_cover_all_four_buckets (Requirement 15.1)
# ---------------------------------------------------------------------------


def test_failure_bucket_predicates_cover_all_four_buckets():
    frame = _frame(
        [
            ("r1", "bm25", "whole_document", "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
            ("r1", "bm25", "whole_document", "2", 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "3", 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 2),
            ("r1", "bm25", "whole_document", "4", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
        ]
    )
    buckets = build_failure_buckets(frame)
    expected = {"1": "total_miss", "2": "mis_ranked", "3": "partial_recall", "4": "full_success"}
    actual = dict(zip(buckets["query_id"], buckets["bucket"]))
    assert actual == expected

    # Directly against the scalar predicate too, without any frame.
    assert assign_failure_bucket(0.0, 0.0, 1) == "total_miss"
    assert assign_failure_bucket(0.0, 0.5, 1) == "mis_ranked"
    assert assign_failure_bucket(1.0, 0.5, 2) == "partial_recall"
    assert assign_failure_bucket(1.0, 1.0, 1) == "full_success"


# ---------------------------------------------------------------------------
# 2. test_contrast_bucket_rules_cover_all_four_buckets (Requirement 15.2)
# ---------------------------------------------------------------------------


def test_contrast_bucket_rules_cover_all_four_buckets():
    frame = _frame(
        [
            ("bm25__whole_document", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("bm25__whole_document", "bm25", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1),
            ("bm25__whole_document", "bm25", "whole_document", "3", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1),
            ("bm25__whole_document", "bm25", "whole_document", "4", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("dense__whole_document", "dense", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1),
            ("dense__whole_document", "dense", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("dense__whole_document", "dense", "whole_document", "3", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1),
            ("dense__whole_document", "dense", "whole_document", "4", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
        ]
    )
    # query 1: bm25 answered, dense missed -> a_only
    # query 2: bm25 missed, dense answered -> b_only
    # query 3: both missed -> both_miss
    # query 4: both answered -> both_answer
    contrast_counts = build_contrast_counts(
        frame, [("bm25__whole_document", "dense__whole_document")]
    )
    counts_by_bucket = dict(zip(contrast_counts["bucket"], contrast_counts["count"]))
    assert counts_by_bucket == {"a_only": 1, "b_only": 1, "both_miss": 1, "both_answer": 1}

    assert assign_contrast_bucket(0.5, 0.0) == "a_only"
    assert assign_contrast_bucket(0.0, 0.5) == "b_only"
    assert assign_contrast_bucket(0.0, 0.0) == "both_miss"
    assert assign_contrast_bucket(0.5, 0.5) == "both_answer"


# ---------------------------------------------------------------------------
# 3. test_totality_assertion_failure_writes_neither_report (Requirement 15.3)
# ---------------------------------------------------------------------------


def test_totality_assertion_failure_writes_neither_report(tmp_path):
    frame = _frame(
        [
            ("r1", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "3", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "4", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
        ]
    )
    per_query = _write_per_query(tmp_path, frame)
    buckets_out = tmp_path / "failure_buckets.csv"
    counts_out = tmp_path / "failure_bucket_counts.csv"
    buckets_out.write_bytes(b"SENTINEL\n")
    counts_out.write_bytes(b"SENTINEL\n")

    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(counts_out),
        ]
    )
    assert rc == 1
    assert buckets_out.read_bytes() == b"SENTINEL\n"
    assert counts_out.read_bytes() == b"SENTINEL\n"


def test_totality_assertion_message_names_summed_and_expected():
    """Companion: assert_partition_total's message names both the
    observed sum and the expected total when they disagree."""
    with pytest.raises(FailureBucketAssertionError) as excinfo:
        assert_partition_total(
            "r1", {"total_miss": 1, "mis_ranked": 1, "partial_recall": 0, "full_success": 0},
            5, FAILURE_BUCKET_ORDER,
        )
    message = str(excinfo.value)
    assert "r1" in message
    assert "2" in message  # observed sum
    assert "5" in message  # expected total


# ---------------------------------------------------------------------------
# 4. test_two_invocations_produce_byte_identical_reports (Requirement 15.4)
# ---------------------------------------------------------------------------


def test_two_invocations_produce_byte_identical_reports(tmp_path):
    frame = _frame(_three_run_fixture_rows())
    covariate_inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,
    )

    run_a_dir = tmp_path / "a"
    run_b_dir = tmp_path / "b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()
    per_query = _write_per_query(tmp_path, frame)

    for out_dir in (run_a_dir, run_b_dir):
        rc = main(
            [
                "--per-query", str(per_query),
                "--buckets-out", str(out_dir / "failure_buckets.csv"),
                "--counts-out", str(out_dir / "failure_bucket_counts.csv"),
            ],
            covariate_inputs=covariate_inputs,
        )
        assert rc == 0

    assert (run_a_dir / "failure_buckets.csv").read_bytes() == (
        run_b_dir / "failure_buckets.csv"
    ).read_bytes()
    assert (run_a_dir / "failure_bucket_counts.csv").read_bytes() == (
        run_b_dir / "failure_bucket_counts.csv"
    ).read_bytes()


def test_shuffled_input_produces_identical_bytes(tmp_path):
    """Companion: the same rows in reverse order produce the same
    bytes -- the artifact's order comes from the declared sort key,
    not from the input (Property 8)."""
    rows = _three_run_fixture_rows()
    covariate_inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,
    )

    forward_dir = tmp_path / "forward"
    reversed_dir = tmp_path / "reversed"
    forward_dir.mkdir()
    reversed_dir.mkdir()

    forward_path = _write_per_query(tmp_path, _frame(rows))
    reversed_path = tmp_path / "per_query_reversed.csv"
    _frame(list(reversed(rows))).to_csv(reversed_path, index=False)

    rc_a = main(
        [
            "--per-query", str(forward_path),
            "--buckets-out", str(forward_dir / "failure_buckets.csv"),
            "--counts-out", str(forward_dir / "failure_bucket_counts.csv"),
        ],
        covariate_inputs=covariate_inputs,
    )
    rc_b = main(
        [
            "--per-query", str(reversed_path),
            "--buckets-out", str(reversed_dir / "failure_buckets.csv"),
            "--counts-out", str(reversed_dir / "failure_bucket_counts.csv"),
        ],
        covariate_inputs=covariate_inputs,
    )
    assert rc_a == 0 and rc_b == 0
    assert (forward_dir / "failure_buckets.csv").read_bytes() == (
        reversed_dir / "failure_buckets.csv"
    ).read_bytes()
    assert (forward_dir / "failure_bucket_counts.csv").read_bytes() == (
        reversed_dir / "failure_bucket_counts.csv"
    ).read_bytes()


# ---------------------------------------------------------------------------
# 5. test_counts_run_id_and_bucket_combinations_are_unique (Requirement 15.5)
# ---------------------------------------------------------------------------


def test_counts_run_id_and_bucket_combinations_are_unique(tmp_path):
    frame = _frame(_three_run_fixture_rows())
    covariate_inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,
    )
    per_query = _write_per_query(tmp_path, frame)
    counts_out = tmp_path / "failure_bucket_counts.csv"
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(tmp_path / "failure_buckets.csv"),
            "--counts-out", str(counts_out),
        ],
        covariate_inputs=covariate_inputs,
    )
    assert rc == 0
    counts = pandas.read_csv(counts_out, dtype=str)
    assert counts.duplicated(subset=["run_id", "bucket"]).sum() == 0

    # Property 6 companions: disjoint bucket name sets, and each run_id
    # kind carries only its own bucket names.
    assert set(FAILURE_BUCKET_ORDER) & set(CONTRAST_BUCKET_ORDER) == set()
    for run_id, bucket in zip(counts["run_id"], counts["bucket"]):
        if "|vs|" in run_id:
            assert bucket in CONTRAST_BUCKET_ORDER
        else:
            assert bucket in FAILURE_BUCKET_ORDER


# ---------------------------------------------------------------------------
# 6. test_bucket_assigner_uses_no_network_no_model_no_data_dir
#    (Requirements 15.6, 15.10)
# ---------------------------------------------------------------------------


def test_bucket_assigner_uses_no_network_no_model_no_data_dir():
    import subprocess

    import src.failure_buckets as module

    # Checked in a FRESH subprocess, not the current pytest process:
    # other tests in this module legitimately call
    # compute_token_length_covariates, which triggers the deferred
    # _import_tokenizer_helpers() import and therefore populates
    # sys.modules with transformers/sentence_transformers/torch for
    # the rest of this test session -- that pollution is expected and
    # is not a regression. What this test checks is the build-time
    # property "importing src.failure_buckets alone pulls none of
    # them in", which only a clean-process import can answer, exactly
    # like Task 4's own done check.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.failure_buckets; "
            "heavy = {'beir', 'sentence_transformers', 'transformers', 'torch', 'huggingface_hub'}; "
            "leaked = sorted(heavy & set(sys.modules)); "
            "assert not leaked, leaked; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout

    source = Path(module.__file__).read_text(encoding="utf-8")
    top_level_imports = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    assert "requests" not in top_level_imports
    assert "urllib.request" not in top_level_imports

    # No fixture path in this test module resolves under data/.
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert not re.search(r"""(read_csv|open|Path)\(['"]data/""", test_source)

    # Requirement 15.10: no new skip-gated real-corpus/real-tokenizer
    # test. Forbidden markers are assembled from fragments so this
    # very assertion does not itself introduce the literal substring
    # into the file (which a grep-based done-check also scans for).
    forbidden_markers = [
        "pytest" + "mark",
        "skip" + "if",
        "skip_" + "if",
        "unittest" + "." + "mock",
        "monkey" + "patch",
    ]
    for marker in forbidden_markers:
        assert marker not in test_source, marker


# ---------------------------------------------------------------------------
# 7. test_covariate_computation_over_stub_corpus_and_stub_tokenizer
#    (Requirement 15.8)
# ---------------------------------------------------------------------------


def test_covariate_computation_over_stub_corpus_and_stub_tokenizer():
    inputs = _stub_covariate_inputs()
    covariates = compute_token_length_covariates(["q1", "q2", "q3"], inputs)
    by_query = {row["query_id"]: row for row in covariates.to_dict(orient="records")}

    # q1: two judged-relevant docs (d1, d4). query text "claim one two"
    # -> 3 words + 2 special tokens = 5.
    q1 = by_query["q1"]
    assert q1["query_token_len__all-MiniLM-L6-v2"] == 5
    assert q1["query_token_len__bge-small-en-v1_5"] == 5
    # d1: "Short title" + " " + "one two three" -> 5 words + 2 = 7.
    # d4: "Long" + " " + 200 w's -> 201 words + 2 = 203. Max must be d4's,
    # never d1's -- the maximum, not the first or the last.
    assert q1["max_relevant_doc_token_len__all-MiniLM-L6-v2"] == 203
    assert q1["max_relevant_doc_token_len__bge-small-en-v1_5"] == 203
    # all-MiniLM-L6-v2 limit is 20: 203 > 20 -> True (exceeds).
    assert bool(q1["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"]) is True
    # bge-small-en-v1.5 limit is 60: 203 > 60 -> also True here.
    assert bool(q1["any_relevant_doc_exceeds_limit__bge-small-en-v1_5"]) is True

    # q2: one judged-relevant doc (d3): "Mid" + " " + 8 w's -> 9 words + 2
    # = 11. Max over one element is not special-cased.
    q2 = by_query["q2"]
    assert q2["max_relevant_doc_token_len__all-MiniLM-L6-v2"] == 11
    assert q2["max_relevant_doc_token_len__bge-small-en-v1_5"] == 11
    assert bool(q2["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"]) is False  # within (11 <= 20)
    assert bool(q2["any_relevant_doc_exceeds_limit__bge-small-en-v1_5"]) is False

    # q3: qrels entry {"d2": 0} -- relevance-0, so NO judged-relevant doc.
    # Both sentinel-bearing covariates must be missing (the sentinel),
    # while query_token_len is still a real count: "claim three" -> 2 + 2
    # = 4. pandas coerces a None-mixed-with-int column to float64/NaN,
    # so the missing check uses pandas.isna() rather than `is None`.
    q3 = by_query["q3"]
    assert q3["query_token_len__all-MiniLM-L6-v2"] == 4
    assert pandas.isna(q3["max_relevant_doc_token_len__all-MiniLM-L6-v2"])
    assert q3["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"] is None
    assert pandas.isna(q3["max_relevant_doc_token_len__bge-small-en-v1_5"])
    assert q3["any_relevant_doc_exceeds_limit__bge-small-en-v1_5"] is None


def test_covariate_computation_disagreement_between_models():
    """Companion: two models with different limits must disagree on
    the SAME document -- catches one model's limit applied to both
    columns."""
    inputs = CovariateInputs(
        corpus={"d1": {"title": "T", "text": " ".join(["w"] * 30)}},  # 31 words + 2 = 33
        queries={"q1": "claim"},
        qrels={"q1": {"d1": 1}},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,  # all-MiniLM-L6-v2: 20, bge-small-en-v1.5: 60
    )
    covariates = compute_token_length_covariates(["q1"], inputs)
    row = covariates.iloc[0]
    assert row["max_relevant_doc_token_len__all-MiniLM-L6-v2"] == 33
    assert bool(row["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"]) is True  # 33 > 20
    assert bool(row["any_relevant_doc_exceeds_limit__bge-small-en-v1_5"]) is False  # 33 <= 60


def test_covariate_computation_raises_for_query_absent_from_queries():
    inputs = _stub_covariate_inputs()
    with pytest.raises(CovariateInputError) as excinfo:
        compute_token_length_covariates(["q_missing"], inputs)
    assert "q_missing" in str(excinfo.value)


def test_max_relevant_doc_token_len_pure_function():
    assert max_relevant_doc_token_len({}, []) is None
    assert max_relevant_doc_token_len({"d": 0}, ["d"]) == 0  # genuine zero-length doc, not None
    assert max_relevant_doc_token_len({"a": 7, "b": 3}, ["a", "b"]) == 7


# ---------------------------------------------------------------------------
# 8. test_missing_judgment_records_sentinel_not_numeric_zero (Requirement 15.9)
# ---------------------------------------------------------------------------


def test_missing_judgment_records_sentinel_not_numeric_zero(tmp_path):
    """Covariate inputs are the stubs above, injected by constructing
    the frames directly and calling build_failure_buckets +
    attach_covariates, not by patching the loader or routing through
    main() -- so this test needs no Declared_Contrast_Set at all."""
    per_query_frame = _frame(
        [
            ("bm25__whole_document", "bm25", "whole_document", "q1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 2),
            ("bm25__whole_document", "bm25", "whole_document", "q3", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0),
            ("bm25__whole_document", "bm25", "whole_document", "q4", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0),
        ]
    )
    failure_buckets = build_failure_buckets(per_query_frame)
    inputs = _stub_covariate_inputs()
    covariates = compute_token_length_covariates(["q1", "q3", "q4"], inputs)
    attached = attach_covariates(failure_buckets, covariates)

    buckets_out = tmp_path / "failure_buckets.csv"
    write_failure_buckets(attached, buckets_out)

    raw_text = buckets_out.read_text(encoding="utf-8")
    lines = raw_text.strip().splitlines()
    header = lines[0].split(",")
    max_len_idx = header.index("max_relevant_doc_token_len__all-MiniLM-L6-v2")
    exceeds_idx = header.index("any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2")

    for line in lines[1:]:
        cells = line.split(",")
        query_id = cells[header.index("query_id")]
        if query_id in ("q3", "q4"):
            assert cells[max_len_idx] == "NA", (query_id, cells[max_len_idx])
            assert cells[exceeds_idx] == "NA", (query_id, cells[exceeds_idx])
            for forbidden in ("0", "0.0", "within", "exceeds", "true", "false", "True", "False", ""):
                assert cells[max_len_idx] != forbidden
                assert cells[exceeds_idx] != forbidden


# ---------------------------------------------------------------------------
# 9. test_covariates_are_identical_across_a_query_id_rows
#    (Requirements 16.9, 6.9)
# ---------------------------------------------------------------------------


def test_covariates_are_identical_across_a_query_id_rows(tmp_path):
    frame = _frame(_three_run_fixture_rows())
    inputs = CovariateInputs(
        corpus={}, queries={"1": "one", "2": "two words", "3": "three word text", "4": "q"},
        qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS},
        limits=_STUB_LIMITS,
    )
    per_query = _write_per_query(tmp_path, frame)
    buckets_out = tmp_path / "failure_buckets.csv"
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(tmp_path / "failure_bucket_counts.csv"),
        ],
        covariate_inputs=inputs,
    )
    assert rc == 0

    written = pandas.read_csv(buckets_out, dtype=str, keep_default_na=False, na_values=[])
    covariate_columns = [c for c in FAILURE_BUCKET_COLUMNS if c not in (
        "run_id", "retriever", "chunking_strategy", "query_id", "bucket", "num_judged_relevant",
    )]
    for query_id, group in written.groupby("query_id"):
        for column in covariate_columns:
            assert group[column].nunique() == 1, (query_id, column, group[column].tolist())


def test_attach_covariates_raises_on_duplicated_query_id_in_covariate_frame():
    """Companion: a covariate frame with a duplicated query_id must
    raise via assert_unique_pairs' fan-out guard rather than silently
    fanning out the joined frame."""
    fb = build_failure_buckets(
        _frame(
            [
                ("r1", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
                ("r1", "bm25", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ]
        )
    )
    duplicated_covariates = pandas.DataFrame(
        [
            {"query_id": "1", "query_token_len__all-MiniLM-L6-v2": 3,
             "max_relevant_doc_token_len__all-MiniLM-L6-v2": 5,
             "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": False,
             "query_token_len__bge-small-en-v1_5": 3,
             "max_relevant_doc_token_len__bge-small-en-v1_5": 5,
             "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": False},
            {"query_id": "1", "query_token_len__all-MiniLM-L6-v2": 4,
             "max_relevant_doc_token_len__all-MiniLM-L6-v2": 6,
             "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": False,
             "query_token_len__bge-small-en-v1_5": 4,
             "max_relevant_doc_token_len__bge-small-en-v1_5": 6,
             "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": False},
            {"query_id": "2", "query_token_len__all-MiniLM-L6-v2": 3,
             "max_relevant_doc_token_len__all-MiniLM-L6-v2": 5,
             "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": False,
             "query_token_len__bge-small-en-v1_5": 3,
             "max_relevant_doc_token_len__bge-small-en-v1_5": 5,
             "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": False},
        ]
    )
    with pytest.raises(FailureBucketAssertionError):
        attach_covariates(fb, duplicated_covariates)


# ---------------------------------------------------------------------------
# 10. test_bucket_level_covariate_count_selector_resolves
#     (Requirements 7.4, 8.1, 12.7)
# ---------------------------------------------------------------------------


def test_bucket_level_covariate_count_selector_resolves(tmp_path):
    from src.verify_writeup_numbers import _resolve_csv_reference

    # Fixture A: any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2 holds
    # only exceeds/within, no "NA" anywhere in that column.
    frame_a = pandas.DataFrame(
        [
            {
                "run_id": "bm25__whole_document", "retriever": "bm25",
                "chunking_strategy": "whole_document", "query_id": "1",
                "bucket": "total_miss", "num_judged_relevant": 1,
                "query_token_len__all-MiniLM-L6-v2": 5,
                "max_relevant_doc_token_len__all-MiniLM-L6-v2": 30,
                "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": "exceeds",
                "query_token_len__bge-small-en-v1_5": 5,
                "max_relevant_doc_token_len__bge-small-en-v1_5": 30,
                "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": "within",
            },
            {
                "run_id": "bm25__whole_document", "retriever": "bm25",
                "chunking_strategy": "whole_document", "query_id": "2",
                "bucket": "full_success", "num_judged_relevant": 1,
                "query_token_len__all-MiniLM-L6-v2": 4,
                "max_relevant_doc_token_len__all-MiniLM-L6-v2": 10,
                "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": "within",
                "query_token_len__bge-small-en-v1_5": 4,
                "max_relevant_doc_token_len__bge-small-en-v1_5": 10,
                "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": "within",
            },
        ]
    )
    # Fixture B: same shape but with one "NA" row added to that column.
    frame_b = pandas.concat(
        [
            frame_a,
            pandas.DataFrame(
                [
                    {
                        "run_id": "bm25__whole_document", "retriever": "bm25",
                        "chunking_strategy": "whole_document", "query_id": "3",
                        "bucket": "total_miss", "num_judged_relevant": 0,
                        "query_token_len__all-MiniLM-L6-v2": 6,
                        "max_relevant_doc_token_len__all-MiniLM-L6-v2": "NA",
                        "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2": "NA",
                        "query_token_len__bge-small-en-v1_5": 6,
                        "max_relevant_doc_token_len__bge-small-en-v1_5": "NA",
                        "any_relevant_doc_exceeds_limit__bge-small-en-v1_5": "NA",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    reference = (
        "run_id=bm25__whole_document,bucket=total_miss,"
        "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2=exceeds.__count__"
    )
    for frame, label in ((frame_a, "a"), (frame_b, "b")):
        count = _resolve_csv_reference(frame, reference, f"failure_buckets_{label}.csv")
        assert count == 1.0, (label, count)

    # No header cell of either fixture contains a "." (Requirement 6.7).
    for frame in (frame_a, frame_b):
        assert not any("." in column for column in frame.columns)

    # Complement-triple-sum: exceeds-count + within-count == bucket's
    # own total-miss count for that run_id.
    complement_reference = (
        "run_id=bm25__whole_document,bucket=total_miss,"
        "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2=within.__count__"
    )
    complement_count = _resolve_csv_reference(
        frame_a, complement_reference, "failure_buckets_a.csv"
    )
    total_miss_count = float(
        len(frame_a[(frame_a["run_id"] == "bm25__whole_document") & (frame_a["bucket"] == "total_miss")])
    )
    exceeds_count = _resolve_csv_reference(frame_a, reference, "failure_buckets_a.csv")
    assert exceeds_count + complement_count == total_miss_count

    # A single-query two-filter covariate read resolves to exactly one value.
    single_value = _resolve_csv_reference(
        frame_a, "run_id=bm25__whole_document,query_id=1.query_token_len__all-MiniLM-L6-v2",
        "failure_buckets_a.csv",
    )
    assert single_value == 5.0


# ---------------------------------------------------------------------------
# Property tests.
# ---------------------------------------------------------------------------


def _reference_failure_bucket_ladder(
    recall_at_1: float, recall_at_20: float, num_judged_relevant: int
) -> str:
    """Independently written full-three-clause ladder, matching
    Requirement 3 Criterion 1's wording verbatim (including the clause
    assign_failure_bucket's own implementation drops as redundant), so
    the property test checks the simplification rather than assuming
    it."""
    if recall_at_20 == 0:
        return "total_miss"
    if recall_at_1 == 0:
        return "mis_ranked"
    if num_judged_relevant > 1 and recall_at_20 > 0 and recall_at_20 < 1:
        return "partial_recall"
    return "full_success"


@settings(max_examples=100)
@given(
    recall_at_1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    recall_at_20=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    num_judged_relevant=st.integers(min_value=1, max_value=20),
)
@example(recall_at_1=0.0, recall_at_20=0.0, num_judged_relevant=1)
@example(recall_at_1=1.0, recall_at_20=1.0, num_judged_relevant=1)
@example(recall_at_1=0.5, recall_at_20=0.5, num_judged_relevant=1)
@example(recall_at_1=0.5, recall_at_20=0.5, num_judged_relevant=2)
def test_property_1_failure_bucket_is_total_and_first_match(
    recall_at_1, recall_at_20, num_judged_relevant
):
    """Feature: analysis-writeup, Property 1: Failure_Bucket assignment
    is total, exclusive, and first-match."""
    result = assign_failure_bucket(recall_at_1, recall_at_20, num_judged_relevant)
    assert result in FAILURE_BUCKET_ORDER
    assert result == _reference_failure_bucket_ladder(
        recall_at_1, recall_at_20, num_judged_relevant
    )


@settings(max_examples=100)
@given(
    ndcg_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ndcg_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@example(ndcg_a=0.0, ndcg_b=0.0)
def test_property_2_contrast_bucket_is_total_and_matches_truth_table(ndcg_a, ndcg_b):
    """Feature: analysis-writeup, Property 2: Contrast_Bucket assignment
    is total, exclusive, and determined by the answered/missed truth
    table."""
    result = assign_contrast_bucket(ndcg_a, ndcg_b)
    assert result in CONTRAST_BUCKET_ORDER
    a, b = is_answered(ndcg_a), is_answered(ndcg_b)
    if a and not b:
        expected = "a_only"
    elif b and not a:
        expected = "b_only"
    elif not a and not b:
        expected = "both_miss"
    else:
        expected = "both_answer"
    assert result == expected


@st.composite
def _small_frame_strategy(draw):
    num_runs = draw(st.integers(min_value=2, max_value=3))
    num_queries = draw(st.integers(min_value=3, max_value=8))
    run_ids = [f"run{i}" for i in range(num_runs)]
    query_ids = [str(i) for i in range(num_queries)]
    rows = []
    for run_id in run_ids:
        for query_id in query_ids:
            recall_1 = draw(st.sampled_from([0.0, 1.0]))
            recall_20 = draw(st.sampled_from([0.0, 0.5, 1.0]))
            njr = draw(st.integers(min_value=1, max_value=3))
            rows.append(
                (run_id, run_id, "whole_document", query_id, recall_1, recall_1,
                 recall_1, recall_20, 0.5, 0.5, njr)
            )
    return _frame(rows)


@settings(max_examples=100)
@given(frame=_small_frame_strategy())
def test_property_4_partitions_are_total_over_generated_frames(frame):
    """Feature: analysis-writeup, Property 4: Both partitions are
    total, and every run_id's fractions sum to one."""
    failure_buckets = build_failure_buckets(frame)
    assert_unique_pairs(failure_buckets)
    run_counts = build_run_counts(failure_buckets)
    for run_id, group in run_counts.groupby("run_id"):
        expected_total = frame[frame["run_id"] == run_id]["query_id"].nunique()
        assert group["count"].sum() == expected_total
        assert abs(group["fraction"].sum() - 1.0) <= 1e-9


@settings(max_examples=100)
@given(
    doc_lens=st.dictionaries(
        st.text(min_size=1, max_size=5), st.integers(min_value=0, max_value=1000), max_size=10
    ),
    subset_indices=st.lists(st.integers(min_value=0, max_value=9), max_size=10),
)
def test_property_14_max_relevant_doc_token_len_is_a_maximum_or_none(doc_lens, subset_indices):
    """Feature: analysis-writeup, Property 14: The covariates are
    run-independent, sentinel-not-zero, limit-from-configuration, and
    untruncated -- this test covers the pure max-or-None half."""
    keys = list(doc_lens.keys())
    subset = [keys[i % len(keys)] for i in subset_indices] if keys else []
    result = max_relevant_doc_token_len(doc_lens, subset)
    if not subset:
        assert result is None
    else:
        assert result in {doc_lens[d] for d in subset}
        assert all(result >= doc_lens[d] for d in subset)


# ---------------------------------------------------------------------------
# Remaining fixture tests from the design's Testing Strategy.
# ---------------------------------------------------------------------------


def test_declared_contrast_set_is_duplicate_free_and_correctly_ordered():
    run_ids = [
        "bm25__whole_document", "bm25__fixed_window",
        "all-MiniLM-L6-v2__whole_document", "all-MiniLM-L6-v2__fixed_window",
        "all-MiniLM-L6-v2__sentence_window",
        "bge-small-en-v1.5__whole_document", "bge-small-en-v1.5__fixed_window",
        "bge-small-en-v1.5__sentence_window",
    ]
    pairs = build_declared_contrast_set(run_ids)
    assert len(set(pairs)) == len(pairs)
    family_aligned = [p for p in pairs if p[0] == "bm25__whole_document"]
    cross_strategy = [p for p in pairs if p[0] != "bm25__whole_document"]
    assert len(family_aligned) == len(run_ids) - 1
    for run_a, run_b in cross_strategy:
        retriever_a = run_a.split("__")[0]
        retriever_b = run_b.split("__")[0]
        assert retriever_a == retriever_b
        assert run_a.endswith("__whole_document")


def test_declared_contrast_set_raises_for_missing_run_id():
    with pytest.raises(ContrastQuerySetError):
        build_declared_contrast_set(["all-MiniLM-L6-v2__whole_document"])


def test_failure_buckets_columns_and_passthrough_values():
    frame = _frame(
        [
            ("r1", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 2),
            ("r1", "bm25", "whole_document", "2", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
        ]
    )
    failure_buckets = build_failure_buckets(frame)
    assert not any("." in column for column in failure_buckets.columns)

    joined = failure_buckets.merge(frame, on=["run_id", "query_id"], suffixes=("", "_input"))
    for column in ("retriever", "chunking_strategy", "num_judged_relevant"):
        assert (joined[column] == joined[f"{column}_input"]).all()


def test_covariate_column_names_match_the_tag_rule():
    assert model_tag("bge-small-en-v1.5") == "bge-small-en-v1_5"
    assert model_tag("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
    assert FAILURE_BUCKET_COLUMNS == (
        "run_id",
        "retriever",
        "chunking_strategy",
        "query_id",
        "bucket",
        "num_judged_relevant",
        "query_token_len__all-MiniLM-L6-v2",
        "max_relevant_doc_token_len__all-MiniLM-L6-v2",
        "any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2",
        "query_token_len__bge-small-en-v1_5",
        "max_relevant_doc_token_len__bge-small-en-v1_5",
        "any_relevant_doc_exceeds_limit__bge-small-en-v1_5",
    )


def test_exceedance_boundary_is_strictly_greater_than():
    limit = 20
    at_limit_tokens = limit - 2  # + 2 special tokens = exactly `limit`
    over_limit_tokens = limit - 1  # + 2 special tokens = limit + 1
    inputs_at_limit = CovariateInputs(
        corpus={"d1": {"title": "", "text": " ".join(["w"] * at_limit_tokens)}},
        queries={"q1": "x"},
        qrels={"q1": {"d1": 1}},
        tokenizers={"all-MiniLM-L6-v2": _StubTokenizer()},
        limits={"all-MiniLM-L6-v2": limit},
    )
    row_at_limit = compute_token_length_covariates(["q1"], inputs_at_limit).iloc[0]
    assert row_at_limit["max_relevant_doc_token_len__all-MiniLM-L6-v2"] == limit
    assert bool(row_at_limit["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"]) is False

    inputs_over_limit = CovariateInputs(
        corpus={"d1": {"title": "", "text": " ".join(["w"] * over_limit_tokens)}},
        queries={"q1": "x"},
        qrels={"q1": {"d1": 1}},
        tokenizers={"all-MiniLM-L6-v2": _StubTokenizer()},
        limits={"all-MiniLM-L6-v2": limit},
    )
    row_over_limit = compute_token_length_covariates(["q1"], inputs_over_limit).iloc[0]
    assert row_over_limit["max_relevant_doc_token_len__all-MiniLM-L6-v2"] == limit + 1
    assert bool(row_over_limit["any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2"]) is True


def test_covariate_cells_are_rendered_as_declared(tmp_path):
    per_query_frame = _frame(
        [
            ("r1", "bm25", "whole_document", "q1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("r1", "bm25", "whole_document", "q3", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0),
        ]
    )
    failure_buckets = build_failure_buckets(per_query_frame)
    covariates = compute_token_length_covariates(["q1", "q3"], _stub_covariate_inputs())
    attached = attach_covariates(failure_buckets, covariates)
    buckets_out = tmp_path / "failure_buckets.csv"
    write_failure_buckets(attached, buckets_out)
    lines = buckets_out.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    query_len_idx = header.index("query_token_len__all-MiniLM-L6-v2")
    max_len_idx = header.index("max_relevant_doc_token_len__all-MiniLM-L6-v2")
    exceeds_idx = header.index("any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2")
    forbidden = {"true", "false", "True", "False", "1", "0", "0.0", ""}
    for line in lines[1:]:
        cells = line.split(",")
        assert re.fullmatch(r"\d+", cells[query_len_idx])
        assert re.fullmatch(r"\d+", cells[max_len_idx]) or cells[max_len_idx] == "NA"
        assert cells[exceeds_idx] in (EXCEEDS_TEXT, WITHIN_TEXT, "NA")
        assert cells[exceeds_idx] not in forbidden
        assert cells[max_len_idx] not in forbidden - {"0"} or cells[max_len_idx] == "NA"


def test_query_id_is_written_as_text_in_lexicographic_order(tmp_path):
    qids = ["1", "100", "1012", "0007"]
    per_query_frame = _frame(
        [
            ("r1", "bm25", "whole_document", qid, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1)
            for qid in qids
        ]
    )
    failure_buckets = build_failure_buckets(per_query_frame)
    inputs = CovariateInputs(
        corpus={}, queries={qid: "q" for qid in qids}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS}, limits=_STUB_LIMITS,
    )
    covariates = compute_token_length_covariates(qids, inputs)
    attached = attach_covariates(failure_buckets, covariates)
    buckets_out = tmp_path / "failure_buckets.csv"
    write_failure_buckets(attached, buckets_out)
    written = pandas.read_csv(buckets_out, dtype=str, keep_default_na=False, na_values=[])
    assert list(written["query_id"]) == sorted(["1", "100", "1012", "0007"])
    assert "0007" in set(written["query_id"])


def test_counts_formatting_is_fixed_width(tmp_path):
    frame = _frame(_three_run_fixture_rows())
    per_query = _write_per_query(tmp_path, frame)
    counts_out = tmp_path / "failure_bucket_counts.csv"
    inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS}, limits=_STUB_LIMITS,
    )
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(tmp_path / "failure_buckets.csv"),
            "--counts-out", str(counts_out),
        ],
        covariate_inputs=inputs,
    )
    assert rc == 0
    lines = counts_out.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    count_idx = header.index("count")
    fraction_idx = header.index("fraction")
    for line in lines[1:]:
        cells = line.split(",")
        assert re.fullmatch(r"\d+", cells[count_idx])
        assert re.fullmatch(r"\d\.\d{6}", cells[fraction_idx])


def test_counts_row_order_matches_declared_total_order(tmp_path):
    frame = _frame(_three_run_fixture_rows())
    per_query = _write_per_query(tmp_path, frame)
    counts_out = tmp_path / "failure_bucket_counts.csv"
    inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS}, limits=_STUB_LIMITS,
    )
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(tmp_path / "failure_buckets.csv"),
            "--counts-out", str(counts_out),
        ],
        covariate_inputs=inputs,
    )
    assert rc == 0
    written = pandas.read_csv(counts_out, dtype=str)

    def _key(row):
        group = 1 if "|vs|" in row["run_id"] else 0
        bucket_rank = (
            CONTRAST_BUCKET_ORDER.index(row["bucket"])
            if group == 1
            else FAILURE_BUCKET_ORDER.index(row["bucket"])
        )
        return (group, row["run_id"], bucket_rank)

    keys = [_key(row) for _, row in written.iterrows()]
    assert keys == sorted(keys)


def test_composite_run_id_selector_resolves_to_exactly_one_row(tmp_path):
    from src.verify_writeup_numbers import _resolve_csv_reference

    frame = _frame(_three_run_fixture_rows())
    per_query = _write_per_query(tmp_path, frame)
    counts_out = tmp_path / "failure_bucket_counts.csv"
    inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS}, limits=_STUB_LIMITS,
    )
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(tmp_path / "failure_buckets.csv"),
            "--counts-out", str(counts_out),
        ],
        covariate_inputs=inputs,
    )
    assert rc == 0
    written = pandas.read_csv(counts_out, dtype=str)

    plain_run_id = "bm25__whole_document"
    plain_value = _resolve_csv_reference(
        written, f"run_id={plain_run_id},bucket=full_success.count", "failure_bucket_counts.csv"
    )
    assert isinstance(plain_value, float)

    composite_run_id = "bm25__whole_document|vs|all-MiniLM-L6-v2__whole_document"
    composite_value = _resolve_csv_reference(
        written, f"run_id={composite_run_id},bucket=both_answer.count", "failure_bucket_counts.csv"
    )
    assert isinstance(composite_value, float)


def test_run_id_containing_separator_raises_and_writes_neither_report(tmp_path):
    frame = _frame(
        [
            ("bm25__whole_document|vs|evil", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
        ]
    )
    per_query = _write_per_query(tmp_path, frame)
    buckets_out = tmp_path / "failure_buckets.csv"
    counts_out = tmp_path / "failure_bucket_counts.csv"
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(counts_out),
        ]
    )
    assert rc == 1
    assert not buckets_out.exists()
    assert not counts_out.exists()


def test_missing_local_cache_raises_covariate_input_error_and_writes_nothing(tmp_path):
    frame = _frame(
        [("bm25__whole_document", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1)]
    )
    per_query = _write_per_query(tmp_path, frame)
    buckets_out = tmp_path / "failure_buckets.csv"
    counts_out = tmp_path / "failure_bucket_counts.csv"
    buckets_out.write_bytes(b"SENTINEL\n")
    counts_out.write_bytes(b"SENTINEL\n")

    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    config_path = _sweep_config_yaml(tmp_path, empty_data_dir)

    rc = main(
        [
            "--per-query", str(per_query),
            "--config", str(config_path),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(counts_out),
        ]
    )
    assert rc == 1
    assert buckets_out.read_bytes() == b"SENTINEL\n"
    assert counts_out.read_bytes() == b"SENTINEL\n"
    # Nothing was downloaded: the empty data_dir is still empty.
    assert list(empty_data_dir.iterdir()) == []


def test_asymmetric_query_set_raises_naming_pair_and_query_id(tmp_path):
    frame = _frame(
        [
            ("bm25__whole_document", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("bm25__whole_document", "bm25", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("all-MiniLM-L6-v2__whole_document", "all-MiniLM-L6-v2", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            # query_id "2" absent for the dense run -> asymmetric.
        ]
    )
    per_query = _write_per_query(tmp_path, frame)
    buckets_out = tmp_path / "failure_buckets.csv"
    counts_out = tmp_path / "failure_bucket_counts.csv"
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(counts_out),
        ]
    )
    assert rc == 1
    assert not buckets_out.exists()
    assert not counts_out.exists()


def test_asymmetric_query_set_error_message_contents():
    frame = _frame(
        [
            ("bm25__whole_document", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("bm25__whole_document", "bm25", "whole_document", "2", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
            ("all-MiniLM-L6-v2__whole_document", "all-MiniLM-L6-v2", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1),
        ]
    )
    with pytest.raises(ContrastQuerySetError) as excinfo:
        build_contrast_counts(
            frame, [("bm25__whole_document", "all-MiniLM-L6-v2__whole_document")]
        )
    message = str(excinfo.value)
    assert "bm25__whole_document" in message
    assert "all-MiniLM-L6-v2__whole_document" in message
    assert "2" in message


@pytest.mark.parametrize("column_to_drop", list(REQUIRED_COLUMNS))
def test_missing_input_file_and_missing_columns_raise_input_error(tmp_path, column_to_drop):
    frame = _frame(
        [("bm25__whole_document", "bm25", "whole_document", "1", 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1)]
    )
    frame = frame.drop(columns=[column_to_drop])
    per_query = tmp_path / "per_query.csv"
    frame.to_csv(per_query, index=False)

    with pytest.raises(FailureBucketInputError) as excinfo:
        load_per_query(per_query)
    assert column_to_drop in str(excinfo.value)


def test_missing_input_file_raises_input_error(tmp_path):
    with pytest.raises(FailureBucketInputError):
        load_per_query(tmp_path / "does_not_exist.csv")


def test_successful_run_prints_derived_counts_and_returns_zero(tmp_path, capsys):
    frame = _frame(_three_run_fixture_rows())
    per_query = _write_per_query(tmp_path, frame)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    buckets_out = output_dir / "failure_buckets.csv"
    counts_out = output_dir / "failure_bucket_counts.csv"
    inputs = CovariateInputs(
        corpus={}, queries={"1": "q", "2": "q", "3": "q", "4": "q"}, qrels={},
        tokenizers={name: _StubTokenizer() for name in _STUB_LIMITS}, limits=_STUB_LIMITS,
    )
    rc = main(
        [
            "--per-query", str(per_query),
            "--buckets-out", str(buckets_out),
            "--counts-out", str(counts_out),
        ],
        covariate_inputs=inputs,
    )
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "7 run_id(s)" in out
    assert "4 query_id(s)" in out
    assert "Pair_Contrasts:" in out
    assert "effective_max_sequence_length:" in out
    for retriever_name, limit in _STUB_LIMITS.items():
        assert f"{retriever_name}={limit}" in out
    assert "covariates computed:" in out
    assert 'covariates recorded as "NA":' in out

    created_files = sorted(p.name for p in output_dir.iterdir())
    assert created_files == sorted(["failure_buckets.csv", "failure_bucket_counts.csv"])


def test_limits_are_not_read_from_a_literal_or_a_config():
    import src.failure_buckets as module

    bad = [
        name
        for name, value in vars(module).items()
        if isinstance(value, int) and not isinstance(value, bool) and value in (256, 512)
    ]
    assert not bad, bad
    assert not hasattr(module, "MAX_SEQUENCE_LENGTH")

    parser_help = module._parse_args([]).__dict__
    # No option name matches a limit-ish pattern.
    assert not any(re.search(r"max.*len|limit|seq", key, re.IGNORECASE) for key in parser_help)


def test_verifier_csv_artifacts_includes_both_new_files():
    from src.verify_writeup_numbers import _ALLOWED_COMPUTATIONS, _CSV_ARTIFACTS

    assert "failure_buckets.csv" in _CSV_ARTIFACTS
    assert "failure_bucket_counts.csv" in _CSV_ARTIFACTS
    assert tuple(_ALLOWED_COMPUTATIONS) == (
        "copy", "ratio", "delta", "mean", "percentage", "sum",
        "half_ci_width", "complement_percentage", "wilson_ci_lower", "wilson_ci_upper",
    )


def test_assert_local_cache_present_raises_naming_absent_paths(tmp_path):
    with pytest.raises(CovariateInputError) as excinfo:
        assert_local_cache_present(tmp_path)
    message = str(excinfo.value)
    assert "scifact" in message


def test_assert_no_separator_collision_raises_naming_offender():
    with pytest.raises(FailureBucketAssertionError) as excinfo:
        assert_no_separator_collision(["ok_run", "bad|vs|run"])
    assert "bad|vs|run" in str(excinfo.value)


def test_make_composite_run_id_and_covariate_column_helpers():
    assert make_composite_run_id("a", "b") == "a|vs|b"
    assert covariate_column("query_token_len", "bge-small-en-v1.5") == "query_token_len__bge-small-en-v1_5"
    assert covariate_column("query_token_len", "all-MiniLM-L6-v2") == "query_token_len__all-MiniLM-L6-v2"
