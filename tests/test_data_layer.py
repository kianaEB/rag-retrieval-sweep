"""Requirement 12 test suite: Data_Layer_Tests and
Real_Corpus_End_To_End_Tests -- the first tests in this spec's suite
to exercise the real, cached BEIR SciFact corpus (and, for the
end-to-end test, the real Sweep_Runner orchestration) rather than an
in-memory stub.

Every test in this module is gated by `_local_cache_available()`
(Local_Cache_Availability, Requirement 12.3) via `pytest.mark.skipif`
at module scope: on a clean checkout -- in particular the GitHub
Actions CI environment, which never downloads a dataset or model
weights (Requirement 12.4) -- every test here is **skipped**, not
failed. Both tests execute and pass locally once the cache has been
populated by a prior manual sweep run (Requirement 12.5).

These tests are additive to, and do not replace or modify,
`tests/test_metrics.py`, `tests/test_orchestration.py`,
`tests/test_significance.py`, `tests/test_claim_segmenter.py`,
`tests/test_quarantine_rule.py`, `tests/test_token_length_analysis.py`,
or `tests/test_verify_writeup_numbers.py` (Requirement 12.6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.corpus_loader import configure_caches, load_scifact
from src.report import MISSING
from src.sweep_runner import (
    make_default_chunker_factory,
    make_default_retriever_factory,
    run_sweep,
)
from src.config import BM25RetrieverConfig, SweepConfig, WholeDocumentChunkingConfig

_DATA_DIR = Path("data")
_HF_CACHE_DIR = _DATA_DIR / "hf_cache"

# The two dense-model snapshot directories this spec's full grid needs
# (bm25 needs none, per design.md's Local_Cache_Availability note).
_REQUIRED_MODEL_CACHE_DIRS = (
    _HF_CACHE_DIR / "models--sentence-transformers--all-MiniLM-L6-v2",
    _HF_CACHE_DIR / "models--BAAI--bge-small-en-v1.5",
)


def _local_cache_available() -> bool:
    """Local_Cache_Availability (Requirement 12.3): `data/scifact` and
    every `data/hf_cache/models--*` directory this spec's grid needs
    (`bm25` needs none; `all-MiniLM-L6-v2` and `bge-small-en-v1.5` each
    need their own snapshot directory) are already present."""
    if not (_DATA_DIR / "scifact").is_dir():
        return False
    return all(cache_dir.is_dir() for cache_dir in _REQUIRED_MODEL_CACHE_DIRS)


pytestmark = pytest.mark.skipif(
    not _local_cache_available(),
    reason="requires a local BEIR SciFact + model weight cache; skipped on a clean checkout",
)


def test_load_scifact_against_real_cached_corpus():
    """Data_Layer_Test (Requirement 12.1): calls the real
    `load_scifact(Path("data"))` against the real cached corpus,
    asserting non-zero `num_documents`/`num_queries`/`num_qrel_pairs`
    counts and referential integrity -- exercising the real
    Corpus_Loader, not a stub. `load_scifact` itself already raises
    `CorpusValidationError` if any qrels-referenced document/query ID
    fails to resolve, so a successful return is itself evidence of
    referential integrity; this test additionally re-derives the qrels
    referential-integrity check directly against the returned bundle,
    so the assertion is not solely "load_scifact didn't raise."
    """
    configure_caches(_DATA_DIR)
    bundle, report = load_scifact(_DATA_DIR)

    assert report.num_documents > 0
    assert report.num_queries > 0
    assert report.num_qrel_pairs > 0
    assert report.num_documents == len(bundle.corpus)
    assert report.num_queries == len(bundle.queries)
    assert report.num_qrel_pairs == sum(len(judged) for judged in bundle.qrels.values())

    # Referential integrity, re-derived directly (not merely inferred
    # from load_scifact's own successful return): every qrels-judged
    # document ID resolves against the loaded corpus, and every qrels
    # query ID resolves against the loaded query set.
    for query_id, judged_docs in bundle.qrels.items():
        assert query_id in bundle.queries
        for doc_id in judged_docs:
            assert doc_id in bundle.corpus


def test_sweep_runner_end_to_end_one_combination_against_real_corpus():
    """Real_Corpus_End_To_End_Test (Requirement 12.2): runs the
    Sweep_Runner (via `run_sweep`, with the production
    `retriever_factory`/`chunker_factory`, not stubs) against the real
    BEIR SciFact corpus for one full retriever x Chunking_Strategy
    combination -- `bm25` x `whole_document` (the cheapest combination:
    no dense model load, no tokenizer download, fastest to run
    locally) -- asserting the resulting rows have the correct columns
    and every metric value is either a float in `[0.0, 1.0]` or
    `MISSING`.
    """
    configure_caches(_DATA_DIR)
    bundle, _report = load_scifact(_DATA_DIR)

    config = SweepConfig(
        seed=42,
        chunking_strategies=(WholeDocumentChunkingConfig(name="whole_document"),),
        cutoffs=(1, 5, 10, 20),
        retrievers=(
            BM25RetrieverConfig(
                name="bm25",
                type="bm25",
                k1=1.5,
                b=0.75,
                tokenizer="regex_word",
                lowercase=True,
                stopwords="none",
                stemming="none",
            ),
        ),
        data_dir=_DATA_DIR,
        output_path=Path("results/sweep.csv"),
    )

    retriever_factory = make_default_retriever_factory(_HF_CACHE_DIR)
    chunker_factory = make_default_chunker_factory(_HF_CACHE_DIR)

    rows, per_query_rows, all_succeeded = run_sweep(
        config, bundle, retriever_factory, chunker_factory
    )

    # 1 chunking strategy x 1 retriever x 4 cutoffs = 4 rows.
    assert len(rows) == 4
    expected_columns = {
        "run_id",
        "retriever",
        "chunking_strategy",
        "k",
        "recall_at_k",
        "ndcg_at_10",
        "mrr_at_10",
        "index_time",
        "query_latency",
        "num_queries_total",
        "num_queries_scored",
    }
    for row in rows:
        assert set(vars(row).keys()) == expected_columns
        assert row.run_id == "bm25__whole_document"
        assert row.retriever == "bm25"
        assert row.chunking_strategy == "whole_document"
        for value in (row.recall_at_k, row.ndcg_at_10, row.mrr_at_10):
            assert value == MISSING or (isinstance(value, float) and 0.0 <= value <= 1.0)

    # bm25 x whole_document is a cheap, network-free combination, so a
    # real run against the real corpus is expected to succeed cleanly.
    assert all_succeeded is True
    for row in rows:
        assert row.recall_at_k != MISSING
        assert row.ndcg_at_10 != MISSING
        assert row.mrr_at_10 != MISSING

    # Per-query rows are emitted for the standard 4-cutoff set.
    assert len(per_query_rows) == bundle_scored_query_count(bundle) 


def bundle_scored_query_count(bundle) -> int:
    """Local helper: the number of queries with at least one
    Qrels-judged relevant document, matching `run_sweep`'s own
    `num_queries_scored` derivation -- used only to size the expected
    `per_query_rows` count above without duplicating `run_sweep`'s
    internals or importing a private helper."""
    from src.metrics import judged_relevant_docs

    return sum(1 for judged in bundle.qrels.values() if judged_relevant_docs(judged))
