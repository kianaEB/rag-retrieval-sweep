# Implementation Plan

Ordering rule: everything network-free (errors, config, seeding, metrics +
tests, the orchestration seam + its stub test, the retrievers, the report
writer) comes before anything that touches the network. The BEIR corpus
load and the full end-to-end run are last. `pytest` must be green
(Tasks 6 and 9) before any download — of model weights or of the corpus —
happens anywhere in the sequence.

- [x] 1. Resolve versions by installing unpinned, then pin `requirements.txt` from the frozen environment
  - In a clean virtual environment, install the 8 top-level packages with
    no version pins: `pip install beir rank_bm25 sentence-transformers
    numpy pandas PyYAML pytest pytrec-eval-terrier`. Do not write any
    version number by hand anywhere in this task — every version in the
    final file comes from what pip actually resolved.
  - Confirm the import check passes before freezing anything:
    `python -c "import beir, rank_bm25, sentence_transformers, numpy, pandas, yaml, pytest, pytrec_eval, torch; print('ok')"`
    must print `ok` and exit 0. If it fails, fix the install and re-check;
    do not proceed to freezing on a failing import.
  - Only after that import check passes, run `pip freeze` and write
    `requirements.txt` from its output, trimmed to the 8 top-level
    packages plus whatever transitive packages they actually pulled in
    (e.g. the CPU-only `torch` wheel `sentence-transformers` installs).
    Every line is copied verbatim (`package==version`) from `pip freeze`
    output — no duplicate package names, no range operators, no
    hand-chosen version.
  - Done check: in a second, separate clean virtual environment,
    `pip install -r requirements.txt` exits 0, then
    `python -c "import beir, rank_bm25, sentence_transformers, numpy, pandas, yaml, pytest, pytrec_eval, torch; print('ok')"`
    prints `ok`. Every non-blank line in `requirements.txt` matches
    `package==version` (spot-check with
    `Select-String -Path requirements.txt -Pattern '==' -NotMatch` returning
    no lines other than blanks/comments), confirming Requirement 9's
    exact-pin rule is satisfied by the frozen output rather than by
    versions chosen in advance.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 11.5_

- [ ] 2. Write `src/errors.py`
  - Define `ConfigError`, `UnsupportedPreprocessingError(ConfigError)`,
    `CorpusLoadError`, `CorpusValidationError`, `SeedApplicationError`,
    `ModelLoadError`, `RetrievalError`, `MetricComputationError`,
    `ZeroQualifyingQueriesError`, `ReportWriteError`.
  - Done check:
    `python -c "from src.errors import *; assert issubclass(UnsupportedPreprocessingError, ConfigError); [cls('x') for cls in (ConfigError, CorpusLoadError, CorpusValidationError, SeedApplicationError, ModelLoadError, RetrievalError, MetricComputationError, ZeroQualifyingQueriesError, ReportWriteError)]; print('ok')"`
    prints `ok` with exit code 0.
  - _Requirements: 1.4, 1.5, 1.6, 1.8, 2.6, 3.7, 4.7, 6.1, 6.2, 6.3, 8.5, 10.4_

- [ ] 3. Write `configs/sweep.yaml` and `src/config.py`
  - Implement `BM25RetrieverConfig`, `DenseRetrieverConfig`, `SweepConfig`,
    and `load_sweep_config(path)` with the validation described in the
    design (exactly 2 retrievers, cutoffs == {1,5,10,20}, chunking ==
    `whole_document`, BM25 preprocessing fields restricted to the
    supported set).
  - Author `configs/sweep.yaml` matching the schema in `design.md`.
  - Done check: a script that (a) calls
    `load_sweep_config(Path("configs/sweep.yaml"))` and asserts
    `config.seed == 42`, `len(config.retrievers) == 2`,
    `config.cutoffs == (1, 5, 10, 20)`; (b) asserts a copy of the YAML
    with `tokenizer: whitespace` raises `UnsupportedPreprocessingError`;
    (c) asserts a copy with only 1 retriever raises `ConfigError`. Script
    prints `ok` and exits 0.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.7, 8.1_

- [ ] 4. Write `src/seeding.py`
  - Implement `apply_seed(seed: int) -> None`, seeding `random`, `numpy`,
    and `torch` in that order, wrapping any failure in
    `SeedApplicationError`.
  - Done check: a script that calls `apply_seed(42)`, records
    `random.random()` and `numpy.random.rand()`, calls `apply_seed(42)`
    again, records the same two calls again, and asserts both pairs are
    exactly equal. Prints `ok` and exits 0.
  - _Requirements: 8.2, 8.5_

- [ ] 5. Write `src/metrics.py`
  - Implement `judged_relevant_docs`, `recall_at_k`, `ndcg_at_10`,
    `mrr_at_10`, `mean_over_qualifying_queries`, `scored_query_count`
    exactly per the formulas in `design.md`.
  - Done check: a script with one hand-built "perfect ranking" fixture
    (all relevant docs at the top) asserting `recall_at_k(...) == 1.0`,
    `ndcg_at_10(...) == 1.0`, `mrr_at_10(...) == 1.0`, and one "no
    relevant doc" fixture asserting all three return `0.0` with no
    exception. Prints `ok` and exits 0. No network call.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ] 6. Write `tests/test_metrics.py`
  - Cover, for each of `recall_at_k`, `ndcg_at_10`, `mrr_at_10`: no
    judged-relevant document, a relevant doc outside the top-k cutoff,
    perfect ranking, and empty ranked list — fixtures ≤10 docs, floats
    compared with `pytest.approx(..., abs=1e-6)`. Add the `pytrec_eval`
    differential cross-check for all three metrics (MRR@10 truncates the
    ranked list to 10 before comparison; recall@k and nDCG@10 do not).
  - Done check: `pytest tests/test_metrics.py -v` reports all tests
    passed (e.g. `12 passed`), with zero network calls made during the
    run (no `requests`/`urllib`/`huggingface_hub` import anywhere in the
    test module — verify with
    `Select-String -Path tests/test_metrics.py -Pattern "beir|sentence_transformers|huggingface"`
    returning no matches).
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 7. Write `src/retrievers/base.py`
  - Define the `Retriever` `Protocol` (`name`, `build_index`,
    `retrieve_all`), the `RetrievalRun` dataclass, and the shared
    `doc_id_sort_key(doc_id: str)` helper used by both retrievers for the
    ascending-numeric-ID tie-break.
  - Done check:
    `python -c "from src.retrievers.base import doc_id_sort_key; assert sorted(['10','9','2','abc','1'], key=doc_id_sort_key) == ['1','2','9','10','abc']; print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 3.4, 4.8, 5.1_

- [ ] 8. Write the `run_sweep` orchestration seam in `src/sweep_runner.py`
  - Implement `run_sweep(config, bundle, retriever_factory)`: exactly one
    `build_index` + one `retrieve_all` per declared retriever at
    `deepest_cutoff = max(config.cutoffs)`, `ndcg_at_10`/`mrr_at_10`
    computed once per `run_id`, `recall_at_k` recomputed per cutoff by
    slicing the single returned ranked list, `run_id` assigned per
    retriever, `num_queries_total`/`num_queries_scored` computed once
    before the retriever loop. Do not wire `main()` or a real retriever
    yet — `retriever_factory` is the only construction path.
  - Done check: a script that builds a 3-document in-memory
    `CorpusBundle`, a 2-cutoff `SweepConfig` (e.g. cutoffs `(1, 2)`), and
    a trivial hand-written factory returning an object whose
    `build_index`/`retrieve_all` return fixed values; asserts
    `run_sweep(...)` returns exactly `len(retrievers) * len(cutoffs)`
    rows and `all_succeeded is True`. Prints `ok` and exits 0. No
    network call, no real retriever imported.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.6, 6.7, 7.3, 7.4, 7.7_

- [ ] 9. Write `tests/test_orchestration.py`
  - Implement `StubRetriever` (records every `build_index`/`retrieve_all`
    call and its arguments; returns a hand-specified ranked list per
    query; no real computation) and a ≤5-document `In_Memory_Test_Corpus`,
    both as literals in the test module. Drive `run_sweep` with two
    `StubRetriever` instances and cutoffs `(1, 2, 3)`.
  - Done check: `pytest tests/test_orchestration.py -v` passes, with
    assertions inside the test verifying: each stub's
    `build_index_calls` and `retrieve_all_calls` both have length exactly
    1; the recorded `retrieve_all` call used `top_k == max(cutoffs)`; and
    for every declared cutoff `k` and every query, the ranked list scored
    for that row equals `full_ranked_list[qid][:k]`. Reports e.g.
    `1 passed`.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9_

- [ ] 10. Write `src/retrievers/bm25_retriever.py`
  - Implement `BM25Retriever`: shared `_tokenize` (regex word split +
    optional lowercase, applied identically to docs and queries), one
    `BM25Okapi` build in `build_index` (timed), `retrieve_all` ranking by
    `(-score, doc_id_sort_key(doc_id))` (timed).
  - Done check: a script that builds a `BM25Retriever` from
    `configs/sweep.yaml`'s BM25 settings over a 4-document in-memory
    corpus including two documents with identical text (to force a score
    tie), calls `build_index` once and `retrieve_all` once with
    `top_k=3`; asserts `index_time`/`query_latency` are non-negative
    floats, every ranked list has length ≤ 3, and the two tied documents
    appear in ascending numeric-ID order. Prints `ok` and exits 0. No
    network call.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 11. Write `src/retrievers/dense_retriever.py`
  - Implement `DenseRetriever`: loads `all-MiniLM-L6-v2` on
    `device="cpu"` with `cache_folder` under `data/`, asserts
    `huggingface_hub`'s resolved cache matches that folder (raising
    `ModelLoadError` otherwise), encodes the corpus once in `build_index`
    (timed, normalized embeddings), and ranks by cosine similarity via a
    brute-force dot product in `retrieve_all` (timed), tie-broken by
    `doc_id_sort_key`.
  - Done check: a script that builds a `DenseRetriever` with
    `cache_folder=Path("data/hf_cache")` over a 3-document in-memory
    corpus, calls `build_index` once and `retrieve_all` once with
    `top_k=2`; asserts `index_time`/`query_latency` are non-negative
    floats and every ranked list has length ≤ 2. Prints `ok` and exits 0.
    `Get-ChildItem data/hf_cache -Recurse` shows downloaded model files
    afterward. **This is the first task in the sequence that touches the
    network** (one-time `all-MiniLM-L6-v2` weight download to `data/`).
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 10.5, 10.6, 10.7_

- [ ] 12. Write `src/report.py`
  - Implement `SweepReportRow`, the `MISSING = "NA"` sentinel,
    `write_sweep_report` (atomic temp-file-then-`os.replace` write), and
    `write_run_config_record` (seed, resolved `SweepConfig`, and
    `importlib.metadata.version(...)` for `beir`, `rank_bm25`,
    `sentence-transformers`, `torch`, `numpy`).
  - Done check: a script that writes 4 `SweepReportRow`s (one with
    `recall_at_k = "NA"`, the rest with real floats including one
    legitimate `0.0`) to a temp CSV, reads it back with `pandas`, and
    asserts: exactly 4 rows; the `"NA"` cell reads back as the string
    `"NA"`; the `0.0` cell reads back as the float `0.0` (not `"NA"`
    and not `NaN`). A second assertion calls `write_run_config_record`
    and confirms the written JSON has `seed`, `sweep_config`, and
    `installed_versions` keys with non-empty version strings. Prints
    `ok` and exits 0.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 8.3_

- [ ] 13. Write `src/corpus_loader.py`
  - Implement `configure_caches(data_dir)` (sets `HF_HOME`/`HF_HUB_CACHE`
    before any `huggingface_hub`-importing module loads) and
    `load_scifact(data_dir)`: downloads/loads BEIR SciFact, raises on
    empty or unresolved qrels references, derives all three counts from
    the loaded structures, and returns `(CorpusBundle, CorpusLoadReport)`.
  - Done check: a script that calls `configure_caches(Path("data"))`
    then `load_scifact(Path("data"))`, prints `report.as_log_line()`,
    and asserts `num_documents > 0`, `num_queries > 0`,
    `num_qrel_pairs > 0`. Output includes a line matching
    `CORPUS_LOAD_REPORT documents=\d+ queries=\d+ qrel_pairs=\d+`. **This
    is the second network-touching task** (one-time BEIR SciFact
    download to `data/`).
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 10.5_

- [ ] 14. Wire `main()` in `src/sweep_runner.py` and run it end to end
  - Implement the CLI entry point: parse `--config` (default
    `configs/sweep.yaml`) → `configure_caches` → `apply_seed` → write
    `results/run_config.json` → `load_scifact` → `run_sweep` with
    `make_default_retriever_factory` (real `BM25Retriever`/
    `DenseRetriever`) → `write_sweep_report` → exit 0 only if no row
    carries a `"NA"` marker.
  - Done check: `python -m src.sweep_runner --config configs/sweep.yaml`
    completes; `$LASTEXITCODE` is `0`; `results/sweep.csv` and
    `results/run_config.json` both exist on disk afterward.
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 8.3, 8.5_

- [ ] 15. Verify `results/sweep.csv` content against the schema
  - Done check: a script that loads `results/sweep.csv` with `pandas`
    and asserts: exactly 8 rows; columns are exactly `run_id, retriever,
    chunking_strategy, k, recall_at_k, ndcg_at_10, mrr_at_10, index_time,
    query_latency, num_queries_total, num_queries_scored`; `k` values
    per retriever are exactly `{1, 5, 10, 20}`; `chunking_strategy` is
    `whole_document` in every row; within each `run_id`, `ndcg_at_10`,
    `mrr_at_10`, `index_time`, and `query_latency` are identical across
    its 4 rows; no cell equals the string `"NA"`. Prints `ok` and exits 0.
  - _Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 16. Verify rerun-identity manually (Requirement 8.4)
  - Done check: run
    `python -m src.sweep_runner --config configs/sweep.yaml` a second
    time to a second output path (e.g. `results/sweep_rerun.csv`, via a
    copy of the config with a different `output_path`), then run a
    script that loads both CSVs and asserts every column except
    `index_time` and `query_latency` is exactly equal cell-for-cell
    between the two runs. Also diff the two `run_config.json` files'
    `sweep_config` and `installed_versions` fields for exact equality.
    Prints `ok` and exits 0. Delete the rerun artifacts afterward so they
    don't linger as untracked files.
  - _Requirements: 8.4_
