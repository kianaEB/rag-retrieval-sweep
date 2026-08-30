# Implementation Plan

## Overview

Ordering rule: this spec is a documentation deliverable with exactly
two pieces of new *code* — `src/token_length_analysis.py` and
`src/verify_writeup_numbers.py` — plus a narrow, one-function
extract-and-import refactor of `src/retrievers/dense_retriever.py` that
the first of those two depends on. Nothing in this spec regenerates
`results/sweep.csv`, `results/per_query.csv`, `results/significance.csv`,
or `results/run_config.json` — all four already exist and are read-only
inputs. The refactor (Task 1) comes first because
`token_length_analysis.py` imports `format_document_text` from it
(design.md's Property 3, "by construction" — there is exactly one
implementation, called from two places). `token_length_analysis.py`
(Task 2) and its pytest suite (Task 3) then follow, and are independent
of `verify_writeup_numbers.py` (Task 4) and its pytest suite (Task 5),
which have no dependency on the refactor or on token-length code at
all — the two pieces of new code can proceed in parallel once Task 1 is
done. Task 6 is the one-time, manual, real-artifact-producing run of
`python -m src.token_length_analysis` against the actual cached corpus
and tokenizer under `data/`, producing the real committed
`results/token_length_report.json` — mirroring session 1's and
significance-testing's own precedent of deferring real-corpus/real-model
runs to a manual step rather than an automated test. `docs/
numeric_traceability.csv` is scaffolded (Task 7) once
`verify_writeup_numbers.py`'s ledger schema is fixed, then grows one row
per Numeric_Claim as `README.md` (Task 8) and `SPEC.md` (Task 9) are
drafted — both documents depend on Task 6's real
`token_length_report.json`, since Requirement 11.6 obliges both the
Readme_Document's "Headline finding" section and the Spec_Document's
threats-to-validity section to state the truncation asymmetry when the
measured fraction exceeds 1%.
The final task (Task 10) is the real Verification_Pass run against the
fully populated ledger and the real committed artifacts, fixing any
`MISMATCH` before the feature is considered done. All done checks are
Git Bash / POSIX shell or Python one-liners, per shell-conventions.md;
none writes into or deletes from the real `results/` or `docs/` files
except the two tasks that are explicitly about producing/growing a real
committed artifact (Tasks 6, 7, 8, 9, 10).

## Tasks

- [x] 1. Extract `format_document_text` in `src/retrievers/dense_retriever.py`
  - Add a new module-level function
    `format_document_text(doc: Dict[str, str]) -> str` returning
    `f"{doc.get('title', '')} {doc.get('text', '')}"`, extracted
    verbatim from the f-string currently inline in
    `DenseRetriever.build_index`'s list comprehension.
  - Change `build_index`'s `texts = [...]` list comprehension to
    `texts = [format_document_text(corpus[doc_id]) for doc_id in self._doc_ids]`.
    No other line of `build_index`, `retrieve_all`, or `__init__`
    changes — same encoding, same normalization, same batching, same
    timing, same ranking.
  - Done check:
    `python -c "from src.retrievers.dense_retriever import format_document_text; assert format_document_text({'title': 'T', 'text': 'X'}) == 'T X'; assert format_document_text({'text': 'X'}) == ' X'; assert format_document_text({}) == ' '; print('ok')"`
    prints `ok` and exits 0. Additionally,
    `grep -c 'format_document_text' src/retrievers/dense_retriever.py`
    reports `2` (the definition plus the one call site inside
    `build_index`), confirming the inline f-string was replaced rather
    than left duplicated alongside the new function.
  - _Requirements: 11.1 (introduction's extract-and-import exception)_

- [x] 2. Write `src/token_length_analysis.py`
  - Implement `TokenLengthStats` (frozen dataclass:
    `num_documents_total`, `num_documents_exceeding`,
    `fraction_exceeding`) and `compute_exceedance_stats(token_counts,
    max_sequence_length)`: `num_documents_exceeding = sum(1 for c in
    token_counts if c > max_sequence_length)` (strictly greater than),
    `fraction_exceeding = num_documents_exceeding /
    num_documents_total`, defined as `0.0` when `num_documents_total ==
    0`.
  - Implement `TokenLengthReport` (frozen dataclass: `model_name`,
    `max_sequence_length`, `num_documents_total`,
    `num_documents_exceeding`, `fraction_exceeding`) and
    `write_token_length_report(report, output_path)`, reusing
    `src.report._atomic_write_text` (temp file + `os.replace`) so
    `output_path` is left absent or in its pre-run state on any
    failure; raise this module's own `TokenLengthReportError` on
    failure.
  - Implement `load_tokenizer_offline(model_name, cache_folder)`: set
    `os.environ["HF_HUB_OFFLINE"] = "1"` and
    `os.environ["TRANSFORMERS_OFFLINE"] = "1"` before the load call,
    then call `transformers.AutoTokenizer.from_pretrained(model_name,
    cache_dir=str(cache_folder), local_files_only=True)`; catch any
    exception from either layer and re-raise as this module's own
    `TokenizerLoadError`, never retrying without the offline flags.
  - Implement `count_tokens(tokenizer, text)`:
    `len(tokenizer(text, add_special_tokens=True,
    truncation=False)["input_ids"])`.
  - Implement `main(argv=None)`: parse `--config` (default
    `configs/sweep.yaml`) and `--output` (default
    `results/token_length_report.json`); load via
    `load_sweep_config` (on `ConfigError`, print and return non-zero,
    write nothing); extract the one `DenseRetrieverConfig`'s
    `model_name`; call `configure_caches(config.data_dir)` before any
    `huggingface_hub`/`transformers`/`beir`-importing call; call
    `load_scifact(config.data_dir)` (on `CorpusLoadError` /
    `CorpusValidationError`, print and return non-zero, write nothing);
    call `load_tokenizer_offline` (on `TokenizerLoadError`, print and
    return non-zero, write nothing, no network call attempted); for
    every document in `bundle.corpus`, append
    `count_tokens(tokenizer, format_document_text(doc))` (importing
    `format_document_text` from `src.retrievers.dense_retriever`,
    Task 1); compute `compute_exceedance_stats(token_counts,
    max_sequence_length=256)`; write the report; return 0.
  - Done check:
    `python -c "from src.token_length_analysis import compute_exceedance_stats, TokenLengthStats; s = compute_exceedance_stats([100, 256, 257, 300], 256); assert s == TokenLengthStats(4, 2, 0.5), s; s2 = compute_exceedance_stats([], 256); assert s2 == TokenLengthStats(0, 0, 0.0), s2; print('ok')"`
    prints `ok` and exits 0. No network call.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.7_

- [x] 3. Write `tests/test_token_length_analysis.py`
  - Cover `compute_exceedance_stats` (Property 1) on: an empty list;
    an all-under-threshold list; an all-over-threshold list; a mixed
    list with an independently hand-computed expected fraction; and
    the `256`-vs-`257` boundary pair (a count of exactly `256` is not
    exceeding, `257` is).
  - Cover the same mixed list, shuffled, asserting identical
    `num_documents_exceeding`/`num_documents_total`/`fraction_exceeding`
    (Property 2 — order independence).
  - Cover `load_tokenizer_offline` (Property 4) raising
    `TokenizerLoadError` when pointed at an empty temporary directory
    (via `tempfile.TemporaryDirectory()`) for a real model name,
    asserting no network call is attempted (safe under "no network
    access inside tests" because `local_files_only=True` makes
    `transformers` raise immediately on a cache miss).
  - Import only `src.token_length_analysis` (which transitively
    imports `src.retrievers.dense_retriever` for
    `format_document_text` — expected, per design.md's "Scope"
    section) — do not import `src.sweep_runner` or `src.significance`.
    Do not load the real cached model or tokenize the real corpus.
  - Done check: `pytest tests/test_token_length_analysis.py -v`
    reports all tests passed, and
    `! grep -Eq 'sentence_transformers|corpus_loader|sweep_runner' tests/test_token_length_analysis.py && echo ok`
    prints `ok`.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.7_

- [x] 4. Write `src/verify_writeup_numbers.py`
  - Implement `TraceabilityRow` (frozen dataclass: `claim_id`,
    `document`, `location`, `stated_value`, `stated_precision`,
    `source_artifact`, `source_fields`, `computation`) and
    `VerificationResult` (frozen dataclass: `claim_id`, `matched`,
    `failure_mode` — `None` | `"value_not_in_document"` |
    `"artifact_mismatch"` — `stated_rounded`, `computed_rounded`,
    `detail`).
  - Implement `stated_value_matches_precision(stated_value,
    stated_precision)`: fixed format checks for `integer` (no `.` in
    `stated_value`), `Ndp` (exactly `N` digits after a single `.`),
    `percentage:Ndp` (trailing `%` with exactly `N` decimal digits
    before it), and the ratio `Nx` suffix (same digit check as
    `integer`/`Ndp` on the digits preceding the `x`); any other
    `stated_precision` shape is treated as malformed.
  - Implement `load_ledger(path)`: parses `docs/numeric_traceability.csv`
    into `TraceabilityRow` instances via `pandas.read_csv`, and for each
    row, immediately calls `stated_value_matches_precision` before any
    artifact I/O; raise this module's own `TraceabilityFileError`
    naming the offending `claim_id` on a missing file, a parse failure,
    or any row failing that check — halting before any row is
    verified.
  - Implement `load_artifact_values(source_artifact, source_fields,
    artifacts_dir)`: for `sweep.csv`/`significance.csv`/`per_query.csv`,
    parse `row_selector` prefixes in `source_fields` (e.g.
    `metric=ndcg_at_10`) as an exact-match row filter, then read the
    named column from the one matching row; for `run_config.json`, walk
    the dotted path (e.g. `corpus_load_report.num_documents`) through
    the parsed JSON; for `token_length_report.json`, read the named
    top-level key directly. Raise this module's own
    `VerificationSourceError` if the file is absent, the row selector
    matches zero or more than one row, or the field/key is absent.
  - Implement `apply_computation(computation, values)` over the fixed
    enum `_ALLOWED_COMPUTATIONS = ("copy", "ratio", "delta", "mean",
    "percentage", "sum", "half_ci_width")`; raise
    `VerificationSourceError` for any other `computation` value.
  - Implement `round_half_up(value, precision_spec)` using
    `decimal.Decimal(str(value)).quantize(decimal.Decimal(target),
    rounding=decimal.ROUND_HALF_UP)` (converting through `str(value)`,
    never `Decimal(float)` directly).
  - Implement `verify_row(row, artifacts_dir, repo_root)`: (1)
    document-presence check — read `repo_root / row.document` and
    confirm `row.stated_value` occurs as a literal substring; on a
    miss, return immediately with `matched=False,
    failure_mode="value_not_in_document"`, without touching the
    artifact; (2) ledger-to-artifact comparison — only reached if (1)
    passes: resolve the value(s) via `load_artifact_values`, apply
    `apply_computation`, round both the stated and computed values with
    `round_half_up` at `row.stated_precision`, and compare the rounded
    strings for exact equality, returning `matched=True,
    failure_mode=None` on agreement or `matched=False,
    failure_mode="artifact_mismatch"` otherwise. A row whose
    `source_fields` is the literal sentinel `"n/a"` or `"NA"` is
    compared as a sentinel-to-sentinel string match, not rounded.
  - Implement `main(argv=None)`: parse `--repo-root` (default cwd) and
    the ledger/artifacts paths; call `load_ledger` (on
    `TraceabilityFileError`, print and return non-zero, no partial
    verification reported as a pass); for each row in file order, call
    `verify_row`; print one line per row (`MATCH` or `MISMATCH` with
    `failure_mode` and both rounded values) plus a summary count;
    return `0` only if every row matched, else `1`.
  - Done check:
    `python -c "from src.verify_writeup_numbers import round_half_up, apply_computation, stated_value_matches_precision; assert round_half_up(0.125, '2dp') == '0.13'; assert round(0.125, 2) == 0.12; assert apply_computation('ratio', [2.0, 1.0]) == 2.0; assert apply_computation('delta', [0.5, 0.3]) == 0.2 or abs(apply_computation('delta', [0.5, 0.3]) - 0.2) < 1e-12; assert stated_value_matches_precision('0.1234', '4dp'); assert not stated_value_matches_precision('0.123', '4dp'); print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 5. Write `tests/test_verify_writeup_numbers.py`
  - Cover `round_half_up`'s tie-breaking (Property 5):
    `round_half_up(0.125, "2dp") == "0.13"` alongside a direct
    comparison statement that Python's own `round(0.125, 2) == 0.12`.
  - Cover each member of `_ALLOWED_COMPUTATIONS` (`copy`, `ratio`,
    `delta`, `mean`, `percentage`, `sum`, `half_ci_width`) against a
    small literal input and an independently hand-computed expected
    output, plus an unrecognized `computation` value raising
    `VerificationSourceError`.
  - Cover `stated_value_matches_precision` against at least one
    matching case per recognized shape (`integer`, `Ndp`,
    `percentage:Ndp`, `Nx`) and at least one deliberately inconsistent
    case (e.g. `stated_value="-0.007"` against `stated_precision="4dp"`),
    plus `load_ledger` raising `TraceabilityFileError` naming the
    offending `claim_id` when a fixture ledger row fails this check
    (use a `tempfile`-written CSV, never the real
    `docs/numeric_traceability.csv`).
  - Cover `verify_row` (Property 6) on a matching pair, and on a pair
    that would falsely match under naive `abs(a - b) < epsilon`
    comparison but correctly mismatches once both sides are rounded to
    the stated (coarser) precision and compared as strings.
  - Cover `verify_row`'s document-presence check (Property 7): succeeds
    when `row.stated_value` is present in a fixture document string
    (written to a temp file standing in for `repo_root / row.document`);
    fails with `failure_mode="value_not_in_document"` when the fixture
    document has been doctored to no longer contain `row.stated_value`,
    even though the cited artifact fixture's value would otherwise
    match.
  - Import only `src.verify_writeup_numbers`, plus `pandas`/`json`/
    `tempfile`/`pytest` as needed for fixtures — do not import
    `src.token_length_analysis` or any retriever/corpus module. Read no
    file under the real `results/` or `docs/` directories.
  - Done check: `pytest tests/test_verify_writeup_numbers.py -v`
    reports all tests passed, and
    `! grep -Eq 'beir|sentence_transformers|corpus_loader|sweep_runner|token_length_analysis' tests/test_verify_writeup_numbers.py && echo ok`
    prints `ok`.
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 6. Run the Token_Length_Analysis for real and commit `results/token_length_report.json`
  - Run `python -m src.token_length_analysis` (using the already-cached
    corpus and `all-MiniLM-L6-v2` tokenizer under `data/` from the
    completed sweep — no download expected). Inspect the printed exit
    code and the resulting `results/token_length_report.json`.
  - Done check: after the real run,
    `python -c "import json; r = json.load(open('results/token_length_report.json')); assert r['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2'; assert r['max_sequence_length'] == 256; assert r['num_documents_total'] == 5183; assert 0 <= r['num_documents_exceeding'] <= r['num_documents_total']; assert abs(r['fraction_exceeding'] - r['num_documents_exceeding']/r['num_documents_total']) < 1e-9 or r['num_documents_total'] == 0; print('ok')"`
    prints `ok` and exits 0 (the `5183` check cross-references the
    already-committed `results/run_config.json`'s
    `corpus_load_report.num_documents`, confirming this run loaded the
    same corpus, not a different or partial one). No network call is
    expected on this run since the cache under `data/` is already
    populated.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 1.4_

- [x] 7. Scaffold `docs/numeric_traceability.csv`
  - Create the CSV with the header row matching
    `src/verify_writeup_numbers.py`'s `TraceabilityRow` schema exactly:
    `claim_id,document,location,stated_value,stated_precision,source_artifact,source_fields,computation`.
    No data rows yet — rows are added incrementally in Tasks 8 and 9,
    one row per Numeric_Claim, in the same edit that adds the claim's
    text to `README.md`/`SPEC.md`.
  - Done check:
    `python -c "import pandas as pd; df = pd.read_csv('docs/numeric_traceability.csv'); assert list(df.columns) == ['claim_id','document','location','stated_value','stated_precision','source_artifact','source_fields','computation']; print('ok')"`
    prints `ok` and exits 0.
  - _Requirements: 12.1, 12.4_

- [x] 8. Author `README.md` at the repository root
  - Write the first paragraph stating: the nDCG@10 mean difference
    (`all-MiniLM-L6-v2` minus BM25) from `results/significance.csv`'s
    nDCG@10 row `mean_diff`; the 95% CI (`ci_lower`, `ci_upper`) from
    the same row; the Holm-Bonferroni adjusted p-value
    (`p_value_adjusted`); and, since the committed `verdict` is
    `indistinguishable`, a statement that the comparison is
    indistinguishable from noise — not described as a win for either
    run (Requirement 2.4).
  - Immediately following that verdict, in the same "Headline finding"
    section, write one or two sentences stating the token-length
    confound (Requirement 11.6): the fraction of corpus documents
    exceeding `all-MiniLM-L6-v2`'s 256-token limit, read from
    `results/token_length_report.json`'s `fraction_exceeding` field;
    the asymmetry that creates under whole-document chunking (BM25
    scores each document's full text, while the dense run scores only
    the first 256 tokens of that same document); and a pointer to
    `SPEC.md`'s "Threats to validity" section for the full discussion.
    This statement SHALL NOT appear only in "What this does not
    claim", which sits below the results table — a reader of the
    Headline finding section alone must learn of the confound.
  - Write the engineering-cost paragraph: the Reference_Run's and
    `all-MiniLM-L6-v2` run's `index_time` from `results/sweep.csv`; the
    ratio of the two (dense over BM25), computed from those two values;
    and, in the same paragraph or the immediately following sentence,
    the nDCG@10 verdict from Requirement 2 restated.
  - Write the 6-row results table (recall@1, recall@5, recall@10,
    recall@20, nDCG@10, MRR@10): each row's BM25 absolute value from
    `results/sweep.csv`; each row's `all-MiniLM-L6-v2` delta read
    directly from `results/significance.csv`'s `mean_diff` column for
    that metric; the nDCG@10 row additionally carrying
    `p_value_adjusted` and `verdict`; every other row carrying the
    literal `"n/a"` in both those positions, matching the committed
    `significance.csv`'s own `n/a` sentinel for those rows.
  - Write the corpus-statistics sentence(s): `num_documents`,
    `num_queries`, `num_qrel_pairs`, each read from
    `results/run_config.json`'s `corpus_load_report` object.
  - Write the reproduction section: the exact sweep entry-point command
    (`python -m src.sweep_runner --config configs/sweep.yaml`); the
    exact significance entry-point command (`python -m src.significance
    --config configs/significance.yaml`); the combined
    indexing-and-retrieval time (sum of both runs' `index_time` +
    `query_latency` from `results/sweep.csv`), explicitly not labeled
    "total"/"wall-clock"/"runtime"; a statement that total wall-clock
    runtime is longer (corpus loading, model loading, metric
    computation, report writing, plus a first-run download); the
    one-time-download / cached-reuse / significance-makes-no-network-call
    statement; and every package version from
    `results/run_config.json`'s `installed_versions` object.
  - Write the "What this does not claim" section: scoped to BEIR
    SciFact only; the exact set of `chunking_strategy` values present
    in `results/sweep.csv`; the exact set of `retriever` values present
    in `results/sweep.csv`; and a statement that the finding is not a
    production RAG recommendation.
  - For every Numeric_Claim added above, add one corresponding row to
    `docs/numeric_traceability.csv` (Task 7) in the same edit, with
    `document=README.md`.
  - Done check:
    `python -c "text = open('README.md', encoding='utf-8').read(); assert 'What this does not claim' in text; assert 'indistinguishable' in text; assert 'python -m src.sweep_runner --config configs/sweep.yaml' in text; assert 'python -m src.significance --config configs/significance.yaml' in text; print('ok')"`
    prints `ok`;
    `python -c "import json, re; text = open('README.md', encoding='utf-8').read(); headline = text.split('## Headline finding', 1)[1].split('## ', 1)[0]; report = json.load(open('results/token_length_report.json')); pct = f\"{report['fraction_exceeding'] * 100:.2f}\"; assert pct in headline, (pct, headline); assert 'SPEC.md' in headline; print('ok')"`
    prints `ok`, confirming the token-length confound sentence(s) with
    the correct percentage and a pointer to `SPEC.md` appear within the
    "Headline finding" section itself, not only in "What this does not
    claim"; and
    `python -c "import pandas as pd; df = pd.read_csv('docs/numeric_traceability.csv'); assert (df['document'] == 'README.md').sum() > 0; assert (df['source_artifact'] == 'token_length_report.json').sum() > 0; print('ok')"`
    prints `ok`, confirming ledger rows were added alongside the
    document's prose, including at least one row sourced from
    `token_length_report.json`.
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5, 11.6_

- [x] 9. Author `SPEC.md` at the repository root
  - Write the design-summary section: the config-driven sweep grid
    (retrievers, cutoffs, chunking strategy) exactly as recorded in
    `results/run_config.json`'s `sweep_config` object; the
    "index once, retrieve once, slice four ways" property; the
    statement that BEIR SciFact qrels are the sole relevance ground
    truth; nDCG@10 as the pre-declared Primary_Metric with recall@k and
    MRR@10 as secondary; and the pre-declared statistical scheme
    (paired bootstrap, paired permutation, Holm-Bonferroni), naming
    `resample_count`, `permutation_count`, and `bootstrap_seed` from
    `results/run_config.json`'s `significance` object.
  - Write the nDCG@10 convention section verbatim per session-1
    Requirement 6.2 (DCG@10 / IDCG@10 / nDCG@10 formulas, mean over
    queries), and a statement that this convention was fixed before any
    sweep result existed.
  - Write the "Threats to validity" section naming: sparse qrels, with
    the average judged-relevant-per-query ratio
    (`num_qrel_pairs / num_queries` from `run_config.json`'s
    `corpus_load_report`); BM25's sensitivity to preprocessing, with the
    exact tokenizer/stopwords/stemming/case-handling/`k1`/`b` values
    from `run_config.json`'s `sweep_config` BM25 entry; single-corpus
    generalization; statistical power, with the nDCG@10 95% CI
    half-width (`(ci_upper - ci_lower) / 2` from
    `results/significance.csv`); the measured BM25 `query_latency` as
    an artifact of `rank_bm25`'s pure-Python implementation, not
    lexical retrieval generally; and the token-length truncation
    fraction from the real, committed `results/token_length_report.json`
    (Task 6), including the confound statement if that fraction is
    strictly greater than 1%.
  - Do not restate the nDCG@10 "indistinguishable" verdict anywhere in
    this document (or in `README.md`) as evidence that no difference
    exists between the compared runs.
  - For every Numeric_Claim added above, add one corresponding row to
    `docs/numeric_traceability.csv` (Task 7) in the same edit, with
    `document=SPEC.md`.
  - Done check:
    `python -c "text = open('SPEC.md', encoding='utf-8').read(); assert 'Threats to validity' in text; assert 'sparse qrels' in text.lower(); assert 'single-corpus' in text.lower() or 'single corpus' in text.lower(); print('ok')"`
    prints `ok`, and
    `python -c "import pandas as pd; df = pd.read_csv('docs/numeric_traceability.csv'); assert (df['document'] == 'SPEC.md').sum() > 0; print('ok')"`
    prints `ok`.
  - _Requirements: 1.1, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.5, 11.6_

- [x] 10. Run the Verification_Pass for real and fix any mismatch
  - Run `python -m src.verify_writeup_numbers --repo-root .` against
    the real, fully populated `docs/numeric_traceability.csv` (Tasks 7,
    8, 9) and the real committed `results/sweep.csv`,
    `results/per_query.csv`, `results/significance.csv`,
    `results/run_config.json`, and `results/token_length_report.json`
    (Task 6).
  - For any `MISMATCH`, correct the Numeric_Claim's text in
    `README.md`/`SPEC.md`, or correct the ledger row's `source_fields`/
    `computation`/`stated_precision` — never by editing any of the five
    cited artifacts, since regenerating or editing them is out of scope
    for this spec. Re-run until every row reports `MATCH`.
  - Perform the manual completeness check named in design.md: re-read
    both finished documents line by line and confirm every number
    appearing in their prose has a matching `docs/numeric_traceability.csv`
    row (Requirement 12.4) — this step is not automatable and is not
    covered by the exit-code check below.
  - Done check: `python -m src.verify_writeup_numbers --repo-root .`
    exits 0 (verify with `echo $?` immediately after, in Git Bash), and
    its printed summary reports zero `MISMATCH` lines across every
    ledgered row.
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

## Notes

- Tasks 1-5 are the entire new-code surface of this spec: the
  extract-and-import refactor (Task 1), `token_length_analysis.py`
  (Task 2) and its tests (Task 3), and `verify_writeup_numbers.py`
  (Task 4) and its tests (Task 5). No task adds test coverage for
  `format_document_text` itself, or for anything in
  `src/retrievers/dense_retriever.py` beyond the refactor — that gap is
  explicitly out of scope per design.md's "What is explicitly not
  tested in this spec."
- No task creates `ANALYSIS.md`, a GitHub Actions workflow, a third
  retriever, an additional chunking strategy, a failure-bucketing
  column, or a data-layer test file — all explicitly out of scope per
  `requirements.md`'s introduction and `scope-guard.md`.
- Tasks 6, 8, 9, and 10 are real-artifact-producing or hand-authoring
  steps, not pytest tasks, matching this repo's established pattern
  (session-1 and significance-testing both defer real-corpus/real-model
  runs and manual reconciliation to non-automated steps).
- `results/sweep.csv`, `results/per_query.csv`, `results/significance.csv`,
  and `results/run_config.json` are never regenerated, edited, or
  overwritten by any task in this spec — they are read-only inputs
  throughout.
- All done checks are Git Bash / POSIX shell (`grep`, `echo $?`) or
  Python one-liners — no PowerShell.
- Every task references specific acceptance criteria for traceability;
  each builds on the prior ones, and no task is a giant
  "implement everything" step.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "4", "7"] },
    { "id": 1, "tasks": ["2", "5"] },
    { "id": 2, "tasks": ["3", "6"] },
    { "id": 3, "tasks": ["8"] },
    { "id": 4, "tasks": ["9"] },
    { "id": 5, "tasks": ["10"] }
  ]
}
```
