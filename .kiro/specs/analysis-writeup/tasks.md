# Implementation Plan

## Overview

Ordering rule: every task below carries an explicit wave number, and the
`## Task Dependency Graph` section at the end is the machine-readable
form of the same assignment. Waves are zero-indexed, matching every
sibling spec's graph (`full-grid-chunking-sweep`,
`significance-testing`, `repo-writeup`, `groundedness-gate`). Tasks
inside one wave are genuinely independent — they write disjoint files and
neither consumes an artifact the other produces — so they can run in
parallel; a task sits in wave N+1 only because it consumes something a
wave-N task produces, or because it writes a file a wave-N task also
writes.

Wave 0 holds the two edits to already-committed modules: the five
appended `src/errors.py` exception classes (Task 1) and the three-part
`src/verify_writeup_numbers.py` edit (Task 2, `_CSV_ARTIFACTS` += 2
strings, module docstring, argparse `description`). Those two touch
different files and neither depends on the other — Task 2 depends on
nothing this spec produces at all, exactly as the repo-writeup spec's own
Task 4 and the groundedness-gate spec's Task 12 were scheduled as early
independent siblings.

**`src/failure_buckets.py` is built in two tasks, one per stage, in two
consecutive waves — and the reason is worth stating, because it is the
one place this plan departs from its own "one file, one task" habit.**
The module now has two separately specified stages (design.md, Overview):
a Bucket_Assignment_Stage that reads `results/per_query.csv` and nothing
else, and a Covariate_Enrichment_Stage that loads the already-cached BEIR
SciFact corpus, its Qrels, and both Dense_Model tokenizers from `data/`
and computes six per-query token-length covariate columns. Together they
are far more than one reviewable unit: seventeen ordered steps in `main`,
five exception tiers, twelve committed columns, two rendering
conventions, and a pre-flight cache check whose *position* relative to
`load_scifact` is itself a requirement (16.12). Splitting them gives the
covariate work its own done checks — the deferred-import surface, the
`data/`-read boundary, the sentinel-not-zero rendering — instead of
burying them in a single task's tail.

They are **not** independent and are **not** in the same wave. They write
the same file, which alone forces sequencing under this plan's wave rule.
They also share state by design: Task 4's `attach_covariates` reindexes to
Task 3's `FAILURE_BUCKET_COLUMNS`, and Task 4's `main` calls every builder
and every assertion Task 3 wrote. So Task 3 (wave 1) declares the
twelve-column schema, the naming rules, the taxonomy, the loader, the
builders, the assertions, and the two writers, and deliberately declares
**no `main`** — an entry point that could only write a covariate-less
twelve-column file would be orphaned code the next task rewrites. Task 4
(wave 2) adds the covariate stage and the one `main` that calls both
stages in the design's order. Every function Task 3 writes has a caller by
the end of Task 4, and no line Task 3 writes is rewritten by it.

Wave 3 is `tests/test_failure_buckets.py` (Task 5), which depends on both
stages existing and must be green before the real run. Wave 4 is the
one-time real invocation of `python -m src.failure_buckets` against the
committed `results/per_query.csv` **and a populated `data/` cache**,
producing and committing `results/failure_buckets.csv` and
`results/failure_bucket_counts.csv` (Task 6) — a manual real-artifact
step, matching how session 1, significance-testing, repo-writeup, and
groundedness-gate each defer their real-artifact runs rather than
automating them in `pytest`. This is the first task in this repository
whose real run needs the gitignored cache: the covariate stage reads
`data/scifact` and both tokenizer snapshot directories under
`data/hf_cache`, and **fails rather than downloading them**, so the cache
is a precondition Task 6 checks before it runs, not a side effect it
produces. Wave 5 is the hand-authored `ANALYSIS.md` (Task 7), which can
only be written after both artifacts exist, because every failure-bucket
and every covariate figure in it is read out of those files rather than
typed from a draft. Wave 6 holds the two independent follow-ups:
appending the `ANALYSIS.md` rows to `docs/numeric_traceability.csv` and
running the Verification_Pass to exit 0 (Task 8), and the one-line
`README.md` filename reference to `ANALYSIS.md` (Task 9) — different
files, no shared artifact. Wave 7 is the final read-only audit (Task 10),
which is its own wave because its entire content is a statement about the
finished working tree, so it cannot run alongside a task that is still
editing it.

Nothing in this plan regenerates `results/sweep.csv`,
`results/per_query.csv`, `results/significance.csv`,
`results/run_config.json`, `results/token_length_report.json`, or any
`results/groundedness*` / `results/generated_answers.csv` /
`results/hand_checked*` artifact; nothing edits `configs/*.yaml` or
`.github/workflows/ci.yml`. `configs/sweep.yaml` is **read** — by
`load_sweep_config`, for its `data_dir` field only (Requirement 16.10) —
and never written. Nothing under `data/` is written, added to, or
downloaded. All done checks are Git Bash / POSIX shell (`grep`, `ls`,
`test -d`, `find`, `cmp`, `diff`, `mktemp -d`, `rm -rf`, `awk`,
`echo $?`) or Python one-liners, per `shell-conventions.md` — no
PowerShell cmdlet appears anywhere in this file.

## Tasks

- [x] 1. Extend `src/errors.py` with the analysis-writeup exception types
  - Append, under a `# --- analysis-writeup spec: extends the hierarchy
    above ---` banner matching the four earlier specs' banners, exactly
    **five** new classes with the design's docstrings:
    `FailureBucketInputError(Exception)` (absent/unparseable
    `results/per_query.csv`, or any of `run_id`, `retriever`,
    `chunking_strategy`, `query_id`, `recall_at_1`, `recall_at_20`,
    `ndcg_at_10`, `num_judged_relevant` missing — a **committed**
    `results/` artifact is broken);
    `CovariateInputError(Exception)` (a Covariate_Enrichment_Stage input
    is unavailable or unresolvable: `data/scifact` absent, Qrels absent
    or empty, a Dense_Model tokenizer snapshot absent, or a `query_id`
    present in `results/per_query.csv` absent from the loaded query set —
    the **gitignored** `data/` cache is unpopulated);
    `ContrastQuerySetError(Exception)` (a Pair_Contrast whose two Run_Ids
    do not cover the same `query_id` set, or whose Run_A/Run_B is absent
    from the input); `FailureBucketAssertionError(Exception)` (every
    pre-write invariant: a failed Totality_Assertion, a duplicated
    `(run_id, query_id)` pair, a `run_id` whose unrounded or rendered
    fractions do not sum to 1 within tolerance, a covariate that is not
    run-independent, a sentinel-required cell holding a numeric `0`, and
    **a Run_Id containing the `|vs|` separator**); and
    `FailureBucketWriteError(Exception)` (either output path could not be
    written — the only tier reached after every assertion passed).
  - Requirement 16.15 is why `CovariateInputError` is one type rather
    than four: its four conditions are one distinguishable failure with
    one remedy (populate `data/`, then re-run), and the *message* names
    which of the four it was.
  - **`CompositeRunIdCollisionError` does not exist.** It was folded into
    `FailureBucketAssertionError`: its only raise site is
    `assert_no_separator_collision`, one of six `assert_*` helpers that
    already share that type, and no caller can act differently on it —
    `main` prints and returns 1 for all of them identically. If an
    earlier draft of `src/errors.py` in this working tree already carries
    that class, remove it; do not leave both.
  - `CovariateInputError` is deliberately **not** folded into
    `FailureBucketInputError`: they are raised from different places and
    carry different remedies (restore a committed artifact versus
    populate the cache), and collapsing them would hand a clean-checkout
    user the wrong instruction. `ContrastQuerySetError` is deliberately
    not folded into `FailureBucketAssertionError` either: it reports a
    data-coverage fault in the input, not a violated internal invariant.
  - Do not touch, rename, or re-base any existing class; do not add a
    config-error subclass for the bucket taxonomy — Requirement 3.4
    forbids reading a predicate, threshold, or bucket name from a
    config file, so no `configs/failure_buckets.yaml` and no
    corresponding error type exists anywhere in this spec.
  - Wave 0. Independent of Task 2 (different file) and a prerequisite
    for Tasks 3 and 4, which import all five names.
  - Done check (the five current names import and instantiate):
    `python -c "from src.errors import FailureBucketInputError, CovariateInputError, ContrastQuerySetError, FailureBucketAssertionError, FailureBucketWriteError; [cls('x') for cls in (FailureBucketInputError, CovariateInputError, ContrastQuerySetError, FailureBucketAssertionError, FailureBucketWriteError)]; print('ok')"`
    prints `ok` and exits 0.
  - Done check (**the fold is verified, not assumed**):
    `python -c "import src.errors as e; assert not hasattr(e, 'CompositeRunIdCollisionError'), 'CompositeRunIdCollisionError must not exist -- it was folded into FailureBucketAssertionError'; import inspect; names = [n for n, o in vars(e).items() if inspect.isclass(o) and issubclass(o, Exception) and n.startswith(('FailureBucket', 'Covariate', 'ContrastQuerySet'))]; assert sorted(names) == ['ContrastQuerySetError', 'CovariateInputError', 'FailureBucketAssertionError', 'FailureBucketInputError', 'FailureBucketWriteError'], sorted(names); print('ok')"`
    prints `ok` and exits 0 — the first assertion fails loudly if the
    folded type survived anywhere, and the second fails if a sixth
    spec-scoped type was added.
  - Done check (the append removed nothing, and no taxonomy config error
    was introduced):
    `python -c "import src.errors as e; assert all(hasattr(e, n) for n in ('ConfigError', 'ChunkingError', 'ChunkingConfigError', 'TraceabilityFileError', 'VerificationSourceError', 'RunConfigMergeError', 'CorpusLoadError', 'CorpusValidationError', 'TokenizerLoadError')); print('ok')"`
    prints `ok`; and
    `! grep -Eq 'FailureBucketConfigError|failure_buckets\.yaml' src/errors.py && echo ok`
    prints `ok`.
  - _Requirements: 2.5, 4.6, 5.5, 7.6, 16.13, 16.15_

- [x] 2. Extend `src/verify_writeup_numbers.py` — the whole three-part edit
  - Append the two strings `"failure_buckets.csv"` and
    `"failure_bucket_counts.csv"` to the existing `_CSV_ARTIFACTS`
    tuple, after `"generated_answers.csv"`. `load_artifact_values`'s
    first branch is a membership test (`if source_artifact in
    _CSV_ARTIFACTS:`), so both files then resolve through the existing
    `_resolve_column_equality_reference` / `_resolve_csv_reference`
    pair — including the `all` row selector and the `__count__` field
    sentinel the groundedness-gate spec already added — with no further
    change.
  - Reword the module docstring's first paragraph so it names
    `ANALYSIS.md` alongside `README.md`/`SPEC.md`, and change the
    document-presence bullet's parenthetical `(README.md or SPEC.md)` to
    `(README.md, SPEC.md, or ANALYSIS.md)`. No other docstring sentence
    changes — the two-check description, the sentinel special case, and
    the "invoked manually, never from CI" note are already correct for a
    third document.
  - Change the argparse `description` text from `its cited document
    (README.md/SPEC.md)` to `its cited document
    (README.md/SPEC.md/ANALYSIS.md)`.
  - Change nothing else: no edit to `_resolve_csv_reference`,
    `_resolve_column_equality_reference`, `_resolve_json_path`,
    `_resolve_top_level_key`, `load_artifact_values`, `load_ledger`,
    `verify_row`, `apply_computation`, `round_half_up`, or
    `stated_value_matches_precision`; no new `_ALLOWED_COMPUTATIONS`
    member; no document allowlist, no per-document branch, and no
    document-name enumeration (`verify_row` already resolves
    `repo_root / row.document`). In particular do **not** teach
    `_read_csv_artifact` a `dtype=str` argument on the covariate
    columns' account — Requirement 8.2 forbids it, and it is
    unnecessary regardless: `exceeds`/`within` are not coercible to a
    boolean or numeric dtype, so `_read_csv_artifact`'s existing
    dtype-less `pandas.read_csv` call already reads
    `any_relevant_doc_exceeds_limit__*` back as text with no further
    change, as Task 8 documents.
  - Wave 0. Independent of Task 1 (different file) and of every other
    task in this plan — it touches only the already-committed
    repo-writeup module.
  - Done check:
    `python -c "import src.verify_writeup_numbers as v; assert v._CSV_ARTIFACTS == ('sweep.csv', 'significance.csv', 'per_query.csv', 'groundedness.csv', 'hand_checked_joined.csv', 'generated_answers.csv', 'failure_buckets.csv', 'failure_bucket_counts.csv'), v._CSV_ARTIFACTS; assert tuple(v._ALLOWED_COMPUTATIONS) == ('copy', 'ratio', 'delta', 'mean', 'percentage', 'sum', 'half_ci_width', 'complement_percentage', 'wilson_ci_lower', 'wilson_ci_upper'), v._ALLOWED_COMPUTATIONS; assert 'ANALYSIS.md' in (v.__doc__ or ''); print('ok')"`
    prints `ok` and exits 0;
    `python -m src.verify_writeup_numbers --help | grep -q 'ANALYSIS.md' && echo ok`
    prints `ok`;
    `python -m pytest tests/test_verify_writeup_numbers.py -q` exits 0
    (the unmodified suite passing is the evidence that Requirement 8.2's
    ten functions are unchanged in behavior); and
    `python -m src.verify_writeup_numbers --repo-root . > /dev/null; echo $?`
    prints `0`, confirming the two-string extension broke none of the
    167 pre-existing ledger rows.
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 3. Write `src/failure_buckets.py` — the Bucket_Assignment_Stage, the twelve-column schema, and the two writers
  - Module docstring per the design, naming **both** stages and what each
    one may touch: the Bucket_Assignment_Stage reads
    `results/per_query.csv` and loads no corpus, no Qrels, no tokenizer,
    no embedding model and no generative model, so no bucket label can
    depend on a token count (Requirement 2.1); the
    Covariate_Enrichment_Stage (Task 4) is the only part that reads
    `data/`. State that both stages complete and both CSV texts are
    fully serialized before the first byte is written, and that no seed
    is required (Requirements 2.7, 16.16).
  - Module-level imports in this task: `dataclasses`, `pathlib`,
    `typing`, `pandas`, the `src.errors` classes this stage raises, and
    `src.report._atomic_write_text`. No `beir`, `sentence_transformers`,
    `transformers`, `torch`, `huggingface_hub`, `requests`/`urllib`,
    `src.token_length_analysis`, or `src.retrievers.*` at module top —
    the latter two are deferred in Task 4. No `random`, `numpy.random`,
    `time`, `datetime`, `uuid`, and no `os.environ` read.
  - Declare the fixed taxonomy constants: `FAILURE_BUCKET_ORDER =
    ("total_miss", "mis_ranked", "partial_recall", "full_success")`,
    `CONTRAST_BUCKET_ORDER = ("a_only", "b_only", "both_miss",
    "both_answer")` (disjoint from the first by construction),
    `COMPOSITE_SEPARATOR = "|vs|"`, `REFERENCE_RUN_ID =
    "bm25__whole_document"`, `CROSS_STRATEGY_BASE = "whole_document"`,
    `CROSS_STRATEGY_VARIANTS = ("fixed_window", "sentence_window")`,
    `DENSE_RETRIEVERS = ("all-MiniLM-L6-v2", "bge-small-en-v1.5")`,
    `REQUIRED_COLUMNS`, `TEXT_COLUMNS = ("run_id", "retriever",
    "chunking_strategy", "query_id")`, `FRACTION_DECIMALS = 6`,
    `FRACTION_SUM_TOLERANCE = 1e-9`, `RENDERED_FRACTION_TOLERANCE =
    2e-6`, and the four default paths (`configs/sweep.yaml`,
    `results/per_query.csv`, and the two outputs). No Run_Id count,
    query count, or row count appears as a literal anywhere in the
    module (Requirement 2.6), and no predicate, threshold, or bucket
    name is read from a config file, a CLI argument, or an environment
    variable (Requirement 3.4).
  - Declare the covariate **naming** constants and helpers here, because
    the twelve-column schema is one declaration and must not be split
    across two tasks: `DENSE_MODEL_NAMES` (retriever name → Hugging Face
    repo id, a literal model *identity*, never a literal *limit*),
    `COVARIATE_NAMES = ("query_token_len", "max_relevant_doc_token_len",
    "any_relevant_doc_exceeds_limit")`, `model_tag(retriever_name)`
    returning `retriever_name.replace(".", "_")` (so `bge-small-en-v1.5`
    yields `bge-small-en-v1_5`, Requirement 6.6), and
    `covariate_column(covariate, retriever_name)` returning
    `f"{covariate}__{model_tag(retriever_name)}"`. **No column name may
    contain a `.`** (Requirement 6.7): the Verifier splits a ledger
    reference's field name from its row selector at the *last* `.`, so a
    dot inside a field name makes the selector resolve the wrong field
    and match zero rows. A dot inside a column *value* is unaffected, so
    the `run_id` and `retriever` columns keep their unsubstituted
    `bge-small-en-v1.5` text.
  - Declare the two frozen dataclasses and derive both writers' column
    lists from them: `CountRow(run_id, bucket, count, fraction)` with
    its column list from `[f.name for f in
    dataclasses.fields(CountRow)]`; and `FailureBucketRow` with twelve
    fields in the committed order (`run_id`, `retriever`,
    `chunking_strategy`, `query_id`, `bucket`, `num_judged_relevant`,
    then the three covariates for `all-MiniLM-L6-v2` and the three for
    `bge-small-en-v1.5` — model-major, covariate-minor), the six
    covariate fields typed `Union[int, str]` / `Union[bool, str]`
    because either may hold the `"NA"` sentinel. Build
    `FAILURE_BUCKET_COLUMNS` once, from `covariate_column(...)` over
    `COVARIATE_NAMES` × `DENSE_MODEL_NAMES`, and add a **module-level
    assertion** pinning it against `dataclasses.fields(FailureBucketRow)`
    and against Requirement 6.1's six literal covariate names, so a
    future edit to the tag rule or the dataclass that desynchronizes
    them fails at import rather than producing a mislabelled artifact.
  - Implement the pure predicates: `assign_failure_bucket(recall_at_1,
    recall_at_20, num_judged_relevant)` as three guarded returns plus an
    unconditional `return "full_success"` — `recall_at_20 == 0` →
    `total_miss`; `recall_at_1 == 0` → `mis_ranked`;
    `num_judged_relevant > 1 and recall_at_20 < 1` → `partial_recall`;
    fallthrough → `full_success`, with exact comparisons against 0 and 1
    and no tolerance; `is_answered(ndcg_at_10)` returning
    `ndcg_at_10 > 0`; and `assign_contrast_bucket(ndcg_a, ndcg_b)`
    exhausting the 2x2 truth table into `a_only`/`b_only`/`both_miss`/
    `both_answer`.
  - Implement `make_composite_run_id(run_a, run_b)` and
    `build_declared_contrast_set(run_ids)`: group (a) pairs
    `REFERENCE_RUN_ID` with every other observed Run_Id in ascending
    lexicographic order (one pair per Pre_Declared_Family row, which
    also supplies the two BM25 cross-strategy contrasts, so group (b)
    must not re-emit them); group (b) pairs each `DENSE_RETRIEVERS`
    entry's `whole_document` Run_Id with its `fixed_window` and
    `sentence_window` Run_Ids; returns (a) then (b), raises
    `ContrastQuerySetError` naming any Run_Id the rule needs that the
    input lacks, and ends with a `len(set(pairs)) == len(pairs)` guard
    raising `FailureBucketAssertionError`.
  - Implement `load_per_query(path)`: `pandas.read_csv` with
    `TEXT_COLUMNS` forced to `str` (the `query_id` dtype is load-bearing
    — default inference makes it `int64`, which would break both
    "copied unchanged" and "lexicographic order of the column's text")
    and `keep_default_na=False, na_values=[]` exactly as
    `_read_csv_artifact` already does; raises `FailureBucketInputError`
    naming the path on an absent or unparseable file, and naming
    *every* missing column at once.
  - Implement the frame builders: `build_failure_buckets` (one row per
    input row, per-row `assign_failure_bucket` call, passthrough of
    `retriever`/`chunking_strategy`/`query_id`/`num_judged_relevant`,
    sorted by `run_id` text then `query_id` text, index reset — the six
    covariate columns are added later by Task 4's `attach_covariates`);
    `build_run_counts` (all four declared buckets per Run_Id including
    zero-count rows, `fraction = count / that Run_Id's distinct
    query_id count`); `build_contrast_counts` (one Contrast_Bucket per
    (Pair_Contrast, `query_id`), `run_id` set to the Composite_Run_Id,
    denominator the shared `query_id` count, raising
    `ContrastQuerySetError` naming the Pair_Contrast and the
    lexicographically smallest offending `query_id` on an asymmetric
    query set); and `build_failure_bucket_counts` (concatenate, then
    sort on the dropped key triple `group` = 1 if
    `COMPOSITE_SEPARATOR in run_id` else 0, `run_id` text ascending,
    declared bucket rank ascending).
  - Implement the assertions, every one raising
    `FailureBucketAssertionError` naming the affected Run_Id,
    Pair_Contrast, `query_id` or `run_id`, the observed value, and the
    expected value: `assert_no_separator_collision` (called before any
    bucket is assigned; **raises `FailureBucketAssertionError`, not a
    dedicated collision type** — see Task 1's fold),
    `assert_unique_pairs`, `assert_partition_total(partition_label,
    bucket_counts, expected_total, declared_buckets)` — one helper
    called once per Run_Id and once per Pair_Contrast, so the two
    partitions cannot drift into two notions of "total" — and
    `assert_fraction_sums`, which runs **both** fraction checks with no
    short-circuit and no early return between them: the four *unrounded*
    float fractions against `FRACTION_SUM_TOLERANCE` (1e-9, Requirement
    5.4) and, in addition, the same four rendered to
    `FRACTION_DECIMALS` places and re-parsed against
    `RENDERED_FRACTION_TOLERANCE` (2e-6, Requirement 5.7 — four values
    each moved by at most 5e-7 move their sum by at most 2e-6). Neither
    check may be skipped, and neither replaces the other.
  - Implement the two writers on the `write_per_query_report` /
    `write_groundedness_report` pattern: fix the column list
    (`FAILURE_BUCKET_COLUMNS` for the per-query report,
    `dataclasses.fields(CountRow)` for the counts report), pre-format
    `fraction` via
    `frame["fraction"].map(lambda v: f"{v:.{FRACTION_DECIMALS}f}")`
    before `to_csv` (so the text asserted and the text written are the
    same), keep `count` a Python `int`, call `to_csv(index=False)`, then
    `_atomic_write_text(path, csv_text, failure_context=...,
    newline="")` — `newline=""` is mandatory or Windows gets `\r\r\n` —
    raising `FailureBucketWriteError` naming the failing path and
    whether the other report had already landed. Every covariate cell
    reaching the writer is already rendered text (Task 4 renders at the
    join), so neither writer formats a covariate.
  - Declare no `main` in this task. An entry point here could only write
    a covariate-less file against a twelve-column schema; Task 4 adds
    the one `main` that calls both stages in the design's order.
  - Wave 1. Depends on Task 1's five exception classes. First of the two
    writers of `src/failure_buckets.py`; Task 4 is the second, which is
    why they are in consecutive waves rather than the same one.
  - Done check (predicates, taxonomy, separator):
    `python -c "from src.failure_buckets import assign_failure_bucket as f, assign_contrast_bucket as g, is_answered, FAILURE_BUCKET_ORDER, CONTRAST_BUCKET_ORDER, COMPOSITE_SEPARATOR, make_composite_run_id; assert f(0.0, 0.0, 1) == 'total_miss'; assert f(0.0, 0.5, 1) == 'mis_ranked'; assert f(1.0, 0.5, 2) == 'partial_recall'; assert f(1.0, 0.5, 1) == 'full_success'; assert f(1.0, 1.0, 3) == 'full_success'; assert g(0.5, 0.0) == 'a_only'; assert g(0.0, 0.5) == 'b_only'; assert g(0.0, 0.0) == 'both_miss'; assert g(0.5, 0.5) == 'both_answer'; assert is_answered(0.0) is False; assert set(FAILURE_BUCKET_ORDER) & set(CONTRAST_BUCKET_ORDER) == set(); assert COMPOSITE_SEPARATOR == '|vs|'; assert make_composite_run_id('a', 'b') == 'a|vs|b'; print('ok')"`
    prints `ok` and exits 0.
  - Done check (the twelve-column schema, the tag rule, and the no-dot
    constraint):
    `python -c "import dataclasses, src.failure_buckets as m; want = ('run_id', 'retriever', 'chunking_strategy', 'query_id', 'bucket', 'num_judged_relevant', 'query_token_len__all-MiniLM-L6-v2', 'max_relevant_doc_token_len__all-MiniLM-L6-v2', 'any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2', 'query_token_len__bge-small-en-v1_5', 'max_relevant_doc_token_len__bge-small-en-v1_5', 'any_relevant_doc_exceeds_limit__bge-small-en-v1_5'); assert m.FAILURE_BUCKET_COLUMNS == want, m.FAILURE_BUCKET_COLUMNS; assert len(dataclasses.fields(m.FailureBucketRow)) == 12; assert [f.name for f in dataclasses.fields(m.CountRow)] == ['run_id', 'bucket', 'count', 'fraction']; assert not [c for c in m.FAILURE_BUCKET_COLUMNS if '.' in c], 'no column name may contain a dot'; assert m.model_tag('bge-small-en-v1.5') == 'bge-small-en-v1_5'; assert m.model_tag('all-MiniLM-L6-v2') == 'all-MiniLM-L6-v2'; print('ok')"`
    prints `ok` and exits 0 — this is the check that pins Requirement
    6.1's exact column order and Requirement 6.7's no-dot rule.
  - Done check (both fraction tolerances exist, with Requirement 5.4's
    and 5.7's values):
    `python -c "import src.failure_buckets as m; assert m.FRACTION_SUM_TOLERANCE == 1e-9, m.FRACTION_SUM_TOLERANCE; assert m.RENDERED_FRACTION_TOLERANCE == 2e-6, m.RENDERED_FRACTION_TOLERANCE; assert m.FRACTION_DECIMALS == 6; print('ok')"`
    prints `ok`; and
    `grep -c 'FRACTION_SUM_TOLERANCE\|RENDERED_FRACTION_TOLERANCE' src/failure_buckets.py`
    prints at least `4` (two declarations plus at least one use each,
    confirming both are actually applied rather than only declared).
  - Done check (import surface, no config read yet, no taxonomy knob):
    `! grep -Eq '^(import|from) +(beir|sentence_transformers|transformers|torch|huggingface_hub|requests|urllib|random|numpy|time|datetime|uuid|scipy|statistics|yaml|src\.token_length_analysis|src\.retrievers)' src/failure_buckets.py && echo ok`
    prints `ok` (the `!` asserts no forbidden top-level import matched);
    and `! grep -Eq 'os\.environ|run_config|MAX_SEQUENCE_LENGTH' src/failure_buckets.py && echo ok`
    prints `ok`, confirming no environment read, no `run_config.json`
    merge path, and no import of `token_length_analysis`' own
    all-MiniLM-L6-v2 limit literal.
    (Task 5's pytest suite is the authoritative verification; these are
    quick structural checks.)
  - _Requirements: 1.2, 1.7, 2.1, 2.5, 2.6, 2.7, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.7, 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_

- [x] 4. Add the Covariate_Enrichment_Stage and `main` to `src/failure_buckets.py`
  - Add the remaining module-top imports — `argparse`, `sys`,
    `load_sweep_config` from `src.config`, `configure_caches` and
    `load_scifact` from `src.corpus_loader`, `judged_relevant_docs` from
    `src.metrics`, `MISSING` from `src.report`, and `ConfigError`,
    `CorpusLoadError`, `CorpusValidationError`, `CovariateInputError`,
    `TokenizerLoadError` from `src.errors`. Importing all of these plus
    `pandas` pulls **none** of `transformers`, `torch`,
    `sentence_transformers`, `beir` or `huggingface_hub` into
    `sys.modules`: `src.corpus_loader` defers `beir` inside
    `load_scifact` by its own design, and `configure_caches` is pure
    `os.environ` assignment.
  - Implement `_import_tokenizer_helpers()`, the deferred import of the
    four reused primitives, returning `(format_document_text,
    count_tokens, load_tokenizer_offline,
    resolve_effective_max_sequence_length)`. **`format_document_text`
    lives in `src/retrievers/dense_retriever.py`, not in
    `src/token_length_analysis.py`** — its docstring records that it was
    extracted there so both the dense retriever and the token-length
    analysis tokenize the same text from one implementation, and this
    module becomes the third caller. Both source modules import heavy
    stacks at their own top (`transformers`; `sentence_transformers` +
    `numpy`), which is why the import is deferred rather than top-level,
    and deferring also gets the ordering right for free: the function is
    only ever called after `configure_caches` has set
    `HF_HOME`/`HF_HUB_CACHE`, which is the ordering
    `src/corpus_loader.py`'s docstring requires.
  - Add the covariate-stage constants: `EXCEEDS_TEXT = "exceeds"`,
    `WITHIN_TEXT = "within"` (Requirement 6.8's literal text encoding —
    the point is not casing, it is that neither string is coercible to
    a boolean or numeric dtype, so the file text and the ledger filter
    literal are always the same string regardless of whether `NA` also
    appears elsewhere in that column), `SCIFACT_CACHE_SUBDIR =
    "scifact"`, and `HF_CACHE_SUBDIR = "hf_cache"`. **No sequence-length
    limit appears as a literal, a config field, a CLI argument, or an
    environment variable** (Requirement 16.6), and
    `src.token_length_analysis.MAX_SEQUENCE_LENGTH` is not imported —
    it is that module's own all-MiniLM-L6-v2 number and would be wrong
    for `bge-small-en-v1.5`, whose limit differs.
  - Implement `CovariateInputs` (frozen dataclass: `corpus`, `queries`,
    `qrels`, `tokenizers`, `limits`), keyed by **retriever name** — the
    Per_Query_Report's `retriever` value — so the covariate column names
    follow from the keys without a second lookup, and so a test can
    construct the whole thing from Python literals.
  - Implement `assert_local_cache_present(data_dir)`: confirms
    `data_dir / SCIFACT_CACHE_SUBDIR` is a directory and that, for every
    model in `DENSE_MODEL_NAMES`, `data_dir / HF_CACHE_SUBDIR /
    f"models--{org}--{name}"` is a directory — the same paths
    `tests/test_data_layer.py`'s own cache-availability gate checks, and
    the same `models--{org}--{name}` convention `huggingface_hub`
    writes. Raises `CovariateInputError` naming **every** absent path.
    A directory-existence check, not a load, so it stays cheap and
    imports neither `transformers` nor `beir`.
  - Implement `load_covariate_inputs(data_dir, retriever_names)` — the
    only impure function in this stage — in exactly this order:
    (1) `assert_local_cache_present(data_dir)`; (2)
    `configure_caches(data_dir)`, **before** the deferred imports,
    because `huggingface_hub` resolves `HF_HOME`/`HF_HUB_CACHE` once at
    its own import time; (3) `_import_tokenizer_helpers()`; (4)
    `bundle, load_report = load_scifact(data_dir)`, wrapping
    `CorpusLoadError`/`CorpusValidationError` as `CovariateInputError`
    and printing `load_report.as_log_line()` so the corpus counts this
    stage actually loaded appear in the run's own output; (5)
    `load_tokenizer_offline(DENSE_MODEL_NAMES[name], data_dir /
    HF_CACHE_SUBDIR)` per retriever, wrapping `TokenizerLoadError` as
    `CovariateInputError`; (6) `resolve_model_limits(...)`. **Step 1
    must precede step 4**: `load_scifact`'s first action is
    `beir_util.download_and_unzip(url, str(data_dir))`, so calling it
    against an empty cache would reach the network and populate `data/`
    — exactly what Requirements 16.12 and 16.14 forbid, and a presence
    check afterwards would be a check after the damage.
  - Implement `resolve_model_limits(tokenizers, data_dir)`: returns each
    retriever name's Effective_Max_Sequence_Length via the committed
    `resolve_effective_max_sequence_length(model_name, tokenizer,
    data_dir / HF_CACHE_SUBDIR)`, from the model's own cached
    configuration and nothing else. Kept as its own function precisely
    so a test can supply `limits` as a plain `{retriever_name: int}`
    dict and never reach the `hf_hub_download` inside it.
  - Implement `max_relevant_doc_token_len(doc_token_lens,
    relevant_doc_ids)` — fully pure, takes **already-tokenized**
    lengths, returns the maximum or `None` (never `0`) for an empty
    relevant set. A `KeyError` for an id absent from `doc_token_lens` is
    left to propagate: `load_scifact` already validates referential
    integrity, and silently skipping an id would understate the maximum.
  - Implement `compute_token_length_covariates(query_ids, inputs)`,
    returning **one row per `query_id`** with columns `["query_id"]` plus
    the six `covariate_column(...)` names. Per (`query_id`,
    retriever_name): `relevant = judged_relevant_docs(
    inputs.qrels.get(query_id, {}))` — `src/metrics.py`'s own
    strictly-greater-than-0 condition, the only source of relevance, with
    no retrieval result, model score or heuristic consulted;
    `query_token_len = count_tokens(tokenizer, inputs.queries[query_id])`
    — untruncated, special tokens included; each relevant document's
    length as `count_tokens(tokenizer,
    format_document_text(inputs.corpus[doc_id]))` — `title` + `" "` +
    `text` over the **source** document, never a Chunk; `max_len`
    from `max_relevant_doc_token_len`; and `exceeds = None if max_len is
    None else max_len > inputs.limits[retriever_name]` — **strictly**
    greater than. Memoize each document's count per (retriever, doc_id)
    across queries; memoization cannot affect determinism because
    `count_tokens` is a pure function of (tokenizer, text). Raises
    `CovariateInputError` naming the `query_id` if it is absent from
    `inputs.queries`.
  - Implement `assert_covariates_run_independent(covariates, query_ids)`:
    raises `FailureBucketAssertionError` if the covariate frame holds
    more than one row for any `query_id`, if any `query_id` is absent
    from it, or if a `max_relevant_doc_token_len__*` /
    `any_relevant_doc_exceeds_limit__*` cell holds a numeric `0` or
    `False` where the sentinel was required.
  - Implement `attach_covariates(failure_buckets, covariates)`:
    **left-joins on `query_id` ALONE, never on `(run_id, query_id)`**.
    That join key is what makes Requirement 6.9 true by construction —
    one covariate row per `query_id`, nine bucket rows per `query_id`,
    all nine fed from the same source row — and a join on the pair would
    make a per-run covariate value *representable*, which Requirement
    16.9 forbids. Render every covariate cell here, at the boundary
    (Requirement 6.8): an int as a base-ten integer with no decimal
    point, a bool as `EXCEEDS_TEXT`/`WITHIN_TEXT`, and a `None` as
    `MISSING`. Rendering at the join rather than in the writer is what
    keeps a mixed int/`"NA"` column out of `float64`, where `to_csv`
    would emit `301.0` and an empty cell — the "no covariate column as
    empty text" failure Requirement 16.14 names. Reindex to exactly
    `FAILURE_BUCKET_COLUMNS`, then call `assert_unique_pairs` again: a
    left join whose right side had a duplicated `query_id` would
    silently fan the frame out, and a row count that grew during a join
    is exactly what a totality assertion exists to catch.
  - Implement `main(argv=None)` with `--per-query`, `--config` (read for
    `data_dir` only), `--buckets-out`, and `--counts-out` and **no other
    argument** — in particular no threshold, no bucket name, no taxonomy
    switch and no sequence-length limit, so Requirements 3.4 and 16.6
    are enforced by the parser's shape — in the design's seventeen-step
    order:
    (1) parse args; *Bucket_Assignment_Stage:* (2) `load_per_query`;
    (3) `assert_no_separator_collision`; (4) `build_failure_buckets` +
    `assert_unique_pairs`; (5) `build_run_counts` (per-Run_Id totality
    inside); (6) `build_declared_contrast_set`; (7)
    `build_contrast_counts` (per-Pair_Contrast totality inside);
    (8) `build_failure_bucket_counts` + `assert_fraction_sums` — **the
    counts frame is final here and the covariate stage never touches it**
    (Requirement 16.17); *Covariate_Enrichment_Stage:* (9)
    `load_sweep_config(args.config)`, using `config.data_dir` and
    nothing else; (10) `load_covariate_inputs`; (11)
    `compute_token_length_covariates(sorted(per_query["query_id"].unique()),
    inputs)`; (12) `assert_covariates_run_independent`; (13)
    `attach_covariates` + `assert_unique_pairs`; *serialize, report,
    write:* (14) build **both** CSV texts in memory; (15) print the
    Requirement 5.6 summary **and** the Requirement 16.18 summary;
    (16) write buckets, then write counts; (17) return 0.
  - The covariate stage sits after step 8 and before step 14 for a
    reason that does not survive moving it: every failure tier —
    `FailureBucketInputError`, `ContrastQuerySetError`,
    `FailureBucketAssertionError`, `ConfigError`, `CovariateInputError` —
    is then raised strictly before the first `_atomic_write_text` call,
    so Requirements 2.5, 4.6, 5.5, 7.6 and 16.13 all keep their shared
    "SHALL write neither report" clause by construction. Putting it after
    the writes breaks Requirement 16.13 outright; putting it between the
    two writes breaks it for one file. And by step 9 every bucket is
    already assigned, so nothing the covariate stage loads can reach a
    predicate — Requirement 2.1 as an ordering fact on top of Requirement
    2.1 as a call-graph fact.
  - Print, at step 15, the Requirement 5.6 figures (row count, Run_Id
    count, per-Run_Id query count, Pair_Contrast count with its
    8-family-aligned / 4-cross-strategy split, and both artifacts' row
    counts with the 36 / 48 split) **and** the Requirement 16.18 figures
    (`CorpusLoadReport.as_log_line()` verbatim, each Dense_Model's
    resolved Effective_Max_Sequence_Length, the number of `query_id`s
    whose covariates were computed, and the number recorded with the
    sentinel). Every figure is a `len(...)`/`nunique()` of something just
    loaded or just built, and both limits come from
    `resolve_effective_max_sequence_length` — no literal count and no
    literal limit (Requirements 2.6, 16.6).
  - Wave 2. Depends on Task 3 (same file, and every builder, assertion
    and writer `main` calls) and on Task 1's `CovariateInputError`.
    Second and final writer of `src/failure_buckets.py`.
  - Done check (the entry point exists and exposes no limit or taxonomy
    knob):
    `python -m src.failure_buckets --help > /dev/null && echo ok`
    prints `ok`;
    `python -m src.failure_buckets --help | grep -Eiq 'threshold|taxonomy|bucket-name|seed|max.*len|limit|seq' && echo BAD || echo ok`
    prints `ok`; and
    `python -m src.failure_buckets --help | grep -c -E '\-\-(per-query|config|buckets-out|counts-out)'`
    prints `4`.
  - Done check (the deferred imports actually stayed deferred):
    `python -c "import sys, src.failure_buckets; heavy = {'beir', 'sentence_transformers', 'transformers', 'torch', 'huggingface_hub'}; leaked = sorted(heavy & set(sys.modules)); assert not leaked, leaked; print('ok')"`
    prints `ok` and exits 0; and
    `! grep -Eq '^(import|from) +(beir|sentence_transformers|transformers|torch|huggingface_hub|requests|urllib|random|numpy|time|datetime|uuid|scipy|statistics|yaml|src\.token_length_analysis|src\.retrievers)' src/failure_buckets.py && echo ok`
    prints `ok`, confirming the four reused primitives are imported
    inside `_import_tokenizer_helpers` rather than at module top.
  - Done check (**no limit is typed in**):
    `python -c "import src.failure_buckets as m; bad = [n for n, v in vars(m).items() if isinstance(v, int) and not isinstance(v, bool) and v in (256, 512)]; assert not bad, bad; assert not hasattr(m, 'MAX_SEQUENCE_LENGTH'); print('ok')"`
    prints `ok`; and
    `! grep -Eq 'MAX_SEQUENCE_LENGTH|\b(256|512)\b' src/failure_buckets.py && echo ok`
    prints `ok` — the negative-space check that the tempting shortcut
    was not taken.
  - Done check (**the pre-flight ordering inside the loader**):
    `python -c "import re; s = open('src/failure_buckets.py', encoding='utf-8').read(); f = [b for b in re.split(r'\ndef ', s) if b.startswith('load_covariate_inputs')]; assert len(f) == 1, len(f); b = f[0]; assert b.index('assert_local_cache_present') < b.index('load_scifact'), 'the pre-flight cache check must run BEFORE load_scifact, which downloads'; assert b.index('configure_caches') < b.index('_import_tokenizer_helpers'), 'configure_caches must run before the deferred tokenizer import'; print('ok')"`
    prints `ok` and exits 0.
  - Done check (the covariate arithmetic, and the sentinel that is not a
    zero):
    `python -c "import src.failure_buckets as m; assert m.max_relevant_doc_token_len({}, []) is None; assert m.max_relevant_doc_token_len({'d': 0}, ['d']) == 0; assert m.max_relevant_doc_token_len({'a': 7, 'b': 3}, ['a', 'b']) == 7; assert m.MISSING == 'NA'; assert (m.EXCEEDS_TEXT, m.WITHIN_TEXT) == ('exceeds', 'within'); print('ok')"`
    prints `ok` — a zero-length document yields `0` and an empty
    relevant set yields `None`, which is the whole distinction
    Requirement 16.8 protects.
  - Done check (the join key is `query_id` alone):
    `! grep -Eq 'on=\[[^]]*run_id[^]]*query_id[^]]*\]' src/failure_buckets.py && echo ok`
    prints `ok`, confirming no merge joins on the `(run_id, query_id)`
    pair; and
    `grep -Eq 'on="query_id"|on=.query_id.' src/failure_buckets.py && echo ok`
    prints `ok`.
  - _Requirements: 1.8, 1.9, 2.2, 2.3, 2.4, 2.8, 5.6, 6.5, 6.8, 6.9, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 16.10, 16.11, 16.12, 16.13, 16.14, 16.15, 16.16, 16.17, 16.18_

- [x] 5. Write `tests/test_failure_buckets.py`
  - Implement the shared `_frame(rows)` helper over the 11
    Per_Query_Report columns (with `run_id`/`retriever`/
    `chunking_strategy`/`query_id` as `str`, matching what
    `load_per_query`'s dtype mapping produces) and a `tmp_path` helper
    returning the paths `main` needs. Every fixture is a Python literal
    or a small CSV under `tmp_path`, and **no fixture exceeds 40 rows** —
    the largest is the 3-Run_Id x 4-`query_id` frame.
  - Implement the covariate fixtures as **hand-written stubs**, following
    the `_ZeroChunkStubChunker` pattern `tests/test_chunking.py`
    established — a plain local class with exactly the duck-typed
    surface the code under test calls, no `unittest.mock`, no
    `monkeypatch`, no skip gate: `_StubTokenizer`, implementing only
    `__call__(text, add_special_tokens=..., truncation=...)` returning a
    mapping with an `"input_ids"` key, asserting
    `add_special_tokens is True` and `truncation is False` so the real
    committed `count_tokens` runs against it unmodified, and tokenizing
    on whitespace plus two special tokens so a fixture's expected count
    is `len(text.split()) + 2`, readable by eye; `_STUB_CORPUS` of **no
    more than 5 documents**, including a long one, a short one, one with
    an empty `title` and one with an empty `text`, so
    `format_document_text`'s `title + " " + text` composition is
    exercised on the shapes it actually meets; `_STUB_QRELS` in which one
    query has two judged-relevant documents, one has exactly one, **one
    has an entry whose only relevance score is `0` — which
    `judged_relevant_docs`' strictly-greater-than-0 condition must treat
    as NO judged-relevant document** — and one query is absent from qrels
    entirely; `_STUB_QUERIES`; and `_STUB_LIMITS` as plain ints,
    **deliberately different per model**, so a test fails if one model's
    limit were applied to the other model's column. The relevance-0
    entry is doing real work: a stub whose only empty case were
    "absent from qrels" would not catch a covariate stage that filtered
    on key presence rather than on score.
  - Implement the **ten** tests Requirement 15 names, with these exact
    function names: `test_failure_bucket_predicates_cover_all_four_buckets`
    (15.1, one query per Failure_Bucket, asserted both through
    `build_failure_buckets` and directly against
    `assign_failure_bucket`);
    `test_contrast_bucket_rules_cover_all_four_buckets` (15.2, two
    Run_Ids x four shared `query_id`s, one query per Contrast_Bucket);
    `test_totality_assertion_failure_writes_neither_report` (15.3,
    pre-creates both output paths with `b"SENTINEL\n"`, triggers a
    duplicated `(run_id, query_id)` pair, asserts
    `FailureBucketAssertionError` names it and that **both files still
    hold exactly those sentinel bytes**);
    `test_two_invocations_produce_byte_identical_reports` (15.4, two
    `main` runs into separate subdirectories compared with
    `read_bytes()`, plus a `test_shuffled_input_produces_identical_bytes`
    companion);
    `test_counts_run_id_and_bucket_combinations_are_unique` (15.5,
    `duplicated(subset=["run_id", "bucket"]).sum() == 0` plus the
    disjoint-name-sets and right-names-on-the-right-`run_id`-kind
    companions);
    `test_bucket_assigner_uses_no_network_no_model_no_data_dir` (15.6,
    15.10 — asserts `sys.modules` is free of `beir`,
    `sentence_transformers`, `transformers`, `torch` and
    `huggingface_hub` after importing the module under test, that its
    module-level import set contains no `requests`/`urllib.request`,
    that no fixture path resolves under `data/`, **and that this test
    module defines no module-scope `pytestmark` and no `skipif`**, so
    Requirement 15.10's "no new skip-gated real-corpus or
    real-tokenizer test" is checked rather than merely intended);
    `test_covariate_computation_over_stub_corpus_and_stub_tokenizer`
    (15.8 — calls `compute_token_length_covariates` against a
    `CovariateInputs` built from the stubs and asserts, for a query with
    two judged-relevant documents, that `query_token_len__*` is the
    query's own count, that `max_relevant_doc_token_len__*` is the
    **maximum** rather than the first or last, and the expected
    exceedance under each model's own limit, with a companion case in
    which the two models' exceedance columns **disagree** — the
    assertion that catches one limit being applied to both; for a query
    with exactly one judged-relevant document, that "max over one
    element" is not special-cased; for the relevance-0 query, that both
    sentinel-bearing covariates are the sentinel while
    `query_token_len__*` is still a real count; plus a
    `pytest.raises(CovariateInputError)` case for a `query_id` absent
    from `_STUB_QUERIES`, and direct unit tests of the pure
    `max_relevant_doc_token_len` including the empty-relevant-set case
    (`None`) and a genuine zero-length document (`0`, not `None`));
    `test_missing_judgment_records_sentinel_not_numeric_zero` (15.9 —
    reads the **raw written text** of `failure_buckets.csv` and asserts
    the relevance-0 query's and the absent-from-qrels query's
    `max_relevant_doc_token_len__*` and
    `any_relevant_doc_exceeds_limit__*` cells are exactly `NA`, never
    `0`, `0.0`, `within`, `true`, `false`, `True` or `False` or empty —
    a numeric zero is the failure this test names in its own title,
    `within` is what a bug that defaulted a missing exceedance to
    `False` instead of `None` would render under the current encoding,
    and the four boolean-literal strings guard against a
    not-yet-migrated implementation still emitting the old encoding;
    asserted against raw text rather than the re-parsed frame on
    purpose, because a `float64` column with `NaN` — the failure this
    test exists to catch — re-parses into something a frame-level
    equality check could be written to tolerate, and the text cannot
    be);
    `test_covariates_are_identical_across_a_query_id_rows` (16.9, 6.9 —
    3 Run_Ids x 4 `query_id`s, each of the six covariate columns having
    exactly one distinct value per `query_id` group, plus a companion
    feeding `attach_covariates` a covariate frame with a duplicated
    `query_id` and asserting `FailureBucketAssertionError` — the
    join-fan-out guard); and
    `test_bucket_level_covariate_count_selector_resolves` (7.4, 8.1,
    12.7, and the covariate half of 15.11 — writes a fixture
    `failure_buckets.csv` in the committed twelve-column schema and
    calls the **real** `_resolve_csv_reference` from
    `src.verify_writeup_numbers` with the three-filter reference
    `run_id=...,bucket=total_miss,any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2=exceeds.__count__`,
    asserting a **non-zero** count; this is a regression guard on two
    things, neither of which is a dtype-inference gap — that gap is
    structurally gone now that `exceeds`/`within` are not coercible to
    a boolean or numeric dtype, so `frame[key].astype(str)` stays the
    identity on the column regardless of what else it holds. What this
    test still guards: (a) that non-coercibility itself, so a future
    change to the rendering constants cannot reintroduce a
    dtype-inferred column silently; and (b) that shipping every
    covariate count alongside its `=within` complement — the mitigation
    Task 8 documents for the one remaining silent-failure mode, where
    `__count__` returns `0.0` for a filter that matched nothing rather
    than raising — still catches a `__count__`-returns-0 failure caused
    by something unrelated, such as a mistyped column name; also
    asserts a two-filter single-query covariate read resolves to
    exactly one value and that no header cell of the written file
    contains a `.`).
  - Implement the property tests, each with `@settings(max_examples=100)`
    or greater and a docstring naming its design property in the form
    **Feature: analysis-writeup, Property N: ...**:
    `test_property_1_failure_bucket_is_total_and_first_match` (Property
    1, `@given` recall floats in `[0.0, 1.0]` and small positive
    `num_judged_relevant`, `@example`s at the exact `0.0`/`1.0`
    boundaries, compared against an independently written
    full-three-clause ladder);
    `test_property_2_contrast_bucket_is_total_and_matches_truth_table`
    (Property 2, `@example` at `(0.0, 0.0)`);
    `test_property_4_partitions_are_total_over_generated_frames`
    (Property 4, 2-3 Run_Ids x 3-8 `query_id`s, sums asserted against
    the fixture's own `nunique()`, never a constant, and each `run_id`'s
    fractions summing to 1 within 1e-9); and
    `test_property_14_max_relevant_doc_token_len_is_a_maximum_or_none`
    (Property 14, `@given` over generated `{doc_id: int}` mappings and
    subsets: the result is a member of the selected lengths and `>=`
    every one of them, and is `None` exactly when the subset is empty).
  - Implement the remaining fixture tests from the design's Testing
    Strategy: `test_declared_contrast_set_is_duplicate_free_and_correctly_ordered`;
    `test_failure_buckets_columns_and_passthrough_values` (header equals
    the declared **twelve**-column list in order, a join back to the
    input frame shows the four passthrough columns equal everywhere, and
    no header cell contains a `.`);
    `test_covariate_column_names_match_the_tag_rule` (the twelve literal
    names written out in the test rather than derived, so the test and
    the module cannot drift the same way);
    `test_exceedance_boundary_is_strictly_greater_than` (a stub document
    at exactly the limit yields `within`, at limit + 1 yields
    `exceeds`);
    `test_covariate_cells_are_rendered_as_declared` (regexes over the
    **raw** written text: `^\d+$` for every `query_token_len__*` cell,
    `^\d+$` or exactly `NA` for every `max_relevant_doc_token_len__*`
    cell, and exactly `exceeds`, `within` or `NA` for every
    `any_relevant_doc_exceeds_limit__*` cell — never `true`, `false`,
    `True`, `False`, `1`, `0`, `0.0` or empty);
    `test_query_id_is_written_as_text_in_lexicographic_order` (fixture
    ids `"1"`, `"100"`, `"1012"`, `"0007"` — the regression test for the
    `dtype=str` decision); `test_counts_formatting_is_fixed_width`
    (regexes over the raw text, `^\d+$` per `count` cell and
    `^\d\.\d{6}$` per `fraction` cell, deliberately not over the
    re-parsed frame); `test_counts_row_order_matches_declared_total_order`;
    `test_composite_run_id_selector_resolves_to_exactly_one_row`;
    `test_run_id_containing_separator_raises_and_writes_neither_report`;
    `test_missing_local_cache_raises_covariate_input_error_and_writes_nothing`
    (points `--config` at a fixture YAML whose `data_dir` is an **empty
    `tmp_path` subdirectory**, pre-creates both outputs with sentinel
    bytes, asserts `main` returns non-zero, that the message names the
    absent `scifact` path, that both outputs still hold their sentinel
    bytes, **and that the temporary `data_dir` is still empty** — the
    "downloads nothing" half of Requirement 16.14, checked by the
    directory staying empty rather than by trusting the offline flags;
    it reaches `assert_local_cache_present` and stops, so no `beir`
    import and no download attempt occurs);
    `test_asymmetric_query_set_raises_naming_pair_and_query_id`;
    `test_missing_input_file_and_missing_columns_raise_input_error`
    (parametrized over `REQUIRED_COLUMNS`);
    `test_successful_run_prints_derived_counts_and_returns_zero`
    (`capsys`; asserts the printed summary contains the fixture's own
    Run_Id count, per-Run_Id query count, Pair_Contrast count, both
    artifacts' row counts, **each stub model's limit, the covariate
    count, and the sentinel count**, and that exactly the two expected
    files and no third were created);
    `test_limits_are_not_read_from_a_literal_or_a_config` (the
    negative-space test: **no module attribute of
    `src.failure_buckets` equals `256` or `512`**, no import of
    `src.token_length_analysis.MAX_SEQUENCE_LENGTH`, and no argparse
    option matching `max.*len|limit|seq`); and
    `test_verifier_csv_artifacts_includes_both_new_files`.
  - No test reads a file under `results/` or `data/`, loads a real
    model, loads a real tokenizer, reads the real corpus, or makes a
    network call. No test calls `resolve_model_limits`,
    `load_covariate_inputs`' success path, or
    `resolve_effective_max_sequence_length` — limits arrive as a plain
    `{retriever_name: int}` dict, which is what makes Requirement
    15.10 hold without a mock, a monkeypatch or a skip gate.
  - Wave 3. Depends on Tasks 3 and 4 (both stages of the module under
    test) and on Task 2 (`_CSV_ARTIFACTS` membership and the
    `_resolve_csv_reference` import). Must be green before Task 6's real
    run.
  - Done check (suite green, and the whole suite still green):
    `python -m pytest tests/test_failure_buckets.py -q` exits 0, and
    `python -m pytest -q` exits 0, confirming no existing test module
    regressed.
  - Done check (**all ten Requirement-15 named tests are collected**):
    `python -m pytest tests/test_failure_buckets.py --collect-only -q | python -c "import sys; names = ('test_failure_bucket_predicates_cover_all_four_buckets', 'test_contrast_bucket_rules_cover_all_four_buckets', 'test_totality_assertion_failure_writes_neither_report', 'test_two_invocations_produce_byte_identical_reports', 'test_counts_run_id_and_bucket_combinations_are_unique', 'test_bucket_assigner_uses_no_network_no_model_no_data_dir', 'test_covariate_computation_over_stub_corpus_and_stub_tokenizer', 'test_missing_judgment_records_sentinel_not_numeric_zero', 'test_covariates_are_identical_across_a_query_id_rows', 'test_bucket_level_covariate_count_selector_resolves'); out = sys.stdin.read(); missing = [n for n in names if n not in out]; assert not missing, missing; print('ok', len(names), 'Requirement 15 tests collected')"`
    prints `ok 10 Requirement 15 tests collected` and exits 0.
  - Done check (**no skip gate, and no real corpus / model / `data/`
    read**):
    `! grep -Eq 'pytestmark|skipif|skip_if|unittest\.mock|monkeypatch' tests/test_failure_buckets.py && echo ok`
    prints `ok` (Requirement 15.10 forbids a new skip-gated
    real-corpus or real-tokenizer test, and the stubs are hand-written
    rather than mocked); and
    `! grep -Eq "(read_csv|open|Path)\(['\"](results|data)/" tests/test_failure_buckets.py && echo ok`
    prints `ok`.
  - Done check (the stub fixtures have the shapes the requirements
    need):
    `python -c "import re; s = open('tests/test_failure_buckets.py', encoding='utf-8').read(); m = re.search(r'_STUB_LIMITS\s*=\s*\{([^}]*)\}', s); assert m, 'no _STUB_LIMITS'; vals = re.findall(r':\s*(\d+)', m.group(1)); assert len(vals) == 2 and vals[0] != vals[1], ('per-model stub limits must differ', vals); i = s.index('_STUB_QRELS'); assert re.search(r':\s*0\b', s[i:i + 600]), 'stub qrels must include a relevance-0 entry'; assert s.count('_STUB_CORPUS') >= 2; docs = re.findall(r'\bd\d+.\s*:\s*\{', s[s.index('_STUB_CORPUS'):s.index('_STUB_QRELS')]); assert len(docs) <= 5, ('stub corpus must hold no more than 5 documents', len(docs)); assert 'class _StubTokenizer' in s; assert 'test_limits_are_not_read_from_a_literal_or_a_config' in s; print('ok', len(docs), 'stub documents')"`
    prints `ok` with the stub document count (at most 5) and exits 0.
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.8, 15.9, 15.10, 15.11, 6.7, 6.8, 6.9, 7.4, 8.1, 8.2, 16.3, 16.6, 16.8, 16.9_

- [x] 6. Run the Bucket_Assigner for real and commit both artifacts
  - **Precondition — a populated `data/` cache.** This is the only task
    in this plan whose run needs the gitignored cache. The
    Covariate_Enrichment_Stage reads `data/scifact` (through
    `load_scifact`) and both tokenizer snapshot directories under
    `data/hf_cache` (through `load_tokenizer_offline`), read-only and
    offline, and **fails with `CovariateInputError` rather than
    downloading anything** if either is absent. Confirm the cache is
    present *before* running, not by interpreting a failure afterwards.
    If it is absent, populate it by running the sweep or the
    token-length analysis first — that is what populates it — and do not
    "fix" the run by letting the assigner download.
  - Run `python -m src.failure_buckets` once against the committed
    `results/per_query.csv` and `configs/sweep.yaml`, with the default
    output paths. Read the printed Requirement 5.6 summary and confirm
    it reports 2700 rows, 9 Run_Ids, 300 `query_id`s for every Run_Id,
    12 Pair_Contrasts (8 family-aligned + 4 dense cross-strategy), 2700
    `failure_buckets.csv` data rows, and 84 `failure_bucket_counts.csv`
    data rows (36 per-run + 48 per-contrast) — a smaller figure anywhere
    means the input was truncated, and the run is not to be committed.
  - Read the printed Requirement 16.18 summary and confirm it reports
    the `CORPUS_LOAD_REPORT` line from the loader's own output, **each
    Dense_Model's resolved Effective_Max_Sequence_Length** (two
    different values — the two models do not share one), the number of
    `query_id`s whose covariates were computed (300), and the number
    recorded with the Missing_Value_Sentinel. **The expected sentinel
    count on the committed input is 0**, because every one of the 300
    `query_id`s in `results/per_query.csv` has `num_judged_relevant >=
    1` (observed range 1 to 5) and `num_judged_relevant` is derived from
    the same Qrels under the same strictly-greater-than-0 condition the
    covariate stage applies. A non-zero sentinel count therefore means
    those two disagree: that is **a finding to investigate, not a
    covariate to commit**. Stop, find out why the Qrels the stage loaded
    differ from the ones the sweep scored against, and do not commit the
    artifact until the count is 0 or the discrepancy is explained.
  - Commit `results/failure_buckets.csv` and
    `results/failure_bucket_counts.csv`. Every bucket figure and every
    covariate figure `ANALYSIS.md` states in Task 7 is read out of these
    two committed files, never out of this run's console output.
  - No row is dropped, filtered, or edited by hand after the run —
    every configuration in the declared grid keeps its rows regardless
    of which retriever the partition favours, and no covariate cell is
    hand-adjusted.
  - Wave 4. Depends on Task 5 being green. Sole producer of both
    artifacts; Tasks 7 and 8 both consume them.
  - Done check (**cache present, run this BEFORE the run**):
    `test -d data/scifact && test -d data/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2 && test -d data/hf_cache/models--BAAI--bge-small-en-v1.5 && echo ok`
    prints `ok`; and
    `ls -d data/scifact data/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2 data/hf_cache/models--BAAI--bge-small-en-v1.5`
    lists all three paths. If any is missing this task cannot proceed —
    the covariate stage will halt, correctly, having written nothing.
  - Done check (**record `data/`'s pre-run state, also BEFORE the run**,
    so Task 10 can prove nothing was written or added there — `data/` is
    gitignored, so `git status` cannot see it):
    `find data -type f | sort > /tmp/analysis_writeup_data_before.txt && wc -l < /tmp/analysis_writeup_data_before.txt`
    prints the pre-run file count and leaves the listing on disk for
    Task 10's `diff`.
  - Done check (the run's own summary, both required blocks, with the
    two limits cross-checked against a committed artifact rather than
    typed in):
    `TMP=$(mktemp -d) && python -m src.failure_buckets > "$TMP/run.txt" 2>&1; echo "exit=$?"; python -c "import json, sys; log = open(sys.argv[1], encoding='utf-8').read(); rep = json.load(open('results/token_length_report.json', encoding='utf-8')); short = {'sentence-transformers/all-MiniLM-L6-v2': 'all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5': 'bge-small-en-v1.5'}; lim = {short[c['model_name']]: c['max_sequence_length'] for c in rep['cells']}; assert len(set(lim.values())) == 2, lim; missing = [k for k in lim if (k + '=' + str(lim[k])) not in log]; assert not missing, ('resolved limit not printed for', missing, lim); assert 'CORPUS_LOAD_REPORT' in log; assert '2700' in log and '84' in log and '300' in log and '12' in log; import re; m = re.search(r'recorded as .NA.: (\d+)', log); assert m, 'the Requirement 16.18 sentinel line was not printed'; assert m.group(1) == '0', ('sentinel count must be 0 on the committed input -- investigate, do not commit', m.group(1)); m2 = re.search(r'covariates computed: (\d+)', log); assert m2 and m2.group(1) == '300', ('covariate count', m2 and m2.group(1)); print('ok')" "$TMP/run.txt"; rm -rf "$TMP"`
    prints `exit=0` and then `ok`, and the final `python -c` exits 0.
  - Done check (**qrels identity — the freshly-loaded qrels the
    Covariate_Enrichment_Stage actually used agree, query by query and
    exactly, with the committed `results/per_query.csv`'s
    `num_judged_relevant`; this is stricter than the sentinel-count-is-0
    check above, which only catches a query with *zero*
    judged-relevant documents in the loaded Qrels and cannot catch a
    qrels snapshot that drifted to a *different* set of relevant
    document IDs while still leaving every query with at least one —
    that drift would still print a sentinel count of 0 while silently
    computing covariates against a different notion of
    "judged relevant" than the one the committed metrics were scored
    against**): loads the real, cached qrels via the same
    `src.config.load_sweep_config` + `src.corpus_loader.load_scifact`
    path the Covariate_Enrichment_Stage itself calls, then compares
    `len(src.metrics.judged_relevant_docs(qrels.get(query_id, {})))`
    against `per_query.csv`'s committed `num_judged_relevant` for every
    one of the (deduplicated) 300 `query_id`s the file contains — an
    exact integer equality, not a tolerance:
    `python -c "import pandas as pd; from src.config import load_sweep_config; from src.corpus_loader import load_scifact; from src.metrics import judged_relevant_docs; config = load_sweep_config('configs/sweep.yaml'); bundle, _report = load_scifact(config.data_dir); pq = pd.read_csv('results/per_query.csv', dtype=str, keep_default_na=False); groups = pq.groupby('query_id')['num_judged_relevant'].apply(lambda s: sorted(set(s))); nonconstant = {qid: vals for qid, vals in groups.items() if len(vals) != 1}; assert not nonconstant, ('num_judged_relevant is not constant per query_id in per_query.csv -- cannot compare', nonconstant); committed = {qid: int(vals[0]) for qid, vals in groups.items()}; fresh = {qid: len(judged_relevant_docs(bundle.qrels.get(qid, {}))) for qid in committed}; compared = len(fresh); assert compared == 300, ('expected exactly 300 query_ids compared, got', compared); mismatches = sorted((qid, committed[qid], fresh[qid]) for qid in committed if committed[qid] != fresh[qid]); assert not mismatches, ('STOP and investigate -- do not commit: per_query.csv num_judged_relevant disagrees with the freshly-loaded qrels judged_relevant_docs count for these (query_id, per_query.csv_value, freshly_loaded_value) triples', mismatches); print('ok', compared, 'query_ids agree exactly between per_query.csv and the freshly-loaded qrels')"`
    prints `ok` followed by `300` and exits 0. A failure names every
    disagreeing `query_id` together with `per_query.csv`'s value and the
    freshly-loaded qrels' value, or reports fewer than 300 compared
    (which itself fails the check, catching a partial comparison from a
    `query_id` mismatch or a silently-defaulting dict lookup) — this is
    a provable claim ("the qrels the covariates use are exactly the
    qrels the published metrics were scored against"), not a
    probabilistic one, so there is no tolerance and no partial-pass
    path: stop and investigate rather than commit.
  - Done check (**the twelve-column header, in Requirement 6.1's exact
    order, and every cell rendered per Requirement 6.8** — asserted
    against the RAW text, never the re-parsed frame):
    `python -c "import re; HDR = 'run_id,retriever,chunking_strategy,query_id,bucket,num_judged_relevant,query_token_len__all-MiniLM-L6-v2,max_relevant_doc_token_len__all-MiniLM-L6-v2,any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2,query_token_len__bge-small-en-v1_5,max_relevant_doc_token_len__bge-small-en-v1_5,any_relevant_doc_exceeds_limit__bge-small-en-v1_5'; raw = open('results/failure_buckets.csv', encoding='utf-8').read().splitlines(); assert raw[0] == HDR, raw[0]; assert '.' not in raw[0], ('no header cell may contain a dot -- the Verifier splits at the LAST dot', raw[0]); body = [l for l in raw[1:] if l.strip()]; assert len(body) == 2700, len(body); cells = [l.split(',') for l in body]; assert all(len(c) == 12 for c in cells), 'every data row must hold exactly 12 fields'; assert all(re.fullmatch(r'\d+', c[i]) for c in cells for i in (5, 6, 9)), 'num_judged_relevant and query_token_len cells must be base-ten integers with no decimal point'; assert all(re.fullmatch(r'\d+|NA', c[i]) for c in cells for i in (7, 10)), 'max_relevant_doc_token_len cells must be a base-ten integer or exactly NA'; bad = sorted({c[i] for c in cells for i in (8, 11)} - {'exceeds', 'within', 'NA'}); assert not bad, ('exceedance cells must be exactly exceeds, within, or NA -- never true/false/True/False/1/0/empty', bad); na = sum(1 for c in cells for i in (7, 8, 10, 11) if c[i] == 'NA'); assert na == 0, ('sentinel cells present on the committed input: a finding to investigate, not an artifact to commit', na); print('ok')"`
    prints `ok` and exits 0.
  - Done check (per-query artifact semantics, passthrough, and covariate
    run-independence):
    `python -c "import pandas as pd; df = pd.read_csv('results/failure_buckets.csv', dtype=str, keep_default_na=False); assert len(df) == 2700; assert df.duplicated(subset=['run_id', 'query_id']).sum() == 0; assert df['bucket'].isin(['total_miss', 'mis_ranked', 'partial_recall', 'full_success']).all(); assert df['run_id'].nunique() == 9; assert (df.groupby('run_id')['query_id'].nunique() == 300).all(); cov = [c for c in df.columns if c.startswith(('query_token_len', 'max_relevant_doc_token_len', 'any_relevant_doc_exceeds_limit'))]; assert len(cov) == 6, cov; assert (df.groupby('query_id')[cov].nunique() == 1).all().all(), 'every covariate must be identical across a query_id rows'; assert df[df['retriever'] == 'bge-small-en-v1.5'].shape[0] > 0 and df['run_id'].str.contains('bge-small-en-v1.5', regex=False).any(), 'run_id and retriever VALUES keep their dot'; pq = pd.read_csv('results/per_query.csv', dtype=str, keep_default_na=False); j = df.merge(pq, on=['run_id', 'query_id'], suffixes=('', '_pq')); assert len(j) == 2700; assert (j['retriever'] == j['retriever_pq']).all(); assert (j['chunking_strategy'] == j['chunking_strategy_pq']).all(); assert (j['num_judged_relevant'] == j['num_judged_relevant_pq']).all(); print('ok')"`
    prints `ok` and exits 0.
  - Done check (counts artifact — four columns only, unique selectors,
    fixed-width formatting, and **the rendered fraction sums**):
    `python -c "import pandas as pd, re; raw = open('results/failure_bucket_counts.csv', encoding='utf-8').read().splitlines(); assert raw[0] == 'run_id,bucket,count,fraction', raw[0]; body = [l for l in raw[1:] if l.strip()]; assert len(body) == 84, len(body); cells = [l.rsplit(',', 2) for l in body]; assert all(re.fullmatch(r'\d+', c[1]) for c in cells), 'count formatting'; assert all(re.fullmatch(r'\d\.\d{6}', c[2]) for c in cells), 'fraction must be fixed-point with exactly 6 decimals'; df = pd.read_csv('results/failure_bucket_counts.csv', dtype={'run_id': str, 'bucket': str}); assert list(df.columns) == ['run_id', 'bucket', 'count', 'fraction'], ('the covariate stage must not add a fifth column', list(df.columns)); assert df.duplicated(subset=['run_id', 'bucket']).sum() == 0; sums = df.groupby('run_id')['fraction'].sum(); bad = {k: v for k, v in sums.items() if abs(v - 1.0) > 2e-6}; assert not bad, ('rendered fractions must sum to 1 within 2e-6 (Requirement 5.7)', bad); per = df[~df['run_id'].str.contains('|vs|', regex=False)]; con = df[df['run_id'].str.contains('|vs|', regex=False)]; assert len(per) == 36 and len(con) == 48, (len(per), len(con)); assert per['run_id'].nunique() == 9 and con['run_id'].nunique() == 12; assert (per.groupby('run_id')['count'].sum() == 300).all(), per.groupby('run_id')['count'].sum().to_dict(); assert per['bucket'].isin(['total_miss', 'mis_ranked', 'partial_recall', 'full_success']).all(); assert con['bucket'].isin(['a_only', 'b_only', 'both_miss', 'both_answer']).all(); flags = list(df['run_id'].str.contains('|vs|', regex=False)); assert flags == sorted(flags), 'every per-run row must precede every contrast row'; print('ok')"`
    prints `ok` and exits 0. The tighter 1e-9 check of Requirement 5.4
    runs inside `assert_fraction_sums` against the *unrounded* floats
    before the write; only the rendered values survive into the file, so
    2e-6 is the correct tolerance to assert here.
  - Done check (**byte-identical rerun, in scratch directories, without
    overwriting either committed artifact** — the covariate columns
    included, since `cmp` compares whole files):
    `TMP=$(mktemp -d) && python -m src.failure_buckets --per-query results/per_query.csv --buckets-out "$TMP/run1_buckets.csv" --counts-out "$TMP/run1_counts.csv" > "$TMP/log1.txt" && python -m src.failure_buckets --per-query results/per_query.csv --buckets-out "$TMP/run2_buckets.csv" --counts-out "$TMP/run2_counts.csv" > "$TMP/log2.txt" && cmp "$TMP/run1_buckets.csv" "$TMP/run2_buckets.csv" && cmp "$TMP/run1_counts.csv" "$TMP/run2_counts.csv" && grep -q '2700' "$TMP/log1.txt" && grep -q '84' "$TMP/log1.txt" && grep -q '12' "$TMP/log1.txt" && grep -q '300' "$TMP/log1.txt" && grep -q 'effective_max_sequence_length' "$TMP/log1.txt" && grep -Eq 'recorded as .NA.: 0' "$TMP/log1.txt" && python -c "import sys; a = open(sys.argv[1], 'rb').read().replace(b'\r\n', b'\n'); b = open(sys.argv[2], 'rb').read().replace(b'\r\n', b'\n'); assert a == b, 'scratch buckets output differs from the committed artifact'; print('ok')" "$TMP/run1_buckets.csv" results/failure_buckets.csv && python -c "import sys; a = open(sys.argv[1], 'rb').read().replace(b'\r\n', b'\n'); b = open(sys.argv[2], 'rb').read().replace(b'\r\n', b'\n'); assert a == b, 'scratch counts output differs from the committed artifact'; print('ok')" "$TMP/run1_counts.csv" results/failure_bucket_counts.csv && rm -rf "$TMP" && echo ok`
    prints `ok` twice and then `ok`, and exits 0. The two `cmp` calls
    are the Requirement 6.5 / 7.9 rerun identity, and they cover the six
    covariate columns as well as the bucket labels; the two Python
    comparisons normalise line endings before comparing, so the check
    holds whether the working tree currently carries CRLF or LF.
  - _Requirements: 1.2, 1.3, 2.4, 2.6, 5.6, 5.7, 6.1, 6.2, 6.4, 6.5, 6.8, 6.9, 7.1, 7.2, 7.3, 7.4, 7.7, 7.8, 7.9, 16.12, 16.17, 16.18, 16.19_

- [x] 7. Hand-author `ANALYSIS.md` at the repository root
  - Write the document by hand, section by section, following the
    design's thirteen-section skeleton (1–8, **8a**, 9–12). Every
    Numeric_Claim is read out of a committed artifact at authoring time —
    `results/failure_bucket_counts.csv`, `results/failure_buckets.csv`,
    `results/sweep.csv`, `results/per_query.csv`,
    `results/significance.csv`, `results/run_config.json`,
    `results/token_length_report.json`, `results/groundedness.csv`,
    `results/hand_checked_joined.csv`, or
    `results/generated_answers.csv` — never typed from memory or from a
    console log. Throwaway `python -c` snippets may be used to print
    candidate values; none is committed.
  - Section 1: what this document is; names both new artifacts and
    `python -m src.failure_buckets`; states that the per-query
    Failure_Bucket label lives in `results/failure_buckets.csv` rather
    than as a column of `results/sweep.csv`, and why —
    `results/sweep.csv` is keyed by (`run_id`, `k`) and has no per-query
    dimension.
  - Section 2: the Pre_Declared_Family (the 8 nDCG@10 comparisons
    against `bm25__whole_document` under Holm-Bonferroni in
    `results/significance.csv`) is the complete set of inferential
    results the study supports; nDCG@10 is the single primary metric,
    designated before any result existed, with recall@k and MRR@10
    secondary; every bucket count, fraction, pairwise disagreement
    figure **and covariate description** below is a Descriptive_Contrast,
    outside that family, carrying no inferential claim.
  - Section 3: the results table with `bm25__whole_document` as the
    reference row and every non-reference Run_Id's nDCG@10 result stated
    as a `mean_diff` delta against it, alongside `p_value_adjusted` and
    `verdict`, each read from that Run_Id's Pre_Declared_Family row and
    reported as recorded — no verdict omitted, softened, or relegated.
  - Section 4: the four Failure_Bucket predicates in Requirement 3
    Criterion 1's order and wording (including `partial_recall`'s full
    three-clause form), the four Contrast_Bucket rules, the
    Declared_Contrast_Set, the statement that the taxonomy was fixed
    before any label was assigned, and the Answered_Query-is-top-10 vs
    `total_miss`-is-top-20 distinction wherever both are reported.
    **Every one of the 12 Pair_Contrasts is reported**, including the
    ones whose counts are unremarkable — none is dropped for being
    uninteresting or unflattering (Requirement 10.8), which is why the
    contrast set was declared before any count existed.
  - Section 5: the `bge-small-en-v1.5` result — above the reference row
    on the primary metric under all three chunking strategies — citing
    each comparison's `mean_diff`, `p_value_adjusted` and `verdict`, and
    accompanied by **either** a mechanism grounded in named
    Failure_Bucket / Contrast_Bucket figures from
    `results/failure_bucket_counts.csv` **or** the explicit statement
    that no mechanism was identified. All three of these verdicts are
    `significant`, so Requirement 11.2 does not apply and a
    **covariate-grounded** mechanism is permitted here: the shape it may
    take is the over-limit share of each run's `total_miss` bucket
    against that run's remaining queries, read from the named
    `any_relevant_doc_exceeds_limit__*` column and cited by name. If the
    two shares do not separate, say no mechanism was identified rather
    than narrating the non-separation away (Requirement 12.8).
  - Section 6: the `bm25__sentence_window` result (below the reference
    row), citing that comparison's `mean_diff`, `p_value_adjusted` and
    `verdict`, then either a bucket-grounded mechanism or the explicit
    statement that no mechanism was identified. Note the specific reason
    a covariate cannot supply a mechanism *here*: both compared runs are
    BM25, and **BM25 does not truncate**, so a token length measured
    against a *dense model's* limit has no bearing on why sentence-window
    chunking hurt a lexical retriever. A covariate *description* of these
    queries is available and may be given; a covariate-grounded
    *mechanism* is not.
  - Section 7: the four comparisons whose verdict is
    `indistinguishable` — the three `all-MiniLM-L6-v2` runs and
    `bm25__fixed_window`, each against the reference run — described as
    indistinguishable from noise and as a win for neither side, with the
    verdict never restated as evidence that no difference exists. **This
    is the sharpest constraint in the spec and the covariate columns do
    not relax it by one inch** (Requirements 11.2, 12.9): a covariate
    column licenses a *description* of a set of queries and never a
    *mechanism* for a comparison the study could not distinguish from
    noise. The line is drawn at the verb, not at the evidence.
    Permitted: "of the N queries where these two runs disagreed, M had a
    judged-relevant document over `all-MiniLM-L6-v2`'s limit
    (`any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2`)", stated
    together with the fact that the aggregate nDCG@10 difference is
    indistinguishable from noise. Forbidden: any sentence asserting that
    truncation *caused*, *explains*, *accounts for*, *is why*, or
    *drove* the direction of that `mean_diff`. A well-sourced story
    about noise is still a story about noise, and is *more* persuasive to
    a reader, which makes it worse rather than better. State the
    constraint in the section itself, at the point the figures are
    presented, not in a footnote.
  - Section 8: truncation as a **corpus-level** property per
    (Chunking_Strategy, dense model) cell, every figure read from
    `results/token_length_report.json`'s `max_sequence_length`,
    `num_documents_total`, `num_documents_exceeding`,
    `fraction_exceeding`, and `cells` entries, with no token count
    computed in the document and **no figure from this artifact
    attributed to an individual query or an individual Failure_Bucket** —
    a corpus-level fraction is a property of the corpus, which is
    exactly why section 8a exists separately.
  - Section 8a — **truncation at the query level, the sentence this
    document exists to be able to state.** Requirement 12.7 names it:
    that one Run_Id's Missed_Queries are disproportionately the queries
    whose Judged_Relevant_Documents exceed that Dense_Model's
    Effective_Max_Sequence_Length. This is what turns section 8's
    corpus-level fraction — a number that attributes to no query — into
    an account of *which* queries were lost. Its constraints, all stated
    in the section's own prose: every supporting figure is read from a
    **named** `results/failure_buckets.csv` covariate column and cited by
    name, with no number computed in the document; it is a
    Descriptive_Contrast outside the Pre_Declared_Family, with no
    p-value, confidence interval, test statistic, significance
    determination, or distributional claim beyond the 300 queries
    measured; it is stated as a mechanism only for a Run_Id whose
    Pre_Declared_Family verdict is `significant`, and for the four
    `indistinguishable` comparisons it is a description only; and if the
    over-limit share of the `total_miss` bucket does not separate from
    the over-limit share of the remaining queries, the section states
    that no mechanism was identified and stops. The covariate makes the
    claim checkable; it does not make it true.
  - Section 9 — per-query covariates behind the buckets, described
    against the corpus-wide distribution as a description of two
    distributions, never as a hypothesis test and never as evidence of a
    distributional difference beyond the queries measured (Requirement
    12.5). Two covariates, both with real content: **(a)
    `num_judged_relevant`**, inherited from `results/per_query.csv` and
    copied into `results/failure_buckets.csv` — the `partial_recall`
    bucket exists only for queries with more than one judged relevant
    document, and the observed range across all 300 is 1 to 5, so
    `partial_recall` is small for every run and cannot carry much of an
    account; reporting that plainly is the correct outcome. **(b) the six
    token-length covariate columns** — per Run_Id and per
    Failure_Bucket, the share of that bucket's queries whose
    judged-relevant documents exceed each Dense_Model's own limit, and
    the distribution of `query_token_len__*` (SciFact claims are short,
    so the query side sits far under either limit and the document side
    is where the mass is). State the figure that makes the two limits
    non-interchangeable: `all-MiniLM-L6-v2` and `bge-small-en-v1.5` do
    **not** share an Effective_Max_Sequence_Length, both values read from
    `results/token_length_report.json`'s `cells[].max_sequence_length`
    rather than typed in, so the same document can be over the limit for
    one model and under it for the other. Cite every figure to its named
    column. Where a bucket's covariate distribution does not separate
    that bucket from the remaining queries, say no mechanism was
    identified rather than substituting a narrative (Requirement 12.8).
  - Section 10 (optional content, include only if it earns its place):
    a short cross-reference to the groundedness gate's quarantine rate
    and hand-checked agreement, framed as a property of a 30-claim
    subset rather than a retrieval result.
  - Section 11: the limits section, containing all six required
    statements — the taxonomy was fixed before assignment and its counts
    describe the partition rather than test a hypothesis about it;
    sparse qrels mean a `total_miss` records the absence of a *judged
    relevant* document, not of any useful document; every number
    describes BEIR SciFact only and may not transfer to another corpus
    or domain; **each Token_Length_Covariate measures the source corpus
    document's own `title` and `text`, not the Chunk a `fixed_window` or
    `sentence_window` run actually encoded** (Requirement 13.7) — a
    mismatch of *unit*, not an absence of measurement: a 400-token
    document is over `all-MiniLM-L6-v2`'s limit as a whole document and
    under it as any of its windows, so "this run's misses are the
    over-limit queries" is a sharper claim for a `whole_document` run
    than for the other two, and the section says so; and any mechanism
    offered is an account consistent with the bucket figures **and the
    covariate columns**, not a causal result the study established.
    Questions the data cannot answer are stated here as limits of this
    study. Nothing anywhere in the document describes hybrid retrieval,
    score fusion, reciprocal-rank fusion, cross-encoder or
    language-model reranking, a fourth retriever, query expansion or
    rewriting, generated pseudo-queries, an approximate nearest
    neighbour index, or fine-tuning as work to build, propose, or
    recommend.
  - Section 12: the two reproduction commands,
    `python -m src.failure_buckets` and
    `python -m src.verify_writeup_numbers --repo-root .`, with the one
    asymmetry between them stated honestly — the verifier needs only
    committed files and runs on a clean checkout, while the assigner's
    Covariate_Enrichment_Stage needs a populated `data/` and **fails
    rather than downloading it**, so a clean checkout can verify every
    number in this document but cannot regenerate the six covariate
    columns without first populating the cache.
  - Wave 5. Depends on Task 6 — both artifacts must exist before the
    prose that cites them is written. Task 8 appends this document's
    ledger rows.
  - Done check (structure and required statements):
    `python -c "t = open('ANALYSIS.md', encoding='utf-8').read(); low = t.lower(); assert 'results/failure_buckets.csv' in t and 'results/failure_bucket_counts.csv' in t; assert 'results/sweep.csv' in t and 'results/token_length_report.json' in t; assert 'python -m src.failure_buckets' in t; assert 'python -m src.verify_writeup_numbers --repo-root .' in t; assert all(b in t for b in ('total_miss', 'mis_ranked', 'partial_recall', 'full_success')); assert all(b in t for b in ('a_only', 'b_only', 'both_miss', 'both_answer')); assert 'holm' in low and 'ndcg@10' in low and 'descriptive' in low; assert 'qrel' in low and 'scifact' in low and 'causal' in low; assert 'no difference exists' not in low; print('ok')"`
    prints `ok` and exits 0.
  - Done check (**the covariate columns are cited by name, and the
    query-level section exists**):
    `python -c "t = open('ANALYSIS.md', encoding='utf-8').read(); low = t.lower(); named = ['any_relevant_doc_exceeds_limit__all-MiniLM-L6-v2', 'any_relevant_doc_exceeds_limit__bge-small-en-v1_5', 'query_token_len__all-MiniLM-L6-v2', 'max_relevant_doc_token_len__all-MiniLM-L6-v2']; hits = [c for c in named if c in t]; assert len(hits) >= 2, ('at least two covariate columns must be cited by name (Requirement 12.6)', hits); assert 'num_judged_relevant' in t; assert 'effective_max_sequence_length' in low or 'max_sequence_length' in low; assert 'source' in low and 'chunk' in low, 'Requirement 13.7: the covariate measures the source document, not the Chunk'; print('ok', hits)"`
    prints `ok` with the cited column names and exits 0.
  - Done check (reference row and the eight family rows):
    `python -c "import pandas as pd; t = open('ANALYSIS.md', encoding='utf-8').read(); sig = pd.read_csv('results/significance.csv', dtype=str, keep_default_na=False); fam = sig[(sig['metric'] == 'ndcg_at_10') & (sig['is_primary'].str.lower() == 'true')]; assert len(fam) == 8, len(fam); missing = [r for r in fam['run_id'] if r not in t]; assert not missing, missing; assert 'bm25__whole_document' in t; print('ok')"`
    prints `ok` and exits 0.
  - Done check (every Pair_Contrast is reported — Requirement 10.8):
    `python -c "import pandas as pd; t = open('ANALYSIS.md', encoding='utf-8').read(); cnt = pd.read_csv('results/failure_bucket_counts.csv', dtype=str, keep_default_na=False); comp = sorted({r for r in cnt['run_id'] if '|vs|' in r}); assert len(comp) == 12, len(comp); missing = [c for c in comp if c not in t and c.split('|vs|')[1] not in t]; assert not missing, ('no Pair_Contrast may be dropped for being unremarkable', missing); print('ok', len(comp))"`
    prints `ok 12` and exits 0.
  - Done check (the four `indistinguishable` verdicts are described as
    indistinguishable):
    `python -c "import pandas as pd; t = open('ANALYSIS.md', encoding='utf-8').read(); sig = pd.read_csv('results/significance.csv', dtype=str, keep_default_na=False); rows = sig[(sig['metric'] == 'ndcg_at_10') & (sig['is_primary'].str.lower() == 'true') & (sig['verdict'] == 'indistinguishable')]['run_id'].tolist(); assert len(rows) == 4, rows; bad = [r for r in rows if 'indistinguishable' not in t[max(0, t.find(r) - 1500): t.find(r) + 1500].lower()]; assert not bad, bad; print('ok', rows)"`
    prints `ok` followed by the four run_ids, and exits 0.
  - Done check (**no covariate-grounded mechanism near an
    `indistinguishable` comparison** — Requirements 11.2, 12.9, the
    sharpest constraint in the spec). This one surfaces hits for human
    review rather than deciding by itself, because it is a judgment about
    prose: a mechanism verb and a truncation term can legitimately
    co-occur near a run_id in a passage that is *stating the
    prohibition*. Read every hit; a real violation is reworded before
    this task is done.
    `python -c "import pandas as pd, re; t = open('ANALYSIS.md', encoding='utf-8').read(); sig = pd.read_csv('results/significance.csv', dtype=str, keep_default_na=False); ind = sig[(sig['metric'] == 'ndcg_at_10') & (sig['is_primary'].str.lower() == 'true') & (sig['verdict'] == 'indistinguishable')]['run_id'].tolist(); assert len(ind) == 4, ind; verbs = re.compile(r'because|caused|explains|explained by|accounts for|due to|is why|driven by|drove|attributable|mechanism', re.I); cov = re.compile(r'truncat|token_len|exceeds_limit|over the limit|token length', re.I); wins = [(r, t[max(0, m.start() - 900): m.start() + 900]) for r in ind for m in re.finditer(re.escape(r), t)]; flag = [(r, ' '.join(w.split())[:200]) for r, w in wins if verbs.search(w) and cov.search(w)]; print('\n'.join(r + ' :: ' + s for r, s in flag)); print('REVIEW' if flag else 'ok')"`
    prints `ok` with no preceding lines when no window pairs a mechanism
    verb with a covariate term; every `REVIEW` line must be read and
    resolved.
  - Done check (prohibited scope): first, surface every mention for
    human review —
    `grep -nEi 'hybrid|fusion|\bRRF\b|rerank|cross-encoder|FAISS|HNSW|query expansion|query rewrit|pseudo-quer|fine-tun|approximate nearest' ANALYSIS.md`
    — a limits-section mention naming a question this study cannot
    answer is legitimate, so the hits are read rather than required to
    be zero. Then the mechanical half:
    `grep -nEi '(hybrid|fusion|rrf|rerank|cross-encoder|faiss|hnsw|query expansion|query rewrit|pseudo-quer|fine-tun|approximate nearest)' ANALYSIS.md | grep -Ei 'future work|next step|we (should|could|plan)|recommend|propose|should be (built|added)|would improve|worth (building|trying)' && echo REVIEW || echo ok`
    prints `ok` (no mention carries a build/propose/recommend framing);
    a `REVIEW` line must be resolved by rewording before this task is
    done.
  - _Requirements: 1.1, 1.5, 1.6, 3.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 8. Append the `ANALYSIS.md` rows to `docs/numeric_traceability.csv` and run the Verification_Pass
  - For every Numeric_Claim in `ANALYSIS.md`, append exactly one ledger
    row with `document` set to the literal `ANALYSIS.md`, after the
    existing 167 `README.md`/`SPEC.md` rows, which keep their content
    and relative order — a pure append, so `git diff` on the ledger
    shows added lines only. Each row's `source_artifact` is one of
    `failure_bucket_counts.csv`, `failure_buckets.csv`, `sweep.csv`,
    `per_query.csv`, `significance.csv`, `groundedness.csv`,
    `hand_checked_joined.csv`, `generated_answers.csv`,
    `run_config.json`, or `token_length_report.json`; each row's
    `computation` is an existing `_ALLOWED_COMPUTATIONS` member
    (`copy` for a count / `mean_diff` / `p_value_adjusted` / `verdict`,
    `ratio` for a count over a count, `percentage` for a fraction stated
    as a percentage); a `source_fields` value containing a `,` is
    double-quoted, exactly as the existing comma-bearing rows are. `|`
    needs no quoting under any CSV dialect and is not a delimiter at any
    level of the `source_fields` grammar, so a Composite_Run_Id selector
    carries `|vs|` through untouched.
  - Prefer the counts artifact over `failure_buckets.csv` wherever both
    could serve a figure: `run_id={value},bucket={value}.count` is one
    selector, one row, one number, with no dependence on the
    `__count__` sentinel, and a zero count is a *stated zero in a real
    row* rather than an empty match. `failure_buckets.csv` is cited only
    where the counts file has nothing to offer — which, after the
    covariate columns, means the two covariate shapes below.
  - **Shape (e), a single-query covariate figure — a two-filter
    single-row selector**:
    `run_id=X,query_id=Y.{covariate}__{model_tag}` against
    `failure_buckets.csv`. The `(run_id, query_id)` key is unique, so
    this resolves to exactly one row. The `run_id` filter does no
    semantic work — the covariate is run-independent, so all nine of that
    query's rows carry the same value — but it is *required*, because a
    `query_id`-only selector matches nine rows and
    `_resolve_csv_reference` raises on that. Cite the Reference_Run by
    convention, so a reader does not read run-specificity into a value
    that has none.
  - **Shape (f), a bucket-level covariate aggregate — a three-filter
    `__count__` selector, committed as a triple**:
    `run_id=X,bucket=B,{covariate}__{model_tag}=V.__count__` against
    `failure_buckets.csv`, committed **alongside its `=within`
    complement and the same bucket's own `count` from
    `failure_bucket_counts.csv`**, so that the two `__count__` values
    must sum to the third — three rows, three selectors, one arithmetic
    identity a reader can check by eye and a mistyped filter breaks.
    Never ledger a covariate `__count__` row alone: `__count__` returns
    `0.0` *silently* for a filter that matched nothing (the branch
    returns before the one-row check), which on its own is
    indistinguishable from a bucket that genuinely has no such query.
    The triple is the mitigation for that one genuinely silent failure
    mode.
  - **The filter literal for the exceedance covariate column is
    `=exceeds` / `=within`, matching the committed file's text
    exactly.** `exceeds` and `within` are not coercible to a boolean or
    numeric dtype, so `_read_csv_artifact`'s plain `pandas.read_csv`
    call (no `dtype` argument) reads the column back as text regardless
    of whether any row also holds the Missing_Value_Sentinel elsewhere
    in that column, and `_resolve_csv_reference`'s
    `frame[key].astype(str) == value` comparison is the identity on it.
    There is no capitalization or dtype-inference trap to route around
    here, unlike the old `true`/`false` encoding — the literal that
    appears in the file is the literal that belongs in the selector.
    The same reasoning applies to `max_relevant_doc_token_len__*`,
    which is `int64` with no sentinel present. Teaching the Verifier a
    `dtype=str` is forbidden by Requirement 8.2, and is unnecessary here
    regardless: `_read_csv_artifact`'s default inference already reads
    this column as text.
  - Run `python -m src.verify_writeup_numbers --repo-root .` over the
    whole ledger. For any `MISMATCH` or `ERROR`, fix `ANALYSIS.md`'s
    stated text, the row's selector, or the row's declared precision —
    never by editing a cited artifact. If a number cannot be resolved
    through a single ledger row, remove it from `ANALYSIS.md` rather
    than stating it unledgered. Re-run until every row reports `MATCH`.
  - Perform the manual completeness read the repo-writeup design
    established: re-read `ANALYSIS.md` line by line and confirm every
    number in its prose has a ledger row. This is not automatable and
    is not covered by the exit-code checks below.
  - Wave 6. Depends on Tasks 6 and 7 (both artifacts and the document
    must exist) and on Task 2 (`_CSV_ARTIFACTS`). Independent of Task 9,
    which touches a different file and states no number.
  - Done check (append-only, and every appended row well-formed):
    `python -c "import pandas as pd, src.verify_writeup_numbers as v; df = pd.read_csv('docs/numeric_traceability.csv', dtype=str, keep_default_na=False); assert set(df.iloc[:167]['document'].unique()) <= {'README.md', 'SPEC.md'}, df.iloc[:167]['document'].unique(); assert (df.iloc[167:]['document'] == 'ANALYSIS.md').all(), 'appended rows must all be ANALYSIS.md rows'; rows = df[df['document'] == 'ANALYSIS.md']; assert len(rows) > 0, 'no ANALYSIS.md ledger rows'; assert df['claim_id'].is_unique; allowed = {'failure_bucket_counts.csv', 'failure_buckets.csv', 'sweep.csv', 'per_query.csv', 'significance.csv', 'groundedness.csv', 'hand_checked_joined.csv', 'generated_answers.csv', 'run_config.json', 'token_length_report.json'}; bad = sorted(set(rows['source_artifact']) - allowed); assert not bad, bad; badc = sorted(set(rows['computation']) - set(v._ALLOWED_COMPUTATIONS)); assert not badc, badc; t = open('ANALYSIS.md', encoding='utf-8').read(); absent = [(r.claim_id, r.stated_value) for r in rows.itertuples() if r.stated_value not in t]; assert not absent, absent; print('ok', len(rows), 'ANALYSIS.md rows')"`
    prints `ok` with the appended row count, and exits 0 — this is the
    "every ledger row's stated value appears verbatim in the document"
    half of the receipt rule.
  - Done check (**every exceedance covariate filter literal matches the
    file's own text**):
    `python -c "import pandas as pd, re; df = pd.read_csv('docs/numeric_traceability.csv', dtype=str, keep_default_na=False); rows = df[df['document'] == 'ANALYSIS.md']; bad = [(r.claim_id, r.source_fields) for r in rows.itertuples() if 'exceeds_limit' in r.source_fields and re.search(r'=(true|false|True|False)[\.;]', r.source_fields)]; assert not bad, ('an exceedance covariate filter must read =exceeds/=within, matching the committed file text exactly -- not the old true/false/True/False encoding', bad); print('ok')"`
    prints `ok` and exits 0.
  - Done check (**every covariate `__count__` triple satisfies its
    arithmetic identity**):
    `python -c "import pandas as pd, re; led = pd.read_csv('docs/numeric_traceability.csv', dtype=str, keep_default_na=False); rows = led[(led['document'] == 'ANALYSIS.md') & (led['source_artifact'] == 'failure_buckets.csv') & led['source_fields'].str.contains('any_relevant_doc_exceeds_limit', regex=False) & led['source_fields'].str.contains('__count__', regex=False)]; pat = re.compile(r'run_id=([^,]+),bucket=([^,]+),(any_relevant_doc_exceeds_limit__[^=]+)=(exceeds|within)\.__count__'); hits = [(pat.fullmatch(s), int(float(val))) for s, val in zip(rows['source_fields'], rows['stated_value'])]; unmatched = [s for s, h in zip(rows['source_fields'], hits) if h[0] is None]; assert not unmatched, ('a covariate __count__ selector does not have the declared three-filter shape', unmatched); assert hits, 'section 8a states a bucket-level covariate aggregate, so at least one triple must be ledgered'; cnt = pd.read_csv('results/failure_bucket_counts.csv', dtype=str, keep_default_na=False); keys = sorted({(m.group(1), m.group(2), m.group(3)) for m, _ in hits}); grp = lambda k: {m.group(4): v for m, v in hits if (m.group(1), m.group(2), m.group(3)) == k}; tot = lambda k: int(cnt[(cnt['run_id'] == k[0]) & (cnt['bucket'] == k[1])]['count'].iloc[0]); bad = [(k, grp(k), tot(k)) for k in keys if set(grp(k)) != {'exceeds', 'within'} or sum(grp(k).values()) != tot(k)]; assert not bad, ('count(exceeds) + count(within) must equal the bucket count', bad); print('ok', len(keys), 'covariate __count__ triples satisfy their identity')"`
    prints `ok` with the triple count and exits 0.
  - Done check (the 167 pre-existing rows were not rewritten): before
    committing,
    `test "$(git diff -- docs/numeric_traceability.csv | grep -c '^-[^-]')" = "0" && echo ok`
    prints `ok`, confirming the edit removed or altered no existing
    line.
  - Done check (Verification_Pass):
    `TMP=$(mktemp -d); python -m src.verify_writeup_numbers --repo-root . > "$TMP/vp.txt"; echo "exit=$?"; python -c "import sys, pandas as pd; out = open(sys.argv[1], encoding='utf-8').read(); df = pd.read_csv('docs/numeric_traceability.csv', dtype=str, keep_default_na=False); rows = df[df['document'] == 'ANALYSIS.md']; assert len(rows) > 0; bad = [c for c in rows['claim_id'] if (c + ': MATCH') not in out]; assert not bad, bad; assert 'MISMATCH' not in out and 'ERROR' not in out; assert f'{len(df)} total' in out; print('ok', len(rows), 'ANALYSIS.md rows matched')" "$TMP/vp.txt"; rm -rf "$TMP"`
    prints `exit=0` and then `ok` with the ANALYSIS.md row count, and
    the final `python -c` exits 0. The pass reads only committed files —
    no corpus, no Qrels, no tokenizer, no model, nothing under `data/`
    (Requirement 15.11).
  - _Requirements: 1.5, 8.1, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 12.6, 15.11_

- [x] 9. Add the one-line `ANALYSIS.md` filename reference to `README.md`
  - Make exactly one edit whose sole effect is to reference
    `ANALYSIS.md` by filename — the natural home is `README.md`'s
    existing "See `SPEC.md` for the full design and threats to validity"
    pointer, extended to name `ANALYSIS.md` as the mechanism /
    failure-bucket analysis. `SPEC.md` may carry the same one-line
    pointer instead, but not a second copy of it.
  - Add no number, no metric, no table row, and no results claim in
    this edit, so no new ledger row is required and no existing ledger
    row's document text changes.
  - Wave 6. Independent of Task 8 (different file, states no number).
  - Done check: `grep -n 'ANALYSIS.md' README.md SPEC.md` prints at
    least one line;
    `test "$(git diff --numstat -- README.md SPEC.md | awk '{s+=$1+$2} END {print s+0}')" -le 4 && echo ok`
    prints `ok`, confirming the edit is a one-line pointer rather than a
    rewrite; and
    `git diff -U0 -- README.md SPEC.md | grep -E '^\+[^+]' | grep -Eq '[0-9]' && echo REVIEW || echo ok`
    prints `ok`, confirming no digit was added — i.e. no Numeric_Claim
    was introduced. Both `git diff` checks must be run before
    committing the edit.
  - _Requirements: 1.4_

- [x] 10. Final read-only audit of the untouched boundary
  - Confirm, without editing anything, that this feature's whole change
    set is: `src/errors.py` (append), `src/verify_writeup_numbers.py`
    (three-part edit), `src/failure_buckets.py` (new, both stages),
    `tests/test_failure_buckets.py` (new),
    `results/failure_buckets.csv` (new),
    `results/failure_bucket_counts.csv` (new), `ANALYSIS.md` (new),
    `docs/numeric_traceability.csv` (append), and the one-line
    `README.md`/`SPEC.md` pointer — and nothing else. In particular
    `requirements.txt` is unchanged: the covariate stage uses only the
    already-pinned `transformers` and `sentence-transformers` entries and
    introduces no new top-level dependency (Requirement 1.9).
  - Confirm every protected file is byte-for-byte unchanged:
    `results/sweep.csv`, `results/per_query.csv`,
    `results/significance.csv`, `results/run_config.json`,
    `results/token_length_report.json`, `results/groundedness.csv`,
    `results/generated_answers.csv`, `results/hand_checked_sample.csv`,
    `results/hand_checked_joined.csv`,
    `results/hand_checked_sample_context.md`,
    `docs/claim_assertion_classification.csv`, `configs/sweep.yaml`,
    `configs/significance.yaml`, `configs/groundedness.yaml`,
    `requirements.txt`, and `.github/workflows/ci.yml`. **`configs/sweep.yaml`
    is now READ** — by `load_sweep_config`, for its `data_dir` field
    only — and must still be unmodified; `configs/significance.yaml` and
    `configs/groundedness.yaml` remain neither read nor written.
  - Confirm **`data/` was read but never written to or added to**. It is
    gitignored, so `git status` cannot see it: instead compare the
    pre-run listing Task 6 recorded against the current one, so the
    claim rests on a file-level comparison rather than on trusting the
    offline flags. The covariate stage's own printed
    `CORPUS_LOAD_REPORT` line is the corroborating evidence that it read
    the cache rather than fetching it.
  - Wave 7. Depends on every preceding task: the audit's entire content
    is a statement about the finished working tree, so it cannot run
    alongside a task still editing it.
  - Done check (**`data/` unchanged — nothing written, nothing
    downloaded**), against the listing Task 6 recorded before its run:
    `find data -type f | sort > /tmp/analysis_writeup_data_after.txt && diff /tmp/analysis_writeup_data_before.txt /tmp/analysis_writeup_data_after.txt && test "$(wc -l < /tmp/analysis_writeup_data_before.txt)" = "$(wc -l < /tmp/analysis_writeup_data_after.txt)" && echo ok`
    prints `ok` with no `diff` output — an added path would appear as a
    `>` line and a removed one as a `<` line. If the pre-run listing was
    not recorded, this check cannot be satisfied retroactively: re-record
    it, re-run Task 6 into a scratch directory, and compare.
  - Done check (protected paths untouched), run before committing:
    `test -z "$(git status --porcelain -- results/sweep.csv results/per_query.csv results/significance.csv results/run_config.json results/token_length_report.json results/groundedness.csv results/generated_answers.csv results/hand_checked_sample.csv results/hand_checked_joined.csv results/hand_checked_sample_context.md docs/claim_assertion_classification.csv configs/sweep.yaml configs/significance.yaml configs/groundedness.yaml requirements.txt .github/workflows/ci.yml)" && echo ok`
    prints `ok`. After committing, the same claim is checked over the
    commit range with
    `git diff --name-only <first-commit-of-this-feature>~1 HEAD -- results/sweep.csv results/per_query.csv results/significance.csv results/run_config.json results/token_length_report.json results/groundedness.csv results/generated_answers.csv results/hand_checked_sample.csv results/hand_checked_joined.csv results/hand_checked_sample_context.md docs/claim_assertion_classification.csv configs/sweep.yaml configs/significance.yaml configs/groundedness.yaml requirements.txt .github/workflows/ci.yml`,
    which must print nothing.
  - Done check (exactly two new `results/` files), run before
    committing: `git status --porcelain -- results/ | awk '{print $2}' | sort`
    prints exactly `results/failure_bucket_counts.csv` and
    `results/failure_buckets.csv`, and no third path.
  - Done check (whole change set, for human review):
    `git status --short` and `git diff --stat` list only the paths
    enumerated in the first sub-bullet.
  - Done check (final gates): `python -m pytest -q` exits 0 and
    `python -m src.verify_writeup_numbers --repo-root . > /dev/null; echo $?`
    prints `0`.
  - _Requirements: 1.3, 1.4, 1.7, 1.9, 15.7_

## Notes

- Requirements coverage: every requirement group 1 through **16** is
  referenced by at least one task. Requirement 1 lands in Tasks 3, 4, 6,
  7, 9 and 10; Requirement 2 in Tasks 3 (the corpus-free assignment
  stage) and 4 (the entry point and the covariate stage); Requirement 3
  in Tasks 3 (predicates) and 7 (the document restating them);
  Requirements 4-7 in Tasks 3 and 6; Requirement 8 in Task 2;
  Requirement 9 in Task 8; Requirements 10-14 in Task 7; Requirement 15
  in Task 5 (15.1-15.6, 15.8-15.11), Task 8 (15.11's
  corpus-free Verification_Pass) and Task 10 (15.7's unchanged CI); and
  Requirement 16 in Task 4 (all eighteen criteria) with 16.12, 16.17 and
  16.18 re-checked against the real run in Task 6. No requirement is
  left to "implied by another task".
- `src/failure_buckets.py` is written by two tasks in two consecutive
  waves (Tasks 3 and 4), one per stage, for the reasons given in the
  Overview: the module is too large for one reviewable unit, and the two
  stages have genuinely different boundaries — one may not touch a
  corpus, a tokenizer or a limit, the other is the only thing in this
  spec that may. They are sequential, not parallel, because they write
  the same file and because Task 4's `main` and `attach_covariates`
  consume Task 3's builders, assertions and twelve-column schema.
- No task is marked optional. Every task in this plan is a release
  gate: the two artifacts, `ANALYSIS.md`, the ledger rows, and the test
  module are all named deliverables, and the sibling specs' task lists
  use no `*` marker either. The only genuinely optional *content* is
  `ANALYSIS.md`'s section 10 (the groundedness cross-reference), which
  is a judgment inside Task 7, not a separate task.
- Property-based testing appears in exactly one place, Task 5:
  `hypothesis` (already pinned at `hypothesis==6.167.1`, already used by
  `tests/test_chunking.py`) drives Properties 1, 2, 4 and the pure
  `max_relevant_doc_token_len` — three pure scalar/collection functions
  plus the partition invariants over small generated frames. Everything
  else in this spec is hand-built fixtures and hand-written stubs,
  because its guarantees are about one artifact's exact bytes, an
  enumerable set of malformed inputs, a single selector resolution, or
  prose. `compute_token_length_covariates` is a deliberate exclusion
  despite being pure: generating a corpus, a qrels mapping and a
  tokenizer consistently enough to state a property means generating an
  oracle, and the oracle would be a second implementation of the
  function under test.
- Task 6 is the only task in this plan that needs the gitignored `data/`
  cache, and it needs it as a **precondition it checks**, not as
  something it produces. `data/scifact` and both tokenizer snapshot
  directories under `data/hf_cache` must already be present; the
  covariate stage reads them read-only and offline, after a pre-flight
  presence check that runs *before* `load_scifact` because that function
  downloads when the cache is empty. Nothing in this plan downloads a
  dataset or a model, and nothing writes under `data/`.
- Task 6 is a manual real-artifact run and Task 7 a hand-authoring
  step, not pytest tasks — matching this repo's established pattern
  (session 1, significance-testing, repo-writeup, and
  groundedness-gate each defer real-artifact runs and document
  authoring to non-automated steps). No automated test reads a file
  under `results/` or `data/`, loads a real model or tokenizer, or makes
  a network call, and `.github/workflows/ci.yml` stays unchanged, so CI
  keeps installing `requirements.txt` and running `pytest` only.
  Requirement 15.10 also rules out adding a *skip-gated* real-corpus or
  real-tokenizer test, which is why Task 5's done check greps for
  `pytestmark`/`skipif` rather than merely intending their absence.
- `ANALYSIS.md` is hand-authored and comes *after* both artifacts
  exist. No task templates, generates, or renders any part of it from
  code — a code path that both computed a value and rendered it into
  the document would leave the Verification_Pass verifying nothing.
- The covariate columns license a **description** of a set of queries and
  never a **mechanism** for a comparison whose Pre_Declared_Family
  verdict is `indistinguishable`. That constraint sits in Task 7's
  section 7 and section 8a sub-bullets, and Task 7 carries a done check
  that surfaces every passage where a mechanism verb and a truncation
  term co-occur near one of the four affected run_ids. Nothing enforces
  it mechanically — it is a judgment about prose, and the check exists to
  make the judgment unavoidable rather than to make it automatically.
- No task adds a `configs/failure_buckets.yaml`, a taxonomy threshold, a
  CLI taxonomy switch, a sequence-length limit in source or config, a new
  `_ALLOWED_COMPUTATIONS` member, a Verifier resolver branch, a
  `dtype=str` in `_read_csv_artifact`, a document allowlist, a
  `run_config.json` merge, a third `results/` artifact, an entry in
  `requirements.txt`, a p-value, a confidence interval, or any post-hoc
  test. No task touches hybrid retrieval or score fusion, cross-encoder
  or LLM reranking, a fourth retriever, query expansion or rewriting, an
  approximate nearest neighbour index, a chat or serving interface, or
  fine-tuning of any model — all out of scope per `requirements.md`'s
  introduction and `.kiro/steering/scope-guard.md`.
- All done checks are Git Bash / POSIX shell (`grep`, `ls`, `test -d`,
  `find`, `cmp`, `diff`, `wc`, `mktemp -d`, `rm -rf`, `awk`, `test`,
  `echo $?`) or Python one-liners — no PowerShell cmdlet, and exit status
  is read with `$?`, never with PowerShell's own exit-status variable.
  Every one verifies the task's claim mechanically (a row count, a
  column header, a cell format, a uniqueness constraint, an arithmetic
  identity, a directory listing, byte identity, an exit status), except
  the three explicitly flagged as human-review steps: the
  covariate-mechanism proximity grep and the prohibited-scope grep in
  Task 7, and the ledger-completeness read in Task 8.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2"] },
    { "id": 1, "tasks": ["3"] },
    { "id": 2, "tasks": ["4"] },
    { "id": 3, "tasks": ["5"] },
    { "id": 4, "tasks": ["6"] },
    { "id": 5, "tasks": ["7"] },
    { "id": 6, "tasks": ["8", "9"] },
    { "id": 7, "tasks": ["10"] }
  ]
}
```
