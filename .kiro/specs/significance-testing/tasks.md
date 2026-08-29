# Implementation Plan

## Overview

Ordering rule: this spec is entirely network-free and model-free — it
adds no new runtime dependency (`numpy`/`pandas`/`PyYAML`/`pytest` are
already pinned) and never touches BEIR or Hugging Face, so the ordering
is driven by dependency, not by a network boundary. The three pure
functions `paired_bootstrap`, `permutation_test`, and `holm_bonferroni`
(Task 2) come FIRST, and their pytest suite (Requirement 7 — the ONLY
automated tests in this spec, Task 3) must be green before the
`Significance_Analyzer` entry point that consumes them is wired (Task
7). Sequence: extend the error hierarchy; write the three pure functions;
test all three; write the config schema + `configs/significance.yaml`;
write the sweep-side per-query writer; extend the `Sweep_Runner` to emit
`per_query.csv` in the same run; wire the analyzer entry point over all
of the above; then verify end-to-end reproducibility manually. The
entry-point end-to-end, per-query-writer real-corpus, and
`run_config.json` merge paths are verified structurally / by manual
rerun (Tasks 7 and 8), consistent with the design's "what is explicitly
not tested in this spec" list. All done checks are Git Bash / POSIX
shell or Python one-liners.

## Tasks

- [ ] 1. Extend `src/errors.py` with the significance exception types
  - Add, alongside session 1's existing types: `PerQueryReportError`
    (top-level `Exception`), `BootstrapConfigError(ConfigError)`,
    `SignificanceInputError`, `MissingReferenceRunError`,
    `RunConfigMergeError`, and `SignificanceWriteError` — matching the
    docstrings and the `ConfigError`-subclass-for-config-failures
    dividing line in the design.
  - Done check:
    `python -c "from src.errors import PerQueryReportError, BootstrapConfigError, SignificanceInputError, MissingReferenceRunError, RunConfigMergeError, SignificanceWriteError, ConfigError; assert issubclass(BootstrapConfigError, ConfigError); [cls('x') for cls in (PerQueryReportError, BootstrapConfigError, SignificanceInputError, MissingReferenceRunError, RunConfigMergeError, SignificanceWriteError)]; print('ok')"`
    prints `ok` with exit code 0.
  - _Requirements: 1.8, 2.4, 2.5, 2.7, 4.5, 4.6_

- [ ] 2. Write the three pure functions `paired_bootstrap`, `permutation_test`, and `holm_bonferroni` in `src/significance.py`
  - Implement `paired_bootstrap(a, b, resample_count, generator)`:
    compute `d = a - b` once; draw one
    `generator.integers(0, n, size=(resample_count, n))` index matrix;
    `resampled = d[idx].mean(axis=1)`; return
    `(float(d.mean()), *np.percentile(resampled, [2.5, 97.5]))` as
    `(observed_mean_diff, ci_lower, ci_upper)`. No Python-level
    per-resample loop.
  - Implement `permutation_test(a, b, permutation_count, generator)`:
    `d = a - b`, `observed = float(d.mean())`; draw one
    `signs = np.where(generator.random((permutation_count, n)) < 0.5, -1.0, 1.0)`;
    `permuted = (signs * d).mean(axis=1)`;
    `count = int((np.abs(permuted) >= abs(observed)).sum())`;
    return `(count + 1) / (permutation_count + 1)`. No Python-level
    per-permutation loop.
  - Implement `holm_bonferroni(raw_p_values)`: family size
    `m = len(raw_p_values)`; sort ascending; multiply the value at
    ascending rank index `i` (from 0) by `(m - i)`; enforce monotonic
    non-decrease across ascending order (running max); clamp each to
    `[0.0, 1.0]`; map back to the family's original input order. Ties
    receive equal adjusted values (order-independent); `m == 1` reduces
    to the raw value clamped (the identity). A pure function of the raw
    p-values and the family size only.
  - All three functions accept an injected `np.random.Generator` (where
    randomness is used), touch no global RNG, read no file, and import
    no retrieval/model code — the module imports only `numpy` (and
    stdlib typing) at this stage.
  - Done check (bootstrap + permutation): a script that builds tiny
    hand-built numpy arrays and asserts, using
    `np.random.default_rng(20240)`: (a) for `a == b` (self-comparison),
    `paired_bootstrap` returns `mean_diff == 0.0` and `permutation_test`
    returns `p == 1.0`; (b) for `a = b + 0.25` (constant positive
    offset), `paired_bootstrap` returns both CI bounds `> 0.0` and
    `permutation_test` returns `p < 0.01`. Prints `ok` and exits 0. No
    network call.
  - Done check (holm_bonferroni quick sanity — the AUTHORITATIVE
    verification is the Task 3 pytest): a script asserting (a)
    `holm_bonferroni([0.03]) == [0.03]` (single-comparison identity);
    (b) the order-preserving worked example
    `holm_bonferroni([0.04, 0.01, 0.03])` equals `[0.06, 0.03, 0.06]`
    (input order differs from sorted order, so it detects a missing
    map-back-to-input-order), compared with `pytest.approx(..., abs=1e-9)`;
    (c) `holm_bonferroni([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]` (three
    tied raw p-values of 0.5: multipliers 3,2,1 give 1.5,1.0,0.5 →
    running-max 1.5,1.5,1.5 → clamp 1.0,1.0,1.0 — exercising both the
    clamp and the tie behavior from reachable inputs). Prints `ok` and
    exits 0.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.3, 5.4, 5.5, 5.6_

- [ ] 3. Write `tests/test_significance.py` (Requirement 7 — the entire automated-test surface)
  - Implement the five bootstrap/permutation assertions from the
    design's Testing Strategy, each constructing its `generator` locally
    via `np.random.default_rng(<fixed seed>)`: self-comparison mean
    difference exactly `0.0` (7.1); self-comparison p ≈ `1.0` within
    `1e-6` (7.3); constant positive offset CI excludes zero with matching
    sign (`lo > 0 and hi > 0`) (7.2); constant offset p `< 0.01` (7.4);
    same seed → identical CI bounds via exact equality of `(lo, hi)`
    across two `default_rng(7)` runs (7.5). Use `pytest.approx(..., abs=1e-6)`
    for the tolerance-based p-value assertion and exact `==` only for the
    exact properties (7.11).
  - Implement the three `holm_bonferroni` assertions: (a)
    single-comparison identity `holm_bonferroni([0.03]) == [0.03]`
    (7.6); (b) the order-preserving worked example
    `holm_bonferroni([0.04, 0.01, 0.03])` equals `[0.06, 0.03, 0.06]`
    within `1e-9` — input order differs from sorted order to catch the
    map-back bug (7.7); (c) the tie case where equal raw p-values
    receive equal adjusted values (7.8).
  - Import ONLY `src.significance` (`paired_bootstrap`,
    `permutation_test`, `holm_bonferroni`), `numpy`, and `pytest` — NOT
    the analyzer `main()`, `src.sweep_runner`, `src.corpus_loader`,
    `src.metrics`, or any retriever module (7.9). Make no network call
    and load no dataset or model (7.10).
  - Done check: `pytest tests/test_significance.py -v` reports all tests
    passed, and
    `! grep -Eq 'beir|sentence_transformers|huggingface|corpus_loader|sweep_runner' tests/test_significance.py && echo ok`
    prints `ok` (the `!` asserts no forbidden-import match, Requirement
    7.9, 7.10). This must be green before Task 7 wires the entry point.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11_

- [ ] 4. Write `configs/significance.yaml` and `src/significance_config.py`
  - Implement the frozen `SignificanceConfig` dataclass
    (`resample_count`, `permutation_count`, `bootstrap_seed`, `alpha`,
    `reference_retriever`, `per_query_path`, `output_path`,
    `run_config_path`) and `load_significance_config(path)` raising
    `BootstrapConfigError` (a `ConfigError` subclass) that names the
    missing / unparsable / non-integer field: `resample_count` /
    `permutation_count` / `bootstrap_seed` must be integers, `alpha`
    must equal the fixed value `0.05`, and `reference_retriever` /
    `per_query_path` / `output_path` / `run_config_path` must be present
    (`run_config_path` defaults to `results/run_config.json` when not
    overridden in the YAML, so the merge target is configurable per
    Requirement 4.7). Never partially applies a config. Imports only
    `PyYAML` and the standard library.
  - Author `configs/significance.yaml` per the design schema:
    `resample_count: 10000`, `permutation_count: 10000`,
    `bootstrap_seed: 20240` (distinct from `configs/sweep.yaml`'s
    `seed: 42`), `alpha: 0.05`, `reference_retriever: bm25`,
    `per_query_path: results/per_query.csv`,
    `output_path: results/significance.csv`,
    `run_config_path: results/run_config.json`.
  - Done check: a script that calls
    `load_significance_config(Path("configs/significance.yaml"))` and
    asserts `resample_count == 10000`, `permutation_count == 10000`,
    `bootstrap_seed == 20240`, `bootstrap_seed != 42`, `alpha == 0.05`,
    `reference_retriever == "bm25"`, and
    `run_config_path == Path("results/run_config.json")`; then asserts a
    copy of the YAML with `alpha: 0.1` raises `BootstrapConfigError`, and
    a copy omitting `bootstrap_seed` raises `BootstrapConfigError`.
    Prints `ok` and exits 0.
  - _Requirements: 4.1, 4.2, 4.5, 4.7, 6.4_

- [ ] 5. Write `src/per_query_report.py` (the Per_Query_Report writer)
  - Implement the frozen `PerQueryReportRow` dataclass with the 11
    columns from Requirement 1.3 in fixed order (`run_id`, `retriever`,
    `chunking_strategy`, `query_id`, `recall_at_1`, `recall_at_5`,
    `recall_at_10`, `recall_at_20`, `ndcg_at_10`, `mrr_at_10`,
    `num_judged_relevant`) and `write_per_query_report(rows, output_path)`
    that writes them to `results/per_query.csv` as CSV via
    `src.report._atomic_write_text` (temp file + `os.replace`, temp
    removed on any failure), raising `PerQueryReportError` on failure so
    the file is left absent or byte-for-byte in its pre-run state, never
    partial.
  - Done check: a script that writes a couple of hand-built
    `PerQueryReportRow`s (one BM25, one dense; distinct `query_id`s) to a
    temp CSV (via `tempfile`), reads it back with `pandas`, and asserts:
    exactly those rows; the columns are exactly the 11 fields in the
    fixed order; every `recall_at_*`/`ndcg_at_10`/`mrr_at_10` cell is a
    float in `[0.0, 1.0]`; `num_judged_relevant` reads back as an
    integer. Prints `ok` and exits 0. No network call.
  - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.8_

- [ ] 6. Extend `src/sweep_runner.py` to emit `results/per_query.csv` in the same run
  - Capture the per-query `recall`/`ndcg`/`mrr` dictionaries `run_sweep`
    already computes on its way to each mean — no new retriever call, no
    recomputation from the corpus — and assemble one `PerQueryReportRow`
    per (run_id, query_id), with the four recall cutoffs as the four
    `recall_at_*` columns on a single wide row and
    `num_judged_relevant = len(judged_relevant_docs(qrels.get(qid, {})))`
    from the loaded qrels. `run_sweep` returns the per-query rows
    alongside the existing sweep rows; `main()` writes them via
    `write_per_query_report` after `write_sweep_report`, and on
    `PerQueryReportError` prints to stderr and returns non-zero, leaving
    no partial `per_query.csv`.
  - Done check: a script driving `run_sweep` with a session-1
    stub-style ≤5-document in-memory `CorpusBundle`, a 4-cutoff
    `SweepConfig` (cutoffs `(1, 5, 10, 20)`), and a trivial hand-written
    factory (no real retriever); asserts the returned per-query row count
    equals `len(retrievers) * len(queries)`, and that every row carries
    all four of `recall_at_1`/`recall_at_5`/`recall_at_10`/`recall_at_20`
    (wide on cutoff) plus `ndcg_at_10`, `mrr_at_10`, and an integer
    `num_judged_relevant`. Prints `ok` and exits 0. No network call, no
    real retriever imported. The real-corpus behavior and the
    Requirement 1.9 reconciliation / 1.10 rerun-identity are verified by
    the end-to-end rerun in Task 8, mirroring session 1's
    manual-verification stance.
  - _Requirements: 1.1, 1.2, 1.5, 1.7, 1.8_

- [ ] 7. Wire the `Significance_Analyzer` `main()` in `src/significance.py`
  - Implement the CLI entry point (`python -m src.significance
    [--config PATH]`, default `configs/significance.yaml`) with the
    design's 9-step orchestration: (1) load config via
    `load_significance_config`, halting on `BootstrapConfigError`; (2)
    read the `run_config_path` declared in the config (default
    `results/run_config.json`) or halt with `RunConfigMergeError` (never
    create a fresh file); (3) read the config's `per_query_path` via
    pandas or halt with `SignificanceInputError` naming the file / parse
    failure / missing Requirement 1.3 column; (4) find the BM25
    `Reference_Run` by `run_id == f"{reference_retriever}__{chunking}"`
    or halt with `MissingReferenceRunError`; (5) build the fixed
    comparison order (non-BM25 `run_id`s sorted ascending), construct the
    single `generator = np.random.default_rng(bootstrap_seed)`, and for
    each comparison (fixed order) × each metric (fixed order:
    `ndcg_at_10`, `recall_at_1`, `recall_at_5`, `recall_at_10`,
    `recall_at_20`, `mrr_at_10`) run `paired_bootstrap` then
    `permutation_test`; a zero-shared-queries comparison records the
    `"NA"` marker and retains its row; (6) `holm_bonferroni` over the
    nDCG@10 family, secondary rows get the `"n/a"` sentinel for
    `p_value_adjusted`/`verdict`; (7) nDCG@10 verdict from
    `p_value_adjusted` vs `alpha`; (8) write the config's `output_path`
    atomically; (9) merge the `"significance"` sub-object into the
    config's `run_config_path` atomically, preserving every existing
    sweep key. Uses the recorded config values actually applied.
  - Done check (structural / manual — no automated test per Requirement
    7, consistent with the design's deferred-testing list): a Git Bash /
    POSIX shell script that NEVER writes into or deletes from the real
    `results/` directory. It creates a temp dir with `TMPDIR="$(mktemp -d)"`;
    fabricates INSIDE that temp dir a tiny `per_query.csv` (a BM25 run
    `bm25__whole_document` and one dense run
    `all-MiniLM-L6-v2__whole_document`, a handful of shared `query_id`s
    with all 11 columns), a minimal `run_config.json` carrying `seed`,
    `sweep_config`, `corpus_load_report`, and `installed_versions`, and a
    `significance.yaml` whose `per_query_path`, `output_path`, and
    `run_config_path` all point at temp-dir paths; runs
    `python -m src.significance --config "$TMPDIR/significance.yaml"` and
    asserts exit 0 via `&& echo ok` (or checking `$?`); asserts the temp
    `significance.csv` exists with the design's columns, one nDCG@10
    (primary) row plus five secondary rows per comparison, and secondary
    rows carry `"n/a"` in `p_value_adjusted` and `verdict`; asserts the
    temp `run_config.json` still contains the original `seed`,
    `sweep_config`, `corpus_load_report`, and `installed_versions` keys
    plus a new `"significance"` object with
    `bootstrap_seed`/`resample_count`/`permutation_count`/`alpha`; then
    `rm -rf "$TMPDIR"`. The real committed `results/` artifacts are never
    created, overwritten, or deleted — this is possible specifically
    because `run_config_path` (and the input/output paths) are
    configurable per Requirement 4.7. Prints `ok` and exits 0.
  - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7, 3.6, 3.7, 3.8, 4.3, 4.4, 4.6, 4.7, 5.1, 5.2, 6.1, 6.2, 6.3, 6.5_

- [ ] 8. Verify end-to-end reproducibility and reconciliation manually
  - Regenerate `results/per_query.csv` from the extended sweep (Task 6),
    either by running `python -m src.sweep_runner --config
    configs/sweep.yaml` or by reusing the artifact from that run. Run
    `python -m src.significance --config configs/significance.yaml`
    twice, to two output paths (via a copy of the config with a
    different `output_path`), and assert the two `significance.csv` files
    are bit-identical (Requirement 3.7). Confirm the nDCG@10 (primary)
    row's `mean_diff` (non-BM25 minus BM25) reconciles in sign and
    magnitude with the delta implied by `results/sweep.csv`'s nDCG@10
    means for the two runs.
  - Done check: a Git Bash comparison of the two `significance.csv` runs
    shows no differences (e.g. `cmp -s run_a.csv run_b.csv && echo identical`,
    or `diff run_a.csv run_b.csv && echo identical`); a script loads
    `results/sweep.csv` and `results/significance.csv`, computes the
    BM25-vs-dense nDCG@10 mean delta from `sweep.csv`, and asserts the
    significance report's nDCG@10 `mean_diff` matches it in sign and to
    within `1e-9`; prints `ok` and exits 0. Delete the rerun artifacts
    afterward with `rm -rf`. This is manual / structural verification
    consistent with the design's deferred-testing stance (Requirement 7
    scopes automated tests to the bootstrap, permutation, and
    Holm-Bonferroni functions only).
  - _Requirements: 1.9, 1.10, 3.7, 6.6_

## Notes

- Task 2 produces all three pure functions (`paired_bootstrap`,
  `permutation_test`, `holm_bonferroni`); Task 3 (the Requirement 7
  pytest suite) tests all three and is the entire automated-test surface
  of this spec — it MUST be green before Task 7 wires the entry point
  that consumes the tested functions.
- The done checks for Tasks 7 and 8 are POSIX shell scripts / manual
  reruns, not pytest tests — Requirement 7 deliberately scopes automated
  tests to `paired_bootstrap`, `permutation_test`, and `holm_bonferroni`
  only; the analyzer entry point, the per-query writer's real-corpus
  behavior, and the `run_config.json` merge are verified structurally,
  consistent with the design's "what is explicitly not tested in this
  spec" list. Task 7's done check runs entirely in a `mktemp -d` temp
  dir and never touches the real `results/` artifacts (enabled by the
  configurable `run_config_path`, Requirement 4.7).
- All done checks are Git Bash / POSIX shell (`$?`, `grep`, `diff` /
  `cmp`, `mktemp -d`, `rm -rf`) or Python one-liners — no PowerShell.
- This spec adds no new runtime dependency: `numpy`, `pandas`, `PyYAML`,
  and `pytest` are already pinned in `requirements.txt` from session 1.
- Every task references specific acceptance criteria for traceability;
  no task is a giant "implement everything" step, and each builds on the
  prior ones.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "4", "5"] },
    { "id": 2, "tasks": ["3", "6"] },
    { "id": 3, "tasks": ["7"] },
    { "id": 4, "tasks": ["8"] }
  ]
}
```
