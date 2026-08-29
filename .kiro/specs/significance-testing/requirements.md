# Requirements Document

## Introduction

This spec makes the honest headline of `rag-retrieval-sweep` statable.
Session 1 produced `results/sweep.csv`: BM25 currently leads
`all-MiniLM-L6-v2` by 0.0068 nDCG@10 (0.6519 vs 0.6451) over 300 test
queries. Nothing in the repo can currently say whether that 0.0068
gap is a real result or noise, so no defensible headline can be
written. This spec adds the two artifacts and the one analysis that
close that gap:

1. A per-query artifact (`results/per_query.csv`) written by the same
   sweep run that already writes `results/sweep.csv`, so the raw
   per-query nDCG@10 values behind every mean become a committed
   receipt rather than a transient in-memory value.
2. A separate significance entry point that reads
   `results/per_query.csv` and writes `results/significance.csv`,
   running a paired bootstrap over per-query nDCG@10 differences
   between each non-BM25 run and the BM25 reference run, so the repo
   can state whether the difference is distinguishable from noise.

The central correctness property of this spec is that the significance
analysis is a pure re-analysis of a committed artifact: it never
re-runs retrieval, never re-encodes the corpus, and never re-touches
the BEIR SciFact data or any model, so it can be repeated cheaply and
deterministically. nDCG@10 remains the single, pre-declared primary
metric (per `evaluation-integrity.md`), and BM25 remains the reference
row against which every other run is reported as a delta. A difference
that the pre-declared paired bootstrap cannot distinguish from noise is
reported as "indistinguishable", never as a win for either side.

This is the session-2 analysis work named in `docs/PROJECT_BRIEF.md`
("per-query metric values persisted as an artifact; paired
significance testing of each retriever against the BM25 reference").
The session-1 release gate (per `.kiro/steering/scope-guard.md`) is
already satisfied: `results/sweep.csv` exists with 8 rows,
`configs/sweep.yaml` and `requirements.txt` are present, and both
`tests/test_metrics.py` and `tests/test_orchestration.py` pass.

Out of scope for this spec (no requirements and no tasks appear below
for any of these): the third retriever (`BAAI/bge-small-en-v1.5`),
fixed-window and sentence-window chunking, failure bucketing,
`README.md`, `SPEC.md`, `ANALYSIS.md`, GitHub Actions CI, and
data-layer tests. Also out of scope, per
`.kiro/steering/scope-guard.md`: hybrid retrieval / score fusion,
cross-encoder or LLM reranking, retrievers beyond the two already run,
query expansion, and approximate nearest neighbour indexes. pytest
coverage in this spec covers the bootstrap function only (Requirement
7); the significance entry point's end-to-end behavior and the
per-query writer's real-corpus behavior are not tested in this spec.

## Glossary

- **Per_Query_Report**: The `results/per_query.csv` file, containing
  exactly one row per (Run_Id, query ID) pair, carrying that query's
  per-query metric values for the run that produced it. The
  granular receipt from which every mean in the Sweep_Report is
  derived.
- **Sweep_Report**: The `results/sweep.csv` file produced in session 1,
  containing one row per retriever x cutoff combination. Unchanged by
  this spec except that the same sweep run now also writes the
  Per_Query_Report.
- **Sweep_Runner**: The single session-1 entry point that executes
  corpus loading, indexing, retrieval, metric computation, and
  Sweep_Report writing end-to-end. This spec extends it to also write
  the Per_Query_Report in the same run.
- **Significance_Analyzer**: The separate entry point introduced by
  this spec that reads the Per_Query_Report and writes the
  Significance_Report. It performs no retrieval, no corpus loading, and
  no model loading.
- **Significance_Report**: The `results/significance.csv` file written
  by the Significance_Analyzer, containing one row per comparison of a
  non-BM25 run against the BM25 Reference_Run, per metric.
- **Reference_Run**: The BM25 run, identified by its Run_Id, against
  which every other run is compared. BM25 is the reference row in every
  comparison; dense results are reported as deltas against it, never as
  standalone numbers (per `evaluation-integrity.md`).
- **Run_Id**: The identifier assigned by the Sweep_Runner to each
  single index-build-and-retrieval run, carried unchanged from the
  Sweep_Report into the Per_Query_Report. Two rows share a Run_Id if
  and only if they share the same retriever and the same chunking
  strategy.
- **Qrels**: The human relevance judgments shipped with BEIR SciFact,
  mapping query IDs to judged document IDs and relevance scores. The
  only source of ground truth for scoring; unchanged by this spec.
- **Paired_Bootstrap**: The resampling procedure that estimates, for a
  given metric, the distribution of the mean per-query difference
  between a non-BM25 run and the Reference_Run by resampling query
  IDs with replacement, pairing each run's per-query value for the same
  resampled query. Produces the observed mean difference and its 95%
  confidence interval (2.5th/97.5th percentiles). The two-sided p-value
  is produced separately by the Permutation_Test, not by this
  procedure.
- **Permutation_Test**: The paired permutation procedure that produces
  the two-sided p-value for a comparison. For each of the declared
  permutation iterations, each paired query's two values are
  independently swapped with probability 0.5, the mean difference is
  recomputed, and the p-value is (count + 1) / (permutation_count + 1)
  where count is the number of permuted mean differences whose absolute
  value is greater than or equal to the absolute observed mean
  difference, giving the p-value a floor of 1 / (permutation_count + 1).
  Uses the same Bootstrap_Seed as the Paired_Bootstrap.
- **Bootstrap_Config**: The configuration, declared as data under
  `configs/`, carrying the bootstrap resample count, the permutation
  iteration count, the bootstrap's own random seed (the
  Bootstrap_Seed), and the path to the run configuration record the
  Significance_Analyzer merges its fields into (defaulting to
  `results/run_config.json`). The Bootstrap_Seed is distinct from, and
  kept separate from, the Sweep_Config's sweep seed.
- **Bootstrap_Seed**: The single fixed integer seed applied to the
  Paired_Bootstrap's resampling randomness. Distinct from the sweep
  seed declared in the Sweep_Config; the two are two separate seeds.
- **Comparison_Family**: The fixed, declared-in-advance set of
  comparisons over which multiple-comparison correction is applied:
  every non-BM25 run compared against the Reference_Run on nDCG@10.
- **Holm_Bonferroni_Adjustment**: The multiple-comparison correction
  applied to the raw two-sided p-values of the Comparison_Family,
  producing an adjusted p-value per comparison.
- **Primary_Metric**: nDCG@10, designated the single primary metric
  before any result exists (per `evaluation-integrity.md`). recall@k
  and MRR@10 are secondary metrics.
- **Alpha**: The significance threshold, fixed in advance at 0.05. A
  comparison's headline verdict is determined by comparing its
  Holm_Bonferroni adjusted p-value against Alpha.

## Requirements

### Requirement 1: Per-Query Artifact Written By The Sweep Run

**User Story:** As a researcher, I want the per-query metric values
behind every mean persisted as a committed artifact by the same sweep
run that writes `results/sweep.csv`, so that the significance analysis
has a receipt to read and never has to re-run retrieval.

#### Acceptance Criteria

1. WHEN the Sweep_Runner completes a run that writes the Sweep_Report, THE Sweep_Runner SHALL, in that same run, write the Per_Query_Report to `results/per_query.csv`, without introducing a separate entry point or a second retrieval run.
2. THE Sweep_Runner SHALL write exactly one Per_Query_Report row per (Run_Id, query ID) pair, for every test query loaded and every Run_Id executed in the run.
3. THE Sweep_Runner SHALL include, in every Per_Query_Report row, the columns: `run_id`, `retriever`, `chunking_strategy`, `query_id`, `recall_at_1`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `ndcg_at_10`, `mrr_at_10`, and `num_judged_relevant`.
4. THE Sweep_Runner SHALL write the recall values for the four cutoffs (1, 5, 10, 20) as four separate columns on a single row per (Run_Id, query ID) pair, such that the Per_Query_Report is wide on cutoff and no per-query metric value is duplicated across multiple rows.
5. THE Sweep_Runner SHALL compute the value of `num_judged_relevant` for each row as the count of that query's Qrels-judged relevant documents (documents with a Qrels relevance score greater than zero), derived from the loaded Qrels and not from any retriever output, and SHALL write it as a non-negative integer.
6. THE Sweep_Runner SHALL compute every per-query metric value written to the Per_Query_Report strictly against the Qrels, treating a query-document pair that is absent from the Qrels, or present with a relevance score of zero or less, as not relevant, regardless of any similarity or relevance score produced by a retriever or model, and SHALL write each `recall_at_1`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `ndcg_at_10`, and `mrr_at_10` value in the inclusive range 0.0 to 1.0.
7. THE Sweep_Runner SHALL assign to each Per_Query_Report row the same `run_id`, `retriever`, and `chunking_strategy` values that identify that run in the Sweep_Report, so that a Per_Query_Report row can be joined to its Sweep_Report run without ambiguity.
8. IF writing the Per_Query_Report fails for any reason, THEN THE Sweep_Runner SHALL terminate with a non-zero exit status and SHALL leave `results/per_query.csv` either absent or byte-for-byte in its pre-run state, never partially written.
9. THE Sweep_Runner SHALL write per-query metric values in the Per_Query_Report that reconcile with the corresponding aggregate means in the Sweep_Report, such that the mean of a metric's per-query values over the queries scored for a Run_Id equals that Run_Id's reported aggregate value for the same metric and cutoff in the Sweep_Report to within a floating-point tolerance of 1e-9.
10. WHILE the sweep seed and the input BEIR SciFact data remain unchanged, THE Sweep_Runner SHALL write Per_Query_Report metric columns (`recall_at_1`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `ndcg_at_10`, `mrr_at_10`, `num_judged_relevant`) that are identical across repeated runs on the same machine.

### Requirement 2: Separate Significance Entry Point That Never Re-Runs Retrieval

**User Story:** As a researcher, I want a distinct command that reads
the per-query artifact and produces the significance results, so that I
can repeat the analysis without re-encoding the corpus or reloading any
model.

#### Acceptance Criteria

1. THE Significance_Analyzer SHALL provide a single command-line entry point, distinct from the Sweep_Runner entry point, that reads `results/per_query.csv` and writes the Significance_Report to `results/significance.csv`.
2. THE Significance_Analyzer SHALL NOT perform corpus loading, index building, retrieval, query encoding, or model loading, and SHALL NOT make any network call, so that the analysis re-runs without re-encoding the corpus.
3. THE Significance_Analyzer SHALL derive every per-query value it uses from the Per_Query_Report it reads, and SHALL NOT recompute any per-query metric from the corpus, the Qrels, or any retriever output.
4. IF `results/per_query.csv` is missing, cannot be parsed, or lacks any column required by Requirement 1.3, THEN THE Significance_Analyzer SHALL halt before writing the Significance_Report, SHALL produce an error message on the standard error stream identifying which file, parse failure, or column is missing or invalid, SHALL terminate with a non-zero exit status, and SHALL NOT leave a partial or corrupted file at `results/significance.csv`.
5. IF the Per_Query_Report contains no run identified as the BM25 Reference_Run, THEN THE Significance_Analyzer SHALL halt before writing the Significance_Report, SHALL produce an error message on the standard error stream stating that the Reference_Run is absent, SHALL terminate with a non-zero exit status, and SHALL NOT leave a partial or corrupted file at `results/significance.csv`, because every comparison in this spec is defined relative to the Reference_Run.
6. WHEN the Significance_Analyzer completes successfully, THE Significance_Analyzer SHALL leave the Significance_Report present at `results/significance.csv` and SHALL terminate with a zero exit status.
7. IF writing the Significance_Report fails for any reason, THEN THE Significance_Analyzer SHALL terminate with a non-zero exit status and SHALL NOT leave a partial or corrupted file at `results/significance.csv`.

### Requirement 3: Paired Bootstrap Over Per-Query nDCG@10 Differences

**User Story:** As a researcher, I want a paired bootstrap over
per-query nDCG@10 differences between each non-BM25 run and the BM25
reference run, so that I can report the mean difference, its
uncertainty, and a p-value rather than a bare point estimate.

#### Acceptance Criteria

1. THE Significance_Analyzer SHALL compute, for each non-BM25 run present in the Per_Query_Report, a Paired_Bootstrap of that run's per-query nDCG@10 values against the Reference_Run's per-query nDCG@10 values, pairing the two runs' values by query ID.
2. THE Significance_Analyzer SHALL compute each Paired_Bootstrap resample by drawing query IDs with replacement from the set of query IDs present for both the non-BM25 run and the Reference_Run, drawing a number of query IDs per resample exactly equal to the count of query IDs in that shared set, and SHALL use the same resampled set of query IDs for both runs within a single resample, so that the comparison remains paired.
3. THE Significance_Analyzer SHALL report, for each comparison, the observed mean per-query nDCG@10 difference (the non-BM25 run minus the Reference_Run) and a 95% confidence interval for that mean difference given as an explicit lower bound at the 2.5th percentile and an explicit upper bound at the 97.5th percentile of the resampled mean differences produced by the Paired_Bootstrap.
4. THE Significance_Analyzer SHALL compute the two-sided p-value for each comparison by a paired permutation test rather than from the Paired_Bootstrap distribution: for each of the declared number of permutation iterations, THE Significance_Analyzer SHALL, independently for each paired query, swap that query's two values with probability 0.5 and recompute the mean difference over all paired queries; and SHALL report the p-value as (count + 1) / (permutation_count + 1), where count is the number of permuted mean differences whose absolute value is greater than or equal to the absolute value of the observed mean difference and permutation_count is the declared number of permutation iterations. This add-one correction gives the p-value a floor of 1 / (permutation_count + 1) so it is never exactly zero: a finite permutation sample cannot justify an exact zero, and the observed sign assignment is itself a valid draw under the null, counted in both the numerator and the denominator.
5. THE Significance_Analyzer SHALL read the bootstrap resample count, the permutation iteration count, and the Bootstrap_Seed from the Bootstrap_Config, and SHALL apply the Bootstrap_Seed to both the bootstrap resampling randomness and the permutation-test randomness before drawing any resample or permutation.
6. THE Significance_Analyzer SHALL report the mean difference as the non-BM25 run's value relative to the Reference_Run (a delta against BM25), and SHALL NOT report a non-BM25 run's nDCG@10 as a standalone number without the comparison to BM25.
7. WHILE the Bootstrap_Seed, the resample count, the permutation iteration count, and the input Per_Query_Report remain unchanged, THE Significance_Analyzer SHALL produce, for each comparison, a mean difference, confidence interval, and p-value that are exactly equal across repeated runs on the same machine, at the numeric precision written to the Significance_Report.
8. IF a comparison has zero query IDs present for both the non-BM25 run and the Reference_Run, THEN THE Significance_Analyzer SHALL record, in place of that comparison's mean difference, both confidence interval bounds, and p-value, a missing-value marker that is distinguishable from every numeric value the report can otherwise contain, and SHALL retain the comparison's row rather than omitting it.

### Requirement 4: Two Distinct, Declared, Recorded Seeds And Resample Count

**User Story:** As a researcher, I want the bootstrap resample count
and the bootstrap's own seed declared as data and recorded in the run
configuration record, kept separate from the sweep seed, so that the
analysis is reproducible and the two randomness sources are never
conflated.

#### Acceptance Criteria

1. THE Bootstrap_Config SHALL declare, as data under `configs/`, the bootstrap resample count as a single explicit integer value, the permutation iteration count as a single explicit integer value, and the Bootstrap_Seed as a single explicit integer value.
2. THE Bootstrap_Seed and the sweep seed SHALL be recorded as two separate named fields, such that neither seed is derived from the other and each can be edited independently, so that the two randomness sources are never conflated.
3. WHEN the Significance_Analyzer runs, THE Significance_Analyzer SHALL merge the bootstrap resample count, the permutation iteration count, and the Bootstrap_Seed applied for the run into the run configuration record declared in the Bootstrap_Config (defaulting to `results/run_config.json`) as fields separate from the sweep seed, and SHALL preserve every key already written there by the Sweep_Runner — including `seed`, `sweep_config`, `corpus_load_report`, and `installed_versions` — removing or altering none of them.
4. THE Significance_Analyzer SHALL derive the recorded bootstrap resample count and Bootstrap_Seed from the values it actually applied during the run, rather than from a literal written independently of the applied values.
5. IF the Bootstrap_Config is missing, cannot be parsed, omits the bootstrap resample count, the permutation iteration count, or the Bootstrap_Seed, or declares any of those three values as a non-integer, THEN THE Significance_Analyzer SHALL halt before writing the Significance_Report or altering `results/run_config.json`, SHALL produce an error message identifying which declaration is missing or invalid, and SHALL terminate with a non-zero exit status.
6. IF the run configuration record declared in the Bootstrap_Config (defaulting to `results/run_config.json`) is absent or cannot be parsed, THEN THE Significance_Analyzer SHALL halt before writing the Significance_Report, SHALL produce an error message on the standard error stream stating that the sweep's run configuration record is missing or unreadable, and SHALL terminate with a non-zero exit status, rather than creating a fresh record that would lack the Sweep_Runner's own record.

7. THE Bootstrap_Config SHALL declare the path to the run configuration record that the Significance_Analyzer merges its fields into, defaulting to `results/run_config.json` when not overridden, so that the merge target is configurable (for example, redirectable to a temporary directory) rather than hard-coded.

### Requirement 5: Multiple-Comparison Handling Declared In Advance

**User Story:** As a researcher, I want the multiple-comparison
correction scheme fixed before results are seen and unchanged as the
grid grows, so that adding runs later cannot be mistaken for adjusting
the analysis after seeing the numbers.

#### Acceptance Criteria

1. THE Significance_Analyzer SHALL define the Comparison_Family as every non-BM25 run compared against the Reference_Run on nDCG@10, and SHALL apply the Holm_Bonferroni_Adjustment over exactly that family.
2. THE Significance_Analyzer SHALL report, for each comparison in the Comparison_Family, both the raw two-sided p-value and the Holm_Bonferroni adjusted p-value.
3. THE Significance_Analyzer SHALL apply the identical Comparison_Family definition, the identical raw-p-value ordering rule, and the identical Holm_Bonferroni step-down multipliers for every Comparison_Family size from one comparison up to nine comparisons inclusive, without changing the scheme based on the number of non-BM25 runs present in the Per_Query_Report.
4. THE Significance_Analyzer SHALL compute the Holm_Bonferroni adjusted p-values as a function only of the raw two-sided p-values in the Comparison_Family and the family size, by: (a) sorting the family's raw p-values in ascending order; (b) multiplying the raw p-value at ascending rank index i (starting at 0) by (family size minus i); (c) enforcing that the resulting adjusted values are non-decreasing across the ascending order, setting each to the running maximum of itself and all lower-ranked adjusted values; and (d) clamping every adjusted value to the range 0.0 to 1.0 inclusive, so that the adjustment is fully determined by the declared scheme and the observed raw p-values.
5. IF two or more comparisons in the Comparison_Family share an equal raw two-sided p-value, THEN THE Significance_Analyzer SHALL apply the ordering and step-down multipliers of Criterion 4 such that those tied comparisons receive equal adjusted p-values, so that the adjusted result does not depend on the input order of tied comparisons.
6. WHERE the Comparison_Family contains exactly one comparison, THE Significance_Analyzer SHALL report the Holm_Bonferroni adjusted p-value equal to the raw two-sided p-value for that comparison, clamped to the range 0.0 to 1.0 inclusive, as the single-comparison step-down multiplier is one and the Holm_Bonferroni_Adjustment reduces to the identity.

### Requirement 6: Primary And Secondary Metric Reporting

**User Story:** As a researcher, I want the same bootstrap reported for
the secondary metrics for reference while the headline is determined by
nDCG@10 alone, so that the primary metric is not chosen or re-chosen
after seeing results.

#### Acceptance Criteria

1. THE Significance_Analyzer SHALL designate nDCG@10 as the Primary_Metric and recall@k (for k in 1, 5, 10, 20) and MRR@10 as secondary metrics, and SHALL record this designation in the Significance_Report before any comparison result is written.
2. THE Significance_Analyzer SHALL compute and report, for each secondary metric, the same outputs it reports for the Primary_Metric — the Paired_Bootstrap mean difference and two-sided 95% confidence interval for the mean difference, and the Permutation_Test two-sided p-value against the Reference_Run — using the identical resample count, permutation iteration count, and Bootstrap_Seed recorded in the Bootstrap_Config that it uses for the Primary_Metric.
3. THE Significance_Analyzer SHALL apply the Holm_Bonferroni_Adjustment over the Comparison_Family, which is defined on nDCG@10 alone (Requirement 5.1), SHALL NOT include any secondary-metric comparison in that Comparison_Family, and SHALL determine the headline result of the analysis from the nDCG@10 comparison alone.
4. THE Significance_Analyzer SHALL determine each comparison's headline verdict from its Holm_Bonferroni adjusted p-value compared against Alpha (0.05, declared in advance), not from whether the 95% confidence interval includes zero; the confidence interval is reported to convey the magnitude and direction of the uncertainty. IF a comparison's Holm_Bonferroni adjusted p-value is greater than or equal to Alpha, THEN THE Significance_Analyzer SHALL mark that comparison's nDCG@10 result as "indistinguishable" in the Significance_Report, and SHALL NOT report it as a win for either the non-BM25 run or the Reference_Run.
5. THE Significance_Analyzer SHALL retain, in the Significance_Report, one row for every comparison in the Comparison_Family and one row for every secondary-metric comparison it computed, and SHALL NOT drop, exclude, or filter any computed comparison from the report on the basis of its result value.
6. THE Significance_Analyzer SHALL write every numeric value in the Significance_Report such that the value is recomputable, to within a floating-point tolerance of 1e-9, from the committed Per_Query_Report together with the recorded Bootstrap_Config (resample count, permutation iteration count, and Bootstrap_Seed), so that no number in the report lacks a committed-artifact receipt.

### Requirement 7: Test Coverage Scope Limited To The Bootstrap, Permutation, And Holm-Bonferroni Functions

**User Story:** As a maintainer, I want pytest coverage over the paired
bootstrap, the permutation test, and the Holm-Bonferroni adjustment
using hand-built synthetic inputs, so that the resampling logic and the
multiple-comparison correction that determines the headline verdict are
verified without any network access or dependency on the significance
entry point or the sweep runner.

#### Acceptance Criteria

1. WHEN the Paired_Bootstrap is given two identical per-query value vectors of equal, non-zero length (a run compared against itself), THE test suite SHALL assert that the reported mean difference equals exactly zero.
2. WHEN the Paired_Bootstrap is given a synthetic pair of per-query value vectors of equal, non-zero length whose per-query values differ by the same non-zero constant offset for every query, THE test suite SHALL assert that the reported 95% confidence interval excludes zero and that the interval lies entirely on the side of zero matching the sign of that constant offset.
3. WHEN the Paired analysis is given two identical per-query value vectors of equal, non-zero length (a run compared against itself), THE test suite SHALL assert that the reported two-sided p-value is approximately 1.0, within a numeric tolerance of 1e-6.
4. WHEN the Paired analysis is given a synthetic pair of per-query value vectors of equal, non-zero length whose per-query values differ by the same non-zero constant offset for every query, THE test suite SHALL assert that the reported two-sided p-value is below 0.01.
5. WHEN the Paired_Bootstrap is run twice with the same Bootstrap_Seed, the same resample count, and the same inputs, THE test suite SHALL assert that the two runs produce confidence interval bounds that are exactly equal.
6. WHEN holm_bonferroni is given a single-element family of raw p-values, THE test suite SHALL assert that the returned adjusted p-value equals that raw p-value (the single-comparison identity).
7. WHEN holm_bonferroni is given the raw p-value family [0.04, 0.01, 0.03] in that input order, THE test suite SHALL assert that the returned adjusted p-values are [0.06, 0.03, 0.06] in that same input order (within a numeric tolerance of 1e-9), so that the adjusted values are verified to be mapped back to each comparison's original input position rather than left in sorted order.
8. WHEN holm_bonferroni is given a family in which two or more comparisons share an equal raw p-value, THE test suite SHALL assert that those tied comparisons receive equal adjusted p-values.
9. THE test suite for this requirement SHALL cover only the paired_bootstrap, permutation_test, and holm_bonferroni functions, SHALL NOT import or invoke the Significance_Analyzer entry point, the Sweep_Runner, or any Corpus_Loader or retriever code, and SHALL defer significance-entry-point end-to-end tests and data-layer tests to a later spec.
10. THE test suite for this requirement SHALL execute without making any network call and without loading any dataset or model.
11. THE test suite SHALL assert exact equality only for the properties that are exact (the zero mean difference of Criterion 1 and the reproduced confidence interval bounds of Criterion 5), and SHALL compare all other float results — including the two p-value assertions of Criteria 3 and 4, which are tolerance-based or threshold-based rather than exact — using a numeric tolerance of 1e-6.
