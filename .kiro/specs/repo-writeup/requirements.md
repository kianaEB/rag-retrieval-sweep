# Requirements Document

## Introduction

This spec covers the write-up of the already-completed `rag-retrieval-sweep`
sweep: producing exactly two human-readable documents at the
**repository root** — `README.md` and `SPEC.md` — that report the
finding already sitting in the committed artifacts
(`results/sweep.csv`, `results/per_query.csv`,
`results/significance.csv`, `results/run_config.json`). No sweep code,
no metrics code, and no behavioral change to any retriever are
introduced by this spec. One narrow exception: the document-formatting
line inside `DenseRetriever.build_index`
(`src/retrievers/dense_retriever.py`) — currently
`f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"`
— is extracted into a module-level function,
`format_document_text(doc: Dict[str, str]) -> str`, in that same file,
and `build_index` is changed to call it. `src/token_length_analysis.py`
(Requirement 11) then imports and calls that same function directly,
rather than duplicating its logic. This is a pure extract-and-import
refactor: `build_index`'s behavior, output, and the model's retrieval
ranking are unchanged; only the formatting line's location moves from
inline to a named, importable function. This is in scope because it is
what makes Requirement 11.1's "exactly as the dense retriever encodes
it" true by construction — a single shared function used by both call
sites — rather than a claim resting on two independently-maintained
copies of the same string-formatting logic staying in sync by
discipline alone. No other line, method, or behavior in
`src/retrievers/dense_retriever.py`, or in any other retriever,
changes. Every number that appears in either document is read from a
committed artifact that already exists (or, for the one new
measurement this spec introduces — the token-length truncation check —
from a new artifact this spec produces alongside the two documents).

`SPEC.md` (capitalised, at the repository root) is a plain-language
design-and-threats-to-validity document for human readers, named in
`.kiro/steering/structure.md`'s layout and definition-of-done section
alongside `README.md` and `ANALYSIS.md`. It has no relationship to the
`.kiro/specs/` folder mechanism that this requirements document itself
lives under; the two must not be confused with each other anywhere in
this document.

This is the session-2 write-up work named in `docs/PROJECT_BRIEF.md`
("`README.md` with the results table and the honest headline;
`SPEC.md` with design and threats to validity"). The sweep and the
significance analysis are both already complete: `results/sweep.csv`
has 8 rows, `results/per_query.csv` and `results/significance.csv`
exist, and `results/run_config.json` carries both the sweep's and the
significance analysis's applied configuration. This spec reports on
those artifacts; it does not regenerate them.

Out of scope for this spec (no requirements and no tasks appear below
for any of these, and none SHALL be produced or begun while this spec
is in progress): `ANALYSIS.md`, a GitHub Actions CI workflow, the
third retriever (`BAAI/bge-small-en-v1.5`), the fixed-window and
sentence-window chunking strategies, failure bucketing, and data-layer
tests. Each belongs to a different or later spec per
`.kiro/steering/scope-guard.md`.

## Glossary

- **Repo_Writeup**: This feature as a whole — the activity of producing
  the Readme_Document and the Spec_Document, plus the one new
  Token_Length_Analysis this spec introduces.
- **Readme_Document**: The `README.md` file at the repository root.
- **Spec_Document**: The `SPEC.md` file at the repository root — a
  plain-markdown design-and-threats-to-validity document for human
  readers, unrelated to the `.kiro/specs/` folder mechanism.
- **Sweep_Report**: The `results/sweep.csv` file produced by the
  session-1 Sweep_Runner: one row per retriever x evaluation-cutoff
  combination, unchanged by this spec.
- **Per_Query_Report**: The `results/per_query.csv` file produced by
  the significance-testing spec's extension to the Sweep_Runner: one
  row per (run, query) pair.
- **Significance_Report**: The `results/significance.csv` file
  produced by the Significance_Analyzer: one row per (comparison,
  metric) pair, including the paired-bootstrap mean difference,
  confidence interval, raw and Holm-Bonferroni-adjusted p-values, and
  verdict.
- **Run_Config_Record**: The `results/run_config.json` file, carrying
  the sweep seed, the resolved sweep configuration, the corpus load
  report, the installed dependency versions, and the significance
  analysis's own seed and resample/permutation counts.
- **Reference_Run**: The BM25 run (`run_id == "bm25__whole_document"`),
  the reference row every other run is reported against, per
  `.kiro/steering/evaluation-integrity.md`.
- **Primary_Metric**: nDCG@10, designated the single primary metric
  before any sweep result existed, per
  `.kiro/steering/evaluation-integrity.md`.
- **Comparison_Family**: The set of nDCG@10 comparisons of every
  non-BM25 run against the Reference_Run, over which the
  Holm-Bonferroni adjustment is applied.
- **Token_Length_Analysis**: The measurement, introduced by this spec,
  that tokenizes every document in the loaded SciFact corpus with the
  `all-MiniLM-L6-v2` tokenizer and computes what fraction of documents
  exceed that model's 256-token maximum sequence length.
- **Token_Length_Report**: The committed artifact file, under
  `results/`, that persists the Token_Length_Analysis's document
  count, exceeding-document count, and resulting fraction.
- **Numeric_Claim**: Any number appearing in the Readme_Document or the
  Spec_Document that states a measured, computed, or configured
  quantity belonging to this repository's sweep, significance
  analysis, or Token_Length_Analysis — including a metric value, a
  count, a timing value, a percentage, a ratio, or a recorded
  configuration value such as a seed or a resample count.
- **Verification_Pass**: The review activity, required by this spec,
  that checks every Numeric_Claim in the Readme_Document and the
  Spec_Document against the specific committed artifact it was read
  from.

## Requirements

### Requirement 1: Exactly Two Deliverables, Nothing Else

**User Story:** As a maintainer, I want this feature to produce exactly
the two root-level files `README.md` and `SPEC.md` and nothing else, so
that the deliverable stays scoped to documenting the completed sweep
rather than growing into session-2 work that belongs to a different
spec.

#### Acceptance Criteria

1. THE Repo_Writeup SHALL produce exactly two documentation
   deliverables: the Readme_Document at the repository root and the
   Spec_Document at the repository root.
2. THE Repo_Writeup SHALL NOT create any root-level file other than the
   Readme_Document (`README.md`) and the Spec_Document (`SPEC.md`).
3. THE Repo_Writeup SHALL NOT create `ANALYSIS.md`, a GitHub Actions
   workflow file under `.github/workflows/`, a third retriever entry in
   `configs/sweep.yaml`'s retriever declarations, an additional chunking
   strategy entry in `configs/sweep.yaml`'s chunking declarations, a
   failure-bucketing column among `results/sweep.csv`'s column headers,
   or a data-layer test file under `tests/`, because each is explicitly
   out of scope for this spec.
4. THE Repo_Writeup SHALL produce the Token_Length_Report as a
   committed artifact file under `results/`, and that file SHALL NOT
   count toward the two documentation deliverables specified in
   Criterion 1, because it is a data artifact consumed by the
   Spec_Document rather than a document written for direct human
   narrative reading.

### Requirement 2: README Headline — The Primary-Metric Comparison

**User Story:** As a reader of the repository, I want the README's
opening paragraph to state the pre-declared primary-metric comparison
against BM25 and whether it is distinguishable from noise, so that I
get the honest headline before reading anything else.

#### Acceptance Criteria

1. THE Readme_Document SHALL state, within its first paragraph, the
   nDCG@10 mean difference between the `all-MiniLM-L6-v2` run and the
   Reference_Run, computed as the `all-MiniLM-L6-v2` run's nDCG@10
   value minus the Reference_Run's nDCG@10 value, read from the
   `mean_diff` value of the nDCG@10 row of the Significance_Report.
2. THE Readme_Document SHALL state, within its first paragraph, the 95%
   confidence interval lower and upper bounds for that nDCG@10 mean
   difference, read from the `ci_lower` and `ci_upper` values of the
   same row of the Significance_Report.
3. THE Readme_Document SHALL state, within its first paragraph, the
   Holm-Bonferroni adjusted p-value for that nDCG@10 comparison, read
   from the `p_value_adjusted` value of the same row of the
   Significance_Report.
4. IF the `verdict` value of the nDCG@10 row of the Significance_Report
   is `indistinguishable`, THEN THE Readme_Document SHALL describe the
   nDCG@10 comparison as indistinguishable from noise in its first
   paragraph, and SHALL NOT describe the comparison as a win for the
   Reference_Run or for the `all-MiniLM-L6-v2` run.
5. IF the `verdict` value of the nDCG@10 row of the Significance_Report
   is `significant`, THEN THE Readme_Document SHALL state, within its
   first paragraph, whether the nDCG@10 comparison favors the
   Reference_Run or favors the `all-MiniLM-L6-v2` run, with the favored
   run determined by the sign of the `mean_diff` value required by
   Criterion 1: a positive `mean_diff` SHALL be described as favoring
   the `all-MiniLM-L6-v2` run, and a negative `mean_diff` SHALL be
   described as favoring the Reference_Run.
6. THE Readme_Document SHALL state the nDCG@10 comparison in its first
   paragraph regardless of whether the comparison favors BM25, favors
   the dense retriever, or is indistinguishable, so that the headline
   is reported whichever way the pre-declared Primary_Metric falls.

### Requirement 3: README Engineering Cost Finding

**User Story:** As a reader deciding whether dense retrieval is worth
its operational cost, I want the README to state the index-time cost of
the dense retriever relative to BM25 in plain terms, so that I can
weigh that cost against the (indistinguishable) accuracy comparison.

#### Acceptance Criteria

1. THE Readme_Document SHALL state the Reference_Run's `index_time`
   value and the `all-MiniLM-L6-v2` run's `index_time` value, both read
   from the Sweep_Report.
2. THE Readme_Document SHALL state the ratio of the `all-MiniLM-L6-v2`
   run's `index_time` to the Reference_Run's `index_time`, computed
   from the two values required by Criterion 1, rather than from a
   ratio typed independently of the Sweep_Report.
3. THE Readme_Document SHALL state, within the same paragraph as, or in
   the sentence immediately following, the index-time comparison
   required by Criteria 1 and 2, the nDCG@10 verdict for the comparison
   between the `all-MiniLM-L6-v2` run and the Reference_Run, read from
   the `verdict` value of the nDCG@10 row of the Significance_Report
   per Requirement 2, so that the engineering cost is stated together
   with the corresponding accuracy result — whichever way that verdict
   falls — rather than in isolation.
4. THE Readme_Document SHALL NOT state an index-time value, an
   index-time ratio, or a query-latency value for either run that
   differs from the value derivable from the Sweep_Report at the
   numeric precision the Readme_Document reports.

### Requirement 4: README Results Table — BM25 As Reference Row

**User Story:** As a reader comparing the two retrievers, I want a
results table with BM25 as the reference row and every dense value
reported as a delta against it, so that I never see a dense number
presented as if it stood on its own.

#### Acceptance Criteria

1. THE Readme_Document SHALL include a results table containing
   exactly 6 rows: one row for each of the six metrics recall@1,
   recall@5, recall@10, recall@20, nDCG@10, and MRR@10, with no row
   representing more than one metric and no metric represented by more
   than one row.
2. THE Readme_Document SHALL report, on each metric's row, that
   metric's name and the Reference_Run's absolute value for that
   metric, read from the Sweep_Report.
3. THE Readme_Document SHALL report, on each metric's row, the
   `all-MiniLM-L6-v2` run's delta against the Reference_Run for that
   same metric, read directly from the `mean_diff` value of that
   metric's row in the Significance_Report, matching the sign
   convention of that `mean_diff` column, rather than computed by
   subtracting Sweep_Report values.
4. THE Readme_Document SHALL include, on the nDCG@10 row only, the
   Holm-Bonferroni adjusted p-value and the verdict for the nDCG@10
   comparison, read from the `p_value_adjusted` and `verdict` values of
   the nDCG@10 row of the Significance_Report.
5. THE Readme_Document SHALL report the literal sentinel "n/a" in the
   adjusted-p-value and verdict positions of the recall@1, recall@5,
   recall@10, recall@20, and MRR@10 rows, matching the `p_value_adjusted`
   and `verdict` values recorded for those non-primary metric rows in
   the Significance_Report, rather than a computed or estimated value.
6. THE Readme_Document's results table SHALL include a Reference_Run
   absolute value and an `all-MiniLM-L6-v2` delta value for every one
   of the six metrics (recall@1, recall@5, recall@10, recall@20,
   nDCG@10, and MRR@10), and SHALL NOT omit, blank, or leave uncomputed
   any such value because it is unfavorable to either retriever.

### Requirement 5: README Corpus Statistics — Read From The Loader's Own Output

**User Story:** As a reader verifying the study's scale, I want the
corpus, query, and judged-pair counts drawn from the loader's own
recorded output, so that I never read a number typed from memory.

#### Acceptance Criteria

1. THE Readme_Document SHALL state three corpus statistics, each read
   directly from the `corpus_load_report` object of the
   Run_Config_Record: the number of corpus documents, equal to that
   object's `num_documents` field; the number of test queries, equal
   to that object's `num_queries` field; and the number of judged
   query-document pairs, equal to that object's `num_qrel_pairs`
   field.
2. THE Readme_Document SHALL NOT state a corpus document count, a
   query count, or a judged-pair count whose numeric value differs
   from the corresponding `num_documents`, `num_queries`, or
   `num_qrel_pairs` field value in the `corpus_load_report` object of
   the Run_Config_Record; a formatting difference alone (for example,
   a thousands separator) SHALL NOT count as a difference so long as
   the underlying integer value is identical.
3. IF the Run_Config_Record does not exist, cannot be parsed, or can
   be parsed but its `corpus_load_report` object is absent, THEN THE
   Repo_Writeup SHALL halt before stating any corpus statistic in the
   Readme_Document, rather than substituting a value from any source
   other than that object.

### Requirement 6: README Reproduction Instructions

**User Story:** As a reader wanting to reproduce the sweep, I want the
exact commands and a realistic runtime expectation, so that I can run
the study myself without guessing at the entry point or the download
behavior.

#### Acceptance Criteria

1. THE Readme_Document SHALL include the exact command-line invocation
   for the sweep entry point, `python -m src.sweep_runner --config
   configs/sweep.yaml`.
2. THE Readme_Document SHALL include the exact command-line invocation
   for the significance-analysis entry point, `python -m
   src.significance --config configs/significance.yaml`.
3. THE Readme_Document SHALL state, labeled as "combined indexing and
   retrieval time" or equivalently precise wording that does not use
   the words "total", "wall-clock", or "runtime" to describe this
   figure, the sum, in seconds, of the Reference_Run's `index_time`
   and `query_latency` values and the `all-MiniLM-L6-v2` run's
   `index_time` and `query_latency` values, each read from the
   Sweep_Report, rather than an estimated or assumed duration.
4. THE Readme_Document SHALL state that the total wall-clock runtime
   of the sweep is longer than the combined indexing-and-retrieval
   figure required by Criterion 3, because that total wall-clock
   runtime also includes corpus loading, model loading, metric
   computation, and report writing, plus, on a first invocation, the
   one-time download described in Criterion 6.
5. THE Readme_Document SHALL NOT present the combined
   indexing-and-retrieval sum required by Criterion 3 as if it were the
   total wall-clock runtime of the sweep.
6. THE Readme_Document SHALL state that the first invocation of the
   sweep entry point (Criterion 1) triggers a one-time download of the
   BEIR SciFact corpus and the `all-MiniLM-L6-v2` model weights to a
   path under `data/`, that a subsequent invocation of the sweep entry
   point reuses the cached copies without a network call, and that the
   significance-analysis entry point (Criterion 2) makes no network
   call and downloads nothing, on either its first or any later
   invocation.
7. THE Readme_Document SHALL state the installed version of every
   package listed in the `installed_versions` object of the
   Run_Config_Record, and SHALL NOT state, for any of those packages, a
   version string that differs from the value recorded in that object.

### Requirement 7: README Scope Disclaimer — "What This Does Not Claim"

**User Story:** As a reader who might be tempted to over-generalize the
finding, I want an explicit section stating what the study does not
cover, so that the finding is not mistaken for a broader claim than the
sweep actually supports.

#### Acceptance Criteria

1. THE Readme_Document SHALL include a section titled "What this does
   not claim".
2. THE Readme_Document SHALL state, within that section, that the
   finding is scoped to one corpus, BEIR SciFact, and SHALL NOT present
   the finding as generalizing to another domain or corpus.
3. THE Readme_Document SHALL state, within that section, the exact,
   complete set of distinct `chunking_strategy` values present in the
   Sweep_Report, and SHALL NOT present the finding as generalizing to a
   fixed-window or sentence-window chunking strategy or to any chunking
   strategy outside that set.
4. THE Readme_Document SHALL state, within that section, the exact set
   of retrievers compared, read from the distinct `retriever` values
   present in the Sweep_Report, and SHALL NOT present the finding as
   generalizing to a retriever, a model size, or a model family outside
   that set.
5. THE Readme_Document SHALL state, within that section, that the
   finding SHALL NOT be read as a recommendation for or against dense
   retrieval in a production retrieval-augmented-generation system.

### Requirement 8: SPEC Design Summary

**User Story:** As a reader evaluating the study's methodology, I want
`SPEC.md` to summarize the design decisions behind the committed
artifacts, so that I can assess the study's validity without reading
every prior Kiro spec.

#### Acceptance Criteria

1. THE Spec_Document SHALL describe the config-driven sweep grid,
   naming the retrievers, the evaluation cutoffs, and the chunking
   strategy exactly as recorded in the `sweep_config` object of the
   Run_Config_Record, rather than read directly from
   `configs/sweep.yaml`, so that the grid described matches the
   configuration actually applied to the reported run and remains
   traceable to a committed artifact per Requirement 12.
2. THE Spec_Document SHALL describe the "index once, retrieve once,
   slice four ways" property: that each retriever is indexed exactly
   once and queried exactly once at the deepest declared cutoff, and
   that every cutoff's metric value is computed by slicing that single
   ranked list rather than by a separate retrieval run.
3. THE Spec_Document SHALL state that the BEIR SciFact qrels are the
   sole source of relevance ground truth used to compute every reported
   metric, and that no model judgment and no manual override changes a
   qrels-derived relevance determination anywhere in the pipeline.
4. THE Spec_Document SHALL state that nDCG@10 was designated the
   Primary_Metric before any sweep result existed, and that recall@k
   (for k in 1, 5, 10, and 20) and MRR@10 are reported as secondary
   metrics.
5. THE Spec_Document SHALL describe the pre-declared statistical
   scheme — the paired-bootstrap confidence interval, the paired
   permutation two-sided p-value, and the Holm-Bonferroni adjustment
   applied over the Comparison_Family — naming the resample count, the
   permutation count, and the bootstrap seed read from the
   `significance` object of the Run_Config_Record.

### Requirement 9: SPEC nDCG@10 Convention

**User Story:** As a reader auditing the metric definitions, I want the
exact nDCG@10 formula recorded in `SPEC.md`, so that I can verify
computed values independently of this repository's code.

#### Acceptance Criteria

1. THE Spec_Document SHALL record the nDCG@10 convention exactly as
   fixed in advance in the session-1 requirements (Requirement 6.2 of
   `session-1-baseline-sweep`): DCG@10 computed as the sum, over ranks
   i = 1 to 10 of the top 10 ranked documents, of each document's
   graded relevance score (taken directly from the qrels, using 0 for a
   document absent from the qrels or judged with a relevance score of
   zero or less) divided by log2(i + 1); IDCG@10 computed with the
   identical formula, applied instead to the query's qrels-judged
   relevant documents sorted by descending relevance score and
   truncated to the first 10; nDCG@10 for a single query defined as
   DCG@10 divided by IDCG@10, and defined as 0 when IDCG@10 is 0; and
   the nDCG@10 value reported for a run equal to the arithmetic mean of
   the per-query nDCG@10 values over all test queries.
2. THE Spec_Document SHALL state that this nDCG@10 convention was fixed
   before any sweep result existed and has not been altered after
   seeing results.

### Requirement 10: SPEC Threats To Validity

**User Story:** As a reader assessing how much to trust the headline
finding, I want a threats-to-validity section naming every material
limitation the steering docs require, each backed by a number read
from a committed artifact, so that the limitations are concrete rather
than generic disclaimers.

#### Acceptance Criteria

1. THE Spec_Document SHALL include a section titled "Threats to
   validity".
2. THE Spec_Document SHALL name sparse qrels as a threat within that
   section, stating the average number of judged relevant documents
   per query, computed as the `num_qrel_pairs` value divided by the
   `num_queries` value of the `corpus_load_report` object of the
   Run_Config_Record, and stating that a relevant-but-unjudged document
   is scored as a miss regardless of whether a retriever actually
   surfaced it.
3. THE Spec_Document SHALL name BM25's sensitivity to its preprocessing
   and scoring settings as a threat within that section, and SHALL
   state the exact tokenizer, stopword, stemming, case-handling, `k1`,
   and `b` values applied for the reported run, read from the BM25
   retriever's entry within the `sweep_config` object of the
   Run_Config_Record.
4. THE Spec_Document SHALL name single-corpus generalization as a
   threat within that section, stating that every reported number
   describes BEIR SciFact only and may not transfer to another domain.
5. THE Spec_Document SHALL name statistical power as a threat within
   that section, stating the 95% confidence interval half-width for the
   nDCG@10 comparison between the `all-MiniLM-L6-v2` run and the
   Reference_Run, computed as the `ci_upper` value minus the `ci_lower`
   value, divided by two, read from the same nDCG@10 Significance_Report
   row referenced in Requirement 2, and stating that a true difference
   smaller than that half-width could exist without this study's paired
   bootstrap being able to detect it.
6. THE Spec_Document SHALL name the measured BM25 `query_latency` as a
   threat within that section, stating that the measured value is a
   property of the `rank_bm25` package's pure-Python scoring
   implementation rather than an inherent property of lexical
   retrieval, so that a reader does not generalize the measured latency
   to lexical retrieval implemented in a compiled or indexed search
   engine.
7. THE Spec_Document and the Readme_Document SHALL NOT restate an
   "indistinguishable" verdict, anywhere in either document, as
   evidence that no difference exists between the compared runs.

### Requirement 11: Token-Length Truncation Measurement

**User Story:** As a researcher, I want to know whether the dense
retriever's 256-token limit truncates a material fraction of SciFact
documents, so that the headline comparison is not silently confounded
by dropped text under whole-document chunking.

#### Acceptance Criteria

1. THE Repo_Writeup SHALL perform a Token_Length_Analysis that tokenizes
   every document in the loaded SciFact corpus with the
   `all-MiniLM-L6-v2` tokenizer already cached under `data/`, computing
   the token count of each document's title-plus-text input exactly as
   the dense retriever encodes it, including any special tokens the
   tokenizer inserts before or after the document text during that
   encoding, so that the counted length matches the exact input length
   the model receives.
2. THE Token_Length_Analysis SHALL compute the fraction of corpus
   documents whose token count, as computed in Criterion 1, is
   strictly greater than the `all-MiniLM-L6-v2` model's 256-token
   maximum sequence length (that is, a token count of 257 or more).
3. THE Token_Length_Analysis SHALL execute using CPU-only code paths and
   SHALL make no network call, consistent with the tokenizer and model
   weights already cached under `data/` from the completed sweep run.
4. THE Token_Length_Analysis SHALL persist the total document count,
   the count of documents exceeding the 256-token limit, and the
   resulting fraction to the Token_Length_Report, so that the fraction
   stated in the Spec_Document (Criterion 5) is verifiable against a
   committed file rather than against an uncommitted or transient
   computation.
5. THE Spec_Document SHALL state, within the "Threats to validity"
   section required by Requirement 10, the fraction of corpus documents
   exceeding the 256-token limit, read from the Token_Length_Report.
6. IF the fraction computed by the Token_Length_Analysis is strictly
   greater than 1% of corpus documents (that is, more than 0.01 as a
   proportion), THEN THE Spec_Document AND THE Readme_Document SHALL
   EACH state that the Reference_Run scores each document's full text
   while the `all-MiniLM-L6-v2` run scores only the first 256 tokens of
   that same document under whole-document chunking, and that this
   asymmetry is a confound in the headline comparison rather than
   evidence about either retriever's ranking quality; within the
   Readme_Document, this statement SHALL appear in the "Headline
   finding" section required by Requirement 2, immediately following
   the nDCG@10 comparison and verdict stated there, so that a reader of
   the Readme_Document alone learns of the confound without needing to
   also read the Spec_Document.
7. IF the `all-MiniLM-L6-v2` tokenizer cannot be loaded from the cache
   under `data/` without making a network call, THEN THE
   Token_Length_Analysis SHALL raise an error identifying that the
   tokenizer failed to load, SHALL NOT make a network call to retrieve
   it, and SHALL NOT write the Token_Length_Report.

### Requirement 12: Cross-Artifact Numeric Verification

**User Story:** As a maintainer safeguarding the evaluation-integrity
rule that no number appears without a receipt, I want every number in
`README.md` and `SPEC.md` checked against its source artifact before
this feature is considered done.

#### Acceptance Criteria

1. THE Repo_Writeup SHALL treat every Numeric_Claim appearing in the
   Readme_Document or the Spec_Document as traceable to exactly one of:
   the Sweep_Report, the Per_Query_Report, the Significance_Report, the
   Run_Config_Record, or the Token_Length_Report, where a Numeric_Claim
   is traceable to a given artifact if it is either (a) a value copied
   unchanged from that artifact, or (b) a value computed solely from
   one or more values stored within that single artifact, using an
   arithmetic operation already specified elsewhere in this spec (for
   example, a ratio, a delta, a mean, or a percentage), with no value
   from any other artifact contributing to the computation.
2. WHEN the Readme_Document and the Spec_Document are both drafted, THE
   Repo_Writeup SHALL perform a Verification_Pass that checks every
   Numeric_Claim in both files against its source artifact and confirms
   the two values agree once each is rounded to the number of decimal
   places (or significant digits, for a value stated as a percentage)
   shown for that Numeric_Claim in the document, using round-half-up
   rounding, with zero discrepancy at that stated precision.
3. IF a Verification_Pass finds a Numeric_Claim that does not match its
   source artifact, THEN THE Repo_Writeup SHALL correct the
   Numeric_Claim's text in the document, or correct the extraction or
   computation step that produced it, before this feature is
   considered complete; correcting the extraction or computation step
   SHALL NOT include changing any value already stored in the
   Sweep_Report, the Per_Query_Report, the Significance_Report, the
   Run_Config_Record, or the Token_Length_Report, because regenerating
   or editing those artifacts is out of scope for this spec; and in no
   case SHALL the Repo_Writeup leave a known mismatch in either file.
4. THE Repo_Writeup SHALL NOT be considered complete while any
   Numeric_Claim in the Readme_Document or the Spec_Document lacks a
   traceable source artifact under Criterion 1.
