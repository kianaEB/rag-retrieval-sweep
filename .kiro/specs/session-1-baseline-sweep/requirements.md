# Requirements Document

## Introduction

This spec covers session 1 of `rag-retrieval-sweep`: producing
`results/sweep.csv` end-to-end from a single entry point on a clean
checkout. It loads the BEIR SciFact corpus, queries, and qrels; runs
two retrievers (BM25 and `all-MiniLM-L6-v2`) over whole-document
chunking only; evaluates at cutoffs k = 1, 5, 10, 20; and writes the 8
resulting rows (2 retrievers x 4 cutoffs) with metrics, timing, and a
fixed seed. The central correctness property of this spec is that
top-k is an evaluation cutoff, not a separate retrieval run: each
retriever is indexed once and queried once at Deepest_Cutoff, and that
single ranked list per query is sliced to compute all four cutoffs'
metrics.

Out of scope for this spec: `README.md`, `SPEC.md`, `ANALYSIS.md`,
GitHub Actions CI, failure bucketing, the `BAAI/bge-small-en-v1.5`
retriever, fixed-window and sentence-window chunking, hybrid
retrieval, reranking, query expansion, and ANN indexes. pytest coverage
in this spec is limited to the metric functions; the data-loading
layer and the sweep runner are not tested in this spec.

## Glossary

- **Corpus_Loader**: The component that loads the BEIR SciFact corpus
  documents, test queries, and qrels in the standard BEIR format.
- **Qrels**: The human relevance judgments shipped with BEIR SciFact,
  mapping query IDs to judged document IDs and relevance scores. The
  only source of ground truth for scoring.
- **Sweep_Config**: The YAML file under `configs/` that declares the
  retrievers, the evaluation cutoffs, the chunking strategy, the BM25
  preprocessing settings, and the fixed random seed as data.
- **Sweep_Runner**: The single entry point that executes corpus
  loading, indexing, retrieval, metric computation, and
  `results/sweep.csv` writing end-to-end, driven entirely by the
  Sweep_Config.
- **BM25_Retriever**: The lexical retriever built on `rank_bm25`
  `BM25Okapi`, indexing whole documents and ranking them against a
  tokenized query.
- **Dense_Retriever**: The retriever that encodes whole documents and
  queries with the `all-MiniLM-L6-v2` sentence-transformers model and
  ranks documents by embedding similarity.
- **Whole_Document_Chunking**: The chunking strategy in which each
  corpus document is indexed as a single, unsplit unit.
- **Deepest_Cutoff**: The largest evaluation cutoff declared in the
  Sweep_Config for a given run (k=20 for this spec).
- **Ranked_List**: The ordered list of document IDs returned by a
  retriever for one query, ordered by descending relevance score.
- **Metrics_Calculator**: The set of functions that compute recall@k,
  nDCG@10, and MRR@10 from a Ranked_List and the Qrels.
- **Sweep_Report**: The `results/sweep.csv` file, containing one row
  per retriever x cutoff combination declared in the Sweep_Config.
- **Index_Time**: The wall-clock duration taken by a retriever to
  build its index over the corpus, measured once per retriever.
- **Query_Latency**: The wall-clock duration taken by a retriever to
  complete retrieval for all queries in a single retrieval run,
  measured once per retriever.
- **Run_Id**: An identifier assigned by the Sweep_Runner to each single
  index-build-and-retrieval run. Every Sweep_Report row derived by
  slicing that run's single Deepest_Cutoff Ranked_List shares the same
  Run_Id. Two Sweep_Report rows share a Run_Id if and only if they
  share the same retriever and the same chunking strategy; in this
  spec, since exactly one chunking strategy (Whole_Document_Chunking)
  is declared, Run_Id is equivalent to retriever identity, but is
  defined in terms of retriever and chunking strategy so the concept
  extends unchanged when additional chunking strategies are added in a
  later spec.

## Requirements

### Requirement 1: Corpus, Query, and Qrels Loading with Count Reporting

**User Story:** As a researcher running the sweep, I want the loader
to load the BEIR SciFact corpus, queries, and qrels and report the
exact counts it loaded, so that I can verify no silent truncation or
bad download occurred.

#### Acceptance Criteria

1. WHEN the Sweep_Runner starts a run, THE Corpus_Loader SHALL load
   the BEIR SciFact corpus documents, test queries, and Qrels in the
   standard BEIR format.
2. WHEN the Corpus_Loader successfully loads the corpus documents,
   test queries, and Qrels described in Criterion 1 without raising
   an error, THE Corpus_Loader SHALL emit, to standard output or a
   run log file, exactly one deterministic, machine-parsable report
   (e.g., one line per count) containing the exact number of corpus
   documents loaded, the exact number of queries loaded, and the
   exact number of judged query-document pairs loaded, such that a
   test can capture and parse the report without ambiguity.
3. THE Corpus_Loader SHALL derive every count included in the report
   described in Criterion 2 from the loaded data structures at load
   time, rather than from a literal value written in source code.
4. IF the Corpus_Loader fails to load the corpus documents, the test
   queries, or the Qrels, for any reason, THEN THE Corpus_Loader
   SHALL raise an error identifying which of the three (corpus
   documents, test queries, or Qrels) failed to load, and SHALL NOT
   emit the count report described in Criterion 2.
5. IF the Corpus_Loader loads zero corpus documents, zero test queries,
   or zero Qrels entries, THEN THE Corpus_Loader SHALL raise an error
   naming which of the three (corpus documents, test queries, or Qrels
   entries) was empty, and SHALL NOT emit the count report described in
   Criterion 2, and THE Sweep_Runner SHALL halt without writing
   `results/sweep.csv`.
6. THE Corpus_Loader SHALL verify that every document ID referenced in
   the Qrels is present in the loaded corpus, and that every query ID
   referenced in the Qrels is present in the loaded query set. IF any
   such reference does not resolve, THEN THE Corpus_Loader SHALL raise
   an error reporting the count of unresolved document ID references
   and the count of unresolved query ID references, and THE
   Sweep_Runner SHALL halt, so that a partial or truncated download is
   detected without hard-coding any expected count.
7. IF the Corpus_Loader fails to load one or more of the corpus
   documents, test queries, or Qrels, THE Corpus_Loader MAY derive and
   include partial counts for any successfully loaded component as
   part of the error described in Criterion 4, and doing so SHALL NOT
   constitute, and SHALL NOT be treated as, the count report described
   in Criterion 2.
8. IF the Corpus_Loader successfully loads the corpus documents, test
   queries, and Qrels described in Criterion 1, but fails to derive or
   emit the count report described in Criterion 2, THEN THE
   Sweep_Runner SHALL abort the run before building any index, and
   SHALL NOT write `results/sweep.csv`, consistent with Criterion 5 of
   Requirement 8 — the count report is the only detector of a silent
   truncation or partial download and is not an optional step.

### Requirement 2: Config-Driven Sweep Grid Definition

**User Story:** As a researcher, I want the retriever x top-k x
chunking grid declared as data in a YAML config, so that the grid for
session 1 is explicit and not buried in hard-coded loops.

#### Acceptance Criteria

1. THE Sweep_Config SHALL declare, as data in a single YAML file under
   `configs/`, the set of retrievers, the set of evaluation cutoffs,
   and the chunking strategy for the sweep.
2. THE Sweep_Config SHALL declare exactly two retrievers (BM25 and
   `all-MiniLM-L6-v2`), exactly four evaluation cutoffs (1, 5, 10, and
   20), and exactly one chunking strategy (Whole_Document_Chunking).
3. THE Sweep_Config SHALL declare an explicit value for each of the
   BM25 tokenizer, stopword list, stemming choice, and case-handling
   setting, using an explicit "none" or "disabled" value for any
   setting that is intentionally not applied rather than omitting the
   field.
4. THE Sweep_Config SHALL declare the fixed random seed used for the
   run as a single explicit integer value.
5. WHEN the Sweep_Runner starts, THE Sweep_Runner SHALL derive the
   sweep grid entirely from the Sweep_Config, producing exactly one
   sweep combination for each retriever x evaluation-cutoff x
   chunking-strategy tuple declared in the Sweep_Config, with no
   combination omitted and no combination added beyond those declared.
6. IF the Sweep_Config file is missing, cannot be parsed as valid
   YAML, omits any required declaration (retrievers, evaluation
   cutoffs, chunking strategy, BM25 preprocessing settings, or random
   seed), or declares a retriever, evaluation cutoff, or chunking
   strategy that the Sweep_Runner does not support, THEN THE
   Sweep_Runner SHALL halt before producing sweep results and SHALL
   produce an error message indicating which declaration is missing,
   invalid, or unsupported.

### Requirement 3: BM25 Retriever Behavior

**User Story:** As a researcher, I want the BM25 retriever to build
one index and retrieve once per run using preprocessing fixed in
advance, so that the lexical baseline is computed consistently and
without post-hoc tuning.

#### Acceptance Criteria

1. WHEN the Sweep_Runner executes the BM25 configuration, THE
   BM25_Retriever SHALL build exactly one `BM25Okapi` index over the
   whole-document corpus.
2. THE BM25_Retriever SHALL tokenize documents and queries using the
   tokenizer, stopword list, stemming choice, and case-handling
   setting declared in the Sweep_Config.
3. THE BM25_Retriever SHALL apply the same text normalization
   pipeline to queries and to documents.
4. WHEN the Sweep_Runner queries the BM25_Retriever, THE
   BM25_Retriever SHALL produce, for each query, one Ranked_List
   containing exactly the top Deepest_Cutoff document IDs ordered by
   descending BM25 score in a single retrieval run, breaking any tie
   between two documents with equal BM25 score by ascending corpus
   document ID, compared numerically (as integers, since BEIR SciFact
   document IDs are numeric strings), so that the Ranked_List order is
   identical across repeated runs on the same corpus and query.
5. WHEN the BM25_Retriever completes the index build, THE
   BM25_Retriever SHALL record the wall-clock duration of the index
   build, in seconds, as Index_Time.
6. WHEN the BM25_Retriever completes the single retrieval run across
   all queries, THE BM25_Retriever SHALL record the wall-clock
   duration of that retrieval run, in seconds, as Query_Latency.
7. IF the Sweep_Config declares a tokenizer, stopword list, stemming
   choice, or case-handling setting that the BM25_Retriever does not
   support, THEN THE BM25_Retriever SHALL raise an error identifying
   which preprocessing setting is unsupported and SHALL NOT build the
   `BM25Okapi` index.

### Requirement 4: Dense Retriever (all-MiniLM-L6-v2) Behavior

**User Story:** As a researcher, I want the dense retriever to build
one index and retrieve once per run, entirely on CPU with model
weights cached inside the repo, so that the dense baseline is
reproducible and stays within the CPU-only, no-paid-API constraints.

#### Acceptance Criteria

1. WHEN the Sweep_Runner executes the `all-MiniLM-L6-v2` configuration,
   THE Dense_Retriever SHALL encode the whole-document corpus into
   embeddings exactly once to build a single index.
2. THE Dense_Retriever SHALL run the `all-MiniLM-L6-v2` model on CPU
   only.
3. THE Dense_Retriever SHALL set the sentence-transformers cache
   folder and the Hugging Face cache environment variables to a path
   under `data/` before loading model weights.
4. WHEN the Sweep_Runner queries the Dense_Retriever, THE
   Dense_Retriever SHALL encode each query into an embedding and
   produce one Ranked_List per query at the Deepest_Cutoff in a
   single retrieval run, ranking corpus documents by cosine
   similarity between the query embedding and each document
   embedding using brute-force exact comparison over every corpus
   embedding.
5. THE Dense_Retriever SHALL record, as Index_Time, the wall-clock
   duration of encoding the whole-document corpus into embeddings,
   excluding any query encoding time.
6. THE Dense_Retriever SHALL record, as Query_Latency, the wall-clock
   duration of the single retrieval run across all queries, including
   the time to encode each query into an embedding and the time to
   compute similarity scores against the corpus embeddings.
7. IF the `all-MiniLM-L6-v2` model weights cannot be downloaded to or
   loaded from the path under `data/`, THEN THE Dense_Retriever SHALL
   raise an error identifying that the model failed to load, without
   producing a Ranked_List.
8. WHEN the Sweep_Runner queries the Dense_Retriever, THE
   Dense_Retriever SHALL break any tie between two documents with
   equal cosine similarity by ascending corpus document ID, compared
   numerically (as integers, the same comparison rule applied by the
   BM25_Retriever in Requirement 3.4), so that the Ranked_List order
   is identical across repeated runs on the same corpus and query.

### Requirement 5: Single Retrieval Sliced to Four Cutoffs

**User Story:** As a researcher, I want each retriever indexed once
and queried once at the deepest cutoff, with that single ranked list
sliced to compute metrics at every cutoff, so that timing columns
reflect the true number of retrieval operations rather than implying
four independent retrievals per retriever.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL build exactly one index for each retriever
   declared in the Sweep_Config.
2. THE Sweep_Runner SHALL execute exactly one retrieval run for each
   retriever declared in the Sweep_Config, across all queries, at the
   Deepest_Cutoff, producing for each query a Ranked_List containing
   exactly Deepest_Cutoff document IDs ordered by descending relevance
   score, or all corpus documents (ordered by descending relevance
   score) if the corpus contains fewer than Deepest_Cutoff documents.
3. THE Sweep_Runner SHALL derive the Ranked_List used to compute
   metrics at each evaluation cutoff k in {1, 5, 10, 20} by taking the
   first k document IDs, in ranked order, from the single
   Deepest_Cutoff Ranked_List produced for the same retriever and
   query.
4. THE Sweep_Runner SHALL reuse the single Deepest_Cutoff Ranked_List
   for all four evaluation cutoffs without issuing an additional
   retrieval call to the retriever for any cutoff.
5. THE Sweep_Runner SHALL perform, across a full run, exactly two
   index builds and exactly two retrieval runs in total, consisting
   of exactly one index build and exactly one retrieval run per
   retriever declared in the Sweep_Config.
6. THE Sweep_Runner SHALL assign the same Index_Time value, taken
   from the retriever's single index build, to every Sweep_Report row
   sharing that run's Run_Id.
7. THE Sweep_Runner SHALL assign the same Query_Latency value, taken
   from the retriever's single Deepest_Cutoff retrieval run, to every
   Sweep_Report row sharing that run's Run_Id.

### Requirement 6: Metric Computation Against Qrels Only

**User Story:** As a researcher, I want recall@k, nDCG@10, and MRR@10
computed strictly against the BEIR SciFact qrels, with nDCG@10 fixed
as the primary metric in advance, so that scoring is honest and the
headline metric is not chosen after seeing results.

#### Acceptance Criteria

1. THE Metrics_Calculator SHALL compute recall@k, for k in {1, 5, 10,
   20}, for each test query as the count of that query's
   Qrels-judged relevant documents appearing in the query's top-k
   Ranked_List divided by the total count of that query's
   Qrels-judged relevant documents, and SHALL report the arithmetic
   mean of this per-query value across all test queries that have at
   least one Qrels-judged relevant document.
2. THE Metrics_Calculator SHALL compute nDCG@10, for each test query,
   as DCG@10 divided by IDCG@10, defined as 0 when IDCG@10 is 0,
   where: DCG@10 is the sum, over ranks i = 1 to 10 of the top 10
   documents of the Ranked_List, of rel_i / log2(i + 1), with rel_i
   equal to the graded relevance score recorded in the Qrels for the
   document at rank i, or 0 if that document is unjudged; and IDCG@10
   is the same sum computed over that query's Qrels-judged relevant
   documents sorted by descending relevance score and truncated to
   the first 10 documents. THE Metrics_Calculator SHALL report the
   arithmetic mean of the per-query nDCG@10 value across all test
   queries that have at least one Qrels-judged relevant document.
   This nDCG@10 convention (the log2(i + 1) discount, graded
   relevance taken from the Qrels, and the 0-when-IDCG@10-is-0 rule)
   SHALL be fixed before any sweep result exists and SHALL be
   recorded in `SPEC.md`.
3. THE Metrics_Calculator SHALL compute MRR@10, for each test query,
   as the reciprocal of the rank position of the first Qrels-judged
   relevant document within the top 10 documents of the Ranked_List,
   or zero if no Qrels-judged relevant document appears within the
   top 10, independent of the evaluation cutoff k assigned to a
   Sweep_Report row, and SHALL report the arithmetic mean of this
   per-query value across all test queries that have at least one
   Qrels-judged relevant document.
4. IF a query-document pair is absent from the Qrels, or is present
   in the Qrels with a relevance score of zero or less, THEN THE
   Metrics_Calculator SHALL treat that query-document pair as not
   relevant, regardless of any similarity or relevance score produced
   by a retriever or model, because the Qrels are the sole source of
   relevance ground truth for this determination.
5. THE Metrics_Calculator SHALL designate nDCG@10 as the primary
   metric and recall@k and MRR@10 as secondary metrics, with all
   three reported for every Sweep_Report row.
6. THE Metrics_Calculator SHALL report the same nDCG@10 value and the
   same MRR@10 value for every Sweep_Report row sharing the same
   Run_Id, because nDCG@10 and MRR@10 are computed once per Run_Id at
   a fixed cutoff of 10 rather than varying by the row's evaluation
   cutoff k.
7. THE Metrics_Calculator SHALL make available, alongside each mean
   computed by Criteria 1, 2, and 3, the total number of test queries
   loaded and the number of those test queries with at least one
   Qrels-judged relevant document (the denominator of that mean), so
   that the Sweep_Runner can record both counts in the Sweep_Report.

### Requirement 7: Sweep Report Output Format and Row Count

**User Story:** As a researcher, I want `results/sweep.csv` to contain
exactly the 8 declared rows with full metric and timing data per row,
so that BM25 can later serve as the reference row for reporting
without missing data.

#### Acceptance Criteria

1. WHEN the Sweep_Runner finishes executing every retriever x cutoff
   combination declared in the Sweep_Config, whether or not an
   individual combination's retrieval or metric computation succeeded,
   THE Sweep_Runner SHALL write the Sweep_Report with exactly one row
   per declared combination.
2. THE Sweep_Runner SHALL write exactly 8 rows to the Sweep_Report,
   corresponding to the 2 retrievers x 4 cutoffs declared in the
   Sweep_Config.
3. THE Sweep_Runner SHALL include, in every Sweep_Report row: a
   run_id identifying the index-build-and-retrieval run the row was
   scored from; the retriever name; the chunking strategy; the row's
   evaluation cutoff; recall@k evaluated at that row's evaluation
   cutoff; nDCG@10 evaluated at a fixed cutoff of 10 independent of
   the row's evaluation cutoff; MRR@10 evaluated at a fixed cutoff of
   10 independent of the row's evaluation cutoff; Index_Time;
   Query_Latency; the total number of test queries loaded; and the
   number of those test queries that have at least one Qrels-judged
   relevant document (the number of queries scored, i.e. the
   denominator of the recall@k, nDCG@10, and MRR@10 means for that
   row), so that the denominator of every reported mean is visible in
   the artifact.
4. THE Sweep_Runner SHALL assign the same run_id value to every
   Sweep_Report row that shares the same retriever and the same
   chunking strategy, and SHALL assign a different run_id value to
   rows produced from a different index build and retrieval run.
5. THE Sweep_Runner SHALL record the chunking strategy value as
   Whole_Document_Chunking for every Sweep_Report row.
6. THE Sweep_Runner SHALL retain every row of the declared grid in the
   Sweep_Report regardless of the metric values computed for that
   row.
7. THE Sweep_Runner SHALL record the same Index_Time value and the
   same Query_Latency value for every Sweep_Report row sharing the
   same run_id.
8. IF a retriever x cutoff combination's retrieval or metric
   computation fails, THEN THE Sweep_Runner SHALL still write the row
   for that combination, recording an explicit missing-value marker
   for each metric or timing value that could not be computed, and
   that marker SHALL be distinguishable from a legitimately computed
   zero value.

### Requirement 8: Reproducibility via Fixed Random Seed

**User Story:** As a researcher, I want every source of randomness
seeded and recorded, so that metric values are identical across
reruns on the same machine.

#### Acceptance Criteria

1. THE Sweep_Config SHALL declare a single fixed random seed, as an
   integer value, used across the run.
2. WHEN the Sweep_Runner starts a run, THE Sweep_Runner SHALL apply
   the fixed random seed declared in the Sweep_Config to Python's
   `random` module, to NumPy's random number generator, to `torch`
   via `torch.manual_seed(seed)`, and to the Dense_Retriever's
   embedding batch ordering, before executing any embedding or
   evaluation step of that run.
3. THE Sweep_Runner SHALL write a single accompanying run
   configuration record to `results/`, alongside the Sweep_Report,
   recording: the fixed random seed value applied for the run; the
   resolved contents of the Sweep_Config used for the run; and the
   installed version of each of `beir`, `rank_bm25`,
   `sentence-transformers`, `torch`, and `numpy`.
4. WHILE the fixed random seed and the input data remain unchanged,
   THE Sweep_Runner SHALL produce recall@k, nDCG@10, and MRR@10
   values that are exactly equal, at the numeric precision written to
   the Sweep_Report, across at least two consecutive runs on the same
   machine. Because Sweep_Runner end-to-end tests are deferred to a
   later spec (see Requirement 11), THE rerun-identity property
   described in this criterion SHALL be verified manually during
   session 1 by running the entry point twice and diffing the two
   resulting `results/sweep.csv` files cell-by-cell on every column
   except Index_Time and Query_Latency, rather than by an automated
   pytest test in this spec; automated verification of rerun-identity
   SHALL be deferred to a later spec.
5. IF applying the fixed random seed to Python's `random` module, to
   NumPy's random number generator, to `torch` via
   `torch.manual_seed(seed)`, or to the Dense_Retriever's embedding
   batch ordering fails for any reason, THEN THE Sweep_Runner SHALL
   abort the run before executing any embedding or evaluation step,
   and SHALL NOT write `results/sweep.csv`.

### Requirement 9: Pinned Dependency Versions

**User Story:** As a researcher, I want `requirements.txt` to pin
exact dependency versions, so that the sweep environment is
reproducible on a clean checkout.

#### Acceptance Criteria

1. THE `requirements.txt` file SHALL list `beir`, `rank_bm25`,
   `sentence-transformers`, `numpy`, `pandas`, `PyYAML`, and `pytest`
   as dependencies.
2. THE `requirements.txt` file SHALL pin each dependency listed in
   Criterion 1 to an exact version using the `==` operator, and SHALL
   NOT use range operators (`>=`, `<=`, `~=`, `>`, `<`, or `*`) for any
   listed dependency.
3. WHERE `requirements.txt` includes any dependency beyond the 7
   packages listed in Criterion 1, THE file SHALL include that
   dependency only if it is a transitive dependency required by
   `beir`, `rank_bm25`, `sentence-transformers`, `numpy`, `pandas`,
   `PyYAML`, or `pytest` (for example, a CPU-only build of `torch`
   required by `sentence-transformers`), and SHALL pin that dependency
   to an exact version using the `==` operator.
4. THE `requirements.txt` file SHALL NOT contain more than one entry
   for the same package name.
5. IF any line in `requirements.txt` declares a dependency without an
   exact version pin using the `==` operator, THEN THE entire
   `requirements.txt` file SHALL be considered invalid, rather than
   treating only that individual entry as invalid.

### Requirement 10: Single Entry Point for End-to-End Sweep Execution

**User Story:** As a researcher, I want one command to run the whole
session-1 sweep from the YAML config to `results/sweep.csv` on a clean
checkout, so that the study is reproducible without manual steps.

#### Acceptance Criteria

1. THE Sweep_Runner SHALL provide a single entry point, invocable as
   one command-line invocation, that accepts a path to a Sweep_Config
   file and, when no path is provided, applies a default Sweep_Config
   file located under `configs/`, and runs the complete sweep from
   that Sweep_Config to the Sweep_Report.
2. WHEN a user invokes the entry point on a clean checkout with
   `requirements.txt` installed, THE Sweep_Runner SHALL perform corpus
   loading, indexing, retrieval, metric computation, and Sweep_Report
   writing without requiring the user to manually download the BEIR
   SciFact corpus, manually download model weights, manually create
   the `data/` or `results/` directories, or edit source code at any
   point between invoking the entry point and the Sweep_Report being
   written.
3. WHEN the Sweep_Runner completes all steps and every Sweep_Report
   row was computed without a missing-value marker, THE Sweep_Runner
   SHALL terminate with a zero exit status and leave the Sweep_Report
   present at `results/sweep.csv`.
4. IF corpus loading, Sweep_Config parsing, or Sweep_Report writing
   fails outright, THEN THE Sweep_Runner SHALL terminate with a
   non-zero exit status, SHALL report an error identifying which step
   failed, and SHALL NOT leave a partial or corrupted Sweep_Report at
   `results/sweep.csv`. IF an individual retriever x cutoff
   combination's retrieval or metric computation fails while other
   combinations succeed, THEN THE Sweep_Runner SHALL apply the
   missing-value-marker recovery described in Requirement 7 for that
   combination, SHALL still write every declared combination's row to
   the Sweep_Report, whether computed or marked missing, and SHALL
   terminate with a non-zero exit status, because a run in which any
   declared combination failed is not a successful run.
5. THE Sweep_Runner SHALL direct every BEIR dataset download and every
   Hugging Face model weight download triggered during the run to a
   path under `data/`.
6. THE Sweep_Runner SHALL execute every step of the sweep using
   CPU-only code paths.
7. THE Sweep_Runner SHALL obtain all model inference from locally
   executed, free, open-weight models downloaded to `data/`, with no
   call to a paid or metered inference API.

### Requirement 11: Test Coverage Scope Limited to Metric Functions

**User Story:** As a maintainer, I want pytest coverage over the
recall@k, nDCG@10, and MRR@10 functions using hand-computed fixtures,
so that the metric logic is verified without any network access or
dependency on the data-loading layer or the sweep runner.

#### Acceptance Criteria

1. THE test suite SHALL include pytest tests for the recall@k
   function, the nDCG@10 function, and the MRR@10 function, with each
   function covering at minimum the following scenarios: no document
   judged relevant for the query, at least one relevant document
   ranked outside the top-k cutoff, a perfect ranking where all
   relevant documents occupy the top positions of the ranked list, and
   an empty ranked list.
2. THE test suite SHALL verify each metric function against fixtures
   containing no more than 10 documents, with expected values computed
   independently of the function under test (by manual calculation or
   a separate reference computation, not by re-running the function
   itself), and SHALL compare float results using a numeric tolerance
   of 1e-6.
3. THE test suite SHALL execute without making any network call and
   without downloading any dataset or model.
4. THE test suite for this spec SHALL cover only the Metrics_Calculator
   functions, SHALL NOT import or invoke any Corpus_Loader or
   Sweep_Runner code, and SHALL defer Corpus_Loader tests and
   Sweep_Runner end-to-end tests to a later spec.
5. THE test suite SHALL additionally assert, for each of recall@k,
   nDCG@10, and MRR@10, that the value produced by the
   Metrics_Calculator function agrees with the value produced by
   `pytrec_eval` on the same fixture to within a numeric tolerance of
   1e-6, so that a convention error, as opposed to only an arithmetic
   error, is caught. Because `pytrec_eval`'s `recip_rank` measure is
   computed over the full submitted ranking rather than a fixed
   cutoff, THE test suite SHALL truncate the ranked list passed to
   `pytrec_eval` to the top 10 documents per query before the MRR@10
   comparison only; the `ndcg_cut_10` and `recall_<k>` comparisons
   SHALL pass the full ranked list to `pytrec_eval` without
   truncation, since those `pytrec_eval` measures already apply their
   own cutoff. `pytrec_eval` SHALL be treated as a transitive
   dependency of `beir` (installed via the `pytrec-eval-terrier` PyPI
   package, which provides the `pytrec_eval` import), and SHALL be
   pinned to an exact version in `requirements.txt` per Requirement 9.
