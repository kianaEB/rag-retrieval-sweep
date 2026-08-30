
# Requirements Document

## Introduction

This spec covers the session-3 stretch goal named in `docs/PROJECT_BRIEF.md`:
"Groundedness gate over generated answers, quarantine rate, hand-checked
sample." The brief further notes this work is "brought forward ahead of
the remaining session-2 work (chunking, failure bucketing,
`ANALYSIS.md`)" because the repo otherwise contains no generative model
and this is the only planned work that demonstrates prompting and
generation. Per `.kiro/steering/tech.md`, local generative models are
in scope for this purpose (CPU-only, no fine-tuning, no paid or metered
inference API), and generated-answer quality is explicitly not a
deliverable — the measured quantities are the quarantine rate and its
agreement with a hand-checked sample, not how good the answers read.

`results/sweep.csv`, `results/per_query.csv`, `results/significance.csv`,
`results/run_config.json`, and `configs/sweep.yaml` already exist and
are **read-only inputs** to this spec. This spec does not regenerate,
edit, or overwrite any of them, and does not change `src/sweep_runner.py`
or the retrieval logic it drives. `results/per_query.csv` carries only
aggregate per-query metric values (recall@k, nDCG@10, MRR@10,
`num_judged_relevant`) — it has no retrieved-document-ID column and
cannot supply "the top-k retrieved documents" for a query. Because that
is the only source of top-k document IDs anywhere in this repo, this
spec obtains retrieved context by re-invoking the exact, already-frozen
retriever class and config block a prior sweep run already used —
identified by that run's `run_id` — for a small, declared, seeded
subset of queries only. This is a read-only reuse of existing frozen
retrieval code and configuration to obtain document IDs, not a new
retrieval experiment, and it changes no retrieval logic, no
`configs/sweep.yaml` schema, and no `results/per_query.csv` schema.
This spec also follows the non-destructive `results/run_config.json`
merge pattern already established by the significance-testing spec:
every existing key (`seed`, `sweep_config`, `corpus_load_report`,
`installed_versions`, `significance`) is preserved unchanged, and this
feature's own parameters are recorded under one new sibling key.

Out of scope for this spec, per `.kiro/steering/scope-guard.md`: the
third retriever (`BAAI/bge-small-en-v1.5`); any chunking strategy beyond
whatever the replayed run already used; any new retriever, reranking,
hybrid/fusion retrieval, query expansion, or approximate nearest
neighbour index; failure bucketing and `ANALYSIS.md`; fine-tuning of any
model; and any chat UI or serving interface/endpoint. This spec also
does not regenerate or modify `results/sweep.csv`, `results/per_query.csv`,
`results/significance.csv`, or the sweep/significance code paths. pytest
coverage in this spec covers the claim-segmentation function and the
quarantine-decision function only (Requirement 12); an end-to-end test
of the orchestrating entry point and of the retrieval-replay component
against the real corpus is not tested in this spec.

## Glossary

- **Groundedness_Config**: The YAML file under `configs/` that declares,
  as data, the Replayed_Run's `run_id`, the replay top-k depth, the
  Generation_Subset size and seed, the Generator_Model and Judge_Model
  identifiers, the Prompt_Template, the Quarantine_Rule's threshold,
  and the Hand_Checked_Sample size and seed.
- **Groundedness_Runner**: The single entry point that orchestrates
  Retrieval_Replay, prompt construction, Generator_Model invocation,
  Claim_Segmenter invocation, Judge_Model invocation, Quarantine_Rule
  application, and Groundedness_Report writing for the
  Generation_Subset.
- **Replayed_Run**: The single existing sweep run, identified by its
  `run_id` in `configs/sweep.yaml` and `results/per_query.csv`, whose
  frozen retriever type and configuration Retrieval_Replay reuses to
  obtain Retrieved_Context.
- **Frozen_Retriever_Config**: The exact retriever configuration block
  (BM25 or dense) named for the Replayed_Run inside `configs/sweep.yaml`,
  reused unchanged by Retrieval_Replay.
- **Retrieval_Replay**: The read-only component that reconstructs the
  Replayed_Run's exact retriever class from its Frozen_Retriever_Config
  and invokes it, on demand, to retrieve the top-k document IDs for one
  Generation_Subset query. Not a new retrieval experiment: it changes
  no retrieval logic, no `configs/sweep.yaml` schema, and no
  `results/sweep.csv` or `results/per_query.csv` schema, and it is the
  only source of document IDs for Retrieved_Context because
  `results/per_query.csv` carries only aggregate per-query metric
  values and no document-ID column.
- **Retrieved_Context**: The ordered set of top-k document texts
  obtained via Retrieval_Replay for one Generation_Subset query, used
  both to build that query's prompt and as the sole evidence Judge_Model
  checks each of that query's Claims against.
- **Generation_Subset**: The fixed set of query IDs sampled, using
  Generation_Subset_Seed, from the query IDs the Replayed_Run actually
  scored (per `results/per_query.csv`), on which this feature's
  generation and judging pipeline runs.
- **Generation_Subset_Seed**: The single fixed integer seed, declared
  in the Groundedness_Config, applied to sampling the Generation_Subset.
- **Prompt_Template**: The fixed, declared-in-advance template,
  declared as data in the Groundedness_Config, that combines a
  Generation_Subset query's text and its Retrieved_Context into the
  text input given to the Generator_Model.
- **Generator_Model**: The small instruction-tuned language model, run
  on CPU with weights cached under `data/`, that generates one
  Generated_Answer per Generation_Subset query from its prompt. Default
  `google/flan-t5-base`.
- **Generated_Answer**: The text the Generator_Model produces for one
  Generation_Subset query.
- **Claim_Segmenter**: The pure-function component that splits a
  Generated_Answer into an ordered list of Claims at sentence
  boundaries.
- **Claim**: One sentence-boundary-delimited segment of a
  Generated_Answer, identified by its `query_id` and its `claim_index`
  (position within that answer).
- **Judge_Model**: The small NLI cross-encoder model, distinct from the
  Generator_Model, that scores one Claim's support by the same query's
  Retrieved_Context. Default `cross-encoder/nli-deberta-v3-xsmall`.
- **Groundedness_Verdict**: The Judge_Model's categorical support
  determination for one Claim against its Retrieved_Context, derived
  by mapping the Judge_Model's native three-way Natural Language
  Inference label (`entailment`, `neutral`, `contradiction`) to
  exactly one of `SUPPORTED` (from `entailment`) or `NOT_SUPPORTED`
  (from `neutral` or `contradiction`), together with a numeric score
  equal to the entailment probability from the softmax over the
  Judge_Model's three logits, in the range 0.0 to 1.0.
- **Quarantine_Rule**: The pure function that maps a Claim's
  Groundedness_Verdict and score, together with the threshold declared
  in the Groundedness_Config, to a Quarantine_Decision.
- **Quarantine_Decision**: The boolean outcome recorded for one Claim:
  whether that Claim is withheld from the Served_Answer.
- **Served_Answer**: The subset of a Generated_Answer's Claims for
  which Quarantine_Decision is false.
- **Groundedness_Report**: The `results/groundedness.csv` file,
  containing one row per (`query_id`, `claim_index`) pair produced for
  the Generation_Subset.
- **Quarantine_Rate**: The fraction of Groundedness_Report rows whose
  Quarantine_Decision is true, computed by aggregation over the
  Groundedness_Report rather than stored as a separate literal.
- **Hand_Checked_Sample**: A fixed number of Claims, selected without
  replacement from a canonical ordering of Claims sorted by `query_id`
  and then by `claim_index`, using the Hand_Checked_Sample_Seed
  independently of any Groundedness_Verdict, exported for manual
  labelling.
- **Hand_Checked_Sample_Seed**: The single fixed integer seed, declared
  in the Groundedness_Config and distinct from Generation_Subset_Seed,
  applied to selecting the Hand_Checked_Sample.
- **Hand_Label**: The manually assigned support label for one
  Hand_Checked_Sample Claim, recorded by a human reviewer.
- **Hand_Label_Import**: The re-imported file containing every
  Hand_Checked_Sample Claim's Hand_Label, read back by the
  Groundedness_Runner to compute Agreement_Rate.
- **Agreement_Rate**: The fraction of Hand_Checked_Sample Claims whose
  Judge_Model Groundedness_Verdict matches the corresponding Hand_Label
  in the Hand_Label_Import.
- **Qrels**: The human relevance judgments shipped with BEIR SciFact.
  The sole ground truth for retrieval metrics in the sessions preceding
  this feature; never read, and never used as a substitute for
  Retrieved_Context, by the Judge_Model.
- **Downstream_Writeup**: Any future `README.md`, `SPEC.md`, or
  `ANALYSIS.md` content — none of which is produced by this spec — that
  states a Quarantine_Rate or Agreement_Rate value attributed to this
  feature.

## Requirements

### Requirement 1: Groundedness_Config Declared As Data

**User Story:** As a researcher, I want the generation subset, the
replay target, and both model identifiers declared as data in a single
YAML file, so that the groundedness gate's parameters are explicit and
not hard-coded.

#### Acceptance Criteria

1. THE Groundedness_Config SHALL declare, as data in a single YAML
   file under `configs/`, the Replayed_Run's `run_id`, the replay
   top-k depth, the Generation_Subset size, the Generation_Subset_Seed,
   the Generator_Model identifier, the Judge_Model identifier, the
   Prompt_Template, the Quarantine_Rule's support-score threshold, the
   Hand_Checked_Sample size, and the Hand_Checked_Sample_Seed, and a
   documented record of the Judge_Model's native-label-to-Groundedness_Verdict
   mapping and the numeric score's definition required by Requirement
   6's Criteria 2, 9, and 10.
2. THE Groundedness_Config SHALL declare the Generation_Subset_Seed and
   the Hand_Checked_Sample_Seed as two separate explicit integer
   fields, such that neither seed is derived from the other.
3. THE Groundedness_Config SHALL declare the Generator_Model identifier
   and the Judge_Model identifier as two distinct string values, such
   that the two identifiers never name the same model.
4. THE Groundedness_Config SHALL declare the Hand_Checked_Sample size
   as a single explicit positive integer value, defaulting to 50
   claims when not overridden.
5. IF the Groundedness_Config file is missing, cannot be parsed as
   valid YAML, omits any declaration required by Criterion 1, declares
   the Generator_Model identifier and the Judge_Model identifier as the
   same value, declares the replay top-k depth, the Generation_Subset
   size, or the Hand_Checked_Sample size as a value other than a
   positive integer, declares the Generation_Subset_Seed or the
   Hand_Checked_Sample_Seed as a value other than an integer, or
   declares the Quarantine_Rule's support-score threshold as a value
   other than numeric, THEN THE Groundedness_Runner SHALL halt before
   generating any answer or writing the Groundedness_Report, and SHALL
   produce an error message identifying which declaration is missing,
   invalid, or in conflict.
6. THE Groundedness_Config SHALL declare the replay top-k depth and
   the Generation_Subset size, each as a single explicit positive
   integer value.
7. THE Groundedness_Config SHALL declare the Quarantine_Rule's
   support-score threshold as a single explicit numeric value.

### Requirement 2: Generation_Subset Sampled From The Replayed Run's Scored Queries

**User Story:** As a researcher, I want the generation subset sampled
only from queries the replayed run actually scored, so that every
sampled query has an established retriever identity and run context to
replay against.

#### Acceptance Criteria

1. IF `results/per_query.csv` is absent, cannot be parsed as a CSV
   file, or does not contain both a `run_id` column and a `query_id`
   column, THEN THE Groundedness_Runner SHALL halt before sampling the
   Generation_Subset, and SHALL produce an error message stating
   whether `results/per_query.csv` is missing, unparsable, or missing
   an expected column.
2. WHEN the Groundedness_Runner starts a run, THE Groundedness_Runner
   SHALL read `results/per_query.csv`, treated as read-only, and SHALL
   determine the set of query IDs that the Replayed_Run's `run_id`
   actually scored as the distinct values of the `query_id` column
   across every row whose `run_id` column equals the Replayed_Run's
   `run_id` declared in the Groundedness_Config.
3. THE Groundedness_Runner SHALL sample the Generation_Subset by
   drawing, uniformly at random and without replacement, the number of
   query IDs declared as the Generation_Subset size in the
   Groundedness_Config from the query IDs identified by Criterion 2,
   seeding that draw with the Generation_Subset_Seed and applying it to
   a canonical ordering of those query IDs sorted in ascending order,
   such that the resulting Generation_Subset is the same regardless of
   the order in which `results/per_query.csv`'s rows were read.
4. THE Groundedness_Runner SHALL NOT read any document ID, ranked
   list, or retrieved-content value from `results/per_query.csv`.
5. IF the Replayed_Run's `run_id` declared in the Groundedness_Config
   is not present in the `run_id` column of `results/per_query.csv`,
   THEN THE Groundedness_Runner SHALL halt before sampling the
   Generation_Subset, and SHALL produce an error message stating that
   the declared `run_id` was not found.
6. IF the declared Generation_Subset size exceeds the number of query
   IDs identified by Criterion 2, THEN THE Groundedness_Runner SHALL
   select every identified query ID, rather than raising an error.

### Requirement 3: Retrieval_Replay Obtains Retrieved_Context From The Frozen Retriever

**User Story:** As a researcher, I want retrieved context obtained by
re-invoking the exact frozen retriever the sweep already used, so that
the groundedness gate never becomes a second retrieval experiment.

#### Acceptance Criteria

1. WHEN the Groundedness_Runner begins processing the Generation_Subset,
   THE Retrieval_Replay component SHALL construct the retriever type
   declared for the Replayed_Run's Frozen_Retriever_Config in
   `configs/sweep.yaml` and build that retriever's index exactly once,
   over the corpus identified by the `data_dir` declared in
   `configs/sweep.yaml`, rather than reconstructing the retriever or
   rebuilding its index separately for each Generation_Subset query.
2. WHEN the Groundedness_Runner processes a Generation_Subset query,
   THE Retrieval_Replay component SHALL retrieve that query's ranked
   document IDs, at the replay top-k declared in the Groundedness_Config,
   from the index already built for the Replayed_Run's retriever, and
   SHALL set that query's Retrieved_Context to the ordered list of
   document texts corresponding to those document IDs, preserving
   retrieval-rank order.
3. THE Retrieval_Replay component SHALL reuse the Frozen_Retriever_Config's
   field values exactly as declared in `configs/sweep.yaml`, without
   modifying any field of that configuration.
4. THE Retrieval_Replay component SHALL NOT modify `configs/sweep.yaml`,
   `results/sweep.csv`, `results/per_query.csv`, or `src/sweep_runner.py`.
5. THE Retrieval_Replay component SHALL limit its retrieval calls to
   the query IDs in the Generation_Subset, and SHALL NOT issue a
   retrieval call for any query outside the Generation_Subset.
6. IF the Frozen_Retriever_Config declared for the Replayed_Run cannot
   be loaded, THEN THE Groundedness_Runner SHALL halt the entire run
   before processing any Generation_Subset query, SHALL NOT write the
   Groundedness_Report, and SHALL produce an error message stating that
   the declared Frozen_Retriever_Config could not be loaded.
7. IF the retriever declared by the Frozen_Retriever_Config fails to
   build its index, or fails to retrieve documents for any
   Generation_Subset query, THEN THE Groundedness_Runner SHALL halt the
   entire run without generating an answer for that query or any
   remaining Generation_Subset query, SHALL NOT write the
   Groundedness_Report, and SHALL produce an error message identifying
   the failed query ID when the failure occurred during retrieval, or
   stating that index construction failed, together with a description
   of the failure.

### Requirement 4: Prompt Construction And CPU-Only Answer Generation

**User Story:** As a researcher, I want each generation-subset query's
retrieved context turned into a prompt and answered by a small
CPU-only model, so that the groundedness gate produces answers to judge
without any paid inference call.

#### Acceptance Criteria

1. WHEN the Retrieval_Replay component returns a Generation_Subset
   query's Retrieved_Context, THE Groundedness_Runner SHALL build that
   query's prompt from the Prompt_Template declared in the
   Groundedness_Config, combining the query text and the
   Retrieved_Context.
2. WHILE any Generated_Answer has been produced for any
   Generation_Subset query during the current run, THE
   Groundedness_Runner SHALL NOT modify the Prompt_Template.
3. THE Generator_Model SHALL run on CPU only, with model weights
   cached under a path under `data/`, using the same Hugging Face
   cache environment variable convention already applied by the
   existing dense retriever.
4. WHEN the Groundedness_Runner submits a Generation_Subset query's
   prompt to the Generator_Model, THE Generator_Model SHALL produce
   exactly one Generated_Answer for that query, and SHALL produce the
   byte-for-byte identical Generated_Answer when that same prompt is
   submitted again on a rerun using the same seed on the same machine.
5. THE Groundedness_Runner SHALL obtain the Generator_Model's inference
   from a locally executed, free, open-weight model, and SHALL NOT
   call any paid or metered inference API.
6. THE Groundedness_Runner SHALL NOT fine-tune the Generator_Model.
7. IF the Generator_Model's weights cannot be downloaded to or loaded
   from the path under `data/`, THEN THE Groundedness_Runner SHALL
   raise an error identifying that the Generator_Model failed to load,
   without producing a Generated_Answer.
8. IF the Generator_Model fails to produce a Generated_Answer for a
   Generation_Subset query after its weights have already been loaded
   successfully, THEN THE Groundedness_Runner SHALL halt before writing
   the Groundedness_Report, SHALL NOT produce a partial Generated_Answer
   for that query, and SHALL produce an error message identifying that
   query's `query_id` together with a description of the failure.

### Requirement 5: Claim Segmentation At Sentence Boundaries

**User Story:** As a maintainer, I want the generated answer split into
claims at sentence boundaries with the heuristic's limitation
documented, so that the segmentation step is never mistaken for a
solved NLP problem.

#### Acceptance Criteria

1. WHEN a Generated_Answer is produced for a Generation_Subset query,
   THE Claim_Segmenter SHALL split that Generated_Answer into an
   ordered list of Claims at sentence boundaries, where a sentence
   boundary is any occurrence of `.`, `!`, or `?` that is immediately
   followed by one or more whitespace characters or by the end of the
   Generated_Answer text.
2. THE Claim_Segmenter SHALL assign each Claim a `claim_index` equal to
   its position, starting at 0, within its Generated_Answer's ordered
   list of Claims, and SHALL set each Claim's text to the
   corresponding segment of the Generated_Answer, including that
   segment's terminating sentence-boundary punctuation character when
   present, with leading and trailing whitespace removed; any segment
   that is empty after whitespace removal SHALL be excluded from the
   ordered list of Claims and SHALL NOT receive a `claim_index`.
3. THE Claim_Segmenter SHALL perform sentence-boundary splitting as a
   pure function of the Generated_Answer text, without loading any
   model and without making any network call.
4. THE Claim_Segmenter's documentation SHALL identify sentence-boundary
   splitting as a crude heuristic rather than a solved
   natural-language-processing problem, and SHALL state that a
   mis-split sentence is a source of measurement error in what counts
   as one Claim.
5. IF a Generated_Answer, after leading and trailing whitespace
   removal, contains no sentence boundary as defined in Criterion 1,
   THEN THE Claim_Segmenter SHALL treat that whitespace-trimmed text
   as a single Claim with `claim_index` 0, rather than raising an
   error, including when that whitespace-trimmed text is the empty
   string.

### Requirement 6: Groundedness Judging Against Retrieved Context By A Distinct Model

**User Story:** As a researcher, I want every claim checked for support
against the same retrieved context by a model different from the one
that generated it, so that the groundedness check is not judged by a
model biased toward accepting its own output.

#### Acceptance Criteria

1. WHEN a Claim is produced by the Claim_Segmenter, THE Judge_Model
   SHALL produce, for that Claim, exactly one Groundedness_Verdict
   whose value is either `SUPPORTED` or `NOT_SUPPORTED`, by checking
   that Claim against the same query's Retrieved_Context.
2. THE Judge_Model SHALL produce, as its native output for each Claim,
   exactly one three-way Natural Language Inference label whose value
   is `entailment`, `neutral`, or `contradiction`, and THE
   Groundedness_Runner SHALL map that native label to the
   Groundedness_Verdict required by Criterion 1 as follows:
   `entailment` maps to `SUPPORTED`; `neutral` maps to
   `NOT_SUPPORTED`; `contradiction` maps to `NOT_SUPPORTED` — so that a
   Claim the Retrieved_Context does not support is withheld whether
   the Retrieved_Context is silent on that Claim or contradicts it.
3. THE label mapping declared in Criterion 2 SHALL be fixed before any
   Quarantine_Rate exists, SHALL be recorded in `SPEC.md`, and SHALL
   NOT be revised after any Quarantine_Rate has been computed.
4. THE Judge_Model SHALL NOT check any Claim against the Qrels.
5. THE Judge_Model SHALL be a model distinct from the Generator_Model.
6. THE Groundedness_Runner's documentation SHALL state that the
   Judge_Model is required to differ from the Generator_Model because
   a model judging its own generated output is biased toward accepting
   it.
7. THE Judge_Model SHALL run on CPU only, with model weights cached
   under a path under `data/`.
8. THE Groundedness_Runner SHALL NOT fine-tune the Judge_Model.
9. THE Judge_Model SHALL report, alongside the categorical
   Groundedness_Verdict for every Claim, a numeric score defined as
   the entailment probability obtained by applying softmax to the
   Judge_Model's three logits (entailment, neutral, contradiction),
   yielding a value in the range 0.0 to 1.0 where a higher value
   indicates stronger support for that Claim by the Retrieved_Context.
10. THE score's definition declared in Criterion 9 SHALL be recorded
    in the Groundedness_Config and in `SPEC.md`, alongside the label
    mapping declared in Criterion 2, so that the threshold declared in
    the Groundedness_Config and applied by the Quarantine_Rule
    (Requirements 7.2 and 7.3) gates a defined measurement.
11. THE Groundedness_Runner's documentation SHALL state that the
    Judge_Model/Generator_Model separation reduces, but does not
    eliminate, judge bias, and SHALL state that the Hand_Checked_Sample
    exists to quantify the Judge_Model's agreement with human judgment.
12. IF the Judge_Model's weights cannot be loaded from the path under
    `data/`, or the Judge_Model fails to produce a Groundedness_Verdict
    for a Claim, THEN THE Groundedness_Runner SHALL halt before writing
    the Groundedness_Report, and SHALL produce an error message
    identifying that the Judge_Model failed to load, or identifying the
    failing Claim's `query_id` and `claim_index`, as applicable.

### Requirement 7: Quarantine Decision And Recording

**User Story:** As a researcher, I want claims judged unsupported
withheld from the served answer and explicitly recorded, so that no
claim is silently dropped.

#### Acceptance Criteria

1. IF a Claim's Groundedness_Verdict is `NOT_SUPPORTED`, THEN THE
   Quarantine_Rule SHALL determine that Claim's Quarantine_Decision to
   be true, regardless of that Claim's score.
2. IF a Claim's Groundedness_Verdict is `SUPPORTED` and that Claim's
   score is strictly below the threshold declared in the
   Groundedness_Config, THEN THE Quarantine_Rule SHALL determine that
   Claim's Quarantine_Decision to be true.
3. IF a Claim's Groundedness_Verdict is `SUPPORTED` and that Claim's
   score is greater than or equal to the threshold declared in the
   Groundedness_Config, THEN THE Quarantine_Rule SHALL determine that
   Claim's Quarantine_Decision to be false.
4. THE Quarantine_Rule SHALL be a pure, deterministic function of a
   Claim's Groundedness_Verdict, score, and the declared threshold,
   taking no other input, such that the same (Groundedness_Verdict,
   score, threshold) tuple always produces the same Quarantine_Decision.
5. IF a Claim's Quarantine_Decision is true, THEN THE
   Groundedness_Runner SHALL withhold that Claim from the
   Served_Answer.
6. THE Groundedness_Runner SHALL record every Claim's
   Quarantine_Decision in the Groundedness_Report, regardless of
   whether that Claim was withheld.

### Requirement 8: Groundedness_Report Schema And Derivable Quarantine_Rate

**User Story:** As a researcher, I want one row per claim with the
judge's verdict, score, and quarantine outcome, so that the quarantine
rate is always derivable from a committed artifact rather than a
separate hard-coded number.

#### Acceptance Criteria

1. WHEN the Groundedness_Runner has produced a Quarantine_Decision for
   every Claim in the Generation_Subset, THE Groundedness_Runner SHALL
   write the Groundedness_Report to `results/groundedness.csv`, with
   exactly one row per (`query_id`, `claim_index`) pair produced for
   the Generation_Subset.
2. THE Groundedness_Runner SHALL include, in every Groundedness_Report
   row, the columns `query_id`, `claim_index`, `claim_text`,
   `groundedness_verdict`, `judge_score`, and `quarantine_decision`,
   holding respectively that row's Claim's `query_id`, its
   `claim_index`, its claim text, its Groundedness_Verdict, its
   Judge_Model score, and its Quarantine_Decision.
3. THE Groundedness_Runner SHALL NOT write a Quarantine_Rate value to
   any file as a separately stored literal, so that Quarantine_Rate
   remains a number derivable by aggregation over the
   Groundedness_Report's `quarantine_decision` column rather than a
   hard-coded aggregate.
4. THE Groundedness_Runner SHALL retain every Claim's row in the
   Groundedness_Report regardless of that Claim's Groundedness_Verdict
   or Quarantine_Decision, and SHALL NOT drop, exclude, or filter any
   produced Claim's row from the Groundedness_Report.
5. IF writing the Groundedness_Report fails for any reason, THEN THE
   Groundedness_Runner SHALL terminate with a non-zero exit status and
   SHALL leave `results/groundedness.csv` either absent or
   byte-for-byte in its pre-run state, never partially written.

### Requirement 9: Non-Destructive Config/Seed Traceability Merge Into run_config.json

**User Story:** As a researcher, I want the declared subset size,
seeds, and model identifiers merged non-destructively into the existing
run configuration record, so that this feature's parameters are
recorded exactly like the significance-testing spec's established
precedent.

#### Acceptance Criteria

1. WHEN the Groundedness_Runner completes writing the
   Groundedness_Report, THE Groundedness_Runner SHALL merge a new
   sibling key into `results/run_config.json`, recording the
   Replayed_Run's `run_id`, the replay top-k, the Generation_Subset
   size, the Generation_Subset_Seed, the Generator_Model identifier,
   the Judge_Model identifier, the Quarantine_Rule's threshold, the
   Hand_Checked_Sample size, and the Hand_Checked_Sample_Seed; WHERE
   that sibling key is already present in `results/run_config.json`
   from an earlier run of the Groundedness_Runner, THE
   Groundedness_Runner SHALL overwrite that earlier sibling key's value
   with the current run's recorded values, rather than raising an
   error or creating a second sibling key.
2. THE Groundedness_Runner SHALL preserve every key already present in
   `results/run_config.json` — including `seed`, `sweep_config`,
   `corpus_load_report`, `installed_versions`, and `significance` —
   unchanged, removing or altering none of them.
3. THE Groundedness_Runner SHALL derive every value merged under
   Criterion 1 from the Groundedness_Config value actually applied
   during the run, rather than from a literal written independently of
   the applied value.
4. IF `results/run_config.json` is absent or cannot be parsed, THEN THE
   Groundedness_Runner SHALL halt before writing the
   Groundedness_Report, SHALL terminate with a non-zero exit status,
   SHALL produce an error message stating that the existing run
   configuration record is missing or unreadable, and SHALL NOT create
   a fresh record in place of the missing one.
5. IF merging the new sibling key into `results/run_config.json` fails
   for any reason, THEN THE Groundedness_Runner SHALL terminate with a
   non-zero exit status and SHALL leave `results/run_config.json`
   byte-for-byte in its pre-run state, never partially written.

### Requirement 10: Hand_Checked_Sample Selection Independent Of Judge Verdicts, And Agreement_Rate

**User Story:** As a researcher, I want a fixed number of claims
selected independently of the judge's verdicts and exported for manual
labelling, so that the agreement rate is not biased by which claims the
judge already favored.

#### Acceptance Criteria

1. WHEN Claims have been produced for the Generation_Subset, THE
   Groundedness_Runner SHALL select the Hand_Checked_Sample by drawing,
   uniformly at random and without replacement, the number of Claims
   declared as the Hand_Checked_Sample size in the Groundedness_Config,
   seeding that draw with the Hand_Checked_Sample_Seed.
2. THE Groundedness_Runner SHALL compute the Hand_Checked_Sample's
   selection from only each Claim's identity (`query_id` and
   `claim_index`), a canonical ordering of Claims sorted first by
   `query_id` and then by `claim_index`, and the
   Hand_Checked_Sample_Seed, and SHALL NOT use any Claim's
   Groundedness_Verdict, score, or Quarantine_Decision as an input to
   that selection.
3. IF the declared Hand_Checked_Sample size exceeds the total number of
   Claims produced for the Generation_Subset, THEN THE
   Groundedness_Runner SHALL select every produced Claim, rather than
   raising an error.
4. WHEN the Hand_Checked_Sample is selected, THE Groundedness_Runner
   SHALL export that sample for manual labelling, including each
   selected Claim's `query_id`, `claim_index`, and claim text, and a
   blank field for the Hand_Label.
5. WHILE a Hand_Label_Import file is present, contains a row
   corresponding to every Claim in the Hand_Checked_Sample, and every
   one of those rows carries a non-blank Hand_Label (a Hand_Label is
   non-blank if it is neither an empty string nor a string containing
   only whitespace), THE Groundedness_Runner SHALL compute
   Agreement_Rate as the fraction of Hand_Checked_Sample Claims whose
   Judge_Model Groundedness_Verdict matches the corresponding
   Hand_Label.
6. WHILE the Hand_Label_Import file is absent, is missing a row for any
   Claim in the Hand_Checked_Sample, or contains a blank Hand_Label (an
   empty string or a string containing only whitespace) for any of its
   rows, THE Groundedness_Runner SHALL NOT compute Agreement_Rate, and
   SHALL leave the exported Hand_Checked_Sample file available for
   manual labelling.
7. IF a previously exported Hand_Checked_Sample file already exists and
   contains a non-blank Hand_Label for one or more of its rows, THEN
   THE Groundedness_Runner SHALL NOT overwrite that file, and SHALL
   leave the existing Hand_Label values unmodified.
8. THE Hand_Checked_Sample export required by Criterion 4 SHALL NOT
   include, for any selected Claim, that Claim's Groundedness_Verdict,
   Judge_Model score, or Quarantine_Decision, so that the independence
   Requirement 10.2 establishes for the Hand_Checked_Sample's selection
   also holds for its labelling — a human reviewer assigning a
   Hand_Label must not be able to see, or be anchored by, the
   Judge_Model's own determination for that Claim.

### Requirement 11: Traceable Account Of How Far The Quarantine Rate Can Be Trusted

**User Story:** As a researcher, I want an explicit, artifact-traceable
account of how far the quarantine rate can be trusted, so that no one
mistakes it for an unqualified accuracy measure.

#### Acceptance Criteria

1. THE Groundedness_Runner's documentation — text describing the
   groundedness-gate feature, committed to the repository, and
   readable without executing any code — SHALL state that
   Quarantine_Rate reflects support against Retrieved_Context only,
   and SHALL state that a Claim can be judged supported while its
   Retrieved_Context is itself not relevant to the query under the
   Qrels, so that groundedness is documented as distinct from
   retrieval relevance.
2. THE documentation required by Criterion 1 SHALL state that the
   Claim_Segmenter's sentence-boundary heuristic, the
   Generator_Model/Judge_Model separation, and the
   Retrieved_Context-versus-Qrels distinction are each a limitation on
   what Quarantine_Rate measures.
3. THE documentation required by Criterion 1 SHALL state that
   Agreement_Rate is part of how far Quarantine_Rate can be trusted,
   and SHALL NOT state an Agreement_Rate value in any section of that
   documentation that omits the limitations stated under Criteria 1
   and 2.
4. IF a Downstream_Writeup states a Quarantine_Rate or Agreement_Rate
   value, THEN THE Downstream_Writeup SHALL derive that value from
   `results/groundedness.csv` or the Hand_Label_Import file, and SHALL
   NOT state either value from any other source.
5. THE documentation required by Criterion 1 SHALL state that
   Quarantine_Rate is model-graded and has no human ground truth, in
   explicit contrast to recall@k, nDCG@10, and MRR@10, each of which is
   computed against the human relevance judgments (Qrels) shipped with
   BEIR SciFact, and SHALL state that the Hand_Checked_Sample's
   Agreement_Rate is the only human anchor Quarantine_Rate has.
6. IF a Downstream_Writeup states a Quarantine_Rate value, THEN THE
   Downstream_Writeup SHALL state, in the same location as that value,
   the Agreement_Rate and the model-graded/no-human-ground-truth
   statement required by Criterion 5, and SHALL NOT state a
   Quarantine_Rate value in any location that omits both.

### Requirement 12: Test Coverage Scope Limited To The Claim_Segmenter And Quarantine_Rule Functions

**User Story:** As a maintainer, I want pytest coverage over claim
segmentation and the quarantine decision rule using hand-built inputs,
so that the pure logic is verified without loading any model, loading
any corpus, or making any network call.

#### Acceptance Criteria

1. THE test suite SHALL include pytest tests for the Claim_Segmenter's
   sentence-boundary splitting function and for the Quarantine_Rule
   function, each covering hand-built input strings or hand-built
   (Groundedness_Verdict, score, threshold) tuples with independently
   reasoned expected outputs.
2. THE test suite covering the Claim_Segmenter and the Quarantine_Rule
   SHALL execute without loading the Generator_Model, without loading
   the Judge_Model, without loading any corpus, and without making any
   network call.
3. THE test suite SHALL NOT import or invoke the Groundedness_Runner
   entry point, the Retrieval_Replay component, `src.corpus_loader`, or
   any retriever module, and SHALL defer Groundedness_Runner end-to-end
   tests and Retrieval_Replay tests over the real corpus to a later
   spec.
4. THE test suite SHALL cover, for the Quarantine_Rule, at minimum the
   following four cases: (a) a Groundedness_Verdict of `SUPPORTED`
   paired with a score numerically equal to the declared threshold, (b)
   a Groundedness_Verdict of `SUPPORTED` paired with a score above the
   declared threshold, (c) a Groundedness_Verdict of `SUPPORTED` paired
   with a score below the declared threshold, and (d) a
   Groundedness_Verdict of `NOT_SUPPORTED` paired with at least two
   distinct score values, one above and one below the declared
   threshold, asserting that the resulting Quarantine_Decision is the
   same for both of those two scores.
5. THE test suite SHALL cover, for the Claim_Segmenter, at minimum the
   following three cases: (a) a Generated_Answer containing multiple
   sentences, asserting that the returned ordered list of Claims
   contains one Claim per sentence and that each Claim's `claim_index`
   equals its 0-based position in that list, (b) a Generated_Answer
   containing exactly one sentence, asserting that the returned list
   contains exactly one Claim, and (c) a Generated_Answer containing no
   sentence-ending punctuation, asserting that the returned list
   contains exactly one Claim whose text is the entire Generated_Answer.
