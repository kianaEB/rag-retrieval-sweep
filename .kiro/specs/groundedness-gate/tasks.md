# Implementation Plan

## Overview

Ordering rule: the design's module layout puts two dependency-light
pure functions — `segment_claims` (`src/claim_segmenter.py`) and
`decide_quarantine` (`src/quarantine_rule.py`) — at the bottom of the
dependency graph, alongside the hard-coded label constant
(`src/groundedness_labels.py`) and the extended exception hierarchy
(`src/errors.py`); none of the four imports `transformers`, `torch`,
`src.corpus_loader`, or any retriever module. Those come first. Task 5
(Requirement 12 — the entire automated-test surface of this spec) must
be green before Task 13 wires the orchestrating entry point that
consumes the tested functions, mirroring the significance-testing
spec's own Task 2/Task 3/Task 7 precedent exactly. `configs/groundedness.yaml`
+ `src/groundedness_config.py`, `src/retrieval_replay.py` (read-only
reuse of session 1's frozen retriever, config loader, and corpus
loader), and the two model wrappers `src/generator_model.py` /
`src/judge_model.py` are independent of each other — each depends only
on `src/errors.py` (and, for `judge_model.py` and `groundedness_config.py`,
`src/groundedness_labels.py`) — so they proceed in the same wave.
`src/groundedness_report.py` and `src/hand_checked_sample.py` likewise
depend only on `src/errors.py` and the already-existing
`src.report._atomic_write_text`. The extension to
`src/verify_writeup_numbers.py` (Task 12) has no dependency on any
other module introduced in this spec — it only touches the existing,
already-committed repo-writeup module — so it is scheduled as an early,
independent sibling task, mirroring how the repo-writeup spec's own
Task 4 (authoring that module) ran in parallel with unrelated early
tasks. `src/groundedness_runner.py`'s `main()` (Task 13) is the single
module that imports everything above and is therefore last in the
code-writing sequence. Per the design's Testing Strategy and Error
Handling sections, the `Groundedness_Runner` entry point,
`Retrieval_Replay` against the real corpus, `groundedness_config.py`'s
validation, and `hand_checked_sample.py`'s selection/export/join logic
are **not** covered by an automated test in this spec (Requirement
12.3 scopes tests to the Claim_Segmenter and Quarantine_Rule functions
only) — they are verified structurally by code shape and, for the
entry point's happy path, by one real end-to-end run (Task 14),
matching session-1's and significance-testing's own precedent of
deferring entry-point and real-corpus tests to manual verification.
The two documentation tasks (15, 16) mirror the repo-writeup spec's own
Task 9/Task 10 split exactly — author `SPEC.md`'s new section and the
ledger rows first, then re-run the Verification_Pass — and Task 15 can
only be completed once a human has hand-labelled
`results/hand_checked_sample.csv` and a rerun of the
`Groundedness_Runner` has produced `results/hand_checked_joined.csv`,
since `Agreement_Rate` has no other source. All done checks are Git
Bash / POSIX shell or Python one-liners per shell-conventions.md.

## Tasks

- [x] 1. Extend `src/errors.py` with the groundedness-gate exception types
  - Add, alongside the existing session-1 and significance-testing
    types: `GroundednessConfigError(ConfigError)`,
    `LabelMappingMismatchError(GroundednessConfigError)`,
    `GenerationSubsetInputError(Exception)`,
    `ReplayedRunNotFoundError(Exception)`,
    `FrozenRetrieverConfigError(Exception)`,
    `RetrievalReplayError(Exception)`,
    `GeneratorModelLoadError(Exception)`,
    `GeneratorGenerationError(Exception)`,
    `JudgeModelLoadError(Exception)`, `JudgeVerdictError(Exception)`,
    `GroundednessReportWriteError(Exception)`,
    `HandCheckedSampleWriteError(Exception)`,
    `HandCheckedJoinedWriteError(Exception)` — matching the design's
    docstrings exactly. Do NOT add a `SpecSectionWriteError` or any
    other type for an automated `SPEC.md`-writing step; none exists in
    the design. `RunConfigMergeError` already exists from the
    significance-testing spec and is reused verbatim — do not
    redefine it.
  - Done check:
    `python -c "from src.errors import ConfigError, GroundednessConfigError, LabelMappingMismatchError, GenerationSubsetInputError, ReplayedRunNotFoundError, FrozenRetrieverConfigError, RetrievalReplayError, GeneratorModelLoadError, GeneratorGenerationError, JudgeModelLoadError, JudgeVerdictError, GroundednessReportWriteError, HandCheckedSampleWriteError, HandCheckedJoinedWriteError, RunConfigMergeError; assert issubclass(GroundednessConfigError, ConfigError); assert issubclass(LabelMappingMismatchError, GroundednessConfigError); [cls('x') for cls in (GroundednessConfigError, LabelMappingMismatchError, GenerationSubsetInputError, ReplayedRunNotFoundError, FrozenRetrieverConfigError, RetrievalReplayError, GeneratorModelLoadError, GeneratorGenerationError, JudgeModelLoadError, JudgeVerdictError, GroundednessReportWriteError, HandCheckedSampleWriteError, HandCheckedJoinedWriteError)]; print('ok')"`
    prints `ok` with exit code 0. Additionally,
    `! grep -q 'SpecSectionWriteError' src/errors.py && echo ok` prints
    `ok`, confirming no automated-`SPEC.md`-writing exception type was
    introduced.
  - _Requirements: 1.1, 1.5, 2.1, 2.5, 3.6, 3.7, 4.7, 4.8, 6.3, 6.10, 6.12, 8.5, 10.4, 10.5_

- [x] 2. Write `src/groundedness_labels.py`
  - Implement the `Verdict` `Literal["SUPPORTED", "NOT_SUPPORTED"]`
    type alias, the hard-coded `NLI_LABEL_TO_VERDICT: Dict[str, Verdict]`
    constant (`entailment` -> `SUPPORTED`, `neutral` -> `NOT_SUPPORTED`,
    `contradiction` -> `NOT_SUPPORTED`), and `ENTAILMENT_LABEL =
    "entailment"`. Standard library only — no `transformers` import —
    so this module stays importable without pulling in any model
    dependency.
  - Done check:
    `python -c "from src.groundedness_labels import NLI_LABEL_TO_VERDICT, ENTAILMENT_LABEL; assert NLI_LABEL_TO_VERDICT == {'entailment': 'SUPPORTED', 'neutral': 'NOT_SUPPORTED', 'contradiction': 'NOT_SUPPORTED'}; assert ENTAILMENT_LABEL == 'entailment'; assert NLI_LABEL_TO_VERDICT[ENTAILMENT_LABEL] == 'SUPPORTED'; print('ok')"`
    prints `ok` and exits 0. No network call.
  - _Requirements: 6.2, 6.3, 6.9, 6.10_

- [x] 3. Write `src/claim_segmenter.py`
  - Implement the frozen `Claim` dataclass (`claim_index: int`,
    `text: str`) and `segment_claims(generated_answer: str) ->
    List[Claim]`: a sentence boundary is any occurrence of `.`, `!`,
    or `?` followed by whitespace or end-of-string
    (`_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s+|$)")`); if the
    whitespace-trimmed input contains no such boundary, return a
    single `Claim(claim_index=0, text=trimmed)` (covers the empty
    string too); otherwise scan matches via `finditer`, slicing out
    each segment (terminating punctuation included), trimming it, and
    appending a `Claim` only when the trimmed segment is non-empty, so
    `claim_index` values are always a contiguous `0..n-1` range. Pure
    function: no model load, no file I/O, no network call. Docstring
    states the sentence-boundary heuristic is crude, not a solved NLP
    problem, and that a mis-split sentence is a source of measurement
    error in what counts as one Claim.
  - Done check:
    `python -c "from src.claim_segmenter import segment_claims; c = segment_claims('BM25 is strong. Dense underperforms here.'); assert [x.text for x in c] == ['BM25 is strong.', 'Dense underperforms here.']; assert [x.claim_index for x in c] == [0, 1]; c2 = segment_claims('One sentence only.'); assert len(c2) == 1 and c2[0].claim_index == 0; c3 = segment_claims('No terminal punctuation'); assert len(c3) == 1 and c3[0].text == 'No terminal punctuation'; c4 = segment_claims('   '); assert len(c4) == 1 and c4[0].text == '' and c4[0].claim_index == 0; print('ok')"`
    prints `ok` and exits 0. No network call. (Task 5's pytest suite is
    the authoritative Requirement 12 verification; this is a quick
    sanity check.)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 4. Write `src/quarantine_rule.py`
  - Implement `decide_quarantine(verdict: Verdict, score: float,
    threshold: float) -> bool`: `if verdict == "NOT_SUPPORTED": return
    True` (unconditional on `score`), otherwise `return score <
    threshold` — a strict `<` comparison, so `score == threshold`
    lands in the "not quarantined" branch. Pure, deterministic,
    three-parameter function; no corpus, no model, no file I/O. Import
    `Verdict` from `src.groundedness_labels` for the type hint only.
  - Done check:
    `python -c "from src.quarantine_rule import decide_quarantine; assert decide_quarantine('SUPPORTED', 0.5, 0.5) is False; assert decide_quarantine('SUPPORTED', 0.9, 0.5) is False; assert decide_quarantine('SUPPORTED', 0.1, 0.5) is True; assert decide_quarantine('NOT_SUPPORTED', 0.9, 0.5) is True; assert decide_quarantine('NOT_SUPPORTED', 0.1, 0.5) is True; print('ok')"`
    prints `ok` and exits 0. No network call. (Task 5's pytest suite is
    the authoritative Requirement 12 verification; this is a quick
    sanity check.)
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 5. Write `tests/test_claim_segmenter.py` and `tests/test_quarantine_rule.py` (Requirement 12 — the entire automated-test surface of this spec)
  - In `tests/test_claim_segmenter.py`, implement the three required
    cases against `segment_claims`/`Claim` from
    `src.claim_segmenter` only: (a) a multi-sentence answer, asserting
    one Claim per sentence with `claim_index` equal to 0-based
    position; (b) a single-sentence answer, asserting exactly one
    Claim; (c) an answer with no terminal punctuation, asserting
    exactly one Claim whose text is the entire input (Requirement
    12.5).
  - In `tests/test_quarantine_rule.py`, implement the four required
    cases against `decide_quarantine` from `src.quarantine_rule` (plus
    `Verdict` from `src.groundedness_labels` for literal construction
    only): (a) `SUPPORTED` at `score == threshold` -> not quarantined;
    (b) `SUPPORTED` above threshold -> not quarantined; (c)
    `SUPPORTED` below threshold -> quarantined; (d) `NOT_SUPPORTED` at
    two distinct scores, one above and one below threshold, both
    quarantined (Requirement 12.4).
  - Neither test module imports or invokes the `Groundedness_Runner`
    entry point, `src.retrieval_replay`, `src.generator_model`,
    `src.judge_model`, `src.corpus_loader`, or any retriever module
    (Requirement 12.3); neither loads a model or a corpus, and neither
    makes a network call (Requirement 12.2).
  - Done check: `pytest tests/test_claim_segmenter.py tests/test_quarantine_rule.py -v`
    reports all tests passed, and
    `! grep -Eq 'beir|sentence_transformers|huggingface|transformers|torch|corpus_loader|sweep_runner|groundedness_runner|retrieval_replay|generator_model|judge_model' tests/test_claim_segmenter.py tests/test_quarantine_rule.py && echo ok`
    prints `ok` (the `!` asserts no forbidden-import match). This must
    be green before Task 13 wires the entry point that consumes these
    two functions.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 6. Write `configs/groundedness.yaml` and `src/groundedness_config.py`
  - Implement the frozen `GroundednessConfig` dataclass with all 18
    fields from the design (`replayed_run_id`, `replay_top_k`,
    `generation_subset_size`, `generation_subset_seed`,
    `generator_model_name`, `judge_model_name`, `prompt_template`,
    `quarantine_threshold`, `hand_checked_sample_size`,
    `hand_checked_sample_seed`, `label_mapping`, `score_definition`,
    `sweep_config_path`, `per_query_path`, `run_config_path`,
    `output_path`, `hand_checked_sample_path`,
    `hand_checked_joined_path`) and `load_groundedness_config(path) ->
    GroundednessConfig`, raising `GroundednessConfigError` naming the
    missing/invalid/conflicting field: any Criterion-1 declaration
    missing; `generator_model_name == judge_model_name`; `replay_top_k`
    / `generation_subset_size` / `hand_checked_sample_size` not a
    positive integer; `generation_subset_seed` /
    `hand_checked_sample_seed` not an integer; `quarantine_threshold`
    not numeric. Never partially applies a config.
  - Implement `_validate_label_mapping(declared, context)`, called
    during `load_groundedness_config`, raising
    `LabelMappingMismatchError` (a `GroundednessConfigError` subclass)
    unless the YAML's `label_mapping` dict is exactly equal to
    `src.groundedness_labels.NLI_LABEL_TO_VERDICT`, and unless
    `score_definition` equals the fixed expected text — both checked
    before any Generation_Subset query is processed. Import only
    `PyYAML`, the standard library, and `src.groundedness_labels`.
  - Author `configs/groundedness.yaml` per the design schema exactly:
    `replayed_run_id: bm25__whole_document`, `replay_top_k: 10`,
    `generation_subset_size: 30`, `generation_subset_seed: 4242`,
    `hand_checked_sample_size: 50`, `hand_checked_sample_seed: 777`,
    `generator_model_name: google/flan-t5-base`,
    `judge_model_name: cross-encoder/nli-deberta-v3-xsmall`, the
    `prompt_template` combining `{query}` and `{context}`,
    `quarantine_threshold: 0.5`, the `label_mapping` and
    `score_definition` records matching `NLI_LABEL_TO_VERDICT`, and
    `sweep_config_path`/`per_query_path`/`run_config_path`/`output_path`/
    `hand_checked_sample_path`/`hand_checked_joined_path` all under
    `configs/`/`results/`.
  - Done check: a script that calls
    `load_groundedness_config(Path("configs/groundedness.yaml"))` and
    asserts `replay_top_k == 10`, `generation_subset_size == 30`,
    `generation_subset_seed == 4242`, `hand_checked_sample_seed == 777`,
    `generation_subset_seed != hand_checked_sample_seed`,
    `generator_model_name != judge_model_name`,
    `quarantine_threshold == 0.5`; then asserts a copy of the YAML with
    `generator_model_name` set equal to `judge_model_name` raises
    `GroundednessConfigError`; a copy with `label_mapping.neutral` set
    to `SUPPORTED` raises `LabelMappingMismatchError`; and a copy
    omitting `generation_subset_seed` raises `GroundednessConfigError`.
    Prints `ok` and exits 0. No network call.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 6.3, 6.10_

- [x] 7. Write `src/retrieval_replay.py`
  - Implement `_retriever_name_from_run_id(run_id)` (split on the
    first `"__"`), `load_frozen_retriever_config(sweep_config_path,
    replayed_run_id)` (loads `configs/sweep.yaml` via
    `load_sweep_config`, returns the one retriever config entry whose
    `name` matches the run_id's prefix, raising
    `FrozenRetrieverConfigError` if the sweep config fails to load or
    no name matches), `build_frozen_retriever(sweep_config,
    retriever_config)` (constructs the ONE matched retriever type via
    `configure_caches` + `load_scifact`, builds its index exactly
    once, raising `FrozenRetrieverConfigError` on corpus load failure
    or `RetrievalReplayError` on index-build failure), and
    `replay_retrieval(retriever, bundle, subset_query_ids, queries,
    replay_top_k)` (pre-filters `queries` to `subset_query_ids` before
    issuing exactly ONE `retrieve_all` call, then maps each query's
    ranked document IDs to `Retrieved_Context` via
    `format_document_text`, raising `RetrievalReplayError` on
    failure). Reuses `src.config`, `src.corpus_loader`,
    `src.retrievers.base`/`bm25_retriever`/`dense_retriever` exactly as
    session 1 already defines them — no retrieval logic changes, no
    `configs/sweep.yaml` schema change.
  - Done check: a two-part script. Part (a), no network/corpus touch:
    `load_frozen_retriever_config(Path("configs/sweep.yaml"),
    "bm25__whole_document")` returns a `BM25RetrieverConfig` whose
    `name == "bm25"`. Part (b), against the already-cached SciFact
    corpus under `data/` (no new download expected for the BM25
    replay target): `build_frozen_retriever` builds the BM25 index
    once, then `replay_retrieval` is called once with a 2-query subset
    at `replay_top_k=3`, asserting the returned dict has exactly 2
    keys and every value is a list of at most 3 document-text strings.
    Prints `ok` and exits 0.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 8. Write `src/generator_model.py`
  - Implement `GeneratorModel(model_name, cache_folder)`: loads via
    `transformers.AutoTokenizer.from_pretrained` +
    `transformers.AutoModelForSeq2SeqLM.from_pretrained` (T5 is
    encoder-decoder, not causal), `.to("cpu")` hard-coded, wrapping any
    load failure in `GeneratorModelLoadError`. Implement
    `generate(prompt) -> str`: tokenizes with `truncation=True`, calls
    `self._model.generate(**inputs, do_sample=False, num_beams=1)`
    (greedy decoding — deterministic by construction, no sampling RNG),
    decodes with `skip_special_tokens=True`, wrapping any failure in
    `GeneratorGenerationError` (raised without `query_id` context; the
    Groundedness_Runner attaches that context, per Requirement 4.8).
  - Done check: a script that constructs
    `GeneratorModel("google/flan-t5-base", Path("data/hf_cache"))`,
    calls `generate("Question: What is BM25?\nContext:\nBM25 is a
    ranking function.\nAnswer:")` twice, and asserts the two returned
    strings are byte-for-byte identical (Requirement 4.4's rerun
    guarantee) and each is a non-empty `str`. Prints `ok` and exits 0.
    This is the first task in this spec's sequence that touches the
    network for a model download (one-time `google/flan-t5-base`
    weight download to `data/hf_cache`).
  - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 9. Write `src/judge_model.py`
  - Implement the frozen `JudgeResult` dataclass (`verdict: Verdict`,
    `score: float`) and `JudgeModel(model_name, cache_folder)`: loads
    via `transformers.AutoTokenizer.from_pretrained` +
    `transformers.AutoModelForSequenceClassification.from_pretrained`
    (not `sentence_transformers.CrossEncoder`), `.to("cpu")` +
    `.eval()`, builds `label2idx` from the loaded model's own
    `model.config.id2label` (lower-cased), raising
    `JudgeModelLoadError` if `ENTAILMENT_LABEL` is not among the
    exposed labels or if loading fails. Implement `judge(premise,
    hypothesis) -> JudgeResult`: tokenizes as a premise/hypothesis
    pair with truncation, computes softmax over the 3 logits under
    `torch.no_grad()`, reads `predicted_label` via `argmax` over the
    logits, maps it to a `Verdict` via `NLI_LABEL_TO_VERDICT`, and sets
    `score` to the entailment class's softmax probability read via
    `label2idx[ENTAILMENT_LABEL]`; wraps any failure in
    `JudgeVerdictError` (raised without `query_id`/`claim_index`
    context; the Groundedness_Runner attaches that context, per
    Requirement 6.12).
  - Done check: a script that constructs
    `JudgeModel("cross-encoder/nli-deberta-v3-xsmall",
    Path("data/hf_cache"))`, calls
    `judge("BM25 is a ranking function used for lexical retrieval.",
    "BM25 is a ranking function.")` and asserts `verdict == "SUPPORTED"`
    and `0.0 <= score <= 1.0`; calls
    `judge("The sky is blue.", "BM25 outperforms every dense retriever
    on every corpus.")` and asserts `verdict == "NOT_SUPPORTED"`.
    Prints `ok` and exits 0. This is the second network-touching task
    in this spec's sequence (one-time
    `cross-encoder/nli-deberta-v3-xsmall` weight download to
    `data/hf_cache`).
  - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.7, 6.8, 6.9, 6.12_

- [x] 10. Write `src/groundedness_report.py`
  - Implement the frozen `GroundednessReportRow` dataclass
    (`query_id`, `claim_index`, `claim_text`, `groundedness_verdict`,
    `judge_score`, `quarantine_decision`, in that fixed column order —
    Requirement 8.2) and `write_groundedness_report(rows,
    output_path)`, reusing `src.report._atomic_write_text` (temp file
    + `os.replace`, removed on failure) so `output_path` is left
    absent or byte-for-byte in its pre-run state on any failure. Never
    filters `rows` regardless of verdict or quarantine_decision.
    Raises `GroundednessReportWriteError` on failure. No
    `Quarantine_Rate` field anywhere in this schema.
  - Done check: a script that writes a handful of hand-built
    `GroundednessReportRow`s (mixing `SUPPORTED`/`NOT_SUPPORTED` and
    `True`/`False` quarantine decisions, including at least one
    `judge_score == 0.0` and one `== 1.0`) to a temp CSV, reads it back
    with `pandas`, and asserts: exactly those rows, in the given order;
    the columns are exactly the 6 fields in the fixed order; every
    `judge_score` reads back as a float in `[0.0, 1.0]`; every
    `quarantine_decision` reads back as a boolean; no row was dropped
    regardless of its verdict. Prints `ok` and exits 0. No network
    call.
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 11. Write `src/hand_checked_sample.py`
  - Implement `select_hand_checked_sample(claim_ids: List[Tuple[str,
    int]], sample_size, seed) -> List[Tuple[str, int]]`: sorts
    `claim_ids` into canonical order, then
    `random.Random(seed).sample(canonical_order, min(sample_size,
    len(canonical_order)))`. The function's parameter list is exactly
    `(claim_ids, sample_size, seed)` — no verdict/score/quarantine
    parameter exists, structurally excluding that information from
    selection (Requirement 10.2).
  - Implement the frozen `HandCheckedSampleRow` dataclass with exactly
    4 fields (`query_id`, `claim_index`, `claim_text`, `hand_label`) —
    no `groundedness_verdict`/`judge_score`/`quarantine_decision`
    field exists on this dataclass (Requirement 10.8) — and
    `export_hand_checked_sample(rows, output_path)`: if `output_path`
    already exists and any row's `hand_label` is non-blank (neither
    empty nor whitespace-only), returns without writing, leaving the
    file untouched (Requirement 10.7); otherwise writes atomically via
    `_atomic_write_text`, raising `HandCheckedSampleWriteError` on
    failure.
  - Implement `read_hand_label_import(path, expected_claim_ids) ->
    Optional[Dict[Tuple[str, int], str]]`: returns the
    `{(query_id, claim_index): hand_label}` mapping only if `path`
    exists, covers every one of `expected_claim_ids`, and every
    covered row's `hand_label` is non-blank; otherwise returns `None`.
    Implement `compute_agreement_rate(judge_verdicts, hand_labels) ->
    float`: fraction of `hand_labels`' keys where
    `judge_verdicts[key] == hand_labels[key]`, `0.0` if `hand_labels`
    is empty.
  - Implement the frozen `HandCheckedJoinedRow` dataclass (`query_id`,
    `claim_index`, `judge_verdict`, `hand_label`),
    `join_hand_labels_with_verdicts(judge_verdicts, hand_labels) ->
    List[HandCheckedJoinedRow]` (one row per `hand_labels` key, in
    canonical sorted order), and `write_hand_checked_joined(rows,
    output_path)` (atomic write via `_atomic_write_text`, safe to
    overwrite unconditionally since this file is always fully derived
    and never hand-edited; raises `HandCheckedJoinedWriteError` on
    failure).
  - Done check: a script that (a) calls
    `select_hand_checked_sample` twice with the same `claim_ids` and
    `seed` and asserts identical results, and with a `sample_size`
    exceeding `len(claim_ids)` asserts every claim ID is returned; (b)
    exports a couple of hand-built `HandCheckedSampleRow`s to a temp
    CSV via `export_hand_checked_sample`, calls it a second time with a
    modified `rows` argument, and asserts the file on disk is
    unchanged after that second call once one row's `hand_label` is
    manually set non-blank between the two calls; (c) calls
    `read_hand_label_import` against a fixture file missing one
    expected claim ID and asserts it returns `None`, then against a
    complete fixture and asserts it returns the expected dict; (d)
    calls `compute_agreement_rate` on hand-built dicts with a known
    match fraction and asserts the returned float matches; (e) calls
    `join_hand_labels_with_verdicts` + `write_hand_checked_joined` and
    reads the result back with `pandas`, asserting both `judge_verdict`
    and `hand_label` columns are present. Prints `ok` and exits 0. No
    network call.
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

- [x] 12. Extend `src/verify_writeup_numbers.py`'s artifact resolution for `groundedness.csv` and `hand_checked_joined.csv`
  - Add `"groundedness.csv"` and `"hand_checked_joined.csv"` to the
    existing `_CSV_ARTIFACTS` tuple (currently `("sweep.csv",
    "significance.csv", "per_query.csv")`) so
    `load_artifact_values`/`_resolve_csv_reference` resolve an ordinary
    `row_selector.field` reference against either new file without
    further change.
  - Extend `_resolve_csv_reference` with two additive branches: (i) if
    the row-selector portion of a reference is the literal string
    `all`, skip per-`key=value` filtering and match every row; (ii) if
    the field portion of a reference is the literal sentinel
    `__count__`, skip the "must match exactly 1 row" check and return
    `float(len(matched))` instead of a single cell's value. Both
    compose with the existing `row_selector.field` dot-split unchanged.
    No new `_ALLOWED_COMPUTATIONS` member is added — `Quarantine_Rate`
    resolves via the existing `ratio` computation over
    `["quarantine_decision=True.__count__", "all.__count__"]`.
  - Add one new resolution branch to `load_artifact_values`, checked
    before the row-selector/`__count__` path: if a reference contains
    the literal substring `==` (e.g. `judge_verdict==hand_label`),
    split into `col_a, col_b`; raise `VerificationSourceError` if
    either is not a column of the artifact or if the artifact has zero
    rows; otherwise resolve to
    `(frame[col_a].astype(str) == frame[col_b].astype(str)).mean()` as
    a single float over the whole file. No new `_ALLOWED_COMPUTATIONS`
    member is added here either — a ledger row for `Agreement_Rate`
    uses `computation=copy` since the reference resolution already
    produces the final ratio.
  - Done check: a script that, inside a `tempfile.TemporaryDirectory()`
    standing in for `artifacts_dir`, writes a tiny fixture
    `groundedness.csv` (a handful of rows with a `quarantine_decision`
    column containing both `True` and `False` values as strings) and a
    tiny fixture `hand_checked_joined.csv` (a handful of rows with
    `judge_verdict`/`hand_label` columns, some equal, some not);
    asserts `load_artifact_values("groundedness.csv",
    "quarantine_decision=True.__count__;all.__count__", tmp_dir)`
    returns `[float(count_of_true_rows), float(total_row_count)]`
    matching a hand count, and that
    `apply_computation("ratio", that_result)` equals the hand-computed
    quarantine rate; asserts `load_artifact_values("hand_checked_joined.csv",
    "judge_verdict==hand_label", tmp_dir)` returns a single-element
    list whose float equals a hand-computed match fraction; asserts a
    reference naming a nonexistent column after `==` raises
    `VerificationSourceError`. Prints `ok` and exits 0. No network
    call, and the real `docs/`/`results/` files are never touched.
  - _Requirements: 11.4, 11.6 (this spec); extends repo-writeup Requirements 12.1, 12.3, 12.4 per design.md's "Extending the shared Verification_Pass module"_

- [x] 13. Wire the `Groundedness_Runner` `main()` in `src/groundedness_runner.py`
  - Implement `build_prompt(template, query_text, retrieved_context)`:
    `template.format(query=query_text, context="\n\n".join(retrieved_context))`
    — the same `"\n\n"` join format `JudgeModel.judge`'s `premise`
    argument uses, so the Generator_Model and Judge_Model are always
    shown literally the same context block.
  - Implement the CLI entry point (`python -m src.groundedness_runner
    [--config PATH]`, default `configs/groundedness.yaml`) with the
    design's 11-step orchestration: (1) load config, halting on
    `GroundednessConfigError`/`LabelMappingMismatchError`; (2) read
    `results/per_query.csv`, determine the Replayed_Run's scored query
    IDs, halting on `GenerationSubsetInputError`/
    `ReplayedRunNotFoundError`; (3) sample the Generation_Subset via a
    seeded, canonical-order draw (sorted query IDs,
    `random.Random(generation_subset_seed).sample(...)`, capped at the
    identified set's size); (4) `load_frozen_retriever_config` +
    `build_frozen_retriever`, halting on `FrozenRetrieverConfigError`;
    (5) `replay_retrieval` over the whole subset in one call, halting
    on `RetrievalReplayError`; (6) construct `GeneratorModel` and
    `JudgeModel` with `cache_folder = sweep_config.data_dir /
    "hf_cache"`, halting on `GeneratorModelLoadError`/
    `JudgeModelLoadError`; (7) for each query_id in sorted order: build
    the prompt, `generator.generate(prompt)` (wrapping
    `GeneratorGenerationError` with that `query_id` attached),
    `segment_claims(...)`, then for each Claim `judge_model.judge(...)`
    (wrapping `JudgeVerdictError` with `query_id`/`claim_index`
    attached) and `decide_quarantine(...)`, appending one
    `GroundednessReportRow`; any exception halts before any report is
    written; (8) `write_groundedness_report`, halting on
    `GroundednessReportWriteError`; (9) `select_hand_checked_sample` +
    `export_hand_checked_sample` (a no-op if the file already carries
    hand labels), halting on `HandCheckedSampleWriteError`; (10)
    `read_hand_label_import` — if non-`None`, build and write the
    joined rows via `join_hand_labels_with_verdicts` +
    `write_hand_checked_joined`, halting on
    `HandCheckedJoinedWriteError`, and print `Agreement_Rate` to stdout
    purely informationally (never written to any file as a stored
    literal); (11) merge the `"groundedness"` sibling key into
    `results/run_config.json` (reusing the exact
    `dict(existing_record)` -> set key -> `json.dumps(...,
    default=_json_default)` -> `_atomic_write_text` pipeline
    `src/significance.py`'s merge already implements), halting on
    `RunConfigMergeError`. There is no step that writes into or
    modifies `SPEC.md` anywhere in this orchestration.
  - Done check: a Git Bash script, entirely inside a `mktemp -d` temp
    dir, that never touches the real `results/` directory. Part (a):
    `python -c "from src.groundedness_runner import main, build_prompt; print('ok')"`
    prints `ok` (importable with no network side effect at import
    time). Part (b): fabricate a `groundedness.yaml` copy in the temp
    dir omitting `generation_subset_seed`; run `python -m
    src.groundedness_runner --config "$TMPDIR/groundedness.yaml"` and
    assert a non-zero exit via `|| echo halted`, and assert no
    `groundedness.csv` was created anywhere under the temp dir. Part
    (c): fabricate a complete, otherwise-valid `groundedness.yaml`
    copy whose `per_query_path` points at a nonexistent file inside the
    temp dir; run the same command and assert a non-zero exit and no
    `groundedness.csv` created. `rm -rf "$TMPDIR"` afterward. Prints
    `ok` and exits 0. The real, model/corpus-driven happy path is
    verified by the manual real run in Task 14, consistent with
    Requirement 12.3 scoping automated tests to the Claim_Segmenter and
    Quarantine_Rule only.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 4.1, 4.2, 4.8, 6.12, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.4, 10.5, 10.6, 10.7, 10.8_

- [x] 14. Run the Groundedness_Runner end-to-end for real and commit the real artifacts
  - Run `python -m src.groundedness_runner --config
    configs/groundedness.yaml` against the real, already-cached
    SciFact corpus and the real, now-cached `google/flan-t5-base` /
    `cross-encoder/nli-deberta-v3-xsmall` weights under `data/hf_cache`
    (from Tasks 8 and 9). Inspect the printed exit code and the
    resulting `results/groundedness.csv`, `results/hand_checked_sample.csv`,
    and `results/run_config.json`.
  - Note: hand-labelling `results/hand_checked_sample.csv` is a
    separate, later manual step performed by a human, not part of this
    task — only after every sampled row's `hand_label` column is
    filled in with `SUPPORTED`/`NOT_SUPPORTED` does a subsequent rerun
    of the same command read back a complete Hand_Label_Import and
    produce `results/hand_checked_joined.csv`. That rerun, and the
    hand-labelling itself, happen after this task and before Task 15.
  - Done check: after the real run,
    `python -c "import pandas as pd; df = pd.read_csv('results/groundedness.csv'); assert len(df) > 0; assert set(df.columns) == {'query_id','claim_index','claim_text','groundedness_verdict','judge_score','quarantine_decision'}; assert df['groundedness_verdict'].isin(['SUPPORTED','NOT_SUPPORTED']).all(); assert df['judge_score'].between(0.0, 1.0).all(); assert df['quarantine_decision'].isin([True, False]).all(); print('ok')"`
    prints `ok`; and
    `python -c "import pandas as pd, json; s = pd.read_csv('results/hand_checked_sample.csv', dtype=str, keep_default_na=False); assert (s['hand_label'].str.strip() == '').all(); rc = json.load(open('results/run_config.json')); assert 'groundedness' in rc; assert set(['seed','sweep_config','corpus_load_report','installed_versions']).issubset(rc.keys()); g = rc['groundedness']; assert g['replayed_run_id'] == 'bm25__whole_document'; assert g['generation_subset_size'] == 30; print('ok')"`
    prints `ok`, confirming the export is blank (Requirement 10.4) and
    every pre-existing `run_config.json` key survived the merge
    (Requirement 9.2).
  - _Requirements: 3.1, 3.2, 4.4, 5.1, 5.2, 6.1, 6.2, 6.9, 7.1, 7.2, 7.3, 8.1, 8.2, 8.4, 8.5, 9.1, 9.2, 9.3, 10.1, 10.3, 10.4, 10.7_

- [ ] 15. Author `SPEC.md`'s "## Groundedness gate" section and extend `docs/numeric_traceability.csv`
  - Prerequisite: a human has hand-labelled every row of
    `results/hand_checked_sample.csv` (from Task 14) and a subsequent
    rerun of `python -m src.groundedness_runner --config
    configs/groundedness.yaml` has produced the real
    `results/hand_checked_joined.csv`, so `Agreement_Rate` has a real,
    committed source to cite. This task cannot be completed before
    that rerun has happened.
  - Hand-edit `SPEC.md` to add one new top-level section (`##
    Groundedness gate`) stating: the Generator_Model/Judge_Model
    separation and its rationale (Requirement 6.6, 6.11); the
    three-native-label-to-binary Groundedness_Verdict mapping, read
    from `configs/groundedness.yaml`'s `label_mapping` field
    (Requirement 6.2, 6.3); the `judge_score` definition and the
    `quarantine_threshold`, read from `configs/groundedness.yaml`'s
    `score_definition` and `quarantine_threshold` fields (Requirement
    6.9, 6.10, 7.2, 7.3); and Requirement 11's full trust account —
    Quarantine_Rate reflects support against Retrieved_Context only,
    distinct from retrieval relevance (11.1); the Claim_Segmenter
    heuristic, the generator/judge separation, and the
    Retrieved_Context-versus-Qrels distinction are each named as a
    limitation (11.2); Agreement_Rate is part of how far
    Quarantine_Rate can be trusted, never stated without the
    limitations (11.3); Quarantine_Rate is model-graded with no human
    ground truth, unlike recall@k/nDCG@10/MRR@10 (11.5). Any
    Quarantine_Rate value stated in this section is accompanied, in the
    same location, by the Agreement_Rate and the
    model-graded/no-human-ground-truth statement (11.6) — never
    presented alone.
  - For every Numeric_Claim added above — at minimum Quarantine_Rate
    (`source_artifact=groundedness.csv`,
    `source_fields="quarantine_decision=True.__count__;all.__count__"`,
    `computation=ratio`), Agreement_Rate
    (`source_artifact=hand_checked_joined.csv`,
    `source_fields="judge_verdict==hand_label"`, `computation=copy`),
    `quarantine_threshold`, `generation_subset_size`,
    `hand_checked_sample_size`, and `replay_top_k` — add one
    corresponding row to `docs/numeric_traceability.csv`,
    `document=SPEC.md`, in the same edit, using the real values read
    from `results/groundedness.csv`, `results/hand_checked_joined.csv`,
    and `results/run_config.json`'s `"groundedness"` sub-object — never
    typed from memory.
  - Done check:
    `python -c "text = open('SPEC.md', encoding='utf-8').read(); assert '## Groundedness gate' in text; assert 'Agreement_Rate' in text.split('## Groundedness gate', 1)[1]; assert 'Quarantine_Rate' in text.split('## Groundedness gate', 1)[1]; print('ok')"`
    prints `ok`; and
    `python -c "import pandas as pd; df = pd.read_csv('docs/numeric_traceability.csv'); new = df[df['document'] == 'SPEC.md']; assert (new['source_artifact'] == 'groundedness.csv').any(); assert (new['source_artifact'] == 'hand_checked_joined.csv').any(); print('ok')"`
    prints `ok`.
  - _Requirements: 1.1, 6.2, 6.3, 6.6, 6.9, 6.10, 6.11, 7.2, 7.3, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 16. Run the Verification_Pass for real and fix any mismatch
  - Run `python -m src.verify_writeup_numbers --repo-root .` against
    the fully updated `docs/numeric_traceability.csv` (Task 15) and
    the real committed `results/groundedness.csv` and
    `results/hand_checked_joined.csv` (Task 14 plus the hand-labelling
    rerun), together with every artifact already cited by the
    pre-existing 52 rows from the repo-writeup spec.
  - For any `MISMATCH`, correct the Numeric_Claim's text in `SPEC.md`,
    or correct the ledger row's `source_fields`/`computation`/
    `stated_precision` — never by editing any cited artifact. Re-run
    until every row reports `MATCH`.
  - Done check: `python -m src.verify_writeup_numbers --repo-root .`
    exits 0 (`echo $?` immediately after, in Git Bash, prints `0`),
    and its printed summary reports zero `MISMATCH` lines across every
    ledgered row, including the new rows added in Task 15.
  - _Requirements: 11.4, 11.6; repo-writeup Requirements 12.1, 12.2, 12.3, 12.4_

## Notes

- Tasks 3, 4, and 5 are, together, the entire automated-test surface
  of this spec (Requirement 12): `segment_claims` and
  `decide_quarantine` are the only two functions with pytest coverage.
  No property-based testing library is introduced — Requirement 12
  fixes the method as hand-built fixtures with independently reasoned
  expected outputs, matching the design's Testing Strategy.
- Tasks 13's done check, Task 7's `build_frozen_retriever`/
  `replay_retrieval` real-corpus behavior, `groundedness_config.py`'s
  validation contract, and every function in
  `src/hand_checked_sample.py` are verified structurally (by code
  shape) or by the one real end-to-end run in Task 14, not by pytest —
  consistent with the design's "What is explicitly not tested in this
  spec" list and this repo's established precedent (session-1 and
  significance-testing each defer their own entry-point/orchestration
  tests the same way).
- No task introduces an automated `SPEC.md`-writing mechanism: Task 15
  is a manual, hand-authored documentation step, matching the design's
  explicit rationale for why the Verification_Pass would stop verifying
  anything if the same code path that computed a value also rendered
  it into `SPEC.md`.
- The `src/verify_writeup_numbers.py` extension (Task 12) modifies a
  module owned by the repo-writeup spec, but the design explicitly
  states this feature's own implementation must make that change — it
  is scheduled here, not in a separate spec, and does not touch
  `README.md`, chunking, failure bucketing, or `ANALYSIS.md`.
- No task in this plan touches the third retriever
  (`BAAI/bge-small-en-v1.5`), any chunking strategy, failure bucketing,
  `ANALYSIS.md`, a chat/serving interface, or fine-tuning of any model
  — all explicitly out of scope per `requirements.md`'s introduction
  and `.kiro/steering/scope-guard.md`.
- All done checks are Git Bash / POSIX shell (`grep`, `mktemp -d`,
  `rm -rf`, `echo $?`) or Python one-liners — no PowerShell.
- Every task references specific acceptance criteria for traceability;
  no task is a giant "implement everything" step, and each builds on
  the prior ones.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2", "3", "12"] },
    { "id": 1, "tasks": ["4", "6", "7", "8", "9", "10", "11"] },
    { "id": 2, "tasks": ["5", "13"] },
    { "id": 3, "tasks": ["14"] },
    { "id": 4, "tasks": ["15"] },
    { "id": 5, "tasks": ["16"] }
  ]
}
```
