"""Groundedness_Runner: the orchestrating entry point for the
groundedness-gate spec (Requirement 2, 9, 11).

Imports everything else this spec introduces -- this is the only
module in this spec that imports both `src.generator_model` and
`src.judge_model` together (Requirement 6.5's "distinct from the
Generator_Model" is a config-load-time check per
`groundedness_config.py`, but this module is where both wrapper
instances actually coexist at run time).

Unlike session 1's `Sweep_Runner` (which recovers per cell with the
`MISSING`/`"NA"` marker) and unlike the significance-testing spec's
`Significance_Analyzer` (which recovers exactly one case -- a
zero-shared-queries comparison -- by retaining a row with a missing
marker), this module has no recoverable case at all: every exception
raised anywhere in the orchestration halts the entire run before
`results/groundedness.csv` is ever written (Requirement 8.5).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas

from src.claim_segmenter import segment_claims
from src.errors import (
    FrozenRetrieverConfigError,
    GeneratedAnswersWriteError,
    GenerationSubsetInputError,
    GeneratorGenerationError,
    GeneratorModelLoadError,
    GroundednessConfigError,
    GroundednessReportWriteError,
    HandCheckedJoinedWriteError,
    HandCheckedSampleWriteError,
    JudgeModelLoadError,
    JudgeVerdictError,
    ReplayedRunNotFoundError,
    RetrievalReplayError,
    RunConfigMergeError,
)
from src.generated_answers_report import GeneratedAnswerRow, write_generated_answers_report
from src.generator_model import GeneratorModel
from src.groundedness_config import GroundednessConfig, load_groundedness_config
from src.groundedness_report import GroundednessReportRow, write_groundedness_report
from src.hand_checked_sample import (
    HandCheckedSampleRow,
    compute_agreement_rate,
    export_hand_checked_sample,
    join_hand_labels_with_verdicts,
    read_hand_label_import,
    select_hand_checked_sample,
    write_hand_checked_joined,
)
from src.judge_model import JudgeModel
from src.quarantine_rule import decide_quarantine
from src.report import _atomic_write_text, _json_default
from src.retrieval_replay import build_frozen_retriever, load_frozen_retriever_config, replay_retrieval

ClaimId = Tuple[str, int]

DEFAULT_CONFIG_PATH = Path("configs/groundedness.yaml")


def _split_context_into_sentences(retrieved_context: List[str]) -> List[str]:
    """Splits `retrieved_context` (the ordered list of retrieved
    document texts for one query) into individual sentences, reusing
    `src.claim_segmenter.segment_claims`'s sentence-boundary rule --
    the same boundary definition the Claim_Segmenter applies to a
    Generated_Answer, applied here to the Retrieved_Context side so
    the Judge_Model's premise granularity matches what
    `cross-encoder/nli-deberta-v3-xsmall` was trained on
    (single-sentence premises), per the sentence-granularity
    correctness fix in `JudgeModel.judge_best_sentence`. One document's
    sentences never merge with the next document's: `segment_claims` is
    called once per document, not once over the whole joined block, so
    a sentence never spans a document boundary.
    """
    sentences: List[str] = []
    for document_text in retrieved_context:
        sentences.extend(claim.text for claim in segment_claims(document_text))
    return sentences


def build_prompt(template: str, query_text: str, retrieved_context: List[str]) -> str:
    """Combines `query_text` and `retrieved_context` into the
    Generator_Model's input text via `template.format(...)`
    (Requirement 4.1).

    `retrieved_context`'s documents are newline-joined in rank order
    (`"\\n\\n".join(retrieved_context)`) before substitution -- the
    exact same join format `JudgeModel.judge`'s `premise` argument
    uses, so the Generator_Model and the Judge_Model are always shown
    literally the same context block for a given query, never two
    independently-formatted variants of it.
    """
    context_block = "\n\n".join(retrieved_context)
    return template.format(query=query_text, context=context_block)


def _read_per_query_report(path: Path) -> pandas.DataFrame:
    """Reads `path` (default results/per_query.csv), read-only.

    Raises `GenerationSubsetInputError` if `path` is absent, cannot be
    parsed as a CSV, or lacks a `run_id` column or a `query_id` column
    (Requirement 2.1).
    """
    path = Path(path)
    if not path.is_file():
        raise GenerationSubsetInputError(f"per-query report file not found: {path}")
    try:
        frame = pandas.read_csv(path)
    except Exception as exc:
        raise GenerationSubsetInputError(
            f"failed to parse per-query report {path}: {exc}"
        ) from exc
    missing_columns = [c for c in ("run_id", "query_id") if c not in frame.columns]
    if missing_columns:
        raise GenerationSubsetInputError(
            f"per-query report {path} is missing required column(s): {missing_columns}"
        )
    return frame


def _scored_query_ids_for_run(frame: pandas.DataFrame, replayed_run_id: str) -> List[str]:
    """Determines the set of query IDs the Replayed_Run's `run_id`
    actually scored: the distinct values of the `query_id` column
    across every row whose `run_id` column equals `replayed_run_id`
    (Requirement 2.2).

    Raises `ReplayedRunNotFoundError` if `replayed_run_id` is not
    present in the `run_id` column at all (Requirement 2.5).
    """
    matching = frame[frame["run_id"] == replayed_run_id]
    if matching.empty:
        raise ReplayedRunNotFoundError(
            f"replayed_run_id {replayed_run_id!r} is not present in the "
            f"per-query report's run_id column"
        )
    return sorted(str(qid) for qid in matching["query_id"].unique())


def _sample_generation_subset(
    scored_query_ids: List[str], subset_size: int, seed: int
) -> List[str]:
    """Samples the Generation_Subset: draws `min(subset_size,
    len(scored_query_ids))` query IDs uniformly at random, without
    replacement, from a canonical ascending sort of
    `scored_query_ids`, seeded with `seed` (Requirement 2.3, 2.6).
    """
    canonical_order = sorted(scored_query_ids)
    rng = random.Random(seed)
    k = min(subset_size, len(canonical_order))
    return sorted(rng.sample(canonical_order, k))


def _truncation_stats(dropped_token_counts: List[int]) -> Dict[str, float]:
    """Summarizes a list of per-item "tokens dropped by truncation"
    counts (one entry per prompt or premise/hypothesis pair, `0` for
    an item that was not truncated) into the aggregate fields recorded
    in `results/run_config.json`'s `"groundedness"` sub-object.

    `replay_top_k` (retrieval depth) and each model's tokenizer input
    budget are independent axes -- deepening retrieval does not shrink
    a model's `model_max_length`, and `configs/groundedness.yaml`
    keeps `replay_top_k: 10` to match the sweep's own retrieval depth
    rather than declaring a second, narrower top-k just for generation.
    Recording these counts, per run, is what makes that truncation
    visible rather than silent (the config's own comment on
    `replay_top_k` explains the tradeoff this function's output makes
    legible).
    """
    total = len(dropped_token_counts)
    truncated = [d for d in dropped_token_counts if d > 0]
    return {
        "total_items": total,
        "truncated_count": len(truncated),
        "tokens_dropped_max": max(dropped_token_counts) if dropped_token_counts else 0,
        "tokens_dropped_mean_when_truncated": (
            sum(truncated) / len(truncated) if truncated else 0.0
        ),
    }


def _merge_groundedness_into_run_config(
    run_config_record: dict,
    config: GroundednessConfig,
    run_config_path: Path,
    generator_max_input_tokens: int,
    judge_max_input_tokens: int,
    generator_truncation: Dict[str, float],
    judge_truncation: Dict[str, float],
) -> None:
    """Merges the `"groundedness"` sub-object into `run_config_record`
    and re-writes `run_config_path` atomically, preserving every
    existing key unchanged (Requirement 9.1, 9.2) -- the identical
    `dict(existing_record)` -> set one sibling key ->
    `json.dumps(..., default=_json_default)` -> `_atomic_write_text`
    pipeline `src/significance.py`'s
    `_merge_significance_into_run_config` already implements.

    `generator_max_input_tokens`/`judge_max_input_tokens` are each
    read from the loaded model's own tokenizer
    (`GeneratorModel.max_input_tokens`/`JudgeModel.max_input_tokens`),
    never hard-coded -- recorded so the truncation asymmetry between
    `replay_top_k` (retrieval depth) and each model's input token
    budget is visible from a committed artifact, the same as
    `results/token_length_report.json` makes the retrieval-side
    truncation visible for the dense retriever.

    Raises `RunConfigMergeError` if serialization or the atomic write
    fails (Requirement 9.5); the original `run_config_path` is left
    untouched on failure.
    """
    record = dict(run_config_record)
    record["groundedness"] = {
        "replayed_run_id": config.replayed_run_id,
        "replay_top_k": config.replay_top_k,
        "generation_subset_size": config.generation_subset_size,
        "generation_subset_seed": config.generation_subset_seed,
        "generator_model_name": config.generator_model_name,
        "judge_model_name": config.judge_model_name,
        "max_new_tokens": config.max_new_tokens,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "repetition_penalty": config.repetition_penalty,
        "generator_max_input_tokens": generator_max_input_tokens,
        "judge_max_input_tokens": judge_max_input_tokens,
        "generator_prompt_truncation": generator_truncation,
        "judge_premise_truncation": judge_truncation,
        "quarantine_threshold": config.quarantine_threshold,
        "hand_checked_sample_size": config.hand_checked_sample_size,
        "hand_checked_sample_seed": config.hand_checked_sample_seed,
    }
    try:
        json_text = json.dumps(record, indent=2, default=_json_default)
    except Exception as exc:
        raise RunConfigMergeError(
            f"failed to serialize merged run config record for {run_config_path}: {exc}"
        ) from exc
    try:
        _atomic_write_text(
            run_config_path,
            json_text,
            failure_context="run config record (groundedness merge)",
        )
    except Exception as exc:
        raise RunConfigMergeError(str(exc)) from exc


def _read_run_config(path: Path) -> dict:
    """Reads and parses the run configuration record at `path` (default
    `results/run_config.json`).

    Raises `RunConfigMergeError` if `path` is absent, unreadable, not
    valid JSON, or not a JSON object -- the runner never creates a
    fresh record in place of a missing/unparsable one (Requirement
    9.4).
    """
    path = Path(path)
    if not path.is_file():
        raise RunConfigMergeError(
            f"the existing run configuration record is missing: {path}"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigMergeError(
            f"the existing run configuration record is unreadable ({path}): {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise RunConfigMergeError(
            f"the existing run configuration record at {path} is not a JSON object"
        )
    return record


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.groundedness_runner",
        description=(
            "Replays a frozen session-1 retriever over a small seeded "
            "subset of queries, generates one answer per query, splits "
            "it into claims, judges each claim's groundedness against "
            "the same retrieved context, applies the quarantine rule, "
            "and writes results/groundedness.csv."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the Groundedness_Config YAML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: `python -m src.groundedness_runner [--config PATH]`.

    Orchestration, in order (`design.md`'s `src/groundedness_runner.py`
    section):

    1. Load config, halting on `GroundednessConfigError` (incl.
       `LabelMappingMismatchError`).
    2. Read `results/per_query.csv`, determine the Replayed_Run's
       scored query IDs, halting on `GenerationSubsetInputError` /
       `ReplayedRunNotFoundError`.
    3. Sample the Generation_Subset via a seeded, canonical-order draw.
    4. `load_frozen_retriever_config` + `build_frozen_retriever`,
       halting on `FrozenRetrieverConfigError`.
    5. `replay_retrieval` over the whole subset in one call, halting on
       `RetrievalReplayError`.
    6. Construct `GeneratorModel` and `JudgeModel`, halting on
       `GeneratorModelLoadError` / `JudgeModelLoadError`.
    7. For each query_id in sorted order: build the prompt,
       `generator.generate(prompt, config.max_new_tokens,
       no_repeat_ngram_size=..., repetition_penalty=...)` (wrapping
       `GeneratorGenerationError` with that `query_id` attached),
       recording one `GeneratedAnswerRow` (with the prompt's
       untruncated token count via `generator.count_tokens(prompt)`),
       `segment_claims(...)`, then splitting the Retrieved_Context into
       individual sentences (`_split_context_into_sentences`, reusing
       `segment_claims`'s own boundary rule) and, for each Claim,
       `judge_model.judge_best_sentence(context_sentences, claim.text)`
       (wrapping `JudgeVerdictError` with `query_id`/`claim_index`
       attached) -- scoring the Claim against each retrieved sentence
       individually and taking the maximum entailment probability,
       matching the Judge_Model's single-sentence training
       distribution -- and `decide_quarantine(...)`, appending one
       `GroundednessReportRow` (including which sentence produced the
       maximum, in `matched_sentence`); any exception halts before any
       report is written.
    8. `write_groundedness_report`, halting on
       `GroundednessReportWriteError`; then
       `write_generated_answers_report`, halting on
       `GeneratedAnswersWriteError`.
    9. `select_hand_checked_sample` + `export_hand_checked_sample` (a
       no-op if the file already carries hand labels), halting on
       `HandCheckedSampleWriteError`.
    10. `read_hand_label_import` -- if non-`None`, build and write the
        joined rows via `join_hand_labels_with_verdicts` +
        `write_hand_checked_joined`, halting on
        `HandCheckedJoinedWriteError`, and print `Agreement_Rate` to
        stdout purely informationally.
    11. Merge the `"groundedness"` sibling key into
        `results/run_config.json`, halting on `RunConfigMergeError`.
        Return 0.
    """
    args = _parse_args(argv)

    try:
        config = load_groundedness_config(args.config)
    except GroundednessConfigError as exc:
        print(
            f"ERROR: failed to load Groundedness_Config from {args.config}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Requirement 9.4: results/run_config.json must be present and
    # parsable before the Groundedness_Report is written -- read (and
    # validated) here, early, rather than only at the merge step at the
    # very end, so a missing/unparsable record halts the run before any
    # model is loaded or any query is processed.
    try:
        run_config_record = _read_run_config(config.run_config_path)
    except RunConfigMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        per_query_frame = _read_per_query_report(config.per_query_path)
        scored_query_ids = _scored_query_ids_for_run(per_query_frame, config.replayed_run_id)
    except (GenerationSubsetInputError, ReplayedRunNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    generation_subset = _sample_generation_subset(
        scored_query_ids, config.generation_subset_size, config.generation_subset_seed
    )

    try:
        sweep_config, retriever_config = load_frozen_retriever_config(
            config.sweep_config_path, config.replayed_run_id
        )
        retriever, bundle = build_frozen_retriever(sweep_config, retriever_config)
    except FrozenRetrieverConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        retrieved_context_by_query = replay_retrieval(
            retriever, bundle, generation_subset, bundle.queries, config.replay_top_k
        )
    except RetrievalReplayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cache_folder = sweep_config.data_dir / "hf_cache"
    try:
        generator = GeneratorModel(config.generator_model_name, cache_folder)
    except GeneratorModelLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        judge = JudgeModel(config.judge_model_name, cache_folder)
    except JudgeModelLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report_rows: List[GroundednessReportRow] = []
    generated_answer_rows: List[GeneratedAnswerRow] = []
    judge_verdicts: Dict[ClaimId, str] = {}
    generator_dropped_tokens: List[int] = []
    judge_dropped_tokens: List[int] = []
    try:
        for query_id in sorted(generation_subset):
            retrieved_context = retrieved_context_by_query[query_id]
            prompt = build_prompt(config.prompt_template, bundle.queries[query_id], retrieved_context)

            # Made explicit and visible, per Requirement's-adjacent
            # correction: truncation=True (no explicit max_length)
            # silently clamps the encoder input to the tokenizer's own
            # model_max_length. Measuring the untruncated token count
            # here -- before generate() truncates -- is what makes the
            # dropped-token count something this run records, rather
            # than something only discoverable by re-running with
            # instrumentation.
            prompt_token_count = generator.count_tokens(prompt)
            generator_dropped_tokens.append(
                max(0, prompt_token_count - generator.max_input_tokens)
            )

            try:
                generated_answer = generator.generate(
                    prompt,
                    config.max_new_tokens,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    repetition_penalty=config.repetition_penalty,
                )
            except GeneratorGenerationError as exc:
                raise GeneratorGenerationError(
                    f"query_id={query_id!r}: {exc}"
                ) from exc

            generated_answer_rows.append(
                GeneratedAnswerRow(
                    query_id=query_id,
                    prompt_token_count=prompt_token_count,
                    answer_text=generated_answer,
                )
            )

            claims = segment_claims(generated_answer)
            # Sentence-granularity correctness fix: the Judge_Model is
            # scored against each Retrieved_Context sentence
            # individually, never against the whole multi-document
            # block at once (see judge_best_sentence's docstring).
            context_sentences = _split_context_into_sentences(retrieved_context)
            for claim in claims:
                try:
                    judge_result, matched_sentence, dropped = judge.judge_best_sentence(
                        context_sentences, claim.text
                    )
                except JudgeVerdictError as exc:
                    raise JudgeVerdictError(
                        f"query_id={query_id!r}, claim_index={claim.claim_index!r}: {exc}"
                    ) from exc
                judge_dropped_tokens.extend(dropped)

                quarantine_decision = decide_quarantine(
                    judge_result.verdict, judge_result.score, config.quarantine_threshold
                )
                judge_verdicts[(query_id, claim.claim_index)] = judge_result.verdict
                report_rows.append(
                    GroundednessReportRow(
                        query_id=query_id,
                        claim_index=claim.claim_index,
                        claim_text=claim.text,
                        groundedness_verdict=judge_result.verdict,
                        judge_score=judge_result.score,
                        quarantine_decision=quarantine_decision,
                        matched_sentence=matched_sentence,
                    )
                )
    except (GeneratorGenerationError, JudgeVerdictError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        write_groundedness_report(report_rows, config.output_path)
    except GroundednessReportWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        write_generated_answers_report(generated_answer_rows, config.generated_answers_path)
    except GeneratedAnswersWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    claim_ids = list(judge_verdicts.keys())
    hand_checked_sample_ids = select_hand_checked_sample(
        claim_ids, config.hand_checked_sample_size, config.hand_checked_sample_seed
    )
    claim_text_by_id = {(row.query_id, row.claim_index): row.claim_text for row in report_rows}
    hand_checked_rows = [
        HandCheckedSampleRow(
            query_id=qid, claim_index=idx, claim_text=claim_text_by_id[(qid, idx)], hand_label=""
        )
        for qid, idx in hand_checked_sample_ids
    ]
    try:
        export_hand_checked_sample(hand_checked_rows, config.hand_checked_sample_path)
    except HandCheckedSampleWriteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    hand_labels = read_hand_label_import(config.hand_checked_sample_path, hand_checked_sample_ids)
    if hand_labels is not None:
        joined_rows = join_hand_labels_with_verdicts(judge_verdicts, hand_labels)
        try:
            write_hand_checked_joined(joined_rows, config.hand_checked_joined_path)
        except HandCheckedJoinedWriteError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        agreement_rate = compute_agreement_rate(judge_verdicts, hand_labels)
        print(f"Agreement_Rate: {agreement_rate:.4f} (informational; never written to any file)")

    try:
        _merge_groundedness_into_run_config(
            run_config_record,
            config,
            config.run_config_path,
            generator_max_input_tokens=generator.max_input_tokens,
            judge_max_input_tokens=judge.max_input_tokens,
            generator_truncation=_truncation_stats(generator_dropped_tokens),
            judge_truncation=_truncation_stats(judge_dropped_tokens),
        )
    except RunConfigMergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Groundedness gate complete: wrote {len(report_rows)} rows to {config.output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
