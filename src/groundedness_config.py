"""Groundedness_Config schema, YAML loading, and validation.

`load_groundedness_config` is the single point where
`configs/groundedness.yaml` is validated, mirroring `src/config.py`'s
`load_sweep_config` and `src/significance_config.py`'s
`load_significance_config`. It is deliberately kept in a file that
imports only `PyYAML`, the standard library, and
`src.groundedness_labels` -- never `transformers`, `torch`,
`src.corpus_loader`, or any retriever module -- so the config can be
loaded and validated without touching a model or the corpus.

The YAML's `label_mapping` and `score_definition` fields are a
documented record that must agree, field-for-field, with the
hard-coded constants in `src.groundedness_labels` (Requirement 6.3,
6.10). That agreement is checked here, once, at config-load time --
before any Generation_Subset query is processed -- which is what makes
"fixed before any Quarantine_Rate exists ... never revised" a
structurally enforced property rather than a documentation promise
alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from src.errors import GroundednessConfigError, LabelMappingMismatchError
from src.groundedness_labels import NLI_LABEL_TO_VERDICT

# The fixed expected text for `score_definition` (Requirement 6.10).
# Declared once, alongside NLI_LABEL_TO_VERDICT's own "declared once,
# cross-validated, never revised" contract -- the YAML's documented
# record of what judge_score means must match this exactly.
EXPECTED_SCORE_DEFINITION = (
    "The entailment probability obtained by applying softmax to the "
    "Judge_Model's three logits (entailment, neutral, contradiction); "
    "a value in [0.0, 1.0] where a higher value indicates stronger "
    "support for the Claim by the Retrieved_Context."
)

# Path fields that default rather than raise when absent from the YAML
# (mirroring significance_config.py's DEFAULT_RUN_CONFIG_PATH
# precedent) -- these make the artifact locations configurable without
# being mandatory boilerplate in every config.
DEFAULT_SWEEP_CONFIG_PATH = Path("configs/sweep.yaml")
DEFAULT_PER_QUERY_PATH = Path("results/per_query.csv")
DEFAULT_RUN_CONFIG_PATH = Path("results/run_config.json")
DEFAULT_OUTPUT_PATH = Path("results/groundedness.csv")
DEFAULT_HAND_CHECKED_SAMPLE_PATH = Path("results/hand_checked_sample.csv")
DEFAULT_HAND_CHECKED_JOINED_PATH = Path("results/hand_checked_joined.csv")
DEFAULT_GENERATED_ANSWERS_PATH = Path("results/generated_answers.csv")
DEFAULT_HAND_CHECKED_CONTEXT_PATH = Path("results/hand_checked_sample_context.md")


@dataclass(frozen=True)
class GroundednessConfig:
    replayed_run_id: str
    replay_top_k: int
    generation_subset_size: int
    generation_subset_seed: int
    generator_model_name: str
    judge_model_name: str
    max_new_tokens: int
    no_repeat_ngram_size: int
    repetition_penalty: float
    prompt_template: str
    quarantine_threshold: float
    hand_checked_sample_size: int
    hand_checked_sample_seed: int
    label_mapping: Dict[str, str]
    score_definition: str
    sweep_config_path: Path
    per_query_path: Path
    run_config_path: Path
    output_path: Path
    hand_checked_sample_path: Path
    hand_checked_joined_path: Path
    generated_answers_path: Path
    hand_checked_context_path: Path


def _require_field(mapping: dict, field: str, context: str) -> Any:
    """Returns mapping[field], raising GroundednessConfigError naming
    the missing field if absent. `context` identifies where the field
    was expected, matching src/config.py's `_require_field` contract."""
    if field not in mapping:
        raise GroundednessConfigError(f"{context}: missing required field '{field}'")
    return mapping[field]


def _require_int(value: Any, field: str, context: str) -> int:
    """Returns value coerced to int, raising GroundednessConfigError if
    value is not an integer. Rejects bool explicitly (bool is an int
    subclass in Python, but a boolean is never a valid seed) and
    rejects float values that are not integral (e.g. 3.5)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundednessConfigError(
            f"{context}: '{field}' must be an integer, got {value!r}"
        )
    if isinstance(value, float) and not value.is_integer():
        raise GroundednessConfigError(
            f"{context}: '{field}' must be an integer, got {value!r}"
        )
    return int(value)


def _require_positive_int(value: Any, field: str, context: str) -> int:
    """Returns value coerced to a positive int, raising
    GroundednessConfigError if value is not an integer, or is not
    strictly positive."""
    coerced = _require_int(value, field, context)
    if coerced <= 0:
        raise GroundednessConfigError(
            f"{context}: '{field}' must be a positive integer, got {value!r}"
        )
    return coerced


def _require_non_negative_int(value: Any, field: str, context: str) -> int:
    """Returns value coerced to a non-negative int, raising
    GroundednessConfigError if value is not an integer, or is
    negative. Unlike `_require_positive_int`, `0` is a valid value
    here -- `no_repeat_ngram_size: 0` is the documented no-op
    (transformers' own default), not an invalid declaration."""
    coerced = _require_int(value, field, context)
    if coerced < 0:
        raise GroundednessConfigError(
            f"{context}: '{field}' must be a non-negative integer, got {value!r}"
        )
    return coerced


def _require_numeric(value: Any, field: str, context: str) -> float:
    """Returns value coerced to float, raising GroundednessConfigError
    if value is not numeric. Rejects bool explicitly, matching
    `_require_int`'s reasoning."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundednessConfigError(
            f"{context}: '{field}' must be numeric, got {value!r}"
        )
    return float(value)


def _validate_label_mapping(declared: Any, context: str) -> None:
    """Raises `LabelMappingMismatchError` unless `declared` is exactly
    equal to `NLI_LABEL_TO_VERDICT` (Requirement 6.3) -- same keys
    (`entailment`, `neutral`, `contradiction`), same values
    (`SUPPORTED`/`NOT_SUPPORTED`). Mirrors `src/config.py`'s
    `_load_bm25_config` validating `tokenizer`/`stopwords`/`stemming`
    against `SUPPORTED_BM25_*` module constants: the YAML is data, but
    data that must agree with a fixed code-side constant, not data the
    code blindly trusts.
    """
    if declared != NLI_LABEL_TO_VERDICT:
        raise LabelMappingMismatchError(
            f"{context}: 'label_mapping' {declared!r} does not match the "
            f"hard-coded src.groundedness_labels.NLI_LABEL_TO_VERDICT "
            f"{NLI_LABEL_TO_VERDICT!r} -- the mapping is fixed before any "
            f"Quarantine_Rate exists and is never revised after seeing "
            f"results"
        )


def _validate_score_definition(declared: Any, context: str) -> None:
    """Raises `LabelMappingMismatchError` unless `declared` equals the
    fixed expected text `EXPECTED_SCORE_DEFINITION` (Requirement 6.10).
    A `GroundednessConfigError` subclass, like
    `_validate_label_mapping`'s check, since both are the same
    "documented record must agree with a fixed code-side constant"
    contract applied to two different fields.
    """
    if declared != EXPECTED_SCORE_DEFINITION:
        raise LabelMappingMismatchError(
            f"{context}: 'score_definition' {declared!r} does not match "
            f"the fixed expected text {EXPECTED_SCORE_DEFINITION!r} -- the "
            f"score definition is fixed before any Quarantine_Rate exists "
            f"and is never revised after seeing results"
        )


def load_groundedness_config(path: Path) -> GroundednessConfig:
    """Reads and validates the Groundedness_Config YAML file at `path`.

    Raises `GroundednessConfigError` if the file is missing, is not
    valid YAML, omits any declaration required by Requirement 1
    Criterion 1, declares `generator_model_name == judge_model_name`
    (Requirement 1.3), declares `replay_top_k` /
    `generation_subset_size` / `hand_checked_sample_size` as anything
    other than a positive integer (Requirement 1.5, 1.6), declares
    `generation_subset_seed` / `hand_checked_sample_seed` as anything
    other than an integer (Requirement 1.5), declares
    `quarantine_threshold` as anything other than numeric (Requirement
    1.5, 1.7), or declares `no_repeat_ngram_size`/`repetition_penalty`
    (when present) as anything other than a non-negative
    integer/numeric value respectively. Raises `LabelMappingMismatchError` (a
    `GroundednessConfigError` subclass) if the YAML's `label_mapping`
    record disagrees with `NLI_LABEL_TO_VERDICT`, or if
    `score_definition` does not match the fixed expected text
    (Requirement 1.1, 6.3, 6.10) -- both checked before any
    Generation_Subset query is processed. Never partially applies a
    config: the first violation found raises and no
    `GroundednessConfig` is returned.
    """
    path = Path(path)
    if not path.is_file():
        raise GroundednessConfigError(f"Groundedness_Config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GroundednessConfigError(
            f"failed to read Groundedness_Config file {path}: {exc}"
        ) from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise GroundednessConfigError(f"failed to parse {path} as YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise GroundednessConfigError(f"{path}: top-level YAML content must be a mapping")

    context = "top-level config"

    replayed_run_id = _require_field(data, "replayed_run_id", context)
    replay_top_k = _require_positive_int(
        _require_field(data, "replay_top_k", context), "replay_top_k", context
    )
    generation_subset_size = _require_positive_int(
        _require_field(data, "generation_subset_size", context),
        "generation_subset_size",
        context,
    )
    generation_subset_seed = _require_int(
        _require_field(data, "generation_subset_seed", context),
        "generation_subset_seed",
        context,
    )
    generator_model_name = _require_field(data, "generator_model_name", context)
    judge_model_name = _require_field(data, "judge_model_name", context)
    max_new_tokens = _require_positive_int(
        _require_field(data, "max_new_tokens", context), "max_new_tokens", context
    )
    # Both default to their transformers no-op value (0 disables
    # no_repeat_ngram_size; 1.0 disables repetition_penalty) when
    # absent from the YAML, rather than raising -- an older config
    # written before this correction stays loadable, decoding
    # unchanged, exactly as if these fields had been declared with
    # their no-op values explicitly.
    no_repeat_ngram_size = _require_non_negative_int(
        data.get("no_repeat_ngram_size", 0), "no_repeat_ngram_size", context
    )
    repetition_penalty = _require_numeric(
        data.get("repetition_penalty", 1.0), "repetition_penalty", context
    )
    prompt_template = _require_field(data, "prompt_template", context)

    quarantine_threshold = _require_numeric(
        _require_field(data, "quarantine_threshold", context),
        "quarantine_threshold",
        context,
    )

    hand_checked_sample_size = _require_positive_int(
        _require_field(data, "hand_checked_sample_size", context),
        "hand_checked_sample_size",
        context,
    )
    hand_checked_sample_seed = _require_int(
        _require_field(data, "hand_checked_sample_seed", context),
        "hand_checked_sample_seed",
        context,
    )

    label_mapping = _require_field(data, "label_mapping", context)
    score_definition = _require_field(data, "score_definition", context)

    if str(generator_model_name) == str(judge_model_name):
        raise GroundednessConfigError(
            f"{context}: 'generator_model_name' and 'judge_model_name' must "
            f"differ, both are {generator_model_name!r} -- a model judging "
            f"its own generated output is biased toward accepting it"
        )

    _validate_label_mapping(label_mapping, context)
    _validate_score_definition(score_definition, context)

    sweep_config_path = data.get("sweep_config_path", DEFAULT_SWEEP_CONFIG_PATH)
    per_query_path = data.get("per_query_path", DEFAULT_PER_QUERY_PATH)
    run_config_path = data.get("run_config_path", DEFAULT_RUN_CONFIG_PATH)
    output_path = data.get("output_path", DEFAULT_OUTPUT_PATH)
    hand_checked_sample_path = data.get(
        "hand_checked_sample_path", DEFAULT_HAND_CHECKED_SAMPLE_PATH
    )
    hand_checked_joined_path = data.get(
        "hand_checked_joined_path", DEFAULT_HAND_CHECKED_JOINED_PATH
    )
    generated_answers_path = data.get(
        "generated_answers_path", DEFAULT_GENERATED_ANSWERS_PATH
    )
    hand_checked_context_path = data.get(
        "hand_checked_context_path", DEFAULT_HAND_CHECKED_CONTEXT_PATH
    )

    return GroundednessConfig(
        replayed_run_id=str(replayed_run_id),
        replay_top_k=replay_top_k,
        generation_subset_size=generation_subset_size,
        generation_subset_seed=generation_subset_seed,
        generator_model_name=str(generator_model_name),
        judge_model_name=str(judge_model_name),
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        prompt_template=str(prompt_template),
        quarantine_threshold=quarantine_threshold,
        hand_checked_sample_size=hand_checked_sample_size,
        hand_checked_sample_seed=hand_checked_sample_seed,
        label_mapping=dict(label_mapping),
        score_definition=str(score_definition),
        sweep_config_path=Path(sweep_config_path),
        per_query_path=Path(per_query_path),
        run_config_path=Path(run_config_path),
        output_path=Path(output_path),
        hand_checked_sample_path=Path(hand_checked_sample_path),
        hand_checked_joined_path=Path(hand_checked_joined_path),
        generated_answers_path=Path(generated_answers_path),
        hand_checked_context_path=Path(hand_checked_context_path),
    )
