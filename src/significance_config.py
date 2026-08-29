"""Bootstrap_Config schema, YAML loading, and validation.

`load_significance_config` is the single point where the Bootstrap_Config
declared under `configs/significance.yaml` is validated, mirroring
`src/config.py`'s `load_sweep_config` (design.md's
`src/significance_config.py` section). It is deliberately kept in a
file separate from `configs/sweep.yaml`: the Significance_Analyzer must
be loadable without parsing or validating a retriever grid it never
uses, and keeping the two config files separate means `bootstrap_seed`
(this module) and `seed` (the sweep seed, `src/config.py`) can never be
conflated or accidentally cross-referenced (Requirement 4.2). This
module imports only `PyYAML` and the standard library -- never `beir`,
`sentence-transformers`, `huggingface_hub`, or any retrieval code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.errors import BootstrapConfigError

# Alpha is declared in advance and never revised after seeing results
# (evaluation-integrity.md); the loader enforces the fixed value here.
REQUIRED_ALPHA = 0.05

# run_config_path is the one field that defaults rather than raising
# when absent from the YAML (Requirement 4.7): it makes the merge
# target configurable (e.g. redirectable to a temp dir for testing)
# instead of hard-coded, while still always carrying a value.
DEFAULT_RUN_CONFIG_PATH = Path("results/run_config.json")


@dataclass(frozen=True)
class SignificanceConfig:
    resample_count: int
    permutation_count: int
    bootstrap_seed: int
    alpha: float
    reference_retriever: str
    per_query_path: Path
    output_path: Path
    run_config_path: Path


def _require_field(mapping: dict, field: str, context: str) -> Any:
    """Returns mapping[field], raising BootstrapConfigError naming the
    missing field if absent. `context` (e.g. "top-level config")
    identifies where the field was expected, matching
    `src/config.py`'s `_require_field` contract."""
    if field not in mapping:
        raise BootstrapConfigError(f"{context}: missing required field '{field}'")
    return mapping[field]


def _require_int(value: Any, field: str, context: str) -> int:
    """Returns value coerced to int, raising BootstrapConfigError if
    value is not an integer. Rejects bool explicitly (bool is a int
    subclass in Python, but a boolean is never a valid resample count,
    permutation count, or seed) and rejects float values that are not
    integral (e.g. 3.5), matching the "declares ... as anything other
    than an integer" failure named in Requirement 4.5."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapConfigError(
            f"{context}: '{field}' must be an integer, got {value!r}"
        )
    if isinstance(value, float) and not value.is_integer():
        raise BootstrapConfigError(
            f"{context}: '{field}' must be an integer, got {value!r}"
        )
    return int(value)


def load_significance_config(path: Path) -> SignificanceConfig:
    """Reads and validates the Bootstrap_Config YAML file at `path`.

    Raises `BootstrapConfigError` (a `ConfigError` subclass) if the
    file is missing, is not valid YAML, omits `resample_count` /
    `permutation_count` / `bootstrap_seed` / `alpha` /
    `reference_retriever` / `per_query_path` / `output_path`, declares
    `resample_count` / `permutation_count` / `bootstrap_seed` as
    anything other than an integer, or declares `alpha` as anything
    other than the fixed value 0.05 (Requirement 4.5, 6.4).
    `run_config_path` is the sole exception: if absent from the YAML it
    defaults to `results/run_config.json` rather than raising
    (Requirement 4.7), so the merge target is configurable without
    being mandatory boilerplate in every config. Never partially
    applies a config: the first violation found raises and no
    `SignificanceConfig` is returned.
    """
    path = Path(path)
    if not path.is_file():
        raise BootstrapConfigError(f"Bootstrap_Config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapConfigError(
            f"failed to read Bootstrap_Config file {path}: {exc}"
        ) from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise BootstrapConfigError(f"failed to parse {path} as YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise BootstrapConfigError(f"{path}: top-level YAML content must be a mapping")

    context = "top-level config"

    resample_count = _require_int(
        _require_field(data, "resample_count", context), "resample_count", context
    )
    permutation_count = _require_int(
        _require_field(data, "permutation_count", context), "permutation_count", context
    )
    bootstrap_seed = _require_int(
        _require_field(data, "bootstrap_seed", context), "bootstrap_seed", context
    )

    alpha_raw = _require_field(data, "alpha", context)
    try:
        alpha = float(alpha_raw)
    except (TypeError, ValueError) as exc:
        raise BootstrapConfigError(
            f"{context}: 'alpha' must be numeric, got {alpha_raw!r}"
        ) from exc
    if alpha != REQUIRED_ALPHA:
        raise BootstrapConfigError(
            f"{context}: 'alpha' must equal the fixed value {REQUIRED_ALPHA}, "
            f"got {alpha_raw!r} -- alpha is declared in advance and never "
            f"revised after seeing results"
        )

    reference_retriever = _require_field(data, "reference_retriever", context)
    per_query_path = _require_field(data, "per_query_path", context)
    output_path = _require_field(data, "output_path", context)
    run_config_path = data.get("run_config_path", DEFAULT_RUN_CONFIG_PATH)

    return SignificanceConfig(
        resample_count=resample_count,
        permutation_count=permutation_count,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
        reference_retriever=str(reference_retriever),
        per_query_path=Path(per_query_path),
        output_path=Path(output_path),
        run_config_path=Path(run_config_path),
    )
