"""Requirement 7 test suite: the entire automated-test surface for the
`significance-testing` spec.

Covers only the three pure functions `paired_bootstrap`,
`permutation_test`, and `holm_bonferroni` from `src.significance`, using
hand-built synthetic numpy arrays with independently reasoned expected
values (Requirement 7.9). This module imports nothing beyond
`src.significance`, `numpy`, and `pytest` -- no analyzer entry point, no
sweep orchestration, no data-loading or retriever code -- and makes no
network call and loads no dataset or model (Requirement 7.10). Each
`generator` is constructed locally via `np.random.default_rng(<seed>)`;
none of these tests touches global RNG state.

Note on the constant-offset permutation fixture (Requirement 7.4): a
paired permutation test's null distribution over a constant per-query
offset is fully discrete, with exactly 2 (all-signs-agree) out of
2**n equally likely sign patterns tying the observed extremity. At
n = 5 that floor is 2/32 = 0.0625, which can never fall below 0.01 no
matter how large `permutation_count` is -- the discreteness, not the
sample count, sets the floor. This suite therefore uses a longer
constant-offset vector (n = 20, floor 2/2**20 ~ 2e-6) for the p < 0.01
assertion specifically, while the CI-excludes-zero assertion (7.2) and
both self-comparison assertions (7.1, 7.3) hold at any n and keep the
shorter arrays.
"""

import numpy as np
import pytest

from src.significance import holm_bonferroni, paired_bootstrap, permutation_test


# ---------------------------------------------------------------------------
# paired_bootstrap / permutation_test (Requirements 7.1-7.5)
# ---------------------------------------------------------------------------


def test_self_comparison_mean_diff_is_exactly_zero():
    # Req 7.1: a run compared against itself has an exactly-zero mean
    # difference (d = a - b is all zeros).
    a = np.array([0.2, 0.5, 0.9, 0.1, 0.7])
    generator = np.random.default_rng(20240)
    mean_diff, _lo, _hi = paired_bootstrap(
        a, a.copy(), resample_count=1000, generator=generator
    )
    assert mean_diff == 0.0  # exact


def test_self_comparison_p_value_is_one():
    # Req 7.3: every sign-flip of an all-zero difference vector leaves
    # it unchanged, so every permuted mean ties the observed 0.0 and
    # the add-one p-value is exactly 1.0.
    a = np.array([0.2, 0.5, 0.9, 0.1, 0.7])
    generator = np.random.default_rng(20240)
    p = permutation_test(a, a.copy(), permutation_count=1000, generator=generator)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_constant_offset_ci_excludes_zero_with_matching_sign():
    # Req 7.2: a constant positive per-query offset makes every
    # resample's mean exactly the offset, so both CI percentiles equal
    # the offset and lie strictly on the positive side of zero.
    b = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
    a = b + 0.25
    generator = np.random.default_rng(20240)
    _mean_diff, lo, hi = paired_bootstrap(a, b, resample_count=2000, generator=generator)
    assert lo > 0.0 and hi > 0.0


def test_constant_offset_p_value_below_threshold():
    # Req 7.4: with permutation_count = 2000 the add-one floor is
    # 1/2001 ~ 5e-4, well below 0.01 -- but a constant offset also
    # needs enough paired queries that the "all signs agree" outcome
    # (the only sign pattern tying the observed extremity) is rare
    # enough in the discrete null. n = 20 (floor 2/2**20 ~ 2e-6) clears
    # the 0.01 threshold comfortably; n = 5, as used in the CI test
    # above, cannot (its exact floor is 2/32 = 0.0625).
    b = np.linspace(0.1, 0.6, 20)
    a = b + 0.25
    generator = np.random.default_rng(20240)
    p = permutation_test(a, b, permutation_count=2000, generator=generator)
    assert p < 0.01


def test_same_seed_reproduces_identical_ci_bounds():
    # Req 7.5: two independently constructed generators seeded
    # identically must draw identical index matrices, hence identical
    # resampled means and percentiles.
    b = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
    a = np.array([0.2, 0.2, 0.8, 0.1, 0.6])
    _m1, lo1, hi1 = paired_bootstrap(
        a, b, resample_count=2000, generator=np.random.default_rng(7)
    )
    _m2, lo2, hi2 = paired_bootstrap(
        a, b, resample_count=2000, generator=np.random.default_rng(7)
    )
    assert (lo1, hi1) == (lo2, hi2)  # exact


# ---------------------------------------------------------------------------
# holm_bonferroni (Requirements 7.6-7.8)
# ---------------------------------------------------------------------------


def test_holm_bonferroni_single_comparison_identity():
    # Req 7.6: family of one -> multiplier (m - 0) == 1, so the
    # adjusted value equals the raw value (clamped) -- the identity.
    assert holm_bonferroni([0.03]) == pytest.approx([0.03], abs=1e-9)


def test_holm_bonferroni_preserves_input_order():
    # Req 7.7: input order [0.04, 0.01, 0.03] differs from sorted order
    # [0.01, 0.03, 0.04] specifically to catch the map-back-to-input-
    # order bug (returning sorted-order values unmapped). Expected:
    # sorted ascending -> multipliers 3,2,1 -> [0.03, 0.06, 0.04] ->
    # running max -> [0.03, 0.06, 0.06] -> mapped back to input order
    # [0.06, 0.03, 0.06].
    assert holm_bonferroni([0.04, 0.01, 0.03]) == pytest.approx(
        [0.06, 0.03, 0.06], abs=1e-9
    )


def test_holm_bonferroni_ties_get_equal_adjusted_values():
    # Req 7.8: equal raw p-values must receive equal adjusted
    # p-values, independent of tied-comparison input order.
    adjusted = holm_bonferroni([0.02, 0.02])
    assert adjusted[0] == pytest.approx(adjusted[1], abs=1e-9)
