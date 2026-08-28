"""Fixed-seed application for reproducibility (Requirement 8).

`apply_seed` is the single point where every source of randomness used
anywhere in the sweep is seeded, in a fixed order: Python's `random`
module, NumPy's global RNG, then `torch` via `torch.manual_seed(seed)`.
The `torch` seed also covers the Dense_Retriever's embedding batch
ordering, since `sentence_transformers.SentenceTransformer.encode()`
consumes the ambient torch RNG state for internal batching/padding
operations (see `design.md`'s `src/seeding.py` section).

The Sweep_Runner calls `apply_seed` before touching the corpus, the
config's own directories, or any retriever, and aborts the run without
writing `results/sweep.csv` if this raises (Requirement 8.5).
"""

from __future__ import annotations

import random

import numpy
import torch

from src.errors import SeedApplicationError


def apply_seed(seed: int) -> None:
    """Applies `seed` to `random`, `numpy`, and `torch`, in that order.

    Wraps any failure from the three underlying calls in
    `SeedApplicationError` so the Sweep_Runner has a single exception
    type to catch and abort on (Requirement 8.5).
    """
    try:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
    except Exception as exc:
        raise SeedApplicationError(f"failed to apply seed {seed!r}: {exc}") from exc
