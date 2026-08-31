"""Hard-coded Judge_Model native-label-to-Groundedness_Verdict mapping.

This is the single source of truth in *code* for Requirement 6
Criterion 2's native-label-to-verdict mapping, and the value
`groundedness_config.py` cross-validates the YAML's documented record
against at load time. Standard library only -- no `transformers`
import here, so this module stays importable from a pure-function test
without pulling in any model dependency.
"""

from __future__ import annotations

from typing import Dict, Literal

Verdict = Literal["SUPPORTED", "NOT_SUPPORTED"]

# The single hard-coded mapping from the Judge_Model's native 3-way NLI
# label to the Groundedness_Verdict (Requirement 6.2). This dict is the
# code-side half of the "declared once, cross-validated against the
# YAML's documented record, never revised after any Quarantine_Rate
# exists" contract (Requirement 6.3) -- see
# groundedness_config.py's _validate_label_mapping.
NLI_LABEL_TO_VERDICT: Dict[str, Verdict] = {
    "entailment": "SUPPORTED",
    "neutral": "NOT_SUPPORTED",
    "contradiction": "NOT_SUPPORTED",
}

# The native NLI label whose softmax probability is the Judge_Model
# score (Requirement 6.9) -- entailment probability, read back through
# the loaded model's own id2label at run time (never a hard-coded
# logit index; see judge_model.py).
ENTAILMENT_LABEL = "entailment"
