"""JudgeModel: CPU-only NLI groundedness judging wrapper around
`cross-encoder/nli-deberta-v3-xsmall` (Requirement 6).

Loads `cross-encoder/nli-deberta-v3-xsmall` directly via
`transformers.AutoTokenizer` + `transformers.AutoModelForSequenceClassification`
-- not `sentence_transformers.CrossEncoder` -- so the label-index order
is read from the loaded model's own `model.config.id2label` at run time
rather than assumed from `sentence_transformers`' own softmax/label
conventions, which the requirements never reference. Same CPU-only,
same `data/hf_cache` root as `GeneratorModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.errors import JudgeModelLoadError, JudgeVerdictError
from src.groundedness_labels import ENTAILMENT_LABEL, NLI_LABEL_TO_VERDICT, Verdict


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    score: float  # entailment probability, softmax over the 3 logits, in [0.0, 1.0]


class JudgeModel:
    """Wraps cross-encoder/nli-deberta-v3-xsmall for CPU-only NLI
    groundedness judging (Requirement 6)."""

    def __init__(self, model_name: str, cache_folder: Path) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(cache_folder)
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir=str(cache_folder)
            ).to("cpu")
            self._model.eval()
        except Exception as exc:
            raise JudgeModelLoadError(
                f"failed to load Judge_Model {model_name!r} with "
                f"cache_folder {cache_folder}: {exc}"
            ) from exc

        # id2label -> label2idx, read from the model's own config
        # rather than a hard-coded logit position (Requirement 6.9).
        id2label = {
            int(i): label.lower() for i, label in self._model.config.id2label.items()
        }
        self._label2idx = {label: i for i, label in id2label.items()}
        if ENTAILMENT_LABEL not in self._label2idx:
            raise JudgeModelLoadError(
                f"loaded Judge_Model {model_name!r}'s id2label does not "
                f"expose an {ENTAILMENT_LABEL!r} class: {id2label!r}"
            )

    def count_premise_tokens(self, premise: str, hypothesis: str) -> int:
        """Returns the *untruncated* token count the premise/hypothesis
        pair would tokenize to -- i.e. the true input length before
        `judge()`'s `truncation=True` clamps it to
        `self._tokenizer.model_max_length`. Pure tokenization, no model
        inference; used to make premise truncation explicit and
        visible rather than silent.
        """
        return len(
            self._tokenizer(premise, hypothesis, truncation=False)["input_ids"]
        )

    @property
    def max_input_tokens(self) -> int:
        """The tokenizer's own `model_max_length` -- the input token
        budget `judge()`'s `truncation=True` clamps every
        premise/hypothesis pair to. Read from the loaded tokenizer,
        never hard-coded."""
        return int(self._tokenizer.model_max_length)

    def judge_best_sentence(
        self, sentences: List[str], hypothesis: str
    ) -> Tuple[JudgeResult, str, List[int]]:
        """Scores `hypothesis` (one Claim's text) against each of
        `sentences` (the Retrieved_Context split into individual
        sentences, via the same sentence-boundary segmenter
        `src.claim_segmenter.segment_claims` uses) independently, via
        `judge()`, and returns the `(JudgeResult, matched_sentence,
        dropped_token_counts)` triple for the sentence whose entailment
        probability is the maximum across all of `sentences`.

        This is a correctness fix, not a tuning change:
        `cross-encoder/nli-deberta-v3-xsmall` is trained on
        single-sentence premises, and scoring it against a
        multi-hundred-token concatenated-abstract premise (this
        wrapper's previous behavior) put every call outside its
        training distribution -- the likely dominant driver of the
        near-zero score distribution observed before this fix. Scoring
        against each sentence individually keeps every `judge()` call
        within that distribution; taking the *maximum* (rather than,
        e.g., the mean) reflects that a Claim is considered supported
        if *any* single retrieved sentence entails it, not only if the
        context as a whole, read all at once, does. The label mapping,
        the quarantine threshold, and the model choice are all
        unchanged by this fix.

        `dropped_token_counts` has one entry per sentence in
        `sentences` -- `max(0, count_premise_tokens(sentence,
        hypothesis) - max_input_tokens)` for that sentence -- so a
        caller can accumulate the same truncation-visibility statistics
        the previous per-document-premise design already recorded, now
        measured at the granularity actually scored.

        Raises `JudgeVerdictError` if `sentences` is empty (there is no
        premise to score the Claim against) or if any per-sentence
        `judge()` call fails.
        """
        if not sentences:
            raise JudgeVerdictError(
                "Judge_Model was given zero sentences to score a Claim "
                "against (Retrieved_Context produced no sentences)"
            )
        best_result = None
        best_sentence = ""
        dropped_token_counts: List[int] = []
        for sentence in sentences:
            token_count = self.count_premise_tokens(sentence, hypothesis)
            dropped_token_counts.append(max(0, token_count - self.max_input_tokens))
            result = self.judge(sentence, hypothesis)
            if best_result is None or result.score > best_result.score:
                best_result = result
                best_sentence = sentence
        assert best_result is not None  # sentences is non-empty, so the loop ran
        return best_result, best_sentence, dropped_token_counts

    def judge(self, premise: str, hypothesis: str) -> JudgeResult:
        """Scores whether `premise` (one sentence of the query's
        Retrieved_Context) entails `hypothesis` (one Claim's text) --
        the standard NLI direction for a support check: "does the
        context entail the claim" (Requirement 6.1).

        Tokenizes as a premise/hypothesis pair (the tokenizer's own
        pair-encoding, matching how this cross-encoder was trained),
        truncated to the tokenizer's own max length. Computes softmax
        over the 3 logits, reads the entailment class's probability via
        `self._label2idx[ENTAILMENT_LABEL]` as the Judge_Model score
        (Requirement 6.9), and determines the predicted native label
        via `argmax` over the same 3 logits (not re-derived from the
        softmax probabilities -- logits and their softmax share the
        same argmax, so this is equivalent but avoids a second,
        redundant computation), mapped to a Groundedness_Verdict via
        `NLI_LABEL_TO_VERDICT` (Requirement 6.2).
        """
        try:
            inputs = self._tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_idx = int(torch.argmax(logits).item())
            predicted_label = [
                label for label, idx in self._label2idx.items() if idx == predicted_idx
            ][0]
            verdict = NLI_LABEL_TO_VERDICT[predicted_label]
            score = float(probabilities[self._label2idx[ENTAILMENT_LABEL]].item())
            return JudgeResult(verdict=verdict, score=score)
        except Exception as exc:
            raise JudgeVerdictError(
                f"Judge_Model failed to produce a verdict: {exc}"
            ) from exc
