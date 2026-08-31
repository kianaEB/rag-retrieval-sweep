"""GeneratorModel: CPU-only, greedy-decoded answer generation wrapper
around `google/flan-t5-base` (Requirement 4).

Loads via `transformers.AutoTokenizer` + `transformers.AutoModelForSeq2SeqLM`
-- T5 is encoder-decoder, not causal, so `AutoModelForCausalLM` would be
the wrong class. `device` is hard-coded to `"cpu"`, never conditional
on CUDA availability, mirroring `DenseRetriever.__init__`'s hard-coded
`device="cpu"` (Requirement 4.3). `cache_folder` is the same
`data_dir / "hf_cache"` root `configure_caches()` already points
`HF_HOME`/`HF_HUB_CACHE` at -- no second cache root is introduced.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.errors import GeneratorGenerationError, GeneratorModelLoadError


class GeneratorModel:
    """Wraps google/flan-t5-base for CPU-only, greedy-decoded answer
    generation (Requirement 4)."""

    def __init__(self, model_name: str, cache_folder: Path) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(cache_folder)
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=str(cache_folder)
            ).to("cpu")
        except Exception as exc:
            raise GeneratorModelLoadError(
                f"failed to load Generator_Model {model_name!r} with "
                f"cache_folder {cache_folder}: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Produces exactly one Generated_Answer for `prompt`
        (Requirement 4.4).

        `max_new_tokens` is passed through explicitly to
        `self._model.generate(...)` -- `transformers` defaults
        `max_length` to 20 tokens when neither `max_length` nor
        `max_new_tokens` is supplied, which silently cut every
        Generated_Answer off mid-word (and, because there was no
        sentence-ending punctuation left to split on, collapsed every
        Claim_Segmenter call to exactly one Claim per query). Declared
        as data in `configs/groundedness.yaml` like every other
        generation parameter, and recorded in `run_config.json`'s
        `"groundedness"` sub-object.

        `no_repeat_ngram_size` (0 disables it) and `repetition_penalty`
        (1.0 disables it) address greedy-decoding repetition collapse
        (observed directly: a Generated_Answer that repeats the same
        clause back-to-back) and title-copying degeneracy (a small
        model, shown a prompt many times its own context window, often
        falls back to reproducing a retrieved document's title
        verbatim). Both are deterministic, seed-free decoding
        constraints -- `no_repeat_ngram_size` forbids repeating any
        n-gram of that size, `repetition_penalty` down-weights the
        logits of already-generated tokens -- neither introduces
        sampling randomness, so Requirement 4.4's byte-for-byte-
        identical-rerun guarantee is unaffected. Declared as data in
        `configs/groundedness.yaml`, defaulting to a no-op (0 / 1.0)
        so an explicit opt-in is always visible in the config rather
        than silently changing decoding behavior.

        Uses greedy decoding (`do_sample=False`, `num_beams=1`) --
        deliberately, and exclusively, because greedy decoding is
        deterministic by construction: the same input_ids always
        produce the same output token sequence on the same model
        weights, with no sampling RNG involved anywhere in the decode
        loop. This is what satisfies Requirement 4.4's byte-for-byte-
        identical-rerun guarantee without introducing a third seed the
        requirements never declare (Generation_Subset_Seed and
        Hand_Checked_Sample_Seed are the only two seeds Requirement 1
        names) -- there is no generation-time randomness to seed in
        the first place.

        `truncation=True` (with no explicit `max_length`) still applies
        to the *encoder input* here, clamped to the tokenizer's own
        `model_max_length` -- a separate, unavoidable axis from
        `max_new_tokens`, which controls only the *decoder output*
        length. A prompt longer than the tokenizer's max length is
        still silently truncated at the input side; see
        `count_tokens` below, which lets a caller measure the
        untruncated prompt length for that visibility.
        """
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
            generate_kwargs = dict(
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            if no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            output_ids = self._model.generate(**inputs, **generate_kwargs)
            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as exc:
            raise GeneratorGenerationError(
                f"Generator_Model failed to produce a Generated_Answer: {exc}"
            ) from exc

    def count_tokens(self, text: str) -> int:
        """Returns the *untruncated* token count `text` would tokenize
        to -- i.e. the true input length before `generate()`'s
        `truncation=True` clamps it to `self._tokenizer.model_max_length`.
        Pure tokenization, no model inference; used to make prompt
        truncation explicit and visible (e.g.
        `results/generated_answers.csv`'s `prompt_token_count` column)
        rather than silent.
        """
        return len(self._tokenizer(text, truncation=False)["input_ids"])

    @property
    def max_input_tokens(self) -> int:
        """The tokenizer's own `model_max_length` -- the input token
        budget `generate()`'s `truncation=True` clamps every prompt to.
        Read from the loaded tokenizer, never hard-coded, so it always
        reflects what the actually-loaded model supports."""
        return int(self._tokenizer.model_max_length)
