# SPEC

Design summary and threats to validity for `rag-retrieval-sweep`. This
is a plain-language design document for human readers; it is unrelated
to the `.kiro/specs/` folder mechanism used to build this repository.

## Design summary

### Sweep grid

The sweep is config-driven (`configs/sweep.yaml`), and the grid
actually applied to the reported run — read from
`results/run_config.json`'s `sweep_config` object — is: two
retrievers, `bm25` (`k1 = 1.50`, `b = 0.75`) and `all-MiniLM-L6-v2`;
evaluation cutoffs `1, 5, 10, 20`; and a single chunking strategy,
`whole_document`.

### Index once, retrieve once, slice four ways

Each retriever is indexed exactly once over the whole corpus, and
queried exactly once, at the deepest declared cutoff (`20`). Every
cutoff's recall@k value is computed by slicing that single returned
ranked list to the first `k` documents, rather than by issuing a
separate retrieval run per cutoff. nDCG@10 and MRR@10 are each
computed once per retriever, at a fixed cutoff of 10, and copied
unchanged into every one of that retriever's rows in
`results/sweep.csv`.

### Relevance ground truth

The BEIR SciFact qrels (human relevance judgments) are the sole source
of relevance ground truth used to compute every reported metric — no
model judgment and no manual override changes a qrels-derived
relevance determination anywhere in this pipeline.

### Primary and secondary metrics

nDCG@10 was designated the primary metric before any sweep result
existed. recall@k (for k = 1, 5, 10, 20) and MRR@10 are reported for
every configuration as secondary metrics.

### Statistical scheme

Every non-BM25 run is compared against the BM25 reference run with a
paired-bootstrap confidence interval and a paired two-sided permutation
p-value, over `resample_count = 10000` bootstrap resamples and
`permutation_count = 10000` permutations, drawn from a single
`numpy` random generator seeded with `bootstrap_seed = 20240`. The
nDCG@10 comparison family's raw p-values are then adjusted with a
Holm-Bonferroni step-down correction before the primary-metric verdict
is decided.

## nDCG@10 convention

nDCG@10 is computed, for each test query, as DCG@10 divided by IDCG@10,
defined as 0 when IDCG@10 is 0, where:

- **DCG@10** is the sum, over ranks i = 1 to 10 of the top 10 documents
  of the ranked list, of `rel_i / log2(i + 1)`, with `rel_i` equal to
  the graded relevance score recorded in the qrels for the document at
  rank i, or 0 if that document is unjudged.
- **IDCG@10** is the same sum computed over that query's qrels-judged
  relevant documents, sorted by descending relevance score and
  truncated to the first 10 documents.

The nDCG@10 value reported for a run is the arithmetic mean of the
per-query nDCG@10 value across all test queries that have at least one
qrels-judged relevant document.

This convention (the `log2(i + 1)` discount, graded relevance taken
directly from the qrels, and the 0-when-IDCG@10-is-0 rule) was fixed
before any sweep result existed and has not been altered after seeing
results.

## Threats to validity

### Sparse qrels

BEIR SciFact judges an average of **1.13** relevant documents per
query (339 judged query-document pairs over 300 test queries). A
document that is actually relevant to a query but was never judged is
scored as a miss regardless of whether a retriever surfaced it — this
penalizes a retriever that surfaces documents unlike the ones that
happened to get annotated, not necessarily a retriever that is
performing worse.

### BM25's sensitivity to preprocessing

BM25's score is sensitive to its tokenizer, stopword list, stemming
choice, and case handling, none of which are inherent to the BM25
algorithm itself. For this run, those choices were fixed once, before
any result existed, and never adjusted after seeing scores: tokenizer
`regex_word`, stopwords `none` (no stopword list applied), stemming
`none` (no stemming applied), lowercasing enabled, `k1 = 1.50`, and
`b = 0.75`. A different, equally defensible set of preprocessing
choices could move BM25's measured score in either direction; the
sensitivity itself, not just this run's fixed choices, is a threat to
validity worth naming.

### Single-corpus generalization

Every number in this document and in `README.md` describes BEIR
SciFact — a corpus of scientific claims — only. None of it is evidence
about how these retrievers would compare on another domain or corpus.

### Statistical power

The nDCG@10 comparison's 95% confidence interval has a half-width of
**0.0384**. A true difference between the two runs smaller than that
half-width could exist without this study's paired bootstrap being
able to detect it at this sample size — the "indistinguishable"
verdict reported in `README.md` and above reflects an inability to
distinguish the comparison from noise at this study's statistical
power, not proof that no difference exists between the two runs.

### BM25 query latency is an implementation artifact, not a property of lexical retrieval

The measured BM25 query latency in this run was **7.50** seconds
across all 300 test queries. That figure is a property of `rank_bm25`,
a pure-Python scoring implementation, not an inherent property of
lexical retrieval — a compiled or indexed lexical search engine (e.g.
Lucene-based) would be expected to score substantially faster. This
measurement should not be generalized to "lexical retrieval is slow."

### Token-length truncation under whole-document chunking

Measured directly against the cached `all-MiniLM-L6-v2` tokenizer over
every document in the loaded corpus (`results/token_length_report.json`):
**71.02%** of corpus documents exceed the model's 256-token maximum
sequence length under whole-document chunking. Because that fraction
is not a small, negligible one, this is stated as a concrete confound
in the headline comparison, not a minor caveat: the BM25 run
scores each document's full text, while the `all-MiniLM-L6-v2` run
scores only the first 256 tokens of that same document. The
"indistinguishable" nDCG@10 verdict reported above and in `README.md`
should be read with this asymmetry in mind — it is a confound in the
headline comparison, not evidence about either retriever's underlying
ranking quality.

## Groundedness gate

Session-3 stretch goal (`docs/PROJECT_BRIEF.md`): a small, local,
CPU-only generator (`google/flan-t5-base`) answers a seeded 30-query
subset of the queries the BM25 run (`bm25__whole_document`) already
scored, replaying that same frozen retriever at `replay_top_k = 10` to
obtain each query's Retrieved_Context. Each Generated_Answer is split
into Claims at sentence boundaries, and a second, distinct model — the
Judge_Model, `cross-encoder/nli-deberta-v3-xsmall` — scores each Claim
for entailment against that same Retrieved_Context. The two models are
required to differ because a model judging its own generated output is
biased toward accepting it; the Judge_Model never sees the Qrels, and
never checks a Claim against anything other than the Retrieved_Context
the Generator_Model was also shown.

### Judging granularity

`nli-deberta-v3-xsmall` is trained on single-sentence NLI premises. An
earlier version of this pipeline scored each Claim against the entire
10-document Retrieved_Context concatenated into one premise (up to
several thousand tokens), which put every judgment outside the
Judge_Model's training distribution and produced a near-uniformly low
score distribution unrelated to actual support. The Judge_Model now
splits the Retrieved_Context into individual sentences (the same
sentence-boundary rule the Claim_Segmenter applies to the
Generated_Answer) and scores the Claim against each sentence
independently, taking the maximum entailment probability as
`judge_score`. `results/groundedness.csv`'s `matched_sentence` column
records which retrieved sentence produced that maximum, so any row can
be audited without a re-run.

### Label mapping and score definition

The Judge_Model's native three-way NLI label maps to a binary
Groundedness_Verdict as declared in `configs/groundedness.yaml`'s
`label_mapping` field, cross-validated against
`src/groundedness_labels.py`'s hard-coded constant at config-load
time: `entailment` maps to `SUPPORTED`; `neutral` and `contradiction`
both map to `NOT_SUPPORTED` — a Claim the Retrieved_Context is merely
silent on is withheld exactly like a Claim it actively contradicts.
`judge_score` is defined, per `configs/groundedness.yaml`'s
`score_definition` field, as the entailment probability obtained by
applying softmax to the Judge_Model's three logits, a value in `[0.0,
1.0]` where higher indicates stronger support. A Claim is quarantined
(withheld from the Served_Answer) if its verdict is `NOT_SUPPORTED`
regardless of score, or if its verdict is `SUPPORTED` and its score
falls strictly below `quarantine_threshold = 0.50`, also declared in
`configs/groundedness.yaml`.

### Generation degeneracy

Two decoding problems were found and addressed, not hidden. First,
`transformers.generate()` defaults to a 20-token output cap when
`max_new_tokens` is not supplied, which cut every answer off mid-word;
`max_new_tokens = 128` is now declared in `configs/groundedness.yaml`.
Second, greedy decoding produced at least one repetition-collapsed
answer (a clause repeated back-to-back); `no_repeat_ngram_size = 3`
and `repetition_penalty = 1.3` are now declared and passed to
`generate()`. Both are deterministic decoding constraints — neither
introduces sampling randomness, so a rerun on the same machine still
reproduces byte-identical answers. After both fixes, a distinct
degeneracy remains and is recorded rather than tuned away: the fraction
of Generation_Subset Claims that are a byte-for-byte copy of the
retrieved sentence the Judge_Model matched them against
(`results/groundedness.csv`: `claim_text` equal to `matched_sentence`)
is **0.0333** (1 of 30), and several more Claims are close paraphrases
of a retrieved sentence without being byte-identical —
`google/flan-t5-base`, shown a prompt many times its own 512-token
input budget (see below), regresses toward reproducing salient input
text rather than synthesizing a new sentence. This is a property of a
small model under acute context-window pressure, not a bug in this
pipeline's decoding parameters.

### Prompt and premise truncation

`replay_top_k = 10` matches the sweep's own retrieval depth rather
than declaring a second, narrower top-k just for generation.
Both `google/flan-t5-base` and `cross-encoder/nli-deberta-v3-xsmall`
have a 512-token input budget (`generator_max_input_tokens` /
`judge_max_input_tokens`, `results/run_config.json`'s `"groundedness"`
sub-object). At the document level, this truncated every one of the 30
Generation_Subset prompts: `generator_prompt_truncation` reports
30/30 prompts truncated, dropping a mean of 3632 tokens per truncated
prompt. Shrinking `replay_top_k` does not avoid this — measured
directly, even `replay_top_k = 1` already exceeds the 512-token budget
for 8 of the 30 queries, and `replay_top_k = 3` exceeds it for all 30
(SciFact abstracts run longer under the flan-t5/DeBERTa tokenizers
than a quick character-count estimate would suggest) — so the
truncation is recorded per run rather than hidden behind a smaller
top-k that still truncates. The Judge_Model, in contrast, is scored
per sentence (see "Judging granularity" above), and at that
granularity `judge_premise_truncation` reports 0 of 3090 per-sentence
premises truncated: the sentence-level fix that corrects the judging
granularity mismatch also happens to eliminate judge-side truncation
entirely, since individual sentences fall well under 512 tokens.

### Quarantine_Rate

Of the 30 Claims produced for the Generation_Subset (one per query),
**11** were quarantined, for a Quarantine_Rate of **0.3667**
(`results/groundedness.csv`: `quarantine_decision = True` for 11 of
30 rows). This is a model-graded quantity with no human ground truth,
unlike recall@k, nDCG@10, and MRR@10 above, each of which is computed
against BEIR SciFact's Qrels — Quarantine_Rate reflects support
against Retrieved_Context only, and a Claim can be judged `SUPPORTED`
while its Retrieved_Context is itself not relevant to the query under
the Qrels; groundedness and retrieval relevance are two different
questions. Quarantine_Rate is never stated without the two numbers
below.

### Agreement_Rate: the only human anchor Quarantine_Rate has

A Hand_Checked_Sample of 30 Claims (every Claim produced for the
Generation_Subset, since `hand_checked_sample_size = 30` meets or
exceeds the population — Requirement 10.3) was selected independently
of the Judge_Model's own verdicts, scores, or quarantine decisions
(`select_hand_checked_sample` takes only Claim identity and a seed;
`results/hand_checked_sample.csv` never carried the Judge_Model's
verdict, score, or quarantine decision, so the human labelling from it
was never anchored by the Judge_Model's own determination), exported
to `results/hand_checked_sample.csv`, and hand-labelled by a human
reviewer using `results/hand_checked_sample_context.md` (query text,
Claim text, and Retrieved_Context only — the same reading aid
deliberately excludes the Judge_Model's verdict, score, matched
sentence, and the Generated_Answer). Against that hand-labelled
sample, `results/hand_checked_joined.csv` reports a pooled Agreement_Rate
of **0.2667** (the fraction of the 30 Claims where the Judge_Model's
verdict matches the human hand label). That pooled number, by itself,
overstates how much it says about judge quality: the Judge_Model's
criterion and the human reviewer's criterion are not the same
criterion, and most of the disagreement is concentrated exactly where
they diverge, not spread evenly across the sample.

The Judge_Model's criterion is textual entailment of the Claim by some
sentence of the Retrieved_Context — nothing more. The human reviewer's
criterion, agreed informally with the reviewer after the Claims
already existed (not declared in this spec before labelling — see
"Threats to validity" below), additionally treated a Claim that is not
a declarative assertion — a copied document title, a bare noun phrase,
a sentence fragment — as `NOT_SUPPORTED` regardless of textual overlap
with the Retrieved_Context, on the reasoning that a non-answer cannot
be "supported" in the sense Quarantine_Rate is meant to protect a
Served_Answer's reader from. Concretely: of the 16 Claims where the
Judge_Model said `SUPPORTED` and the human said `NOT_SUPPORTED`,
inspection shows they are concentrated on Claims that are copied
document titles or bare noun-phrase fragments (e.g. `"cathelicidins
and proBac7."`, `"HNF4A gene mutations."`, `"Aliskiren and losartan in
type 2 diabetes and nephropathy."`) — the Judge_Model correctly found
textual entailment, and the human reviewer declined to call a
non-answer supported under a criterion the Judge_Model was never given.

The 16 judge-`SUPPORTED`/human-`NOT_SUPPORTED` cases above do not, on
their own, indict the Judge_Model's entailment judgment — a majority
of them reflect the two criteria diverging on non-assertion Claims
(see the partition below). The **6** judge-`NOT_SUPPORTED`/human-
`SUPPORTED` cases are the disagreement that *does* bear on judge
quality: there, the human reviewer found support in the
Retrieved_Context that the Judge_Model's entailment check missed.
Reported separately from the 16, this is the more direct signal that
the Judge_Model under-detects support on some real assertions,
independent of the non-assertion criterion mismatch above.

### Agreement_Rate, partitioned by declarative assertion vs. non-assertion

A pooled 0.2667 conflates the two disagreement mechanisms above into
one number. Partitioning the 30 Hand_Checked_Sample Claims makes the
split itself the finding. The partition rule: **a Claim is a
declarative assertion if its text contains a finite main verb, making
it a declarative assertion rather than a title, noun phrase, or
fragment.** A Claim with no finite main verb is a **non-assertion**.

This rule is applied via a committed, per-claim classification file,
`docs/claim_assertion_classification.csv` — not a mechanical
heuristic. An earlier version of this analysis used a fixed
marker-word list (is/are/was/has/may/...) as a mechanical proxy for
"contains a finite verb," which caught copular sentences ("CHD4 *is*
a ...") but missed ordinary present-tense assertions whose only verb
is a bare main verb ("Sildenafil *improves* ...", "Aspirin *inhibits*
...", "CD11b+ monocytes *abrogate* ..."), misclassifying 7 of the 30
Claims as non-assertions. `docs/claim_assertion_classification.csv`
replaces that heuristic with an explicit classification obtained by
reading each of the 30 committed Claim texts individually. Unlike
`hand_label`, this classification is a grammatical fact about the
committed `claim_text` — every row is mechanically auditable by
reading the cited `claim_text` and checking it for a finite main verb,
without needing a human reviewer's judgment call or a re-run — but,
like the partition analysis as a whole, it was applied after the hand
labels already existed, not declared before labelling (see "Threats
to validity" below).

Under this partition (`results/hand_checked_joined.csv`'s
`is_declarative_assertion` column, read from the committed
classification file by the Groundedness_Runner and never re-derived
elsewhere):

- **Declarative assertions:** 19 of the 30 Claims. Agreement_Rate
  **0.3684** (7 of 19).
- **Non-assertions:** 11 of the 30 Claims. Agreement_Rate **0.0909**
  (1 of 11).

This split is sharper than the pooled 0.2667 suggests, and sharper
than an earlier, since-corrected version of this partition (which had
misclassified 7 assertions as non-assertions and reported 0.3333 /
0.2222). On non-assertions, agreement nearly disappears: of 11
non-assertion Claims, the Judge_Model said `SUPPORTED` for 10 of them,
and the human reviewer agreed on only 1 (a Claim both correctly called
`NOT_SUPPORTED`) — the human reviewer never once agreed that a
non-assertion Claim was `SUPPORTED`, because the human criterion
treats every non-assertion as `NOT_SUPPORTED` by construction, while
the Judge_Model's textual-entailment criterion finds most of them
textually entailed by some retrieved sentence (they are frequently
close paraphrases or copies of a retrieved title). The two criteria
are not merely different in the abstract — on this sample, they are
almost maximally opposed on non-assertion Claims specifically, which
is exactly what "the disagreement is concentrated where the criteria
diverge" means concretely.

On declarative assertions, by contrast, the criterion mismatch above
does not apply — both raters were answering the same question (does
the Retrieved_Context support this Claim?), not two different ones.
The assertion-partition Agreement_Rate of **0.3684** (7 of 19) is
below the ~0.5 that two raters would reach by chance alone on a binary
label, and the 12 disagreements are balanced rather than one-sided: 6
are judge-`SUPPORTED`/human-`NOT_SUPPORTED` and 6 are
judge-`NOT_SUPPORTED`/human-`SUPPORTED`. Unlike the non-assertion
partition, there is no criterion-mismatch story available here to
explain the low rate — if anything, a rate below chance agreement on
a shared question is a genuine signal that the Judge_Model's
entailment check and the human reviewer's support judgment track each
other poorly on real assertions, not merely that they are answering
different questions.

That said, n = 19 is small enough that 0.3684 itself should not be
read as a precise estimate of judge quality. A 95% Wilson score
interval around 7 successes out of 19 trials spans **0.19 to 0.59** —
wide enough to be compatible with the judge tracking the human
somewhat worse than chance, at chance, or somewhat better, and this
study cannot distinguish those possibilities at this sample size. This
is the same
statistical-power limitation already named for the nDCG@10 comparison
in "Threats to validity" above (a 95% confidence interval half-width
of 0.0384 on 300 queries) — here it is sharper only because n = 19 is
far smaller than n = 300, not because a different kind of uncertainty
is at play. The assertion-partition Agreement_Rate is reported for
transparency about where the disagreement concentrates, not as a
performance estimate of the Judge_Model.

### Limitations this Quarantine_Rate inherits

Beyond the Agreement_Rate discussion above, four further limitations
bound what this Quarantine_Rate measures:

- **The Claim_Segmenter's sentence-boundary heuristic** splits on
  `.`/`!`/`?` followed by whitespace or end-of-string — a crude rule,
  not a solved natural-language-processing problem. A mis-split
  sentence changes what counts as one Claim.
- **The Generator_Model/Judge_Model separation** reduces, but does not
  eliminate, judge bias — the two models were trained on different
  data by different teams, but neither is a substitute for a human
  reviewer, which is exactly why the Hand_Checked_Sample above exists.
- **Retrieved_Context is not the Qrels.** The Judge_Model checks a
  Claim against whatever BM25 retrieved for that query at
  `replay_top_k = 10`, not against BEIR SciFact's human relevance
  judgments. A Claim can be `SUPPORTED` by a Retrieved_Context that is
  itself irrelevant under the Qrels; Quarantine_Rate says nothing
  about retrieval quality.
- **Neither the human hand-labelling criterion nor the assertion/
  non-assertion partition was pre-registered.** Unlike nDCG@10
  (declared primary before any results existed) and the Holm
  correction scheme (fixed in `configs/significance.yaml` before the
  bootstrap ran), the human reviewer's "a non-assertion cannot be
  supported" criterion was agreed informally after the 30 Claims
  already existed, not declared in `configs/groundedness.yaml` or
  anywhere else before labelling began. The declarative-assertion
  classification in `docs/claim_assertion_classification.csv` was
  applied after that, later still — after the hand labels themselves
  already existed. Because of this, Agreement_Rate — pooled or
  partitioned — is not a clean measure of Judge_Model quality in the
  way a pre-registered comparison would be: a differently-agreed
  labelling criterion or a differently-drawn assertion/non-assertion
  line, either decided in advance, could move both partition rates.
  The one difference between the two: the labelling criterion is a
  human judgment call that cannot be independently re-checked without
  a human, while the assertion/non-assertion classification is a
  grammatical fact about the committed `claim_text` that anyone can
  audit row-by-row against `results/groundedness.csv` without
  consulting a human reviewer or re-running anything. The partition
  above is offered as a post-hoc explanation of where the disagreement
  concentrates, not as a validated quality metric.

### Generation and hand-checked sample sizes

The Generation_Subset (`generation_subset_size = 30`,
`generation_subset_seed = 4242`) is sampled from the query IDs the
Replayed_Run (`bm25__whole_document`) actually scored, per
`results/per_query.csv`. The Hand_Checked_Sample
(`hand_checked_sample_size = 30`, `hand_checked_sample_seed = 777`,
distinct from `generation_subset_seed`) draws from the Claims that
Generation_Subset produced.
