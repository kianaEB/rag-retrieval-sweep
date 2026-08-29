---
inclusion: always
---

# Evaluation integrity

Every rule here is mechanically checkable against a committed artifact —
not a matter of judgment.

- **Qrels are the only judge.** All retrieval metrics (recall@k,
  nDCG@10, MRR@10) are computed against the BEIR SciFact qrels. No
  model, including this one, judges relevance. There is no
  LLM-as-judge step and no manual relevance override anywhere in the
  pipeline.

- **Dataset stats come from the loader's own output, never hard-coded.**
  Corpus size, query count, and every other dataset statistic that
  appears anywhere (code, `README.md`, `SPEC.md`, `ANALYSIS.md`) must be
  read from what the loading code printed when it ran, not typed in
  from memory or from a prior run. If a number changes because the
  loader loaded fewer or more items, the written number changes too.

- **No number without a receipt.** No number appears in `README.md` or
  `ANALYSIS.md` unless it can be read out of `results/sweep.csv` or
  another committed artifact. If a claim needs a number that isn't in a
  committed file, either add it to the artifact or don't state it.

- **BM25 is the reference row.** BM25 is the reference row in every
  results table. Dense retriever results are always reported as a delta
  against BM25 (e.g. "+0.03 nDCG@10" or "-0.05 recall@10"), never as
  standalone numbers presented without the comparison.

- **nDCG@10 is the single primary metric, declared in advance.** It is
  named as primary before any results exist. recall@k and MRR@10 are
  reported for every configuration but are secondary. The headline
  finding in `README.md` is determined by nDCG@10 alone. Which metric
  is primary is never revised after results are seen. If there is ever
  a reason to change it, that reason is recorded in `SPEC.md` under
  threats to validity, along with what the headline would have been
  under the original choice.

- **The nDCG@10 comparison is the headline, whichever way it falls.**
  `README.md`'s first paragraph states the nDCG@10 comparison against
  BM25 together with whether the difference is distinguishable from
  noise, as determined by the pre-declared paired bootstrap. A
  difference that is not statistically significant is reported as
  "indistinguishable", never as a win for either side. A dense loss is
  never buried in a table or qualified away in a later section; neither
  is a dense win, and neither is a tie.

- **No row gets dropped for being unflattering.** Never drop, exclude,
  or filter a run from `results/sweep.csv` because its numbers are
  unflattering to a particular retriever or to the project. Every
  configuration in the declared grid produces a row, and every row that
  was produced stays in the file.
