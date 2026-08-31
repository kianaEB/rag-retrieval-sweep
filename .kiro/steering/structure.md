---
inclusion: always
---

# Structure

## Layout

```
rag-retrieval-sweep/
├── configs/          # YAML/config files declaring the sweep grid
│                      (retriever x top-k x chunking). The grid is data,
│                      not hard-coded loops.
├── data/             # gitignored cache for downloaded BEIR corpora
│                      and model weights — downloads never scatter into
│                      the repo root.
├── docs/
│   ├── PROJECT_BRIEF.md
│   └── numeric_traceability.csv    # Numeric_Claim ledger: every number
│                                     in README.md/SPEC.md mapped to the
│                                     committed artifact it was read from
│                                     (repo-writeup spec)
├── results/
│   ├── sweep.csv                   # one row per configuration: metrics,
│   │                                 timing, failure bucket
│   ├── token_length_report.json    # all-MiniLM-L6-v2 tokenizer
│   │                                  truncation check: fraction of
│   │                                  corpus documents exceeding its
│   │                                  256-token limit (repo-writeup spec)
│   ├── groundedness.csv            # one row per (query_id, claim_index)
│   │                                  claim: the judge's verdict, score,
│   │                                  and quarantine decision
│   │                                  (groundedness-gate spec)
│   ├── generated_answers.csv       # one row per Generation_Subset
│   │                                  query: the raw Generated_Answer
│   │                                  plus its prompt's untruncated
│   │                                  token count, so a Claim traces
│   │                                  back to the text it was segmented
│   │                                  from (groundedness-gate spec)
│   └── hand_checked_sample.csv     # exported-then-re-imported
│                                      hand-labelling sample, carrying
│                                      the judge's verdict alongside the
│                                      human hand label for the
│                                      Agreement_Rate check
│                                      (groundedness-gate spec)
├── src/               # loading, retrievers, chunking, metrics,
│                      the sweep entry point
├── tests/            # pytest over metric functions and the data layer
├── requirements.txt  # pinned exact versions
├── README.md         # results table + honest headline finding
├── SPEC.md           # design + threats to validity
├── ANALYSIS.md        # mechanism / failure-bucket analysis
└── .github/
    └── workflows/    # CI: install requirements, run pytest
```

- One entry point runs the sweep end-to-end from a config file, writing
  `results/sweep.csv`.
- The corpus loading code reports the exact counts it loads (documents,
  queries, qrels) — counts are never hard-coded, so a silent truncation
  or bad download is visible immediately.
- Every row of `results/sweep.csv` includes the retriever, top-k,
  chunking strategy, the metrics, wall-clock index time, query latency,
  and the failure bucket — the analysis in `ANALYSIS.md` is derived from
  this file, not from separate ad hoc scripts.

## Build order (from the brief)

1. **Session 1** — Load SciFact + qrels; BM25 and one dense retriever;
   whole-document chunking only; recall@k / nDCG / MRR; write
   `results/sweep.csv`; seeded; `requirements.txt`; one entry point;
   config-driven grid from day one.
2. **Session 2** — Full grid (all retrievers x all top-k x all chunking
   strategies); failure bucketing; `ANALYSIS.md`; pytest over the metric
   functions and the data layer; GitHub Actions CI; `README.md` with the
   results table and the honest headline; `SPEC.md` with design and
   threats to validity.
3. **Session 3 (stretch)** — groundedness gate over generated answers,
   quarantine rate, hand-checked sample.

## Definition of done

The repo does not count as done until it ships all of the following:

- `README.md` — results table and the honest headline finding.
- `SPEC.md` — design and threats to validity.
- `ANALYSIS.md` — the mechanism / failure-bucket analysis.
- `tests/` — pytest coverage over the metric functions and the data
  layer.
- A passing GitHub Actions workflow (CI installs `requirements.txt` and
  runs the test suite on every push) — shipped as
  `.github/workflows/ci.yml`, triggered on `push` and `pull_request`,
  running on `ubuntu-latest` with Python 3.13. It installs
  `requirements.txt` and runs `pytest` only — no sweep, no dataset
  download, no model download, consistent with the network-free test
  suite (session-1 Requirement 11.3, significance-testing Requirement
  7.7).

`SPEC.md` must contain a "Threats to validity" section, and that section
must name at minimum:

- **Sparse qrels** — unjudged documents are scored as non-relevant,
  which penalises retrievers that surface documents unlike those that
  were annotated.
- **Sensitivity of BM25 to preprocessing choices** — tokenizer,
  stopwords, stemming, and case handling all move BM25's score; the
  choices are fixed once (see tech.md) but the sensitivity itself is a
  threat to validity worth naming.
- **Single-corpus generalisation** — SciFact is scientific claims;
  results may not transfer to other domains.

Missing any one of these means the project is still in progress,
regardless of how complete the sweep results look.
