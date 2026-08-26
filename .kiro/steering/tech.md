---
inclusion: always
---

# Tech

## Libraries

The sweep is built from small, boring, CPU-friendly libraries — no
frameworks, no orchestration layers.

- **beir** — loads the SciFact corpus, queries, and qrels (human
  relevance judgments) in the standard BEIR format. This is the only
  source of ground truth for scoring.
- **rank_bm25**, specifically `BM25Okapi` — the lexical baseline. This
  is the one pinned implementation; no Elasticsearch, no Lucene, no
  external search service. `k1` and `b` are recorded in `configs/`
  alongside the tokenizer settings.
- **sentence-transformers** — runs `all-MiniLM-L6-v2` and
  `BAAI/bge-small-en-v1.5` for dense retrieval. Models run locally on
  CPU; weights are downloaded once from the Hugging Face Hub (a
  one-time, free download, not a paid inference API).
- **numpy** — vector math for dense similarity scoring.
- **pandas** — assembling and writing `results/sweep.csv`.
- **PyYAML** (or similar) — config-driven grid definitions under
  `configs/`, so retriever x top-k x chunking combinations are declared
  as data, not hard-coded loops.
- **pytest** — tests over the metric functions and the data-loading
  layer.
- **GitHub Actions** — CI that installs `requirements.txt` and runs
  `pytest` on every push.

Pin exact versions in `requirements.txt` as they're added — no open
ranges.

## Hard constraints

These are non-negotiable properties of this project, not preferences:

- **CPU-only.** Every retriever, every model, every step of the sweep
  must run on a CPU. No CUDA-only code paths, no assuming a GPU is
  available. Model choices (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`) are
  already picked for CPU feasibility — don't swap in larger models that
  require a GPU to run in reasonable time.
- **No paid API calls.** No OpenAI, Anthropic, Cohere, or any other
  metered inference endpoint. Everything runs from locally-executed,
  free, open-weight models. If a step seems to need a paid API, that's a
  sign it's out of scope for this repo.
- **Fixed random seed.** Every source of randomness (embedding batch
  order, any sampling in chunking or evaluation) is seeded, and the seed
  is recorded. The metric columns in `results/sweep.csv` (recall@k,
  nDCG@10, MRR@10) must be identical across reruns on the same machine
  with the same seed — that's the reproducibility bar. Timing columns
  (index time, query latency) are expected to vary run to run and
  machine to machine; they are never compared across machines and never
  used as a result in their own right.
- **No network access inside tests.** `pytest` runs must not download
  datasets or models, and must not make any network call. Tests use
  small fixtures / cached / pre-fetched data, not live BEIR or Hugging
  Face Hub downloads. If a test needs data, it needs a local fixture.
- **Metrics computed only against BEIR qrels.** recall@k, nDCG@10, and
  MRR@10 are computed strictly against the human relevance judgments
  shipped with the BEIR SciFact dataset. No self-graded relevance, no
  LLM-as-judge, no heuristic substitute for ground truth. If qrels don't
  cover a query/document pair, it is not relevant — full stop.
- **Lexical preprocessing is declared once, not tuned.** The BM25
  tokenizer, stopword list, stemming choice, and case handling are
  declared once in `configs/`, recorded in `results/sweep.csv` (or the
  run config), and documented in `SPEC.md`. These choices are fixed
  before the sweep runs and never adjusted per-run or after seeing
  results — changing preprocessing after seeing scores is p-hacking the
  baseline. The same text normalization pipeline is applied to both
  queries and documents. Switching the BM25 implementation, or changing
  `k1`/`b` after seeing results, is the same violation as tuning
  preprocessing after seeing results — it does not happen.
- **Downloads are pinned to `data/`.** The code sets the Hugging Face
  and BEIR cache directories explicitly to `data/` (e.g. via the
  `sentence-transformers` `cache_folder` argument and the relevant
  environment variables, such as `HF_HOME` / `TRANSFORMERS_CACHE`).
  Nothing downloads to a default location outside the repo.
- **CI runs pytest only.** The GitHub Actions workflow installs
  `requirements.txt` and runs `pytest`. It never runs the full sweep,
  never downloads BEIR datasets, and never downloads model weights. CI
  exists to catch regressions in the metric and data-loading code, not
  to reproduce results.
