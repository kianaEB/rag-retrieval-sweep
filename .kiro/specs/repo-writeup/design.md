# Design Document: Repo Writeup

## Overview

This design covers three things, all in service of `requirements.md`'s
12 requirements:

1. **A new, small analysis script** — `src/token_length_analysis.py` —
   that tokenizes the loaded SciFact corpus with the already-cached
   `all-MiniLM-L6-v2` tokenizer and writes a new committed artifact,
   `results/token_length_report.json` (the Token_Length_Report,
   Requirement 11). This is the only new *code* this spec introduces.
2. **Two hand-authored markdown documents at the repository root** —
   `README.md` and `SPEC.md` (Requirement 1) — whose every Numeric_Claim
   is read from one of five committed artifacts: `results/sweep.csv`,
   `results/per_query.csv`, `results/significance.csv`,
   `results/run_config.json`, and the new `results/token_length_report.json`.
3. **A Verification_Pass mechanism** (Requirement 12) — a committed
   traceability ledger, `docs/numeric_traceability.csv`, plus a small
   checking script, `src/verify_writeup_numbers.py`, that checks every
   ledgered Numeric_Claim two ways: that its stated value still appears
   verbatim in the document that cites it, and that the value
   mechanically re-derived from its cited artifact agrees with it —
   flagging either kind of mismatch, distinguishably.

Nothing in this design touches `src/corpus_loader.py`, `src/metrics.py`,
`src/config.py`, `src/report.py`, `src/sweep_runner.py`,
`src/significance.py`, `src/significance_config.py`,
`src/per_query_report.py`, or `src/retrievers/base.py` — per
`requirements.md`'s introduction, "no sweep code, no metrics code, and
no behavioral change to any retriever are introduced by this spec."
The two new modules *import* several of those (`configure_caches`,
`load_scifact`, `load_sweep_config`, `_atomic_write_text`,
`_json_default`) but never modify them.

The one narrow exception is `src/retrievers/dense_retriever.py`: it
gains exactly one new module-level function, `format_document_text`,
extracted verbatim from the f-string that already lives inline inside
`DenseRetriever.build_index`, with no other line, method, or behavior
change. `src/token_length_analysis.py` then imports and calls that same
function directly, rather than re-typing its logic a second time. This
is a single extract-and-import refactor, in scope because it makes
Requirement 11.1's "exactly as the dense retriever encodes it" true by
construction — a single shared function used by both call sites —
rather than a claim resting on two independently-maintained copies of
the same string-formatting logic staying in sync by discipline alone.

The design is organized around one non-negotiable data flow: **every
number in `README.md`/`SPEC.md` traces to exactly one artifact field,
and that trace is itself a committed, re-checkable row**, not a claim
resting on the author's memory of having looked something up once.
Concretely:

- `results/sweep.csv`, `results/per_query.csv`, `results/significance.csv`,
  and `results/run_config.json` already exist and are read-only inputs
  to this spec — nothing regenerates or edits them (`requirements.md`'s
  introduction: "This spec reports on those artifacts; it does not
  regenerate them").
- `results/token_length_report.json` is the one artifact this spec's
  code produces, and it is produced by reusing session 1's own
  cache-configuration and corpus-loading code, not by re-implementing
  either.
- `README.md`/`SPEC.md` are authored directly, by hand — this design
  deliberately does **not** introduce a templating engine or a
  markdown-generation script (see "Why README.md/SPEC.md are hand-
  authored, not generated" below).
- Every Numeric_Claim placed into either document is, at authoring
  time, also added as one row to `docs/numeric_traceability.csv`
  (artifact + field + stated precision). `src/verify_writeup_numbers.py`
  is the Verification_Pass: for each row, it first confirms the stated
  value still appears verbatim in the cited document, then re-derives
  the value from the cited artifact and confirms agreement at the
  row's stated precision, using round-half-up rounding (Requirement
  12.2) — checking the claim against both files it names, per
  Requirement 12.2's "in both files" wording.

## Architecture

### Module layout

```
configs/
  sweep.yaml                        # unchanged (session 1). Also the source
                                     # of data_dir and the dense retriever's
                                     # model_name for token_length_analysis.py.
  significance.yaml                 # unchanged (significance-testing spec)

docs/
  PROJECT_BRIEF.md                  # unchanged
  numeric_traceability.csv          # NEW: the Numeric_Claim ledger (Requirement 12)

src/
  __init__.py
  errors.py                         # unchanged
  config.py                         # unchanged
  seeding.py                        # unchanged (NOT used by this spec's new code)
  corpus_loader.py                  # unchanged; REUSED by token_length_analysis.py
  metrics.py                        # unchanged
  report.py                         # unchanged; REUSED (_atomic_write_text, _json_default)
  per_query_report.py               # unchanged
  significance.py                   # unchanged
  significance_config.py            # unchanged
  sweep_runner.py                   # unchanged
  token_length_analysis.py          # NEW: Token_Length_Analysis entry point (Requirement 11)
  verify_writeup_numbers.py         # NEW: Verification_Pass entry point (Requirement 12)
  retrievers/
    __init__.py
    base.py                         # unchanged
    bm25_retriever.py                # unchanged
    dense_retriever.py               # gains ONE new module-level function,
                                     # format_document_text, extracted from
                                     # build_index's existing inline f-string
                                     # (no behavioral change). IMPORTED
                                     # (not duplicated) by
                                     # token_length_analysis.py.

tests/
  test_metrics.py                   # unchanged
  test_orchestration.py             # unchanged
  test_significance.py              # unchanged
  test_token_length_analysis.py     # NEW: pure aggregation-function tests
  test_verify_writeup_numbers.py    # NEW: rounding/comparison-function tests

results/
  sweep.csv                         # unchanged (session 1)
  per_query.csv                     # unchanged (significance-testing)
  significance.csv                  # unchanged (significance-testing)
  run_config.json                   # unchanged
  token_length_report.json          # NEW artifact (Requirement 11.4)

README.md                           # NEW deliverable (repo root, Requirement 1)
SPEC.md                             # NEW deliverable (repo root, Requirement 1)

data/                                # unchanged; gitignored. Already has
                                     # all-MiniLM-L6-v2's model + tokenizer
                                     # cached under data/hf_cache/ from the
                                     # completed sweep.
```

This matches `structure.md`'s layout: `docs/` gains a supporting ledger
(not one of the two documentation deliverables), `src/` gains two more
small, single-purpose entry points alongside the existing ones,
`tests/` gains their unit tests, and `results/` gains one more
artifact file. No root-level file other than `README.md`/`SPEC.md` is
created (Requirement 1.2).

### Component diagram

```mermaid
graph TD
    subgraph existing["Already committed (read-only inputs to this spec)"]
        SWEEPCSV["results/sweep.csv"]
        PQCSV["results/per_query.csv"]
        SIGCSV["results/significance.csv"]
        RUNCFG["results/run_config.json"]
        HFCACHE[("data/hf_cache/<br/>(cached model + tokenizer)")]
        SCIFACT[("data/ BEIR SciFact<br/>(cached corpus)")]
    end

    subgraph tla["Token_Length_Analysis (NEW, this spec)"]
        CFG["configs/sweep.yaml"] --> TLA["src/token_length_analysis.py"]
        SCIFACT -.->|load_scifact, configure_caches<br/>REUSED, not modified| TLA
        HFCACHE -.->|tokenizer, local_files_only| TLA
        TLA --> TLR["results/token_length_report.json<br/>(NEW artifact)"]
    end

    subgraph writeup["Hand-authored documents (this spec)"]
        SWEEPCSV --> DOCS
        PQCSV -.->|via significance.csv only,<br/>never read directly| DOCS
        SIGCSV --> DOCS
        RUNCFG --> DOCS
        TLR --> DOCS["README.md<br/>SPEC.md"]
    end

    subgraph verify["Verification_Pass (NEW, this spec)"]
        DOCS -->|every Numeric_Claim gets one row| LEDGER["docs/numeric_traceability.csv"]
        LEDGER --> VWN["src/verify_writeup_numbers.py"]
        DOCS -.->|repo_root: document-presence check,<br/>literal substring only| VWN
        SWEEPCSV --> VWN
        SIGCSV --> VWN
        RUNCFG --> VWN
        TLR --> VWN
        VWN -->|mismatch report (failure_mode),<br/>non-zero exit on any mismatch| RESULT["pass / fail"]
    end
```

`per_query.csv` is never read directly by `README.md`/`SPEC.md` or by
either new script — every requirement that needs a per-query-derived
number (the bootstrap mean difference, CI bounds, adjusted p-value)
reads it from `results/significance.csv`, which was already computed
from `per_query.csv` by the significance-testing spec. This keeps the
Numeric_Claim traceability chain to *one hop* per number (Requirement
12.1), never "recomputed from `per_query.csv` a second time by this
spec's code," which would risk a second, independent (and possibly
divergent) computation of the same statistic.

### Sequence: Token_Length_Analysis run

```mermaid
sequenceDiagram
    participant U as python -m src.token_length_analysis
    participant CFG as load_sweep_config (REUSED)
    participant CACHE as configure_caches (REUSED)
    participant LOADER as load_scifact (REUSED)
    participant TOK as HF tokenizer (local cache only)
    participant AGG as compute_exceedance_stats (pure, NEW)
    participant OUT as results/token_length_report.json

    U->>CFG: load_sweep_config(config_path)
    CFG-->>U: SweepConfig (data_dir, dense retriever's model_name)
    U->>CACHE: configure_caches(config.data_dir)
    U->>LOADER: load_scifact(config.data_dir)
    LOADER-->>U: CorpusBundle (no re-download; already cached)
    U->>TOK: load tokenizer(model_name, cache_folder, local_files_only=True)
    Note over TOK: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE also set (Requirement 11.7)
    TOK-->>U: tokenizer, or raises TokenizerLoadError (no network attempted)
    loop every document in corpus (Requirement 11.1)
        U->>U: text = format_document_text(doc)  [imported from src.retrievers.dense_retriever]
        U->>TOK: tokenizer(text, add_special_tokens=True, truncation=False)
        TOK-->>U: token_count (includes special tokens)
    end
    U->>AGG: compute_exceedance_stats(token_counts, max_sequence_length=256)
    AGG-->>U: TokenLengthStats(total, exceeding, fraction)
    U->>OUT: write_token_length_report(...) [atomic: temp file + os.replace]
```

### Sequence: Verification_Pass

```mermaid
sequenceDiagram
    participant Author as Document author (human)
    participant Ledger as docs/numeric_traceability.csv
    participant VWN as python -m src.verify_writeup_numbers
    participant Docs as README.md / SPEC.md (repo_root)
    participant Artifacts as results/*.csv, results/*.json

    Author->>Author: write a Numeric_Claim into README.md or SPEC.md
    Author->>Ledger: add one row: claim_id, document, location,<br/>stated_value, stated_precision,<br/>source_artifact, source_fields, computation
    Note over Author: repeated for every Numeric_Claim (Requirement 12.1, 12.4)
    Author->>VWN: run python -m src.verify_writeup_numbers --repo-root .
    VWN->>Ledger: read all rows (load_ledger checks each row's<br/>stated_value against stated_precision; halts on mismatch)
    loop each row
        VWN->>Docs: read row.document; check stated_value is a<br/>literal substring (document-presence check)
        alt stated_value not found in document
            VWN->>VWN: record failure_mode="value_not_in_document"
        else stated_value found
            VWN->>Artifacts: read source_artifact, select source_fields
            VWN->>VWN: apply `computation` (copy/ratio/delta/mean/percentage/sum)
            VWN->>VWN: round both stated_value and computed_value<br/>to stated_precision, ROUND_HALF_UP
            VWN->>VWN: compare; record match, or mismatch with<br/>failure_mode="artifact_mismatch"
        end
    end
    VWN-->>Author: prints mismatches (row + failure_mode + expected + got); exit 0 iff none
    Author->>Author: manual completeness check: every number that appears<br/>in README.md/SPEC.md prose has one ledger row (Requirement 12.4)
```

## Components and Interfaces

### `src/token_length_analysis.py` — Token_Length_Analysis

```python
@dataclass(frozen=True)
class TokenLengthStats:
    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float

def compute_exceedance_stats(
    token_counts: Sequence[int], max_sequence_length: int
) -> TokenLengthStats: ...

@dataclass(frozen=True)
class TokenLengthReport:
    model_name: str
    max_sequence_length: int
    num_documents_total: int
    num_documents_exceeding: int
    fraction_exceeding: float

def load_tokenizer_offline(model_name: str, cache_folder: Path) -> "PreTrainedTokenizerBase": ...
def count_tokens(tokenizer, text: str) -> int: ...
def write_token_length_report(report: TokenLengthReport, output_path: Path) -> None: ...
def main(argv: Optional[List[str]] = None) -> int: ...
```

`token_length_analysis.py` has no `format_document_text` of its own —
it imports the one that now lives in
`src/retrievers/dense_retriever.py` (see below).

**`compute_exceedance_stats`** is the sole pure, unit-under-test
function (mirroring how `src/metrics.py` and `src/significance.py`
each isolate one pure aggregation core). It takes a plain list of
integers and a threshold — no corpus, no tokenizer, no file I/O — and
returns the three numbers Requirement 11.2/11.4 requires:
`num_documents_exceeding = sum(1 for c in token_counts if c > max_sequence_length)`
("strictly greater than," per Requirement 11.2 — a count of exactly
`max_sequence_length` does **not** count as exceeding), and
`fraction_exceeding = num_documents_exceeding / num_documents_total`
(defined as `0.0` if `num_documents_total == 0`, an edge case that
should be unreachable in production since `load_scifact` already
raises `CorpusLoadError` on an empty corpus, but the pure function
handles it without a `ZeroDivisionError` rather than assuming its
caller always guards it).

**`format_document_text`** is defined once, in
`src/retrievers/dense_retriever.py`, and imported by
`token_length_analysis.py` rather than re-typed. Requirement 11.1
requires tokenizing "exactly" the same input string `DenseRetriever
.build_index` encodes, and the only way to make that true *by
construction* — rather than by two independently-maintained copies
staying in sync through discipline alone — is for both call sites to
call the same function. That extraction is in scope for this spec
specifically because it is what makes Requirement 11.1 hold by
construction, per `requirements.md`'s introduction: a single,
narrowly-scoped extract-and-import refactor, not a general invitation
to modify retriever code.

The line that moves, unchanged in content, from inline inside
`DenseRetriever.build_index` to a new module-level function in the
same file:

```python
# src/retrievers/dense_retriever.py
def format_document_text(doc: Dict[str, str]) -> str:
    return f"{doc.get('title', '')} {doc.get('text', '')}"
```

`build_index`'s list comprehension changes from building this f-string
inline to calling the named function for each `doc_id`:

```python
# before (inline f-string)
texts = [
    f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}"
    for doc_id in self._doc_ids
]

# after (calls the extracted function; same output, same behavior)
texts = [format_document_text(corpus[doc_id]) for doc_id in self._doc_ids]
```

No other line of `build_index`, `retrieve_all`, or `__init__` changes:
same encoding, same normalization, same batching, same timing, same
ranking. `token_length_analysis.py` then imports the function directly:

```python
from src.retrievers.dense_retriever import format_document_text
```

**`load_tokenizer_offline`** loads the corpus's already-cached
tokenizer with two independent layers of "never touch the network,"
mirroring the defense-in-depth style `DenseRetriever.__init__` already
uses for its own cache-path assertion:

1. Sets `os.environ["HF_HUB_OFFLINE"] = "1"` and
   `os.environ["TRANSFORMERS_OFFLINE"] = "1"` before the load call —
   this is enforced globally by `huggingface_hub`/`transformers`
   regardless of which specific loading API is used.
2. Passes `local_files_only=True` explicitly to
   `transformers.AutoTokenizer.from_pretrained(model_name,
   cache_dir=cache_folder, local_files_only=True)` — the same
   `cache_folder` (`data_dir / "hf_cache"`) that `configure_caches`
   points `HF_HOME`/`HF_HUB_CACHE` at, so the tokenizer is resolved
   from the exact directory the sweep already populated.

Any exception from either layer (a missing snapshot, a corrupted
cache, `huggingface_hub`'s own `LocalEntryNotFoundError`, or a generic
`OSError`) is caught and re-raised as this module's own
`TokenizerLoadError` — never retried without the offline flags, and
never allowed to silently fall through to a network request
(Requirement 11.7). `AutoTokenizer` (not the full `SentenceTransformer`
wrapper `DenseRetriever` uses) is loaded directly: the analysis needs
only tokenization, not the PyTorch encoder weights, so there's no
reason to force-load model weights an analysis script never uses.

**`count_tokens`** calls `tokenizer(text, add_special_tokens=True,
truncation=False)` and returns `len(result["input_ids"])`.
`add_special_tokens=True` matches what `SentenceTransformer.encode`
does internally (Requirement 11.1's "including any special tokens");
`truncation=False` is essential and deliberate: `DenseRetriever`
itself truncates at encode time, but this analysis exists specifically
to measure *how much* would be truncated, so it must count the true,
untruncated length to compare against the 256-token limit (Requirement
11.2). Truncating first would make every document's counted length
`<= max_sequence_length` by construction and the whole measurement
would be vacuous.

**`main`** orchestration:

1. Parse `--config` (default `configs/sweep.yaml`) and `--output`
   (default `results/token_length_report.json`). Load via
   `load_sweep_config` (reused, unmodified) — on `ConfigError`, print
   and return non-zero, write nothing.
2. Extract the one `DenseRetrieverConfig` entry from
   `config.retrievers` (there is exactly one, per `load_sweep_config`'s
   own validation) for `model_name`.
3. `configure_caches(config.data_dir)` (reused, unmodified) — before
   anything that imports `huggingface_hub`/`transformers`/`beir`
   transitively, same ordering discipline `sweep_runner.py` already
   follows.
4. `bundle, report = load_scifact(config.data_dir)` (reused,
   unmodified) — on `CorpusLoadError`/`CorpusValidationError`, print
   and return non-zero, write nothing. This also means a truncated or
   corrupted corpus is caught by session 1's own referential-integrity
   check before this analysis ever runs (Requirement 1.6, session 1).
5. `tokenizer = load_tokenizer_offline(model_name, config.data_dir /
   "hf_cache")` — on `TokenizerLoadError`, print and return non-zero,
   write nothing, having made no network call (Requirement 11.7).
6. For every document in `bundle.corpus` (iteration order is
   irrelevant to the result — see Property 2 below):
   `token_counts.append(count_tokens(tokenizer,
   format_document_text(doc)))` — `format_document_text` here is the
   function imported from `src.retrievers.dense_retriever` (see
   "Components and Interfaces" above), not a locally-defined one; the
   call itself is unchanged.
7. `stats = compute_exceedance_stats(token_counts, max_sequence_length=256)`.
8. `write_token_length_report(TokenLengthReport(model_name=model_name,
   max_sequence_length=256, **dataclasses.asdict(stats)), args.output)`
   — on a write failure, raises this module's `TokenLengthReportError`,
   printed and non-zero; the atomic temp-file-plus-`os.replace` pattern
   (reusing `src.report._atomic_write_text`) means `args.output` is
   left absent or in its pre-run state, never partially written.
9. Return 0.

The `256` threshold is written as a literal matching Requirement
11.2's exact wording ("the `all-MiniLM-L6-v2` model's 256-token maximum
sequence length"), not read from a config field — there is no
`max_sequence_length` field anywhere in `configs/sweep.yaml`, and
introducing one for a single hard-coded model constant would be
config-schema churn for no benefit in a single-run analysis. This is
a deliberate exception to "derive constants from data, don't hardcode
them": the *document counts and fraction* are always derived from the
loaded corpus (never hardcoded — Requirement 1.3/evaluation-integrity's
"dataset stats come from the loader's own output" rule applies to
those), but the *threshold being measured against* is itself the
subject of Requirement 11.2, fixed by the model's published
architecture, not a sweep parameter.

### Why `README.md`/`SPEC.md` are hand-authored, not generated

This design considered, and rejects, writing a templating script that
would render `README.md`/`SPEC.md` from the five artifacts
automatically. Reasons:

- `scope-guard.md` is explicit that this project uses "small, boring"
  tooling and names "no frameworks, no orchestration layers" as a
  standing constraint; a templating engine (Jinja2 is already a
  transitive dependency of `huggingface_hub`, but is not pinned or
  used anywhere in `src/` today) would be new machinery introduced
  solely to render two files, once, that will not be regenerated on a
  schedule — unlike `results/sweep.csv` or `results/significance.csv`,
  which genuinely are regenerated by reruns.
- `requirements.md`'s own Requirement 12 anticipates hand-authored
  prose: it defines a Verification_Pass as a review activity performed
  *after* both documents are drafted, not a property a generator could
  guarantee by construction. A template only guarantees traceability
  for the values it was told to substitute; it does nothing to prevent
  an author from also typing a stray, untraced number directly into
  the surrounding prose — the exact failure mode Requirement 12.4
  guards against. The Verification_Pass is required either way.
- Sections like Requirement 7's "What this does not claim" and
  Requirement 10's threats-to-validity narrative are qualitative prose
  that a template cannot generate from artifact fields; only the
  numeric portions of the two documents are template-shaped at all,
  and those are a minority of each document's content.

Instead: the two files are authored directly, with each numeric value
looked up from its specific artifact and field as it is written, and a
corresponding row added to `docs/numeric_traceability.csv` in the same
edit. A throwaway, uncommitted helper (e.g. a one-off `python -c`
snippet, or a scratch script deleted before commit) may be used at
drafting time purely to print candidate values from the artifacts and
reduce transcription error — this is explicitly *not* a deliverable of
this spec and is not described further here, because it produces
nothing that needs to be correct on its own; only the committed
`docs/numeric_traceability.csv` and its verification (below) are
load-bearing.

### `docs/numeric_traceability.csv` — the Numeric_Claim ledger

**Decision: a plain CSV, not a markdown table embedded in `SPEC.md`.**

Justification: Requirement 12.1's traceability condition is precise
and mechanical (copy, or a named arithmetic operation over one
artifact's fields) — exactly the shape a small fixed-schema CSV can
represent without ambiguity, and exactly the shape a hand-rolled parser
of free-form markdown prose cannot represent reliably. A markdown table
embedded in a hand-edited document is one loose sentence away from a
malformed row (an extra `|`, a value split across lines); a dedicated
CSV under `docs/` is edited only by adding ledger rows, never mixed
with narrative prose, and is trivially read by `pandas.read_csv` the
same way every other artifact in this repo already is.

Schema (one row per Numeric_Claim, Requirement 12.1/12.4):

| Column | Type | Meaning |
|---|---|---|
| `claim_id` | str | short stable identifier, e.g. `readme-headline-mean-diff` |
| `document` | str | `README.md` or `SPEC.md` |
| `location` | str | human-readable pointer, e.g. "first paragraph" or "Threats to validity, sparse qrels" |
| `stated_value` | str | the exact text of the number as it appears in the document (e.g. `"-0.0068"`, `"268x"`, `"5183"`) |
| `stated_precision` | str | how many decimal places / significant digits / "integer" the stated text uses — the rounding target for comparison (Requirement 12.2) |
| `source_artifact` | str | one of `sweep.csv`, `per_query.csv` (never used directly by this spec, kept only as a valid enum value), `significance.csv`, `run_config.json`, `token_length_report.json` |
| `source_fields` | str | one or more `field` or `row_selector.field` references *within that one artifact* — e.g. `ndcg_at_10.mean_diff` (a `significance.csv` row selected by its `metric` column) or `corpus_load_report.num_qrel_pairs;corpus_load_report.num_queries` for a two-field ratio |
| `computation` | str | one of a fixed enum: `copy`, `ratio`, `delta`, `mean`, `percentage`, `sum`, `half_ci_width` |

`stated_precision` is authoritative; `stated_value`'s text must be
formatted consistent with it (e.g. a `stated_precision` of `4dp`
requires `stated_value` to show exactly 4 digits after the decimal
point; a `stated_precision` of `integer` requires `stated_value` to
contain no decimal point; a `stated_precision` of `percentage:1dp`
requires a `%`-suffixed value with exactly 1 decimal place before the
`%`), and this consistency is itself checked mechanically, not merely
documented — see `stated_value_matches_precision` below. The two
columns are kept separate rather than collapsed into one (Requirement
12.2's "the number of decimal places ... shown for that Numeric_Claim"
still needs a place to be stated independently of the value's own
text so `round_half_up`'s comparison has an explicit target), but
`stated_value` may never silently disagree with the precision it
claims to be written at.

`source_fields` never names more than one artifact per row, by
construction of the column's meaning (Requirement 12.1's "no value
from any other artifact contributing to the computation"). A
computation spanning *rows* of the same artifact (e.g. Requirement
6.3's sum of `index_time` + `query_latency` across both `sweep.csv`
runs) is still a single-artifact computation and is expressed with
multiple `row_selector.field` references joined by `;` in
`source_fields`.

A row whose `source_fields` value is the literal sentinel string
`"n/a"` or `"NA"` (Requirement 4.5's non-primary-metric table cells)
is compared as a **sentinel-to-sentinel string match**, not rounded or
treated as a number — `verify_writeup_numbers.py` special-cases this
before attempting any arithmetic, since `"n/a"` and `"NA"` are exactly
the two distinct non-numeric sentinels `results/significance.csv`
itself defines (see the significance-testing `design.md`'s "two
distinct sentinels" section) and neither is a Numeric_Claim under the
glossary's definition, even though a ledger row is still the simplest
way to assert "this cell must read exactly this sentinel."

### `src/verify_writeup_numbers.py` — the Verification_Pass

```python
_ALLOWED_COMPUTATIONS = ("copy", "ratio", "delta", "mean", "percentage", "sum", "half_ci_width")

@dataclass(frozen=True)
class TraceabilityRow:
    claim_id: str
    document: str
    location: str
    stated_value: str
    stated_precision: str
    source_artifact: str
    source_fields: str
    computation: str

@dataclass(frozen=True)
class VerificationResult:
    claim_id: str
    matched: bool
    failure_mode: Optional[str]  # None | "value_not_in_document" | "artifact_mismatch"
    stated_rounded: str
    computed_rounded: str
    detail: str

def load_ledger(path: Path) -> List[TraceabilityRow]: ...
def load_artifact_values(source_artifact: str, source_fields: str, artifacts_dir: Path) -> List[float | str]: ...
def apply_computation(computation: str, values: List[float | str]) -> float | str: ...
def round_half_up(value: float, precision_spec: str) -> str: ...
def stated_value_matches_precision(stated_value: str, stated_precision: str) -> bool: ...
def verify_row(row: TraceabilityRow, artifacts_dir: Path, repo_root: Path) -> VerificationResult: ...
def main(argv: Optional[List[str]] = None) -> int: ...
```

**`round_half_up`** exists because Python's built-in `round()` performs
*round-half-to-even* ("banker's rounding"), not round-half-up —
`round(0.125, 2)` gives `0.12` in Python, not the `0.13` Requirement
12.2's "round-half-up rounding" specifies. This function instead uses
`decimal.Decimal(str(value)).quantize(decimal.Decimal(target),
rounding=decimal.ROUND_HALF_UP)`, converting through `str(value)`
first (never `Decimal(float)` directly) so the decimal digits rounded
are the same digits a human reading the float's usual string
representation would see, avoiding a spurious mismatch caused by binary
floating-point representation noise several digits past what either
document actually states.

**`stated_value_matches_precision`** is a small fixed set of format
checks, one per recognized `stated_precision` shape — a format check
against `stated_value`'s literal text, not a numeric derivation from
it: `integer` requires no `.` in `stated_value`; `Ndp` (e.g. `4dp`)
requires exactly `N` digits after a single `.`; `percentage:Ndp` (e.g.
`percentage:1dp`) requires a trailing `%` with exactly `N` decimal
digits before it; a ratio's `Nx` suffix (e.g. `268x`) is treated the
same as `integer`/`Ndp` on the digits preceding the `x`. Any
`stated_precision` value outside this fixed set of shapes is itself a
malformed row, handled the same way as an unrecognized shape below.
This function never parses `stated_value` into a `float` and never
infers `stated_precision` from it in the other direction — the
direction is fixed: `stated_precision` is declared, and `stated_value`
is checked against it, matching the ledger schema's "`stated_precision`
is authoritative" rule above.

**`apply_computation`** implements exactly the fixed enum
`_ALLOWED_COMPUTATIONS` — the same named operations `requirements.md`
already specifies in prose (`ratio`: Requirement 3.2, 10.2; `delta`:
evaluation-integrity's "dense results always reported as a delta";
`mean`: Requirement 6.3's sum-of-four-values read as "combined... time"
is a `sum`, not a `mean` — the two are kept as separate enum members
rather than one generic "arithmetic" escape hatch specifically so a
ledger row cannot smuggle in an unreviewed computation; `half_ci_width`:
Requirement 10.5's `(ci_upper - ci_lower) / 2`). Adding a computation
this enum doesn't already support requires editing this function, not
just adding a new value to a ledger row — a deliberate friction point,
matching Requirement 12.1's closing clause, "using an arithmetic
operation already specified elsewhere in this spec."

**`load_ledger`** parses `docs/numeric_traceability.csv` into
`TraceabilityRow` instances and, immediately after parsing each row —
before any artifact I/O, before `verify_row` is ever called — calls
`stated_value_matches_precision(row.stated_value, row.stated_precision)`
on that row. A row that fails this check is a malformed ledger row (its
own two columns disagree about how the value is formatted), not a
verification mismatch against an artifact, so `load_ledger` raises
`TraceabilityFileError` naming the offending `claim_id`, matching this
design's existing "failures discovered before any output is produced
halt outright" philosophy: the run halts before verifying *any* row,
rather than reporting the malformed row as a `MISMATCH` alongside
correctly-formed rows that did get checked.

**`load_artifact_values`** dispatches on `source_artifact`: for
`sweep.csv`/`significance.csv`/`per_query.csv`, parses `row_selector`
prefixes in `source_fields` (e.g. `run_id=bm25__whole_document,k=1` or
`metric=ndcg_at_10`) as an exact-match row filter, then reads the named
column from the one matching row; for `run_config.json`, walks the
dotted path (e.g. `corpus_load_report.num_documents`,
`sweep_config.retrievers[0].k1`) through the parsed JSON object; for
`token_length_report.json`, reads the named top-level key directly.
Raises this module's own `VerificationSourceError` if the artifact file
is absent, the row selector matches zero or more than one row, or the
named field/key is absent — a row that cannot be resolved is a hard
failure, not a skipped row, since Requirement 12.4 does not allow any
Numeric_Claim to go unverified.

**`verify_row`** performs two independent checks, in order, and the
result distinguishes which one (if either) failed:

1. **Document-presence check.** Given `row.document` (`README.md` or
   `SPEC.md`), reads that file from `repo_root / row.document` and
   confirms that `row.stated_value` occurs in the file's full text as
   a literal substring — no regex, no numeric parsing, no prose
   extraction, just `row.stated_value in document_text`. This is
   presence, not extraction: it answers "does the document still say
   this," not "what does the document say here." If the substring is
   not found, `verify_row` returns immediately with
   `matched=False, failure_mode="value_not_in_document"`, without
   attempting the artifact comparison below — a document that no
   longer contains the value it's ledgered against is a failure in its
   own right, regardless of whether the cited artifact would otherwise
   agree.
2. **Ledger-to-artifact comparison.** Only reached once the
   document-presence check passes. Resolves the cited artifact
   value(s) via `load_artifact_values`, applies the computation via
   `apply_computation`, rounds both the stated and computed values with
   `round_half_up` at the row's stated precision, and compares the two
   rounded strings for exact equality. A disagreement here returns
   `matched=False, failure_mode="artifact_mismatch"`; agreement returns
   `matched=True, failure_mode=None`.

The two failure modes are kept distinguishable in `VerificationResult`
because they call for different fixes: `"value_not_in_document"` means
the document (or the ledger's `stated_value`) has drifted since the
row was written; `"artifact_mismatch"` means the value is present in
the document as claimed but disagrees with what the cited artifact
actually holds. Collapsing both into one generic `MISMATCH` would
require a maintainer to re-derive which fix applies by hand every time;
`failure_mode` states it directly.

**`main`** orchestration:

1. Parse `--repo-root` (optional CLI arg, default the current working
   directory) in addition to the ledger/artifacts paths, and pass it
   through to every `verify_row` call as `repo_root: Path`.
2. Read `docs/numeric_traceability.csv` via `load_ledger`. On a missing
   file, a parse failure lacking a required column, or a row whose
   `stated_value` fails `stated_value_matches_precision` against its
   own `stated_precision`, print and return non-zero — no partial
   verification is reported as a pass.
3. For each row (in file order), call `verify_row(row, artifacts_dir,
   repo_root)`, which performs the document-presence check and, if
   that passes, the ledger-to-artifact comparison described above.
4. Print one line per row: `claim_id`: `MATCH`, or `MISMATCH` together
   with which `failure_mode` occurred (`value_not_in_document` or
   `artifact_mismatch`) and both rounded values shown where applicable
   — and a summary count at the end.
5. Return `0` only if every row matched; otherwise return `1`, and the
   run is not considered a completed Verification_Pass (Requirement
   12.2, 12.3) — the feature is not done while any row still fails
   (Requirement 12.4).

This script is invoked manually, as-needed while drafting and once
more before this spec is considered complete — it is **not** wired
into any CI workflow, since a GitHub Actions workflow is explicitly out
of scope for this spec (`requirements.md`'s introduction) and belongs
to a different, later piece of work per `scope-guard.md`.

**What the Verification_Pass does *not* automate.** Requirement 12.4
requires that no Numeric_Claim *lacks* a ledger row — i.e., that the
ledger is complete, not just that every row it contains is correct.
Detecting "a number appears in the prose of `README.md`/`SPEC.md` with
no corresponding ledger row" would require parsing arbitrary numbers
out of free-form markdown text and then judging which of them are
Numeric_Claims under the glossary's definition (a percentage, a count,
a configured value, and a plain descriptive number like "two
retrievers" are not all the same kind of thing, and the glossary's
definition is itself a judgment call in places). A regex-based prose
scanner would produce both false positives (flagging a number that
isn't a Numeric_Claim, e.g. a markdown heading level or a requirement
number cited for context) and false negatives (missing a number spelled
out as a word, or embedded inside a table cell the regex didn't
anticipate) often enough that maintaining the scanner's exception list
would itself become the fragile, high-maintenance artifact the
"no number without a receipt" rule is trying to avoid. Ledger
*completeness* is therefore a **documented manual step**: the author
re-reads both finished documents line by line and confirms every number
has a matching ledger row, as the last step of the Verification_Pass
sequence diagram above — a one-time human review over two short,
finished documents, not an ongoing automated gate.

Ledger *staleness*, by contrast — a ledger row whose `stated_value` no
longer matches what the document actually says, because someone edited
the document's prose after the row was written and forgot to update
the ledger to match — is **not** a manual-review concern: `verify_row`'s
document-presence check (above) catches this automatically on every
run, by confirming `stated_value` still appears verbatim in the cited
document before ever comparing against the artifact. Completeness (does
every number have a row) remains manual; correctness of each existing
row, including whether it still reflects the document's current text,
is fully automated.

## Data Models

### `results/token_length_report.json` schema (Requirement 11.4)

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "max_sequence_length": 256,
  "num_documents_total": 5183,
  "num_documents_exceeding": 0,
  "fraction_exceeding": 0.0
}
```

(Values above are illustrative placeholders only — every real value in
the committed file is derived from the actual corpus and tokenizer at
run time, never typed from memory, per evaluation-integrity's "dataset
stats come from the loader's own output" rule extended to this new
artifact.)

| Field | Type | Meaning |
|---|---|---|
| `model_name` | str | the dense retriever's `model_name`, read from `configs/sweep.yaml` via `load_sweep_config` — never hardcoded independently of the config |
| `max_sequence_length` | int | the fixed threshold this analysis measures against; `256`, per Requirement 11.2 |
| `num_documents_total` | int | `len(bundle.corpus)` — every document `load_scifact` loaded, matching `run_config.json`'s `corpus_load_report.num_documents` for the same run |
| `num_documents_exceeding` | int | count of documents whose untruncated token count is `> max_sequence_length` |
| `fraction_exceeding` | float | `num_documents_exceeding / num_documents_total` |

Never a missing-value sentinel: the analysis either fully succeeds
(corpus loads, tokenizer loads offline, every document is counted) and
writes all five fields, or it halts before writing anything at all —
there is no partial-row or partial-field recovery path, unlike
`results/sweep.csv`'s per-cell `"NA"` marker. A single artifact
describing one whole-corpus measurement has no smaller unit to
partially degrade to.

### `docs/numeric_traceability.csv` schema

Documented above in "Components and Interfaces" (the schema is the
component's interface, so it is defined once, there, rather than
duplicated here).

## Correctness Properties

*A property is a characteristic or behavior that should hold true
across all valid executions of a system — essentially, a formal
statement about what the system should do. Properties serve as the
bridge between human-readable specifications and machine-verifiable
correctness guarantees.*

**Property-based testing (generated random inputs via a library such
as Hypothesis) is deliberately not used in this spec**, matching the
precedent both `session-1-baseline-sweep/design.md` and
`significance-testing/design.md` already set in this repository.
Applying the Requirement-11-era decision guide directly:
`compute_exceedance_stats` is a pure function, but its input space (a
list of non-negative token counts and one threshold) has essentially
three interesting boundaries — the empty case, the exact-threshold
case, and the strictly-greater case — and a handful of hand-picked
fixtures covering those exercises the same logic 100 generated
iterations would, without adding a new dependency
(`hypothesis`/similar) for a two-branch arithmetic function. The same
reasoning applies to `round_half_up` and `apply_computation`: each has
a small number of closed-form boundary cases (the exact `.5` tie for
rounding; one value vs. multiple values for the arithmetic operations)
that hand-built fixtures state directly and exactly, which is this
repo's established correctness bar for this class of function (see
both prior `design.md` files' identical "Property-based testing is
deliberately not used" sections). The properties below are still
stated as universal "for all" claims, in the same house style as the
two prior specs, and are verified either by the fixture-based unit
tests in Testing Strategy or, where noted, structurally.

### Property 1: Exceedance count and fraction are internally consistent

For any list of token counts and any non-negative threshold, the
computed `num_documents_exceeding` equals the count of elements
strictly greater than the threshold, `0 <= num_documents_exceeding <=
num_documents_total`, and `fraction_exceeding` equals
`num_documents_exceeding / num_documents_total` exactly (or `0.0` when
`num_documents_total == 0`).

**Validates: Requirements 11.2, 11.4**

Upheld by: `compute_exceedance_stats`'s single-pass `sum(1 for c in
token_counts if c > max_sequence_length)` and division. Verified by
`tests/test_token_length_analysis.py`'s hand-picked fixtures: empty
input, all-under-threshold, all-over-threshold, and a mixed case with
an independently hand-computed expected fraction, plus the boundary
pair (a count of exactly `256` is not exceeding; a count of `257` is)
that directly exercises the "strictly greater than" wording of
Requirement 11.2.

### Property 2: The aggregate does not depend on document order

For any list of per-document token counts, the values of
`num_documents_exceeding`, `num_documents_total`, and
`fraction_exceeding` are unchanged by any permutation of that list —
the corpus dictionary's iteration order (which `dict` in Python 3.7+
preserves as insertion order, but which is otherwise incidental to
BEIR's own file-loading order) never affects the reported statistics.

**Validates: Requirements 11.2, 11.4**

Upheld by: `compute_exceedance_stats` reducing over the list with `sum`
and `len`, both order-independent reductions with no dependency on
position. Verified by `tests/test_token_length_analysis.py` asserting
identical output for a fixture list and a shuffled copy of the same
list.

### Property 3: The tokenized input matches the dense retriever's own input, verbatim

For any corpus document, the string `token_length_analysis.py` tokenizes
is character-for-character identical to the string `DenseRetriever
.build_index` encodes for that same document (`title + " " + text`,
using an empty string for either field when absent).

**Validates: Requirements 11.1**

This property is no longer a testable-behavior claim; it is a
structural, by-construction guarantee. `token_length_analysis.py` does
not maintain its own copy of the title+text join — it imports and
calls the exact same `format_document_text` function that
`DenseRetriever.build_index` calls (see "Components and Interfaces"
above). There are not two independently-maintained implementations
that could drift apart; there is exactly one implementation, called
from two places. Identity holds trivially and permanently by
construction, the same way `a is a` holds regardless of what `a` is.

This property therefore requires no dedicated regression test for
drift — there are no two things left to drift apart, so no test can
meaningfully exercise "did they stay in sync." A basic unit test of
`format_document_text` itself (for example, against a document with a
missing `title` key) still has value, but it belongs wherever
`DenseRetriever`'s own test coverage lives, not in
`tests/test_token_length_analysis.py`. No test file currently covers
`src/retrievers/dense_retriever.py` (there is no `tests/
test_dense_retriever.py` today), and adding one is not something this
spec's requirements call for — see "What is explicitly not tested in
this spec" below.

### Property 4: No invocation of the Token_Length_Analysis makes a network call

For any invocation of `python -m src.token_length_analysis` — whether
the tokenizer cache under `data/` is present or missing — no network
request is made; a missing or incomplete cache produces a
`TokenizerLoadError` instead of a download attempt.

**Validates: Requirements 11.3, 11.7**

Upheld by: `load_tokenizer_offline`'s two independent offline layers
(the `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` environment variables, set
before any load call, and the explicit `local_files_only=True` argument
to `AutoTokenizer.from_pretrained`) — either layer alone is sufficient
to prevent a network call in `huggingface_hub`/`transformers`, and the
two together are defense in depth, mirroring `DenseRetriever.__init__`'s
own two-layer cache-path assertion. Verified by
`tests/test_token_length_analysis.py`'s "missing cache" test, which
points `cache_folder` at an empty temporary directory for a real model
name and asserts `TokenizerLoadError` is raised — safe to run under
"no network access inside tests" specifically *because*
`local_files_only=True` makes `transformers` raise immediately on a
cache miss rather than falling back to a request.

### Property 5: Round-half-up rounding matches Requirement 12.2 exactly at the tie boundary

For any decimal value whose digit immediately past the stated
precision is exactly `5` (a rounding tie), `round_half_up` rounds away
from zero at that digit, even where Python's built-in `round()` would
round to the nearest even digit instead.

**Validates: Requirements 12.2**

Upheld by: `round_half_up`'s use of `decimal.Decimal(str(value))
.quantize(..., rounding=decimal.ROUND_HALF_UP)`, converting through the
value's string representation rather than constructing a `Decimal`
from the raw `float` (which would round on already-imprecise binary
digits). Verified by `tests/test_verify_writeup_numbers.py` asserting
`round_half_up(0.125, "2dp") == "0.13"` alongside a direct comparison
statement that Python's own `round(0.125, 2) == 0.12`, so the test
documents *why* a bespoke function is needed rather than merely
asserting one hand-picked expected value.

### Property 6: A verification row reports a match if and only if the rounded values are equal

For any traceability row, `verify_row` reports `matched=True` exactly
when the stated value and the value it recomputes from the cited
artifact are equal after both are independently rounded to the row's
stated precision with `round_half_up` — never on unrounded equality,
and never a false match from two values that merely round similarly
under ordinary (non-half-up) rounding.

**Validates: Requirements 12.2, 12.3**

Upheld by: `verify_row` calling `round_half_up` on both the ledger's
`stated_value` and the freshly computed value before comparing the two
resulting strings, rather than comparing raw floats with a tolerance
(a tolerance-based comparison would silently accept a stated value that
is correct only to a coarser precision than the document claims,
undermining Requirement 12.2's "the number of decimal places ... shown
for that Numeric_Claim in the document"). Verified by
`tests/test_verify_writeup_numbers.py`'s matched/mismatched fixture
pairs, including one pair that would incorrectly report a match under
naive `abs(a - b) < epsilon` comparison but correctly reports a
mismatch once both are rounded to the stated (coarser) precision and
compared as strings.

### Property 7: A stale document is always caught as a document-presence failure, never silently reported as a match

For any traceability row whose `stated_value` no longer appears
verbatim in the text of its cited `document` (`README.md` or
`SPEC.md`) — whether because the document's prose was edited after the
row was written, or because the ledger row itself drifted — `verify_row`
always reports `matched=False, failure_mode="value_not_in_document"`,
and never reports `matched=True` on the strength of the cited
artifact's value agreeing, regardless of what that artifact contains.

**Validates: Requirements 12.2**

This closes the "checks... in both files" gap directly: Requirement
12.2 requires the Verification_Pass to check every Numeric_Claim "in
both files" — meaning both the document's own text and the cited
artifact — and a `verify_row` that only ever compared the ledger's
`stated_value` against the recomputed artifact value, without reading
the document at all, could report `MATCH` for a row whose document
text had since changed to say something else entirely. That gap is
exactly what the document-presence check closes.

Upheld by: `verify_row`'s document-presence check running first and
unconditionally — reading `repo_root / row.document` and testing
`row.stated_value in document_text` as a literal substring — and
returning `failure_mode="value_not_in_document"` immediately on a
miss, before the ledger-to-artifact comparison ever runs, so a stale
document can never be masked by an artifact that still happens to
agree with the ledger. Verified by a new
`tests/test_verify_writeup_numbers.py` test case, e.g.
`test_verify_row_document_presence_failure_even_when_artifact_matches`:
a fixture where the cited artifact's value matches `row.stated_value`
exactly (so the ledger-to-artifact comparison alone would report
`MATCH`), but the fixture standing in for the document's text has been
doctored to no longer contain `row.stated_value` anywhere — asserting
the result is `matched=False, failure_mode="value_not_in_document"`,
not `matched=True`.

## Error Handling

| Failure | Detected by | Exception | Behavior | Requirement |
|---|---|---|---|---|
| `configs/sweep.yaml` missing/unparsable/unsupported declaration | `load_sweep_config` (reused) | `ConfigError` | Halt before any corpus/tokenizer access. No `token_length_report.json` written. | reuses session-1 contract |
| Corpus/queries/qrels fail to load, load empty, or fail referential-integrity validation | `load_scifact` (reused) | `CorpusLoadError` / `CorpusValidationError` | Halt before tokenizer load. No `token_length_report.json` written. | reuses session-1 contract |
| Tokenizer cannot be loaded from the local cache without a network call | `load_tokenizer_offline` | `TokenizerLoadError` (new, this module) | Halt before tokenizing any document. No `token_length_report.json` written. No network call attempted. | 11.7 |
| `results/token_length_report.json` write fails (disk full, permissions) | `write_token_length_report` (atomic write via reused `_atomic_write_text`) | `TokenLengthReportError` (new, this module) | Halt. Temp file removed. No partial/corrupted file left at the output path. | 11.4 |
| `docs/numeric_traceability.csv` missing, unparsable, or missing a required column | `load_ledger` | `TraceabilityFileError` (new, this module) | Halt before verifying any row. Non-zero exit. No row is reported as verified. | 12.1, 12.4 |
| A ledger row's `stated_value` text is not formatted consistent with its own declared `stated_precision` | `load_ledger`, via `stated_value_matches_precision` | `TraceabilityFileError` (new, this module) | Halt before verifying any row. Non-zero exit. No row is reported as verified. | 12.2 |
| A ledger row's cited artifact file is absent, its row selector matches zero or more than one row, or the named field/key is absent | `load_artifact_values` | `VerificationSourceError` (new, this module) | That row is reported as a hard failure (not skipped). Non-zero exit for the whole run. | 12.1, 12.3 |
| A ledger row's `computation` value is not one of the fixed enum members | `apply_computation` | `VerificationSourceError` | Same as above — treated as a resolution failure, not a mismatch, since the row itself is malformed rather than merely wrong. | 12.1 |
| A ledger row's `stated_value` does not appear verbatim in its cited document (`README.md` or `SPEC.md`) | `verify_row`'s document-presence check | (no new exception type — a recorded `VerificationResult` with `matched=False, failure_mode="value_not_in_document"`) | Printed as a `MISMATCH` with that `failure_mode`; `main` returns non-zero once all rows are checked. | 12.2 |
| A ledger row's stated value and recomputed value disagree once both are rounded to the stated precision (document-presence check already passed) | `verify_row`'s ledger-to-artifact comparison | (no exception — a recorded `VerificationResult` with `matched=False, failure_mode="artifact_mismatch"`) | Printed as a `MISMATCH` with that `failure_mode`; `main` returns non-zero once all rows are checked. Per Requirement 12.3, the mismatch must be corrected (in the document text, or in the ledger's `source_fields`/`computation`/`stated_precision` — never by editing the cited artifact itself) before this feature is considered complete. | 12.2, 12.3 |

The dividing line matches the two prior specs in this repository:
**failures discovered before any output is produced** (bad
`Sweep_Config`, bad/empty/unresolvable corpus, an unloadable tokenizer,
a malformed ledger — including a row whose `stated_value` doesn't
match the format its own `stated_precision` declares) halt outright
with nothing written or reported as passing. The **two conditions that
are a reported result rather than an exception** — a row's document-
presence failure (`failure_mode="value_not_in_document"`) and a row's
ledger-to-artifact mismatch (`failure_mode="artifact_mismatch"`) — each
still fail the whole Verification_Pass run (non-zero exit), because
Requirement 12.4 explicitly does not allow the feature to be considered
complete while any known mismatch of either kind remains; unlike
`results/sweep.csv`'s Requirement 7's "row-count guarantee always
wins," there is no artifact-completeness requirement here that would
justify recording a mismatched row and moving on regardless.

## Testing Strategy

**Property-based testing is deliberately not used in this spec** — see
the explanation at the top of "Correctness Properties" above. The
required assertions for both new modules are closed-form, hand-picked
boundary cases (an empty corpus; a token count exactly at 256 vs. 257;
a rounding tie at exactly `.5`; a case where naive tolerance-based
float comparison would wrongly report a match), which fixture-based
unit tests state and check directly, matching the correctness bar this
repository has already established in `test_metrics.py` and
`test_significance.py`.

### Scope

This spec adds two test modules: `tests/test_token_length_analysis.py`
and `tests/test_verify_writeup_numbers.py`. Together they are the
entire automated test surface this spec introduces; `test_metrics.py`,
`test_orchestration.py`, and `test_significance.py` are unchanged. Both
new modules resolve `src.*` imports the same way the existing three do
— via `pyproject.toml`'s already-configured `pythonpath = ["."]` /
`testpaths = ["tests"]` — so no new pytest configuration is needed.

### `tests/test_token_length_analysis.py`

- Imports `src.token_length_analysis` (`compute_exceedance_stats`,
  `load_tokenizer_offline`, `TokenizerLoadError`). Does not import
  `src.sweep_runner` or `src.significance`. It does, transitively,
  import `src.retrievers.dense_retriever` for `format_document_text` —
  `token_length_analysis.py` imports that name from there rather than
  defining it locally (see "Components and Interfaces" above), so any
  test that exercises `token_length_analysis.py`'s own module-level
  names pulls that import in along with it. This is a small, deliberate
  exception to keeping this test module's dependencies limited to
  `src.token_length_analysis`: it is the direct, intended consequence
  of the extract-and-import refactor, not a sign the test is reaching
  into unrelated retriever internals. `format_document_text` itself is
  not re-tested here (see Property 3's note above) — this module tests
  `token_length_analysis.py`'s own logic, not the imported function's.
- Makes no network call anywhere, including in the "missing cache"
  test (Property 4 above) — `local_files_only=True` guarantees a local
  `OSError`/`LocalEntryNotFoundError` rather than a request when the
  cache directory is empty. Importing `format_document_text` is a plain
  Python import of a pure function and makes no network call either.
- Does not load the real cached model or tokenize the real ~5k-document
  corpus — that would require the actual cached artifacts under
  `data/` and is deferred to a one-time manual run of
  `python -m src.token_length_analysis` while producing
  `results/token_length_report.json` for real, the same way session 1
  and significance-testing both defer their own real-corpus,
  real-model end-to-end runs to a manual step rather than an automated
  test (see both prior specs' "What is explicitly not tested" lists).
- Covers, at minimum: `compute_exceedance_stats` on an empty list, an
  all-under-threshold list, an all-over-threshold list, a mixed list
  with a hand-computed expected fraction, and the `256`-vs-`257`
  boundary pair (Property 1); the same mixed list shuffled (Property
  2); and `load_tokenizer_offline` raising `TokenizerLoadError` against
  an empty temporary cache directory (Property 4). Property 3 (the
  tokenized input matches the dense retriever's own input, verbatim)
  has no test here — it is upheld structurally, by there being a single
  shared `format_document_text` rather than two copies to compare (see
  "Correctness Properties" above).

### `tests/test_verify_writeup_numbers.py`

- Imports only `src.verify_writeup_numbers` (`round_half_up`,
  `apply_computation`, `verify_row`, `load_ledger`,
  `stated_value_matches_precision`, `TraceabilityRow`,
  `VerificationSourceError`, `TraceabilityFileError`). Does not import
  `src.token_length_analysis` or any retriever/corpus module.
- Makes no network call and reads no file under `results/` or `docs/`
  — every "artifact" a test needs is a small Python literal dict
  standing in for a parsed CSV row or JSON object, and every "document"
  a test needs is a small literal string standing in for the contents
  of `README.md`/`SPEC.md` (e.g. written to a temporary file or passed
  via a fixture `repo_root`), matching this repository's "tests use
  small fixtures... not live... data" rule.
- Covers, at minimum: `round_half_up`'s tie-breaking behavior against
  Python's own `round()` on the same input (Property 5); each member of
  `_ALLOWED_COMPUTATIONS` (`copy`, `ratio`, `delta`, `mean`,
  `percentage`, `sum`, `half_ci_width`) against a small literal input
  and an independently hand-computed expected output; an unrecognized
  `computation` value raising `VerificationSourceError`; and
  `verify_row` on both a matching pair and a pair that would falsely
  match under naive tolerance-based float comparison but correctly
  mismatches once both sides are rounded to the stated (coarser)
  precision (Property 6).
- `stated_value_matches_precision` against at least one matching case
  per recognized `stated_precision` shape (`integer`, `Ndp`,
  `percentage:Ndp`, and the ratio `Nx` suffix) and at least one
  deliberately inconsistent case — e.g. `stated_value="-0.007"` against
  `stated_precision="4dp"`, which must fail since `-0.007` shows 3
  decimal digits, not 4 — plus `load_ledger` raising
  `TraceabilityFileError` naming the offending `claim_id` when a fixture
  ledger row fails this check.
- `verify_row`'s document-presence check succeeding when
  `row.stated_value` is present in a fixture document string, and
  failing with `failure_mode="value_not_in_document"` when a fixture
  document string has been doctored to no longer contain
  `row.stated_value`, even though the cited artifact fixture's value
  would otherwise match (Property 7) — e.g.
  `test_verify_row_document_presence_failure_even_when_artifact_matches`.

### What is explicitly not tested in this spec

- `src/token_length_analysis.py`'s `main()` end-to-end against the real
  cached BEIR SciFact corpus and the real cached tokenizer — no
  automated test; verified once by manually running
  `python -m src.token_length_analysis` and inspecting
  `results/token_length_report.json` before citing its numbers in
  `SPEC.md`.
- `src/verify_writeup_numbers.py`'s `main()` end-to-end against the
  real, committed `docs/numeric_traceability.csv` and the real
  `results/*` artifacts — no automated test; this *is* the
  Verification_Pass itself (Requirement 12.2), run manually and
  required to pass at least once before this feature is considered
  complete, not gated behind pytest or CI.
- The prose content of `README.md`/`SPEC.md` — no automated test
  asserts wording, section presence, or narrative framing; those
  requirements (e.g. Requirement 7's "What this does not claim"
  section, Requirement 10's threats-to-validity narrative) are
  satisfied by the authored documents themselves and are not
  mechanically checkable, consistent with this feature being a
  documentation deliverable rather than a piece of regenerated,
  tested software.
- The manual "every prose number has a ledger row" completeness check
  (Requirement 12.4) — deliberately not automated; see "What the
  Verification_Pass does not automate" in Components and Interfaces.
- `format_document_text`'s own correctness (for example, its behavior
  against a document missing a `title` field) — no test is added for
  it by this spec, in this spec's test modules or elsewhere. No test
  file currently covers `src/retrievers/dense_retriever.py` (`tests/`
  has no `test_dense_retriever.py`), and adding one is out of scope
  here: this spec's requirements call only for `token_length_analysis
  .py` to reuse the function correctly (Property 3, upheld
  structurally by the import), not for establishing test coverage of
  the retriever module itself. That gap, if it is to be closed,
  belongs to whichever spec first adds retriever test coverage.
- `ANALYSIS.md`, CI, the third retriever, additional chunking
  strategies, and failure bucketing — out of scope for this entire
  spec per `requirements.md`'s introduction and `scope-guard.md`; no
  test surface is introduced for any of them here.
