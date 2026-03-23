# imperial-capstone-project

Capstone project as part of the Imperial College AI/ML certification.

## BBO Capstone Project

**Public repository:** [github.com/houman44/imperial-capstone-project](https://github.com/houman44/imperial-capstone-project)

Black-box optimisation capstone: iterative query submission and next-point recommendation for eight benchmark functions (Imperial College AI/ML certification).

## For a general audience (about 100 words)

How do you improve when you cannot see the rules? Each capstone “function” is a hidden scorer: you pick numeric settings in a fixed range, submit them, and get one score back—never the underlying formula. This project provides a workflow that learns from past scores, proposes sensible next settings, and logs every round for reproducibility. It balances refining the best results so far with probing unfamiliar regions—like disciplined experimentation. Documentation alongside the code states assumptions, limitations, and how to interpret results responsibly. It was built for the Imperial College AI/ML black-box optimisation capstone.

## Repository map

| Location | Purpose |
| -------- | ------- |
| `src/capstone_manager.py` | Main CLI: init, import results, `sync-history`, recommend next queries |
| `data/functions/` | Per-function observation history (`function_1.csv` … `function_8.csv`) |
| `data/history/` | Versioned round snapshots for reproducible rebuilds (commit these for auditability) |
| `data/submissions/` | Generated next-query CSVs per round |
| `notebooks/BBO_Capstone_Method_and_Results.ipynb` | **Method & results** narrative (portfolio) |
| `docs/DATASHEET.md` | Dataset context, collection, limitations |
| `docs/MODEL_CARD.md` | Optimisation approach, intended use, limitations |
| `docs/BBO_Capstone_Presentation.pdf` | Short presentation (regenerate via `scripts/generate_bbo_presentation_pdf.py`) |

## Data hosted externally

**Official evaluations (scores `y`)** are produced by the **course / Kaggle-style capstone evaluator**, not bundled in this repository. This repo stores **local copies** of inputs and returned scores that you export and import via CSV (`import-results`, `data/history/`). Do not commit secrets or huge raw dumps; keep canonical round exports small and documented (see Datasheet).

## Strategy summary (current)

- **Goal:** maximise or minimise unknown scalar objectives \(f_i(\mathbf{x})\) on \([0,1]^{d_i}\) with a limited query budget, using only noisy evaluations returned by an external simulator (Kaggle).
- **Core method:** Gaussian Process (Matérn kernel) surrogate + **Expected Improvement** (ξ = 0.01), with random candidate search (`n_candidates` = 5000) per function (`src/capstone_manager.py`).
- **Evolution:** Early rounds combined manual reasoning (e.g. boundary tests where optima often lie) with ad hoc scripts; later rounds use one reproducible pipeline: import results → fit GP → propose next \(\mathbf{x}\) → submit.
- **Exploration vs exploitation:** EI balances improving on the current best (exploitation) and visiting uncertain regions (exploration); high-dimensional functions receive more queries in practice because the surrogate needs more evidence.

See **[Model Card](docs/MODEL_CARD.md)** for limitations, per-round notes, and performance snapshot.

## Approach

Recommendations are generated using Bayesian optimization with a Gaussian Process (Matérn kernel) surrogate and Expected Improvement (EI) acquisition. See the [Model Card](docs/MODEL_CARD.md) for full details.

## Documentation

- **[Datasheet](docs/DATASHEET.md)** – Motivation, composition, collection process, and intended uses of the dataset.
- **[Model Card](docs/MODEL_CARD.md)** – Optimisation approach, intended use, performance, limitations, and ethical considerations.

---

## Workflow

1. Generate query points.
2. Submit/evaluate externally (Kaggle).
3. Import returned results.
4. Generate the next queries.

Use `src/capstone_manager.py` for steps 1, 3, and 4.

## Why this setup

- Keeps all round history in one place (`data/functions/`).
- Avoids hard-coded arrays spread across many scripts.
- Makes external processing easy with one import CSV format.

## Commands

Use a virtual environment (recommended on macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Initialize local store (one-time)

**Option A – start from scratch:**

```bash
python src/capstone_manager.py init
```

**Option B – start with Week 5 historical data pre-loaded:**

```bash
python src/capstone_manager.py seed-legacy --overwrite
```

This creates (or overwrites):

- `data/metadata.json` (dims + objective direction per function)
- `data/functions/function_*.csv` (history per function)
- `data/templates/external_results_template.csv` (input format)

### 2) Import externally processed results

```bash
python src/capstone_manager.py import-results --results-file data/templates/external_results_template.csv
```

If needed, force a round number for all imported rows:

```bash
python src/capstone_manager.py import-results --results-file path/to/results.csv --round 6
```

### 2b) Rebuild `data/functions/` from versioned history (reproducible)

Keep canonical snapshots under `data/history/` as `round_1_results.csv`, `round_2_results.csv`, … (same columns as import-results: `function_id`, `round`, `source`, `x1`–`x8`, `y`). After clone or when you need a clean store:

```bash
python src/capstone_manager.py sync-history
```

Optional: `--history-dir path/to/history`, `--dry-run` to list files only. This **clears** existing per-function rows and re-imports all history files in round order.

### 3) Generate recommendations for next round

```bash
python src/capstone_manager.py recommend --round 7
```

Output is saved to:

- `data/submissions/round_7_next_queries.csv`

### 4) Check completeness/status

```bash
python src/capstone_manager.py status
```

## Required import CSV columns

Minimum:

- `function_id`
- `y`

Recommended full format (matches template):

- `function_id,round,source,x1,x2,...,x8,y`

Only the first `N` dimensions for each function are read, where `N` is defined in metadata.

## Existing legacy scripts

Your older one-off scripts were moved to `src/legacy/` (no deletion), so you can still reference them if needed.  
The intended path going forward is to use `src/capstone_manager.py` as the source of truth.
