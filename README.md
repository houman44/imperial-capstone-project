<<<<<<< HEAD
# imperial-capstone-project
Capstone project as part of the Imperial College AI/ML certification
=======
# BBO Capstone Project

Black-Box Optimization capstone: iterative query submission and next-point recommendation for eight benchmark functions.

## Documentation

- **[Datasheet](docs/DATASHEET.md)** – Motivation, composition, collection process, and intended uses of the dataset.
- **[Model Card](docs/MODEL_CARD.md)** – Optimization approach, intended use, performance, limitations, and ethical considerations.

---

## Workflow

This repository provides a repeatable workflow for the capstone loop:

1. Generate query points.
2. Submit/evaluate externally.
3. Import returned results.
4. Generate the next queries.

Use `src/capstone_manager.py` for all of that.

## Why this setup

- Keeps all round history in one place (`data/functions/`).
- Avoids hard-coded arrays spread across many scripts.
- Makes external processing easy with one import CSV format.

## Commands

Run with your Python environment that has `numpy`, `scipy`, and `scikit-learn`:
```bash
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
>>>>>>> b1b396e (Initial commit: BBO capstone with datasheet and model card)
