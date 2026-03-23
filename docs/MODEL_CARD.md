# Model Card: BBO Capstone Optimization Approach

## Overview

**Name:** Bayesian optimization with GP surrogate and Expected Improvement (hybrid strategies).

**Type:** Black-box optimization pipeline for iterative query recommendation.

**Version:** 1.1 (final capstone submission). Implemented in `src/capstone_manager.py`. Legacy exploratory code may exist under `src/legacy/`; the supported path is the unified manager + CSV workflow.

## Intended Use

**Suitable for:**  
- Iterative optimization of expensive black-box functions with limited evaluation budget.  
- Functions with inputs in [0, 1]^d and scalar outputs.  
- Settings where a Gaussian Process surrogate and acquisition-based sampling are appropriate (smooth or moderately noisy objectives).

**Avoid:**  
- Functions with discrete or categorical inputs (without adaptation).  
- Very high-dimensional spaces (d >> 10) with very few observations.  
- Settings requiring formal safety constraints or hard feasibility regions (current implementation assumes box constraints only).  
- Real-time or latency-critical deployment (GP fitting and acquisition optimization have non-trivial compute).

## Details

**Strategy across rounds:**  
- **Rounds 1–5:** Data accumulated from initial submissions and legacy scripts. Hybrid strategies assigned by function: pure GP+EI for F1–F3; GP+SVM screening for F4, F6, F7; GP+gradient refinement for F5; full hybrid (SVM+GP+gradients) for F8.  
- **Rounds 6–9:** Unified pipeline via `capstone_manager.py`: single GP+EI with ξ=0.01, n_candidates=5000. Strategy is consistent; per-function tuning (e.g. different acquisitions) was not reintroduced in later rounds.  
- **Evolution:** Early rounds relied on manual inspection and strategic reasoning (e.g. boundary tests for F5). Later rounds automated recommendations via the capstone manager. The pipeline supports CSV import of external results and incremental updates.

**Techniques used:**  
- Gaussian Process regression (Matérn kernel, sklearn).  
- Expected Improvement acquisition.  
- Random candidate sampling with EI maximization.  
- GP fit-quality checks: convergence warnings and length-scale bounds; retry with n_restarts=10 when poor.  
- Explicit random exploration when max EI < 1e-6 (surrogate uninformative).  
- Optional (in legacy code): SVM pre-screening, gradient ascent on GP surrogate.

## Performance

**Best observed scores (local mirror, `data/functions/` through round 8)**  

These are the **best `y` achieved** over all rows currently stored per function (maximise = highest `y`; minimise = lowest `y`), per `data/metadata.json`.

| Function | Objective | Best observed \(y\) | Round (where that best occurred) |
|----------|-------------|---------------------|-----------------------------------|
| F1 | maximise | `5.722038229264154e-26` | 8 |
| F2 | maximise | `0.3594551047918857` | 1 |
| F3 | minimise | `-0.16504019029582775` | 6 |
| F4 | minimise | `-25.154033310264733` | 6 |
| F5 | maximise | `6065.515500492326` | 6 |
| F6 | minimise | `-2.152736614425762` | 1 |
| F7 | maximise | `1.5518772385254398` | 7 |
| F8 | maximise | `7.939020817385101` | 1 |

**Qualitative snapshot (latest round-8 evaluations):** F1–F2 remain near-zero scale; F4 sits around −24.0 on the most recent row; F5’s latest round-8 evaluation is `3215.66` (not the global best); F6’s latest is about `−1.11`; F7’s latest is about `0.27`; F8’s latest is about `5.79`. Use the table above for **global best-so-far** in this repo.

**Notes:**  
- The **official capstone leaderboard** may differ if additional submissions exist only on the platform. Refresh this table if you re-import further rounds.  
- No single aggregate metric is defined across functions (scales and objectives differ).  
- **Qualitative pattern:** higher-dimensional functions (e.g. F8) can remain sensitive to exploration; boundary-heavy optima are common under box constraints.

## Assumptions and Limitations

**Assumptions:**  
- Inputs are continuous and lie in [0, 1].  
- The objective is reasonably smooth (GP is appropriate).  
- Observations are conditionally independent given inputs (no temporal correlation).  
- One evaluation per (input, output) pair; no batching or async evaluation modeled.

**Constraints and failure modes:**  
- Sparse data in high dimensions (e.g. F8) can lead to unreliable surrogates.  
- GP fitting can fail to converge (kernel bounds, numerical issues); pipeline retries with more restarts and logs warnings.  
- EI can favor exploration in uninformative regions when the surrogate is poor; pipeline falls back to random exploration when max EI < 1e-6.  
- No explicit handling of heteroscedastic noise or non-stationarity.

## Ethical Considerations

**Transparency and reproducibility:**  
- The model card documents the strategy, assumptions, and limitations.  
- Code is organized in `capstone_manager.py` with a clear init → import → recommend workflow.  
- Data format (CSV with function_id, round, x1–x8, y) is specified to support reproduction and auditing.  
- A datasheet documents the dataset; linking both in the README supports responsible use and critique.

**Real-world adaptation:**  
- Practitioners adapting this approach should: validate on their own functions; adjust acquisition and strategy for domain constraints; and document any modifications.  
- The model card supports informed adoption by stating intended and inappropriate uses clearly.
