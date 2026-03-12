# Model Card: BBO Capstone Optimization Approach

## Overview

**Name:** Bayesian optimization with GP surrogate and Expected Improvement (hybrid strategies).

**Type:** Black-box optimization pipeline for iterative query recommendation.

**Version:** 1.0 (as of capstone submission). Implemented in `src/capstone_manager.py`.

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

**Summary across eight functions:**  
- Best observed values (as of round 9): F1 (very small y, near zero); F2 (~0.05); F3 (~-0.08); F4 (~-24.0); F5 (3215.66 at [0.81, 0.83, 0.92, 0.99]); F6 (~-1.11); F7 (~1.89); F8 (5.79).  
- Metrics: best y per function (max for maximize objectives, min for minimize). No formal aggregate metric; performance is assessed per function and per round.  
- Variability: F5 and F7 showed large swings; F4 and F6 improved incrementally; F8 improved with more data.

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
