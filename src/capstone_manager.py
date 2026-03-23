#!/usr/bin/env python3
"""
Capstone Kaggle round manager.

This script organizes an iterative optimization workflow:
1) Initialize local data store
2) Append externally processed results
3) Generate next query recommendations
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern


MAX_DIMS = 8


@dataclass(frozen=True)
class FunctionSpec:
    function_id: int
    dims: int
    objective: str  # "maximize" or "minimize"


DEFAULT_SPECS: List[FunctionSpec] = [
    FunctionSpec(1, 2, "maximize"),
    FunctionSpec(2, 2, "maximize"),
    FunctionSpec(3, 3, "minimize"),
    FunctionSpec(4, 4, "minimize"),
    FunctionSpec(5, 4, "maximize"),
    FunctionSpec(6, 5, "minimize"),
    FunctionSpec(7, 6, "maximize"),
    FunctionSpec(8, 8, "maximize"),
]


def metadata_path(data_dir: Path) -> Path:
    return data_dir / "metadata.json"


def function_csv_path(data_dir: Path, function_id: int) -> Path:
    return data_dir / "functions" / f"function_{function_id}.csv"


def load_specs(data_dir: Path) -> List[FunctionSpec]:
    raw = json.loads(metadata_path(data_dir).read_text(encoding="utf-8"))
    return [FunctionSpec(**item) for item in raw["functions"]]


def ensure_store_exists(data_dir: Path) -> None:
    if not metadata_path(data_dir).exists():
        raise FileNotFoundError(
            f"Missing store metadata at {metadata_path(data_dir)}. "
            "Run `init` first."
        )


def create_store(data_dir: Path, overwrite: bool) -> None:
    functions_dir = data_dir / "functions"
    submissions_dir = data_dir / "submissions"
    templates_dir = data_dir / "templates"

    if data_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{data_dir} already exists. Pass --overwrite to recreate it."
        )

    functions_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": 1,
        "functions": [spec.__dict__ for spec in DEFAULT_SPECS],
    }
    metadata_path(data_dir).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for spec in DEFAULT_SPECS:
        path = function_csv_path(data_dir, spec.function_id)
        header = ["round", "source", *[f"x{i}" for i in range(1, spec.dims + 1)], "y"]
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    template_path = templates_dir / "external_results_template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["function_id", "round", "source", *[f"x{i}" for i in range(1, MAX_DIMS + 1)], "y"]
        )
        writer.writerow([1, 1, "kaggle", 0.5, 0.5, "", "", "", "", "", "", 0.1234])

    print(f"Initialized store at: {data_dir}")
    print(f"Template for external results: {template_path}")


def _to_float(value: str) -> float:
    return float(str(value).strip())


def _spec_map(specs: Sequence[FunctionSpec]) -> Dict[int, FunctionSpec]:
    return {s.function_id: s for s in specs}


def clear_function_history(data_dir: Path) -> None:
    """Reset each function_*.csv to header only (keeps metadata.json)."""
    ensure_store_exists(data_dir)
    specs = load_specs(data_dir)
    for spec in specs:
        path = function_csv_path(data_dir, spec.function_id)
        header = ["round", "source", *[f"x{i}" for i in range(1, spec.dims + 1)], "y"]
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    print(f"Cleared per-function history under {data_dir / 'functions'}")


def _history_csv_sort_key(path: Path) -> Tuple[int, int, str]:
    """Sort key: canonical round_N_results.csv first by N, then other files by name."""
    m = re.match(r"^round_(\d+)_results\.csv$", path.name, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)), path.name)
    m2 = re.match(r"^round(\d+)[._-]", path.name, re.IGNORECASE)
    if m2:
        return (0, int(m2.group(1)), path.name)
    return (1, 0, path.name)


def list_history_csvs(history_dir: Path) -> List[Path]:
    if not history_dir.is_dir():
        return []
    paths = [p for p in history_dir.glob("*.csv") if not p.name.startswith(".")]
    return sorted(paths, key=_history_csv_sort_key)


def sync_history(data_dir: Path, history_dir: Path, *, dry_run: bool) -> None:
    """
    Rebuild data/functions/*.csv from CSV snapshots in history_dir.

    Expected layout: one or more files named like round_1_results.csv, round_2_results.csv, ...
    Each file should have columns function_id, round, source, x1..x8, y (same as import-results).

    Files are applied in ascending round order (from filename). This gives a reproducible
    history: commit data/history/*.csv under version control, then run sync-history after clone.
    """
    ensure_store_exists(data_dir)
    files = list_history_csvs(history_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {history_dir}. "
            "Add round_N_results.csv files (see README) or pass --history-dir."
        )

    print("History files (order):")
    for p in files:
        print(f"  - {p.name}")

    if dry_run:
        print("Dry run: no changes written.")
        return

    clear_function_history(data_dir)
    total = 0
    for results_file in files:
        total += append_external_results(data_dir, results_file, round_override=None)
    print(f"Sync complete. Imported {len(files)} file(s), {total} total row(s).")


def append_external_results(data_dir: Path, results_file: Path, round_override: int | None) -> int:
    ensure_store_exists(data_dir)
    specs = load_specs(data_dir)
    by_id = _spec_map(specs)

    inserted = 0
    with results_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"function_id", "y"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Results CSV must include at least: function_id,y")

        for row in reader:
            function_id = int(row["function_id"])
            if function_id not in by_id:
                raise ValueError(f"Unknown function_id={function_id}")
            spec = by_id[function_id]

            round_value = round_override if round_override is not None else int(row.get("round") or 0)
            source = row.get("source") or "external"
            xs = [_to_float(row[f"x{i}"]) for i in range(1, spec.dims + 1)]
            y = _to_float(row["y"])

            path = function_csv_path(data_dir, function_id)
            with path.open("a", newline="", encoding="utf-8") as wf:
                csv.writer(wf).writerow([round_value, source, *xs, y])
            inserted += 1

    print(f"Appended {inserted} rows from: {results_file}")
    return inserted


def load_function_dataset(data_dir: Path, spec: FunctionSpec) -> tuple[np.ndarray, np.ndarray]:
    path = function_csv_path(data_dir, spec.function_id)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"No rows for function {spec.function_id}. "
            "Import at least one external result first."
        )

    X = np.array(
        [[_to_float(row[f"x{i}"]) for i in range(1, spec.dims + 1)] for row in rows],
        dtype=float,
    )
    y = np.array([_to_float(row["y"]) for row in rows], dtype=float)
    return X, y


# Thresholds for fit-quality checks
LENGTH_SCALE_LOW = 1e-4
LENGTH_SCALE_HIGH = 1e4
EI_EXPLORATION_THRESHOLD = 1e-6  # Below this max EI, use random exploration


@dataclass
class GPFitQuality:
    """Result of GP fit quality checks."""
    convergence_ok: bool = True
    length_scale_ok: bool = True
    warnings: List[str] = field(default_factory=list)


def _check_gp_fit_quality(gp: GaussianProcessRegressor, dims: int) -> GPFitQuality:
    """Check GP fit quality: convergence and length-scale bounds."""
    result = GPFitQuality()
    try:
        # Access Matern kernel (k2 in C * Matern)
        matern = gp.kernel_.k2
        ls = np.atleast_1d(matern.length_scale)
        for i, scale in enumerate(ls):
            if scale <= LENGTH_SCALE_LOW or scale >= LENGTH_SCALE_HIGH:
                result.length_scale_ok = False
                result.warnings.append(
                    f"length_scale[{i}]={scale:.2e} near bounds ({LENGTH_SCALE_LOW}, {LENGTH_SCALE_HIGH})"
                )
    except (AttributeError, TypeError) as e:
        result.warnings.append(f"Could not check length scales: {e}")
    return result


def expected_improvement(X: np.ndarray, gp: GaussianProcessRegressor, y_best: float, xi: float) -> np.ndarray:
    mu, sigma = gp.predict(X, return_std=True)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    sigma = np.maximum(sigma, 1e-12)

    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def suggest_point(
    X: np.ndarray,
    y: np.ndarray,
    *,
    maximize: bool,
    n_candidates: int,
    xi: float,
    random_seed: int,
    function_id: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, GPFitQuality]:
    transformed_y = y if maximize else -y
    dims = X.shape[1]
    y_best = float(np.max(transformed_y))
    rng = np.random.default_rng(random_seed)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dims), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_seed,
    )

    # Capture GP convergence warnings explicitly
    gp_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gp.fit(X, transformed_y)
        for _w in w:
            msg = str(_w.message)
            gp_warnings.append(f"{_w.category.__name__}: {msg}")
            if verbose:
                logging.warning(f"F{function_id} GP fit: {msg}")

    fit_quality = _check_gp_fit_quality(gp, dims)
    fit_quality.warnings = gp_warnings + fit_quality.warnings
    fit_quality.convergence_ok = not any("ConvergenceWarning" in _w for _w in gp_warnings)

    # Poor fit: retry with more optimizer restarts
    if not fit_quality.convergence_ok or not fit_quality.length_scale_ok:
        if verbose:
            logging.info(f"F{function_id} retrying GP fit with n_restarts=10")
        gp_retry = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=random_seed + 1000,
        )
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            gp_retry.fit(X, transformed_y)
            for _w in w2:
                if verbose:
                    logging.warning(f"F{function_id} GP retry: {_w.message}")
        gp = gp_retry
        fit_quality = _check_gp_fit_quality(gp, dims)

    candidates = rng.uniform(0.0, 1.0, size=(n_candidates, dims))
    ei = expected_improvement(candidates, gp, y_best=y_best, xi=xi)
    max_ei = float(np.max(ei))

    # Explicit exploration when EI is uniformly low (surrogate uninformative)
    if max_ei < EI_EXPLORATION_THRESHOLD:
        if verbose:
            logging.info(
                f"F{function_id} max EI={max_ei:.2e} < {EI_EXPLORATION_THRESHOLD}, "
                "using random exploration"
            )
        p = rng.uniform(0.0, 1.0, size=(dims,))
        # Ensure not too close to existing points
        for _ in range(100):
            if not np.any(np.all(np.isclose(X, p, atol=1e-6), axis=1)):
                return p, 0.0, fit_quality
            p = rng.uniform(0.0, 1.0, size=(dims,))
        return p, 0.0, fit_quality

    order = np.argsort(ei)[::-1]
    for idx in order:
        p = candidates[idx]
        if not np.any(np.all(np.isclose(X, p, atol=1e-6), axis=1)):
            return p, float(ei[idx]), fit_quality

    return rng.uniform(0.0, 1.0, size=(dims,)), 0.0, fit_quality


def write_recommendations_csv(
    path: Path,
    rows: Iterable[tuple[int, np.ndarray, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function_id", *[f"x{i}" for i in range(1, MAX_DIMS + 1)], "acquisition_score"])
        for function_id, point, score in rows:
            padded = [*point.tolist(), *([""] * (MAX_DIMS - len(point)))]
            writer.writerow([function_id, *padded, score])


def recommend_next_round(
    data_dir: Path,
    round_id: int,
    n_candidates: int,
    xi: float,
    random_seed: int,
    verbose: bool = True,
) -> None:
    ensure_store_exists(data_dir)
    specs = load_specs(data_dir)
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")

    results: List[tuple[int, np.ndarray, float]] = []
    print(f"Generating recommendations for round {round_id}")
    for spec in specs:
        X, y = load_function_dataset(data_dir, spec)
        point, score, fit_quality = suggest_point(
            X,
            y,
            maximize=(spec.objective == "maximize"),
            n_candidates=n_candidates,
            xi=xi,
            random_seed=random_seed + spec.function_id,
            function_id=spec.function_id,
            verbose=verbose,
        )
        results.append((spec.function_id, point, score))
        pt_str = "-".join(f"{x:.6f}" for x in point)
        print(f"F{spec.function_id}: {pt_str}")
        if verbose and fit_quality.warnings:
            for w in fit_quality.warnings[:2]:  # Limit to first 2 per function
                print(f"  [fit] {w}")

    out_file = data_dir / "submissions" / f"round_{round_id}_next_queries.csv"
    write_recommendations_csv(out_file, results)
    print(f"Saved recommendations: {out_file}")

    print("\n--- Copy for submission ---")
    for fid, point, _ in results:
        pt_str = "-".join(f"{x:.6f}" for x in point)
        print(f"F{fid}: {pt_str}")


def print_status(data_dir: Path) -> None:
    ensure_store_exists(data_dir)
    specs = load_specs(data_dir)
    print("Current store status")
    for spec in specs:
        path = function_csv_path(data_dir, spec.function_id)
        with path.open("r", newline="", encoding="utf-8") as f:
            n_rows = max(0, sum(1 for _ in f) - 1)
        print(
            f"Function {spec.function_id}: dims={spec.dims}, "
            f"objective={spec.objective}, rows={n_rows}"
        )


def _get_week5_legacy_data() -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """Historical (X, y) from Week 5 for all 8 functions."""
    return {
        1: (
            np.array([
                [0.19, 0.04], [0.76, 0.54], [0.44, 0.80], [0.35, 0.32],
                [0.30, 0.32], [0.57, 0.03], [0.57, 0.18], [0.83, 0.79],
                [0.61, 0.84], [0.04, 0.29], [0.88, 0.29], [0.96, 0.03],
                [0.58, 0.54], [0.58, 0.54],
            ]),
            np.array([0.26, 0.08, 5.83, 7.76, 5.18, 0.0, 0.0, 4.03, 5.84, 0.0, 0.0, 0.0, 0.02, 5.494870748162966e-8]),
        ),
        2: (
            np.array([
                [0.19, 0.04], [0.76, 0.54], [0.44, 0.80], [0.35, 0.32],
                [0.30, 0.32], [0.57, 0.03], [0.57, 0.18], [0.83, 0.79],
                [0.61, 0.84], [0.04, 0.29], [0.88, 0.29], [0.96, 0.03],
                [0.70, 0.93], [0.70, 0.93],
            ]),
            np.array([0.40, -0.09, 0.23, 0.68, 0.43, -0.23, 0.12, 0.18, 0.22, -0.06, 0.09, -0.09, 0.49, 0.49954705055291193]),
        ),
        3: (
            np.array([
                [0.19, 0.04, 0.61], [0.76, 0.54, 0.66], [0.44, 0.80, 0.21],
                [0.35, 0.32, 0.12], [0.30, 0.32, 0.62], [0.57, 0.03, 0.88],
                [0.57, 0.18, 0.34], [0.83, 0.79, 0.11], [0.61, 0.84, 0.98],
                [0.04, 0.29, 0.98], [0.88, 0.29, 0.41], [0.96, 0.03, 0.09],
                [0.93, 0.85, 0.64], [0.85, 0.94, 0.48], [0.99, 0.90, 0.91],
                [0.70, 0.75, 0.35], [0.70, 0.30, 0.60],
            ]),
            np.array([-0.27, -0.50, -0.77, -0.41, -0.30, -0.70, -0.40, -0.69, -0.74, -0.73, -0.47, -0.63, -0.85, -0.83, -0.93, -0.79, -0.06382715038922124]),
        ),
        4: (
            np.array([
                [0.19, 0.04, 0.61, 0.41], [0.76, 0.54, 0.66, 0.36],
                [0.44, 0.80, 0.21, 0.15], [0.35, 0.32, 0.12, 0.47],
                [0.30, 0.32, 0.62, 0.10], [0.57, 0.03, 0.88, 0.82],
                [0.57, 0.18, 0.34, 0.79], [0.83, 0.79, 0.11, 0.69],
                [0.61, 0.84, 0.98, 0.21], [0.04, 0.29, 0.98, 0.56],
                [0.88, 0.29, 0.41, 0.03], [0.96, 0.03, 0.09, 0.95],
                [0.93, 0.85, 0.64, 0.59], [0.85, 0.94, 0.48, 0.76],
                [0.99, 0.90, 0.91, 0.33], [0.30, 0.29, 0.46, 0.93],
                [0.90, 0.55, 0.95, 0.48], [0.50, 0.66, 0.31, 0.26],
                [0.06, 0.58, 0.76, 0.66], [0.67, 0.71, 0.77, 0.86],
                [0.22, 0.85, 0.88, 0.88], [0.26, 0.84, 0.95, 0.89],
                [0.00, 1.00, 1.00, 1.00], [0.00, 1.00, 1.00, 0.95],
                [0.55, 0.45, 0.52, 0.28], [0.54, 0.42, 0.55, 0.29],
                [0.51, 0.42, 0.46, 0.32], [0.51, 0.44, 0.47, 0.30],
                [0.51, 0.44, 0.46, 0.30], [0.51, 0.43, 0.46, 0.31],
                [0.53, 0.47, 0.38, 0.30],
            ]),
            np.array([-3.16, -3.00, -3.15, -3.03, -3.10, -2.95, -3.02, -2.98, -3.14, -3.07, -2.99, -2.92, -3.17, -3.19, -3.24, -3.04, -3.25, -3.08, -3.12, -3.22, -3.28, -3.29, -3.36, -3.36, -3.38, -3.38, -3.38, -3.38, -3.38, -3.38, -3.3897164075807775]),
        ),
        5: (
            np.array([
                [0.19, 0.04, 0.61, 0.41], [0.76, 0.54, 0.66, 0.36],
                [0.44, 0.80, 0.21, 0.15], [0.35, 0.32, 0.12, 0.47],
                [0.30, 0.32, 0.62, 0.10], [0.57, 0.03, 0.88, 0.82],
                [0.57, 0.18, 0.34, 0.79], [0.83, 0.79, 0.11, 0.69],
                [0.61, 0.84, 0.98, 0.21], [0.04, 0.29, 0.98, 0.56],
                [0.88, 0.29, 0.41, 0.03], [0.96, 0.03, 0.09, 0.95],
                [0.93, 0.85, 0.64, 0.59], [0.85, 0.94, 0.48, 0.76],
                [0.99, 0.90, 0.91, 0.33], [0.30, 0.29, 0.46, 0.93],
                [0.90, 0.55, 0.95, 0.48], [0.50, 0.66, 0.31, 0.26],
                [0.06, 0.58, 0.76, 0.66], [0.67, 0.71, 0.77, 0.86],
                [0.22, 0.85, 0.88, 0.88], [0.26, 0.84, 0.95, 0.89],
                [0.00, 1.00, 1.00, 1.00], [0.00, 1.00, 1.00, 0.95],
                [0.00, 1.00, 0.95, 1.00],
            ]),
            np.array([64.4, 18.3, 0.11, 109.6, 63.4, 258.4, 8.85, 4.21, 28.3, 55.5, 0.51, 18.9, 113.7, 356.9, 67.1, 78.5, 432.9, 18.2, 44.3, 257.8, 1088.9, 1550.9, 4440.5, 3819.7, 3819.7407576895994]),
        ),
        6: (
            np.array([
                [0.19, 0.04, 0.61, 0.41, 0.93], [0.76, 0.54, 0.66, 0.36, 0.03],
                [0.44, 0.80, 0.21, 0.15, 0.12], [0.35, 0.32, 0.12, 0.47, 0.10],
                [0.30, 0.32, 0.62, 0.10, 0.50], [0.57, 0.03, 0.88, 0.82, 0.48],
                [0.57, 0.18, 0.34, 0.79, 0.66], [0.83, 0.79, 0.11, 0.69, 0.41],
                [0.61, 0.84, 0.98, 0.21, 0.21], [0.04, 0.29, 0.98, 0.56, 0.06],
                [0.88, 0.29, 0.41, 0.03, 0.88], [0.96, 0.03, 0.09, 0.95, 0.28],
                [0.93, 0.85, 0.64, 0.59, 0.36], [0.85, 0.94, 0.48, 0.76, 0.26],
                [0.99, 0.90, 0.91, 0.33, 0.03], [0.30, 0.29, 0.46, 0.93, 0.70],
                [0.90, 0.55, 0.95, 0.48, 0.07], [0.50, 0.66, 0.31, 0.26, 0.46],
                [0.06, 0.58, 0.76, 0.66, 0.30], [0.67, 0.71, 0.77, 0.86, 0.28],
                [0.22, 0.85, 0.88, 0.88, 0.13], [0.26, 0.84, 0.95, 0.89, 0.13],
                [0.00, 1.00, 1.00, 1.00, 0.10], [0.00, 1.00, 1.00, 0.95, 0.07],
                [0.55, 0.20, 0.75, 1.00, 0.03], [0.55, 0.18, 0.72, 1.00, 0.03],
            ]),
            np.array([-0.58, -0.47, -0.49, -0.55, -0.50, -0.63, -0.57, -0.63, -0.56, -0.50, -0.50, -0.52, -0.67, -0.67, -0.58, -0.55, -0.63, -0.52, -0.58, -0.67, -0.64, -0.64, -0.64, -0.63, -0.64, -0.6368618203511591]),
        ),
        7: (
            np.array([
                [0.19, 0.04, 0.61, 0.41, 0.93, 0.29], [0.76, 0.54, 0.66, 0.36, 0.03, 0.12],
                [0.44, 0.80, 0.21, 0.15, 0.12, 0.10], [0.35, 0.32, 0.12, 0.47, 0.10, 0.50],
                [0.30, 0.32, 0.62, 0.10, 0.50, 0.48], [0.57, 0.03, 0.88, 0.82, 0.48, 0.66],
                [0.57, 0.18, 0.34, 0.79, 0.66, 0.41], [0.83, 0.79, 0.11, 0.69, 0.41, 0.21],
                [0.61, 0.84, 0.98, 0.21, 0.21, 0.06], [0.04, 0.29, 0.98, 0.56, 0.06, 0.88],
                [0.88, 0.29, 0.41, 0.03, 0.88, 0.28], [0.96, 0.03, 0.09, 0.95, 0.28, 0.36],
                [0.93, 0.85, 0.64, 0.59, 0.36, 0.26], [0.85, 0.94, 0.48, 0.76, 0.26, 0.03],
                [0.99, 0.90, 0.91, 0.33, 0.03, 0.70], [0.30, 0.29, 0.46, 0.93, 0.70, 0.07],
                [0.90, 0.55, 0.95, 0.48, 0.07, 0.46], [0.50, 0.66, 0.31, 0.26, 0.46, 0.30],
                [0.06, 0.58, 0.76, 0.66, 0.30, 0.28], [0.67, 0.71, 0.77, 0.86, 0.28, 0.13],
                [0.22, 0.85, 0.88, 0.88, 0.13, 0.13], [0.26, 0.84, 0.95, 0.89, 0.13, 0.10],
                [0.00, 1.00, 1.00, 1.00, 0.10, 0.07], [0.01, 0.36, 0.51, 0.21, 0.43, 0.74],
                [0.01, 0.36, 0.51, 0.21, 0.43, 0.74],
            ]),
            np.array([0.63, 0.17, 0.09, 0.22, 0.25, 0.55, 0.33, 0.29, 0.45, 0.59, 0.13, 0.09, 0.56, 0.54, 0.70, 0.23, 0.66, 0.24, 0.48, 0.69, 0.74, 0.78, 0.87, 1.89, 1.8942067659620383]),
        ),
        8: (
            np.array([
                [0.19, 0.04, 0.61, 0.41, 0.93, 0.29, 0.74, 0.59],
                [0.76, 0.54, 0.66, 0.36, 0.03, 0.12, 0.10, 0.15],
                [0.44, 0.80, 0.21, 0.15, 0.12, 0.10, 0.50, 0.48],
                [0.03, 0.07, 0.015, 0.05, 1.00, 0.85, 0.55, 0.92],
            ]),
            np.array([64.4, 18.3, 0.11, 9.536385]),
        ),
    }


def seed_legacy(data_dir: Path, overwrite: bool) -> None:
    """Pre-populate store with Week 5 historical data from legacy scripts."""
    create_store(data_dir=data_dir, overwrite=overwrite)
    legacy = _get_week5_legacy_data()
    for fid, (X, y) in legacy.items():
        path = function_csv_path(data_dir, fid)
        spec = next(s for s in DEFAULT_SPECS if s.function_id == fid)
        header = ["round", "source", *[f"x{i}" for i in range(1, spec.dims + 1)], "y"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(X)):
                w.writerow([5, "legacy_week5", *X[i].tolist(), float(y[i])])
        print(f"Seeded function {fid}: {len(X)} rows")
    print(f"Legacy data loaded. Run: python src/capstone_manager.py recommend --round 6")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capstone optimization workflow manager")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory used for metadata, per-function datasets, and submissions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Create a clean local data store")
    p_init.add_argument("--overwrite", action="store_true")

    p_seed = subparsers.add_parser("seed-legacy", help="Initialize store and pre-fill with Week 5 historical data")
    p_seed.add_argument("--overwrite", action="store_true")

    p_import = subparsers.add_parser("import-results", help="Append externally processed results CSV")
    p_import.add_argument("--results-file", required=True)
    p_import.add_argument("--round", type=int, default=None, help="Override round for all imported rows")

    p_recommend = subparsers.add_parser("recommend", help="Generate next query recommendations")
    p_recommend.add_argument("--round", type=int, required=True)
    p_recommend.add_argument("--n-candidates", type=int, default=5000)
    p_recommend.add_argument("--xi", type=float, default=0.01)
    p_recommend.add_argument("--seed", type=int, default=42)
    p_recommend.add_argument("--quiet", action="store_true", help="Suppress fit-quality warnings and logs")

    subparsers.add_parser("status", help="Show data completeness per function")

    p_sync = subparsers.add_parser(
        "sync-history",
        help="Rebuild data/functions/*.csv from CSV snapshots in a history directory",
    )
    p_sync.add_argument(
        "--history-dir",
        default="data/history",
        help="Directory containing round_N_results.csv files (default: data/history)",
    )
    p_sync.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be imported without modifying the store",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    if args.command == "init":
        create_store(data_dir=data_dir, overwrite=args.overwrite)
    elif args.command == "seed-legacy":
        seed_legacy(data_dir=data_dir, overwrite=args.overwrite)
    elif args.command == "import-results":
        append_external_results(
            data_dir=data_dir,
            results_file=Path(args.results_file),
            round_override=args.round,
        )
    elif args.command == "sync-history":
        sync_history(
            data_dir=data_dir,
            history_dir=Path(args.history_dir),
            dry_run=args.dry_run,
        )
    elif args.command == "recommend":
        recommend_next_round(
            data_dir=data_dir,
            round_id=args.round,
            n_candidates=args.n_candidates,
            xi=args.xi,
            random_seed=args.seed,
            verbose=not getattr(args, "quiet", False),
        )
    elif args.command == "status":
        print_status(data_dir=data_dir)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
