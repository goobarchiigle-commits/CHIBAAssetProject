"""Lightweight correlation analysis for batch experiment results.

Purpose: visibility only — identify near-duplicate strategies.
No portfolio optimization. No allocation decisions.

All functions are pure and deterministic: same inputs → same outputs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_pairwise_correlation(
    returns_map: Dict[str, pd.Series],
    min_overlap: int = 30,
) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix on aligned daily returns.

    Keys of returns_map become column/row labels.
    Pairs with fewer than min_overlap overlapping observations get NaN.
    Output is deterministic: same input → same matrix.
    """
    if not returns_map:
        return pd.DataFrame()

    # Deterministic column order: sort by key
    ordered_keys = sorted(returns_map.keys())
    df = pd.DataFrame({k: returns_map[k] for k in ordered_keys})
    df = df.sort_index()  # ensure row order is deterministic

    corr = df.corr(method="pearson", min_periods=min_overlap)
    return corr


def flag_near_duplicates(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.95,
) -> List[Tuple[str, str, float]]:
    """Return sorted list of (id1, id2, correlation) for |corr| >= threshold.

    Only upper triangle (i < j) to avoid duplicates.
    Sorted descending by |correlation|, then by id1, id2 for tie-breaking.
    """
    if corr_matrix.empty:
        return []

    ids = list(corr_matrix.columns)
    pairs: List[Tuple[str, str, float]] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            val = corr_matrix.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append((ids[i], ids[j], float(val)))

    pairs.sort(key=lambda x: (-abs(x[2]), x[0], x[1]))
    return pairs


def rolling_correlation_stability(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 60,
) -> float:
    """Std of rolling Pearson correlation between two return series.

    Lower value = more stable relationship.
    Returns nan if insufficient data.
    """
    if len(s1) < window or len(s2) < window:
        return float("nan")

    aligned = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    if len(aligned) < window:
        return float("nan")

    rolling = aligned["a"].rolling(window).corr(aligned["b"])
    return float(rolling.std())


def correlation_hash(corr_matrix: pd.DataFrame) -> str:
    """Stable 16-char hash of a correlation matrix. Rounded to 4dp for stability."""
    if corr_matrix.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    rounded = corr_matrix.round(4).fillna(0.0)
    # Use sorted column order for determinism
    cols = sorted(rounded.columns)
    payload = json.dumps(
        {c: {r: rounded.loc[r, c] for r in sorted(rounded.index)} for c in cols},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_correlation_report(
    corr_matrix: pd.DataFrame,
    near_duplicate_threshold: float = 0.95,
) -> dict:
    """Build a structured correlation report for artifact storage."""
    if corr_matrix.empty:
        return {"strategy_count": 0, "near_duplicates": [], "matrix_hash": correlation_hash(corr_matrix)}

    near_dups = flag_near_duplicates(corr_matrix, threshold=near_duplicate_threshold)
    ids = list(corr_matrix.columns)

    # Mean absolute off-diagonal correlation per strategy
    off_diag: Dict[str, float] = {}
    for sid in ids:
        vals = [corr_matrix.loc[sid, other] for other in ids if other != sid]
        vals = [v for v in vals if pd.notna(v)]
        off_diag[sid] = round(float(np.mean(np.abs(vals))), 4) if vals else 0.0

    return {
        "strategy_count": len(ids),
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_count": len(near_dups),
        "near_duplicates": [
            {"id1": a, "id2": b, "correlation": round(c, 4)} for a, b, c in near_dups
        ],
        "mean_abs_correlation_by_strategy": dict(sorted(off_diag.items())),
        "matrix_hash": correlation_hash(corr_matrix),
    }
