"""ParameterTopologyAnalyzer — deterministic parameter surface topology analysis.

Distinguishes robust plateau regions from isolated overfit spikes.
No randomness, no smoothing, no stochastic sampling.

All neighbor relationships are derived from the observed parameter grid.
Same inputs → same outputs, always.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.research.analysis.contract import TopologyResult

# Fraction of self's score that defines "in the plateau"
_PLATEAU_TOL = 0.10
# Maximum score magnitude considered near-zero (avoid division)
_EPS = 1e-9
# Number of bins for surface entropy computation
_ENTROPY_BINS = 10
# Robust-core survivorship threshold
_ROBUST_SURVIVORSHIP_THRESH = 0.60


class ParameterTopologyAnalyzer:
    """Analyze parameter topology for a population of experiments.

    Args:
        linf_radius:  L∞ grid distance considered "neighbor" (default 1 = adjacent cells).
        plateau_tol:  fraction of |score| defining plateau tolerance (default 0.10 = 10%).
    """

    def __init__(self, linf_radius: int = 1, plateau_tol: float = _PLATEAU_TOL) -> None:
        if linf_radius < 1:
            raise ValueError(f"linf_radius must be >= 1, got {linf_radius}")
        if not (0.0 < plateau_tol < 1.0):
            raise ValueError(f"plateau_tol must be in (0, 1), got {plateau_tol}")
        self.linf_radius = linf_radius
        self.plateau_tol = plateau_tol

    def analyze(
        self,
        experiments: List[Tuple[str, dict, float, bool]],
    ) -> Dict[str, TopologyResult]:
        """Compute topology metrics for each experiment.

        Args:
            experiments: List of (experiment_id, params_dict, score, survived).
                         params_dict values must be numeric (int/float) or will be skipped.

        Returns:
            Dict mapping experiment_id → TopologyResult.
            Deterministic: same input list (any order) → same output dict.
        """
        if not experiments:
            return {}

        # Canonical index: sorted by experiment_id for stable iteration
        exp_index: Dict[str, Tuple[dict, float, bool]] = {}
        for eid, params, score, survived in experiments:
            if eid in exp_index:
                raise ValueError(f"Duplicate experiment_id: {eid}")
            exp_index[eid] = (params, float(score), bool(survived))

        grid = self._build_grid(exp_index)

        return {
            eid: self._analyze_one(eid, exp_index, grid)
            for eid in sorted(exp_index.keys())
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_grid(self, exp_index: Dict[str, Tuple[dict, float, bool]]) -> Dict[str, List]:
        """Sorted unique values per numeric parameter dimension."""
        param_values: Dict[str, set] = {}
        for params, _, _ in exp_index.values():
            for k, v in params.items():
                if _is_numeric(v):
                    param_values.setdefault(k, set()).add(float(v))
        return {k: sorted(vs) for k, vs in sorted(param_values.items())}

    def _get_numeric_params(self, params: dict, grid: Dict[str, List]) -> Dict[str, float]:
        return {k: float(v) for k, v in params.items() if _is_numeric(v) and k in grid}

    def _grid_position(self, numeric_params: Dict[str, float], grid: Dict[str, List]) -> Dict[str, int]:
        pos = {}
        for k, v in numeric_params.items():
            grid_vals = grid[k]
            # Binary search would be faster but list.index is deterministic and correct
            try:
                pos[k] = grid_vals.index(v)
            except ValueError:
                # Value not in grid (shouldn't happen if built from same data)
                pass
        return pos

    def _find_neighbors(
        self,
        exp_id: str,
        my_pos: Dict[str, int],
        my_keys: frozenset,
        exp_index: Dict[str, Tuple[dict, float, bool]],
        grid: Dict[str, List],
        radius: int,
    ) -> List[str]:
        """Find experiments within L∞ radius in grid space. Sorted for stability."""
        neighbors = []
        for other_id in sorted(exp_index.keys()):
            if other_id == exp_id:
                continue
            other_params, _, _ = exp_index[other_id]
            other_numeric = self._get_numeric_params(other_params, grid)
            if frozenset(other_numeric.keys()) != my_keys:
                continue  # different parameter space
            other_pos = self._grid_position(other_numeric, grid)
            if all(
                abs(my_pos.get(k, 0) - other_pos.get(k, 0)) <= radius
                for k in my_keys
            ):
                neighbors.append(other_id)
        return neighbors

    def _analyze_one(
        self,
        exp_id: str,
        exp_index: Dict[str, Tuple[dict, float, bool]],
        grid: Dict[str, List],
    ) -> TopologyResult:
        params, score, survived = exp_index[exp_id]
        numeric = self._get_numeric_params(params, grid)
        my_keys = frozenset(numeric.keys())
        my_pos = self._grid_position(numeric, grid)

        neighbors = self._find_neighbors(exp_id, my_pos, my_keys, exp_index, grid, self.linf_radius)

        if not neighbors:
            return TopologyResult(
                experiment_id=exp_id,
                topology_fragility_score=1.0,
                plateau_width=0.0,
                neighbor_decay_rate=0.0,
                parameter_elasticity=0.0,
                surface_entropy=0.0,
                neighborhood_survivorship=0.0,
                neighbor_count=0,
            )

        neighbor_scores = [exp_index[nid][1] for nid in neighbors]
        neighbor_survived = [exp_index[nid][2] for nid in neighbors]
        mean_neighbor = sum(neighbor_scores) / len(neighbor_scores)

        fragility = _fragility(score, mean_neighbor)
        plateau = _plateau_width(score, neighbor_scores, self.plateau_tol)
        decay = max(0.0, score - mean_neighbor)
        entropy = _surface_entropy(neighbor_scores)
        n_survivorship = sum(1 for s in neighbor_survived if s) / len(neighbor_survived)
        elasticity = self._compute_elasticity(exp_id, numeric, my_keys, my_pos, score, exp_index, grid)

        return TopologyResult(
            experiment_id=exp_id,
            topology_fragility_score=round(fragility, 6),
            plateau_width=round(plateau, 6),
            neighbor_decay_rate=round(decay, 6),
            parameter_elasticity=round(elasticity, 6),
            surface_entropy=round(entropy, 6),
            neighborhood_survivorship=round(n_survivorship, 6),
            neighbor_count=len(neighbors),
        )

    def _compute_elasticity(
        self,
        exp_id: str,
        numeric: Dict[str, float],
        my_keys: frozenset,
        my_pos: Dict[str, int],
        score: float,
        exp_index: Dict[str, Tuple[dict, float, bool]],
        grid: Dict[str, List],
    ) -> float:
        """Mean |Δscore| / |Δparam_normalized| across single-dimension perturbations."""
        samples: List[float] = []
        for k in sorted(my_keys):
            grid_vals = grid[k]
            n = len(grid_vals)
            if n <= 1:
                continue
            my_idx = my_pos[k]
            for other_id in sorted(exp_index.keys()):
                if other_id == exp_id:
                    continue
                other_params, other_score, _ = exp_index[other_id]
                other_numeric = self._get_numeric_params(other_params, grid)
                if frozenset(other_numeric.keys()) != my_keys:
                    continue
                # Must match on all OTHER dimensions exactly
                if not all(other_numeric[kk] == numeric[kk] for kk in my_keys if kk != k):
                    continue
                other_pos = self._grid_position(other_numeric, grid)
                delta_idx = abs(my_idx - other_pos[k])
                if delta_idx == 0:
                    continue
                delta_param = delta_idx / (n - 1)  # normalized [0,1]
                delta_score = abs(score - other_score)
                samples.append(delta_score / delta_param)
        return sum(samples) / len(samples) if samples else 0.0


# ── Module-level helpers ───────────────────────────────────────────────────────

def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fragility(score: float, mean_neighbor: float) -> float:
    """0=stable plateau, 1=isolated spike."""
    drop = max(0.0, score - mean_neighbor)
    denom = abs(score) + _EPS
    return min(1.0, drop / denom)


def _plateau_width(score: float, neighbor_scores: List[float], tol: float) -> float:
    """Fraction of neighbors within tol*|score| of score."""
    if not neighbor_scores:
        return 0.0
    threshold = max(abs(score) * tol, 0.05)
    within = sum(1 for ns in neighbor_scores if abs(ns - score) <= threshold)
    return within / len(neighbor_scores)


def _surface_entropy(scores: List[float]) -> float:
    """Shannon entropy (nats) of discretized neighbor score distribution."""
    if len(scores) <= 1:
        return 0.0
    lo, hi = min(scores), max(scores)
    if hi - lo < _EPS:
        return 0.0
    n_bins = min(_ENTROPY_BINS, len(scores))
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for s in scores:
        idx = min(int((s - lo) / width), n_bins - 1)
        counts[idx] += 1
    total = len(scores)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return entropy
