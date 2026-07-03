"""
exposure_core.py — Economic exposure reconstruction

Distinguishes nominal positions from economic exposure.
4 correlated positions ≠ 4 independent risks.

Tracks:
- Rolling pairwise Pearson correlation (20-day window)
- Effective independent exposure count (correlation-adjusted)
- Theme clustering (sector + RSR quartile)
- Correlation collapse detection

Avoids deep factor models. All arithmetic is pure Python for
deterministic replay on Windows without numpy.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CORR_WINDOW: int = 20              # rolling days for correlation
COLLAPSE_THRESHOLD: float = 0.85   # pairwise correlation → collapse alert
COLLAPSE_STREAK_REQUIRED: int = 5  # consecutive days above threshold
MIN_OBSERVATIONS_FOR_CORR: int = 5 # min return points before correlation valid
MAX_RETURN_BUFFER: int = CORR_WINDOW + 5  # keep a small tail beyond window


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PairCorrelation:
    symbol_a: str
    symbol_b: str
    correlation: float         # Pearson [-1.0, 1.0]; None serialized as 0.0
    n_observations: int
    valid: bool                # False if insufficient data


@dataclass
class ThemeCluster:
    theme_key: str             # e.g. "sector:電気機器" or "rsr_q:1"
    symbols: list
    cluster_type: str          # "sector" | "rsr_quartile"
    overlap_count: int         # symbols sharing the same theme


@dataclass
class ExposureReport:
    nominal_positions: int
    effective_independent_n: float     # < nominal when correlated
    pair_correlations: list           # list[PairCorrelation] as dicts
    theme_clusters: list              # list[ThemeCluster] as dicts
    avg_abs_pairwise_corr: float
    collapse_alert: bool
    collapse_pairs: list              # list of [sym_a, sym_b] strings
    concentration_risk: str           # "LOW" | "MEDIUM" | "HIGH"
    schema_version: int = 1


@dataclass
class ExposureState:
    """Mutable rolling state. Persisted to disk between runs."""
    return_buffers: dict = field(default_factory=dict)   # {symbol: [float]}
    collapse_streak: dict = field(default_factory=dict)  # {pair_key: int}
    schema_version: int = 1


# ── Pure math helpers ──────────────────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson correlation between two equal-length series. Returns None if invalid."""
    n = min(len(xs), len(ys))
    if n < MIN_OBSERVATIONS_FOR_CORR:
        return None
    xs = xs[-n:]
    ys = ys[-n:]
    sx = sum(xs)
    sy = sum(ys)
    sx2 = sum(x * x for x in xs)
    sy2 = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    num = n * sxy - sx * sy
    den_sq = (n * sx2 - sx * sx) * (n * sy2 - sy * sy)
    if den_sq <= 0.0:
        return None
    den = math.sqrt(den_sq)
    r = num / den
    return round(max(-1.0, min(1.0, r)), 4)


def _pair_key(a: str, b: str) -> str:
    """Canonical pair key — order-invariant."""
    return f"{min(a, b)}|{max(a, b)}"


def _effective_n_from_corr(n_positions: int, avg_abs_corr: float) -> float:
    """
    Effective independent position count under average pairwise correlation.
    Formula: N / (1 + (N-1) * rho)   — equi-correlation approximation.
    Returns in [1.0, N].
    """
    if n_positions <= 0:
        return 0.0
    if n_positions == 1:
        return 1.0
    rho = max(0.0, min(1.0, avg_abs_corr))
    denom = 1.0 + (n_positions - 1) * rho
    eff = n_positions / denom
    return round(max(1.0, min(float(n_positions), eff)), 4)


def _concentration_risk_label(eff_n: float, nominal_n: int) -> str:
    if nominal_n == 0:
        return "LOW"
    ratio = eff_n / nominal_n
    if ratio >= 0.75:
        return "LOW"
    if ratio >= 0.50:
        return "MEDIUM"
    return "HIGH"


def _rsr_quartile(rsr_rank: float, n_universe: int = 42) -> int:
    """Map raw RSR rank (1-based, lower=better) to quartile 1–4."""
    if n_universe <= 0 or rsr_rank <= 0:
        return 4
    pct = rsr_rank / n_universe
    if pct <= 0.25:
        return 1
    if pct <= 0.50:
        return 2
    if pct <= 0.75:
        return 3
    return 4


# ── Core computation ───────────────────────────────────────────────────────────

def compute_pair_correlations(
    state: ExposureState,
    symbols: list[str],
) -> list[PairCorrelation]:
    """Compute all pairwise correlations among active symbols."""
    result: list[PairCorrelation] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            buf_a = state.return_buffers.get(a, [])
            buf_b = state.return_buffers.get(b, [])
            # align on common window
            common_n = min(len(buf_a), len(buf_b), CORR_WINDOW)
            if common_n < MIN_OBSERVATIONS_FOR_CORR:
                result.append(PairCorrelation(
                    symbol_a=a, symbol_b=b,
                    correlation=0.0, n_observations=common_n, valid=False,
                ))
                continue
            r = _pearson(buf_a[-common_n:], buf_b[-common_n:])
            result.append(PairCorrelation(
                symbol_a=a, symbol_b=b,
                correlation=r if r is not None else 0.0,
                n_observations=common_n,
                valid=(r is not None),
            ))
    return result


def compute_theme_clusters(
    symbols: list[str],
    sector_map: dict,           # {symbol: sector_str}
    rsr_rank_map: dict,         # {symbol: int rank (1-based)}
    n_universe: int = 42,
) -> list[ThemeCluster]:
    """Group symbols by sector and RSR quartile."""
    clusters: list[ThemeCluster] = []

    # Sector clusters
    sector_groups: dict[str, list[str]] = {}
    for sym in symbols:
        sec = sector_map.get(sym, "不明")
        sector_groups.setdefault(sec, []).append(sym)
    for sec, members in sector_groups.items():
        if len(members) >= 2:
            clusters.append(ThemeCluster(
                theme_key=f"sector:{sec}",
                symbols=members,
                cluster_type="sector",
                overlap_count=len(members),
            ))

    # RSR quartile clusters
    quartile_groups: dict[int, list[str]] = {}
    for sym in symbols:
        rank = rsr_rank_map.get(sym, n_universe)
        q = _rsr_quartile(rank, n_universe)
        quartile_groups.setdefault(q, []).append(sym)
    for q, members in quartile_groups.items():
        if len(members) >= 2:
            clusters.append(ThemeCluster(
                theme_key=f"rsr_q:{q}",
                symbols=members,
                cluster_type="rsr_quartile",
                overlap_count=len(members),
            ))

    return clusters


def compute_exposure_report(
    state: ExposureState,
    symbols: list[str],
    sector_map: dict,
    rsr_rank_map: dict,
    n_universe: int = 42,
) -> ExposureReport:
    """
    Full exposure report for current positions.
    Deterministic — given same state and inputs, output is identical.
    """
    nominal_n = len(symbols)
    if nominal_n == 0:
        return ExposureReport(
            nominal_positions=0,
            effective_independent_n=0.0,
            pair_correlations=[],
            theme_clusters=[],
            avg_abs_pairwise_corr=0.0,
            collapse_alert=False,
            collapse_pairs=[],
            concentration_risk="LOW",
        )

    pairs = compute_pair_correlations(state, symbols)
    valid_pairs = [p for p in pairs if p.valid]

    avg_abs_corr = (
        sum(abs(p.correlation) for p in valid_pairs) / len(valid_pairs)
        if valid_pairs else 0.0
    )
    avg_abs_corr = round(avg_abs_corr, 4)

    eff_n = _effective_n_from_corr(nominal_n, avg_abs_corr)

    # Collapse detection
    collapse_pairs: list[list[str]] = []
    for p in pairs:
        if p.valid and abs(p.correlation) >= COLLAPSE_THRESHOLD:
            key = _pair_key(p.symbol_a, p.symbol_b)
            if state.collapse_streak.get(key, 0) >= COLLAPSE_STREAK_REQUIRED:
                collapse_pairs.append([p.symbol_a, p.symbol_b])

    clusters = compute_theme_clusters(symbols, sector_map, rsr_rank_map, n_universe)
    risk = _concentration_risk_label(eff_n, nominal_n)

    return ExposureReport(
        nominal_positions=nominal_n,
        effective_independent_n=eff_n,
        pair_correlations=[asdict(p) for p in pairs],
        theme_clusters=[asdict(c) for c in clusters],
        avg_abs_pairwise_corr=avg_abs_corr,
        collapse_alert=len(collapse_pairs) > 0,
        collapse_pairs=collapse_pairs,
        concentration_risk=risk,
    )


# ── State update ───────────────────────────────────────────────────────────────

def update_exposure_state(
    state: ExposureState,
    daily_returns: dict,   # {symbol: float daily return}
) -> ExposureState:
    """
    Append today's returns to buffers and update collapse streaks.
    Returns a new ExposureState (immutable update pattern).
    """
    new_buffers: dict[str, list[float]] = {}
    for sym, buf in state.return_buffers.items():
        new_buffers[sym] = list(buf)

    for sym, ret in daily_returns.items():
        buf = list(new_buffers.get(sym, []))
        buf.append(float(ret))
        new_buffers[sym] = buf[-MAX_RETURN_BUFFER:]

    # Update collapse streaks for all current symbol pairs
    symbols = list(daily_returns.keys())
    new_streak: dict[str, int] = dict(state.collapse_streak)
    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            key = _pair_key(a, b)
            buf_a = new_buffers.get(a, [])
            buf_b = new_buffers.get(b, [])
            common_n = min(len(buf_a), len(buf_b), CORR_WINDOW)
            if common_n < MIN_OBSERVATIONS_FOR_CORR:
                new_streak[key] = 0
                continue
            r = _pearson(buf_a[-common_n:], buf_b[-common_n:])
            if r is not None and abs(r) >= COLLAPSE_THRESHOLD:
                new_streak[key] = new_streak.get(key, 0) + 1
            else:
                new_streak[key] = 0

    return ExposureState(
        return_buffers=new_buffers,
        collapse_streak=new_streak,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def load_exposure_state(path: Path) -> ExposureState:
    if not path.exists():
        return ExposureState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExposureState(
            return_buffers=data.get("return_buffers", {}),
            collapse_streak=data.get("collapse_streak", {}),
        )
    except Exception as exc:
        logger.warning("exposure_state load failed (%s) — default", exc)
        return ExposureState()


def save_exposure_state(state: ExposureState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
