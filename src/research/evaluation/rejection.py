"""Auto-rejection criteria for experiment results.

Fail closed: any violated criterion → reject.
Thresholds match CLAUDE.md VALIDATION section.
"""
from __future__ import annotations

from typing import List, Tuple

from src.research.contracts.result import ResultContract

# ── Rejection thresholds ───────────────────────────────────────────────────────
MAX_SHARPE: float = 3.0          # above → suspect lookahead (CLAUDE.md: sharpe_max=3.0)
MIN_TRADES: int = 5              # below → statistically meaningless (CLAUDE.md: trade_min=5)
MAX_DRAWDOWN_FLOOR: float = -0.5 # below → strategy non-viable (CLAUDE.md: dd_max=0.5)
MAX_TURNOVER: float = 5.0        # above → excessive churn
MAX_SHARPE_DECAY: float = 2.0    # IS - OOS Sharpe excess


def check_rejection(result: ResultContract) -> Tuple[bool, List[str]]:
    """Returns (should_reject, [reasons]).

    No silent fallback: every violated criterion is listed.
    """
    reasons: List[str] = []

    if result.sharpe > MAX_SHARPE:
        reasons.append(
            f"sharpe={result.sharpe:.3f} > {MAX_SHARPE} — suspect lookahead leak"
        )
    if result.trade_count < MIN_TRADES:
        reasons.append(
            f"trade_count={result.trade_count} < {MIN_TRADES} — statistically meaningless"
        )
    if result.max_drawdown < MAX_DRAWDOWN_FLOOR:
        reasons.append(
            f"max_drawdown={result.max_drawdown:.3f} < {MAX_DRAWDOWN_FLOOR} — non-viable"
        )
    if result.turnover > MAX_TURNOVER:
        reasons.append(
            f"turnover={result.turnover:.3f} > {MAX_TURNOVER} — excessive churn"
        )
    if result.stability_metrics.sharpe_decay > MAX_SHARPE_DECAY:
        reasons.append(
            f"sharpe_decay={result.stability_metrics.sharpe_decay:.3f} > {MAX_SHARPE_DECAY}"
        )

    return (len(reasons) > 0, reasons)


def check_regime_fragility(
    result: ResultContract,
    min_regime_sharpe: float = -1.0,
) -> Tuple[bool, List[str]]:
    """Reject if any regime has Sharpe below min_regime_sharpe."""
    reasons: List[str] = []
    for regime, metrics in result.regime_metrics.by_regime.items():
        s = metrics.get("sharpe", 0.0)
        if s < min_regime_sharpe:
            reasons.append(f"regime '{regime}' sharpe={s:.3f} < {min_regime_sharpe}")
    return (len(reasons) > 0, reasons)
