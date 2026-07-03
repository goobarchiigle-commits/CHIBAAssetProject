"""BatchSummary — deterministic result aggregation for a completed batch.

Aggregates ResultContracts into ranking table, rejection report, regime summaries.
All outputs are deterministic: same results → same summary, always.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.research.contracts.result import ResultContract
from src.research.evaluation.ranking import rank_results, RankedResult
from src.research.evaluation.rejection import check_rejection


@dataclass
class BatchSummary:
    batch_id: str
    total_experiments: int
    ranking: List[RankedResult]          # deterministically ordered
    rejection_report: dict
    regime_summary: dict

    # ── public views ───────────────────────────────────────────────────────────

    def ranking_table(self) -> List[dict]:
        """Deterministic ranking table, one row per result."""
        rows = []
        rank = 1
        for r in self.ranking:
            rows.append({
                "rank": rank if not r.rejected else None,
                "experiment_id": r.result.experiment_id,
                "fold_id": r.result.fold_id,
                "score": round(r.score, 6) if r.score != float("-inf") else None,
                "sharpe": round(r.result.sharpe, 4),
                "sortino": round(r.result.sortino, 4),
                "max_drawdown": round(r.result.max_drawdown, 4),
                "turnover": round(r.result.turnover, 4),
                "hit_rate": round(r.result.hit_rate, 4),
                "oos_robustness": round(r.result.stability_metrics.oos_robustness, 4),
                "sharpe_decay": round(r.result.stability_metrics.sharpe_decay, 4),
                "trade_count": r.result.trade_count,
                "rejected": r.rejected,
                "rejection_reasons": r.rejection_reasons,
            })
            if not r.rejected:
                rank += 1
        return rows

    def top_n(self, n: int = 10) -> List[RankedResult]:
        return [r for r in self.ranking if not r.rejected][:n]

    def to_dict(self) -> dict:
        non_rejected = [r for r in self.ranking if not r.rejected]
        return {
            "batch_id": self.batch_id,
            "total_experiments": self.total_experiments,
            "total_ranked": len(non_rejected),
            "total_rejected": self.total_experiments - len(non_rejected),
            "rejection_report": self.rejection_report,
            "regime_summary": self.regime_summary,
            "top_3": self.ranking_table()[:3],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    # ── factory ────────────────────────────────────────────────────────────────

    @staticmethod
    def from_results(batch_id: str, results: List[ResultContract]) -> BatchSummary:
        """Build deterministic BatchSummary from a list of ResultContracts."""
        ranking = rank_results(results)
        rejection_report = _build_rejection_report(results)
        regime_summary = _build_regime_summary(results)
        return BatchSummary(
            batch_id=batch_id,
            total_experiments=len(results),
            ranking=ranking,
            rejection_report=rejection_report,
            regime_summary=regime_summary,
        )


# ── Private aggregation helpers ────────────────────────────────────────────────

def _build_rejection_report(results: List[ResultContract]) -> dict:
    """Count and categorize rejection reasons."""
    total = len(results)
    rejected_count = 0
    reason_counts: Dict[str, int] = {}

    for r in results:
        is_rejected, reasons = check_rejection(r)
        if is_rejected:
            rejected_count += 1
            for reason in reasons:
                # Extract the metric name from the reason string
                metric = reason.split("=")[0].strip() if "=" in reason else reason[:30]
                reason_counts[metric] = reason_counts.get(metric, 0) + 1

    return {
        "total": total,
        "rejected": rejected_count,
        "passed": total - rejected_count,
        "rejection_rate": round(rejected_count / max(total, 1), 4),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _build_regime_summary(results: List[ResultContract]) -> dict:
    """Aggregate per-regime metrics across all experiments."""
    regime_sharpes: Dict[str, List[float]] = {}
    regime_trade_counts: Dict[str, int] = {}

    for r in results:
        for regime, metrics in r.regime_metrics.by_regime.items():
            regime_sharpes.setdefault(regime, []).append(metrics.get("sharpe", 0.0))
            regime_trade_counts[regime] = (
                regime_trade_counts.get(regime, 0) + int(metrics.get("trade_count", 0))
            )

    summary = {}
    for regime in sorted(regime_sharpes.keys()):
        sharpes = regime_sharpes[regime]
        summary[regime] = {
            "experiment_count": len(sharpes),
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0,
            "total_trades": regime_trade_counts.get(regime, 0),
        }
    return summary
