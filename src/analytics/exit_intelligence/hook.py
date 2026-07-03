"""
src/analytics/exit_intelligence/hook.py — Live Signal Integration Hook

run_exit_intelligence_hook(): called from run_live_signal.py.
Captures daily observations from current positions + computes velocity scores.
Fail-open: any error → no modification to live signals.

Pipeline per call:
  1. Capture PortfolioExitPressureSnapshot
  2. For each held position: capture ExitPathSnapshot
  3. Run all 5 deterioration analyzers
  4. Score ExitVelocityScore per symbol
  5. Store all records (append-only)
  6. Return velocity scores dict: symbol → ExitVelocityScore
  7. Generate daily report
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def run_exit_intelligence_hook(
    portfolio_summary: Optional[Dict[str, Any]] = None,
    broker_positions: Optional[List[Dict[str, Any]]] = None,
    regime_info: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    obs_path: Path,
    path_snapshots_path: Path,
    exec_obs_path: Path,
    portfolio_pressure_path: Path,
    velocity_scores_path: Path,
    lifecycle_path: Path,
    dataset_path: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """
    Main integration hook for run_live_signal.py.

    Args:
        portfolio_summary: dict from result.portfolio_summary
        broker_positions: list of current positions from broker
        regime_info: dict with "regime", "confidence", etc.
        market_data: dict symbol → {close, entry_price, entry_date,
                                    peak_price, rs_rank, atr_pct,
                                    breadth_score, momentum, momentum_prev,
                                    volume_ratio, volatility_compression,
                                    upper_shadow_ratio, exit_signal_count,
                                    trend_persistence, day_number}

    Returns:
        dict with "velocity_scores" (symbol → ExitVelocityScore.to_dict())
        and "report" summary.
    Fail-open: returns {} on any error.
    """
    try:
        from .observation import (
            ExitObservationStore,
            ExitPathSnapshotStore,
            ExecutionExitObservationStore,
            PortfolioExitPressureStore,
            build_exit_path_snapshot,
        )
        from .deterioration import (
            ConvexityRetentionAnalyzer,
            MomentumDecayAnalyzer,
            PeakDecayVelocityAnalyzer,
            PersistenceExtensionLayer,
            StructuralFailureDetector,
        )
        from .exit_velocity import ExitVelocityScorer, ExitVelocityStore
        from .models import ExitVelocityScore, PortfolioExitPressureSnapshot
        from .dataset import ExitResearchDatasetStore
        from .daily_report import run_daily_report

        today = _today_str()

        # Stores
        path_store = ExitPathSnapshotStore(path_snapshots_path)
        velocity_store = ExitVelocityStore(velocity_scores_path)
        portfolio_store = PortfolioExitPressureStore(portfolio_pressure_path)

        # ── 1. Portfolio pressure snapshot ─────────────────────────────────────
        ps = portfolio_summary or {}
        pos_list = broker_positions or []
        concurrent = len(pos_list)
        gross_exp = float(ps.get("gross_exposure_jpy", 0.0))
        capital_total = float(ps.get("capital", 3_000_000))
        cash = float(ps.get("available_cash_jpy", capital_total))
        capital_util = _clamp(gross_exp / capital_total) if capital_total > 0 else 0.0
        cash_pressure = _clamp(1.0 - cash / capital_total) if capital_total > 0 else 0.0

        # Portfolio heat: avg distance from peak across positions
        heat_vals: List[float] = []
        if market_data:
            for sym, md in market_data.items():
                pk = float(md.get("peak_price", md.get("close", 1.0)))
                cl = float(md.get("close", pk))
                if pk > 0:
                    heat_vals.append((pk - cl) / pk)
        portfolio_heat = sum(heat_vals) / len(heat_vals) if heat_vals else 0.0
        worst_det = max(heat_vals) if heat_vals else 0.0

        replacement_pressure = _clamp(float(ps.get("replacement_pressure", 0.0)))

        pressure_snap = PortfolioExitPressureSnapshot(
            pressure_id=PortfolioExitPressureSnapshot.make_pressure_id(today),
            snapshot_date=today,
            concurrent_positions=concurrent,
            gross_exposure_jpy=gross_exp,
            sector_concentration_max=float(ps.get("sector_concentration_max", 0.0)),
            signal_queue_pressure=float(ps.get("signal_queue_pressure", 0.0)),
            replacement_pressure=replacement_pressure,
            portfolio_heat=portfolio_heat,
            cash_pressure=cash_pressure,
            capital_utilization=capital_util,
            avg_holding_days=float(ps.get("avg_holding_days", 0.0)),
            worst_position_deterioration=worst_det,
        )
        portfolio_store.append(pressure_snap)

        # ── 2 & 3. Per-symbol path snapshots + deterioration ───────────────────
        scorers = {
            "momentum": MomentumDecayAnalyzer(),
            "structural": StructuralFailureDetector(),
            "peak_decay": PeakDecayVelocityAnalyzer(),
            "convexity": ConvexityRetentionAnalyzer(),
            "persistence": PersistenceExtensionLayer(),
        }
        evs_scorer = ExitVelocityScorer()
        velocity_results: Dict[str, Any] = {}

        if market_data:
            for sym, md in market_data.items():
                try:
                    entry_date = str(md.get("entry_date", today))
                    day_number = int(md.get("day_number", 1))
                    snap = build_exit_path_snapshot(
                        symbol=sym,
                        entry_date=entry_date,
                        snapshot_date=today,
                        day_number=day_number,
                        close_price=float(md.get("close", 0.0)),
                        entry_price=float(md.get("entry_price", 0.0)),
                        peak_price_so_far=float(md.get("peak_price", md.get("close", 0.0))),
                        rs_rank=float(md.get("rs_rank", 50.0)),
                        atr_pct=float(md.get("atr_pct", 0.02)),
                        breadth_score=float(md.get("breadth_score",
                                                    float((regime_info or {}).get("breadth", 0.5)))),
                        regime=str((regime_info or {}).get("regime", "unknown")),
                        trend_persistence=float(md.get("trend_persistence", 0.5)),
                        momentum=float(md.get("momentum", 0.0)),
                        momentum_prev=float(md.get("momentum_prev", 0.0)),
                        volume_ratio=float(md.get("volume_ratio", 1.0)),
                        volatility_compression=float(md.get("volatility_compression", 0.5)),
                        upper_shadow_ratio=float(md.get("upper_shadow_ratio", 0.3)),
                        exit_signal_count=int(md.get("exit_signal_count", 0)),
                    )
                    path_store.append(snap)

                    # Load all snapshots for this symbol to compute deterioration
                    all_snaps = path_store.load_open_position(sym, entry_date)
                    if len(all_snaps) >= 1:
                        mom_metrics = scorers["momentum"].analyze(all_snaps, today)
                        struct_metrics = scorers["structural"].analyze(all_snaps, today)
                        peak_metrics = scorers["peak_decay"].analyze(all_snaps, today)
                        conv_metrics = scorers["convexity"].analyze(all_snaps, today)
                        pers_metrics = scorers["persistence"].analyze(all_snaps, today)

                        evs = evs_scorer.score(
                            symbol=sym,
                            scored_date=today,
                            latest_snapshot=snap,
                            momentum_decay=mom_metrics,
                            structural_failure=struct_metrics,
                            peak_decay=peak_metrics,
                            convexity_retention=conv_metrics,
                            persistence=pers_metrics,
                            portfolio_pressure=pressure_snap,
                        )
                        velocity_store.append(evs)
                        velocity_results[sym] = evs.to_dict()
                        logger.info(
                            "[EXIT_INTEL] %s score=%.2f band=%s",
                            sym, evs.total_score, evs.action_band,
                        )
                except Exception as sym_err:
                    logger.warning("[EXIT_INTEL] symbol=%s error: %s", sym, sym_err)

        # ── 4. Daily report ─────────────────────────────────────────────────────
        try:
            report = run_daily_report(
                obs_path=obs_path,
                path_snapshots_path=path_snapshots_path,
                velocity_path=velocity_scores_path,
                portfolio_path=portfolio_pressure_path,
                dataset_path=dataset_path,
                reports_dir=reports_dir,
                report_date=today,
            )
            alerts = report.get("section_6_alerts", {})
        except Exception as rpt_err:
            logger.warning("[EXIT_INTEL] daily report error: %s", rpt_err)
            alerts = {}

        return {
            "velocity_scores": velocity_results,
            "portfolio_heat": portfolio_heat,
            "high_pressure_count": sum(
                1 for vs in velocity_results.values() if vs.get("total_score", 0) >= 5.0
            ),
            "alerts": alerts,
        }

    except Exception as hook_err:
        logger.warning("[EXIT_INTEL] hook failed: %s — continuing", hook_err)
        return {}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
