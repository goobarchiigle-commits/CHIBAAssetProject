"""
src/analytics/exit_intelligence/daily_report.py — Automated Daily Report

Generates deterministic daily markdown + JSON report from exit intelligence data.
Runs as part of morning routine (called from run_live_signal.py).
Also runnable standalone.

Output:
  runtime/reports/exit_intelligence/exit_intelligence_daily_YYYYMMDD.md
  runtime/reports/exit_intelligence/exit_intelligence_daily_YYYYMMDD.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ExitIntelligenceDailyReporter:
    """
    Build daily exit intelligence report from all stores.
    Deterministic: same inputs → same output.
    """

    def __init__(
        self,
        obs_store,          # ExitObservationStore
        path_store,         # ExitPathSnapshotStore
        velocity_store,     # ExitVelocityStore
        portfolio_store,    # PortfolioExitPressureStore
        dataset_store,      # ExitResearchDatasetStore
        reports_dir: Path,
    ) -> None:
        self._obs = obs_store
        self._paths = path_store
        self._velocity = velocity_store
        self._portfolio = portfolio_store
        self._dataset = dataset_store
        self._reports_dir = reports_dir

    def generate(self, report_date: Optional[str] = None) -> Dict[str, Any]:
        date = report_date or _today_str()
        report = self._build_report(date)

        # Write markdown
        md_path = self._reports_dir / f"exit_intelligence_daily_{date}.md"
        json_path = self._reports_dir / f"exit_intelligence_daily_{date}.json"
        _atomic_write(md_path, self._render_markdown(report))
        _atomic_write(json_path, json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
        logger.info("[EXIT_REPORT] written: %s", md_path)
        return report

    def _build_report(self, date: str) -> Dict[str, Any]:
        observations = self._obs.load_all()
        velocity_scores = self._velocity.load_all()
        portfolio_latest = self._portfolio.load_latest()
        dataset_stats = self._dataset.snapshot_stats()

        # Observations today
        today_obs = [o for o in observations if o.exit_date == date]
        # Active velocity scores (any date)
        active_vel = {}
        for vs in velocity_scores:
            if vs.symbol not in active_vel or vs.scored_date > active_vel[vs.symbol].scored_date:
                active_vel[vs.symbol] = vs

        # Action band distribution
        band_counts: Dict[str, int] = {
            "hold": 0, "tighten_risk": 0, "staged_reduction": 0, "aggressive_exit": 0
        }
        for vs in active_vel.values():
            band_counts[vs.action_band] = band_counts.get(vs.action_band, 0) + 1

        # Recent exits summary (last 5)
        recent_exits = sorted(observations, key=lambda o: o.exit_date, reverse=True)[:5]

        # Portfolio pressure
        port_summary: Dict[str, Any] = {}
        if portfolio_latest:
            port_summary = {
                "snapshot_date": portfolio_latest.snapshot_date,
                "concurrent_positions": portfolio_latest.concurrent_positions,
                "gross_exposure_jpy": portfolio_latest.gross_exposure_jpy,
                "portfolio_heat": portfolio_latest.portfolio_heat,
                "cash_pressure": portfolio_latest.cash_pressure,
                "replacement_pressure": portfolio_latest.replacement_pressure,
            }

        # Dataset integrity
        integrity_ok = dataset_stats.get("integrity_ok", True)

        # High-pressure alerts
        high_pressure = [
            {"symbol": vs.symbol, "score": vs.total_score, "band": vs.action_band}
            for vs in active_vel.values()
            if vs.total_score >= 5.0
        ]
        high_pressure.sort(key=lambda x: x["score"], reverse=True)

        return {
            "report_date": date,
            "generated_at": _now_jst(),
            "section_1_portfolio_pressure": port_summary,
            "section_2_active_velocity_scores": {
                "count": len(active_vel),
                "band_distribution": band_counts,
                "high_pressure_positions": high_pressure,
            },
            "section_3_today_exits": {
                "count": len(today_obs),
                "exits": [
                    {
                        "symbol": o.symbol,
                        "return": round(o.realized_return, 4),
                        "holding_days": o.holding_days,
                        "exit_reason": o.exit_reason,
                        "velocity_score": round(o.exit_velocity_score, 2),
                    }
                    for o in today_obs
                ],
            },
            "section_4_recent_exits": [
                {
                    "symbol": o.symbol,
                    "exit_date": o.exit_date,
                    "return": round(o.realized_return, 4),
                    "velocity_score_at_exit": round(o.exit_velocity_score, 2),
                }
                for o in recent_exits
            ],
            "section_5_research_dataset": dataset_stats,
            "section_6_alerts": {
                "high_pressure_count": len(high_pressure),
                "integrity_ok": integrity_ok,
                "dataset_total_records": dataset_stats.get("total_records", 0),
            },
        }

    def _render_markdown(self, r: Dict[str, Any]) -> str:
        date = r["report_date"]
        port = r["section_1_portfolio_pressure"]
        vel = r["section_2_active_velocity_scores"]
        exits = r["section_3_today_exits"]
        recent = r["section_4_recent_exits"]
        ds = r["section_5_research_dataset"]
        alerts = r["section_6_alerts"]

        lines = [
            f"# Exit Intelligence Daily Report — {date}",
            f"**Generated:** {r['generated_at']}",
            "",
            "## 1. Portfolio Exit Pressure",
        ]
        if port:
            lines += [
                f"- Concurrent positions: {port.get('concurrent_positions', 0)}",
                f"- Portfolio heat: {port.get('portfolio_heat', 0.0):.2%}",
                f"- Cash pressure: {port.get('cash_pressure', 0.0):.2%}",
                f"- Replacement pressure: {port.get('replacement_pressure', 0.0):.2%}",
                f"- Gross exposure: ¥{port.get('gross_exposure_jpy', 0):,.0f}",
            ]
        else:
            lines.append("*No portfolio pressure snapshot available.*")

        lines += [
            "",
            "## 2. Active Exit Velocity Scores",
            f"Monitoring **{vel['count']}** positions.",
            "",
            "| Band | Count |",
            "|---|---|",
        ]
        for band, cnt in vel["band_distribution"].items():
            lines.append(f"| {band} | {cnt} |")

        if vel["high_pressure_positions"]:
            lines += ["", "### High-Pressure Positions (score ≥ 5.0)", ""]
            lines += ["| Symbol | Score | Band |", "|---|---|---|"]
            for pos in vel["high_pressure_positions"]:
                lines.append(f"| {pos['symbol']} | {pos['score']:.2f} | **{pos['band']}** |")

        lines += [
            "",
            f"## 3. Today's Exits ({date})",
        ]
        if exits["count"] == 0:
            lines.append("*No exits today.*")
        else:
            lines += ["| Symbol | Return | Days | Reason | Velocity |", "|---|---|---|---|---|"]
            for e in exits["exits"]:
                lines.append(
                    f"| {e['symbol']} | {e['return']:+.2%} | {e['holding_days']} | "
                    f"{e['exit_reason']} | {e['velocity_score']:.1f} |"
                )

        lines += ["", "## 4. Recent Exit History", ""]
        if recent:
            lines += ["| Symbol | Date | Return | Velocity at Exit |", "|---|---|---|---|"]
            for e in recent:
                lines.append(
                    f"| {e['symbol']} | {e['exit_date']} | {e['return']:+.2%} | "
                    f"{e['velocity_score_at_exit']:.1f} |"
                )
        else:
            lines.append("*No exit history yet.*")

        lines += [
            "",
            "## 5. Research Dataset",
            f"- Total records: {ds.get('total_records', 0)}",
            f"- Mean realized return: {ds.get('mean_realized_return', 0.0):+.2%}",
            f"- Positive return rate: {ds.get('positive_return_rate', 0.0):.1%}",
            f"- Premature exit rate: {ds.get('premature_exit_rate', 0.0):.1%}",
            f"- Integrity: {'✅ OK' if ds.get('integrity_ok', True) else '❌ CORRUPTED'}",
            "",
            "## 6. Alerts",
            f"- High-pressure positions: {alerts['high_pressure_count']}",
            f"- Dataset integrity: {'✅' if alerts['integrity_ok'] else '❌ CHECK IMMEDIATELY'}",
        ]

        return "\n".join(lines) + "\n"


def run_daily_report(
    obs_path: Path,
    path_snapshots_path: Path,
    velocity_path: Path,
    portfolio_path: Path,
    dataset_path: Path,
    reports_dir: Path,
    report_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Standalone runner. Used from run_live_signal.py and Task Scheduler."""
    from .observation import ExitObservationStore, ExitPathSnapshotStore, PortfolioExitPressureStore
    from .exit_velocity import ExitVelocityStore
    from .dataset import ExitResearchDatasetStore

    reporter = ExitIntelligenceDailyReporter(
        obs_store=ExitObservationStore(obs_path),
        path_store=ExitPathSnapshotStore(path_snapshots_path),
        velocity_store=ExitVelocityStore(velocity_path),
        portfolio_store=PortfolioExitPressureStore(portfolio_path),
        dataset_store=ExitResearchDatasetStore(dataset_path),
        reports_dir=reports_dir,
    )
    return reporter.generate(report_date)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import os as _os
    _base = Path(_os.environ.get("AI_TRADING_HOME", "C:/ai-trading"))
    _rt = _base / "runtime"

    result = run_daily_report(
        obs_path=_rt / "analytics" / "exit_intelligence" / "exit_observations.jsonl",
        path_snapshots_path=_rt / "analytics" / "exit_intelligence" / "exit_path_snapshots.jsonl",
        velocity_path=_rt / "analytics" / "exit_intelligence" / "exit_velocity_scores.jsonl",
        portfolio_path=_rt / "analytics" / "exit_intelligence" / "portfolio_exit_pressure.jsonl",
        dataset_path=_rt / "analytics" / "exit_intelligence" / "research_dataset.jsonl",
        reports_dir=_rt / "reports" / "exit_intelligence",
    )
    print(json.dumps(result["section_6_alerts"], ensure_ascii=False, indent=2))
