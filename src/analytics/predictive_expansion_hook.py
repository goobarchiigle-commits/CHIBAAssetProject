"""
predictive_expansion_hook.py — Live signal integration hook

Loads OHLCV + RSR history, computes PredictiveExpansionScores,
writes JSONL + markdown report. Fail-open: never affects live execution.

Called from run_live_signal.py after bridge.run() completes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _today_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d")


def _load_ohlcv(
    symbols: List[str],
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
) -> Dict[str, Any]:
    """Load OHLCV parquet files. Returns {symbol: DataFrame}."""
    import pandas as pd

    result: Dict[str, Any] = {}
    for sym in symbols:
        candidates = []
        if default_data_version:
            candidates.append(backtest_dataset_dir / default_data_version / f"{sym}.parquet")
        candidates.append(cache_dir / "ohlcv" / f"{sym}.parquet")

        for p in candidates:
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                # Normalize Close column
                if "Close" not in df.columns and "Adj Close" in df.columns:
                    df = df.copy()
                    df["Close"] = df["Adj Close"]
                # Ensure High/Low
                for col in ("High", "Low", "Close"):
                    if col not in df.columns:
                        break
                else:
                    if "Volume" not in df.columns:
                        df = df.copy()
                        df["Volume"] = 0
                    result[sym] = df
                    break
            except Exception as e:
                logger.debug("OHLCV load failed %s@%s: %s", sym, p, e)

    return result


def _build_rsr_df(ohlcv: Dict[str, Any]):
    """Compute RSR DataFrame from OHLCV dict."""
    import pandas as pd
    from src.backtest.rsr import calc_universe_rsr

    prices: Dict[str, Any] = {}
    for sym, df in ohlcv.items():
        close = df.get("Close", df.get("Adj Close"))
        if close is not None:
            prices[sym] = df["Close"].dropna()

    if len(prices) < 2:
        return pd.DataFrame()

    return calc_universe_rsr(prices)


def _append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_report(scores, reports_dir: Path) -> None:
    """Write deterministic markdown report for today's predictive scores."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = _today_str()
    out_path = reports_dir / f"predictive_expansion_{today}.md"

    top10 = scores.top_k(10)
    top_accel = sorted(scores.rsr_acceleration_score.items(), key=lambda x: (-x[1], x[0]))[:5]
    top_compress = sorted(scores.compression_breakout_score.items(), key=lambda x: (-x[1], x[0]))[:5]
    top_volume = sorted(scores.volume_regime_score.items(), key=lambda x: (-x[1], x[0]))[:5]
    igniting = sorted(s for s, v in scores.sector_ignition_active.items() if v)
    followers = sorted(s for s, v in scores.is_follower.items() if v)

    lines: List[str] = [
        f"# Predictive Expansion Report — {today}",
        "",
        "## Top Predictive Alpha Candidates",
        "| Rank | Symbol | PredAlpha | RSRAccel | Compress | SectIgn | Volume | Vel5d | Leader | Follower |",
        "|------|--------|-----------|----------|----------|---------|--------|-------|--------|----------|",
    ]
    for rank, (sym, score) in enumerate(top10, 1):
        lines.append(
            f"| {rank} | {sym} | {score:.1f}"
            f" | {scores.rsr_acceleration_score.get(sym, 0):.1f}"
            f" | {scores.compression_breakout_score.get(sym, 0):.1f}"
            f" | {scores.sector_ignition_score.get(sym, 0):.1f}"
            f" | {scores.volume_regime_score.get(sym, 0):.1f}"
            f" | {scores.rsr_velocity_5d.get(sym, 0):+.1f}"
            f" | {'✓' if scores.is_leader.get(sym) else ''}"
            f" | {'✓' if scores.is_follower.get(sym) else ''} |"
        )

    lines += [
        "",
        "## RSR Acceleration Leaders",
        "| Symbol | Accel_Score | Vel_5d | Vel_10d | Accel_Raw |",
        "|--------|-------------|--------|---------|-----------|",
    ]
    for sym, s in top_accel:
        lines.append(
            f"| {sym} | {s:.1f}"
            f" | {scores.rsr_velocity_5d.get(sym, 0):+.1f}"
            f" | {scores.rsr_velocity_10d.get(sym, 0):+.1f}"
            f" | {scores.rsr_acceleration.get(sym, 0):+.1f} |"
        )

    lines += [
        "",
        "## Compression Breakout Watchlist",
        "| Symbol | CB_Score | CompressRatio | Expansion |",
        "|--------|----------|---------------|-----------|",
    ]
    for sym, s in top_compress:
        lines.append(
            f"| {sym} | {s:.1f}"
            f" | {scores.compression_ratio.get(sym, 1):.3f}"
            f" | {scores.breakout_expansion.get(sym, 1):.2f}× |"
        )

    lines += [
        "",
        "## Highest Volume Regime Shift",
        "| Symbol | Vol_Score | Vol_5d | Vol_20d |",
        "|--------|-----------|--------|---------|",
    ]
    for sym, s in top_volume:
        lines.append(
            f"| {sym} | {s:.1f}"
            f" | {scores.volume_ratio_5d.get(sym, 1):.2f}×"
            f" | {scores.volume_ratio_20d.get(sym, 1):.2f}× |"
        )

    if igniting:
        lines += [
            "",
            f"## Sector Ignition Active ({len(igniting)} symbols)",
            ", ".join(igniting),
        ]

    if followers:
        lines += [
            "",
            f"## Leader→Follower Propagation ({len(followers)} candidates)",
            ", ".join(followers),
        ]

    lines += ["", f"*{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')} JST*"]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[PREDICTIVE] report → %s", out_path)


def _print_predictive_summary(scores) -> None:
    """Print [PREDICTIVE_ALPHA] block to stdout."""
    top5 = scores.top_k(5)
    igniting = [s for s, v in scores.sector_ignition_active.items() if v]
    followers = [s for s, v in scores.is_follower.items() if v]

    print("\n  ── [PREDICTIVE_ALPHA] Expansion Detection ──────────────────────────")
    if not top5:
        print("  (データ不足)")
        return

    print(f"  {'Symbol':<10} {'Score':>6} {'Vel5d':>6} {'Compress':>8} {'SectIgn':>8} {'Flag'}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for sym, score in top5:
        vel5 = scores.rsr_velocity_5d.get(sym, 0.0)
        comp = scores.compression_ratio.get(sym, 1.0)
        sig  = "IGNITING" if scores.sector_ignition_active.get(sym) else ""
        sig  = "FOLLOWER" if scores.is_follower.get(sym) else sig
        sig  = "LEADER"   if scores.is_leader.get(sym) else sig
        print(
            f"  {sym:<10} {score:>6.1f} {vel5:>+6.1f} {comp:>8.3f}"
            f" {scores.sector_ignition_score.get(sym, 0):>8.1f} {sig}"
        )
    if igniting:
        print(f"  Sector ignition: {', '.join(sorted(igniting)[:5])}")
    if followers:
        print(f"  Followers:       {', '.join(sorted(followers)[:5])}")
    print()


def run_predictive_expansion_hook(
    universe: Dict[str, str],
    *,
    backtest_dataset_dir: Path,
    default_data_version: str,
    cache_dir: Path,
    scores_path: Path,
    reports_dir: Path,
    run_id: str = "",
    print_summary: bool = True,
) -> Dict[str, Any]:
    """
    Main integration hook for run_live_signal.py. Fail-open.

    Args:
        universe:             {symbol: sector} — full RSR context universe
        backtest_dataset_dir: path to parquet dataset files
        default_data_version: version subdirectory (e.g. "v1")
        cache_dir:            fallback OHLCV cache directory
        scores_path:          JSONL path for score persistence
        reports_dir:          directory for markdown reports
        run_id:               live run identifier for tracing
        print_summary:        print [PREDICTIVE_ALPHA] block to stdout

    Returns:
        {
          "scores":                   PredictiveExpansionScores.to_dict(),
          "top_candidates":           [(symbol, score), ...],
          "sector_igniting_symbols":  [symbol, ...],
          "followers":                [symbol, ...],
        }
        Returns {} on any error.
    """
    try:
        from src.analytics.predictive_expansion import compute_predictive_expansion_scores

        symbols = list(universe.keys())
        logger.info("[PREDICTIVE] loading OHLCV for %d symbols", len(symbols))

        ohlcv = _load_ohlcv(
            symbols=symbols,
            backtest_dataset_dir=backtest_dataset_dir,
            default_data_version=default_data_version,
            cache_dir=cache_dir,
        )

        if len(ohlcv) < 3:
            logger.warning("[PREDICTIVE] insufficient OHLCV (%d loaded) — skip", len(ohlcv))
            return {}

        rsr_df = _build_rsr_df(ohlcv)
        if rsr_df.empty:
            logger.warning("[PREDICTIVE] RSR DataFrame empty — skip")
            return {}

        scores = compute_predictive_expansion_scores(
            rsr_df=rsr_df,
            ohlcv=ohlcv,
            sector_map=universe,
        )

        # Persist
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(JST).isoformat(),
            **scores.to_dict(),
        }
        _append_jsonl(record, scores_path)
        _write_report(scores, reports_dir)

        top = scores.top_k(10)
        igniting = sorted(s for s, v in scores.sector_ignition_active.items() if v)
        followers = sorted(s for s, v in scores.is_follower.items() if v)

        if print_summary:
            _print_predictive_summary(scores)

        logger.info(
            "[PREDICTIVE] done: top3=%s igniting=%d followers=%d",
            [s for s, _ in top[:3]], len(igniting), len(followers),
        )

        return {
            "scores":                  scores.to_dict(),
            "top_candidates":          top,
            "sector_igniting_symbols": igniting,
            "followers":               followers,
        }

    except Exception as exc:
        logger.warning("[PREDICTIVE] hook failed (%s) — continuing", exc)
        return {}


def load_latest_predictive_scores(scores_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load the most recent predictive scores from JSONL (last line).

    Used by predictive entry overlay to read prior-day scores.
    Returns None if file absent or unreadable.
    """
    if not scores_path.exists():
        return None
    try:
        lines = scores_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
    except Exception as e:
        logger.debug("load_latest_predictive_scores: %s", e)
    return None
