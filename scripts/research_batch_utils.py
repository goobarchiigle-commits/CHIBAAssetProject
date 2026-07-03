from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.pop("DATA_VERSION", None)

import src.backtest.composite_alpha_bt as _bt
from src.backtest.rsr import calc_composite_alpha_matrix, calc_universe_rsr
from src.backtest.universe_builder import download_universe
from src.config_loader import load_strategy_config
from src.paths import ALLOW_YFINANCE_NETWORK
from src.utils.memory import collect_and_log_process_memory, get_process_rss_mb


FULL_START = "2018-01-01"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
RESULTS_DIR = REPO_ROOT / "results"

CURRENT_VOL_THRESHOLD = 1.15
CURRENT_VOLUME_MULTIPLIER = 1.3
CURRENT_TRAIL_ATR_MULT = 3.0
CURRENT_RSR_EXIT_THRESHOLD = 75.0
CURRENT_MAX_HOLD_DAYS = 60


def build_cfg(*, max_hold_days: int | None = None):
    cfg = load_strategy_config()
    risk_cfg = cfg.risk
    if max_hold_days is not None:
        risk_cfg = replace(risk_cfg, max_hold_days=int(max_hold_days))
    return replace(
        cfg,
        fujiko=replace(
            cfg.fujiko,
            mom_period=21,
            turtle_exit=55,
            min_rsr=75.0,
        ),
        portfolio=replace(
            cfg.portfolio,
            capital=3_000_000,
            max_positions=3,
        ),
        risk=replace(
            risk_cfg,
            min_hold_days=3,
        ),
    )


def load_oos_dataset() -> tuple:
    rsr_syms = _bt._load_rsr_universe()
    trade_syms = rsr_syms
    universe_raw = download_universe({**rsr_syms, **trade_syms}, start=FULL_START, end=OOS_END, verbose=False)
    topix_close = _bt._download_topix(FULL_START, OOS_END)

    rsr_df = calc_universe_rsr({
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms
        if sym in universe_raw
    })
    alpha_df = calc_composite_alpha_matrix({
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms
        if sym in universe_raw
    }, window=_bt.COMP_ALPHA_WINDOW).shift(1)
    regime_df = _bt._calc_regime(topix_close)
    tech_matrices = _bt._precompute_tech_matrices(universe_raw, list(trade_syms))
    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close, tech_matrices


def safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(val):
        return default
    return val


def _profit_factor(wins: pd.DataFrame, losses: pd.DataFrame) -> float:
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize_result(result: dict, *, scenario_name: str | None = None) -> dict:
    trades = pd.DataFrame(result.get("_trades", []))
    if trades.empty:
        exit_pct = {"RSR_EXIT": 0.0, "ATR_TRAIL": 0.0, "TRAIL_EXIT": 0.0, "TIME_STOP": 0.0}
        return {
            "scenario_name": scenario_name or result.get("scenario", ""),
            "trade_count": int(result.get("n_trades", 0)),
            "hit_rate": safe_float(result.get("win_rate", 0.0)),
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "sharpe": safe_float(result.get("sharpe", 0.0)),
            "avg_hold_days": safe_float(result.get("avg_hold_days", 0.0)),
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "count_win": 0,
            "count_loss": 0,
            "max_drawdown": safe_float(result.get("max_dd", 0.0)),
            "exit_rsr_pct": 0.0,
            "exit_atr_pct": 0.0,
            "exit_turtle_pct": 0.0,
            "exit_reason_distribution": json.dumps(exit_pct, ensure_ascii=False),
        }

    trades = trades.copy()
    trades["ret_pct"] = (trades["exit"] / trades["entry"] - 1.0) * 100.0
    trades["hold_days"] = trades["exit_idx"] - trades["entry_idx"]
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    hit_rate = (len(wins) / max(1, len(trades))) * 100.0
    avg_win = float(wins["ret_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["ret_pct"].mean()) if not losses.empty else 0.0
    expectancy = (hit_rate / 100.0) * avg_win + (1.0 - hit_rate / 100.0) * avg_loss
    profit_factor = _profit_factor(wins, losses)

    reason_counts = result.get("exit_reason_counts", {}) or {}
    total = max(1, len(trades))
    atr_count = int(reason_counts.get("ATR_TRAIL", 0)) + int(reason_counts.get("TRAIL_EXIT", 0))
    turtle_count = int(reason_counts.get("STRATEGY_EXIT", 0))
    dist = {
        key: round(count / total * 100.0, 2)
        for key, count in sorted(reason_counts.items())
    }
    return {
        "scenario_name": scenario_name or result.get("scenario", ""),
        "trade_count": int(len(trades)),
        "hit_rate": round(hit_rate, 2),
        "expectancy": round(expectancy, 3),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else float("inf"),
        "sharpe": safe_float(result.get("sharpe", 0.0)),
        "avg_hold_days": safe_float(result.get("avg_hold_days", 0.0)),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "count_win": int(len(wins)),
        "count_loss": int(len(losses)),
        "max_drawdown": safe_float(result.get("max_dd", 0.0)),
        "exit_rsr_pct": round(reason_counts.get("RSR_EXIT", 0) / total * 100.0, 2),
        "exit_atr_pct": round(atr_count / total * 100.0, 2),
        "exit_turtle_pct": round(turtle_count / total * 100.0, 2),
        "exit_reason_distribution": json.dumps(dist, ensure_ascii=False),
    }


def run_scenario_oos(
    dataset: tuple,
    cfg,
    *,
    scenario_name: str,
    enable_filters: bool = True,
    enable_volatility_filter: bool = True,
    enable_atr_filter: bool = True,
    enable_volume_filter: bool = True,
    enable_market_filter: bool = True,
    rsr_exit_threshold: float | None = None,
    trail_atr_mult: float = CURRENT_TRAIL_ATR_MULT,
) -> dict:
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close, tech_matrices = dataset
    return _bt.run_scenario(
        scenario="STEP5",
        universe_raw=universe_raw,
        rsr_df=rsr_df,
        alpha_df=alpha_df,
        regime_df=regime_df,
        trade_syms=trade_syms,
        rsr_syms=rsr_syms,
        cfg=cfg,
        start=OOS_START,
        end=OOS_END,
        capital=float(cfg.portfolio.capital),
        verbose=False,
        min_hold=cfg.risk.min_hold_days,
        rsr_exit_threshold=rsr_exit_threshold,
        enable_filters=enable_filters,
        volatility_threshold=CURRENT_VOL_THRESHOLD,
        volume_multiplier=CURRENT_VOLUME_MULTIPLIER,
        enable_volatility_filter=enable_volatility_filter,
        enable_atr_filter=enable_atr_filter,
        enable_volume_filter=enable_volume_filter,
        enable_market_filter=enable_market_filter,
        topix_close=topix_close,
        tech_matrices=tech_matrices,
        trail_atr_mult=trail_atr_mult,
    )


def scenario_log(
    name: str,
    index: int,
    total: int,
    result: dict,
    started_at: float,
    *,
    baseline_sharpe: float | None = None,
) -> None:
    elapsed = time.perf_counter() - started_at
    sharpe = safe_float(result.get("sharpe", 0.0))
    n_trades = int(result.get("n_trades", 0))
    if baseline_sharpe is None:
        baseline_part = ""
    else:
        baseline_part = f" baseline_delta={sharpe - baseline_sharpe:+.3f}"
    print(f"[SCENARIO] {index:02d}/{total:02d} {name} trades={n_trades} sharpe={sharpe:.3f}{baseline_part}")
    collect_and_log_process_memory(name)
    print(f"[ELAPSED] {name}: {elapsed:.2f}s")


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(["sharpe", "expectancy", "trade_count"], ascending=[False, False, False]).iloc[0]


def try_load_usdjpy(start: str = FULL_START, end: str = OOS_END) -> pd.Series:
    candidates = [
        REPO_ROOT / "cache" / "ohlcv" / "JPY=X.parquet",
        REPO_ROOT / "src" / "cache" / "ohlcv" / "JPY=X.parquet",
        REPO_ROOT / "data" / "backtest_dataset" / "2026-03-28" / "JPY=X.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        series = df[close_col].dropna()
        series.index = pd.to_datetime(series.index)
        series = series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
        if not series.empty:
            return series

    if not ALLOW_YFINANCE_NETWORK:
        return pd.Series(dtype=float)

    try:
        import yfinance as yf

        df = yf.download("JPY=X", start=start, end=end, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        return close
    except Exception:
        return pd.Series(dtype=float)


def report_footer(*, scenario_count: int, best_sharpe: float, baseline_sharpe: float) -> list[str]:
    rss = get_process_rss_mb()
    rss_text = f"{rss:.1f} MB" if rss is not None else "n/a"
    return [
        f"scenario_count={scenario_count}",
        f"best_sharpe={best_sharpe:.3f}",
        f"baseline_sharpe={baseline_sharpe:.3f}",
        f"rss={rss_text}",
    ]
