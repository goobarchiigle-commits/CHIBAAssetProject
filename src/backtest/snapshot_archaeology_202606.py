"""
backtest/snapshot_archaeology_202606.py

考古学的スナップショット再現バックテスト（改善案・パラメータ探索・最適化は一切行わない）。

docs/research/strategy_snapshot_manifest.md の確定値のみを使用し、
src/configs/config_snapshot_S{1..5}.yaml を読み込んで5時点のロジックスナップショットを
同一データセット・同一コスト構造・同一資本(3,000,000円)で実行する。

S1 (2026-03-15) は別アーキテクチャ（topk_live_equivalent.py, TEMPORAL24+top_k）。
S2-S5 は composite_alpha_bt.run_scenario の BASELINE/PROD_FAITHFULスタイル + 明示flagsで再現。

IS: 2018-01-01〜2024-12-31 / OOS: 2025-01-01〜直近データ（独立run、capitalはIS/OOS共通3,000,000で再スタート）。
出力: backtests/snapshot_archaeology_202606_<date>.json
"""

from __future__ import annotations

import sys
import json
import warnings
import dataclasses
from pathlib import Path
from datetime import date, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import yaml
import numpy as np
import pandas as pd

from src.paths import CONFIGS_DIR, RESULTS_DIR
from src.config_loader import load_strategy_config

from src.backtest import composite_alpha_bt as cab
from src.backtest import topk_live_equivalent as tke

IS_START, IS_END = "2018-01-01", "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # 直近取得可能日（前日）
CAPITAL = 3_000_000
CONFIG_DIR = CONFIGS_DIR
SNAPSHOTS = ["S1", "S2", "S3", "S4", "S4_CDOS_OVERRIDE_VARIANT", "S5"]


def _load_yaml(snapshot_id: str) -> dict:
    fname = "config_snapshot_S4.yaml" if snapshot_id.startswith("S4") else f"config_snapshot_{snapshot_id}.yaml"
    with open(CONFIG_DIR / fname, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unit_metrics(cagr_frac, sharpe, sortino, max_dd_frac, calmar, pf, avg_exp_frac, avg_hold, n_yr) -> dict:
    """全エンジン共通: CAGR/MaxDD/avgExposureは%単位(0-100)に統一。"""
    return {
        "cagr_pct": round(cagr_frac * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd_frac * 100, 2),
        "calmar": round(calmar, 3) if np.isfinite(calmar) else None,
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "avg_exposure_pct": round(avg_exp_frac * 100, 1),
        "avg_hold_days": round(avg_hold, 1),
        "trades_per_year": round(n_yr, 1),
    }


def _segment_metrics(equity_full: pd.Series, exposure_full: pd.Series, trades_full: list,
                      eq_dates: pd.DatetimeIndex, seg_start: str, seg_end: str) -> dict:
    """単一連続runの結果からIS/OOS部分区間を独立評価として切り出す（capitalはseg開始時点equityで再基準化）。"""
    mask = (equity_full.index >= pd.Timestamp(seg_start)) & (equity_full.index <= pd.Timestamp(seg_end))
    eq_seg = equity_full[mask]
    exp_seg = exposure_full[mask]
    if len(eq_seg) < 2:
        return None
    years = len(eq_seg) / 252
    cagr = (eq_seg.iloc[-1] / eq_seg.iloc[0]) ** (1 / max(years, 0.01)) - 1
    daily_ret = eq_seg.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    sortino = float(daily_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0
    roll_max = eq_seg.expanding().max()
    dd = (eq_seg - roll_max) / roll_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    idx_pos = {d: i for i, d in enumerate(eq_dates)}
    seg_start_idx = idx_pos.get(eq_seg.index[0], 0)
    seg_end_idx = idx_pos.get(eq_seg.index[-1], len(eq_dates) - 1)
    seg_trades = [t for t in trades_full if t["side"] == "SELL" and t.get("pnl") is not None
                  and seg_start_idx <= t.get("exit_idx", -1) <= seg_end_idx]
    wins = [t["pnl"] for t in seg_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in seg_trades if t["pnl"] <= 0]
    gross_loss = abs(sum(losses))
    pf = (sum(wins) / gross_loss) if gross_loss > 0 else float("inf")
    hold_days = [t["exit_idx"] - t["entry_idx"] for t in seg_trades]
    avg_hold = float(np.mean(hold_days)) if hold_days else 0.0
    n_yr = len(seg_trades) / max(years, 0.01)
    avg_exp = float(exp_seg.mean()) if len(exp_seg) else 0.0

    return _unit_metrics(cagr, sharpe, sortino, max_dd, calmar, pf, avg_exp, avg_hold, n_yr)


def run_s1(full_start: str, full_end: str) -> tuple[dict, dict]:
    """一回の連続runでIS/OOSを切り出す（OOS単独呼び出しはmin_days未達で空集合になるため）。"""
    res = tke.run_backtest(
        top_k=4, start=full_start, end=full_end, capital=CAPITAL,
        max_hold_days=60, max_single_weight=0.25, max_new_positions_per_day=2,
        min_sepa=6, verbose=False,
    )
    eq = res["equity_curve"]
    exp = res["exposure_curve"]
    trades = res["trades"]
    is_m = _segment_metrics(eq, exp, trades, eq.index, IS_START, IS_END)
    oos_m = _segment_metrics(eq, exp, trades, eq.index, OOS_START, full_end)
    return is_m, oos_m


def build_common_dataset(end: str):
    """S2-S5共通: RSR42 + dyn_rsr42_bear_rs0 の全期間プリコンピュート（一度だけ実行・再利用）。"""
    rsr_syms = cab._load_rsr_universe(verbose=True)
    trade_syms = rsr_syms
    all_syms = {**rsr_syms, **trade_syms}
    universe_raw = cab.download_universe(all_syms, start=IS_START, end=end, min_days=500, verbose=False)
    topix_close = cab._download_topix(IS_START, end, verbose=False)
    rsr42_prices = {sym: universe_raw[sym]["df"]["Close"] for sym in rsr_syms if sym in universe_raw}
    rsr_df = cab.calc_universe_rsr(rsr42_prices)

    base_cfg = load_strategy_config()
    bear_cfg = base_cfg.risk_controls.bear_universe_filter
    bear_exclude = list(bear_cfg.excluded_sectors) if bear_cfg.enabled else None
    sym_active_df = cab.build_dyn_rsr42_active(
        universe_raw=universe_raw, topix_close=topix_close, rsr_df=rsr_df,
        all_syms=list(trade_syms.keys()), start=IS_START, end=end,
        bear_exclude_sectors=bear_exclude,
        sym_sector_map=dict(trade_syms) if bear_exclude else None,
    )
    regime_df = cab._calc_regime(topix_close)
    tech_matrices = cab._precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth_series = cab._calc_breadth(universe_raw)
    return dict(
        base_cfg=base_cfg, trade_syms=trade_syms, rsr_syms=rsr_syms,
        universe_raw=universe_raw, rsr_df=rsr_df, topix_close=topix_close,
        sym_active_df=sym_active_df, regime_df=regime_df,
        tech_matrices=tech_matrices, breadth_series=breadth_series,
    )


def run_snapshot_window(ds: dict, snap_cfg: dict, start: str, end: str, max_positions_override: int | None = None) -> dict:
    cfg = ds["base_cfg"]
    mp = max_positions_override if max_positions_override is not None else snap_cfg["portfolio"]["max_positions"]
    cfg = dataclasses.replace(cfg, portfolio=dataclasses.replace(cfg.portfolio, max_positions=mp, capital=CAPITAL))

    rsr_exit_cfg = snap_cfg.get("rsr_exit", {})
    trail_cfg = snap_cfg.get("trailing_stop", {}) or {}
    sizing_cfg = snap_cfg.get("sizing", {}) or {}

    res = cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=cfg, start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=snap_cfg["risk"]["min_hold_days"],
        topix_close=ds["topix_close"],
        market_shock_mode=snap_cfg["shock_exit"]["mode"],
        sym_active_df=ds["sym_active_df"] if snap_cfg["universe"]["dynamic_universe_enabled"] else None,
        enable_simple_rsr_exit=rsr_exit_cfg.get("enable_simple_rsr_exit", True),
        rsr_exit_threshold=rsr_exit_cfg.get("rsr_exit_threshold") or cfg.fujiko.rsr_exit,
        enable_multilayer_rsr=rsr_exit_cfg.get("enable_multilayer_rsr", False),
        ml_exit_single_only=rsr_exit_cfg.get("ml_exit_single_only", False),
        use_fixed_pct_trail=(trail_cfg.get("mode") == "fixed_pct"),
        fixed_trail_pct=trail_cfg.get("fixed_trail_pct", 0.025),
        enable_atr_trailing_prod=bool(trail_cfg.get("enable_atr_trailing_prod", False)),
        enable_mtf_filter=bool(snap_cfg["mtf"]["enabled"]),
        enable_atr_risk_sizing=bool(sizing_cfg.get("enable_atr_risk_sizing", False)),
        risk_sizing_pct=sizing_cfg.get("risk_pct", cab.PROD_RISK_PCT),
        sizing_mode="existing",
    )
    if not res:
        return None
    return _unit_metrics(
        res["cagr"] / 100.0, res["sharpe"], res["sortino"], res["max_dd"] / 100.0,
        res["calmar"], res["profit_factor"], res["avg_exposure"] / 100.0,
        res["avg_hold_days"], res["n_trades_yr"],
    )


def main() -> int:
    print(f"IS={IS_START}..{IS_END}  OOS={OOS_START}..{OOS_END}  capital={CAPITAL:,}")

    results: dict[str, dict] = {}

    # ---- S1 (別アーキテクチャ) ----
    print("\n[S1] topk_live_equivalent 実行中（連続run、IS/OOSは事後切り出し）...")
    s1_is, s1_oos = run_s1(IS_START, OOS_END)
    results["S1"] = {"IS": s1_is, "OOS": s1_oos}
    print(f"  IS:  {s1_is}")
    print(f"  OOS: {s1_oos}")

    # ---- S2-S5 共通データセット構築 ----
    print("\n[S2-S5] 共通データセット構築中（RSR42 + dyn_rsr42_bear_rs0, 1回のみ）...")
    ds = build_common_dataset(OOS_END)

    snap_ids = ["S2", "S3", "S4", "S5"]
    for sid in snap_ids:
        snap_cfg = _load_yaml(sid)
        print(f"\n[{sid}] composite_alpha_bt 実行中（max_positions={snap_cfg['portfolio']['max_positions']}）...")
        is_res = run_snapshot_window(ds, snap_cfg, IS_START, IS_END)
        oos_res = run_snapshot_window(ds, snap_cfg, OOS_START, OOS_END)
        results[sid] = {"IS": is_res, "OOS": oos_res}
        print(f"  IS:  {is_res}")
        print(f"  OOS: {oos_res}")

    # ---- S4 CDOS override variant (max_positions=5 上限相当) ----
    snap_cfg_s4 = _load_yaml("S4")
    print(f"\n[S4_CDOS_OVERRIDE_VARIANT] max_positions=5 で実行中...")
    is_res = run_snapshot_window(ds, snap_cfg_s4, IS_START, IS_END, max_positions_override=5)
    oos_res = run_snapshot_window(ds, snap_cfg_s4, OOS_START, OOS_END, max_positions_override=5)
    results["S4_CDOS_OVERRIDE_VARIANT"] = {"IS": is_res, "OOS": oos_res}
    print(f"  IS:  {is_res}")
    print(f"  OOS: {oos_res}")

    out_path = RESULTS_DIR / f"snapshot_archaeology_202606_{date.today().isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
