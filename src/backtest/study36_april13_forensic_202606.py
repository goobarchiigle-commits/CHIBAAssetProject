"""
backtest/study36_april13_forensic_202606.py

Study36 — April-13 Report Forensic Reproduction.
感度分析・最適化・パラメータ探索は一切行わない。22.4%/1.582/-12.32%の出所特定のみ。

APRIL_REPRO config (docs/research/april13_baseline_identification_and_rollback_plan.md +
decay_audit_20260413_vs_20260624.md の確定済みSTEP1抽出をそのまま使用):
  scenario=BASELINE, sym_active_df=None(静的RSR42), market_shock_mode="full_exit",
  rsr_exit_threshold=75.0(min_rsrフォールバック、当時真値はUNKNOWN), min_hold=3,
  sizing_mode="existing", capital=3,000,000, max_positions=3, turtle_exit=55,
  全PROD_FAITHFULフラグ=False, enable_filters=False(デフォルトのまま明示しない)。

期間監査: 2018-2024 / 2019-2024 / 2020-2024 / 2021-2024 を同一ロジックで比較。
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.backtest import composite_alpha_bt as cab
from src.backtest.snapshot_archaeology_202606 import build_common_dataset

PERIODS = [("2018-01-01", "2024-12-31"), ("2019-01-01", "2024-12-31"),
           ("2020-01-01", "2024-12-31"), ("2021-01-01", "2024-12-31")]

TARGET = {"cagr": 22.4, "sharpe": 1.582, "max_dd": -12.32}


def main():
    ds = build_common_dataset("2024-12-31")
    cfg = ds["base_cfg"]
    print(f"TARGET: CAGR={TARGET['cagr']}%  Sharpe={TARGET['sharpe']}  MaxDD={TARGET['max_dd']}%\n")
    for start, end in PERIODS:
        res = cab.run_scenario(
            scenario="BASELINE",
            universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
            regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
            cfg=cfg, start=start, end=end, verbose=False,
            tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
            capital=3_000_000, min_hold=3,
            topix_close=ds["topix_close"],
            market_shock_mode="full_exit",
            sym_active_df=None,
            enable_simple_rsr_exit=True,
            rsr_exit_threshold=75.0,
            enable_multilayer_rsr=False,
            use_fixed_pct_trail=False,
            enable_atr_trailing_prod=False,
            enable_mtf_filter=False,
            enable_atr_risk_sizing=False,
            sizing_mode="existing",
        )
        gap_cagr = res["cagr"] - TARGET["cagr"]
        gap_sharpe = res["sharpe"] - TARGET["sharpe"]
        gap_dd = res["max_dd"] - TARGET["max_dd"]
        print(f"IS={start}..{end}  n_trades={res['n_trades']}  "
              f"CAGR={res['cagr']:+.2f}%(gap{gap_cagr:+.2f}pp)  "
              f"Sharpe={res['sharpe']:.3f}(gap{gap_sharpe:+.3f})  "
              f"MaxDD={res['max_dd']:.2f}%(gap{gap_dd:+.2f}pp)  "
              f"avg_hold={res['avg_hold_days']:.1f}  avg_exposure={res['avg_exposure']:.1f}%")


if __name__ == "__main__":
    main()
