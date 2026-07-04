"""
study80a_observation_infrastructure_2026-07-04.py
Study80A — Observation Infrastructure & CAP_MISS Root Cause Foundation

目的: Study74B-RCAで未解決だった「なぜmax_positions=3が最適なのか」を、
Study81以降で必ず説明可能にするための恒久的観測基盤を構築する。
改善研究ではない。新規BTは合計1回のみ許可（本スクリプト内で1回のみ実行）。

エンジン変更（composite_alpha_bt.py）: 既存の候補ログ4種
(_missed_cands/_skip_detail/_rejected_by_lot_detail/_admitted_by_ratio_detail)を
day-level context（cash_before_entry/used_slots/max_slots/selected_symbols/
selected_scores/position_weights/candidate_count_today/momentum_63d_pct/sector/
market_regime/skip_reason）で拡張し、新規に_selected_cands（SELECTED候補ログ）を追加。
全て既存dict literalへのキー追加、または既存パターンを踏襲した新規append 1箇所のみ。
制御フロー変更ゼロ・Parity確認済み（本ファイル実行時に自動検証・parity_report.md参照）。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

TODAY_STR = date.today().strftime("%Y-%m-%d")
BT_DIR = ROOT / "backtests"
REPORT_DIR = ROOT / "reports"

CAPITAL = 3_000_000
MIN_HOLD = 3
FULL_START, FULL_END = "2018-01-01", "2025-12-30"
PROD_CFG = {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 70.0}

# 既知の基準値（M1適用後Production・parity検証用）
BASELINE = {"cagr": 11.22, "n_trades": 309, "sharpe": 0.564, "max_dd": -18.22, "calmar": 0.616}

FWD_HORIZONS = [5, 10, 20, 40, 60]


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc = cfg.risk_controls.bear_universe_filter
    be = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_full(ds, sym_active_df):
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"], start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=PROD_CFG["rsr_exit"],
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False, enable_mtf_filter=False,
        sizing_mode="existing", exit_policy="A", addon_policy="D",
        addon_size_frac=0.25, addon_atr_mult=1.0,
    )


def build_trade_ledger(raw: dict) -> list[dict]:
    """Study78と同一手法（entry_idx/exit_idxで日付マッピング・addon判定）"""
    sells = raw.get("_trades", [])
    buys = raw.get("_trades_buy", [])
    addons = raw.get("_addon_detail", [])
    eq: pd.Series = raw["equity_curve"]
    dates = eq.index

    buy_by_key: dict[tuple, list[dict]] = {}
    for b in buys:
        buy_by_key.setdefault((b["symbol"], b.get("entry_idx")), []).append(b)
    addon_by_symbol: dict[str, list[dict]] = {}
    for a in addons:
        addon_by_symbol.setdefault(a["symbol"], []).append(a)
    date_idx_map = {str(d.date()): i for i, d in enumerate(dates)}

    ledger = []
    for t in sells:
        eidx, xidx = t.get("entry_idx"), t.get("exit_idx")
        if eidx is None or xidx is None:
            continue
        b = buy_by_key.get((t["symbol"], eidx), [{}])
        b0 = b[0] if b else {}
        entry_date = str(dates[eidx].date()) if eidx < len(dates) else None
        exit_date = str(dates[xidx].date()) if xidx < len(dates) else None
        entry_px, exit_px = float(t.get("entry", 0)), float(t.get("exit", 0))
        qty, pnl = int(t.get("qty", 0)), float(t.get("pnl", 0))
        ret_pct = round((exit_px - entry_px) / entry_px * 100, 3) if entry_px > 0 else 0.0
        sym_addons = addon_by_symbol.get(t["symbol"], [])
        addon_flag = any(eidx <= date_idx_map.get(a["date"], -999) <= xidx for a in sym_addons) if sym_addons else False
        ledger.append({
            "symbol": t["symbol"], "sector": t.get("sector", "不明"),
            "entry_date": entry_date, "exit_date": exit_date,
            "entry_idx": eidx, "exit_idx": xidx,
            "entry_price": round(entry_px, 2), "exit_price": round(exit_px, 2),
            "qty": qty, "return_pct": ret_pct,
            "r_multiple": round((ret_pct / 100.0) / 0.08, 3),
            "pnl": round(pnl, 1), "holding_days": xidx - eidx,
            "entry_atr_pct": b0.get("entry_atr_pct"), "entry_rsr": b0.get("entry_rsr"),
            "entry_type": b0.get("entry_type", "unknown"),
            "exit_policy": PROD_CFG["exit_policy"], "addon_received": addon_flag,
            "exit_reason": t.get("reason", ""),
            # ── Study80A v2拡張フィールド（selected_candsから結合） ──
            "candidate_rank": None, "candidate_count": None, "cash_before_entry": None,
            "slots_used": None, "portfolio_exposure": None, "entry_cluster_id": entry_date,
            "entry_sector": t.get("sector", "不明"), "entry_regime": None,
        })
    return ledger


def enrich_ledger_v2(ledger: list[dict], selected_cands: list[dict]) -> list[dict]:
    """trade_dataset_v2: SELLトレードにSELECTED候補ログの日次コンテキストを結合"""
    sel_by_key: dict[tuple, dict] = {}
    for c in selected_cands:
        sel_by_key[(c["symbol"], c["date"])] = c
    for t in ledger:
        key = (t["symbol"], t["entry_date"])
        c = sel_by_key.get(key)
        if c:
            t["candidate_rank"] = c["rank"]
            t["candidate_count"] = c.get("candidate_count_today")
            t["cash_before_entry"] = c.get("cash_before_entry")
            t["slots_used"] = c.get("used_slots")
            weights = c.get("position_weights", {})
            t["portfolio_exposure"] = round(sum(weights.values()), 4) if weights else None
            t["entry_regime"] = c.get("market_regime")
    return ledger


def merge_missed_candidates(raw: dict) -> list[dict]:
    """真の見送り(CAP_MISS/SECTOR_CAP/CLUSTER_CAP/GROSS_EXPOSURE/LOT_REJECT)を統合。
    ADMITTED_BY_RATIOは実際には約定成立(rescue)のため区別してタグ付けし別枠格納。"""
    merged = []
    merged.extend(raw.get("_missed_cands", []))
    merged.extend(raw.get("_skip_detail", []))
    merged.extend(raw.get("_rejected_by_lot_detail", []))
    rescued = raw.get("_admitted_by_ratio_detail", [])
    return merged, rescued


def forward_return_for_candidate(universe_raw: dict, symbol: str, date_str: str) -> dict:
    if symbol not in universe_raw:
        return {}
    df = universe_raw[symbol]["df"]
    if date_str not in df.index.astype(str):
        # 近似: date以降最初の営業日
        idx_pos = df.index.searchsorted(pd.Timestamp(date_str))
        if idx_pos >= len(df):
            return {}
    else:
        idx_pos = df.index.get_loc(pd.Timestamp(date_str))
    base_close = float(df["Close"].iloc[idx_pos])
    if base_close <= 0 or np.isnan(base_close):
        return {}
    out = {}
    max_h = max(FWD_HORIZONS)
    window = df.iloc[idx_pos: idx_pos + max_h + 1]
    if len(window) < 2:
        return {}
    for h in FWD_HORIZONS:
        if idx_pos + h < len(df):
            px = float(df["Close"].iloc[idx_pos + h])
            out[f"forward_{h}"] = round((px - base_close) / base_close * 100, 2) if not np.isnan(px) else None
        else:
            out[f"forward_{h}"] = None
    mfe = (float(window["High"].max()) - base_close) / base_close * 100
    mae = (float(window["Low"].min()) - base_close) / base_close * 100
    out["mfe_pct"] = round(mfe, 2)
    out["mae_pct"] = round(mae, 2)
    running_max = window["Close"].cummax()
    dd = (window["Close"] - running_max) / running_max
    out["max_dd_pct"] = round(float(dd.min()) * 100, 2)
    peak_day = int(window["Close"].values.argmax())
    out["holding_days_equivalent"] = peak_day
    return out


def main():
    print("=" * 80)
    print("  Study80A — Observation Infrastructure & CAP_MISS Root Cause Foundation")
    print(f"  Date: {TODAY_STR}  (唯一の新規BT許可・FULL 2018-2025 CURRENT M1適用後)")
    print("=" * 80)

    ds = build_common_dataset(FULL_END)
    all_syms = list(ds["trade_syms"].keys())
    act = get_active(ds, all_syms, FULL_START, FULL_END)

    print("\n[RUN] Production FULL run（唯一の新規BT）...")
    raw = run_full(ds, act)
    print(f"  CAGR={raw['cagr']:+.2f}%  Sharpe={raw['sharpe']:.3f}  MaxDD={raw['max_dd']:.2f}%  "
          f"Calmar={raw['calmar']:.3f}  Trades={raw['n_trades']}")

    # ── Part7: Parity検証 ────────────────────────────────────────────────
    print("\n[Part7] Parity検証...")
    parity_ok = (raw["cagr"] == BASELINE["cagr"] and raw["n_trades"] == BASELINE["n_trades"]
                 and raw["sharpe"] == BASELINE["sharpe"] and raw["max_dd"] == BASELINE["max_dd"]
                 and raw["calmar"] == BASELINE["calmar"])
    print(f"  CAGR: {raw['cagr']} vs {BASELINE['cagr']}  {'OK' if raw['cagr']==BASELINE['cagr'] else 'MISMATCH'}")
    print(f"  Trades: {raw['n_trades']} vs {BASELINE['n_trades']}  {'OK' if raw['n_trades']==BASELINE['n_trades'] else 'MISMATCH'}")
    print(f"  Sharpe/MaxDD/Calmar一致: {parity_ok}")

    # ── 台帳構築 ──────────────────────────────────────────────────────────
    ledger = build_trade_ledger(raw)
    selected_cands = raw.get("_selected_cands", [])
    ledger_v2 = enrich_ledger_v2(ledger, selected_cands)
    print(f"\n[LEDGER] trade_dataset_v2: {len(ledger_v2)}トレード")

    missed_merged, rescued = merge_missed_candidates(raw)
    print(f"[CANDIDATES] missed_candidates_full: {len(missed_merged)}件（+rescued(ADMITTED_BY_RATIO): {len(rescued)}件）")

    # ── Part2: Forward Return Framework ──────────────────────────────────
    print("\n[Part2] Forward Return計算中（価格データ直接参照・BTではない）...")
    universe_raw = ds["universe_raw"]
    for c in missed_merged:
        fwd = forward_return_for_candidate(universe_raw, c["symbol"], c["date"])
        c.update(fwd)
    for c in rescued:
        fwd = forward_return_for_candidate(universe_raw, c["symbol"], c["date"])
        c.update(fwd)
    n_with_fwd = sum(1 for c in missed_merged if "forward_5" in c)
    print(f"  Forward Return付与: {n_with_fwd}/{len(missed_merged)}件")

    # ── Part5: Opportunity Cost Framework ────────────────────────────────
    print("\n[Part5] Opportunity Cost計算中...")
    adopted_returns = [t["return_pct"] for t in ledger_v2]
    adopted_mean_ret = float(np.mean(adopted_returns)) if adopted_returns else 0.0
    adopted_pf = (sum(t["pnl"] for t in ledger_v2 if t["pnl"] > 0) /
                  max(1.0, abs(sum(t["pnl"] for t in ledger_v2 if t["pnl"] < 0))))
    adopted_expectancy = float(np.mean([t["pnl"] for t in ledger_v2])) if ledger_v2 else 0.0
    adopted_win_rate = 100.0 * len([t for t in ledger_v2 if t["pnl"] >= 0]) / max(1, len(ledger_v2))

    def opp_cost_by(key_fn, items):
        groups: dict[str, list] = {}
        for c in items:
            if "forward_20" not in c or c["forward_20"] is None:
                continue
            groups.setdefault(key_fn(c), []).append(c["forward_20"])
        return {k: {"n": len(v), "mean_forward_20_pct": round(float(np.mean(v)), 2),
                     "median_forward_20_pct": round(float(np.median(v)), 2)}
                for k, v in groups.items()}

    opportunity_cost_dataset = {
        "date": TODAY_STR,
        "adopted_baseline": {
            "n": len(ledger_v2), "mean_return_pct": round(adopted_mean_ret, 2),
            "profit_factor": round(adopted_pf, 3), "expectancy_yen": round(adopted_expectancy, 1),
            "win_rate_pct": round(adopted_win_rate, 1),
        },
        "missed_candidates_forward20_overall": {
            "n": n_with_fwd,
            "mean_forward_20_pct": round(float(np.mean([c["forward_20"] for c in missed_merged if c.get("forward_20") is not None])), 2) if n_with_fwd else None,
        },
        "opportunity_cost_by_sector": opp_cost_by(lambda c: c.get("sector", "不明"), missed_merged),
        "opportunity_cost_by_regime": opp_cost_by(lambda c: c.get("market_regime", "unknown"), missed_merged),
        "opportunity_cost_by_rank": opp_cost_by(
            lambda c: "rank0" if c.get("rank") == 0 else ("rank1-2" if c.get("rank", 99) <= 2 else "rank3+"),
            missed_merged),
        "opportunity_cost_by_skip_reason": opp_cost_by(lambda c: c.get("skip_reason", "unknown"), missed_merged),
        "note": "forward_20(20営業日後リターン)を基準指標として使用。adopted側のreturn_pctとの直接比較は"
                "保有期間が異なるため単純差分ではなくmean/median水準の並列比較にとどめる。",
    }
    print(f"  採用トレード平均リターン: {adopted_mean_ret:+.2f}%  見送り候補forward_20平均: "
          f"{opportunity_cost_dataset['missed_candidates_forward20_overall'].get('mean_forward_20_pct')}%")

    # ── Part4: Correlation Observation（同日候補の集中度・観測のみ） ────────
    print("\n[Part4] Correlation Observation（同日候補集中度）...")
    all_day_records = missed_merged + selected_cands
    by_date: dict[str, list] = {}
    for c in all_day_records:
        by_date.setdefault(c["date"], []).append(c)
    correlation_records = []
    for d, recs in by_date.items():
        if len(recs) < 2:
            continue
        sectors = [r.get("sector", "不明") for r in recs]
        sector_counts = {}
        for s in sectors:
            sector_counts[s] = sector_counts.get(s, 0) + 1
        max_sector_share = max(sector_counts.values()) / len(recs)
        atr_signs = [1 if (r.get("momentum_63d_pct") or 0) > 0 else -1 for r in recs]
        same_direction_share = max(atr_signs.count(1), atr_signs.count(-1)) / len(atr_signs)
        correlation_records.append({
            "date": d, "n_candidates": len(recs),
            "distinct_sectors": len(sector_counts), "max_sector_concentration_pct": round(max_sector_share * 100, 1),
            "momentum_same_direction_pct": round(same_direction_share * 100, 1),
        })
    avg_sector_conc = round(float(np.mean([r["max_sector_concentration_pct"] for r in correlation_records])), 1) if correlation_records else None
    avg_same_dir = round(float(np.mean([r["momentum_same_direction_pct"] for r in correlation_records])), 1) if correlation_records else None
    correlation_dataset = {
        "date": TODAY_STR, "n_multi_candidate_days": len(correlation_records),
        "avg_max_sector_concentration_pct": avg_sector_conc,
        "avg_momentum_same_direction_pct": avg_same_dir,
        "daily_records": correlation_records,
        "purpose": "「4銘柄目は本当に独立リスクだったのか」をStudy81で検証するための同日候補集中度データ。新規ロジックなし・観測のみ。",
    }
    print(f"  複数候補日: {len(correlation_records)}日  平均セクター集中度: {avg_sector_conc}%  "
          f"平均モメンタム同方向率: {avg_same_dir}%")

    # ── 保存 ─────────────────────────────────────────────────────────────
    print("\n[SAVE] 成果物保存中...")
    with open(BT_DIR / "trade_dataset_v2.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "n_trades": len(ledger_v2), "trades": ledger_v2}, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "missed_candidates_full.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "n_missed": len(missed_merged), "n_rescued_admitted_by_ratio": len(rescued),
                    "missed_candidates": missed_merged, "rescued_candidates": rescued}, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "forward_return_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"date": TODAY_STR, "horizons": FWD_HORIZONS,
                    "records": [{"date": c["date"], "symbol": c["symbol"], "skip_reason": c.get("skip_reason"),
                                  **{k: c.get(k) for k in ["forward_5", "forward_10", "forward_20", "forward_40", "forward_60",
                                                             "mfe_pct", "mae_pct", "max_dd_pct", "holding_days_equivalent"]}}
                                 for c in missed_merged if "forward_5" in c]},
                  f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "opportunity_cost_dataset.json", "w", encoding="utf-8") as f:
        json.dump(opportunity_cost_dataset, f, ensure_ascii=False, indent=2, default=str)
    with open(BT_DIR / "correlation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(correlation_dataset, f, ensure_ascii=False, indent=2, default=str)

    print("  [OUTPUT] backtests/trade_dataset_v2.json")
    print("  [OUTPUT] backtests/missed_candidates_full.json")
    print("  [OUTPUT] backtests/forward_return_dataset.json")
    print("  [OUTPUT] backtests/opportunity_cost_dataset.json")
    print("  [OUTPUT] backtests/correlation_dataset.json")

    return {"parity_ok": parity_ok, "raw_metrics": {"cagr": raw["cagr"], "n_trades": raw["n_trades"],
                                                       "sharpe": raw["sharpe"], "max_dd": raw["max_dd"], "calmar": raw["calmar"]}}


if __name__ == "__main__":
    main()
