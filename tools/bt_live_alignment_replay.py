"""
tools/bt_live_alignment_replay.py — BT/live整合化監査 replay（research-only）

目的: live専用制約（MAX_NEW_POS_PER_DAY=1 / sector max_names=1 / cash_buffer=10%）を
      Pattern A バックテスト台帳に「後適用」し、IS再現性の劣化を定量化する。

方式: capital_allocation_abc.run_period(pattern="A", return_trades=True) を読み取り実行し、
      確定トレード台帳に対して制約フィルタ＋サイズスケールを適用して日次エクイティを再構成。
      プロダクションコード・パラメータは一切変更しない。

⚠ 近似の限界（レポートに明記）:
  - トレード除去のみ可能。空いたスロットへの「代替エントリー」は生成不能
    → B/C/E の劣化値は上限推定（実エンジンなら部分補填があり得る）。
  - D は qty×0.90 の線形スケール。lot丸めは無視（実liveはより劣化）。

Run: cd C:/ai-trading && python tools/bt_live_alignment_replay.py
出力: backtests/bt_live_alignment_replay_2026-06-12.json + コンソール表
"""
from __future__ import annotations

import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, run_period, COST_ONE_WAY,
    IS_START, IS_END, OOS_START, OOS_END,
)

OUT_JSON = Path("backtests/bt_live_alignment_replay_2026-06-12.json")
FULL_START, FULL_END = "2018-01-01", "2025-12-31"


# ─────────────────────────────────────────────────────────────────────
# 台帳 → 完結トレードリスト
# ─────────────────────────────────────────────────────────────────────
def build_round_trips(raw_trades, n_dates, close_lookup):
    """SELLレコード＝完結ペア。未決済BUYは最終日closeで合成クローズ。"""
    sells = [t for t in raw_trades if t["side"] == "SELL"]
    buys  = [t for t in raw_trades if t["side"] == "BUY"]

    # BUY側RSR（reason="RSR=xx.x"）を (symbol, entry_idx) で索引
    rsr_at_entry = {}
    for b in buys:
        try:
            rsr_at_entry[(b["symbol"], b["entry_idx"])] = float(
                str(b["reason"]).split("=")[1])
        except Exception:
            rsr_at_entry[(b["symbol"], b["entry_idx"])] = 0.0

    trips = []
    matched = set()
    for s in sells:
        key = (s["symbol"], s["entry_idx"])
        matched.add(key)
        trips.append({
            "symbol": s["symbol"], "qty": float(s["qty"]),
            "entry_px": float(s["entry"]), "exit_px": float(s["exit"]),
            "entry_idx": int(s["entry_idx"]), "exit_idx": int(s["exit_idx"]),
            "reason": s["reason"],
            "rsr": rsr_at_entry.get(key, 0.0),
        })
    # 未決済 BUY → 合成クローズ（最終日close）
    for b in buys:
        key = (b["symbol"], b["entry_idx"])
        if key in matched:
            continue
        last_px = close_lookup(b["symbol"], n_dates - 1)
        if last_px is None or not np.isfinite(last_px):
            continue
        trips.append({
            "symbol": b["symbol"], "qty": float(b["qty"]),
            "entry_px": float(b["entry"]), "exit_px": float(last_px),
            "entry_idx": int(b["entry_idx"]), "exit_idx": n_dates - 1,
            "reason": "OPEN_END",
            "rsr": rsr_at_entry.get(key, 0.0),
        })
    trips.sort(key=lambda t: (t["entry_idx"], -t["rsr"]))
    return trips


# ─────────────────────────────────────────────────────────────────────
# 制約フィルタ
# ─────────────────────────────────────────────────────────────────────
def filter_max_new(trips, max_new=1):
    """同一 entry_idx（執行日）の上位 max_new 件のみ残す（RSR降順=エンジン執行順）。"""
    kept, dropped = [], []
    per_day = {}
    for t in trips:
        c = per_day.get(t["entry_idx"], 0)
        if c < max_new:
            kept.append(t); per_day[t["entry_idx"]] = c + 1
        else:
            dropped.append(t)
    return kept, dropped


def filter_sector_names(trips, sectors, max_names=1):
    """エントリー時点で同一セクターの保有（残存トレード）があればBUY却下。"""
    kept, dropped = [], []
    open_list = []  # (exit_idx, sector)
    for t in trips:
        i = t["entry_idx"]
        open_list = [(e, sec) for (e, sec) in open_list if e > i]
        sec = sectors.get(t["symbol"], "不明")
        n_same = sum(1 for (_, s) in open_list if s == sec)
        if n_same >= max_names:
            dropped.append(t)
        else:
            kept.append(t)
            open_list.append((t["exit_idx"], sec))
    return kept, dropped


# ─────────────────────────────────────────────────────────────────────
# 日次エクイティ再構成
# ─────────────────────────────────────────────────────────────────────
def simulate(trips, dates, close_lookup, capital, scale=1.0):
    n = len(dates)
    cash = capital
    cash_flow = np.zeros(n + 1)       # 日次キャッシュ変動（exit翌日精算近似→exit_idxで計上）
    holdings_val = np.zeros(n)

    for t in trips:
        q = t["qty"] * scale
        buy_cost  = q * t["entry_px"] * (1 + COST_ONE_WAY)
        sell_gain = q * t["exit_px"]  * (1 - COST_ONE_WAY)
        cash_flow[t["entry_idx"]] -= buy_cost
        cash_flow[t["exit_idx"]]  += sell_gain
        for i in range(t["entry_idx"], t["exit_idx"]):
            px = close_lookup(t["symbol"], i)
            if px is not None and np.isfinite(px):
                holdings_val[i] += q * px

    cash_series = capital + np.cumsum(cash_flow[:n])
    equity = cash_series + holdings_val
    exposure = np.divide(holdings_val, equity,
                         out=np.zeros(n), where=equity > 0)
    return pd.Series(equity, index=dates), pd.Series(exposure, index=dates)


def seg_metrics(equity, exposure, trips, dates, start, end, capital):
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    mask = (equity.index >= ts) & (equity.index <= te)
    eq = equity[mask]
    if len(eq) < 5:
        return {}
    base = eq.iloc[0]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / base) ** (1 / max(years, 1e-9)) - 1
    rebased = eq / eq.cummax()
    maxdd = (rebased - 1).min()
    idx_map = {d: i for i, d in enumerate(dates)}
    lo, hi = idx_map[eq.index[0]], idx_map[eq.index[-1]]
    seg_trips = [t for t in trips if lo <= t["exit_idx"] <= hi]
    pnls = []
    for t in seg_trips:
        q = t["qty"]
        net = (t["exit_px"] - t["entry_px"]) * q \
              - (t["entry_px"] + t["exit_px"]) * q * COST_ONE_WAY
        pnls.append(net)
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p <= 0)
    pf = wins / losses if losses > 0 else float("inf")
    wr = (sum(1 for p in pnls if p > 0) / len(pnls) * 100) if pnls else 0.0
    return {
        "cagr": round(cagr * 100, 2),
        "pf": round(pf, 3) if np.isfinite(pf) else None,
        "max_dd": round(maxdd * 100, 2),
        "win_rate": round(wr, 1),
        "exposure": round(float(exposure[mask].mean()) * 100, 1),
        "n_trades": len(seg_trips),
    }


def yearly_returns(equity):
    out = {}
    for y in range(2018, 2026):
        eq = equity[(equity.index >= f"{y}-01-01") & (equity.index <= f"{y}-12-31")]
        if len(eq) < 5:
            continue
        ret = eq.iloc[-1] / eq.iloc[0] - 1
        dd = (eq / eq.cummax() - 1).min()
        out[str(y)] = {"ret": round(ret * 100, 2), "max_dd": round(dd * 100, 2)}
    return out


# ─────────────────────────────────────────────────────────────────────
def main():
    cfg = load_strategy_config()
    capital = float(cfg.portfolio.capital)
    print("=" * 70)
    print("  BT/live整合化監査 replay — A/B/C/D/E  (research-only)")
    print("=" * 70)

    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("[2/4] Pattern A full 2018-2025 実行 (return_trades=True)...")
    t0 = time.time()
    m = run_period(universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                   topix_close, rsr_syms, cfg,
                   start=FULL_START, end=FULL_END, pattern="A",
                   return_trades=True, verbose=False)
    print(f"  engine: CAGR={m.get('cagr', 0):+.1f}%  Sharpe={m.get('sharpe', 0):.3f} "
          f" MaxDD={m.get('max_dd', 0):.1f}%  Trades={m.get('n_trades', 0)} "
          f" ({time.time()-t0:.0f}s)")

    dates_dt = pd.DatetimeIndex(pd.to_datetime(list(m["_common_dates"])))
    dates = list(dates_dt)
    n_dates = len(dates)

    # close lookup
    close_cache = {}
    def close_lookup(sym, i):
        if sym not in close_cache:
            ser = universe_raw[sym]["df"]["Close"]
            ser.index = pd.to_datetime(ser.index)
            close_cache[sym] = ser.reindex(dates_dt).values.astype(float)
        return close_cache[sym][i]

    trips_all = build_round_trips(m["_trades_raw"], n_dates, close_lookup)
    sectors = {s: sec for s, sec in rsr_syms.items()}
    print(f"  round trips: {len(trips_all)} (合成クローズ含む)")

    # ── ケース定義 ────────────────────────────────────────────────
    kept_B, drop_B = filter_max_new(trips_all, 1)
    kept_C, drop_C = filter_sector_names(trips_all, sectors, 1)
    kept_BC, drop_BC2 = filter_sector_names(kept_B, sectors, 1)

    cases = {
        "A_baseline":        (trips_all, 1.0),
        "B_max_new_1":       (kept_B,    1.0),
        "C_sector_names_1":  (kept_C,    1.0),
        "D_cash_buffer_10":  (trips_all, 0.90),
        "E_all":             (kept_BC,   0.90),
    }
    print(f"  B drop: {len(drop_B)}  C drop: {len(drop_C)} "
          f" E drop: {len(drop_B) + len(drop_BC2)}")

    # ── 各ケース集計 ──────────────────────────────────────────────
    results = {}
    for name, (trips, scale) in cases.items():
        eq, exp = simulate(trips, dates_dt, close_lookup, capital, scale)
        is_m  = seg_metrics(eq, exp, trips, dates, IS_START, IS_END, capital)
        oos_m = seg_metrics(eq, exp, trips, dates, OOS_START, OOS_END, capital)
        results[name] = {
            "IS": is_m, "OOS": oos_m,
            "oos_is_ratio": round(oos_m["cagr"] / is_m["cagr"], 3)
                            if is_m.get("cagr") else None,
            "yearly": yearly_returns(eq),
            "n_kept": len(trips), "scale": scale,
        }
        print(f"\n  {name}: IS CAGR={is_m['cagr']:+.2f}% PF={is_m['pf']} "
              f"DD={is_m['max_dd']}% Exp={is_m['exposure']}% N={is_m['n_trades']} | "
              f"OOS CAGR={oos_m['cagr']:+.2f}%")

    results["_meta"] = {
        "date": "2026-06-12",
        "engine": "capital_allocation_abc.run_period pattern=A",
        "engine_full_metrics": {k: m.get(k) for k in
                                ("cagr", "sharpe", "max_dd", "calmar", "n_trades")},
        "rsr_exit_in_cfg": float(cfg.fujiko.rsr_exit),
        "approximation": [
            "B/C/E: 除去のみ・代替エントリー生成不能 → 劣化の上限推定",
            "D: 線形スケール・lot丸め無視 → 劣化の下限推定",
            "OPEN_END合成クローズ含む",
        ],
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[4/4] 保存: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
