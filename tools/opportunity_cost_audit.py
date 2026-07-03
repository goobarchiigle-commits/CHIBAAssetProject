"""
opportunity_cost_audit.py — 2025-01-01〜現在の機会損失集計

シミュレーション方針:
  • RSR42 >= 75 + 20日高値ブレイク = BUY候補
  • 以下 6 理由で候補を分類し、各理由ごとのカウントと仮想 P&L を算出
    1. CB_ACTIVE       : サーキットブレーカー発動中
    2. SLOT_FULL       : max_positions=3 のポジション満杯
    3. WEIGHT_LIMIT    : price×100 > position_budget (価格超過)
    4. RISK_LIMIT      : qty_risk = 0 (ATR 過大でロット 0)
    5. ALREADY_HOLDING : 既に保有中
    6. ACCEPTED        : 実際にエントリー (参照用)

仮想 P&L:
  entry  = シグナル発生日の終値
  exit   = 55 日安値割れ OR 60 日タイムストップ OR 本日 (open position)
  qty    = 実際に使われる would-be サイジング (current bridge.capital ベース)
  P&L    = (exit - entry) × qty − コスト
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parents[1]
BD_DIR    = ROOT / "data" / "backtest_dataset" / "2026-03-28"
CACHE_DIR = ROOT / "cache" / "ohlcv"

# ── 定数 ─────────────────────────────────────────────────────────────────────
CAPITAL_CONFIG    = 3_000_000
CASH_BUFFER       = 0.10
MAX_SINGLE_WEIGHT = 0.25
RISK_PCT          = 0.0125
MAX_POSITIONS     = 3
LOT               = 100
COST              = 0.001 + 0.00055
MIN_RSR           = 75.0
ENTRY_N           = 20
EXIT_N            = 55
MAX_HOLD          = 60
MAX_DD_LIMIT      = 0.15         # CB_ACTIVE trigger

SIM_START         = "2025-01-01"
SIM_END           = "2026-06-02"    # 本日

# 実資産同期後の bridge.capital (定常状態)
ACTUAL_EQUITY     = 4_015_209   # equity_snapshots.jsonl 最新値 (2026-06-01)
BRIDGE_AFTER      = ACTUAL_EQUITY * (1 - CASH_BUFFER)

TOPIX_SYM = "1306.T"

# ── データロード ──────────────────────────────────────────────────────────────
def _parse(df: pd.DataFrame) -> pd.DataFrame:
    cc = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df[[c for c in [cc, "High", "Low", "Volume"] if c in df.columns]].copy()
    df.columns = [{"Adj Close": "Close"}.get(c, c) for c in df.columns]
    df.index   = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    return df[~df.index.duplicated()].sort_index()

def load_sym(sym: str) -> pd.DataFrame | None:
    bd = BD_DIR    / f"{sym}.parquet"
    ca = CACHE_DIR / f"{sym}.parquet"
    if bd.exists():
        df = _parse(pd.read_parquet(bd))
        if ca.exists():
            dc = _parse(pd.read_parquet(ca))
            new = dc[dc.index > df.index[-1]]
            if not new.empty:
                common = [c for c in df.columns if c in new.columns]
                df = pd.concat([df[common], new[common]])
        return df
    if ca.exists():
        return _parse(pd.read_parquet(ca))
    return None

def load_universe() -> dict[str, str]:
    from src.paths import LIVE_UNIVERSE_FILE
    d = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding="utf-8"))
    return d.get("symbols", d)

def build_panels(uni: dict[str, str]):
    cl_d, hi_d, lo_d = {}, {}, {}
    for sym in uni:
        df = load_sym(sym)
        if df is not None and len(df) > 100:
            cl_d[sym] = df["Close"]
            hi_d[sym] = df["High"]
            lo_d[sym] = df["Low"]
    cl = pd.DataFrame(cl_d).sort_index()
    hi = pd.DataFrame(hi_d).sort_index()
    lo = pd.DataFrame(lo_d).sort_index()
    atr_d: dict[str, pd.Series] = {}
    for sym in cl.columns:
        tr = pd.concat([hi[sym]-lo[sym],
                        (hi[sym]-cl[sym].shift()).abs(),
                        (lo[sym]-cl[sym].shift()).abs()], axis=1).max(axis=1)
        atr_d[sym] = tr.rolling(20).mean()
    atr = pd.DataFrame(atr_d)
    tp = load_sym(TOPIX_SYM)
    topix = tp["Close"].reindex(cl.index).ffill()
    return cl, hi, lo, atr, topix

def rsr42(cl: pd.DataFrame, topix: pd.Series) -> pd.DataFrame:
    return (cl.pct_change(42).sub(topix.pct_change(42), axis=0)
              .rank(axis=1, pct=True) * 100)

# ── シミュレーション ───────────────────────────────────────────────────────────
@dataclass
class Opp:
    date:        str
    symbol:      str
    sector:      str
    signal_close: float
    atr20:        float
    rsr:          float
    reason:       str    # ACCEPTED/CB_ACTIVE/SLOT_FULL/WEIGHT_LIMIT/RISK_LIMIT/ALREADY_HOLDING
    # would-be sizing (AFTER capital ベース)
    qty_cap:      int
    qty_risk:     int
    qty_final:    int
    binding:      str
    # 仮想 P&L
    exit_date:    str    = ""
    exit_price:   float  = 0.0
    virtual_pnl:  float  = 0.0
    virtual_ret:  float  = 0.0
    hold_days:    int    = 0
    exit_reason:  str    = ""
    # 実 P&L (ACCEPTED のみ)
    actual_pnl:   float  = 0.0

def sizing_after(price: float, atr: float, rsr: float) -> dict:
    """AFTER 実資産ベースのサイジング計算"""
    pos_bud  = BRIDGE_AFTER * MAX_SINGLE_WEIGHT
    risk_bud = BRIDGE_AFTER * RISK_PCT
    lot_cost = price * LOT

    qty_cap  = int(pos_bud  // lot_cost) * LOT
    qty_raw  = int(risk_bud / max(atr, 0.01))
    qty_risk = max(LOT, (qty_raw // LOT) * LOT)
    qty_fin  = min(qty_risk, qty_cap) if qty_cap > 0 else 0

    if qty_cap <= 0:
        binding = "price_too_high"
    elif qty_risk < qty_cap:
        binding = "qty_risk"
    elif qty_cap < qty_risk:
        binding = "qty_cap"
    else:
        binding = "equal"
    return {"qty_cap": qty_cap, "qty_risk": qty_risk,
            "qty_final": qty_fin, "binding": binding,
            "pos_budget": pos_bud}

def virtual_exit(sym: str, entry_date: pd.Timestamp, entry_price: float,
                 lo: pd.DataFrame, cl: pd.DataFrame) -> tuple[str, float, int, str]:
    """55日安値割れ / タイムストップ / 本日 のうち最も早い出口。"""
    future = cl.index[cl.index > entry_date]
    if sym not in cl.columns:
        return SIM_END, entry_price, 0, "no_data"
    for k, dt in enumerate(future):
        if k >= MAX_HOLD:
            p = float(cl[sym].get(dt, entry_price))
            return str(dt.date()), p, k, "time_stop"
        p = float(cl[sym].get(dt, entry_price))
        lo_hist = lo[sym].loc[:dt].iloc[-EXIT_N-1:-1]
        stop = float(lo_hist.min()) if not lo_hist.empty else 0.0
        if p <= stop:
            return str(dt.date()), p, k, "turtle_exit"
    # まだオープン
    last_price = float(cl[sym].dropna().iloc[-1])
    return SIM_END, last_price, len(future), "open"

def run_simulation(uni: dict, cl: pd.DataFrame, hi: pd.DataFrame,
                   lo: pd.DataFrame, atr: pd.DataFrame, rsr: pd.DataFrame) -> list[Opp]:
    dates = cl.loc[SIM_START:SIM_END].index

    cash      = float(CAPITAL_CONFIG)
    peak_eq   = float(CAPITAL_CONFIG)
    pos: dict[str, dict]  = {}   # {sym: {entry_price, qty, entry_idx}}
    opps: list[Opp]       = []
    cb_active = False
    cb_end_date: date | None = None

    pos_bud_cfg = CAPITAL_CONFIG * MAX_SINGLE_WEIGHT

    for i, dt in enumerate(dates):
        cl_row = cl.loc[dt]

        # 評価額
        mval = sum(pos[s]["qty"] * float(cl_row.get(s, pos[s]["ep"])) for s in pos)
        eq   = cash + mval
        peak_eq = max(peak_eq, eq)

        # CB判定
        dd = (eq - peak_eq) / peak_eq if peak_eq > 0 else 0.0
        if dd < -MAX_DD_LIMIT and not cb_active:
            cb_active   = True
            cb_end_date = dt.date() + timedelta(days=60)
        if cb_active and cb_end_date and dt.date() >= cb_end_date:
            # CB解除後、equity が recovery_threshold (= peak × 0.85) を超えていれば解除
            if eq >= peak_eq * (1 - MAX_DD_LIMIT):
                cb_active   = False
                cb_end_date = None

        # Exit
        to_exit = []
        for sym, p in pos.items():
            lo_hist = lo[sym].loc[:dt].iloc[-EXIT_N-1:-1] if sym in lo.columns else pd.Series(dtype=float)
            stop    = float(lo_hist.min()) if not lo_hist.empty else 0.0
            price   = float(cl_row.get(sym, p["ep"]))
            held    = i - p["entry_idx"]
            if price <= stop or held >= MAX_HOLD:
                to_exit.append((sym, price))
        for sym, price in to_exit:
            p = pos.pop(sym)
            cash += p["qty"] * price * (1 - COST)
            # 保有中だった機会を ACCEPTED として記録済み → actual_pnl を更新
            for opp in reversed(opps):
                if opp.symbol == sym and opp.reason == "ACCEPTED":
                    opp.exit_date  = str(dt.date())
                    opp.exit_price = price
                    opp.hold_days  = i - p["entry_idx"]
                    opp.actual_pnl = (price - p["ep"]) * p["qty"]
                    break

        # Entry candidates
        if dt not in rsr.index:
            continue
        rsr_row    = rsr.loc[dt].dropna()
        cands      = rsr_row[rsr_row >= MIN_RSR].sort_values(ascending=False)
        new_n      = 0
        used_syms  = set(pos.keys())

        for sym in cands.index:
            if sym not in cl.columns or sym not in hi.columns:
                continue
            price = float(cl_row.get(sym, np.nan))
            if np.isnan(price) or price <= 0:
                continue

            # 20日高値ブレイク確認
            hi_hist = hi[sym].loc[:dt].iloc[-ENTRY_N-1:-1]
            if hi_hist.empty or price <= float(hi_hist.max()):
                continue  # ブレイクアウト未達 → BUY候補にも数えない

            # ATR
            av = atr[sym].get(dt, np.nan) if sym in atr.columns else np.nan
            if np.isnan(av) or av <= 0:
                av = price * 0.02  # fallback 2%

            rsrval = float(rsr_row.get(sym, 0))
            sz     = sizing_after(price, av, rsrval)

            # 拒否理由の判定 (優先順)
            if cb_active:
                reason = "CB_ACTIVE"
            elif sym in used_syms:
                reason = "ALREADY_HOLDING"
            elif len(pos) >= MAX_POSITIONS and new_n == 0:
                reason = "SLOT_FULL"
            elif sz["qty_cap"] <= 0:
                reason = "WEIGHT_LIMIT"
            elif sz["qty_final"] <= 0:
                reason = "RISK_LIMIT"
            else:
                reason = "ACCEPTED"

            opp = Opp(
                date=str(dt.date()), symbol=sym, sector=uni.get(sym, ""),
                signal_close=price, atr20=round(av, 1), rsr=round(rsrval, 1),
                reason=reason,
                qty_cap=sz["qty_cap"], qty_risk=sz["qty_risk"],
                qty_final=sz["qty_final"], binding=sz["binding"],
            )

            # 仮想 P&L (ACCEPTED 以外の拒否シグナル)
            if reason != "ACCEPTED":
                xdate, xprice, xdays, xreason = virtual_exit(sym, dt, price, lo, cl)
                qty = sz["qty_final"] if sz["qty_final"] > 0 else LOT  # fallback 1lot
                pnl = (xprice - price) * qty * (1 - 2 * COST)
                opp.exit_date  = xdate
                opp.exit_price = xprice
                opp.hold_days  = xdays
                opp.exit_reason= xreason
                opp.virtual_pnl= round(pnl, 0)
                opp.virtual_ret= round((xprice - price) / price, 4) if price > 0 else 0.0

            opps.append(opp)

            if reason == "ACCEPTED" and new_n == 0 and len(pos) < MAX_POSITIONS:
                cash -= sz["qty_final"] * price * (1 + COST)
                pos[sym] = {"ep": price, "qty": sz["qty_final"], "entry_idx": i}
                used_syms.add(sym)
                new_n += 1
                opp.exit_date  = SIM_END
                opp.exit_price = float(cl[sym].dropna().iloc[-1])
                opp.hold_days  = 0
                opp.actual_pnl = (opp.exit_price - price) * sz["qty_final"]

    return opps

# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    uni  = load_universe()
    cl, hi, lo, atr, topix = build_panels(uni)
    print(f"  Symbols:{len(cl.columns)}  Range:{cl.index[0].date()}-{cl.index[-1].date()}")
    rsr = rsr42(cl, topix)

    print(f"  Simulating {SIM_START} → {SIM_END}  (BRIDGE_AFTER=¥{BRIDGE_AFTER:,.0f})")
    opps = run_simulation(uni, cl, hi, lo, atr, rsr)
    print(f"  Total signals (BUY candidates): {len(opps)}")

    SEP = "=" * 72

    # ── 1. 拒否理由サマリー ─────────────────────────────────────────────────
    from collections import Counter
    reason_cnt   = Counter(o.reason for o in opps)
    reason_order = ["ACCEPTED","CB_ACTIVE","SLOT_FULL","WEIGHT_LIMIT",
                    "RISK_LIMIT","ALREADY_HOLDING"]

    print()
    print(SEP)
    print(f"  BUY候補シグナル集計  {SIM_START} 〜 {SIM_END}")
    print(SEP)
    total = len(opps)
    print(f"  {'理由':<22}  {'件数':>6}  {'割合':>7}  {'仮想P&L合計':>14}  {'仮想P&L平均':>13}")
    print("  " + "-" * 68)
    pnl_by_reason: dict[str, float] = {}
    for r in reason_order:
        cnt   = reason_cnt.get(r, 0)
        rej   = [o for o in opps if o.reason == r and o.reason != "ACCEPTED"]
        pnl_t = sum(o.virtual_pnl for o in rej)
        pnl_a = pnl_t / max(len(rej), 1)
        pnl_by_reason[r] = pnl_t
        if r == "ACCEPTED":
            ap = sum(o.actual_pnl for o in opps if o.reason == "ACCEPTED")
            print(f"  {'ACCEPTED (実エントリー)':<22}  {cnt:>6}  {cnt/max(total,1):>7.1%}  {'(actual P&L)':>14}  ¥{ap:>+12,.0f}")
        else:
            print(
                f"  {r:<22}  {cnt:>6}  {cnt/max(total,1):>7.1%}"
                f"  ¥{pnl_t:>+13,.0f}  ¥{pnl_a:>+12,.0f}"
            )
    print(f"  {'TOTAL':<22}  {total:>6}  {'100%':>7}")

    # ── 2. 機会損失ランキング (上位20) ─────────────────────────────────────
    rejected = [o for o in opps if o.reason != "ACCEPTED" and o.virtual_pnl != 0]
    rejected.sort(key=lambda x: x.virtual_pnl, reverse=True)

    print()
    print(SEP)
    print("  機会損失ランキング TOP20 (仮想P&L 降順 = 最大missed gain)")
    print(SEP)
    print(f"  {'#':>3} {'date':<11} {'sym':<8} {'sector':<10} {'RSR':>5}"
          f" {'reason':<18} {'entry':>7} {'exit':>7} {'qty':>5}"
          f" {'v_pnl':>10} {'v_ret':>7} {'days':>5} {'bind':<8} {'x_reason'}")
    print("  " + "-" * 115)
    for rank, o in enumerate(rejected[:20], 1):
        print(
            f"  {rank:>3} {o.date:<11} {o.symbol:<8} {o.sector:<10} {o.rsr:>5.1f}"
            f" {o.reason:<18} {o.signal_close:>7,.0f} {o.exit_price:>7,.0f}"
            f" {o.qty_final:>5} ¥{o.virtual_pnl:>+9,.0f} {o.virtual_ret:>+7.2%}"
            f" {o.hold_days:>5}d {o.binding:<8} {o.exit_reason}"
        )

    # ── 3. 機会損失ランキング BOTTOM20 (最も「拒否して正解」) ──────────────
    rejected.sort(key=lambda x: x.virtual_pnl)
    print()
    print(SEP)
    print("  「拒否して正解」ランキング TOP20 (仮想P&L 昇順 = 最大avoided loss)")
    print(SEP)
    print(f"  {'#':>3} {'date':<11} {'sym':<8} {'sector':<10} {'RSR':>5}"
          f" {'reason':<18} {'entry':>7} {'exit':>7} {'qty':>5}"
          f" {'v_pnl':>10} {'v_ret':>7} {'days':>5} {'bind':<8} {'x_reason'}")
    print("  " + "-" * 115)
    for rank, o in enumerate(rejected[:20], 1):
        print(
            f"  {rank:>3} {o.date:<11} {o.symbol:<8} {o.sector:<10} {o.rsr:>5.1f}"
            f" {o.reason:<18} {o.signal_close:>7,.0f} {o.exit_price:>7,.0f}"
            f" {o.qty_final:>5} ¥{o.virtual_pnl:>+9,.0f} {o.virtual_ret:>+7.2%}"
            f" {o.hold_days:>5}d {o.binding:<8} {o.exit_reason}"
        )

    # ── 4. 理由別 × 銘柄別集計 ──────────────────────────────────────────────
    print()
    print(SEP)
    print("  理由別 × 銘柄別 仮想P&L 集計 (上位10)")
    print(SEP)
    for r in [r for r in reason_order if r != "ACCEPTED"]:
        grp = [o for o in opps if o.reason == r]
        if not grp:
            continue
        by_sym: dict[str, list[float]] = {}
        for o in grp:
            by_sym.setdefault(o.symbol, []).append(o.virtual_pnl)
        sym_pnl = {s: sum(v) for s, v in by_sym.items()}
        top = sorted(sym_pnl.items(), key=lambda x: -x[1])[:5]
        total_pnl = sum(sym_pnl.values())
        print(f"  {r:<18}  total=¥{total_pnl:>+12,.0f}  n={len(grp)}")
        for sym, pnl in top:
            cnt2 = len(by_sym[sym])
            print(f"    {sym:<8}  ¥{pnl:>+10,.0f}  ({cnt2}回)")

    # ── 5. 月別拒否件数の推移 ───────────────────────────────────────────────
    print()
    print(SEP)
    print("  月別 BUY候補 × 拒否理由 推移")
    print(SEP)
    months_set = sorted(set(o.date[:7] for o in opps))
    print(f"  {'月':<8}  {'総候補':>6}  {'ACCEPT':>7}  {'CB':>7}  {'SLOT':>7}  {'WEIGHT':>7}  {'RISK':>7}  {'HOLD':>7}  {'v_pnl合計':>12}")
    print("  " + "-" * 80)
    for m in months_set:
        mo = [o for o in opps if o.date[:7] == m]
        ac = sum(1 for o in mo if o.reason == "ACCEPTED")
        cb = sum(1 for o in mo if o.reason == "CB_ACTIVE")
        sl = sum(1 for o in mo if o.reason == "SLOT_FULL")
        wt = sum(1 for o in mo if o.reason == "WEIGHT_LIMIT")
        ri = sum(1 for o in mo if o.reason == "RISK_LIMIT")
        ho = sum(1 for o in mo if o.reason == "ALREADY_HOLDING")
        vp = sum(o.virtual_pnl for o in mo if o.reason != "ACCEPTED")
        print(f"  {m:<8}  {len(mo):>6}  {ac:>7}  {cb:>7}  {sl:>7}  {wt:>7}  {ri:>7}  {ho:>7}  ¥{vp:>+11,.0f}")

    # ── 6. 集計最終結論 ──────────────────────────────────────────────────────
    all_rejected = [o for o in opps if o.reason != "ACCEPTED"]
    total_missed_gain  = sum(o.virtual_pnl for o in all_rejected if o.virtual_pnl > 0)
    total_avoided_loss = sum(o.virtual_pnl for o in all_rejected if o.virtual_pnl < 0)
    net_opp_cost       = sum(o.virtual_pnl for o in all_rejected)
    accepted_pnl       = sum(o.actual_pnl for o in opps if o.reason == "ACCEPTED")

    print()
    print(SEP)
    print("  最終集計")
    print(SEP)
    print(f"  シミュレーション期間  : {SIM_START} 〜 {SIM_END}")
    print(f"  BRIDGE_AFTER          : ¥{BRIDGE_AFTER:>12,.0f}  (実資産同期後)")
    print(f"  総BUY候補数           : {total:>12,}")
    print(f"  実エントリー数        : {reason_cnt.get('ACCEPTED',0):>12,}")
    print(f"  拒否合計              : {len(all_rejected):>12,}")
    print()
    print(f"  missed gain (拒否→後上昇): ¥{total_missed_gain:>+12,.0f}")
    print(f"  avoided loss(拒否→後下落): ¥{total_avoided_loss:>+12,.0f}")
    print(f"  net 機会損失           : ¥{net_opp_cost:>+12,.0f}")
    print(f"  実エントリー P&L      : ¥{accepted_pnl:>+12,.0f}")
    print()
    # 最大ボトルネック
    rej_pnl = {r: sum(o.virtual_pnl for o in opps if o.reason == r and o.virtual_pnl > 0)
               for r in reason_order if r != "ACCEPTED"}
    worst_r = max(rej_pnl, key=rej_pnl.get)
    print(f"  最大機会損失理由: {worst_r}  (missed gain ¥{rej_pnl[worst_r]:>+,.0f})")
    for r, pnl in sorted(rej_pnl.items(), key=lambda x: -x[1]):
        cnt2 = reason_cnt.get(r, 0)
        print(f"    {r:<18} ¥{pnl:>+11,.0f}  ({cnt2}回)")

    # ── CSV ───────────────────────────────────────────────────────────────────
    out = ROOT / "logs" / "opportunity_cost_audit.csv"
    out.parent.mkdir(exist_ok=True)
    fields = ["date","symbol","sector","signal_close","atr20","rsr","reason",
              "qty_cap","qty_risk","qty_final","binding",
              "exit_date","exit_price","virtual_pnl","virtual_ret",
              "hold_days","exit_reason","actual_pnl"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in opps:
            w.writerow({f: getattr(o, f) for f in fields})
    print(f"\n  CSV: {out}  ({len(opps)} rows)")


if __name__ == "__main__":
    main()
