"""
replacement_opportunity_audit.py — Replacement Opportunity Audit

SLOT_FULL 発生時：「保有継続 vs 弱者入替」を全イベントでシミュレーション

弱者 = 当日の RSR が最も低い保有銘柄
入替シナリオ = 弱者を当日終値で売却 → 新規候補を当日終値で購入
比較期間     = 弱者の仮想保有期間と新規候補の保有期間（同じ開始日）

出力:
  1. 乗り換え勝率
  2. CAGR/Sharpe/MaxDD 改善量（2戦略のポートフォリオシミュレーション比較）
  3. 銘柄別ランキング
  4. セクター別ランキング
  5. 結論: max_positions増 vs 早期入替
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

CAPITAL       = 3_000_000
CASH_BUFFER   = 0.10
MSW           = 0.25
RISK_PCT      = 0.0125
MAX_POS       = 3
LOT           = 100
COST          = 0.001 + 0.00055
MIN_RSR       = 75.0
ENTRY_N       = 20
EXIT_N        = 55
MAX_HOLD      = 60
MAX_DD_LIMIT  = 0.15
BRIDGE        = CAPITAL * (1 - CASH_BUFFER)

SIM_START = "2025-01-01"
SIM_END   = "2026-06-02"
TOPIX_SYM = "1306.T"

# ── データロード ──────────────────────────────────────────────────────────────
def _parse(df):
    cc = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df[[c for c in [cc,"High","Low","Volume"] if c in df.columns]].copy()
    df.columns = [{"Adj Close":"Close"}.get(c,c) for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    return df[~df.index.duplicated()].sort_index()

def load_sym(sym):
    bd = BD_DIR/f"{sym}.parquet"; ca = CACHE_DIR/f"{sym}.parquet"
    if bd.exists():
        df = _parse(pd.read_parquet(bd))
        if ca.exists():
            dc = _parse(pd.read_parquet(ca))
            new = dc[dc.index > df.index[-1]]
            if not new.empty:
                common = [c for c in df.columns if c in new.columns]
                df = pd.concat([df[common], new[common]])
        return df
    if ca.exists(): return _parse(pd.read_parquet(ca))
    return None

def load_universe():
    from src.paths import LIVE_UNIVERSE_FILE
    d = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding="utf-8"))
    return d.get("symbols", d)

def build_panels(uni):
    cl_d, hi_d, lo_d = {}, {}, {}
    for sym in uni:
        df = load_sym(sym)
        if df is not None and len(df) > 100:
            cl_d[sym]=df["Close"]; hi_d[sym]=df["High"]; lo_d[sym]=df["Low"]
    cl = pd.DataFrame(cl_d).sort_index()
    hi = pd.DataFrame(hi_d).sort_index()
    lo = pd.DataFrame(lo_d).sort_index()
    atr_d = {}
    for sym in cl.columns:
        tr = pd.concat([hi[sym]-lo[sym],(hi[sym]-cl[sym].shift()).abs(),
                        (lo[sym]-cl[sym].shift()).abs()],axis=1).max(axis=1)
        atr_d[sym] = tr.rolling(20).mean()
    atr = pd.DataFrame(atr_d)
    tp  = load_sym(TOPIX_SYM)
    topix = tp["Close"].reindex(cl.index).ffill()
    return cl, hi, lo, atr, topix

def rsr42(cl, topix):
    return (cl.pct_change(42).sub(topix.pct_change(42),axis=0)
              .rank(axis=1, pct=True)*100)

# ── 出口計算 ──────────────────────────────────────────────────────────────────
def calc_exit(sym, entry_idx, entry_price, dates, lo, cl):
    """entry_idx 以降の最初の exit を返す。(exit_price, exit_days, exit_reason)"""
    if sym not in cl.columns:
        return entry_price, 0, "no_data"
    for k in range(1, MAX_HOLD + 1):
        j = entry_idx + k
        if j >= len(dates):
            break
        dt = dates[j]
        p = float(cl[sym].get(dt, entry_price))
        if np.isnan(p): p = entry_price
        lo_hist = lo[sym].loc[:dt].iloc[-EXIT_N-1:-1] if sym in lo.columns else pd.Series(dtype=float)
        stop = float(lo_hist.min()) if not lo_hist.empty else 0.0
        if p <= stop:
            return p, k, "turtle"
    # time_stop
    last_j = min(entry_idx + MAX_HOLD, len(dates)-1)
    p = float(cl[sym].get(dates[last_j], entry_price))
    if np.isnan(p): p = entry_price
    return p, MAX_HOLD, "time_stop"

# ── イベント構造 ───────────────────────────────────────────────────────────────
@dataclass
class HoldInfo:
    sym: str; sector: str
    entry_price: float; entry_idx: int
    entry_rsr: float     # RSR at entry date
    current_rsr: float   # RSR at SLOT_FULL date
    unrealized_pct: float
    hold_days: int

@dataclass
class ReplaceEvent:
    date: str; date_idx: int
    candidate: str; cand_sector: str
    cand_price: float; cand_rsr: float; cand_atr: float
    holdings: list[HoldInfo]
    weakest_sym: str     # symbol of weakest holding
    # per-event forward P&L
    keep_weak_pnl: float   = 0.0   # weakest if kept further
    repl_cand_pnl: float   = 0.0   # candidate P&L from same date
    replace_wins: bool     = False
    keep_weak_ret: float   = 0.0
    repl_cand_ret: float   = 0.0
    keep_weak_exit_days: int = 0
    repl_exit_days: int  = 0

# ── メインシミュレーション ─────────────────────────────────────────────────────
def simulate(uni, cl, hi, lo, atr, rsr):
    """
    2 ストラテジーを並行シミュレーション:
      A: 現行 (SLOT_FULL = 拒否)
      B: 弱者入替 (SLOT_FULL → weakest を新候補と入れ替え)
    """
    dates = cl.loc[SIM_START:SIM_END].index
    pos_bud  = BRIDGE * MSW
    risk_bud = BRIDGE * RISK_PCT

    def sizing(price, av):
        lc = price * LOT
        qc = int(pos_bud // lc) * LOT
        qr = max(LOT, (int(risk_bud / max(av,0.01)) // LOT) * LOT)
        return min(qr, qc) if qc > 0 else 0

    # ── Strategy A: 現行 ─────────────────────────────────────────────────────
    cashA = float(CAPITAL); peakA = float(CAPITAL)
    posA: dict[str, dict]  = {}   # {sym: {ep, qty, idx, entry_rsr}}
    eqA: list[float]       = []

    # ── Strategy B: 弱者入替 ───────────────────────────────────────────────
    cashB = float(CAPITAL); peakB = float(CAPITAL)
    posB: dict[str, dict]  = {}
    eqB: list[float]       = []

    events: list[ReplaceEvent] = []
    cb_active = False; cb_end: date | None = None

    for i, dt in enumerate(dates):
        cl_row = cl.loc[dt]

        # ── Equity 更新 ────────────────────────────────────────────────────
        for pos_x, eq_x, cash_x in [(posA, eqA, cashA),(posB, eqB, cashB)]:
            mv = sum(pos_x[s]["qty"]*float(cl_row.get(s, pos_x[s]["ep"])) for s in pos_x)
            eq = cash_x + mv
            eq_x.append(eq)
        # Use Strategy A equity for CB
        eq_now = eqA[-1]
        peakA = max(peakA, eq_now); peakB = max(peakB, eqB[-1])
        dd = (eq_now - peakA) / peakA if peakA > 0 else 0.0
        if dd < -MAX_DD_LIMIT and not cb_active:
            cb_active = True; cb_end = dt.date() + timedelta(days=60)
        if cb_active and cb_end and dt.date() >= cb_end:
            if eq_now >= peakA*(1-MAX_DD_LIMIT): cb_active=False; cb_end=None

        # ── Exit 両ストラテジー ─────────────────────────────────────────────
        for pos_x, cash_ref in [(posA,"A"),(posB,"B")]:
            to_ex = []
            for sym, p in pos_x.items():
                price = float(cl_row.get(sym, p["ep"]))
                if np.isnan(price): price = p["ep"]
                lo_hist = lo[sym].loc[:dt].iloc[-EXIT_N-1:-1] if sym in lo.columns else pd.Series(dtype=float)
                stop = float(lo_hist.min()) if not lo_hist.empty else 0.0
                held = i - p["idx"]
                if price <= stop or held >= MAX_HOLD:
                    to_ex.append((sym, price))
            for sym, price in to_ex:
                p = pos_x.pop(sym)
                if pos_x is posA: cashA += p["qty"]*price*(1-COST)
                else:             cashB += p["qty"]*price*(1-COST)

        # ── シグナル判定 ────────────────────────────────────────────────────
        if dt not in rsr.index: continue
        rsr_row = rsr.loc[dt].dropna()
        cands   = rsr_row[rsr_row>=MIN_RSR].sort_values(ascending=False)
        new_A = 0; new_B = 0

        for sym in cands.index:
            if sym not in cl.columns or sym not in hi.columns: continue
            price = float(cl_row.get(sym, np.nan))
            if np.isnan(price) or price<=0: continue
            hi_hist = hi[sym].loc[:dt].iloc[-ENTRY_N-1:-1]
            if hi_hist.empty or price<=float(hi_hist.max()): continue
            av = atr[sym].get(dt, np.nan) if sym in atr.columns else np.nan
            if np.isnan(av) or av<=0: av = price*0.02
            rsrv = float(rsr_row.get(sym, 0))
            qty  = sizing(price, av)

            # ── Strategy A ─────────────────────────────────────────────────
            a_held = sym in posA
            a_full = len(posA) >= MAX_POS
            if not a_held and not a_full and not cb_active and qty>0 and new_A==0:
                cashA -= qty*price*(1+COST)
                posA[sym] = {"ep":price,"qty":qty,"idx":i,"entry_rsr":rsrv}
                new_A += 1
            elif not a_held and a_full and not cb_active and qty>0:
                # SLOT_FULL event → record replacement event
                holdings_info = []
                for hs, hp in posA.items():
                    h_price = float(cl_row.get(hs, hp["ep"]))
                    if np.isnan(h_price): h_price = hp["ep"]
                    h_rsr = float(rsr_row.get(hs, 0)) if hs in rsr_row.index else 0.0
                    holdings_info.append(HoldInfo(
                        sym=hs, sector=uni.get(hs,""),
                        entry_price=hp["ep"], entry_idx=hp["idx"],
                        entry_rsr=hp.get("entry_rsr",0),
                        current_rsr=h_rsr,
                        unrealized_pct=(h_price-hp["ep"])/max(hp["ep"],1),
                        hold_days=i-hp["idx"],
                    ))
                # weakest by current RSR
                holdings_info.sort(key=lambda h: h.current_rsr)
                weakest = holdings_info[0]
                ev = ReplaceEvent(
                    date=str(dt.date()), date_idx=i,
                    candidate=sym, cand_sector=uni.get(sym,""),
                    cand_price=price, cand_rsr=rsrv, cand_atr=round(av,1),
                    holdings=holdings_info, weakest_sym=weakest.sym,
                )
                # Forward P&L: weakest if kept
                weak_xp, weak_days, _ = calc_exit(weakest.sym, i, weakest.entry_price, dates, lo, cl)
                weak_cur = float(cl_row.get(weakest.sym, weakest.entry_price))
                if np.isnan(weak_cur): weak_cur = weakest.entry_price
                ev.keep_weak_pnl = (weak_xp - weak_cur) * posA[weakest.sym]["qty"] * (1-2*COST)
                ev.keep_weak_ret = (weak_xp - weak_cur)/max(weak_cur,1)
                ev.keep_weak_exit_days = weak_days
                # Forward P&L: candidate over same period
                # Use weakest's exit horizon as candidate exit horizon too
                cand_xp, cand_days, _ = calc_exit(sym, i, price, dates, lo, cl)
                ev.repl_cand_pnl = (cand_xp - price) * qty * (1-2*COST)
                ev.repl_cand_ret  = (cand_xp - price)/max(price,1)
                ev.repl_exit_days = cand_days
                # Win condition: candidate outperforms weakest (return-based)
                ev.replace_wins = ev.repl_cand_ret > ev.keep_weak_ret
                events.append(ev)

            # ── Strategy B ─────────────────────────────────────────────────
            b_held = sym in posB
            b_full = len(posB) >= MAX_POS
            if not b_held and not b_full and not cb_active and qty>0 and new_B==0:
                cashB -= qty*price*(1+COST)
                posB[sym] = {"ep":price,"qty":qty,"idx":i,"entry_rsr":rsrv}
                new_B += 1
            elif not b_held and b_full and not cb_active and qty>0:
                # Replace weakest in B
                b_rsrs = {s: float(rsr_row.get(s,0)) if s in rsr_row.index else 0.0
                          for s in posB}
                weakest_b = min(b_rsrs, key=b_rsrs.get)
                if rsrv > b_rsrs[weakest_b]:   # only replace if new is stronger
                    w_price = float(cl_row.get(weakest_b, posB[weakest_b]["ep"]))
                    if np.isnan(w_price): w_price = posB[weakest_b]["ep"]
                    cashB += posB[weakest_b]["qty"]*w_price*(1-COST)
                    del posB[weakest_b]
                    cashB -= qty*price*(1+COST)
                    posB[sym] = {"ep":price,"qty":qty,"idx":i,"entry_rsr":rsrv}
                    new_B += 1

    # Close remaining positions
    final_dt = dates[-1]
    for pos_x, cash_ref in [(posA,"A"),(posB,"B")]:
        for sym, p in list(pos_x.items()):
            pr = float(cl[sym].dropna().iloc[-1]) if sym in cl.columns else p["ep"]
            if np.isnan(pr): pr = p["ep"]
            if pos_x is posA: cashA += p["qty"]*pr*(1-COST)
            else:             cashB += p["qty"]*pr*(1-COST)

    eqA_s = pd.Series(eqA, index=dates)
    eqB_s = pd.Series(eqB, index=dates)
    return events, eqA_s, eqB_s

# ── メトリクス計算 ─────────────────────────────────────────────────────────────
def metrics(eq: pd.Series, label: str) -> dict:
    yr    = len(eq) / 252
    fin   = eq.iloc[-1]
    cagr  = (fin/CAPITAL)**(1/max(yr,.01)) - 1
    dd    = eq/eq.cummax() - 1
    mdd   = float(dd.min())
    dr    = eq.pct_change().dropna()
    sh    = float(dr.mean()/dr.std()*252**.5) if dr.std()>0 else 0.0
    cal   = cagr/abs(mdd) if mdd<0 else 0.0
    return {"label":label,"cagr":cagr,"sharpe":sh,"max_dd":mdd,"calmar":cal,
            "final":fin,"years":round(yr,2)}

# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    uni  = load_universe()
    cl, hi, lo, atr, topix = build_panels(uni)
    print(f"  Symbols:{len(cl.columns)}  Range:{cl.index[0].date()}-{cl.index[-1].date()}")
    rsr = rsr42(cl, topix)

    print(f"  Simulating {SIM_START}→{SIM_END} (Strategy A=keep / B=replace_weakest)")
    events, eqA, eqB = simulate(uni, cl, hi, lo, atr, rsr)
    print(f"  SLOT_FULL replacement events: {len(events)}")

    SEP = "=" * 76

    # ── 1. 乗り換え勝率 ──────────────────────────────────────────────────────
    n_ev   = len(events)
    n_win  = sum(1 for e in events if e.replace_wins)
    n_lose = n_ev - n_win
    win_rate = n_win / max(n_ev, 1)

    mA = metrics(eqA, "A: Keep Existing")
    mB = metrics(eqB, "B: Replace Weakest")

    print()
    print(SEP)
    print(f"  1. 乗り換え勝率  ({SIM_START} 〜 {SIM_END})")
    print(SEP)
    print(f"  SLOT_FULL イベント数 : {n_ev}")
    print(f"  乗り換え優位       : {n_win}件  ({win_rate:.1%})")
    print(f"  保有継続優位       : {n_lose}件  ({n_lose/max(n_ev,1):.1%})")
    print()
    print(f"  平均 Δreturn (candidate - weakest):")
    avg_dr = np.mean([e.repl_cand_ret - e.keep_weak_ret for e in events]) if events else 0
    std_dr = np.std([e.repl_cand_ret  - e.keep_weak_ret for e in events]) if events else 0
    print(f"    mean={avg_dr:>+.4f}  std={std_dr:.4f}")
    print(f"    positive events={sum(1 for e in events if e.repl_cand_ret>e.keep_weak_ret)}/{n_ev}")

    # ── 2. CAGR/Sharpe/MaxDD 比較 ────────────────────────────────────────────
    print()
    print(SEP)
    print("  2. ポートフォリオシミュレーション比較")
    print(SEP)
    print(f"  {'指標':<22}  {'A:Keep':>12}  {'B:Replace':>12}  {'Δ(B-A)':>12}")
    print("  " + "-" * 64)
    for key, is_pct, is_yen in [("cagr",True,False),("sharpe",False,False),
                                 ("max_dd",True,False),("calmar",False,False),("final",False,True)]:
        va = mA[key]; vb = mB[key]; dv = vb - va
        if is_yen:
            print(f"  {'final_equity':<22}  ¥{va:>11,.0f}  ¥{vb:>11,.0f}  ¥{dv:>+11,.0f}")
        elif is_pct:
            print(f"  {key:<22}  {va:>12.2%}  {vb:>12.2%}  {dv:>+12.2%}")
        else:
            print(f"  {key:<22}  {va:>12.3f}  {vb:>12.3f}  {dv:>+12.3f}")

    # ── 3. 銘柄別ランキング ──────────────────────────────────────────────────
    # (a) 新規候補として最も乗り換え価値が高かった銘柄
    from collections import defaultdict
    cand_stats: dict[str,list] = defaultdict(list)
    weak_stats: dict[str,list] = defaultdict(list)
    for e in events:
        cand_stats[e.candidate].append((e.replace_wins, e.repl_cand_ret, e.repl_cand_pnl))
        weak_stats[e.weakest_sym].append((e.replace_wins, e.keep_weak_ret, e.keep_weak_pnl))

    print()
    print(SEP)
    print("  3a. 新規候補ランキング (乗り換え優位率 × missed P&L 降順)")
    print(SEP)
    print(f"  {'symbol':<10} {'sector':<10} {'n':>4} {'win%':>6} {'avg_ret':>9} {'total_missed_pnl':>17} {'avg_rsr':>8}")
    print("  " + "-" * 72)
    cand_rank = []
    for sym, stats in cand_stats.items():
        wins = sum(1 for s in stats if s[0])
        avg_ret = np.mean([s[1] for s in stats])
        total_pnl = sum(s[2] for s in stats)
        ev_rsr = [e.cand_rsr for e in events if e.candidate == sym]
        avg_rsr = np.mean(ev_rsr) if ev_rsr else 0
        cand_rank.append((sym, uni.get(sym,""), len(stats), wins/len(stats),
                          avg_ret, total_pnl, avg_rsr))
    cand_rank.sort(key=lambda x: -x[5])
    for row in cand_rank[:15]:
        sym, sec, n, wr, ar, tp, rsr_v = row
        print(f"  {sym:<10} {sec:<10} {n:>4} {wr:>6.1%} {ar:>+9.2%} ¥{tp:>+16,.0f} {rsr_v:>8.1f}")

    print()
    print(SEP)
    print("  3b. 弱者候補ランキング (最も「入れ替えるべきだった」保有銘柄)")
    print(SEP)
    print(f"  {'symbol':<10} {'sector':<10} {'n':>4} {'replaced%':>10} {'avg_weak_ret':>13} {'total_keep_pnl':>15}")
    print("  " + "-" * 72)
    weak_rank = []
    for sym, stats in weak_stats.items():
        replaced = sum(1 for s in stats if s[0])
        avg_ret  = np.mean([s[1] for s in stats])
        total_pnl= sum(s[2] for s in stats)
        weak_rank.append((sym, uni.get(sym,""), len(stats), replaced/len(stats),
                          avg_ret, total_pnl))
    weak_rank.sort(key=lambda x: x[4])  # worst performance first
    for row in weak_rank[:15]:
        sym, sec, n, rp, ar, tp = row
        print(f"  {sym:<10} {sec:<10} {n:>4} {rp:>10.1%} {ar:>+13.2%} ¥{tp:>+14,.0f}")

    # ── 4. セクター別ランキング ──────────────────────────────────────────────
    sec_cand: dict[str,list] = defaultdict(list)
    sec_weak: dict[str,list] = defaultdict(list)
    for e in events:
        sec_cand[e.cand_sector].append((e.replace_wins, e.repl_cand_ret))
        sec_weak[uni.get(e.weakest_sym,"")].append((e.replace_wins, e.keep_weak_ret))

    print()
    print(SEP)
    print("  4. セクター別ランキング")
    print(SEP)
    print(f"  {'sector':<12} {'as_cand n':>10} {'cand_winrate':>13} {'cand_avg_ret':>13}"
          f" {'as_weak n':>10} {'weak_avg_ret':>13}")
    print("  " + "-" * 74)
    all_secs = sorted(set(list(sec_cand.keys())+list(sec_weak.keys())))
    for sec in all_secs:
        cn = sec_cand.get(sec, [])
        wn = sec_weak.get(sec, [])
        cwr = sum(1 for s in cn if s[0])/max(len(cn),1)
        car = np.mean([s[1] for s in cn]) if cn else 0
        war = np.mean([s[1] for s in wn]) if wn else 0
        print(f"  {sec:<12} {len(cn):>10} {cwr:>13.1%} {car:>+13.2%}"
              f" {len(wn):>10} {war:>+13.2%}")

    # ── 5. イベント詳細サマリー ──────────────────────────────────────────────
    # 月別 win rate
    print()
    print(SEP)
    print("  5. 月別 乗り換え勝率 推移")
    print(SEP)
    print(f"  {'月':<8} {'n':>5} {'win':>5} {'win%':>7} {'avg_cand_ret':>14} {'avg_weak_ret':>14} {'avg_Δret':>10}")
    print("  " + "-" * 68)
    months = sorted(set(e.date[:7] for e in events))
    for m in months:
        me = [e for e in events if e.date[:7]==m]
        mw = sum(1 for e in me if e.replace_wins)
        acr = np.mean([e.repl_cand_ret  for e in me])
        awr = np.mean([e.keep_weak_ret  for e in me])
        adr = np.mean([e.repl_cand_ret - e.keep_weak_ret for e in me])
        print(f"  {m:<8} {len(me):>5} {mw:>5} {mw/len(me):>7.1%} {acr:>+14.2%} {awr:>+14.2%} {adr:>+10.2%}")

    # ── 6. 最終結論 ──────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  最終結論")
    print(SEP)

    # Key stats
    replace_cagr_delta = mB["cagr"] - mA["cagr"]
    replace_dd_delta   = mB["max_dd"] - mA["max_dd"]
    replace_sh_delta   = mB["sharpe"] - mA["sharpe"]

    # max_positions+1 の期待値 (全 SLOT_FULL イベントの仮想P&L 平均)
    all_missed_ret = [e.repl_cand_ret for e in events if e.repl_cand_pnl>0]
    avg_missed_ret_if_entered = np.mean(all_missed_ret) if all_missed_ret else 0

    print(f"  [乗り換え戦略 B vs 現行 A]")
    print(f"    勝率              : {win_rate:.1%}  ({n_win}/{n_ev}件)")
    print(f"    CAGR 改善量       : {replace_cagr_delta:>+.3%}")
    print(f"    Sharpe 改善量     : {replace_sh_delta:>+.4f}")
    print(f"    MaxDD 変化        : {replace_dd_delta:>+.3%}")
    print()
    print(f"  [参考: もし max_positions=4 なら]")
    print(f"    全 SLOT_FULL 候補の平均リターン: {avg_missed_ret_if_entered:>+.2%}")
    top_missed = sorted(events, key=lambda e: -e.repl_cand_pnl)[:3]
    print(f"    Top3 missed gain (最大機会損失) →")
    for e in top_missed:
        print(f"      {e.date} {e.candidate:<8} +{e.repl_cand_ret:.1%}"
              f"  P&L≈¥{e.repl_cand_pnl:>+,.0f}")
    print()
    # 判定
    if win_rate >= 0.55 and replace_cagr_delta > 0.01:
        verdict = "早期入替有利"
        detail  = (f"乗り換え勝率{win_rate:.0%} + CAGR+{replace_cagr_delta:.2%}pt。"
                   "弱者入替を積極採用する価値がある。")
    elif avg_missed_ret_if_entered > 0.10:
        verdict = "max_positions増加有利"
        detail  = (f"SLOT_FULL候補の平均リターン{avg_missed_ret_if_entered:.1%}は高いが、"
                   f"乗り換え勝率{win_rate:.0%}は低い。既存保有も強く、売る必要はない。"
                   " 4スロット化で両取りが最善。")
    else:
        verdict = "現行維持 (差異軽微)"
        detail  = "改善幅が小さく、変更コストが見合わない。"

    print(f"  ★ 判定: {verdict}")
    print(f"    {detail}")
    print()
    print(f"  補足:")
    print(f"    現行(A) CAGR={mA['cagr']:>+.2%}  最終資産 ¥{mA['final']:>,.0f}")
    print(f"    入替(B) CAGR={mB['cagr']:>+.2%}  最終資産 ¥{mB['final']:>,.0f}")
    print(f"    入替適用: {sum(1 for e in events if e.replace_wins)}件のみ優位。{n_ev-sum(1 for e in events if e.replace_wins)}件は現行が優位。")

    # ── CSV 出力 ─────────────────────────────────────────────────────────────
    out = ROOT/"logs"/"replacement_opportunity_audit.csv"
    out.parent.mkdir(exist_ok=True)
    fields = ["date","candidate","cand_sector","cand_rsr","cand_price","cand_atr",
              "weakest_sym","weakest_rsr","weakest_unreal_pct","weakest_hold_days",
              "keep_weak_ret","repl_cand_ret","delta_ret",
              "keep_weak_pnl","repl_cand_pnl","replace_wins"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            wk = next((h for h in e.holdings if h.sym==e.weakest_sym), None)
            w.writerow({
                "date":e.date,"candidate":e.candidate,
                "cand_sector":e.cand_sector,"cand_rsr":e.cand_rsr,
                "cand_price":e.cand_price,"cand_atr":e.cand_atr,
                "weakest_sym":e.weakest_sym,
                "weakest_rsr":wk.current_rsr if wk else 0,
                "weakest_unreal_pct":round(wk.unrealized_pct,4) if wk else 0,
                "weakest_hold_days":wk.hold_days if wk else 0,
                "keep_weak_ret":round(e.keep_weak_ret,4),
                "repl_cand_ret":round(e.repl_cand_ret,4),
                "delta_ret":round(e.repl_cand_ret-e.keep_weak_ret,4),
                "keep_weak_pnl":round(e.keep_weak_pnl,0),
                "repl_cand_pnl":round(e.repl_cand_pnl,0),
                "replace_wins":e.replace_wins,
            })
    print(f"\n  CSV: {out}  ({len(events)} events)")


if __name__ == "__main__":
    main()
