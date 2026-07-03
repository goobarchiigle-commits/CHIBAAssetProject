"""
AUTO_PROMOTE_V2 Walk-Forward Validation: P2 (UNCLASSIFIED + SI=50-89 only)
P2: Promote ONLY UNCLASSIFIED candidates with 50 <= sector_ignition < 90
    EARLY_IGNITION / CONTINUATION / MATURE_LEADER / HIGH_RSR are excluded
Robustness: leave-one-out per symbol
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import json
import statistics
import math

OHLCV_DIR  = Path("C:/ai-trading/data/ohlcv")
CAPITAL    = 3_000_000
SLIPPAGE   = 0.001
COMMISSION = 0.00055
MAX_POS    = 3
ALLOC_MUL  = 0.25   # probation allocation multiplier

SHADOW_SYMS = ["5706.T", "6146.T", "6857.T", "6920.T", "8035.T"]

LIVE_SYMS = [
    "8035.T","6702.T","6501.T","6762.T","6920.T","6857.T","6594.T",
    "7203.T","7201.T","9432.T","8306.T","8411.T","8309.T","7182.T",
    "8725.T","8058.T","8053.T","8002.T","8001.T","4021.T","2914.T",
    "7011.T","3382.T","5401.T","5411.T","9531.T","9101.T","9104.T",
]

SECTOR_MAP = {
    "8035.T":"電機精密","6645.T":"電機精密","6762.T":"電機精密",
    "6920.T":"電機精密","6857.T":"電機精密","6594.T":"電機精密",
    "6146.T":"電機精密","6479.T":"電機精密","6506.T":"電機精密",
    "6702.T":"電機","6501.T":"電機","6752.T":"電機","6758.T":"電機",
    "7011.T":"機械","6301.T":"機械","6326.T":"機械","6273.T":"機械",
    "7203.T":"輸送機器","7201.T":"輸送機器",
    "8306.T":"銀行","8411.T":"銀行","8309.T":"銀行","7182.T":"銀行",
    "8725.T":"保険","8766.T":"保険",
    "8058.T":"商社","8053.T":"商社","8002.T":"商社","8001.T":"商社",
    "4021.T":"化学","4063.T":"化学","4188.T":"化学","4452.T":"化学",
    "4901.T":"化学","3407.T":"化学","3402.T":"化学",
    "2914.T":"食品","2801.T":"食品","2802.T":"食品","2503.T":"食品",
    "9432.T":"情報通信","4307.T":"情報通信",
    "3382.T":"小売","3099.T":"小売",
    "5401.T":"鉄鋼","5411.T":"鉄鋼",
    "5706.T":"非鉄金属","5713.T":"非鉄金属","5714.T":"非鉄金属",
    "9101.T":"海運","9104.T":"海運","9107.T":"海運",
    "9531.T":"ガス",
    "4503.T":"医薬品","4507.T":"医薬品","4519.T":"医薬品",
    "4523.T":"医薬品","4543.T":"医薬品","4568.T":"医薬品",
    "1925.T":"建設","1928.T":"建設","1801.T":"建設",
    "6146.T":"電機精密",
}

WF_FOLDS = [
    {"name":"Fold1","train_end":"2020-12-31","test_start":"2021-01-01","test_end":"2021-12-31"},
    {"name":"Fold2","train_end":"2021-12-31","test_start":"2022-01-01","test_end":"2022-12-31"},
    {"name":"Fold3","train_end":"2022-12-31","test_start":"2023-01-01","test_end":"2023-12-31"},
    {"name":"Fold4","train_end":"2023-12-31","test_start":"2024-01-01","test_end":"2024-12-31"},
]

# Taxonomy thresholds (identical to production)
ML_MIN_RSR   = 87.0
ML_MAX_COMP  = 35.0
MR_MAX_RSR   = 55.0
MR_MAX_SI    = 55.0
EI_MIN_SI    = 75.0
EI_MAX_RSR   = 85.0
CT_MIN_RSR   = 70.0
CT_MIN_COMP  = 28.0
CT_MAX_RSR   = 88.0
HR_MIN_RSR   = 90.0
HR_MIN_SI    = 90.0
P2_SI_LO     = 50.0
P2_SI_HI     = 90.0  # exclusive


def load_prices(sym):
    f = OHLCV_DIR / f"{sym}.csv"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding='utf-8').splitlines()[1:]:
        p = line.split(',')
        if len(p) >= 2:
            try:
                out[p[0].strip()] = float(p[1].strip())
            except:
                pass
    return out


def all_trading_dates(prices_dict):
    dates = set()
    for p in prices_dict.values():
        dates.update(p.keys())
    return sorted(d for d in dates if "2018-01-01" <= d <= "2024-12-31")


def compute_rsr(all_prices, dates):
    monthly = dates[::21]
    out = {}
    for i, dt in enumerate(monthly):
        if i < 12:
            continue
        dt_12m = monthly[i-12]
        rets = {}
        for sym, prices in all_prices.items():
            p0, p1 = prices.get(dt_12m), prices.get(dt)
            if p0 and p1 and p0 > 0:
                rets[sym] = p1/p0 - 1
        if not rets:
            continue
        ranked = sorted(rets, key=lambda s: rets[s])
        n = len(ranked)
        out[dt] = {sym: round(rank/(n-1)*100,1) if n>1 else 50.0
                   for rank, sym in enumerate(ranked)}
    return out


def compute_si(all_prices, dates):
    monthly = dates[::21]
    out = {}
    for i, dt in enumerate(monthly):
        if i < 3:
            continue
        dt_prev = monthly[i-1]
        vel = {}
        for sym, prices in all_prices.items():
            p0, p1 = prices.get(dt_prev), prices.get(dt)
            if p0 and p1 and p0 > 0:
                vel[sym] = (p1/p0-1)*100
        breakout = {}
        for sym, prices in all_prices.items():
            prev4 = [prices.get(d) for d in monthly[max(0,i-4):i] if prices.get(d)]
            p1 = prices.get(dt)
            if prev4 and p1:
                breakout[sym] = p1 >= max(prev4)*0.98
        sectors = defaultdict(list)
        for sym in all_prices:
            sectors[SECTOR_MAP.get(sym,"_unk")].append(sym)
        sec_raw = {}
        for sec, syms in sectors.items():
            n = len(syms)
            if n == 0:
                sec_raw[sec] = 0.0
                continue
            breadth  = sum(1 for s in syms if vel.get(s,0) > 0)/n
            vels_avg = sum(vel.get(s,0) for s in syms)/n
            vel_norm = max(0.0, min(1.0, (vels_avg+2.5)/5.0))
            bo_ratio = sum(1 for s in syms if breakout.get(s,False))/n
            sec_raw[sec] = 0.35*breadth + 0.25*vel_norm + 0.25*bo_ratio + 0.15*0.5
        sym_raw = {sym: sec_raw.get(SECTOR_MAP.get(sym,"_unk"),0.0) for sym in all_prices}
        vals = sorted(sym_raw.values())
        n = len(vals)
        if n <= 1:
            out[dt] = {sym: 50.0 for sym in all_prices}
            continue
        out[dt] = {sym: round(sum(1 for x in vals if x <= v)/n*100,1)
                   for sym, v in sym_raw.items()}
    return out


def compute_comp(all_prices, dates):
    monthly = dates[::21]
    out = {}
    for i, dt in enumerate(monthly):
        if i < 6:
            continue
        comp = {}
        for sym, prices in all_prices.items():
            a20 = [prices.get(d) for d in monthly[max(0,i-2):i+1] if prices.get(d)]
            a60 = [prices.get(d) for d in monthly[max(0,i-6):i+1] if prices.get(d)]
            if len(a20) >= 2 and len(a60) >= 3:
                r20 = (max(a20)-min(a20))/min(a20) if min(a20)>0 else 0
                r60 = (max(a60)-min(a60))/min(a60) if min(a60)>0 else 0
                comp[sym] = round(max(0,min(100,(r20/r60 if r60>0 else 1)*50)),1)
            else:
                comp[sym] = 50.0
        out[dt] = comp
    return out


def classify(rsr, si, comp):
    if rsr >= ML_MIN_RSR and comp < ML_MAX_COMP:
        return "MATURE_LEADER"
    if rsr < MR_MAX_RSR and si < MR_MAX_SI:
        return "MEAN_REVERSION"
    if si >= EI_MIN_SI and rsr < EI_MAX_RSR:
        return "EARLY_IGNITION"
    if rsr >= CT_MIN_RSR and comp >= CT_MIN_COMP and rsr < CT_MAX_RSR:
        return "CONTINUATION"
    if rsr >= HR_MIN_RSR and si >= HR_MIN_SI:
        return "HIGH_RSR"
    return "UNCLASSIFIED"


def gate_p2(rsr, si, comp):
    """P2 gate: ONLY UNCLASSIFIED with 50 <= si < 90"""
    if rsr < 8.0:
        return False, classify(rsr, si, comp), "rsr_too_low"
    ctype = classify(rsr, si, comp)
    if ctype != "UNCLASSIFIED":
        return False, ctype, f"taxonomy:{ctype}"
    if si < P2_SI_LO:
        return False, ctype, f"si<{P2_SI_LO}"
    if si >= P2_SI_HI:
        return False, ctype, f"si>={P2_SI_HI}"
    return True, ctype, ""


def fwd_return(prices, dates, entry_dt, n_days):
    idx = next((i for i,d in enumerate(dates) if d >= entry_dt), None)
    if idx is None or idx+n_days >= len(dates):
        return None
    p0, p1 = prices.get(dates[idx]), prices.get(dates[idx+n_days])
    if p0 and p1 and p0 > 0:
        return (p1/p0-1) - (SLIPPAGE+COMMISSION)*2
    return None


def equity_metrics(eq_curve, ts, te):
    dates = sorted(d for d in eq_curve if ts <= d <= te)
    if len(dates) < 50:
        return {"cagr":0,"max_dd":0,"sharpe":0,"calmar":0,"n_days":len(dates)}
    vals = [eq_curve[d] for d in dates]
    years = len(dates)/252
    cagr = (vals[-1]/vals[0])**(1/years) - 1
    peak, max_dd = vals[0], 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v-peak)/peak
        if dd < max_dd:
            max_dd = dd
    rets = [vals[i]/vals[i-1]-1 for i in range(1,len(vals))]
    avg = sum(rets)/len(rets)
    std = statistics.stdev(rets) if len(rets)>1 else 1e-6
    sharpe = avg*252/(std*math.sqrt(252)) if std>0 else 0
    calmar = abs(cagr/max_dd) if max_dd!=0 else 0
    return {"cagr":round(cagr*100,2),"max_dd":round(max_dd*100,2),
            "sharpe":round(sharpe,3),"calmar":round(calmar,3),"n_days":len(dates)}


def simulate_baseline(all_prices, live_syms, dates, rsr_all):
    cash = float(CAPITAL)
    positions = {}
    eq = {}
    monthly = set(rsr_all.keys())
    for dt in dates:
        mval = cash + sum(pos['qty']*all_prices.get(sym,{}).get(dt,pos['ep'])
                          for sym,pos in positions.items())
        eq[dt] = mval
        if dt not in monthly:
            continue
        rsr = rsr_all.get(dt,{})
        to_close = [sym for sym,pos in positions.items()
                    if rsr.get(sym,50)<70 or
                    sum(1 for d in dates if pos['ed']<=d<=dt)>60]
        for sym in to_close:
            pos = positions.pop(sym)
            p = all_prices.get(sym,{}).get(dt,pos['ep'])
            cash += pos['qty']*p*(1-SLIPPAGE-COMMISSION)
        slots = MAX_POS - len(positions)
        if slots <= 0:
            continue
        buys = sorted([(s,rsr.get(s,0)) for s in live_syms
                       if s not in positions and rsr.get(s,0)>=75 and all_prices.get(s,{}).get(dt)],
                      key=lambda x:-x[1])
        for sym,_ in buys[:slots]:
            alloc = cash/(slots+len(positions)+1)
            p = all_prices[sym][dt]
            qty = int(alloc/(p*(1+SLIPPAGE+COMMISSION)))
            if qty <= 0:
                continue
            cash -= qty*p*(1+SLIPPAGE+COMMISSION)
            positions[sym] = {'ep':p,'qty':qty,'ed':dt}
    return eq


def apply_overlay(eq_base, promotions, all_prices, dates):
    eq = dict(eq_base)
    for pr in promotions:
        if pr['fwd30'] is None:
            continue
        sym = pr['sym']
        idx = next((i for i,d in enumerate(dates) if d >= pr['date']),None)
        if idx is None:
            continue
        exit_idx = min(idx+30, len(dates)-1)
        alloc = eq.get(dates[idx], CAPITAL) * (ALLOC_MUL/MAX_POS)
        p0 = all_prices.get(sym,{}).get(dates[idx],0)
        p1 = all_prices.get(sym,{}).get(dates[exit_idx],0)
        if p0>0 and p1>0:
            pnl = alloc*(p1/p0-1) - alloc*(SLIPPAGE+COMMISSION)*2
            exit_dt = dates[exit_idx]
            for d in dates:
                if d >= exit_dt:
                    eq[d] = eq.get(d, CAPITAL) + pnl
    return eq


def run_p2_fold(shadow_syms, all_prices, dates, rsr_all, si_all, comp_all,
                monthly_dates, eq_base, fold):
    ts, te = fold['test_start'], fold['test_end']
    test_monthly = [d for d in monthly_dates if ts <= d <= te]
    all_evals = []
    type_counts = defaultdict(int)
    reject_detail = defaultdict(int)

    for dt in test_monthly:
        rsr = rsr_all.get(dt,{})
        si  = si_all.get(dt,{})
        comp= comp_all.get(dt,{})
        for sym in shadow_syms:
            rv, sv, cv = rsr.get(sym,0), si.get(sym,0), comp.get(sym,50)
            passed, ctype, reason = gate_p2(rv, sv, cv)
            type_counts[ctype] += 1
            all_evals.append({"sym":sym,"date":dt,"rsr":rv,"si":sv,"comp":cv,
                               "type":ctype,"passed":passed,"reason":reason})
            if not passed:
                reject_detail[reason] += 1

    # Dedup: one per symbol per >=2 monthly windows
    passed_evals = [e for e in all_evals if e['passed']]
    seen = {}
    promotions = []
    for ev in sorted(passed_evals, key=lambda x: x['date']):
        sym = ev['sym']
        prev = seen.get(sym)
        if prev is None or sum(1 for d in test_monthly if prev < d <= ev['date']) >= 2:
            f30 = fwd_return(all_prices[sym], dates, ev['date'], 30)
            f60 = fwd_return(all_prices[sym], dates, ev['date'], 60)
            promotions.append({
                "sym":sym,"date":ev['date'],"type":ev['type'],
                "rsr":ev['rsr'],"si":ev['si'],"comp":ev['comp'],
                "fwd30":f30,"fwd60":f60,
            })
            seen[sym] = ev['date']

    # Leak check
    leaks = [p for p in promotions if not (ts <= p['date'] <= te)]

    f30 = [p['fwd30']*100 for p in promotions if p['fwd30'] is not None]
    f60 = [p['fwd60']*100 for p in promotions if p['fwd60'] is not None]

    eq_aug = apply_overlay(eq_base, promotions, all_prices, dates)
    m_base = equity_metrics(eq_base, ts, te)
    m_aug  = equity_metrics(eq_aug, ts, te)

    # Per-symbol
    sym_stats = {}
    for sym in shadow_syms:
        sps = [p for p in promotions if p['sym']==sym]
        s30 = [p['fwd30']*100 for p in sps if p['fwd30'] is not None]
        s60 = [p['fwd60']*100 for p in sps if p['fwd60'] is not None]
        wins30 = sum(1 for x in s30 if x>0)
        sym_stats[sym] = {
            "n":len(sps),
            "fwd30_mean":round(sum(s30)/len(s30),2) if s30 else None,
            "fwd30_wr":round(wins30/len(s30)*100,1) if s30 else None,
            "fwd60_mean":round(sum(s60)/len(s60),2) if s60 else None,
            "fwd30_pf":(round(sum(x for x in s30 if x>0) /
                        abs(sum(x for x in s30 if x<=0)),2)
                        if any(x<=0 for x in s30) and any(x>0 for x in s30) else None),
        }

    pf_all = None
    wins  = sum(x for x in f30 if x > 0)
    losses= sum(x for x in f30 if x <= 0)
    if losses < 0 and wins > 0:
        pf_all = round(wins / abs(losses), 2)

    return {
        "n_promotions": len(promotions),
        "n_syms": len(set(p['sym'] for p in promotions)),
        "type_counts": dict(type_counts),
        "reject_detail": dict(reject_detail),
        "fwd30_mean":   round(statistics.mean(f30),2) if f30 else None,
        "fwd30_median": round(statistics.median(f30),2) if f30 else None,
        "fwd30_wr":     round(sum(1 for x in f30 if x>0)/len(f30)*100,1) if f30 else None,
        "fwd30_pf":     pf_all,
        "fwd30_n":      len(f30),
        "fwd60_mean":   round(statistics.mean(f60),2) if f60 else None,
        "fwd60_wr":     round(sum(1 for x in f60 if x>0)/len(f60)*100,1) if f60 else None,
        "fwd60_n":      len(f60),
        "base_cagr":    m_base['cagr'],
        "base_dd":      m_base['max_dd'],
        "base_sharpe":  m_base['sharpe'],
        "base_calmar":  m_base['calmar'],
        "aug_cagr":     m_aug['cagr'],
        "aug_dd":       m_aug['max_dd'],
        "aug_sharpe":   m_aug['sharpe'],
        "aug_calmar":   m_aug['calmar'],
        "delta_cagr":   round(m_aug['cagr']-m_base['cagr'],2),
        "delta_dd":     round(m_aug['max_dd']-m_base['max_dd'],2),
        "delta_sharpe": round(m_aug['sharpe']-m_base['sharpe'],3),
        "delta_calmar": round(m_aug['calmar']-m_base['calmar'],3),
        "leak_count":   len(leaks),
        "sym_stats":    sym_stats,
        "promotions":   promotions,
    }


def main():
    print("Loading OHLCV...")
    all_syms = list(set(SHADOW_SYMS+LIVE_SYMS))
    all_prices = {sym: load_prices(sym) for sym in all_syms}
    all_prices = {k:v for k,v in all_prices.items() if v}

    dates = all_trading_dates(all_prices)
    print(f"Dates: {dates[0]} → {dates[-1]}  ({len(dates)} days)")

    print("Computing RSR...")
    rsr_all  = compute_rsr(all_prices, dates)
    print("Computing SI...")
    si_all   = compute_si(all_prices, dates)
    print("Computing compression...")
    comp_all = compute_comp(all_prices, dates)

    monthly_dates = sorted(set(rsr_all) & set(si_all))
    print(f"Monthly eval dates: {monthly_dates[0]} → {monthly_dates[-1]}  ({len(monthly_dates)})")
    print()

    live_syms = [s for s in LIVE_SYMS if s in all_prices and s not in SHADOW_SYMS]
    eq_base = simulate_baseline(all_prices, live_syms, dates, rsr_all)

    # ── Full P2 (all 5 shadow symbols) ──────────────────────────────────────────
    print("=== FULL P2 (all 5 symbols) ===")
    fold_results = []
    for fold in WF_FOLDS:
        res = run_p2_fold(SHADOW_SYMS, all_prices, dates, rsr_all, si_all,
                          comp_all, monthly_dates, eq_base, fold)
        fold_results.append({"fold":fold['name'],"test_start":fold['test_start'],
                              "test_end":fold['test_end'],"train_end":fold['train_end'],
                              "p2":res})
        print(f"  {fold['name']} ({fold['test_start'][:4]}): "
              f"promo={res['n_promotions']} n_syms={res['n_syms']} "
              f"fwd30={res['fwd30_mean']}% wr={res['fwd30_wr']}% pf={res['fwd30_pf']} "
              f"ΔCAGR={res['delta_cagr']:+.2f}pp "
              f"ΔDD={res['delta_dd']:+.2f}pp "
              f"ΔSharpe={res['delta_sharpe']:+.3f} "
              f"leaks={res['leak_count']}")

    # Aggregate
    dc = [fr['p2']['delta_cagr']  for fr in fold_results]
    dd = [fr['p2']['delta_dd']    for fr in fold_results]
    ds = [fr['p2']['delta_sharpe'] for fr in fold_results]
    dm = [fr['p2']['delta_calmar'] for fr in fold_results]
    f30_all = [m for fr in fold_results
               for m in [fr['p2']['fwd30_mean']] if m is not None]
    print(f"\n  AVG: ΔCAGR={sum(dc)/len(dc):+.2f}pp "
          f"ΔDD={sum(dd)/len(dd):+.2f}pp "
          f"ΔSharpe={sum(ds)/len(ds):+.3f} "
          f"ΔCalmar={sum(dm)/len(dm):+.3f} "
          f"fwd30_avg={sum(f30_all)/len(f30_all):.2f}%")
    print(f"  per-fold ΔCAGR: {[f'{v:+.2f}' for v in dc]}")
    print(f"  per-fold ΔDD:   {[f'{v:+.2f}' for v in dd]}")
    std_dc = statistics.stdev(dc) if len(dc)>1 else 0
    print(f"  ΔCAGR std:      {std_dc:.2f}pp")

    # ── Per-symbol full OOS breakdown ────────────────────────────────────────────
    print("\n=== PER-SYMBOL OOS (full P2) ===")
    all_promos = [p for fr in fold_results for p in fr['p2']['promotions']]
    sym_agg = defaultdict(lambda: {'fwd30':[],'fwd60':[],'fold':defaultdict(list)})
    for pr in all_promos:
        sym = pr['sym']
        if pr['fwd30'] is not None:
            sym_agg[sym]['fwd30'].append(pr['fwd30']*100)
        if pr['fwd60'] is not None:
            sym_agg[sym]['fwd60'].append(pr['fwd60']*100)

    sym_full_stats = {}
    for sym in SHADOW_SYMS:
        f30 = sym_agg[sym]['fwd30']
        f60 = sym_agg[sym]['fwd60']
        wins30 = sum(1 for x in f30 if x>0)
        losses30 = sum(1 for x in f30 if x<=0)
        gross_w = sum(x for x in f30 if x>0)
        gross_l = abs(sum(x for x in f30 if x<=0))
        pf = round(gross_w/gross_l,2) if gross_l>0 and gross_w>0 else None
        sym_full_stats[sym] = {
            "n": len(f30),
            "fwd30_mean": round(sum(f30)/len(f30),2) if f30 else None,
            "fwd30_wr":   round(wins30/len(f30)*100,1) if f30 else None,
            "fwd30_pf":   pf,
            "fwd60_mean": round(sum(f60)/len(f60),2) if f60 else None,
            "fwd60_wr":   round(sum(1 for x in f60 if x>0)/len(f60)*100,1) if f60 else None,
        }
        print(f"  {sym}: n={len(f30)} fwd30={sym_full_stats[sym]['fwd30_mean']}% "
              f"wr={sym_full_stats[sym]['fwd30_wr']}% pf={pf} "
              f"fwd60={sym_full_stats[sym]['fwd60_mean']}%")

    # ── Robustness: leave-one-out ────────────────────────────────────────────────
    print("\n=== ROBUSTNESS: leave-one-out ===")
    leave_one_out = {}
    for exclude_sym in SHADOW_SYMS:
        syms_subset = [s for s in SHADOW_SYMS if s != exclude_sym]
        fold_res_loo = []
        for fold in WF_FOLDS:
            res = run_p2_fold(syms_subset, all_prices, dates, rsr_all, si_all,
                              comp_all, monthly_dates, eq_base, fold)
            fold_res_loo.append(res)
        dc_loo = [r['delta_cagr']  for r in fold_res_loo]
        dd_loo = [r['delta_dd']    for r in fold_res_loo]
        ds_loo = [r['delta_sharpe'] for r in fold_res_loo]
        avg_dc = sum(dc_loo)/len(dc_loo)
        avg_ds = sum(ds_loo)/len(ds_loo)
        n_total = sum(r['n_promotions'] for r in fold_res_loo)
        leave_one_out[exclude_sym] = {
            "avg_delta_cagr": round(avg_dc,2),
            "avg_delta_sharpe": round(avg_ds,3),
            "avg_delta_dd": round(sum(dd_loo)/len(dd_loo),2),
            "n_promotions": n_total,
            "per_fold_delta_cagr": dc_loo,
        }
        print(f"  Excl {exclude_sym}: avg ΔCAGR={avg_dc:+.2f}pp "
              f"avg ΔSharpe={avg_ds:+.3f} n={n_total}")

    # ── Robustness: leave-two-out for key combos ─────────────────────────────────
    print("\n=== ROBUSTNESS: leave-two-out (key combos) ===")
    combos = [
        ("5706.T","6857.T"),
        ("5706.T","8035.T"),
        ("5706.T","6920.T"),
        ("6857.T","6920.T"),
    ]
    leave_two_out = {}
    for excl_a, excl_b in combos:
        syms_sub = [s for s in SHADOW_SYMS if s not in (excl_a, excl_b)]
        fold_res_l2 = []
        for fold in WF_FOLDS:
            res = run_p2_fold(syms_sub, all_prices, dates, rsr_all, si_all,
                              comp_all, monthly_dates, eq_base, fold)
            fold_res_l2.append(res)
        dc_l2 = [r['delta_cagr'] for r in fold_res_l2]
        ds_l2 = [r['delta_sharpe'] for r in fold_res_l2]
        avg_dc2 = sum(dc_l2)/len(dc_l2)
        avg_ds2 = sum(ds_l2)/len(ds_l2)
        key = f"excl_{excl_a}+{excl_b}"
        leave_two_out[key] = {
            "avg_delta_cagr": round(avg_dc2,2),
            "avg_delta_sharpe": round(avg_ds2,3),
            "per_fold": dc_l2,
        }
        print(f"  Excl {excl_a}+{excl_b}: avg ΔCAGR={avg_dc2:+.2f}pp ΔSharpe={avg_ds2:+.3f}")

    # ── Fold-level per-symbol breakdown ─────────────────────────────────────────
    fold_sym_matrix = {}
    for fr in fold_results:
        yr = fr['test_start'][:4]
        fold_sym_matrix[yr] = {}
        for pr in fr['p2']['promotions']:
            sym = pr['sym']
            if sym not in fold_sym_matrix[yr]:
                fold_sym_matrix[yr][sym] = {'fwd30':[],'fwd60':[]}
            if pr['fwd30'] is not None:
                fold_sym_matrix[yr][sym]['fwd30'].append(pr['fwd30']*100)
            if pr['fwd60'] is not None:
                fold_sym_matrix[yr][sym]['fwd60'].append(pr['fwd60']*100)

    print("\n=== FOLD × SYMBOL MATRIX (fwd30 mean) ===")
    header = f"{'':8s}" + "".join(f"{s:>12s}" for s in SHADOW_SYMS)
    print(f"  {header}")
    for yr, sym_d in sorted(fold_sym_matrix.items()):
        row = f"  {yr:8s}"
        for sym in SHADOW_SYMS:
            vals = sym_d.get(sym,{}).get('fwd30',[])
            if vals:
                row += f"{sum(vals)/len(vals):>+11.1f}%"
            else:
                row += f"{'—':>12s}"
        print(row)

    # Lease check total
    leak_total = sum(fr['p2']['leak_count'] for fr in fold_results)
    print(f"\nLeak check total: {leak_total} violations")

    return {
        "fold_results": fold_results,
        "agg": {
            "delta_cagr": dc,
            "delta_dd": dd,
            "delta_sharpe": ds,
            "delta_calmar": dm,
            "avg_delta_cagr": round(sum(dc)/len(dc),2),
            "avg_delta_dd": round(sum(dd)/len(dd),2),
            "avg_delta_sharpe": round(sum(ds)/len(ds),3),
            "avg_delta_calmar": round(sum(dm)/len(dm),3),
            "std_delta_cagr": round(std_dc,2),
        },
        "sym_full_stats": sym_full_stats,
        "leave_one_out": leave_one_out,
        "leave_two_out": leave_two_out,
        "fold_sym_matrix": fold_sym_matrix,
        "leak_total": leak_total,
        "all_promotions_count": len(all_promos),
    }


if __name__ == "__main__":
    res = main()

    out = Path("C:/ai-trading/backtests/auto_promote_wf_p2_results.json")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            return None if obj != obj else round(obj, 4)
        return obj

    out.write_text(json.dumps(clean(res), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: {out}")
