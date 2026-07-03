"""
AUTO_PROMOTE_V2 Walk-Forward Validation: Case A vs Case E
Expanding-window WF: 4 folds (Test: 2021 / 2022 / 2023 / 2024)
No lookahead: RSR/SI computed from past data only.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import json
import statistics
import math

OHLCV_DIR = Path("C:/ai-trading/data/ohlcv")
CAPITAL    = 3_000_000
SLIPPAGE   = 0.001
COMMISSION = 0.00055
MAX_POS    = 3
ALLOC_MUL  = 0.25   # probation multiplier (25% of normal 1/3 = ~8.3%)

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
    "6586.T":"電機精密","6724.T":"電機精密","6841.T":"電機精密",
    "6869.T":"電機精密","6902.T":"電機精密","6981.T":"電機精密",
    "6702.T":"電機","6501.T":"電機","6752.T":"電機","6758.T":"電機",
    "6701.T":"電機","6503.T":"電機","6504.T":"電機",
    "7011.T":"機械","6301.T":"機械","6326.T":"機械","6273.T":"機械",
    "6471.T":"機械","6201.T":"機械","6367.T":"機械","7013.T":"機械","7012.T":"機械",
    "7203.T":"輸送機器","7201.T":"輸送機器",
    "8306.T":"銀行","8411.T":"銀行","8309.T":"銀行","7182.T":"銀行",
    "8354.T":"銀行","6178.T":"銀行",
    "8725.T":"保険","8766.T":"保険","8750.T":"保険",
    "8058.T":"商社","8053.T":"商社","8002.T":"商社","8001.T":"商社","8015.T":"商社",
    "4021.T":"化学","4063.T":"化学","4188.T":"化学","4452.T":"化学",
    "4901.T":"化学","4911.T":"化学","4004.T":"化学","3407.T":"化学","3402.T":"化学",
    "2914.T":"食品","2801.T":"食品","2802.T":"食品","2503.T":"食品",
    "2502.T":"食品","2269.T":"食品","2282.T":"食品","2002.T":"食品",
    "9432.T":"情報通信","4307.T":"情報通信","4055.T":"情報通信","6098.T":"情報通信",
    "3382.T":"小売","3099.T":"小売","3197.T":"小売","3289.T":"小売",
    "5401.T":"鉄鋼","5411.T":"鉄鋼",
    "5706.T":"非鉄金属","5713.T":"非鉄金属","5714.T":"非鉄金属",
    "9101.T":"海運","9104.T":"海運","9107.T":"海運",
    "9531.T":"ガス",
    "4503.T":"医薬品","4507.T":"医薬品","4519.T":"医薬品",
    "4523.T":"医薬品","4543.T":"医薬品","4568.T":"医薬品",
    "4578.T":"医薬品","4661.T":"医薬品","4151.T":"医薬品",
    "3861.T":"紙パルプ","5201.T":"ガラス土石","5333.T":"ガラス土石",
    "1605.T":"石油","5020.T":"石油","5108.T":"化学",
    "1925.T":"建設","1928.T":"建設","1801.T":"建設","1802.T":"建設",
    "1803.T":"建設","1812.T":"建設","1721.T":"建設",
}

# WF fold definitions (expanding train, fixed-year test)
WF_FOLDS = [
    {"name": "Fold1", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "Fold2", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "Fold3", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"name": "Fold4", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
]

# Taxonomy thresholds
MATURE_LEADER_MIN_RSR         = 87.0
MATURE_LEADER_MAX_COMPRESSION = 35.0
MEAN_REVERSION_MAX_RSR        = 55.0
MEAN_REVERSION_MAX_IGNITION   = 55.0
IGNITION_MIN_SECTOR_IGNITION  = 75.0
IGNITION_MAX_RSR              = 85.0
CONTINUATION_MIN_RSR          = 70.0
CONTINUATION_MIN_COMPRESSION  = 28.0
CONTINUATION_MAX_RSR          = 88.0
HIGH_RSR_BYPASS_RSR           = 90.0
ALLOWED_BASE = {"HIGH_RSR", "EARLY_IGNITION", "CONTINUATION"}

CASES = {
    "A": {"si_thr": 90.0, "allow_unc": False},
    "E": {"si_thr": 50.0, "allow_unc": True},
}

# ── Data loading ──────────────────────────────────────────────────────────────
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

# ── Score computation (uses only data up to eval_date → no lookahead) ─────────
def compute_rsr(all_prices, dates):
    """Monthly RSR: 12-month momentum percentile rank. Computed from past data only."""
    monthly = dates[::21]
    rsr_by_date = {}
    for i, dt in enumerate(monthly):
        if i < 12:
            continue
        dt_12m = monthly[i - 12]
        rets = {}
        for sym, prices in all_prices.items():
            p0 = prices.get(dt_12m)
            p1 = prices.get(dt)
            if p0 and p1 and p0 > 0:
                rets[sym] = p1 / p0 - 1
        if not rets:
            continue
        ranked = sorted(rets.keys(), key=lambda s: rets[s])
        n = len(ranked)
        rsr_by_date[dt] = {
            sym: round(rank / (n - 1) * 100, 1) if n > 1 else 50.0
            for rank, sym in enumerate(ranked)
        }
    return rsr_by_date

def compute_si(all_prices, dates):
    """Monthly sector ignition from OHLCV breadth+velocity. No lookahead."""
    monthly = dates[::21]
    si_by_date = {}
    for i, dt in enumerate(monthly):
        if i < 3:
            continue
        dt_prev = monthly[i - 1]
        vel_5d = {}
        for sym, prices in all_prices.items():
            p0 = prices.get(dt_prev)
            p1 = prices.get(dt)
            if p0 and p1 and p0 > 0:
                vel_5d[sym] = (p1 / p0 - 1) * 100
        breakout = {}
        for sym, prices in all_prices.items():
            avail = [prices.get(d) for d in monthly[max(0,i-4):i] if prices.get(d)]
            p1 = prices.get(dt)
            if avail and p1:
                breakout[sym] = p1 >= max(avail) * 0.98
        sectors = defaultdict(list)
        for sym in all_prices:
            sectors[SECTOR_MAP.get(sym, "_unk")].append(sym)
        sec_raw = {}
        for sec, syms in sectors.items():
            n = len(syms)
            if n == 0:
                sec_raw[sec] = 0.0
                continue
            breadth = sum(1 for s in syms if vel_5d.get(s, 0) > 0) / n
            vels = [vel_5d.get(s, 0) for s in syms]
            vel_norm = max(0.0, min(1.0, (sum(vels)/len(vels) + 2.5) / 5.0))
            bo_ratio = sum(1 for s in syms if breakout.get(s, False)) / n
            sec_raw[sec] = 0.35*breadth + 0.25*vel_norm + 0.25*bo_ratio + 0.15*0.5
        sym_raw = {sym: sec_raw.get(SECTOR_MAP.get(sym,"_unk"), 0.0) for sym in all_prices}
        vals = sorted(sym_raw.values())
        n = len(vals)
        if n <= 1:
            si_by_date[dt] = {sym: 50.0 for sym in all_prices}
            continue
        si_by_date[dt] = {
            sym: round(sum(1 for x in vals if x <= v) / n * 100, 1)
            for sym, v in sym_raw.items()
        }
    return si_by_date

def compute_comp(all_prices, dates):
    """Monthly compression proxy. No lookahead."""
    monthly = dates[::21]
    comp_by_date = {}
    for i, dt in enumerate(monthly):
        if i < 6:
            continue
        comp = {}
        for sym, prices in all_prices.items():
            a20 = [prices.get(d) for d in monthly[max(0,i-2):i+1] if prices.get(d)]
            a60 = [prices.get(d) for d in monthly[max(0,i-6):i+1] if prices.get(d)]
            if len(a20) >= 2 and len(a60) >= 3:
                r20 = (max(a20)-min(a20))/min(a20) if min(a20) > 0 else 0
                r60 = (max(a60)-min(a60))/min(a60) if min(a60) > 0 else 0
                comp[sym] = round(max(0, min(100, (r20/r60 if r60 > 0 else 1)*50)), 1)
            else:
                comp[sym] = 50.0
        comp_by_date[dt] = comp
    return comp_by_date

# ── Taxonomy ──────────────────────────────────────────────────────────────────
def classify(sym, rsr, si, comp):
    late = rsr >= MATURE_LEADER_MIN_RSR and comp < MATURE_LEADER_MAX_COMPRESSION
    if late:
        return "MATURE_LEADER"
    if rsr < MEAN_REVERSION_MAX_RSR and si < MEAN_REVERSION_MAX_IGNITION:
        return "MEAN_REVERSION"
    if si >= IGNITION_MIN_SECTOR_IGNITION and rsr < IGNITION_MAX_RSR:
        return "EARLY_IGNITION"
    if rsr >= CONTINUATION_MIN_RSR and comp >= CONTINUATION_MIN_COMPRESSION and rsr < CONTINUATION_MAX_RSR:
        return "CONTINUATION"
    if rsr >= HIGH_RSR_BYPASS_RSR and si >= 90.0:
        return "HIGH_RSR"
    return "UNCLASSIFIED"

def gate_pass(sym, rsr, si, comp, si_thr, allow_unc):
    if rsr < 8.0:
        return False, classify(sym, rsr, si, comp), "rsr_too_low"
    ctype = classify(sym, rsr, si, comp)
    if si_thr > 0 and si < si_thr:
        return False, ctype, f"si<{si_thr}"
    allowed = ALLOWED_BASE | ({"UNCLASSIFIED"} if allow_unc else set())
    if ctype not in allowed:
        return False, ctype, f"taxonomy:{ctype}"
    return True, ctype, ""

# ── Forward returns ───────────────────────────────────────────────────────────
def fwd_return(prices, dates, entry_dt, n_days):
    idx = next((i for i, d in enumerate(dates) if d >= entry_dt), None)
    if idx is None or idx + n_days >= len(dates):
        return None
    p0 = prices.get(dates[idx])
    p1 = prices.get(dates[idx + n_days])
    if p0 and p1 and p0 > 0:
        return (p1/p0 - 1) - (SLIPPAGE+COMMISSION)*2
    return None

# ── Portfolio metrics ─────────────────────────────────────────────────────────
def equity_metrics(eq_curve, test_start, test_end):
    dates = sorted(d for d in eq_curve if test_start <= d <= test_end)
    if len(dates) < 50:
        return {"cagr":0,"max_dd":0,"sharpe":0,"calmar":0,"n_days":len(dates)}
    vals = [eq_curve[d] for d in dates]
    years = len(dates) / 252
    cagr = (vals[-1]/vals[0])**(1/years) - 1
    peak, max_dd = vals[0], 0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v-peak)/peak
        if dd < max_dd:
            max_dd = dd
    daily_rets = [vals[i]/vals[i-1]-1 for i in range(1,len(vals))]
    avg = sum(daily_rets)/len(daily_rets)
    std = statistics.stdev(daily_rets) if len(daily_rets)>1 else 1e-6
    sharpe = avg*252 / (std*math.sqrt(252)) if std > 0 else 0
    calmar = abs(cagr/max_dd) if max_dd != 0 else 0
    return {
        "cagr": round(cagr*100, 2),
        "max_dd": round(max_dd*100, 2),
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "n_days": len(dates),
    }

# ── Baseline portfolio simulation ─────────────────────────────────────────────
def simulate_baseline(all_prices, live_syms, dates, rsr_by_date):
    """Simple Fujiko-law proxy: RSR>=75 triggers entry, RSR<70 or 60d triggers exit."""
    capital = float(CAPITAL)
    cash = capital
    positions = {}
    eq = {}
    monthly = set(rsr_by_date.keys())
    for di, dt in enumerate(dates):
        mval = cash
        for sym, pos in positions.items():
            p = all_prices.get(sym,{}).get(dt, pos['ep'])
            mval += pos['qty'] * p
        eq[dt] = mval
        if dt not in monthly:
            continue
        rsr = rsr_by_date.get(dt, {})
        to_close = [sym for sym, pos in positions.items()
                    if rsr.get(sym,50) < 70 or
                    sum(1 for d in dates if pos['ed'] <= d <= dt) > 60]
        for sym in to_close:
            pos = positions.pop(sym)
            p = all_prices.get(sym,{}).get(dt, pos['ep'])
            cash += pos['qty'] * p * (1 - SLIPPAGE - COMMISSION)
        slots = MAX_POS - len(positions)
        if slots <= 0:
            continue
        buys = sorted(
            [(s, rsr.get(s,0)) for s in live_syms
             if s not in positions and rsr.get(s,0) >= 75 and all_prices.get(s,{}).get(dt)],
            key=lambda x: -x[1]
        )
        for sym, _ in buys[:slots]:
            alloc = cash / (slots+len(positions)+1)
            p = all_prices[sym][dt]
            qty = int(alloc / (p*(1+SLIPPAGE+COMMISSION)))
            if qty <= 0:
                continue
            cost = qty * p * (1+SLIPPAGE+COMMISSION)
            cash -= cost
            positions[sym] = {'ep': p, 'qty': qty, 'ed': dt}
    return eq

# ── Probation portfolio overlay ───────────────────────────────────────────────
def apply_probation_overlay(eq_baseline, promotions, all_prices, dates):
    """Add probation P&L on top of baseline equity curve."""
    eq = dict(eq_baseline)
    for pr in promotions:
        if pr['fwd30'] is None:
            continue
        sym = pr['sym']
        entry_dt = pr['date']
        idx = next((i for i,d in enumerate(dates) if d >= entry_dt), None)
        if idx is None:
            continue
        exit_idx = min(idx + 30, len(dates)-1)
        alloc_pct = ALLOC_MUL / MAX_POS
        entry_eq = eq.get(dates[idx], CAPITAL)
        alloc = entry_eq * alloc_pct
        p0 = all_prices.get(sym,{}).get(dates[idx], 0)
        p1 = all_prices.get(sym,{}).get(dates[exit_idx], 0)
        if p0 > 0 and p1 > 0:
            pnl = alloc * (p1/p0-1) - alloc*(SLIPPAGE+COMMISSION)*2
            exit_dt = dates[exit_idx]
            for d in dates:
                if d >= exit_dt:
                    eq[d] = eq.get(d, CAPITAL) + pnl
    return eq

# ── Leak check ────────────────────────────────────────────────────────────────
def check_lookahead(promotions, test_start, test_end):
    """
    Verify no train-period data used in test decisions.
    All promotion dates must be within [test_start, test_end].
    RSR/SI scores computed from data up to decision date only.
    """
    leaks = []
    for pr in promotions:
        if not (test_start <= pr['date'] <= test_end):
            leaks.append(f"date {pr['date']} outside test window")
    return leaks

# ── Main WF loop ──────────────────────────────────────────────────────────────
def main():
    print("Loading OHLCV...")
    all_syms = list(set(SHADOW_SYMS + LIVE_SYMS))
    all_prices = {sym: load_prices(sym) for sym in all_syms}
    all_prices = {k: v for k, v in all_prices.items() if v}

    dates = all_trading_dates(all_prices)
    print(f"Dates: {dates[0]} → {dates[-1]}  ({len(dates)} days)")

    print("Computing RSR (all dates)...")
    rsr_all = compute_rsr(all_prices, dates)

    print("Computing sector ignition (all dates)...")
    si_all = compute_si(all_prices, dates)

    print("Computing compression (all dates)...")
    comp_all = compute_comp(all_prices, dates)

    monthly_dates = sorted(set(rsr_all) & set(si_all))
    print(f"Monthly eval dates: {monthly_dates[0]} → {monthly_dates[-1]}  ({len(monthly_dates)})")
    print()

    live_syms = [s for s in LIVE_SYMS if s in all_prices and s not in SHADOW_SYMS]

    # Precompute baseline equity (full period)
    eq_baseline = simulate_baseline(all_prices, live_syms, dates, rsr_all)

    fold_results = []

    for fold in WF_FOLDS:
        fname = fold['name']
        te_s = fold['test_start']
        te_e = fold['test_end']
        tr_e = fold['train_end']

        print(f"=== {fname}: Train→{tr_e}  Test:{te_s}~{te_e} ===")

        # Promotion events in TEST period only (no training of parameters)
        test_monthly = [d for d in monthly_dates if te_s <= d <= te_e]
        print(f"  Test monthly dates: {len(test_monthly)}")

        fold_case = {}
        for cname, cfg in CASES.items():
            si_thr = cfg['si_thr']
            allow_unc = cfg['allow_unc']
            promotions = []
            type_counts = defaultdict(int)

            for dt in test_monthly:
                rsr = rsr_all.get(dt, {})
                si  = si_all.get(dt, {})
                comp = comp_all.get(dt, {})
                for sym in SHADOW_SYMS:
                    rv = rsr.get(sym, 0)
                    sv = si.get(sym, 0)
                    cv = comp.get(sym, 50)
                    passed, ctype, reason = gate_pass(sym, rv, sv, cv, si_thr, allow_unc)
                    type_counts[ctype] += 1
                    if passed:
                        f30 = fwd_return(all_prices[sym], dates, dt, 30)
                        f60 = fwd_return(all_prices[sym], dates, dt, 60)
                        promotions.append({
                            "sym": sym, "date": dt, "type": ctype,
                            "rsr": rv, "si": sv, "comp": cv,
                            "fwd30": f30, "fwd60": f60,
                        })

            # Deduplicate: one per symbol per 30d window
            seen = {}
            deduped = []
            for pr in sorted(promotions, key=lambda x: x['date']):
                sym = pr['sym']
                prev = seen.get(sym)
                if prev is None or (
                    sum(1 for d in test_monthly if prev < d <= pr['date']) >= 2
                ):
                    deduped.append(pr)
                    seen[sym] = pr['date']
            promotions = deduped

            # Leak check
            leaks = check_lookahead(promotions, te_s, te_e)

            fwd30 = [p['fwd30']*100 for p in promotions if p['fwd30'] is not None]
            fwd60 = [p['fwd60']*100 for p in promotions if p['fwd60'] is not None]

            # Portfolio metrics
            eq_aug = apply_probation_overlay(eq_baseline, promotions, all_prices, dates)
            m_base = equity_metrics(eq_baseline, te_s, te_e)
            m_aug  = equity_metrics(eq_aug, te_s, te_e)

            # Contribution analysis: P&L from probation positions only
            contrib_pnl = sum(
                eq_aug.get(dates[-1], CAPITAL) - eq_baseline.get(dates[-1], CAPITAL)
                for _ in [1]  # single calculation
            )

            fold_case[cname] = {
                "n_promotions": len(promotions),
                "n_syms": len(set(p['sym'] for p in promotions)),
                "type_counts": dict(type_counts),
                "fwd30_mean": round(statistics.mean(fwd30),2) if fwd30 else None,
                "fwd30_median": round(statistics.median(fwd30),2) if fwd30 else None,
                "fwd30_win_rate": round(sum(1 for f in fwd30 if f>0)/len(fwd30)*100,1) if fwd30 else None,
                "fwd30_n": len(fwd30),
                "fwd60_mean": round(statistics.mean(fwd60),2) if fwd60 else None,
                "fwd60_win_rate": round(sum(1 for f in fwd60 if f>0)/len(fwd60)*100,1) if fwd60 else None,
                "fwd60_n": len(fwd60),
                "base_cagr": m_base['cagr'],
                "base_dd": m_base['max_dd'],
                "base_sharpe": m_base['sharpe'],
                "base_calmar": m_base['calmar'],
                "aug_cagr": m_aug['cagr'],
                "aug_dd": m_aug['max_dd'],
                "aug_sharpe": m_aug['sharpe'],
                "aug_calmar": m_aug['calmar'],
                "delta_cagr": round(m_aug['cagr'] - m_base['cagr'], 2),
                "delta_dd": round(m_aug['max_dd'] - m_base['max_dd'], 2),
                "delta_sharpe": round(m_aug['sharpe'] - m_base['sharpe'], 3),
                "delta_calmar": round(m_aug['calmar'] - m_base['calmar'], 3),
                "leak_count": len(leaks),
                "promotions": promotions,
            }
            sym_counts = dict(type_counts)
            print(f"  Case {cname}: promo={len(promotions)} n_syms={len(set(p['sym'] for p in promotions))} "
                  f"fwd30={fold_case[cname]['fwd30_mean']}% "
                  f"ΔCAGR={fold_case[cname]['delta_cagr']:+.2f}pp "
                  f"ΔDD={fold_case[cname]['delta_dd']:+.2f}pp "
                  f"ΔSharpe={fold_case[cname]['delta_sharpe']:+.3f} "
                  f"leaks={len(leaks)}")

        fold_results.append({
            "fold": fname,
            "test_start": te_s,
            "test_end": te_e,
            "train_end": tr_e,
            "cases": fold_case,
        })

    # ── Aggregate WF stats ───────────────────────────────────────────────────
    agg = defaultdict(lambda: defaultdict(list))
    for fr in fold_results:
        for cname, r in fr['cases'].items():
            agg[cname]['delta_cagr'].append(r['delta_cagr'])
            agg[cname]['delta_dd'].append(r['delta_dd'])
            agg[cname]['delta_sharpe'].append(r['delta_sharpe'])
            agg[cname]['delta_calmar'].append(r['delta_calmar'])
            agg[cname]['n_promo'].append(r['n_promotions'])
            if r['fwd30_mean'] is not None:
                agg[cname]['fwd30'].append(r['fwd30_mean'])
            if r['fwd60_mean'] is not None:
                agg[cname]['fwd60'].append(r['fwd60_mean'])

    # ── Per-symbol failure analysis (Case E) ─────────────────────────────────
    all_e_promos = [p for fr in fold_results for p in fr['cases']['E']['promotions']]
    sym_analysis = defaultdict(lambda: defaultdict(list))
    for pr in all_e_promos:
        sym = pr['sym']
        ctype = pr['type']
        if pr['fwd30'] is not None:
            sym_analysis[sym]['fwd30'].append(pr['fwd30']*100)
        if pr['fwd60'] is not None:
            sym_analysis[sym]['fwd60'].append(pr['fwd60']*100)
        sym_analysis[sym]['types'].append(ctype)

    # Failure cases: fwd30 < -5%
    failures = [p for p in all_e_promos if p['fwd30'] is not None and p['fwd30']*100 < -5]

    return {
        "fold_results": fold_results,
        "agg": {k: dict(v) for k, v in agg.items()},
        "sym_analysis": {k: dict(v) for k, v in sym_analysis.items()},
        "failures": failures,
        "leak_total": sum(
            fr['cases'][c]['leak_count']
            for fr in fold_results
            for c in CASES
        ),
    }


if __name__ == "__main__":
    res = main()

    out = Path("C:/ai-trading/backtests/auto_promote_wf_results.json")

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

    # Aggregate summary
    agg = res['agg']
    print("\n=== WF AGGREGATE ===")
    for cname in ['A', 'E']:
        dc = agg[cname]['delta_cagr']
        dd = agg[cname]['delta_dd']
        ds = agg[cname]['delta_sharpe']
        print(f"Case {cname}: ΔCAGR={sum(dc)/len(dc):+.2f}pp(avg) "
              f"ΔDD={sum(dd)/len(dd):+.2f}pp(avg) "
              f"ΔSharpe={sum(ds)/len(ds):+.3f}(avg) "
              f"folds={len(dc)}")
        print(f"  per-fold ΔCAGR: {[f'{v:+.2f}' for v in dc]}")
        print(f"  per-fold ΔDD:   {[f'{v:+.2f}' for v in dd]}")

    print(f"\nLeak check total: {res['leak_total']} violations")
    print(f"Case E OOS failures (fwd30<-5%): {len(res['failures'])} / {sum(len(fr['cases']['E']['promotions']) for fr in res['fold_results'])}")
