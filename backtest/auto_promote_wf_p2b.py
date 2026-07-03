"""
AUTO_PROMOTE P2-B Walk-Forward
Base: P2-A (UNCLASSIFIED + 50<=SI<90, excl 5706.T)
Compare: B1 (60<=SI<90), B2 (65<=SI<90)
Focus: 2021 Fold improvement vs P2-A
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import json, statistics, math

OHLCV_DIR  = Path("C:/ai-trading/data/ohlcv")
CAPITAL    = 3_000_000
SLIPPAGE   = 0.001
COMMISSION = 0.00055
MAX_POS    = 3
ALLOC_MUL  = 0.25

TARGET_SYMS = ["6146.T", "6857.T", "6920.T", "8035.T"]  # excl 5706.T

LIVE_SYMS = [
    "8035.T","6702.T","6501.T","6762.T","6920.T","6857.T","6594.T",
    "7203.T","7201.T","9432.T","8306.T","8411.T","8309.T","7182.T",
    "8725.T","8058.T","8053.T","8002.T","8001.T","4021.T","2914.T",
    "7011.T","3382.T","5401.T","5411.T","9531.T","9101.T","9104.T",
]

SECTOR_MAP = {
    "8035.T":"電機精密","6762.T":"電機精密","6920.T":"電機精密",
    "6857.T":"電機精密","6594.T":"電機精密","6146.T":"電機精密",
    "6702.T":"電機","6501.T":"電機",
    "7011.T":"機械","6301.T":"機械",
    "7203.T":"輸送機器","7201.T":"輸送機器",
    "8306.T":"銀行","8411.T":"銀行","8309.T":"銀行","7182.T":"銀行",
    "8725.T":"保険","8766.T":"保険",
    "8058.T":"商社","8053.T":"商社","8002.T":"商社","8001.T":"商社",
    "4021.T":"化学","4063.T":"化学","4188.T":"化学","4452.T":"化学",
    "4901.T":"化学","3407.T":"化学","3402.T":"化学",
    "2914.T":"食品","2801.T":"食品","2503.T":"食品",
    "9432.T":"情報通信","4307.T":"情報通信",
    "3382.T":"小売","3099.T":"小売",
    "5401.T":"鉄鋼","5411.T":"鉄鋼",
    "5706.T":"非鉄金属","5713.T":"非鉄金属",
    "9101.T":"海運","9104.T":"海運","9107.T":"海運",
    "9531.T":"ガス",
}

CASES = {
    "P2-A": {"si_lo": 50.0, "si_hi": 90.0},
    "B1":   {"si_lo": 60.0, "si_hi": 90.0},
    "B2":   {"si_lo": 65.0, "si_hi": 90.0},
}

WF_FOLDS = [
    {"name":"Fold1","train_end":"2020-12-31","test_start":"2021-01-01","test_end":"2021-12-31"},
    {"name":"Fold2","train_end":"2021-12-31","test_start":"2022-01-01","test_end":"2022-12-31"},
    {"name":"Fold3","train_end":"2022-12-31","test_start":"2023-01-01","test_end":"2023-12-31"},
    {"name":"Fold4","train_end":"2023-12-31","test_start":"2024-01-01","test_end":"2024-12-31"},
]

def classify(rsr, si, comp):
    if rsr >= 87.0 and comp < 35.0:           return "MATURE_LEADER"
    if rsr < 55.0  and si  < 55.0:            return "MEAN_REVERSION"
    if si  >= 75.0 and rsr < 85.0:            return "EARLY_IGNITION"
    if rsr >= 70.0 and comp >= 28.0 and rsr < 88.0: return "CONTINUATION"
    if rsr >= 90.0 and si  >= 90.0:           return "HIGH_RSR"
    return "UNCLASSIFIED"

def make_gate(si_lo, si_hi):
    def gate(rsr, si, comp):
        if rsr < 8.0: return False, classify(rsr, si, comp), "rsr_too_low"
        ct = classify(rsr, si, comp)
        if ct != "UNCLASSIFIED": return False, ct, f"taxonomy:{ct}"
        if si < si_lo:  return False, ct, f"si<{si_lo}"
        if si >= si_hi: return False, ct, f"si>={si_hi}"
        return True, ct, ""
    return gate

def load_prices(sym):
    f = OHLCV_DIR / f"{sym}.csv"
    if not f.exists(): return {}
    out = {}
    for line in f.read_text(encoding='utf-8').splitlines()[1:]:
        p = line.split(',')
        if len(p) >= 2:
            try: out[p[0].strip()] = float(p[1].strip())
            except: pass
    return out

def all_dates(prices_dict):
    ds = set()
    for p in prices_dict.values(): ds.update(p.keys())
    return sorted(d for d in ds if "2018-01-01" <= d <= "2024-12-31")

def compute_rsr(all_prices, dates):
    monthly, out = dates[::21], {}
    for i, dt in enumerate(monthly):
        if i < 12: continue
        dt12 = monthly[i - 12]
        rets = {}
        for s, prices in all_prices.items():
            p0, p1 = prices.get(dt12), prices.get(dt)
            if p0 and p1: rets[s] = p1 / p0 - 1
        if not rets: continue
        ranked = sorted(rets, key=lambda s: rets[s])
        n = len(ranked)
        out[dt] = {sym: round(rank / (n - 1) * 100, 1) if n > 1 else 50.0
                   for rank, sym in enumerate(ranked)}
    return out

def compute_si(all_prices, dates):
    monthly, out = dates[::21], {}
    for i, dt in enumerate(monthly):
        if i < 3: continue
        dt_prev = monthly[i - 1]
        vel = {s: (all_prices[s].get(dt, 0) / all_prices[s].get(dt_prev, 1e9) - 1) * 100
               for s in all_prices if all_prices[s].get(dt_prev)}
        bo  = {s: all_prices[s].get(dt, 0) >= max(
                   (all_prices[s].get(d, 0) for d in monthly[max(0, i - 4):i]), default=0) * 0.98
               for s in all_prices if all_prices[s].get(dt)}
        secs = defaultdict(list)
        for s in all_prices: secs[SECTOR_MAP.get(s, "_unk")].append(s)
        sr = {}
        for sec, syms in secs.items():
            n = len(syms)
            breadth  = sum(1 for s in syms if vel.get(s, 0) > 0) / n if n else 0
            vel_avg  = sum(vel.get(s, 0) for s in syms) / n if n else 0
            vel_norm = max(0.0, min(1.0, (vel_avg + 2.5) / 5.0))
            bo_r     = sum(1 for s in syms if bo.get(s, False)) / n if n else 0
            sr[sec]  = 0.35 * breadth + 0.25 * vel_norm + 0.25 * bo_r + 0.15 * 0.5
        sym_raw = {s: sr.get(SECTOR_MAP.get(s, "_unk"), 0.0) for s in all_prices}
        vals = sorted(sym_raw.values()); nv = len(vals)
        if nv <= 1:
            out[dt] = {s: 50.0 for s in all_prices}
            continue
        out[dt] = {s: round(sum(1 for x in vals if x <= v) / nv * 100, 1)
                   for s, v in sym_raw.items()}
    return out

def compute_comp(all_prices, dates):
    monthly, out = dates[::21], {}
    for i, dt in enumerate(monthly):
        if i < 6: continue
        comp = {}
        for sym, prices in all_prices.items():
            a20 = [prices.get(d) for d in monthly[max(0, i - 2):i + 1] if prices.get(d)]
            a60 = [prices.get(d) for d in monthly[max(0, i - 6):i + 1] if prices.get(d)]
            if len(a20) >= 2 and len(a60) >= 3:
                r20 = (max(a20) - min(a20)) / min(a20) if min(a20) > 0 else 0
                r60 = (max(a60) - min(a60)) / min(a60) if min(a60) > 0 else 0
                comp[sym] = round(max(0, min(100, (r20 / r60 if r60 > 0 else 1) * 50)), 1)
            else:
                comp[sym] = 50.0
        out[dt] = comp
    return out

def fwd_ret(prices, dates, entry_dt, n):
    idx = next((i for i, d in enumerate(dates) if d >= entry_dt), None)
    if idx is None or idx + n >= len(dates): return None
    p0, p1 = prices.get(dates[idx]), prices.get(dates[idx + n])
    if p0 and p1 and p0 > 0: return (p1 / p0 - 1) - (SLIPPAGE + COMMISSION) * 2
    return None

def metrics(eq, ts, te):
    ds = sorted(d for d in eq if ts <= d <= te)
    if len(ds) < 50: return dict(cagr=0, max_dd=0, sharpe=0, calmar=0)
    vs = [eq[d] for d in ds]; yr = len(ds) / 252
    cagr = (vs[-1] / vs[0]) ** (1 / yr) - 1
    pk, mdd = vs[0], 0.0
    for v in vs:
        if v > pk: pk = v
        dd = (v - pk) / pk
        if dd < mdd: mdd = dd
    rets = [vs[i] / vs[i - 1] - 1 for i in range(1, len(vs))]
    avg = sum(rets) / len(rets)
    std = statistics.stdev(rets) if len(rets) > 1 else 1e-6
    sh  = avg * 252 / (std * math.sqrt(252)) if std > 0 else 0
    cal = abs(cagr / mdd) if mdd != 0 else 0
    return dict(cagr=round(cagr * 100, 2), max_dd=round(mdd * 100, 2),
                sharpe=round(sh, 3), calmar=round(cal, 3))

def sim_base(all_prices, live_syms, dates, rsr_all):
    cash = float(CAPITAL); pos = {}; eq = {}
    monthly = set(rsr_all)
    for dt in dates:
        mv = cash + sum(p['qty'] * all_prices.get(s, {}).get(dt, p['ep']) for s, p in pos.items())
        eq[dt] = mv
        if dt not in monthly: continue
        rsr = rsr_all.get(dt, {})
        to_close = [s for s, p in pos.items()
                    if rsr.get(s, 50) < 70 or sum(1 for d in dates if p['ed'] <= d <= dt) > 60]
        for s in to_close:
            p = pos.pop(s); pr = all_prices.get(s, {}).get(dt, p['ep'])
            cash += p['qty'] * pr * (1 - SLIPPAGE - COMMISSION)
        sl = MAX_POS - len(pos)
        if sl <= 0: continue
        buys = sorted([(s, rsr.get(s, 0)) for s in live_syms
                       if s not in pos and rsr.get(s, 0) >= 75 and all_prices.get(s, {}).get(dt)],
                      key=lambda x: -x[1])
        for s, _ in buys[:sl]:
            al = cash / (sl + len(pos) + 1)
            pr = all_prices[s][dt]
            qty = int(al / (pr * (1 + SLIPPAGE + COMMISSION)))
            if qty <= 0: continue
            cash -= qty * pr * (1 + SLIPPAGE + COMMISSION)
            pos[s] = {'ep': pr, 'qty': qty, 'ed': dt}
    return eq

def overlay(eq_b, promos, all_prices, dates):
    eq = dict(eq_b)
    for pr in promos:
        if pr['fwd30'] is None: continue
        idx = next((i for i, d in enumerate(dates) if d >= pr['date']), None)
        if idx is None: continue
        ex = min(idx + 30, len(dates) - 1)
        al = eq.get(dates[idx], CAPITAL) * (ALLOC_MUL / MAX_POS)
        p0 = all_prices.get(pr['sym'], {}).get(dates[idx], 0)
        p1 = all_prices.get(pr['sym'], {}).get(dates[ex], 0)
        if p0 > 0 and p1 > 0:
            pnl = al * (p1 / p0 - 1) - al * (SLIPPAGE + COMMISSION) * 2
            exit_dt = dates[ex]
            for d in dates:
                if d >= exit_dt: eq[d] = eq.get(d, CAPITAL) + pnl
    return eq

def run_fold(gate_fn, all_prices, dates, rsr_all, si_all, comp_all, monthly_dates, eq_base, fold):
    ts, te = fold['test_start'], fold['test_end']
    tm = [d for d in monthly_dates if ts <= d <= te]
    all_ev = []
    for dt in tm:
        rsr = rsr_all.get(dt, {}); si = si_all.get(dt, {}); comp = comp_all.get(dt, {})
        for sym in TARGET_SYMS:
            rv, sv, cv = rsr.get(sym, 0), si.get(sym, 0), comp.get(sym, 50)
            passed, ct, reason = gate_fn(rv, sv, cv)
            all_ev.append(dict(sym=sym, date=dt, rsr=rv, si=sv, comp=cv,
                               type=ct, passed=passed, reason=reason))
    seen = {}; promos = []
    for ev in sorted((e for e in all_ev if e['passed']), key=lambda x: x['date']):
        sym = ev['sym']; prev = seen.get(sym)
        if prev is None or sum(1 for d in tm if prev < d <= ev['date']) >= 2:
            f30 = fwd_ret(all_prices[sym], dates, ev['date'], 30)
            f60 = fwd_ret(all_prices[sym], dates, ev['date'], 60)
            promos.append(dict(sym=sym, date=ev['date'], type=ev['type'],
                               rsr=ev['rsr'], si=ev['si'], comp=ev['comp'],
                               fwd30=f30, fwd60=f60))
            seen[sym] = ev['date']
    f30 = [p['fwd30'] * 100 for p in promos if p['fwd30'] is not None]
    f60 = [p['fwd60'] * 100 for p in promos if p['fwd60'] is not None]
    eq_aug = overlay(eq_base, promos, all_prices, dates)
    mb = metrics(eq_base, ts, te)
    ma = metrics(eq_aug, ts, te)
    gw = sum(x for x in f30 if x > 0); gl = abs(sum(x for x in f30 if x < 0))
    pf = round(gw / gl, 2) if gw > 0 and gl > 0 else (None if gl == 0 and gw == 0 else float('inf'))
    # per-symbol stats for this fold
    sym_stats = {}
    for sym in TARGET_SYMS:
        sp = [p for p in promos if p['sym'] == sym]
        s30 = [p['fwd30'] * 100 for p in sp if p['fwd30'] is not None]
        gw2 = sum(x for x in s30 if x > 0); gl2 = abs(sum(x for x in s30 if x < 0))
        sym_stats[sym] = dict(
            n=len(sp),
            fwd30_mean=round(sum(s30) / len(s30), 2) if s30 else None,
            fwd30_wr=round(sum(1 for x in s30 if x > 0) / len(s30) * 100, 1) if s30 else None,
            fwd30_pf=round(gw2 / gl2, 2) if gw2 > 0 and gl2 > 0 else None,
        )
    # SI distribution of passed promotions
    si_vals = [p['si'] for p in promos]
    return dict(
        n_promotions=len(promos),
        n_syms=len(set(p['sym'] for p in promos)),
        fwd30_mean=round(statistics.mean(f30), 2) if f30 else None,
        fwd30_wr=round(sum(1 for x in f30 if x > 0) / len(f30) * 100, 1) if f30 else None,
        fwd30_pf=pf, fwd30_n=len(f30),
        fwd60_mean=round(statistics.mean(f60), 2) if f60 else None,
        fwd60_wr=round(sum(1 for x in f60 if x > 0) / len(f60) * 100, 1) if f60 else None,
        base_cagr=mb['cagr'], base_dd=mb['max_dd'], base_sharpe=mb['sharpe'], base_calmar=mb['calmar'],
        aug_cagr=ma['cagr'],  aug_dd=ma['max_dd'],  aug_sharpe=ma['sharpe'],  aug_calmar=ma['calmar'],
        delta_cagr=round(ma['cagr']   - mb['cagr'],   2),
        delta_dd  =round(ma['max_dd'] - mb['max_dd'], 2),
        delta_sharpe=round(ma['sharpe'] - mb['sharpe'], 3),
        delta_calmar=round(ma['calmar'] - mb['calmar'], 3),
        sym_stats=sym_stats,
        si_vals=si_vals,
        promotions=promos,
    )

def run_case(case_name, gate_fn, all_prices, dates, rsr_all, si_all, comp_all, monthly, eq_base):
    folds = []
    for fold in WF_FOLDS:
        r = run_fold(gate_fn, all_prices, dates, rsr_all, si_all, comp_all, monthly, eq_base, fold)
        folds.append({"fold": fold['name'], "test_start": fold['test_start'],
                      "test_end": fold['test_end'], "r": r})
    dc = [f['r']['delta_cagr']   for f in folds]
    dd = [f['r']['delta_dd']     for f in folds]
    ds = [f['r']['delta_sharpe'] for f in folds]
    dm = [f['r']['delta_calmar'] for f in folds]
    n_tot = sum(f['r']['n_promotions'] for f in folds)
    return dict(case=case_name, folds=folds, n_total=n_tot,
                agg=dict(
                    dc=dc, dd=dd, ds=ds, dm=dm,
                    avg_dc=round(sum(dc)/len(dc), 2),
                    avg_dd=round(sum(dd)/len(dd), 2),
                    avg_ds=round(sum(ds)/len(ds), 3),
                    avg_dm=round(sum(dm)/len(dm), 3),
                    std_dc=round(statistics.stdev(dc), 2),
                    pos_folds=sum(1 for x in dc if x > 0),
                    pos_folds_ratio=f"{sum(1 for x in dc if x>0)}/{len(dc)}",
                ))

def main():
    print("Loading OHLCV...")
    all_syms = list(set(TARGET_SYMS + LIVE_SYMS))
    all_prices = {s: load_prices(s) for s in all_syms}
    all_prices = {k: v for k, v in all_prices.items() if v}
    dates = all_dates(all_prices)
    print(f"Dates: {dates[0]} → {dates[-1]}  ({len(dates)}d)")

    print("RSR..."); rsr_all = compute_rsr(all_prices, dates)
    print("SI ...");  si_all  = compute_si(all_prices, dates)
    print("Comp..."); comp_all = compute_comp(all_prices, dates)
    monthly = sorted(set(rsr_all) & set(si_all))
    print(f"Monthly: {monthly[0]} → {monthly[-1]}  ({len(monthly)})")

    live_syms = [s for s in LIVE_SYMS if s in all_prices and s not in TARGET_SYMS and s != "5706.T"]
    eq_base = sim_base(all_prices, live_syms, dates, rsr_all)

    results = {}
    for case_name, params in CASES.items():
        print(f"\n=== {case_name} (SI=[{params['si_lo']},{params['si_hi']}) ===")
        gate_fn = make_gate(params['si_lo'], params['si_hi'])
        res = run_case(case_name, gate_fn, all_prices, dates,
                       rsr_all, si_all, comp_all, monthly, eq_base)
        results[case_name] = res
        agg = res['agg']
        print(f"  n={res['n_total']}  avg ΔCAGR={agg['avg_dc']:+.2f}pp "
              f"ΔDD={agg['avg_dd']:+.2f}pp ΔSharpe={agg['avg_ds']:+.3f} "
              f"ΔCalmar={agg['avg_dm']:+.3f}  std={agg['std_dc']:.2f}  "
              f"pos={agg['pos_folds_ratio']}")
        for f in res['folds']:
            r = f['r']
            print(f"    {f['test_start'][:4]}: ΔCAGR={r['delta_cagr']:+.2f}pp "
                  f"ΔSharpe={r['delta_sharpe']:+.3f} ΔDD={r['delta_dd']:+.2f}pp "
                  f"n={r['n_promotions']} fwd30={r['fwd30_mean']}% wr={r['fwd30_wr']}%")

    # ── 2021 Fold deep dive ──────────────────────────────────────────────────
    print("\n=== 2021 FOLD COMPARISON (deep dive) ===")
    for case_name, res in results.items():
        f2021 = next(f for f in res['folds'] if f['test_start'][:4] == '2021')
        r = f2021['r']
        si_lo = CASES[case_name]['si_lo']
        print(f"  {case_name} (SI≥{si_lo:.0f}): "
              f"n={r['n_promotions']} fwd30={r['fwd30_mean']}% wr={r['fwd30_wr']}% "
              f"ΔCAGR={r['delta_cagr']:+.2f}pp")
        for sym, ss in r['sym_stats'].items():
            if ss['n'] > 0:
                print(f"    {sym}: n={ss['n']} fwd30={ss['fwd30_mean']}% wr={ss['fwd30_wr']}%")
        if r['si_vals']:
            si_sorted = sorted(r['si_vals'])
            print(f"    SI range: {si_sorted[0]:.1f}–{si_sorted[-1]:.1f} "
                  f"mean={sum(r['si_vals'])/len(r['si_vals']):.1f}")

    # ── Fold × Case ΔCAGR matrix ─────────────────────────────────────────────
    print("\n=== FOLD × CASE ΔCAGR MATRIX ===")
    years = [f['test_start'][:4] for f in next(iter(results.values()))['folds']]
    header = f"  {'':6}" + "".join(f"{c:>12}" for c in results)
    print(header)
    for i, yr in enumerate(years):
        row = f"  {yr:6}"
        for case_name, res in results.items():
            v = res['folds'][i]['r']['delta_cagr']
            row += f"{v:>+11.2f}pp"
        print(row)
    # avg row
    row = f"  {'avg':6}"
    for case_name, res in results.items():
        v = res['agg']['avg_dc']
        row += f"{v:>+11.2f}pp"
    print(row)

    # ── ΔSharpe matrix ──────────────────────────────────────────────────────
    print("\n=== FOLD × CASE ΔSharpe MATRIX ===")
    print(header)
    for i, yr in enumerate(years):
        row = f"  {yr:6}"
        for case_name, res in results.items():
            v = res['folds'][i]['r']['delta_sharpe']
            row += f"{v:>+12.3f}"
        print(row)
    row = f"  {'avg':6}"
    for case_name, res in results.items():
        v = res['agg']['avg_ds']
        row += f"{v:>+12.3f}"
    print(row)

    # ── n昇格 matrix ────────────────────────────────────────────────────────
    print("\n=== FOLD × CASE 昇格件数 MATRIX ===")
    print(header)
    for i, yr in enumerate(years):
        row = f"  {yr:6}"
        for case_name, res in results.items():
            v = res['folds'][i]['r']['n_promotions']
            row += f"{v:>12}"
        print(row)
    row = f"  {'total':6}"
    for case_name, res in results.items():
        v = res['n_total']
        row += f"{v:>12}"
    print(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    criteria = [
        ("avg ΔCAGR", lambda r: r['agg']['avg_dc'], ">+0.3pp", lambda v: v > 0.3),
        ("avg ΔSharpe", lambda r: r['agg']['avg_ds'], "≥0", lambda v: v >= 0),
        ("avg ΔDD", lambda r: r['agg']['avg_dd'], "<+2.0pp", lambda v: v < 2.0),
        ("pos folds", lambda r: r['agg']['pos_folds'], "≥3/4", lambda v: v >= 3),
    ]
    for case_name, res in results.items():
        si_lo = CASES[case_name]['si_lo']
        print(f"\n  {case_name} (SI≥{si_lo:.0f}): n={res['n_total']}")
        all_pass = True
        for cname, getter, threshold, check in criteria:
            v = getter(res)
            ok = check(v)
            if not ok: all_pass = False
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {cname:15} = {v:+.3f}  (threshold {threshold})")
        print(f"    → {'ALL PASS' if all_pass else 'HAS FAIL'}")

    out = Path("C:/ai-trading/backtests/auto_promote_wf_p2b_results.json")
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        if isinstance(obj, float): return None if obj != obj else round(obj, 4)
        return obj
    out.write_text(json.dumps(clean(results), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved: {out}")
    return results

if __name__ == "__main__":
    main()
