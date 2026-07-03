"""
AUTO_PROMOTE_V2 Gate Validation Backtest
Cases A-E: sector_ignition threshold × UNCLASSIFIED taxonomy
Period: 2018-2024 (OHLCV data available)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import defaultdict
import json
import statistics
import math

# ── Configuration ─────────────────────────────────────────────────────────────
OHLCV_DIR = Path("C:/ai-trading/data/ohlcv")
CAPITAL    = 3_000_000
SLIPPAGE   = 0.001
COMMISSION = 0.00055
MAX_POS    = 3
ALLOC_MUL  = 0.25   # probation allocation multiplier
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# Shadow candidates (currently blocked from live)
SHADOW_SYMS = ["5706.T", "6146.T", "6857.T", "6920.T", "8035.T"]

# Live baseline universe (G29_V2 intersected with OHLCV data)
LIVE_SYMS = [
    "8035.T","6702.T","6501.T","6762.T","6920.T","6857.T","6594.T",
    "7203.T","7201.T","9432.T","8306.T","8411.T","8309.T","7182.T",
    "8725.T","8058.T","8053.T","8002.T","8001.T","4021.T","2914.T",
    "7011.T","3382.T","5401.T","5411.T","9531.T","9101.T","9104.T",
]

# Sector mapping (known + extended for shadow candidates)
SECTOR_MAP = {
    # 電機精密
    "8035.T":"電機精密","6645.T":"電機精密","6762.T":"電機精密",
    "6920.T":"電機精密","6857.T":"電機精密","6594.T":"電機精密",
    "6146.T":"電機精密","6479.T":"電機精密","6506.T":"電機精密",
    "6586.T":"電機精密","6724.T":"電機精密","6841.T":"電機精密",
    "6869.T":"電機精密","6902.T":"電機精密","6981.T":"電機精密",
    # 電機
    "6702.T":"電機","6501.T":"電機","6752.T":"電機","6758.T":"電機",
    "6701.T":"電機","6503.T":"電機","6504.T":"電機",
    # 機械
    "7011.T":"機械","6301.T":"機械","6326.T":"機械","6273.T":"機械",
    "6471.T":"機械","6201.T":"機械","6367.T":"機械","7013.T":"機械",
    "7012.T":"機械",
    # 輸送機器
    "7203.T":"輸送機器","7201.T":"輸送機器",
    # 銀行
    "8306.T":"銀行","8411.T":"銀行","8309.T":"銀行","7182.T":"銀行",
    "8354.T":"銀行","6178.T":"銀行",
    # 保険
    "8725.T":"保険","8766.T":"保険","8750.T":"保険",
    # 商社
    "8058.T":"商社","8053.T":"商社","8002.T":"商社","8001.T":"商社",
    "8015.T":"商社",
    # 化学
    "4021.T":"化学","4063.T":"化学","4188.T":"化学","4452.T":"化学",
    "4901.T":"化学","4911.T":"化学","4004.T":"化学","3407.T":"化学",
    # 食品
    "2914.T":"食品","2801.T":"食品","2802.T":"食品","2503.T":"食品",
    "2502.T":"食品","2269.T":"食品","2282.T":"食品","2002.T":"食品",
    # 情報通信
    "9432.T":"情報通信","4307.T":"情報通信","4055.T":"情報通信",
    "6098.T":"情報通信",
    # 小売
    "3382.T":"小売","3099.T":"小売","3197.T":"小売","3289.T":"小売",
    # 鉄鋼
    "5401.T":"鉄鋼","5411.T":"鉄鋼",
    # 非鉄金属
    "5706.T":"非鉄金属","5713.T":"非鉄金属","5714.T":"非鉄金属",
    # 海運
    "9101.T":"海運","9104.T":"海運","9107.T":"海運",
    # ガス
    "9531.T":"ガス",
    # 医薬品
    "4503.T":"医薬品","4507.T":"医薬品","4519.T":"医薬品",
    "4523.T":"医薬品","4543.T":"医薬品","4568.T":"医薬品",
    "4578.T":"医薬品","4661.T":"医薬品",
    # その他
    "3861.T":"紙パルプ","3402.T":"化学","3861.T":"紙パルプ",
    "5201.T":"ガラス土石","5333.T":"ガラス土石",
    "4151.T":"医薬品","4503.T":"医薬品",
    "1605.T":"石油","5020.T":"石油","5108.T":"化学",
    "1925.T":"建設","1928.T":"建設","1801.T":"建設","1802.T":"建設",
    "1803.T":"建設","1812.T":"建設","1721.T":"建設",
}

# AUTO_PROMOTE_V2 taxonomy thresholds (from source)
MATURE_LEADER_MIN_RSR         = 87.0
MATURE_LEADER_MAX_COMPRESSION = 35.0
MEAN_REVERSION_MAX_RSR        = 55.0
MEAN_REVERSION_MAX_IGNITION   = 55.0
IGNITION_MIN_SECTOR_IGNITION  = 75.0
IGNITION_MAX_PREDICTIVE_RANK  = 5
IGNITION_MAX_RSR              = 85.0
IGNITION_MIN_DRIFT            = 35.0
CONTINUATION_MIN_RSR          = 70.0
CONTINUATION_MIN_COMPRESSION  = 28.0
CONTINUATION_MAX_RSR          = 88.0
HIGH_RSR_BYPASS_RSR           = 90.0

ALLOWED_TYPES = {"HIGH_RSR", "EARLY_IGNITION", "CONTINUATION"}

# Case definitions: (si_threshold, allow_unclassified)
CASES = {
    "A": (90.0, False),   # 現行
    "B": (0.0,  False),   # SI gate無効化
    "C": (50.0, False),   # SI=50
    "D": (90.0, True),    # UNCLASSIFIED許可
    "E": (50.0, True),    # SI=50 + UNCLASSIFIED許可
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_prices(sym: str) -> dict:
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

def trading_dates(all_prices: dict) -> list:
    dates = set()
    for p in all_prices.values():
        dates.update(p.keys())
    return sorted(d for d in dates if START_DATE <= d <= END_DATE)

# ── RSR computation ───────────────────────────────────────────────────────────
def compute_rsr_monthly(all_prices: dict, dates: list) -> dict:
    """Monthly RSR: 12-month momentum percentile rank (0-100)."""
    # Build monthly snapshots every ~21 trading days
    monthly_dates = dates[::21]
    rsr_by_date = {}
    for i, dt in enumerate(monthly_dates):
        if i < 12:
            continue
        dt_12m_ago = monthly_dates[i - 12]
        rets = {}
        for sym, prices in all_prices.items():
            p_now = prices.get(dt)
            # find closest prior date
            p_12m = prices.get(dt_12m_ago)
            if p_now and p_12m and p_12m > 0:
                rets[sym] = (p_now / p_12m - 1)
        if not rets:
            continue
        sorted_syms = sorted(rets.keys(), key=lambda s: rets[s])
        n = len(sorted_syms)
        rsr_by_date[dt] = {
            sym: round((rank / (n - 1)) * 100, 1) if n > 1 else 50.0
            for rank, sym in enumerate(sorted_syms)
        }
    return rsr_by_date

# ── Sector ignition computation ───────────────────────────────────────────────
def compute_sector_ignition_monthly(all_prices: dict, dates: list) -> dict:
    """Monthly sector ignition: reconstructed from OHLCV breadth + velocity."""
    monthly_dates = dates[::21]
    si_by_date = {}

    for i, dt in enumerate(monthly_dates):
        if i < 6:
            continue
        dt_5d_ago = monthly_dates[max(0, i-1)]  # approx 5 trading days ago

        # 5-day RSR velocity proxy: 5d return for each symbol
        vel_5d = {}
        for sym, prices in all_prices.items():
            p_now = prices.get(dt)
            p_5d  = prices.get(dt_5d_ago)
            if p_now and p_5d and p_5d > 0:
                vel_5d[sym] = (p_now / p_5d - 1) * 100

        # Breakout: price at 20-day high
        breakout = {}
        for sym, prices in all_prices.items():
            avail = [prices.get(d) for d in monthly_dates[max(0,i-4):i] if prices.get(d)]
            p_now = prices.get(dt)
            if avail and p_now:
                high_20d = max(avail)
                breakout[sym] = p_now >= high_20d * 0.98  # within 2% of high

        # Volume ratio proxy: none (no volume data), use price volatility as proxy
        # → set to 1.0 for all (neutral)

        # Sector-level aggregation
        sectors: dict = defaultdict(list)
        for sym in all_prices:
            sec = SECTOR_MAP.get(sym, "_unknown")
            sectors[sec].append(sym)

        sector_raw = {}
        for sec, syms in sectors.items():
            n = len(syms)
            if n == 0:
                sector_raw[sec] = 0.0
                continue
            adv = sum(1 for s in syms if vel_5d.get(s, 0) > 0)
            breadth = adv / n
            vels = [vel_5d.get(s, 0) for s in syms]
            mean_vel = sum(vels) / len(vels) if vels else 0
            vel_norm = max(0.0, min(1.0, (mean_vel + 2.5) / 5.0))
            bo = sum(1 for s in syms if breakout.get(s, False))
            bo_ratio = bo / n
            raw = 0.35 * breadth + 0.25 * vel_norm + 0.25 * bo_ratio + 0.15 * 0.5
            sector_raw[sec] = raw

        # Cross-sectional rank
        sym_raw = {
            sym: sector_raw.get(SECTOR_MAP.get(sym, "_unknown"), 0.0)
            for sym in all_prices
        }
        vals = sorted(sym_raw.values())
        n = len(vals)
        if n <= 1:
            si_by_date[dt] = {sym: 50.0 for sym in all_prices}
            continue

        si_ranked = {}
        for sym, v in sym_raw.items():
            rank = sum(1 for x in vals if x <= v)
            si_ranked[sym] = round((rank / n) * 100, 1)
        si_by_date[dt] = si_ranked

    return si_by_date

# ── Compression proxy ─────────────────────────────────────────────────────────
def compute_compression_monthly(all_prices: dict, dates: list) -> dict:
    """Compression proxy: 20d range / 60d range (0-100 scale)."""
    monthly_dates = dates[::21]
    comp_by_date = {}
    for i, dt in enumerate(monthly_dates):
        if i < 6:
            continue
        comp = {}
        for sym, prices in all_prices.items():
            avail_20 = [prices.get(d) for d in monthly_dates[max(0,i-2):i+1] if prices.get(d)]
            avail_60 = [prices.get(d) for d in monthly_dates[max(0,i-6):i+1] if prices.get(d)]
            if len(avail_20) >= 2 and len(avail_60) >= 3:
                r20 = (max(avail_20) - min(avail_20)) / min(avail_20) if min(avail_20) > 0 else 0
                r60 = (max(avail_60) - min(avail_60)) / min(avail_60) if min(avail_60) > 0 else 0
                ratio = r20 / r60 if r60 > 0 else 1.0
                comp[sym] = round(max(0, min(100, ratio * 50)), 1)
            else:
                comp[sym] = 50.0
        comp_by_date[dt] = comp
    return comp_by_date

# ── Candidate classification (mirrors auto_promote_safe_v2.py) ────────────────
def classify_candidate(sym, rsr, si, comp, predictive_rank=None):
    late_entry_risk = rsr >= MATURE_LEADER_MIN_RSR and comp < MATURE_LEADER_MAX_COMPRESSION

    if late_entry_risk:
        return "MATURE_LEADER"
    if rsr < MEAN_REVERSION_MAX_RSR and si < MEAN_REVERSION_MAX_IGNITION:
        return "MEAN_REVERSION"
    if (si >= IGNITION_MIN_SECTOR_IGNITION
            and rsr < IGNITION_MAX_RSR
            and (predictive_rank is None or predictive_rank <= IGNITION_MAX_PREDICTIVE_RANK)):
        return "EARLY_IGNITION"
    if rsr >= CONTINUATION_MIN_RSR and comp >= CONTINUATION_MIN_COMPRESSION and rsr < CONTINUATION_MAX_RSR:
        return "CONTINUATION"
    if rsr >= HIGH_RSR_BYPASS_RSR and si >= 90.0:
        return "HIGH_RSR"
    return "UNCLASSIFIED"

def gate_pass(sym, rsr, si, comp, si_threshold, allow_unclassified):
    cand_type = classify_candidate(sym, rsr, si, comp)
    fail = []
    # Gate 1: RSR >= 8
    if rsr < 8.0:
        fail.append("rsr_too_low")
    # Gate 2: predictive_rank (skip for RSR>=90)
    # (simplified: no predictive_rank available historically, skip)
    # Gate 3: SI >= threshold
    if si_threshold > 0 and si < si_threshold:
        fail.append(f"si:{si:.1f}<{si_threshold}")
    gate_ok = len(fail) == 0

    if not gate_ok:
        return False, cand_type
    if cand_type not in ALLOWED_TYPES and not (allow_unclassified and cand_type == "UNCLASSIFIED"):
        return False, cand_type
    return True, cand_type

# ── Forward return computation ─────────────────────────────────────────────────
def get_fwd_return(prices: dict, dates: list, entry_date: str, n_days: int = 30) -> float | None:
    idx = next((i for i, d in enumerate(dates) if d >= entry_date), None)
    if idx is None:
        return None
    p_entry = prices.get(dates[idx])
    # Find N trading days forward
    fwd_idx = idx + n_days
    if fwd_idx >= len(dates):
        return None
    p_fwd = prices.get(dates[fwd_idx])
    if p_entry and p_fwd and p_entry > 0:
        cost = SLIPPAGE + COMMISSION
        return (p_fwd / p_entry - 1) - cost
    return None

# ── Portfolio simulation ───────────────────────────────────────────────────────
def simulate_portfolio(
    all_prices: dict,
    baseline_syms: list,
    promoted_candidates: dict,  # {sym: [entry_date, exit_date, case]}
    dates: list,
) -> dict:
    """
    Simplified Fujiko-law portfolio simulation.
    Entry: RSR > 75 + price >= 20d high (proxy)
    Exit: RSR < 70 or 55d low break or hold > 60d
    Returns: {date: equity}
    """
    capital = float(CAPITAL)
    equity_curve = {}
    positions = {}  # {sym: {entry_date, entry_price, qty, is_probation}}
    cash = capital

    rsr_series = {}
    for sym in baseline_syms:
        p = all_prices.get(sym, {})
        rsr_series[sym] = {}

    # Pre-compute RSR for baseline
    rsr_by_date = compute_rsr_monthly(
        {s: all_prices[s] for s in baseline_syms if s in all_prices},
        dates
    )

    monthly_dates = set(rsr_by_date.keys())

    for di, dt in enumerate(dates):
        if dt < "2019-01-01":
            equity_curve[dt] = capital
            continue

        # Mark-to-market
        port_value = cash
        for sym, pos in positions.items():
            p_now = all_prices.get(sym, {}).get(dt, pos['entry_price'])
            port_value += pos['qty'] * p_now
        equity_curve[dt] = port_value

        # Monthly rebalance
        if dt not in monthly_dates:
            continue

        rsr = rsr_by_date.get(dt, {})

        # Exit checks
        to_close = []
        for sym, pos in positions.items():
            hold_days = sum(1 for d in dates if pos['entry_date'] <= d <= dt)
            rsr_val = rsr.get(sym, 50)
            if rsr_val < 70 or hold_days > 60:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            p_sell = all_prices.get(sym, {}).get(dt, pos['entry_price'])
            proceeds = pos['qty'] * p_sell * (1 - SLIPPAGE - COMMISSION)
            cash += proceeds

        # Entry signals
        current_syms = set(positions.keys())
        n_slots = MAX_POS - len(current_syms)
        if n_slots <= 0:
            continue

        # Find buy signals
        candidates_buy = [
            (sym, rsr.get(sym, 0))
            for sym in baseline_syms
            if sym not in current_syms
            and rsr.get(sym, 0) >= 75
            and sym in all_prices
            and all_prices[sym].get(dt)
        ]
        candidates_buy.sort(key=lambda x: -x[1])

        for sym, rsr_val in candidates_buy[:n_slots]:
            if cash <= 0:
                break
            alloc = min(cash * 0.33, cash)
            p_buy = all_prices[sym].get(dt, 0)
            if p_buy <= 0:
                continue
            qty = int(alloc / (p_buy * (1 + SLIPPAGE + COMMISSION)))
            if qty <= 0:
                continue
            cost = qty * p_buy * (1 + SLIPPAGE + COMMISSION)
            cash -= cost
            positions[sym] = {
                'entry_date': dt, 'entry_price': p_buy,
                'qty': qty, 'is_probation': False
            }

    return equity_curve

def equity_to_metrics(eq_curve: dict, start: str = "2019-01-01") -> dict:
    dates = sorted(d for d in eq_curve if d >= start)
    if len(dates) < 252:
        return {"cagr": 0, "max_dd": 0, "sharpe": 0}

    vals = [eq_curve[d] for d in dates]
    years = len(dates) / 252
    cagr = (vals[-1] / vals[0]) ** (1 / years) - 1

    peak = vals[0]
    max_dd = 0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    daily_rets = [(vals[i] / vals[i-1] - 1) for i in range(1, len(vals))]
    avg_ret = sum(daily_rets) / len(daily_rets)
    std_ret = statistics.stdev(daily_rets) if len(daily_rets) > 1 else 0.001
    sharpe = (avg_ret * 252) / (std_ret * math.sqrt(252)) if std_ret > 0 else 0

    return {
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 3),
    }

# ── Main analysis ──────────────────────────────────────────────────────────────
def main():
    print("Loading OHLCV data...")
    all_syms = list(set(SHADOW_SYMS + LIVE_SYMS))
    all_prices = {}
    for sym in all_syms:
        p = load_prices(sym)
        if p:
            all_prices[sym] = p

    dates = trading_dates(all_prices)
    print(f"Trading dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    print("Computing RSR...")
    rsr_by_date = compute_rsr_monthly(all_prices, dates)

    print("Computing sector ignition...")
    si_by_date = compute_sector_ignition_monthly(all_prices, dates)

    print("Computing compression...")
    comp_by_date = compute_compression_monthly(all_prices, dates)

    monthly_eval_dates = sorted(set(rsr_by_date.keys()) & set(si_by_date.keys()))
    print(f"Monthly eval dates: {monthly_eval_dates[0]} to {monthly_eval_dates[-1]} ({len(monthly_eval_dates)} dates)")

    # ── Case evaluation ────────────────────────────────────────────────────────
    results = {}

    for case_name, (si_thr, allow_unc) in CASES.items():
        promotions = []  # {sym, date, type, fwd30}
        type_counts = defaultdict(int)

        for dt in monthly_eval_dates:
            rsr = rsr_by_date.get(dt, {})
            si  = si_by_date.get(dt, {})
            comp = comp_by_date.get(dt, {})

            for sym in SHADOW_SYMS:
                rsr_val  = rsr.get(sym, 0)
                si_val   = si.get(sym, 0)
                comp_val = comp.get(sym, 50)

                passed, ctype = gate_pass(sym, rsr_val, si_val, comp_val, si_thr, allow_unc)
                type_counts[ctype] += 1

                if passed:
                    fwd = get_fwd_return(all_prices[sym], dates, dt, 30)
                    promotions.append({
                        "sym": sym, "date": dt, "type": ctype,
                        "rsr": rsr_val, "si": si_val, "comp": comp_val,
                        "fwd30": fwd,
                    })

        # Deduplicate: one promotion per symbol per 30-day window
        seen = {}
        deduped = []
        for pr in promotions:
            sym, dt = pr["sym"], pr["date"]
            if sym not in seen or (pr["date"] > seen[sym]):
                if sym not in seen or days_gap(seen[sym], dt) >= 30:
                    deduped.append(pr)
                    seen[sym] = dt
        promotions = deduped

        fwd_vals = [p["fwd30"] for p in promotions if p["fwd30"] is not None]
        n_pass_dates = len(promotions)
        n_syms = len(set(p["sym"] for p in promotions))

        # Average promotion interval per symbol
        sym_dates = defaultdict(list)
        for pr in promotions:
            sym_dates[pr["sym"]].append(pr["date"])
        avg_interval = None
        intervals = []
        for sym, dts in sym_dates.items():
            dts.sort()
            for i in range(1, len(dts)):
                gap = sum(1 for d in monthly_eval_dates if dts[i-1] < d <= dts[i])
                intervals.append(gap * 21)  # approx calendar days
        if intervals:
            avg_interval = round(statistics.mean(intervals))

        results[case_name] = {
            "si_threshold": si_thr,
            "allow_unclassified": allow_unc,
            "n_promotions": n_pass_dates,
            "n_unique_syms": n_syms,
            "avg_interval_days": avg_interval,
            "fwd30_mean": round(statistics.mean(fwd_vals) * 100, 2) if fwd_vals else None,
            "fwd30_median": round(statistics.median(fwd_vals) * 100, 2) if fwd_vals else None,
            "fwd30_win_rate": round(sum(1 for f in fwd_vals if f > 0) / len(fwd_vals) * 100, 1) if fwd_vals else None,
            "fwd30_n": len(fwd_vals),
            "type_dist": dict(type_counts),
            "promotions": promotions,
        }

    # ── Per-symbol forward return analysis ────────────────────────────────────
    sym_baseline_fwd = {}
    live_syms_in_data = [s for s in LIVE_SYMS if s in all_prices and s not in SHADOW_SYMS]
    for sym in live_syms_in_data:
        fwds = []
        for dt in monthly_eval_dates:
            rsr = rsr_by_date.get(dt, {}).get(sym, 0)
            if rsr >= 75:
                f = get_fwd_return(all_prices[sym], dates, dt, 30)
                if f is not None:
                    fwds.append(f * 100)
        if fwds:
            sym_baseline_fwd[sym] = {
                "mean": round(statistics.mean(fwds), 2),
                "median": round(statistics.median(fwds), 2),
                "win_rate": round(sum(1 for f in fwds if f > 0) / len(fwds) * 100, 1),
                "n": len(fwds),
            }

    # ── Simplified portfolio delta computation ─────────────────────────────────
    # For each Case: simulate adding promoted candidates at 25% allocation
    # vs baseline (no additional candidates)
    # Use monthly RSR signals for baseline + probation positions
    print("Computing portfolio metrics...")
    baseline_live = [s for s in LIVE_SYMS if s in all_prices and s not in SHADOW_SYMS]
    eq_baseline = simulate_portfolio(all_prices, baseline_live, {}, dates)

    portfolio_metrics = {}
    base_m = equity_to_metrics(eq_baseline)
    portfolio_metrics["Baseline"] = base_m

    for case_name, res in results.items():
        # Augmented: baseline + probation candidate entries
        # Simplified: add a fixed 8.3% allocation (ALLOC_MUL * 1/3) per promoted entry
        # at the promotion date, hold for 30bd
        eq_aug = dict(eq_baseline)
        for pr in res["promotions"]:
            if pr["fwd30"] is None:
                continue
            # Find entry and exit dates in main date list
            sym = pr["sym"]
            entry_dt = pr["date"]
            idx = next((i for i, d in enumerate(dates) if d >= entry_dt), None)
            if idx is None:
                continue
            exit_idx = min(idx + 30, len(dates) - 1)

            # Allocate from baseline equity
            alloc_pct = ALLOC_MUL / MAX_POS
            entry_eq = eq_baseline.get(entry_dt, CAPITAL)
            alloc = entry_eq * alloc_pct

            p_entry = all_prices[sym].get(dates[idx], 0)
            p_exit  = all_prices[sym].get(dates[exit_idx], 0)
            if p_entry > 0 and p_exit > 0:
                gross_ret = p_exit / p_entry - 1
                net_pnl = alloc * gross_ret - alloc * (SLIPPAGE + COMMISSION) * 2
                # Add P&L to equity from exit date onward
                exit_dt = dates[exit_idx]
                for d in dates:
                    if d >= exit_dt:
                        eq_aug[d] = eq_aug.get(d, CAPITAL) + net_pnl

        aug_m = equity_to_metrics(eq_aug)
        portfolio_metrics[case_name] = aug_m

    # ── Sector distribution ────────────────────────────────────────────────────
    sector_dist = {}
    for case_name, res in results.items():
        sd = defaultdict(int)
        for pr in res["promotions"]:
            sec = SECTOR_MAP.get(pr["sym"], "不明")
            sd[sec] += 1
        sector_dist[case_name] = dict(sd)

    # ── Year-by-year forward returns for UNCLASSIFIED ─────────────────────────
    unc_syms = ["5706.T","6857.T","8035.T"]
    unc_by_year = defaultdict(list)
    for sym in unc_syms:
        if sym not in all_prices:
            continue
        for dt in monthly_eval_dates:
            yr = dt[:4]
            rsr = rsr_by_date.get(dt, {}).get(sym, 0)
            if rsr >= 75:
                f = get_fwd_return(all_prices[sym], dates, dt, 30)
                if f is not None:
                    unc_by_year[yr].append(f * 100)

    return {
        "cases": results,
        "portfolio_metrics": portfolio_metrics,
        "sym_baseline_fwd": sym_baseline_fwd,
        "sector_dist": sector_dist,
        "unc_by_year": dict(unc_by_year),
        "baseline_live_syms": baseline_live,
        "monthly_eval_count": len(monthly_eval_dates),
    }

def days_gap(d1, d2):
    from datetime import date
    a = date.fromisoformat(d1)
    b = date.fromisoformat(d2)
    return abs((b - a).days)

if __name__ == "__main__":
    data = main()

    # Save for report generation
    out_path = Path("C:/ai-trading/backtests/auto_promote_gate_validation_raw.json")
    # Convert non-serializable types
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 4) if not (obj != obj) else None  # handle NaN
        return obj

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2)
    print(f"\nRaw data saved: {out_path}")

    # Quick summary
    print("\n=== QUICK SUMMARY ===")
    for case_name in ["A","B","C","D","E"]:
        r = data["cases"][case_name]
        pm = data["portfolio_metrics"].get(case_name, {})
        print(f"Case {case_name}: promotions={r['n_promotions']} syms={r['n_unique_syms']} "
              f"fwd30_mean={r['fwd30_mean']}% wr={r['fwd30_win_rate']}% "
              f"CAGR={pm.get('cagr','N/A')}% DD={pm.get('max_dd','N/A')}% "
              f"Sharpe={pm.get('sharpe','N/A')}")
    print(f"Baseline: CAGR={data['portfolio_metrics']['Baseline']['cagr']}% "
          f"DD={data['portfolio_metrics']['Baseline']['max_dd']}% "
          f"Sharpe={data['portfolio_metrics']['Baseline']['sharpe']}")
