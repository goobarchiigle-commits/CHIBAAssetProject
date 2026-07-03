"""sizing_audit_csv.py — 全候補銘柄のポジションサイジング監査 CSV 出力"""
import sys, json, csv
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ── パラメータ（直近 DRY ラン metrics から） ──────────────────────────────
CAPITAL           = 3_000_000   # bridge.capital (metrics["capital"])
MAX_SINGLE_WEIGHT = 0.25        # strategy.yaml
RISK_PCT          = 0.0125      # signal_bridge.py _risk_pct (1.25%)
LOT               = 100
LEADER_RSR        = 85.0        # _LEADER_RSR_THRESHOLD
LEADER_SLOT       = 0.35        # _LEADER_SLOT_WEIGHT
MIN_RSR           = 75.0        # fujiko.min_rsr

position_budget_normal = CAPITAL * MAX_SINGLE_WEIGHT   # 750,000
position_budget_leader = CAPITAL * LEADER_SLOT         # 1,050,000
risk_per_trade         = CAPITAL * RISK_PCT            # 30,000

# ── データロード ──────────────────────────────────────────────────────────
from src.paths import LIVE_UNIVERSE_FILE
universe_raw = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding='utf-8'))
symbols_map = universe_raw.get('symbols', universe_raw)  # {symbol: sector}

rsr_lines = Path('logs/diagnostics/rsr_distribution.jsonl').read_text(encoding='utf-8').strip().split('\n')
last_rsr_entry = json.loads(rsr_lines[-1])
rsr_scores = {item['symbol']: item['rsr'] for item in last_rsr_entry.get('top20', [])}

# portfolio_state (保有中・CB状態)
ps_data    = json.loads(Path('runtime/portfolio_state.json').read_text(encoding='utf-8'))
cb_active  = ps_data['cb_state'] in ('CB_ACTIVE', 'RECOVERY')
held_syms  = set(ps_data.get('position_qtys', {}).keys())

# ── OHLCV → Close / ATR20 計算 ───────────────────────────────────────────
OHLCV_DIR = Path('cache/ohlcv')

def load_close_atr(symbol: str):
    p = OHLCV_DIR / f'{symbol}.parquet'
    if not p.exists():
        return None, None
    try:
        df = pd.read_parquet(p)
        if df.empty or 'Close' not in df.columns:
            return None, None
        close = float(df['Close'].dropna().iloc[-1])
        # True Range
        hi, lo, cl = df['High'], df['Low'], df['Close'].shift(1)
        tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
        atr20_series = tr.rolling(20).mean()
        atr20 = float(atr20_series.dropna().iloc[-1]) if not atr20_series.dropna().empty else None
        return close, atr20
    except Exception:
        return None, None

# ── サイジング計算 ─────────────────────────────────────────────────────────
def compute_sizing(symbol, close, atr20, rsr):
    if close is None or close <= 0:
        return None

    lot_cost = close * LOT

    # leader slot
    is_leader = (rsr is not None and rsr >= LEADER_RSR)
    pb = position_budget_leader if is_leader else position_budget_normal

    # qty_cap
    qty_cap = int(pb // lot_cost) * 100

    # qty_risk (ATR-based)
    if atr20 and atr20 > 0:
        qty_raw  = int(risk_per_trade / atr20)
        qty_risk = (qty_raw // 100) * 100
        if qty_risk == 0:
            qty_risk = 100  # 最低1単元フォールバック
    else:
        qty_risk = None   # ATR 不明

    # qty_final
    if qty_cap == 0:
        qty_final           = 0
        binding_constraint  = 'price_too_high'
    elif qty_risk is None:
        qty_final           = qty_cap
        binding_constraint  = 'atr_unavailable'
    else:
        qty_final = min(qty_risk, qty_cap)
        if qty_risk <= qty_cap:
            binding_constraint = 'qty_risk'
        elif qty_cap < qty_risk:
            binding_constraint = 'qty_cap'
        else:
            binding_constraint = 'equal'

    return {
        'position_budget': pb,
        'qty_cap'        : qty_cap,
        'qty_risk'       : qty_risk,
        'qty_final'      : qty_final,
        'binding_constraint': binding_constraint,
    }

# ── 全銘柄ループ ──────────────────────────────────────────────────────────
rows = []
for symbol, sector in sorted(symbols_map.items()):
    close, atr20 = load_close_atr(symbol)
    rsr          = rsr_scores.get(symbol)
    sizing       = compute_sizing(symbol, close, atr20, rsr) if close else None

    # 状態フラグ
    currently_holding = symbol in held_syms
    rsr_pass          = (rsr is not None and rsr >= MIN_RSR)
    cb_blocked        = cb_active and rsr_pass and not currently_holding

    row = {
        'symbol'            : symbol,
        'sector'            : sector,
        'price'             : f'{close:.1f}' if close else '',
        'lot_cost'          : f'{close * 100:.0f}' if close else '',
        'ATR20'             : f'{atr20:.1f}' if atr20 else '',
        'RSR'               : f'{rsr:.1f}' if rsr else '',
        'rsr_pass'          : rsr_pass,
        'currently_holding' : currently_holding,
        'cb_blocked'        : cb_blocked,
        'position_budget'   : f'{sizing["position_budget"]:.0f}' if sizing else '',
        'qty_cap'           : str(sizing['qty_cap']) if sizing else '',
        'qty_risk'          : str(sizing['qty_risk']) if sizing and sizing['qty_risk'] is not None else '',
        'qty_final'         : str(sizing['qty_final']) if sizing else '',
        'est_amount'        : f'{sizing["qty_final"] * close:.0f}' if sizing and close else '',
        'binding_constraint': sizing['binding_constraint'] if sizing else 'no_price',
    }
    rows.append(row)

# ── ソート: RSR降順 → symbol ──────────────────────────────────────────────
def sort_key(r):
    try:
        rsr_val = float(r['RSR']) if r['RSR'] else -1
    except Exception:
        rsr_val = -1
    return (-rsr_val, r['symbol'])

rows.sort(key=sort_key)

# ── CSV 出力 ─────────────────────────────────────────────────────────────
OUT = Path('logs/sizing_audit.csv')
OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    'symbol', 'sector', 'price', 'lot_cost', 'ATR20', 'RSR',
    'rsr_pass', 'currently_holding', 'cb_blocked',
    'position_budget', 'qty_cap', 'qty_risk', 'qty_final',
    'est_amount', 'binding_constraint',
]
with OUT.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f'Written: {OUT}  ({len(rows)} rows)')

# ── コンソールサマリー ────────────────────────────────────────────────────
print()
print(f'  capital={CAPITAL:,}  max_single_weight={MAX_SINGLE_WEIGHT:.0%}  risk_pct={RISK_PCT:.0%}')
print(f'  position_budget(normal)=¥{position_budget_normal:,.0f}  (leader)=¥{position_budget_leader:,.0f}')
print(f'  risk_per_trade=¥{risk_per_trade:,.0f}  CB_ACTIVE={cb_active}')
print()
print(f'  {"symbol":<10} {"sector":<10} {"price":>8} {"ATR20":>7} {"RSR":>6} {"pb":>9} {"qcap":>6} {"qrisk":>6} {"qfin":>6} {"est_amt":>10}  binding')
print('  ' + '-' * 98)
for r in rows:
    rsr_str   = r['RSR'] or '-'
    price_str = r['price'] or '-'
    atr_str   = r['ATR20'] or '-'
    pb_str    = r['position_budget'] or '-'
    qc_str    = r['qty_cap'] or '-'
    qr_str    = r['qty_risk'] or '-'
    qf_str    = r['qty_final'] or '-'
    ea_str    = r['est_amount'] or '-'
    flag = ''
    if r['currently_holding']: flag = ' [HOLD]'
    elif r['cb_blocked']:       flag = ' [CB]'
    elif not r['rsr_pass']:     flag = ''
    print(
        f'  {r["symbol"]:<10} {r["sector"]:<10} {price_str:>8} {atr_str:>7} {rsr_str:>6}'
        f' {pb_str:>9} {qc_str:>6} {qr_str:>6} {qf_str:>6} {ea_str:>10}'
        f'  {r["binding_constraint"]}{flag}'
    )
