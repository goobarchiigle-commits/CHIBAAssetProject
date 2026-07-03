"""capital_utilization_audit.py — Capital utilization audit tool (LIVE/DRY only)"""
import sys, json, math
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# 実データ取得
# ================================================================
ps = json.loads(Path('runtime/portfolio_state.json').read_text(encoding='utf-8'))

PRICE_6506 = ps['position_current_prices']['6506.T']
PRICE_6981 = ps['position_current_prices']['6981.T']
QTY_6506   = ps['position_qtys']['6506.T']
QTY_6981   = ps['position_qtys']['6981.T']

MARKET_VALUE   = PRICE_6506 * QTY_6506 + PRICE_6981 * QTY_6981
AVAILABLE_CASH = ps['available_cash']
ACTUAL_EQUITY  = AVAILABLE_CASH + MARKET_VALUE  # broker cash + positions (computed)

# ================================================================
# 資本階層パラメータ
# ================================================================
CASH_BUFFER       = 0.10
MAX_SINGLE_WEIGHT = 0.25
MAX_POSITIONS     = 3
SECTOR_CAP        = 0.25
CLUSTER_CAP       = 0.35
RISK_PCT          = 0.0125
REJECT_THRESHOLD  = 0.25
LOT               = 100
ASSUMED_CAGR      = 0.1626      # V2 OOS CAGR
ASSUMED_DD        = 0.0612      # V2 MaxDD

effective_capital  = ACTUAL_EQUITY
deployable_capital = effective_capital * (1 - CASH_BUFFER)
risk_adjusted      = deployable_capital  # scalars=1.0 assumption

SEP = '=' * 72

# ================================================================
# SECTION 1: 資本階層
# ================================================================
print(SEP)
print('  SECTION 1: 現在値 — 資本階層')
print(SEP)
rows_s1 = [
    ('actual_equity',         ACTUAL_EQUITY,    1.0),
    ('effective_capital',     effective_capital, effective_capital/ACTUAL_EQUITY),
    ('deployable_capital',    deployable_capital, deployable_capital/ACTUAL_EQUITY),
    ('risk_adjusted_capital', risk_adjusted,    risk_adjusted/ACTUAL_EQUITY),
]
for name, val, pct in rows_s1:
    print(f'  {name:<26} {val:>12,.0f}  ({pct:.1%})')
print()
print(f'  -- 実ポートフォリオ (portfolio_state.json) --')
print(f'  6506.T  {QTY_6506:>4}株  @{PRICE_6506:>8,.0f}  = {QTY_6506*PRICE_6506:>11,.0f}')
print(f'  6981.T  {QTY_6981:>4}株  @{PRICE_6981:>8,.0f}  = {QTY_6981*PRICE_6981:>11,.0f}')
print(f'  市場価値                           {MARKET_VALUE:>11,.0f}')
print(f'  可用現金                           {AVAILABLE_CASH:>11,.0f}')
print(f'  actual_equity (computed)           {ACTUAL_EQUITY:>11,.0f}')
print(f'  CB状態: {ps["cb_state"]}  cooldown_end: {ps["cb_cooldown_end_date"]}')

# ================================================================
# SECTION 2: Utilization Ratio
# ================================================================
print()
print(SEP)
print('  SECTION 2: Capital Utilization Ratio')
print(SEP)
util = MARKET_VALUE / ACTUAL_EQUITY
cash_ratio = AVAILABLE_CASH / ACTUAL_EQUITY
print(f'  capital_utilization_ratio = 実際投資額 / actual_equity')
print(f'    = {MARKET_VALUE:,.0f} / {ACTUAL_EQUITY:,.0f} = {util:.2%}')
print()
print(f'  {'ラベル':<24} {'円':>12}  {'比率':>8}')
print('  ' + '-' * 48)
print(f'  {'投資中 (deployed)'::<24} {MARKET_VALUE:>12,.0f}  {util:>8.2%}')
print(f'  {'現金 (idle cash)'::<24} {AVAILABLE_CASH:>12,.0f}  {cash_ratio:>8.2%}')
print(f'  {'合計 actual_equity'::<24} {ACTUAL_EQUITY:>12,.0f}  {"100.00%":>8}')

# ================================================================
# SECTION 3: 制約別分析
# ================================================================
print()
print(SEP)
print('  SECTION 3: 制約別分析 — ロック額・機会損失')
print(SEP)
print(f'  前提: CAGR想定={ASSUMED_CAGR:.2%} (V2 OOS), 各制約は独立評価')
print()

# 制約1: 10% cash buffer
c1_lock = ACTUAL_EQUITY * CASH_BUFFER
c1_opp  = c1_lock * ASSUMED_CAGR
print(f'  [C1] 10% cash buffer')
print(f'       ロック額     {c1_lock:>12,.0f}  (actual × 10%)')
print(f'       機会損失     {c1_opp:>12,.0f}/年')

# 制約2: max_single_weight=25%
c2_no_cap   = deployable_capital / MAX_POSITIONS       # 上限なし時
c2_with_cap = deployable_capital * MAX_SINGLE_WEIGHT   # 25%上限
c2_waste    = (c2_no_cap - c2_with_cap) * MAX_POSITIONS
c2_opp      = c2_waste * ASSUMED_CAGR
print()
print(f'  [C2] max_single_weight = 25%')
print(f'       上限なし per-pos  {c2_no_cap:>12,.0f}  (deployable ÷ 3 = 33.3%)')
print(f'       25%上限 per-pos   {c2_with_cap:>12,.0f}  (deployable × 25%)')
print(f'       差分 × 3pos       {c2_waste:>12,.0f}')
print(f'       機会損失          {c2_opp:>12,.0f}/年')

# 制約3: max_positions=3 (現在 2 保有 / 1 空き)
n_held   = len(ps['position_qtys'])
n_open   = MAX_POSITIONS - n_held
c3_idle  = n_open * c2_with_cap
c3_opp   = c3_idle * ASSUMED_CAGR
print()
print(f'  [C3] max_positions = 3  (現在 {n_held}保有 / {n_open}スロット空き)')
print(f'       空きスロットアイドル {c3_idle:>11,.0f}  ({n_open} × position_budget={c2_with_cap:,.0f})')
print(f'       機会損失            {c3_opp:>12,.0f}/年')

# 制約4: lot丸め (2銘柄の実ポジションから計算)
def lot_waste(budget, price):
    lc = price * LOT
    if lc <= 0: return 0, 0
    lots = int(budget // lc)
    return lots * lc, budget - lots * lc

d1, w1 = lot_waste(c2_with_cap, PRICE_6506)
d2, w2 = lot_waste(c2_with_cap, PRICE_6981)
c4_waste = w1 + w2
c4_opp   = c4_waste * ASSUMED_CAGR
print()
print(f'  [C4] lot丸め waste')
print(f'       6506.T (@{PRICE_6506:,.0f}): budget={c2_with_cap:,.0f} → deployed={d1:,.0f}  waste={w1:,.0f}')
print(f'       6981.T (@{PRICE_6981:,.0f}): budget={c2_with_cap:,.0f} → deployed={d2:,.0f}  waste={w2:,.0f}')
print(f'       合計 lot waste      {c4_waste:>12,.0f}')
print(f'       機会損失            {c4_opp:>12,.0f}/年')

# 制約5: sector_cap=25%
c5_worst  = c2_with_cap          # 2銘柄目同一セクターが丸ごとブロックされた場合
c5_opp    = c5_worst * ASSUMED_CAGR
print()
print(f'  [C5] sector_cap = 25% (同一セクター重複時 worst case)')
print(f'       2銘柄目ブロック時 機会損失  {c5_worst:>9,.0f}/エントリー')
print(f'                          ≈       {c5_opp:>9,.0f}/年 (1エントリー/年想定)')

# 制約6: CB_ACTIVE
c6_opp = (c3_idle + c4_waste) * ASSUMED_CAGR  # 空きスロット + lot waste が全部アイドル
print()
print(f'  [C6] CB_ACTIVE (回路遮断器: 2026-06-30 まで新規 BUY 停止)')
print(f'       影響額 (空きスロット+lot waste)  {c3_idle+c4_waste:>9,.0f}')
print(f'       機会損失               {c6_opp:>12,.0f}/年')

# 機会損失サマリー
print()
print(f'  -- 機会損失サマリー (円/年) --')
print(f'  {'制約':<30} {'ロック額':>12}  {'機会損失/年':>12}')
print('  ' + '-' * 58)
constraints = [
    ('C1: 10% cash buffer',       c1_lock, c1_opp),
    ('C2: max_single_weight=25%', c2_waste, c2_opp),
    ('C3: 空きスロット(1枠)',     c3_idle,  c3_opp),
    ('C4: lot丸め',               c4_waste, c4_opp),
    ('C5: sector_cap (worst)',     c5_worst, c5_opp),
    ('C6: CB_ACTIVE',             c3_idle+c4_waste, c6_opp),
]
for name, lock, opp in constraints:
    print(f'  {name:<30} {lock:>12,.0f}  {opp:>12,.0f}')

# ================================================================
# SECTION 4: 理論最大投資率
# ================================================================
print()
print(SEP)
print('  SECTION 4: 現在設定での理論最大投資率')
print(SEP)

# 全 3 スロット埋まった場合
avg_price  = (PRICE_6506 + PRICE_6981) / 2   # 代表株価
lot_cost_r = avg_price * LOT
pos_budget = deployable_capital * MAX_SINGLE_WEIGHT
lots_each  = int(pos_budget // lot_cost_r)
max_invest = MAX_POSITIONS * lots_each * lot_cost_r
max_util   = max_invest / ACTUAL_EQUITY
# 連続値（lot丸め前）
max_cont   = min(deployable_capital, MAX_POSITIONS * deployable_capital * MAX_SINGLE_WEIGHT)
max_cont_u = max_cont / ACTUAL_EQUITY

print(f'  代表株価: (6506: {PRICE_6506:,.0f} + 6981: {PRICE_6981:,.0f}) / 2 = {avg_price:,.0f}円')
print(f'  lot_cost = {lot_cost_r:,.0f}円')
print()
print(f'  連続値 (lot丸め前):')
print(f'    max_invest  = min(deployable, 3 × deployable × 25%)')
print(f'              = {max_cont:,.0f}')
print(f'    max_util    = {max_cont_u:.2%}')
print()
print(f'  lot丸め後:')
print(f'    lots/pos    = int({pos_budget:,.0f} / {lot_cost_r:,.0f}) = {lots_each} 単元')
print(f'    max_invest  = 3 × {lots_each} × {lot_cost_r:,.0f} = {max_invest:,.0f}')
print(f'    max_util    = {max_util:.2%}')
print()
print(f'  現在の実投資率:  {util:.2%}')
print(f'  理論最大:        {max_util:.2%}')
print(f'  未活用余地:      {max_util - util:.2%} pt')
print(f'  (主因: CB_ACTIVE による新規BUY停止 + {n_open}スロット空き)')

# ================================================================
# SECTION 5 & 6: max_single_weight 変更比較
# ================================================================
print()
print(SEP)
print('  SECTION 5-6: max_single_weight 変更シミュレーション')
print(SEP)

cases_w = [
    ('A (現状)', 0.25),
    ('B',        0.30),
    ('C',        0.33),
    ('D',        0.40),
]

# ヘッダー
h = f'  {"ケース":<10}  {"weight":>7}  {"pos_budget":>11}  {"lots(6506)":>10}  {"lots(6981)":>10}'
print(h)
h2 = f'  {"":10}  {"":7}  {"":11}  {"[6639]":>10}  {"[4864]":>10}'
print(h2)
print('  ' + '-' * 56)

case_results = []
for label, w in cases_w:
    pb = deployable_capital * w
    l1 = int(pb // (PRICE_6506 * LOT))
    l2 = int(pb // (PRICE_6981 * LOT))
    inv_6506 = l1 * PRICE_6506 * LOT
    inv_6981 = l2 * PRICE_6981 * LOT
    avg_inv  = (inv_6506 + inv_6981) / 2
    max_3pos = MAX_POSITIONS * avg_inv
    util_3   = max_3pos / ACTUAL_EQUITY
    # lot丸め前投資率
    util_cont = min(1.0, MAX_POSITIONS * w * (1 - CASH_BUFFER))
    case_results.append((label, w, pb, l1, l2, avg_inv, max_3pos, util_3, util_cont))
    print(f'  {label:<10}  {w:>7.0%}  {pb:>11,.0f}  {l1:>7}単元  {l2:>7}単元')

print()
print(f'  {"ケース":<10}  {"weight":>7}  {"想定投資率":>11}  {"平均pos":>12}  {"CAGR試算":>10}  {"DD試算":>8}')
print('  ' + '-' * 66)
for label, w, pb, l1, l2, avg_inv, max_3pos, util_3, util_cont in case_results:
    # CAGR試算: 投資率に比例してCAGRもスケール（粗い線形仮定、MAX_DDも同様にスケール）
    # ベース: ケースA の利率 per deployed unit から換算
    cagr_est  = ASSUMED_CAGR * (util_cont / (MAX_POSITIONS * 0.25 * (1-CASH_BUFFER)))
    dd_est    = ASSUMED_DD   * (util_cont / (MAX_POSITIONS * 0.25 * (1-CASH_BUFFER)))
    cagr_imp  = cagr_est - ASSUMED_CAGR
    print(f'  {label:<10}  {w:>7.0%}  {util_3:>11.2%}  {avg_inv:>12,.0f}  {cagr_est:>10.2%}  {dd_est:>8.2%}')

print()
print('  注: CAGR試算は deployed/capital 比例仮定。実際は銘柄スリッページ・集中リスク要因あり')
print('      バックテスト再検証必須。weight変更前に ASK_FIRST_ON_CHANGE 手続きが必要。')

# ================================================================
# SECTION 6 拡張: ケース別詳細比較表
# ================================================================
print()
print(SEP)
print('  SECTION 6: ケース別詳細比較 (代表2銘柄、ATR=150円想定)')
print(SEP)

ATR = 150
print(f'  {"":10}  {"weight":>7}  {"pos_budget":>11}  {"risk_budget":>11}  {"qty_risk":>9}  {"qty_cap":>8}  {"FINAL":>6}  {"投資額":>11}  {"utiliz.":>8}')
print('  ' + '-' * 95)
for label, w in cases_w:
    pb          = deployable_capital * w
    rb          = risk_adjusted * RISK_PCT
    qty_raw     = int(rb / ATR)
    qty_risk    = (qty_raw // 100) * 100 or 100
    for price_label, price in [('6506@6639', PRICE_6506), ('6981@4864', PRICE_6981)]:
        lc      = price * LOT
        qty_cap = int(pb // lc) * 100
        qty_f   = min(qty_risk, qty_cap) if qty_cap > 0 else 0
        inv     = qty_f * price
        util_s  = inv * MAX_POSITIONS / ACTUAL_EQUITY
        print(
            f'  {label:<10}  {w:>7.0%}  {pb:>11,.0f}  {rb:>11,.0f}  '
            f'{qty_risk:>8}株  {qty_cap:>7}株  {qty_f:>5}株  '
            f'{inv:>11,.0f}  {util_s:>8.2%}'
        )
    print()

print(SEP)
print('  KEY FINDINGS')
print(SEP)
print(f'  1. 現在 utilization = {util:.2%} (CB_ACTIVE + 1空きスロットが主因)')
print(f'  2. max_single_weight=25% での lot_cost jump:')
print(f'     - 6506.T (6,639円): 1単元 = 663,900 < budget={c2_with_cap:,.0f} → 1lot ONLY')
print(f'     - 6981.T (4,864円): 1単元 = 486,400 < budget={c2_with_cap:,.0f} → 1lot ONLY')
print(f'  3. 33%に上げると 6506.T: budget=1,002K → int(1,002K/664K)=1 (変わらず)')
print(f'                           6981.T: budget=1,002K → int(1,002K/486K)=2 (2lots!)')
pos_budget_33 = deployable_capital * 0.33
print(f'     6981.T 2lots threshold: deployable × w × {PRICE_6981*LOT/deployable_capital:.4f}')
print(f'       33%時 budget={pos_budget_33:,.0f} / lot_cost={PRICE_6981*100:,.0f} = {pos_budget_33/(PRICE_6981*100):.2f}')
print(f'  4. 最大利用率は weight=C(33%)/D(40%) で {min(1.0, MAX_POSITIONS*0.33*(1-CASH_BUFFER)):.2%} (連続値)')
print(f'  5. lot丸め後の実際の max util は銘柄価格に依存（今回={max_util:.2%}）')
