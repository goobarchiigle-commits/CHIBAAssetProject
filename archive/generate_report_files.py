"""
generate_report_files.py
D_IDM_sec1（Calmar 2.235達成シナリオ）のバックテストを実行し
レポート用ファイルを出力する

出力先: C:/Users/owner/.claude/レポート/
  - equity_curve.png
  - yearly_returns.csv
  - trades.csv
"""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

OUTPUT_DIR = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

SECTOR_STRATEGY = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}

print("=" * 60)
print("  レポート用ファイル生成")
print(f"  シナリオ: D_IDM_sec1（min_sectors=1, IDM有効）")
print(f"  期間: {START} 〜 {END} / 初期資本: ¥{CAPITAL:,}")
print("=" * 60)

# 1. 銘柄リスト
df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
sym_to_strategy = dict(zip(df_sel["symbol"], df_sel["strategy"]))

broadest = df_sel[(df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)]
broad_tickers = {r["symbol"]: r["sector"] for _, r in broadest.iterrows()}

print(f"\n[1/4] データ取得中（{len(broad_tickers)}銘柄）...")
universe_all = download_universe(broad_tickers, start=START, end=END)
print(f"  取得完了: {len(universe_all)} 銘柄")

# 2. RSR計算
print("\n[2/4] RSR計算中...")
universe_prices = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
rsr_universe    = calc_universe_rsr(universe_prices)
print(f"  完了: {rsr_universe.shape[1]} 銘柄")

# 3. D_IDM_sec1シナリオ実行
print("\n[3/4] バックテスト実行（D_IDM_sec1）...")
mask = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
tier_syms = set(df_sel[mask]["symbol"].tolist())
universe_t = {sym: info for sym, info in universe_all.items() if sym in tier_syms}

strat_dict = {}
for sym, info in universe_t.items():
    rsr_s = rsr_universe[sym] if sym in rsr_universe.columns else None
    lbl_  = sym_to_strategy.get(sym, "フジコ法")
    sector = info["sector"]
    rule = SECTOR_STRATEGY.get(sector, "dynamic")
    if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl_):
        strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
    else:
        strat_dict[sym] = FujikoStrategy(
            rsr_series=rsr_s,
            min_sepa=6, min_rsr=75.0,
            mom_period=21, turtle_entry=20, turtle_exit=10, use_turtle_entry=True,
        )

cost = TradeCost()
engine = PortfolioEngine(
    universe          = universe_t,
    strategy          = strat_dict,
    capital           = CAPITAL,
    max_dd_limit      = 0.15,
    min_sectors       = 1,
    max_positions     = 3,
    cost              = cost,
    vol_target        = 0.02,
    max_single_weight = 0.25,
    vol_window        = 20,
    use_idm           = True,
    use_dd_scalar     = False,
    clenow_scores     = None,
)
res = engine.run()

def calmar(r):
    dd = abs(r.max_drawdown)
    return r.cagr / dd if dd > 0 else float("inf")

c = calmar(res)
print(f"\n  CAGR:    {res.cagr*100:+.2f}%")
print(f"  MaxDD:   {res.max_drawdown*100:.2f}%")
print(f"  Calmar:  {c:.3f}")
print(f"  Sharpe:  {res.sharpe_ratio:.3f}")
print(f"  取引数:  {res.n_trades}回")
print(f"  勝率:    {res.win_rate*100:.1f}%")
print(f"  CB発動:  {res.n_circuit_breaker}回")
print(f"  最終資産: ¥{res.equity_curve.iloc[-1]:,.0f}")

# 4. ファイル出力
print("\n[4/4] ファイル出力中...")

# ---- equity_curve.png ----
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(
    f"シン・フジコ投資法 ポートフォリオバックテスト\n"
    f"D_IDM_sec1（27銘柄/max3/IDM/min_sectors=1）  2018〜2024\n"
    f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
    f"Calmar={c:.3f}  Sharpe={res.sharpe_ratio:.3f}",
    fontsize=11,
)

ax = axes[0]
ax.plot(res.equity_curve.index, res.equity_curve.values / 10_000,
        color="royalblue", linewidth=2.0, label=f"資産推移 (CAGR={res.cagr*100:+.1f}%)")
ax.axhline(CAPITAL / 10_000, color="gray", linestyle="--", linewidth=0.8, label="元本 ¥200万")
cb_days = res.equity_curve[res.cb_active].index
if len(cb_days):
    ax.scatter(cb_days, res.equity_curve[cb_days].values / 10_000,
               color="red", s=50, zorder=5, label=f"CB発動 ({res.n_circuit_breaker}回)")
ax.set_ylabel("資産（万円）")
ax.set_title("資産推移")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(res.dd_series.index, res.dd_series.values * 100, 0,
                color="tomato", alpha=0.55, label="ドローダウン")
ax.axhline(-15, color="darkred", linestyle="--", linewidth=1.2, label="CB閾値 -15%")
ax.axhline(-5.90, color="darkorange", linestyle=":", linewidth=1.0,
           label=f"最大DD {res.max_drawdown*100:.2f}%")
ax.set_ylabel("ドローダウン（%）")
ax.set_title("ドローダウン推移")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.fill_between(res.n_positions.index, res.n_positions.values,
                alpha=0.4, color="steelblue", label="保有銘柄数")
ax.axhline(res.n_positions.mean(), color="navy", linestyle="--", linewidth=0.8,
           label=f"平均 {res.n_positions.mean():.2f}銘柄")
ax.set_ylabel("保有銘柄数")
ax.set_title(f"保有銘柄数推移（平均={res.n_positions.mean():.2f}銘柄）")
ax.set_ylim(0, 5)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3)

plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  equity_curve.png → {png_path}")

# ---- yearly_returns.csv ----
years = list(range(2018, 2025))
yr_rows = []
eq = res.equity_curve
dd = res.dd_series

for yr in years:
    yr_eq = eq[eq.index.year == yr]
    yr_dd = dd[dd.index.year == yr]
    if yr_eq.empty:
        continue
    ret = yr_eq.iloc[-1] / yr_eq.iloc[0] - 1
    max_dd_yr = float(yr_dd.min())
    calmar_yr = ret / abs(max_dd_yr) if max_dd_yr < 0 else None
    yr_rows.append({
        "年": yr,
        "年間リターン(%)": round(ret * 100, 2),
        "年間最大DD(%)": round(max_dd_yr * 100, 2),
        "年間Calmar": round(calmar_yr, 2) if calmar_yr is not None else "∞",
        "期末資産(円)": int(yr_eq.iloc[-1]),
    })

# 全期間行を追加
total_ret = eq.iloc[-1] / eq.iloc[0] - 1
total_dd  = float(dd.min())
yr_rows.append({
    "年": "全期間",
    "年間リターン(%)": round(total_ret * 100, 2),
    "年間最大DD(%)": round(total_dd * 100, 2),
    "年間Calmar": round(c, 3),
    "期末資産(円)": int(eq.iloc[-1]),
})

df_yr = pd.DataFrame(yr_rows)
csv_yr_path = os.path.join(OUTPUT_DIR, "yearly_returns.csv")
df_yr.to_csv(csv_yr_path, index=False, encoding="utf-8-sig")
print(f"  yearly_returns.csv → {csv_yr_path}")
print("\n  年次リターン内訳:")
print(f"  {'年':<6} {'リターン':>10} {'MaxDD':>10} {'Calmar':>8}")
for row in yr_rows:
    print(f"  {str(row['年']):<6} {str(row['年間リターン(%)'])+' %':>10} "
          f"{str(row['年間最大DD(%)'])+' %':>10} {str(row['年間Calmar']):>8}")

# ---- trades.csv ----
if not res.trades.empty:
    df_trades = res.trades.copy()
    df_trades["date"] = pd.to_datetime(df_trades["date"]).dt.strftime("%Y-%m-%d")
    df_trades["price"] = df_trades["price"].round(1)
    df_trades["pnl"] = df_trades["pnl"].round(0).astype(int)
    csv_tr_path = os.path.join(OUTPUT_DIR, "trades.csv")
    df_trades.to_csv(csv_tr_path, index=False, encoding="utf-8-sig")
    print(f"\n  trades.csv → {csv_tr_path}  ({len(df_trades)} 行)")
else:
    print("\n  ⚠ trades が空のため trades.csv はスキップ")

print("\n完了。")
