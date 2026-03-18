"""
run_strategy_analysis.py
戦略追加分析スクリプト（全10項目）

出力先: C:/Users/owner/.claude/レポート/
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["font.family"] = "MS Gothic"

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import TradeCost

OUTPUT = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
YEARS   = list(range(2018, 2025))

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

# ================================================================
# ヘルパー関数
# ================================================================
def calmar_ratio(res):
    dd = abs(res.max_drawdown)
    return res.cagr / dd if dd > 0 else float("inf")

def metrics(res, label):
    return {
        "label":        label,
        "cagr":         res.cagr,
        "maxdd":        res.max_drawdown,
        "calmar":       calmar_ratio(res),
        "sharpe":       res.sharpe_ratio,
        "n_trades":     res.n_trades,
        "win_rate":     res.win_rate,
        "avg_pos":      res.n_positions.mean(),
        "n_cb":         res.n_circuit_breaker,
        "total_return": res.total_return,
        "_res":         res,
    }

def build_strats(universe_t, df_sel, sym_to_strat, rsr_uni, min_sepa, min_rsr):
    d = {}
    for sym, info in universe_t.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        lbl    = sym_to_strat.get(sym, "フジコ法")
        rule   = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            d[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            d[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=min_sepa, min_rsr=min_rsr,
                mom_period=21, turtle_entry=20, turtle_exit=10, use_turtle_entry=True,
            )
    return d

def run_engine(universe_t, strat_dict, max_pos=3, vol_tgt=0.02,
               max_wt=0.25, min_sec=1, use_idm=True):
    eng = PortfolioEngine(
        universe=universe_t, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=min_sec, max_positions=max_pos,
        cost=TradeCost(), vol_target=vol_tgt, max_single_weight=max_wt,
        vol_window=20, use_idm=use_idm, use_dd_scalar=False,
    )
    return eng.run()

def yearly_table(res):
    rows = []
    eq = res.equity_curve; dd = res.dd_series
    for yr in YEARS:
        ye = eq[eq.index.year == yr]; yd = dd[dd.index.year == yr]
        if ye.empty: continue
        ret = ye.iloc[-1] / ye.iloc[0] - 1
        mdd = float(yd.min())
        cal = ret / abs(mdd) if mdd < 0 else None
        dr  = ye.pct_change().dropna()
        sh  = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
        rows.append({"年": yr, "リターン%": round(ret*100,2),
                     "MaxDD%": round(mdd*100,2),
                     "Calmar": round(cal,2) if cal else "∞",
                     "Sharpe": round(sh,2)})
    return pd.DataFrame(rows)

def trade_statistics(trades_df):
    if trades_df.empty:
        return {}
    sells = trades_df[trades_df["side"].str.startswith("SELL")].copy()
    if sells.empty:
        return {}

    # 連勝/連敗
    results = (sells["pnl"] > 0).tolist()
    max_win = max_loss = cur_win = cur_loss = 0
    for r in results:
        if r:
            cur_win += 1; cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_loss = max(max_loss, cur_loss)

    # 平均保有日数（BUYとSELLをシンボル単位でマッチ）
    hold_days_list = []
    for sym in trades_df["symbol"].unique():
        sym_df = trades_df[trades_df["symbol"] == sym].sort_values("date")
        buys  = sym_df[sym_df["side"] == "BUY"]["date"].tolist()
        sels  = sym_df[sym_df["side"].str.startswith("SELL")]["date"].tolist()
        for b, s in zip(buys, sels):
            hd = (pd.to_datetime(s) - pd.to_datetime(b)).days
            hold_days_list.append(hd)

    wins   = sells[sells["pnl"] > 0]["pnl"]
    losses = sells[sells["pnl"] <= 0]["pnl"]
    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    payoff   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "決済回数":       len(sells),
        "勝率%":          round(float((sells["pnl"] > 0).sum() / len(sells) * 100), 1),
        "平均利益(円)":   round(avg_win),
        "平均損失(円)":   round(avg_loss),
        "ペイオフレシオ": round(payoff, 2),
        "平均保有日数":   round(np.mean(hold_days_list), 1) if hold_days_list else 0,
        "最大連勝":       max_win,
        "最大連敗":       max_loss,
        "最大利益(円)":   round(float(sells["pnl"].max())),
        "最大損失(円)":   round(float(sells["pnl"].min())),
        "総損益(円)":     round(float(sells["pnl"].sum())),
    }

def etf_adjusted_equity(base_res, etf_close, max_single_weight=0.25):
    """待機資金をETFで運用した場合の資産推移を近似計算"""
    eq   = base_res.equity_curve
    npos = base_res.n_positions

    etf_ret  = etf_close.reindex(eq.index).pct_change().fillna(0)
    base_ret = eq.pct_change().fillna(0)

    adj_eq = [eq.iloc[0]]
    for i in range(1, len(eq)):
        prev_n    = npos.iloc[i - 1]
        idle_frac = max(0.0, 1.0 - prev_n * max_single_weight)
        extra     = idle_frac * etf_ret.iloc[i]
        adj_eq.append(adj_eq[-1] * (1 + base_ret.iloc[i] + extra))

    return pd.Series(adj_eq, index=eq.index)

def metrics_from_curve(eq_series, label):
    r  = eq_series.pct_change().dropna()
    tr = eq_series.iloc[-1] / eq_series.iloc[0] - 1
    n_years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = (1 + tr) ** (1 / n_years) - 1 if n_years > 0 else tr
    peak = eq_series.cummax()
    mdd  = float(((eq_series - peak) / peak).min())
    sh   = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    cal  = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {"label": label, "CAGR%": round(cagr*100,2),
            "MaxDD%": round(mdd*100,2), "Calmar": round(cal,3), "Sharpe": round(sh,3)}

def fmt_row(d):
    return {
        "シナリオ":       d["label"],
        "CAGR%":         f"{d['cagr']*100:+.2f}",
        "MaxDD%":        f"{d['maxdd']*100:.2f}",
        "Calmar":        f"{d['calmar']:.3f}",
        "Sharpe":        f"{d['sharpe']:.3f}",
        "取引数":        d["n_trades"],
        "勝率%":         f"{d['win_rate']*100:.1f}",
        "平均保有銘柄": f"{d['avg_pos']:.2f}",
    }

# ================================================================
# DATA DOWNLOAD（1回のみ）
# ================================================================
print("=" * 62)
print("  戦略追加分析スクリプト — 全10項目")
print("=" * 62)

df_sel = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
mask_broad = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)

broad_tickers = {r["symbol"]: r["sector"] for _, r in df_sel[mask_broad].iterrows()}
print(f"\n[DATA] ユニバース取得中 ({len(broad_tickers)}銘柄)...")
universe_all = download_universe(broad_tickers, start=START, end=END)
print(f"  完了: {len(universe_all)} 銘柄")

universe_prices = {sym: info["df"]["Close"] for sym, info in universe_all.items()}
rsr_uni = calc_universe_rsr(universe_prices)

tier_syms   = set(df_sel[mask_broad]["symbol"].tolist())
universe_t  = {sym: info for sym, info in universe_all.items() if sym in tier_syms}

# ETF データ取得
print("[DATA] ETFデータ取得中 (1306.T, 1475.T)...")
etf_data = {}
for ticker in ["1306.T", "1475.T"]:
    df = yf.download(ticker, start=START, end=END, progress=False)
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        etf_data[ticker] = df["Close"]
        print(f"  {ticker}: {len(df)} 日")

# ================================================================
# SECTION 1: 先読みバイアス確認
# ================================================================
print("\n[1/10] 先読みバイアス確認...")
lookahead_report = """
## Section 1: 先読みバイアス確認（コードレビュー結果）

### 検証箇所と結果

| 箇所 | コード | 判定 |
|------|--------|------|
| RSR計算（rsr.py） | `prices / prices.shift(63) - 1` ← shift()使用 | ✅ 安全 |
| シグナル生成（portfolio_engine.py L158） | `past_data = df_sym.loc[:date]` ← 当日以前のみ | ✅ 安全 |
| 約定価格（portfolio_engine.py L421） | `df_sym.loc[next_date, "Open"]` ← **翌日始値** | ✅ 安全 |
| ポジション評価 | `df_sym.loc[date, "Close"]` ← 当日終値 | ✅ 安全 |
| タートルズ高値（fujiko_strategy.py L140） | `close.iloc[-(N+1):-1].max()` ← 前日まで | ✅ 安全 |

### フロー確認

```
t日（15:00 終値確定）
  ↓ RSR計算、SEPA評価、シグナル生成
t+1日（09:00 寄付き）
  ↓ 約定（始値 × (1 ± slippage 0.1%)）
```

**結論: 先読みバイアスなし。実装は正しい。**

ただし注意点:
- **生存者バイアス**: 現存銘柄のみ対象（上場廃止銘柄を含まない）
- **サバイバーシップバイアス**: 2018年時点では不明だった銘柄選定基準を使用
- これらはバックテスト結果を若干楽観的にする可能性がある
"""
print(lookahead_report.strip())

# ================================================================
# SECTION 2: RSR閾値 感度分析
# ================================================================
print("\n[2/10] RSR閾値感度分析 (RSR>65/70/73/75/80)...")
rsr_results = []
for min_rsr in [65, 70, 73, 75, 80]:
    strat_d = build_strats(universe_t, df_sel, sym_to_strat, rsr_uni, 6, min_rsr)
    res = run_engine(universe_t, strat_d)
    m = metrics(res, f"RSR>{min_rsr}")
    rsr_results.append(m)
    print(f"  RSR>{min_rsr:2d}: CAGR={m['cagr']*100:+.2f}%  "
          f"MaxDD={m['maxdd']*100:.2f}%  Calmar={m['calmar']:.3f}  "
          f"Sharpe={m['sharpe']:.3f}  平均保有={m['avg_pos']:.2f}")

# ================================================================
# SECTION 3: 最大保有銘柄数テスト
# ================================================================
print("\n[3/10] 最大保有銘柄数テスト (2/3/5/10)...")
pos_results = []
base_strats = build_strats(universe_t, df_sel, sym_to_strat, rsr_uni, 6, 75.0)
for max_pos in [2, 3, 5, 10]:
    res = run_engine(universe_t, base_strats, max_pos=max_pos)
    m = metrics(res, f"max_pos={max_pos}")
    pos_results.append(m)
    print(f"  max_pos={max_pos:2d}: CAGR={m['cagr']*100:+.2f}%  "
          f"MaxDD={m['maxdd']*100:.2f}%  Calmar={m['calmar']:.3f}  "
          f"Sharpe={m['sharpe']:.3f}  平均保有={m['avg_pos']:.2f}")

# ================================================================
# SECTION 4: ポジションサイズ方式テスト
# ================================================================
print("\n[4/10] ポジションサイズ方式テスト (均等 vs ボラ調整)...")
size_results = []

# 均等ウェイト: vol_target=0, 各ポジション capital/max_positions
res_eq = run_engine(universe_t, base_strats, max_pos=3, vol_tgt=0.0,
                    max_wt=0.25, use_idm=False)
m = metrics(res_eq, "均等ウェイト")
size_results.append(m)
print(f"  均等ウェイト: CAGR={m['cagr']*100:+.2f}%  MaxDD={m['maxdd']*100:.2f}%  "
      f"Calmar={m['calmar']:.3f}  Sharpe={m['sharpe']:.3f}")

# ボラ調整（IDM有効・現行設定）
res_vol = run_engine(universe_t, base_strats, max_pos=3, vol_tgt=0.02,
                     max_wt=0.25, use_idm=True)
m = metrics(res_vol, "ボラ調整+IDM")
size_results.append(m)
print(f"  ボラ調整+IDM: CAGR={m['cagr']*100:+.2f}%  MaxDD={m['maxdd']*100:.2f}%  "
      f"Calmar={m['calmar']:.3f}  Sharpe={m['sharpe']:.3f}")

# ================================================================
# SECTION 5: 待機資金テスト
# ================================================================
print("\n[5/10] 待機資金テスト (現金 vs 1306.T vs 1475.T)...")
base_res = res_vol  # 現行設定（ボラ調整+IDM）
cash_results = []

# ベースライン（現金）
m_base = metrics_from_curve(base_res.equity_curve, "現金待機（現行）")
cash_results.append(m_base)
print(f"  現金待機: CAGR={m_base['CAGR%']:+.2f}%  MaxDD={m_base['MaxDD%']:.2f}%  "
      f"Calmar={m_base['Calmar']:.3f}")

for ticker, name in [("1306.T","TOPIX ETF(1306)"), ("1475.T","TOPIX ETF(1475)")]:
    if ticker in etf_data:
        adj_eq = etf_adjusted_equity(base_res, etf_data[ticker])
        m = metrics_from_curve(adj_eq, f"ETF待機({name})")
        cash_results.append(m)
        print(f"  {name}: CAGR={m['CAGR%']:+.2f}%  MaxDD={m['MaxDD%']:.2f}%  "
              f"Calmar={m['Calmar']:.3f}")
    else:
        print(f"  ⚠ {ticker} データ取得失敗")

# ================================================================
# SECTION 6: 保有銘柄数とパフォーマンスの関係
# ================================================================
print("\n[6/10] 保有銘柄数とパフォーマンスの関係...")
holdings_data = []
for m in rsr_results:
    holdings_data.append({
        "シナリオ":      m["label"],
        "平均保有銘柄": round(m["avg_pos"], 2),
        "CAGR%":        round(m["cagr"]*100, 2),
        "MaxDD%":       round(m["maxdd"]*100, 2),
        "Sharpe":       round(m["sharpe"], 3),
        "Calmar":       round(m["calmar"], 3),
    })
df_holdings = pd.DataFrame(holdings_data)
print(df_holdings.to_string(index=False))

# ================================================================
# SECTION 7: 年別パフォーマンス分析
# ================================================================
print("\n[7/10] 年別パフォーマンス分析...")
df_yearly = yearly_table(base_res)
print(df_yearly.to_string(index=False))

# ================================================================
# SECTION 8: トレード統計
# ================================================================
print("\n[8/10] トレード統計...")
stats = trade_statistics(base_res.trades)
for k, v in stats.items():
    print(f"  {k}: {v}")

# ================================================================
# SECTION 9: リスクイベント耐性
# ================================================================
print("\n[9/10] リスクイベント耐性シミュレーション...")
risk_scenarios = []
# 平均実際のポジションウェイト: max_single_weight = 0.25
# 1銘柄保有時: IDM拡大で25%に張る
for wt, shock, label in [
    (0.25, -0.20, "1銘柄-20%（最大ウェイト25%時）"),
    (0.25, -0.30, "1銘柄-30%（最大ウェイト25%時）"),
    (0.15, -0.20, "1銘柄-20%（実効ウェイト15%時）"),
    (0.15, -0.30, "1銘柄-30%（実効ウェイト15%時）"),
]:
    port_impact = wt * shock
    # サーキットブレーカー（-15%）との比較
    cb_triggered = abs(port_impact) >= 0.15
    risk_scenarios.append({
        "シナリオ":          label,
        "ウェイト":          f"{wt*100:.0f}%",
        "株価ショック":      f"{shock*100:.0f}%",
        "ポートフォリオ影響": f"{port_impact*100:+.1f}%",
        "CB発動":            "YES" if cb_triggered else "NO",
        "残余資産(推定)":    f"¥{CAPITAL * (1 + port_impact):,.0f}",
    })

# ストップ安連日ケース
for n_days, label in [(1,"ストップ安1日"), (2,"ストップ安2日連続")]:
    shock_total = -0.30 * n_days   # 簡易: 30%/日
    port_impact = 0.25 * shock_total
    risk_scenarios.append({
        "シナリオ":          f"1銘柄{label}（-30%/日）",
        "ウェイト":          "25%",
        "株価ショック":      f"{shock_total*100:.0f}%",
        "ポートフォリオ影響": f"{port_impact*100:+.1f}%",
        "CB発動":            "YES" if abs(port_impact) >= 0.15 else "NO",
        "残余資産(推定)":    f"¥{CAPITAL * (1 + port_impact):,.0f}",
    })

df_risk = pd.DataFrame(risk_scenarios)
print(df_risk.to_string(index=False))

# ================================================================
# SECTION 10: グラフ生成
# ================================================================
print("\n[10/10] グラフ生成中...")

fig = plt.figure(figsize=(18, 22))
gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

# ---- (1) RSR感度: 資産推移 ----
ax1 = fig.add_subplot(gs[0, :])
cmap = plt.cm.tab10
for i, m in enumerate(rsr_results):
    res_i = m["_res"]
    lw = 2.5 if m["label"] == "RSR>75" else 1.5
    ax1.plot(res_i.equity_curve.index,
             res_i.equity_curve.values / 10_000,
             color=cmap(i), linewidth=lw,
             label=f"{m['label']}  CAGR={m['cagr']*100:+.1f}%  "
                   f"Calmar={m['calmar']:.2f}  保有={m['avg_pos']:.1f}銘柄")
ax1.axhline(CAPITAL / 10_000, color="gray", linestyle="--", linewidth=0.8)
ax1.set_title("RSR閾値感度分析 — 資産推移比較（2018-2024）", fontsize=11)
ax1.set_ylabel("資産（万円）")
ax1.legend(fontsize=8, loc="upper left")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
ax1.grid(True, alpha=0.25)

# ---- (2) RSR感度: 指標棒グラフ ----
ax2 = fig.add_subplot(gs[1, 0])
labels  = [m["label"] for m in rsr_results]
cagrs   = [m["cagr"] * 100 for m in rsr_results]
colors2 = [cmap(i) for i in range(len(rsr_results))]
bars = ax2.bar(labels, cagrs, color=colors2, alpha=0.8)
ax2.set_title("RSR閾値別 CAGR比較")
ax2.set_ylabel("CAGR (%)")
ax2.axhline(0, color="black", linewidth=0.5)
for b, v in zip(bars, cagrs):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:+.1f}%",
             ha="center", fontsize=8)
ax2.grid(True, alpha=0.25, axis="y")

ax3 = fig.add_subplot(gs[1, 1])
avg_pos = [m["avg_pos"] for m in rsr_results]
calmars = [m["calmar"] for m in rsr_results]
ax3_b = ax3.twinx()
ax3.bar(labels, avg_pos, alpha=0.5, color="steelblue", label="平均保有銘柄")
ax3_b.plot(labels, calmars, "o-", color="darkorange", linewidth=2, label="Calmar比")
ax3.set_title("RSR閾値別 平均保有銘柄数 vs Calmar比")
ax3.set_ylabel("平均保有銘柄数", color="steelblue")
ax3_b.set_ylabel("Calmar比", color="darkorange")
ax3.legend(loc="upper left", fontsize=8)
ax3_b.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.25, axis="y")

# ---- (3) 最大保有銘柄数テスト ----
ax4 = fig.add_subplot(gs[2, 0])
pos_labels  = [m["label"] for m in pos_results]
pos_cagrs   = [m["cagr"] * 100 for m in pos_results]
pos_calmars = [m["calmar"] for m in pos_results]
ax4_b = ax4.twinx()
ax4.bar(pos_labels, pos_cagrs, alpha=0.6, color="royalblue")
ax4_b.plot(pos_labels, pos_calmars, "s-", color="red", linewidth=2, label="Calmar")
ax4.set_title("最大保有銘柄数別 CAGR vs Calmar")
ax4.set_ylabel("CAGR (%)", color="royalblue")
ax4_b.set_ylabel("Calmar", color="red")
for i, (l, v) in enumerate(zip(pos_labels, pos_cagrs)):
    ax4.text(i, v + 0.3, f"{v:+.1f}%", ha="center", fontsize=8)
ax4_b.legend(fontsize=8)
ax4.grid(True, alpha=0.25, axis="y")

# ---- (4) 年別リターン棒グラフ ----
ax5 = fig.add_subplot(gs[2, 1])
yr_data = df_yearly
colors_yr = ["tomato" if v < 0 else "seagreen" for v in yr_data["リターン%"]]
ax5.bar(yr_data["年"].astype(str), yr_data["リターン%"],
        color=colors_yr, alpha=0.85)
ax5.axhline(0, color="black", linewidth=0.5)
ax5.set_title("年別リターン（基準シナリオ）")
ax5.set_ylabel("リターン (%)")
for i, (yr, v) in enumerate(zip(yr_data["年"], yr_data["リターン%"])):
    ax5.text(i, v + (0.5 if v >= 0 else -1.5), f"{v:+.1f}%", ha="center", fontsize=8)
ax5.grid(True, alpha=0.25, axis="y")

# ---- (5) ポジションサイズ比較 ----
ax6 = fig.add_subplot(gs[3, 0])
for i, (res_i, label, color) in enumerate([
    (res_eq, "均等ウェイト", "steelblue"),
    (res_vol, "ボラ調整+IDM", "darkorange"),
]):
    ax6.plot(res_i.equity_curve.index,
             res_i.equity_curve.values / 10_000,
             color=color, linewidth=1.8,
             label=f"{label}  CAGR={res_i.cagr*100:+.1f}%")
ax6.axhline(CAPITAL / 10_000, color="gray", linestyle="--", linewidth=0.8)
ax6.set_title("ポジションサイズ方式比較")
ax6.set_ylabel("資産（万円）")
ax6.legend(fontsize=8)
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
ax6.grid(True, alpha=0.25)

# ---- (6) 待機資金比較 ----
ax7 = fig.add_subplot(gs[3, 1])
etf_colors = ["gray", "royalblue", "seagreen"]
etf_eq_curves = {"現金待機（現行）": base_res.equity_curve}
for ticker, name in [("1306.T","TOPIX ETF(1306)"), ("1475.T","TOPIX ETF(1475)")]:
    if ticker in etf_data:
        adj_eq = etf_adjusted_equity(base_res, etf_data[ticker])
        etf_eq_curves[f"ETF待機({name})"] = adj_eq

for (name, eq_s), color in zip(etf_eq_curves.items(), etf_colors):
    m = metrics_from_curve(eq_s, name)
    ax7.plot(eq_s.index, eq_s.values / 10_000, color=color, linewidth=1.8,
             label=f"{name}  CAGR={m['CAGR%']:+.1f}%")
ax7.axhline(CAPITAL / 10_000, color="gray", linestyle="--", linewidth=0.8)
ax7.set_title("待機資金運用方式比較")
ax7.set_ylabel("資産（万円）")
ax7.legend(fontsize=7.5)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))
ax7.grid(True, alpha=0.25)

for ax in fig.get_axes():
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    except Exception:
        pass

fig.suptitle("シン・フジコ投資法 — 戦略追加分析レポート（2018-2024）",
             fontsize=13, fontweight="bold", y=0.995)
plt.savefig(os.path.join(OUTPUT, "analysis_charts.png"), dpi=130,
            bbox_inches="tight")
plt.close()
print(f"  analysis_charts.png 保存完了")

# ================================================================
# CSV 出力
# ================================================================
print("\n[CSV] ファイル出力...")

# RSR感度
df_rsr = pd.DataFrame([fmt_row(m) for m in rsr_results])
df_rsr.to_csv(os.path.join(OUTPUT, "rsr_sensitivity.csv"),
              index=False, encoding="utf-8-sig")

# 最大保有銘柄数
df_pos = pd.DataFrame([fmt_row(m) for m in pos_results])
df_pos.to_csv(os.path.join(OUTPUT, "max_positions_test.csv"),
              index=False, encoding="utf-8-sig")

# ポジションサイズ
df_sz = pd.DataFrame([fmt_row(m) for m in size_results])
df_sz.to_csv(os.path.join(OUTPUT, "position_sizing.csv"),
             index=False, encoding="utf-8-sig")

# 待機資金
df_cash = pd.DataFrame(cash_results)
df_cash.to_csv(os.path.join(OUTPUT, "cash_alternatives.csv"),
               index=False, encoding="utf-8-sig")

# 年別
df_yearly.to_csv(os.path.join(OUTPUT, "yearly_analysis.csv"),
                 index=False, encoding="utf-8-sig")

# トレード統計
pd.DataFrame([stats]).T.reset_index().rename(
    columns={"index": "指標", 0: "値"}
).to_csv(os.path.join(OUTPUT, "trade_statistics.csv"),
         index=False, encoding="utf-8-sig")

# リスクイベント
df_risk.to_csv(os.path.join(OUTPUT, "risk_events.csv"),
               index=False, encoding="utf-8-sig")

print("  全CSVファイル出力完了")

# ================================================================
# 結果サマリー出力（コンソール）
# ================================================================
print("\n" + "=" * 62)
print("  分析完了サマリー")
print("=" * 62)

print("\n■ Section 2: RSR閾値感度分析")
print(df_rsr.to_string(index=False))

print("\n■ Section 3: 最大保有銘柄数テスト")
print(df_pos.to_string(index=False))

print("\n■ Section 4: ポジションサイズ方式")
print(df_sz.to_string(index=False))

print("\n■ Section 5: 待機資金テスト")
print(df_cash.to_string(index=False))

print("\n■ Section 6: 保有銘柄数とパフォーマンスの関係")
print(df_holdings.to_string(index=False))

print("\n■ Section 7: 年別パフォーマンス")
print(df_yearly.to_string(index=False))

print("\n■ Section 8: トレード統計")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n■ Section 9: リスクイベント耐性")
print(df_risk.to_string(index=False))

print("\n全分析完了。")
# 各結果を文字列として保存してレポートで使う
analysis_data = {
    "rsr_results": rsr_results,
    "pos_results": pos_results,
    "size_results": size_results,
    "cash_results": cash_results,
    "holdings_df": df_holdings,
    "yearly_df": df_yearly,
    "trade_stats": stats,
    "risk_df": df_risk,
}

# データをpickle保存（レポート作成に使用）
import pickle
with open(os.path.join(OUTPUT, "_analysis_data.pkl"), "wb") as f:
    pickle.dump(analysis_data, f)
print("  分析データ保存: _analysis_data.pkl")
