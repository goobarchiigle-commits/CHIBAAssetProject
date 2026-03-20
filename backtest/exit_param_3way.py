"""
backtest/exit_param_3way.py
エグジットパラメータ 3way 比較バックテスト

【検証仮説】
  PnL vs 保有日数分析で「11-15日バケットがマイナス」となった原因:
  → turtle_exit=10 が勝ちトレードを早期に刈っている

【3シナリオ】
  A (baseline): turtle_exit=10  ← 現行 V2 設定
  B (仮説):     turtle_exit=20  ← ホールドを延ばす
  C (ATR stop): turtle_exit=20 + entry_price - 2×ATR20 を下限ストップに追加
                （ATRベース動的ストップでノイズ損切りを減らしつつ大損を防ぐ）

【ATRストップの仕組み】
  エントリー時に ATR20 を記録 → stop_price = entry_price - 2 × ATR20
  終値 < stop_price → SELL_ATR_STOP（turtle_exitより先に発動することもある）
  turtle_exit=20 の N日安値割れも引き続き有効（どちらか先の方でエグジット）

【使い方】
  cd C:/Users/owner/.gemini/antigravity/scratch/asset_simulation
  python -m backtest.exit_param_3way
"""

from __future__ import annotations
import os, sys, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine, Position, _IDM_TABLE
from backtest.engine                  import TradeCost
from portfolio.volatility_allocator   import calc_atr

# ================================================================
# 設定
# ================================================================
START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
OUT_DIR = "C:/Users/owner/.claude/レポート"

UNIVERSE_FILE = "configs/universe/2026Q1_temporal24.json"

SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "fujiko", "機械":    "fujiko", "電機精密":"fujiko", "商社":   "fujiko",
    "電機":    "fujiko", "食品":    "fujiko", "情報通信":"dynamic","小売":   "dynamic",
    "医薬品":  "dynamic","不動産":  "dynamic","サービス":"dynamic","レジャー":"fujiko",
    "ガス":    "mean_rev","鉄鋼":   "mean_rev","銀行":   "mean_rev","保険":   "mean_rev",
    "輸送機器":"mean_rev","化学":   "mean_rev","陸運":   "mean_rev","石油":   "mean_rev",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

# ATRストップのデフォルトフォールバック（ATR計算不能時）
ATR_FALLBACK_PCT = 0.07


# ================================================================
# ポジション拡張（ATRストップ価格を保持）
# ================================================================
@dataclass
class ExtendedPosition(Position):
    """stop_price を追加したポジション（ATRストップ用）"""
    stop_price: float = 0.0


# ================================================================
# ATRストップ付きエンジン
# ================================================================
class AtrStopEngine(PortfolioEngine):
    """
    PortfolioEngine に ATRベース動的ストップを追加したエンジン。

    エントリー時に ATR20 を記録し、
    stop_price = entry_price - atr_mult × ATR20 を設定する。
    毎日の損切りチェックで「strategy signal OR stop_price割れ」を判定する。
    """

    def __init__(self, *args, atr_mult: float = 2.0, atr_period: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.atr_mult   = atr_mult
        self.atr_period = atr_period

    def _calc_stop_price(self, exec_price: float, past_df: pd.DataFrame) -> float:
        """ATR20 × atr_mult を entry_price から引いたストップ価格を返す。"""
        atr = calc_atr(past_df, period=self.atr_period)
        if np.isnan(atr) or atr <= 0:
            return exec_price * (1 - ATR_FALLBACK_PCT)
        return exec_price - self.atr_mult * atr

    # ------------------------------------------------------------------
    # run() をオーバーライド: エントリー時に stop_price を計算・保存
    #                         エグジット時に stop_price チェックを追加
    # ------------------------------------------------------------------
    def run(self) -> "PortfolioResult":  # type: ignore[override]
        from backtest.portfolio_engine import PortfolioResult

        cash        = self.capital
        positions: dict[str, ExtendedPosition] = {}
        equity_records: list[dict] = []
        trades:         list[dict] = []
        peak_equity = float(self.capital)
        cb_active   = False

        dates = self.all_dates

        for i, date in enumerate(dates):

            # 1. 時価評価
            mkt_value = sum(
                pos.market_value(self.universe[sym]["df"].loc[date, "Close"])
                for sym, pos in positions.items()
                if date in self.universe[sym]["df"].index
            )
            port_value  = cash + mkt_value
            peak_equity = max(peak_equity, port_value)
            current_dd  = (port_value - peak_equity) / peak_equity

            sector_held = list({p.sector for p in positions.values()})

            # 2. サーキットブレーカー
            if current_dd < -self.max_dd_limit and not cb_active:
                for sym in list(positions.keys()):
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_position(
                        sym, positions[sym], price, "CIRCUIT_BREAKER", date, cash, trades
                    )
                positions.clear()
                cb_active = True

            if cb_active and current_dd > -self.max_dd_limit * 0.5:
                cb_active = False

            # 3. 損切りチェック（ATRストップ + strategyシグナル）
            if not cb_active:
                for sym in list(positions.keys()):
                    pos    = positions[sym]
                    df_sym = self.universe[sym]["df"]
                    price  = float(df_sym.loc[date, "Close"])

                    # ATRストップ: entry_price - 2×ATR を下回ったら即決済
                    if price < pos.stop_price:
                        cash = self._close_position(
                            sym, pos, price, "ATR_STOP", date, cash, trades
                        )
                        del positions[sym]
                        continue

                    # 通常の戦略シグナル（turtle_exit N日安値 or RSRモメンタム）
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == -1:
                        cash = self._close_position(
                            sym, pos, price, "STOP", date, cash, trades
                        )
                        del positions[sym]

            # 4. エントリーチェック
            if not cb_active and i + 1 < len(dates):
                next_date = dates[i + 1]
                regime_ok = self._market_regime_ok(date)
                candidates = list(self.universe.items())

                for sym, info in candidates:
                    if sym in positions:
                        continue

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    if not regime_ok:
                        continue

                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal != 1:
                        continue

                    sector = info["sector"]
                    if not self._sector_ok(sector, positions):
                        continue

                    exec_price = df_sym.loc[next_date, "Open"] * (1 + self.cost.slippage_rate)

                    alloc = self._position_alloc()
                    alloc *= self._dd_scalar(current_dd)
                    qty   = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue

                    total_cost = qty * exec_price + max(
                        qty * exec_price * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    if total_cost > cash:
                        continue

                    # ATRストップ価格を計算して ExtendedPosition に保存
                    stop_price = self._calc_stop_price(exec_price, past)

                    cash -= total_cost
                    positions[sym] = ExtendedPosition(
                        symbol      = sym,
                        sector      = sector,
                        qty         = qty,
                        entry_price = exec_price,
                        entry_date  = next_date,
                        stop_price  = stop_price,
                    )
                    trades.append({
                        "date":        next_date,
                        "symbol":      sym,
                        "sector":      sector,
                        "side":        "BUY",
                        "qty":         qty,
                        "price":       exec_price,
                        "pnl":         0.0,
                        "stop_price":  stop_price,
                        "atr_at_entry": exec_price - stop_price,
                    })

            # 5. 記録
            equity_records.append({
                "date":        date,
                "value":       port_value,
                "dd":          current_dd,
                "n_positions": len(positions),
                "n_sectors":   len({p.sector for p in positions.values()}),
                "cb_active":   cb_active,
                "sectors_held":"",
                "regime_ok":   True,
            })

        rec_df = pd.DataFrame(equity_records).set_index("date")
        return PortfolioResult(
            equity_curve    = rec_df["value"],
            dd_series       = rec_df["dd"],
            n_positions     = rec_df["n_positions"],
            n_sectors       = rec_df["n_sectors"],
            cb_active       = rec_df["cb_active"],
            regime_ok       = rec_df["regime_ok"],
            trades          = pd.DataFrame(trades),
            initial_capital = self.capital,
            strategy_name   = "AtrStop",
            max_dd_limit    = self.max_dd_limit,
            min_sectors     = self.min_sectors,
        )


# ================================================================
# ヘルパー
# ================================================================
def build_strats(universe: dict, rsr_uni: pd.DataFrame, turtle_exit: int) -> dict:
    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule   = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0, mom_period=21,
                turtle_entry=20, turtle_exit=turtle_exit, use_turtle_entry=True,
            )
    return strats


def _common_engine_kwargs(universe, strats):
    return dict(
        universe=universe, strategy=strats, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=3,
        cost=TradeCost(), vol_target=0.0, max_single_weight=0.25,
        vol_window=20, use_idm=False, use_dd_scalar=False,
    )


def calmar(cagr: float, maxdd: float) -> float:
    return cagr / abs(maxdd) if maxdd != 0 else float("inf")


def avg_hold(trades_df: pd.DataFrame) -> float:
    pairs = []
    for sym, grp in trades_df.groupby("symbol"):
        buy_q = []
        for _, row in grp.sort_values("date").iterrows():
            if row["side"] == "BUY":
                buy_q.append(row["date"])
            elif row["side"].startswith("SELL") and buy_q:
                pairs.append((row["date"] - buy_q.pop(0)).days)
    return float(np.mean(pairs)) if pairs else float("nan")


def exit_breakdown(trades_df: pd.DataFrame) -> dict:
    """SELL系の内訳（side別件数）を返す。"""
    sells = trades_df[trades_df["side"].str.startswith("SELL")]
    return sells.groupby("side").size().to_dict()


def metrics(res, label: str) -> dict:
    cagr  = res.cagr
    maxdd = res.max_drawdown
    ah    = avg_hold(res.trades)
    bd    = exit_breakdown(res.trades)
    return {
        "label":    label,
        "cagr":     cagr,
        "maxdd":    maxdd,
        "calmar":   calmar(cagr, maxdd),
        "sharpe":   res.sharpe_ratio,
        "n_trades": res.n_trades,
        "avg_hold": ah,
        "win_rate": res.win_rate,
        "exit_bd":  bd,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    SEP = "=" * 76
    print(SEP)
    print("  エグジットパラメータ 3way 比較バックテスト")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("  A: turtle_exit=10 (現行)")
    print("  B: turtle_exit=20 (ホールド延長)")
    print("  C: turtle_exit=20 + ATR20×2 動的ストップ")
    print(SEP)

    # ---- 1. データ ----
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        tickers = json.load(f)["symbols"]

    print(f"\n[1/4] データ取得中（{len(tickers)}銘柄）...")
    universe = download_universe(tickers, start=START, end=END, verbose=False)
    prices   = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr      = calc_universe_rsr(prices)
    print(f"  取得完了: {len(universe)} 銘柄 / RSR計算: {rsr.shape[1]} 銘柄")

    # ---- 2. テスト A ----
    print("\n[2/4] テスト A: turtle_exit=10 (baseline)...")
    strats_a = build_strats(universe, rsr, turtle_exit=10)
    res_a = PortfolioEngine(**_common_engine_kwargs(universe, strats_a)).run()
    m_a   = metrics(res_a, "A: exit=10")
    print(f"  CAGR={m_a['cagr']*100:+.2f}%  MaxDD={m_a['maxdd']*100:.2f}%"
          f"  Calmar={m_a['calmar']:.3f}  Sharpe={m_a['sharpe']:.3f}"
          f"  avg_hold={m_a['avg_hold']:.1f}日  勝率={m_a['win_rate']*100:.1f}%")
    print(f"  exit内訳: {m_a['exit_bd']}")

    # ---- 3. テスト B ----
    print("\n[3/4] テスト B: turtle_exit=20 (ホールド延長)...")
    strats_b = build_strats(universe, rsr, turtle_exit=20)
    res_b = PortfolioEngine(**_common_engine_kwargs(universe, strats_b)).run()
    m_b   = metrics(res_b, "B: exit=20")
    print(f"  CAGR={m_b['cagr']*100:+.2f}%  MaxDD={m_b['maxdd']*100:.2f}%"
          f"  Calmar={m_b['calmar']:.3f}  Sharpe={m_b['sharpe']:.3f}"
          f"  avg_hold={m_b['avg_hold']:.1f}日  勝率={m_b['win_rate']*100:.1f}%")
    print(f"  exit内訳: {m_b['exit_bd']}")

    # ---- 4. テスト C ----
    print("\n[4/4] テスト C: turtle_exit=20 + ATR20×2 動的ストップ...")
    strats_c = build_strats(universe, rsr, turtle_exit=20)
    res_c = AtrStopEngine(
        **_common_engine_kwargs(universe, strats_c),
        atr_mult=2.0, atr_period=20,
    ).run()
    m_c   = metrics(res_c, "C: exit=20+ATR")
    print(f"  CAGR={m_c['cagr']*100:+.2f}%  MaxDD={m_c['maxdd']*100:.2f}%"
          f"  Calmar={m_c['calmar']:.3f}  Sharpe={m_c['sharpe']:.3f}"
          f"  avg_hold={m_c['avg_hold']:.1f}日  勝率={m_c['win_rate']*100:.1f}%")
    print(f"  exit内訳: {m_c['exit_bd']}")

    # ---- 5. 比較表 ----
    print(f"\n{SEP}")
    print("  比較サマリー")
    print(f"  {'指標':<14} {'A: exit=10':>12} {'B: exit=20':>12} {'C: exit=20+ATR':>14} {'B-A':>8} {'C-A':>8}")
    print("-" * 76)

    rows = [
        ("CAGR%",     m_a["cagr"]*100,     m_b["cagr"]*100,     m_c["cagr"]*100,     False),
        ("MaxDD%",    m_a["maxdd"]*100,     m_b["maxdd"]*100,    m_c["maxdd"]*100,    True),
        ("Calmar",    m_a["calmar"],        m_b["calmar"],       m_c["calmar"],       False),
        ("Sharpe",    m_a["sharpe"],        m_b["sharpe"],       m_c["sharpe"],       False),
        ("n_trades",  m_a["n_trades"],      m_b["n_trades"],     m_c["n_trades"],     None),
        ("avg_hold日",m_a["avg_hold"],      m_b["avg_hold"],     m_c["avg_hold"],     None),
        ("勝率%",     m_a["win_rate"]*100,  m_b["win_rate"]*100, m_c["win_rate"]*100, False),
    ]
    for name, va, vb, vc, lower_is_better in rows:
        try:
            diff_ba = vb - va
            diff_ca = vc - va
            if lower_is_better is None:
                ba_s = f"{diff_ba:+.1f}"
                ca_s = f"{diff_ca:+.1f}"
            elif lower_is_better:
                ba_s = f"{diff_ba:+.3f}{'↑' if diff_ba < 0 else '↓'}"
                ca_s = f"{diff_ca:+.3f}{'↑' if diff_ca < 0 else '↓'}"
            else:
                ba_s = f"{diff_ba:+.3f}{'↑' if diff_ba > 0 else '↓'}"
                ca_s = f"{diff_ca:+.3f}{'↑' if diff_ca > 0 else '↓'}"
            print(f"  {name:<14} {va:>12.3f} {vb:>12.3f} {vc:>14.3f} {ba_s:>8} {ca_s:>8}")
        except Exception:
            print(f"  {name:<14} {va!s:>12} {vb!s:>12} {vc!s:>14}")

    print(SEP)

    # ---- 6. 判定 ----
    best_label = max(
        [("A", m_a["calmar"]), ("B", m_b["calmar"]), ("C", m_c["calmar"])],
        key=lambda x: x[1]
    )
    print(f"\n  Calmarベスト: {best_label[0]}（{best_label[1]:.3f}）")

    if m_b["calmar"] > m_a["calmar"]:
        b_effect = f"✅ turtle_exit=20 有効: Calmar {m_a['calmar']:.3f}→{m_b['calmar']:.3f}"
    else:
        b_effect = f"→ turtle_exit=20 効果なし: Calmar {m_a['calmar']:.3f}→{m_b['calmar']:.3f}"
    print(f"  B vs A: {b_effect}")

    if m_c["calmar"] > m_b["calmar"]:
        c_effect = f"✅ ATRストップ追加効果あり: Calmar {m_b['calmar']:.3f}→{m_c['calmar']:.3f}"
    elif m_c["calmar"] > m_a["calmar"]:
        c_effect = f"△ ATRストップ: A より改善だが B より劣る ({m_c['calmar']:.3f})"
    else:
        c_effect = f"→ ATRストップ効果なし: Calmar {m_b['calmar']:.3f}→{m_c['calmar']:.3f}"
    print(f"  C vs B: {c_effect}")

    # ATRストップ発動比率（Test C）
    if not res_c.trades.empty:
        sells_c   = res_c.trades[res_c.trades["side"].str.startswith("SELL")]
        atr_exits = (sells_c["side"] == "SELL_ATR_STOP").sum()
        total_exits = len(sells_c)
        print(f"\n  [C] ATRストップ発動: {atr_exits}/{total_exits}回 "
              f"({atr_exits/total_exits*100:.0f}%)")
        # ATRストップのみの勝率
        atr_trades = sells_c[sells_c["side"] == "SELL_ATR_STOP"]
        if not atr_trades.empty:
            atr_wr = (atr_trades["pnl"] > 0).mean()
            atr_avg_pnl = atr_trades["pnl"].mean()
            print(f"  [C] ATRストップ 勝率={atr_wr*100:.0f}%  avg_pnl={atr_avg_pnl:+,.0f}円")

    print()

    # ---- 7. グラフ ----
    BG, PANEL, GRID = "#0d1117", "#161b22", "#21262d"
    TEXT, MUTED     = "#e6edf3", "#8b949e"
    COLORS = {"A": "#2ea043", "B": "#388bfd", "C": "#f0b429"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BG)
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    # ax0: 資産推移
    for label, res in [("A", res_a), ("B", res_b), ("C", res_c)]:
        eq = res.equity_curve / 10_000
        axes[0].plot(eq.index, eq.values, color=COLORS[label], linewidth=1.5,
                     label=f"{metrics(res, label)['label']} ({res.cagr*100:+.1f}%)")
    axes[0].set_title("資産推移 (万円)", color=TEXT, fontsize=11)
    axes[0].set_ylabel("万円", color=MUTED)
    axes[0].tick_params(colors=MUTED)
    axes[0].legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    axes[0].grid(color=GRID, linewidth=0.5, zorder=0)

    # ax1: DD推移
    for label, res in [("A", res_a), ("B", res_b), ("C", res_c)]:
        axes[1].plot(res.dd_series.index, res.dd_series.values * 100,
                     color=COLORS[label], linewidth=1.2, label=label)
    axes[1].axhline(-15, color="#da3633", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_title("ドローダウン (%)", color=TEXT, fontsize=11)
    axes[1].set_ylabel("DD%", color=MUTED)
    axes[1].tick_params(colors=MUTED)
    axes[1].legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    axes[1].grid(color=GRID, linewidth=0.5, zorder=0)

    # ax2: 指標バー比較
    metrics_keys  = ["CAGR%", "Calmar", "Sharpe"]
    metrics_vals  = {
        "A": [m_a["cagr"]*100, m_a["calmar"], m_a["sharpe"]],
        "B": [m_b["cagr"]*100, m_b["calmar"], m_b["sharpe"]],
        "C": [m_c["cagr"]*100, m_c["calmar"], m_c["sharpe"]],
    }
    x_pos = np.arange(len(metrics_keys))
    width = 0.25
    for j, (label, vals) in enumerate(metrics_vals.items()):
        axes[2].bar(x_pos + j * width, vals, width,
                    color=COLORS[label], label=label, alpha=0.9, zorder=3)
    axes[2].set_xticks(x_pos + width)
    axes[2].set_xticklabels(metrics_keys, color=MUTED)
    axes[2].set_title("主要指標比較", color=TEXT, fontsize=11)
    axes[2].tick_params(colors=MUTED)
    axes[2].legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
    axes[2].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    axes[2].axhline(0, color=MUTED, linewidth=0.5)

    # ax3: 平均保有日数 & 勝率
    labels_ = ["A", "B", "C"]
    ah_vals = [m_a["avg_hold"], m_b["avg_hold"], m_c["avg_hold"]]
    wr_vals = [m_a["win_rate"]*100, m_b["win_rate"]*100, m_c["win_rate"]*100]
    ax3b = axes[3].twinx()
    axes[3].bar(labels_, ah_vals, color=[COLORS[l] for l in labels_], alpha=0.7, zorder=3)
    ax3b.plot(labels_, wr_vals, color="#e6edf3", marker="o", linewidth=2, zorder=4)
    axes[3].set_title("avg_hold日 (棒) / 勝率% (線)", color=TEXT, fontsize=11)
    axes[3].set_ylabel("保有日数", color=MUTED)
    ax3b.set_ylabel("勝率%", color=MUTED)
    axes[3].tick_params(colors=MUTED)
    ax3b.tick_params(colors=MUTED)
    axes[3].grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    fig.suptitle(
        f"エグジットパラメータ 3way 比較  {START}〜{END}  (資本¥{CAPITAL//10000}万)",
        color=TEXT, fontsize=13, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "exit_param_3way.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  グラフ保存: {out_png}")

    # ---- 8. JSON保存 ----
    result = {
        "generated_at": pd.Timestamp.now().isoformat()[:19],
        "period":        f"{START}/{END}",
        "A_exit10":      {k: (v if not isinstance(v, (float, np.floating))
                              else round(float(v), 4))
                          for k, v in m_a.items() if k != "exit_bd"},
        "B_exit20":      {k: (v if not isinstance(v, (float, np.floating))
                              else round(float(v), 4))
                          for k, v in m_b.items() if k != "exit_bd"},
        "C_exit20_atr":  {k: (v if not isinstance(v, (float, np.floating))
                              else round(float(v), 4))
                          for k, v in m_c.items() if k != "exit_bd"},
        "exit_breakdown": {
            "A": m_a["exit_bd"],
            "B": m_b["exit_bd"],
            "C": m_c["exit_bd"],
        },
        "best": best_label[0],
    }
    os.makedirs("results", exist_ok=True)
    out_json = "results/exit_param_3way.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  データ保存: {out_json}")


if __name__ == "__main__":
    main()
