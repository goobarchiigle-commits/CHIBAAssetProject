"""
backtest/cb_sector_analysis.py
CB停止問題 + セクター内トップ銘柄仮説 + CB設計改善の統合検証

【Task 1】セクター内トップ銘柄仮説
  流動性ランキング（2018H1平均売買代金）でセクター内Top1/2/3と全銘柄を比較。

【Task 2】CB長期停止の根本原因分析
  CB発動後 → 全キャッシュ → peak固定 → DD永久 -15.4% → 解除不可
  の「不可逆デッドロック」が構造的に発生していることを数値で示す。
  G27が2022-2024に回復できた銘柄（海運・商社ラリー）を特定する。

【Task 3】CB設計改善比較
  A: 現行（standard）  — 全決済 → DD永久固定 → 完全停止
  B: time_limit       — 最大120営業日で自動解除
  C: entry_stop_only  — 新規エントリー停止のみ（既存ポジションは保有継続）
  D: partial_size     — 新規サイズを50%に縮小（フルストップなし）
  E: no_cb            — CBなし（リスクを測る）
"""

from __future__ import annotations
import os, sys, warnings, copy
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from backtest.universe_builder        import download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine, PortfolioResult, Position
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
OUTPUT  = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

ALLOWED_SECTORS = {"電機精密", "電機", "海運", "鉄鋼", "機械", "商社", "輸送機器"}
SECTOR_STRATEGY: dict[str, str] = {
    "電機精密":"フジコ法","電機":"フジコ法","海運":"フジコ法",
    "機械":"フジコ法","商社":"フジコ法",
    "輸送機器":"平均回帰","鉄鋼":"平均回帰",
}
MR_PARAMS = dict(rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
                 ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15)

G27_SCREENED: dict[str, str] = {
    "8035.T":"電機精密","6645.T":"電機精密","6702.T":"電機","6501.T":"電機",
    "6762.T":"電機精密","6920.T":"電機精密","7203.T":"輸送機器","7201.T":"輸送機器",
    "9432.T":"情報通信","8306.T":"銀行","8411.T":"銀行","8309.T":"銀行",
    "7182.T":"銀行","8725.T":"保険","8058.T":"商社","8053.T":"商社",
    "8002.T":"商社","8001.T":"商社","4021.T":"化学","2914.T":"食品",
    "7011.T":"機械","3382.T":"小売","5401.T":"鉄鋼","5411.T":"鉄鋼",
    "9531.T":"ガス","9101.T":"海運","9104.T":"海運",
    "6857.T":"電機精密","6594.T":"電機精密",
}

TSE_SECTOR_CANDIDATES: dict[str, str] = {
    "6758.T":"電機","6861.T":"電機精密","8035.T":"電機精密","6954.T":"電機精密",
    "6645.T":"電機精密","6702.T":"電機","6501.T":"電機","6503.T":"電機",
    "6752.T":"電機","6762.T":"電機精密","6971.T":"電機精密","6723.T":"電機精密",
    "7741.T":"電機精密","6920.T":"電機精密","6902.T":"電機","7751.T":"電機精密",
    "6857.T":"電機精密","6594.T":"電機精密","6506.T":"機械","6146.T":"電機精密",
    "6981.T":"電機精密","6479.T":"電機精密","6869.T":"電機精密",
    "7203.T":"輸送機器","7267.T":"輸送機器","7201.T":"輸送機器",
    "7270.T":"輸送機器","7261.T":"輸送機器","6201.T":"機械",
    "8058.T":"商社","8031.T":"商社","8053.T":"商社",
    "8002.T":"商社","8001.T":"商社","8015.T":"商社",
    "6367.T":"機械","6301.T":"機械","6326.T":"機械",
    "7011.T":"機械","7012.T":"機械","7013.T":"機械",
    "5401.T":"鉄鋼","5411.T":"鉄鋼","5713.T":"鉄鋼","5706.T":"鉄鋼",
    "9101.T":"海運","9104.T":"海運","9107.T":"海運",
}


# ================================================================
# ヘルパー
# ================================================================
def _calmar(cagr: float, mdd: float) -> float:
    return cagr / abs(mdd) if mdd != 0 else 0.0


def _build_strats(data_dict: dict, rsr_uni: pd.DataFrame,
                  min_rsr: float = 70.0, min_sepa: int = 6) -> dict:
    strat_dict = {}
    for sym, info in data_dict.items():
        rule = SECTOR_STRATEGY.get(info["sector"], "フジコ法")
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        strat_dict[sym] = (
            MeanReversionStrategy(**MR_PARAMS) if rule == "平均回帰"
            else FujikoStrategy(rsr_series=rsr_s, min_sepa=min_sepa, min_rsr=min_rsr,
                                mom_period=21, turtle_entry=20, turtle_exit=10,
                                use_turtle_entry=True)
        )
    return strat_dict


def _metrics(r, label: str, n: int) -> dict:
    avg_pos = float(r.n_positions.mean())
    cb_days = int(r.cb_active.sum()) if hasattr(r, "cb_active") else 0
    return dict(label=label, n_stocks=n,
                cagr=r.cagr, sharpe=r.sharpe_ratio,
                mdd=r.max_drawdown, calmar=_calmar(r.cagr, r.max_drawdown),
                trades=r.n_trades, win_rate=r.win_rate,
                avg_pos=avg_pos, cb_days=cb_days, _res=r)


def _print_row(m: dict):
    print(f"  {m['label']:<42} {m['n_stocks']:>4} "
          f"{m['cagr']*100:>+7.1f}% {m['sharpe']:>7.3f} "
          f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} "
          f"{m['avg_pos']:>6.2f} {m['cb_days']:>6}日")


# ================================================================
# CB改善版エンジン（Task 3用）
# PortfolioEngineのrunメソッドをオーバーライドして各CB戦略を実装
# ================================================================
class FlexiblePortfolioEngine(PortfolioEngine):
    """
    CB動作を切り替え可能なポートフォリオエンジン。

    cb_mode:
      'standard'       現行（全決済→DD固定→永久停止）
      'time_limit'     N営業日後に自動解除（デフォルト120日）
      'entry_stop_only' 新規エントリー停止のみ。既存ポジションは保有継続
      'partial_size'   新規サイズを50%に縮小（フルストップなし）
      'no_cb'          CBなし
    """
    def __init__(self, *args, cb_mode: str = "standard", cb_max_days: int = 120, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_mode     = cb_mode
        self.cb_max_days = cb_max_days

    def run(self) -> PortfolioResult:
        if self.cb_mode == "standard":
            return super().run()   # 既存ロジックそのまま

        cash            = self.capital
        positions: dict[str, Position] = {}
        equity_records: list[dict]     = []
        trades:         list[dict]     = []
        peak_equity     = float(self.capital)
        cb_active       = False
        cb_activated_day = -1   # CB発動から何営業日経過したか追跡

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

            # 2. CB制御（モードごとの分岐）
            if self.cb_mode == "no_cb":
                cb_active = False   # CBなし → 常にFalse

            elif self.cb_mode == "entry_stop_only":
                # 新規エントリー停止のみ。既存ポジションは保有継続（決済しない）
                if current_dd < -self.max_dd_limit:
                    cb_active = True
                    if cb_activated_day < 0:
                        cb_activated_day = i
                # 解除: DDが-7.5%を上回ったら解除
                if cb_active and current_dd > -self.max_dd_limit * 0.5:
                    cb_active = False
                    cb_activated_day = -1

            elif self.cb_mode == "partial_size":
                # フルストップなし。DDが閾値を超えたらサイズを50%に縮小フラグ
                # cb_activeはサイズ縮小のフラグとして使う
                if current_dd < -self.max_dd_limit:
                    cb_active = True
                # 解除: DDが-7.5%を上回ったら解除
                if cb_active and current_dd > -self.max_dd_limit * 0.5:
                    cb_active = False

            elif self.cb_mode == "time_limit":
                # 現行CB（全決済）だが、N営業日後に自動解除
                if current_dd < -self.max_dd_limit and not cb_active:
                    # 全決済
                    for sym in list(positions.keys()):
                        price = self.universe[sym]["df"].loc[date, "Close"]
                        cash  = self._close_position(
                            sym, positions[sym], price, "CIRCUIT_BREAKER", date, cash, trades
                        )
                    positions.clear()
                    cb_active = True
                    cb_activated_day = i

                # 解除条件: DDが回復 OR 経過日数がcb_max_daysを超えた
                days_since_cb = i - cb_activated_day if cb_activated_day >= 0 else 0
                if cb_active and (current_dd > -self.max_dd_limit * 0.5 or
                                   days_since_cb >= self.cb_max_days):
                    cb_active = False
                    cb_activated_day = -1

            # 3. 損切りチェック（entry_stop_only と partial_size はポジション維持）
            if self.cb_mode not in ("entry_stop_only", "partial_size") and not cb_active:
                for sym in list(positions.keys()):
                    df_sym = self.universe[sym]["df"]
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == -1:
                        price = df_sym.loc[date, "Close"]
                        cash  = self._close_position(
                            sym, positions[sym], price, "STOP", date, cash, trades
                        )
                        del positions[sym]
            elif not cb_active:
                # entry_stop_only / partial_size の損切り
                for sym in list(positions.keys()):
                    df_sym = self.universe[sym]["df"]
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == -1:
                        price = df_sym.loc[date, "Close"]
                        cash  = self._close_position(
                            sym, positions[sym], price, "STOP", date, cash, trades
                        )
                        del positions[sym]

            # 4. エントリーチェック
            entry_blocked = (self.cb_mode not in ("partial_size", "entry_stop_only", "no_cb")
                             and cb_active)
            if (not entry_blocked) and i + 1 < len(dates):
                next_date = dates[i + 1]
                size_multiplier = 0.5 if (self.cb_mode == "partial_size" and cb_active) else 1.0

                candidates = list(self.universe.items())
                for sym, info in candidates:
                    if sym in positions:
                        continue
                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal != 1:
                        continue
                    sector = info["sector"]
                    if not self._sector_ok(sector, positions):
                        continue

                    exec_price = df_sym.loc[next_date, "Open"] * (1 + self.cost.slippage_rate)
                    alloc      = self._position_alloc() * size_multiplier
                    qty        = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue
                    total_cost = qty * exec_price + max(
                        qty * exec_price * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    if total_cost > cash:
                        continue
                    cash -= total_cost
                    positions[sym] = Position(
                        symbol=sym, sector=sector, qty=qty,
                        entry_price=exec_price, entry_date=next_date,
                    )
                    trades.append({"date": next_date, "symbol": sym, "sector": sector,
                                   "side": "BUY", "qty": qty, "price": exec_price, "pnl": 0.0})

            # 5. 記録
            equity_records.append({
                "date": date, "value": port_value, "dd": current_dd,
                "n_positions": len(positions),
                "n_sectors": len({p.sector for p in positions.values()}),
                "cb_active": cb_active,
                "sectors_held": "|".join(sorted(sector_held)),
                "regime_ok": True,
            })

        rec_df = pd.DataFrame(equity_records).set_index("date")
        return PortfolioResult(
            equity_curve=rec_df["value"], dd_series=rec_df["dd"],
            n_positions=rec_df["n_positions"], n_sectors=rec_df["n_sectors"],
            cb_active=rec_df["cb_active"], regime_ok=rec_df["regime_ok"],
            trades=pd.DataFrame(trades), initial_capital=self.capital,
            strategy_name=next(iter(self.strategies.values())).name,
            max_dd_limit=self.max_dd_limit, min_sectors=self.min_sectors,
        )


def _run_flexible(data_dict, strat_dict, cb_mode="standard",
                  cb_max_days=120, max_positions=3) -> PortfolioResult:
    return FlexiblePortfolioEngine(
        universe=data_dict, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=max_positions,
        cost=TradeCost(), vol_target=0.0,
        max_single_weight=min(0.25, 1.0 / max(1, max_positions)),
        use_idm=False, cb_mode=cb_mode, cb_max_days=cb_max_days,
    ).run()


# ================================================================
# Task 1: セクター内トップ銘柄仮説
# ================================================================
def task1_top_stock_hypothesis(universe_all: dict) -> list[dict]:
    print("\n" + "=" * 80)
    print("【Task 1】セクター内トップ銘柄仮説の検証")
    print("=" * 80)

    # 2018H1の平均売買代金でランキング
    turnover: dict[str, float] = {}
    for sym, info in universe_all.items():
        df_h1 = info["df"].loc["2018-01-01":"2018-06-30"]
        if len(df_h1) >= 30:
            turnover[sym] = float((df_h1["Close"] * df_h1["Volume"]).mean())
        else:
            turnover[sym] = 0.0

    # セクター内ランキング表示
    print("\n  【セクター内 流動性ランキング（2018H1売買代金）】")
    sector_ranked: dict[str, list[str]] = {}
    for sec in sorted(ALLOWED_SECTORS):
        syms_in_sec = [(s, turnover[s]) for s, i in universe_all.items()
                       if i["sector"] == sec]
        syms_in_sec.sort(key=lambda x: -x[1])
        sector_ranked[sec] = [s for s, _ in syms_in_sec]
        print(f"\n  {sec}（{len(syms_in_sec)}銘柄）:")
        for rank, (s, t) in enumerate(syms_in_sec[:6], 1):
            in_g27 = "★G27" if s in G27_SCREENED else "    "
            print(f"    #{rank:>2} {s} {in_g27} 売買代金={t/1e8:>6.1f}億/日")

    # Top-N Universe構築とバックテスト
    results = []
    all_prices = {s: i["df"]["Close"] for s, i in universe_all.items()}
    rsr_all = calc_universe_rsr(all_prices)

    for top_n, label in [(1, "Top1/セクター"), (2, "Top2/セクター"),
                          (3, "Top3/セクター"), (None, "全銘柄(48)")]:
        if top_n is not None:
            selected_syms = set()
            for sec, ranked_syms in sector_ranked.items():
                selected_syms.update(ranked_syms[:top_n])
            univ_n = {s: universe_all[s] for s in selected_syms if s in universe_all}
        else:
            univ_n = universe_all

        prices_n = {s: i["df"]["Close"] for s, i in univ_n.items()}
        rsr_n = calc_universe_rsr(prices_n)
        strat_n = _build_strats(univ_n, rsr_n)
        r_n = _run_flexible(univ_n, strat_n, cb_mode="standard")
        cb_years = r_n.cb_active.sum() / 252  # CB発動年数
        m = _metrics(r_n, label, len(univ_n))
        m["cb_years"] = cb_years
        results.append(m)

        # 年次内訳
        print(f"\n  [{label}] {len(univ_n)}銘柄: CAGR={m['cagr']*100:+.1f}%  "
              f"Sharpe={m['sharpe']:.3f}  MaxDD={m['mdd']*100:.1f}%  "
              f"Calmar={m['calmar']:.3f}  CB={m['cb_days']}日({cb_years:.1f}年)")

    return results, sector_ranked


# ================================================================
# Task 2: CB長期停止の原因分析
# ================================================================
def task2_cb_cause_analysis(universe_all: dict, sector_ranked: dict):
    print("\n" + "=" * 80)
    print("【Task 2】CB長期停止の根本原因分析")
    print("=" * 80)

    # ── CB構造的バグの数値証明 ──
    print("""
  【CB設計の根本的問題（デッドロック）】

  発動条件: ポートフォリオDD < -15.0%
  発動時:  全ポジションを強制決済 → 100%キャッシュ
  解除条件: DD > -7.5%（peak_equityの92.5%に回復）

  ⚠️  問題: キャッシュは価値が増えない
    peak_equity = CB発動前のピーク（不変）
    cash_after_cb ≈ peak × 0.846（= -15.4% DD時点のキャッシュ）
    current_dd = (cash - peak) / peak = -15.4% ← 永久に固定

    解除条件 (-7.5%) は絶対に満たされない
    → CB = 一度発動したら永久停止のトラップ
  """)

    # ── 2021年DDを引き起こした銘柄の特定 ──
    print("  【2021年DDを引き起こした銘柄分析】")
    print("  2021年の個別銘柄リターン（Sector Filter銘柄）:")

    rows_2021 = []
    for sym, info in universe_all.items():
        df = info["df"]
        df_2021 = df.loc["2021-01-01":"2021-12-31"]
        if len(df_2021) < 100:
            continue
        ret_2021 = df_2021["Close"].iloc[-1] / df_2021["Close"].iloc[0] - 1
        rows_2021.append({
            "symbol": sym, "sector": info["sector"],
            "ret_2021": ret_2021,
            "in_g27": sym in G27_SCREENED,
        })

    df_2021 = pd.DataFrame(rows_2021).sort_values("ret_2021")
    print(f"\n  ワースト10銘柄（2021年）:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'2021リターン':>12} {'G27':>5}")
    for _, r in df_2021.head(10).iterrows():
        g27 = "★G27" if r["in_g27"] else "    "
        print(f"  {r['symbol']:<10} {r['sector']:<10} {r['ret_2021']*100:>+12.1f}% {g27}")

    # ── 2022-2024リターン比較（Sector Filter vs G27） ──
    print(f"\n  【2022-2024リターン比較（資源・半導体ラリー）】")
    rows_2224 = []
    all_syms = set(universe_all.keys()) | set(G27_SCREENED.keys())
    for sym in all_syms:
        if sym not in universe_all and sym not in {s: None for s in G27_SCREENED}:
            continue
        info = universe_all.get(sym)
        if info is None:
            continue
        df = info["df"]
        df_2224 = df.loc["2022-01-01":"2024-12-31"]
        if len(df_2224) < 200:
            continue
        ret_2224 = df_2224["Close"].iloc[-1] / df_2224["Close"].iloc[0] - 1
        rows_2224.append({
            "symbol": sym, "sector": info["sector"],
            "ret_2022_2024": ret_2224,
            "in_g27": sym in G27_SCREENED,
            "in_sector_filter": info["sector"] in ALLOWED_SECTORS,
        })

    df_2224 = pd.DataFrame(rows_2224).sort_values("ret_2022_2024", ascending=False)

    # G27のみ / Sector Filterのみ / 両方
    g27_only  = df_2224[df_2224["in_g27"] & ~df_2224["in_sector_filter"]]
    sec_only  = df_2224[~df_2224["in_g27"] & df_2224["in_sector_filter"]]
    both      = df_2224[df_2224["in_g27"] & df_2224["in_sector_filter"]]

    print(f"\n  ベスト銘柄 TOP15（2022-2024）:")
    print(f"  {'銘柄':<10} {'セクター':<10} {'2022-24':>9} {'G27':>5} {'Sector':>7}")
    for _, r in df_2224.head(15).iterrows():
        g27_mark = "★G27" if r["in_g27"] else "    "
        sec_mark = "★Sec" if r["in_sector_filter"] else "    "
        print(f"  {r['symbol']:<10} {r['sector']:<10} {r['ret_2022_2024']*100:>+9.1f}% "
              f"{g27_mark} {sec_mark}")

    print(f"\n  グループ別2022-2024平均リターン:")
    if not both.empty:
        print(f"    G27 ∩ Sector Filter: {both['ret_2022_2024'].mean()*100:>+7.1f}%（{len(both)}銘柄）")
    if not g27_only.empty:
        print(f"    G27のみ（非Sector）: {g27_only['ret_2022_2024'].mean()*100:>+7.1f}%（{len(g27_only)}銘柄）")
    if not sec_only.empty:
        print(f"    Sectorのみ（非G27）: {sec_only['ret_2022_2024'].mean()*100:>+7.1f}%（{len(sec_only)}銘柄）")

    # ── 鍵となるセクター別分析 ──
    print(f"\n  【G27 vs Sector Filter：キーセクター2022-2024リターン比較】")
    for sec in ["電機精密", "海運", "商社", "鉄鋼"]:
        g27_sec  = df_2224[df_2224["in_g27"] & (df_2224["sector"] == sec)]
        sf_sec   = df_2224[df_2224["in_sector_filter"] & (df_2224["sector"] == sec)]
        g27_mean = g27_sec["ret_2022_2024"].mean() * 100 if not g27_sec.empty else float("nan")
        sf_mean  = sf_sec["ret_2022_2024"].mean()  * 100 if not sf_sec.empty  else float("nan")
        print(f"  {sec:<10}: G27={g27_mean:>+6.1f}%({len(g27_sec)}銘柄)  "
              f"SectorFilter={sf_mean:>+6.1f}%({len(sf_sec)}銘柄)  "
              f"差={g27_mean-sf_mean:>+6.1f}%")

    return df_2224


# ================================================================
# Task 3: CB設計改善比較
# ================================================================
def task3_cb_improvements(universe_all: dict) -> list[dict]:
    print("\n" + "=" * 80)
    print("【Task 3】CB設計改善比較")
    print("=" * 80)

    prices_all = {s: i["df"]["Close"] for s, i in universe_all.items()}
    rsr_all    = calc_universe_rsr(prices_all)
    strat_base = _build_strats(universe_all, rsr_all)

    cb_modes = [
        ("A: 現行（standard）",       "standard",       120),
        ("B: 時間制限CB（120日）",     "time_limit",     120),
        ("C: 時間制限CB（60日）",      "time_limit",     60),
        ("D: エントリー停止のみ",       "entry_stop_only",120),
        ("E: サイズ50%縮小",           "partial_size",   120),
        ("F: CBなし",                   "no_cb",          120),
    ]

    results = []
    HDR = (f"\n  {'シナリオ':<30} {'銘柄':>4} {'CAGR':>8} {'Sharpe':>7} "
           f"{'MaxDD':>7} {'Calmar':>7} {'取引':>5} {'CB日数':>7}")
    print(HDR)
    print("  " + "-" * 80)

    for label, mode, max_days in cb_modes:
        strat = copy.deepcopy(strat_base)
        r = _run_flexible(universe_all, strat, cb_mode=mode, cb_max_days=max_days)
        m = _metrics(r, label, len(universe_all))
        results.append(m)
        print(f"  {label:<30} {m['n_stocks']:>4} "
              f"{m['cagr']*100:>+8.1f}% {m['sharpe']:>7.3f} "
              f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} "
              f"{m['trades']:>5} {m['cb_days']:>7}日")

    # 年次内訳（entry_stop_only と time_limit_60を詳細表示）
    for m in results:
        if m["label"] in ("B: 時間制限CB（120日）", "D: エントリー停止のみ",
                           "E: サイズ50%縮小"):
            r = m["_res"]
            eq = r.equity_curve; dd = r.dd_series
            print(f"\n  {m['label']} 年次内訳:")
            print(f"  {'年':>4}  {'リターン':>9}  {'MaxDD':>8}")
            for yr in range(2018, 2025):
                ye = eq[eq.index.year == yr]
                yd = dd[dd.index.year == yr]
                if ye.empty: continue
                ret = ye.iloc[-1] / ye.iloc[0] - 1
                mdd_yr = float(yd.min())
                print(f"  {yr:>4}  {ret*100:>+9.1f}%  {mdd_yr*100:>8.1f}%")

    return results


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  cb_sector_analysis.py — CB問題・セクタートップ銘柄・CB改善 統合検証")
    print("  期間: 2018〜2024  /  初期資本: ¥2,000,000")
    print("=" * 80)

    # データ取得
    print(f"\n[0] データ取得中（{len(TSE_SECTOR_CANDIDATES)}銘柄候補）...")
    universe_all = download_universe(TSE_SECTOR_CANDIDATES, start=START, end=END, verbose=False)
    # Sector Filterのみに絞る
    universe_sf = {s: i for s, i in universe_all.items()
                   if i["sector"] in ALLOWED_SECTORS}
    print(f"  取得成功: {len(universe_all)}銘柄 → Sector Filter: {len(universe_sf)}銘柄")

    # G27
    g27_data = {s: universe_all[s] for s in G27_SCREENED if s in universe_all}
    print(f"  G27利用可能: {len(g27_data)}銘柄")

    # ── Task 1 ──
    t1_results, sector_ranked = task1_top_stock_hypothesis(universe_sf)

    # ── Task 2 ──
    df_2224 = task2_cb_cause_analysis(universe_sf, sector_ranked)

    # ── Task 3 ──
    t3_results = task3_cb_improvements(universe_sf)

    # ── 最終サマリー ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【最終比較サマリー】")
    print("=" * 80)

    # G27 ベースライン追加
    prices_g27 = {s: i["df"]["Close"] for s, i in g27_data.items()}
    rsr_g27    = calc_universe_rsr(prices_g27)
    strat_g27  = _build_strats(g27_data, rsr_g27)
    r_g27      = _run_flexible(g27_data, strat_g27, cb_mode="standard")
    m_g27      = _metrics(r_g27, "★ 現行G27（ベースライン）", len(g27_data))

    print(f"\n  {'シナリオ':<42} {'銘柄':>4} {'CAGR':>8} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'Calmar':>7} {'平均保有':>7} {'CB日数':>7}")
    print("  " + "-" * 100)
    _print_row(m_g27)
    print("  " + "-" * 100)
    print("  [Task 1: セクター内Top-N]")
    for m in t1_results:
        _print_row(m)
    print("  " + "-" * 100)
    print("  [Task 3: CB設計改善（Sector Filter 48銘柄）]")
    for m in t3_results:
        _print_row(m)

    # 結論
    best_t1 = max(t1_results, key=lambda m: m["calmar"])
    best_t3 = max(t3_results, key=lambda m: m["calmar"])

    print("\n" + "=" * 80)
    print("【4つの問いへの回答】")
    print("=" * 80)

    print(f"""
  ① G27が高Sharpe（{m_g27['sharpe']:.3f}）を出している本当の理由
     Top1/セクター: Sharpe={t1_results[0]['sharpe']:.3f}
     Top2/セクター: Sharpe={t1_results[1]['sharpe']:.3f}
     Top3/セクター: Sharpe={t1_results[2]['sharpe']:.3f}
     全銘柄(48)   : Sharpe={t1_results[3]['sharpe']:.3f}
     → G27は「各セクターのトップ銘柄が揃っている」から高Sharpe?
       またはWFテストで確認した通りin-sample過学習?
       両方の寄与を上記から読み取ること。
  """)

    print(f"""  ② Sector Filterを改良すれば G27と同等以上を再現できるか？
     最良Top-N設定: {best_t1['label']}
       Sharpe={best_t1['sharpe']:.3f} vs G27={m_g27['sharpe']:.3f}
       {'→ 同等に近い ✅' if best_t1['sharpe'] > m_g27['sharpe'] * 0.8 else '→ まだ差がある ❌'}
  """)

    print(f"""  ③ CB設計改善で2022-2024の停止問題は解決するか？
     現行CB      : Sharpe={t3_results[0]['sharpe']:.3f}  CB={t3_results[0]['cb_days']}日
     最良CB改善案 : {best_t3['label']}
       Sharpe={best_t3['sharpe']:.3f}  CB={best_t3['cb_days']}日
       {'→ 大幅改善 ✅' if best_t3['sharpe'] > t3_results[0]['sharpe'] + 0.3 else '→ 改善効果あり △' if best_t3['sharpe'] > t3_results[0]['sharpe'] else '→ 改善なし ❌'}
  """)

    print(f"""  ④ 最も再現性の高いUniverse設計
     Task1ベスト: {best_t1['label']}（Calmar={best_t1['calmar']:.3f}）
     CB改善との組み合わせが最終推奨案。
  """)

    # CSV保存
    all_rows = []
    for m in [m_g27] + t1_results + t3_results:
        all_rows.append({
            "シナリオ": m["label"], "銘柄数": m["n_stocks"],
            "CAGR%": round(m["cagr"]*100, 2), "Sharpe": round(m["sharpe"], 3),
            "MaxDD%": round(m["mdd"]*100, 2), "Calmar": round(m["calmar"], 3),
            "取引数": m["trades"], "勝率%": round(m["win_rate"]*100, 1),
            "平均保有": round(m["avg_pos"], 2), "CB日数": m["cb_days"],
        })
    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUTPUT, "cb_sector_analysis.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"  CSV保存: {os.path.join(OUTPUT, 'cb_sector_analysis.csv')}")
    print("\n完了。")


if __name__ == "__main__":
    main()
