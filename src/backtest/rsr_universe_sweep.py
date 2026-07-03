"""
backtest/rsr_universe_sweep.py

RSRユニバースサイズ感度テスト
  RSR42 / RSR76 / RSR91 の3サイズで BASELINE を比較

目的:
  ユニバースが小さすぎると RSR ランキングが固着し
  avgHoldings が 2 台に落ちる仮説を検証する。

追加メトリクス:
  avg_holdings       : 平均同時保有銘柄数
  entries_per_year   : 年間エントリー数
  top5_concentration : 上位5銘柄の RSR 固着度（HHI, 0=完全分散 1=固定）
  rsr_rank_turnover  : 上位 top-k の日次入れ替え率（高いほど流動的）
  avg_entry_lag_days : エントリー日→価格ピーク の平均遅延日数（Step2）

実行:
  python -m backtest.rsr_universe_sweep
  DATA_VERSION=2026-03-28 python -m backtest.rsr_universe_sweep
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import BACKTEST_DATASET_DIR

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.rsr            import calc_composite_return, calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.universe_builder        import download_universe, TOPIX100_TICKERS


# ------------------------------------------------------------------ #
# 定数（composite_alpha_bt と揃える）
# ------------------------------------------------------------------ #
START         = "2018-01-01"
END           = "2024-12-31"
CAPITAL       = 3_000_000
MAX_POSITIONS = 3
MAX_HOLD_DAYS = 60
MIN_RSR       = 75.0
MAX_DD_LIMIT  = 0.15
LOT           = 100
SLIPPAGE      = 0.001
COMMISSION    = 0.00055
COST_ONE_WAY  = SLIPPAGE + COMMISSION
MIN_SEPA      = 6
MOM_PERIOD    = 21
TURTLE_ENTRY  = 20
TURTLE_EXIT   = 20

SECTOR_STRATEGY: dict[str, str] = {
    "海運": "fujiko", "機械": "fujiko", "電機精密": "fujiko",
    "商社": "fujiko", "電機": "fujiko", "ゲーム": "fujiko",
    "レジャー": "fujiko", "食品": "fujiko",
    "ガス": "mean_rev", "鉄鋼": "mean_rev", "銀行": "mean_rev",
    "保険": "mean_rev", "輸送機器": "mean_rev", "化学": "mean_rev",
    "小売": "mean_rev",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

SNAPSHOT_BASE = BACKTEST_DATASET_DIR
DATA_VERSION  = os.environ.get("DATA_VERSION", "")


# ------------------------------------------------------------------ #
# ユニバース定義
# ------------------------------------------------------------------ #
def _build_universe_configs() -> dict[str, dict[str, str]]:
    """
    RSRコンテキストサイズ別ユニバース辞書を生成する。

    設計:
      トレードユニバースは常に RSR42 固定。
      RSRコンテキスト（ランク計算に使う銘柄プール）のみ拡大する。
      → 同じ銘柄で RSR が「42銘柄中での順位」vs「91銘柄中での順位」を比較できる。
    """
    # RSR42: 既存 CSV（トレードユニバース = RSRコンテキスト）
    rsr42_df = pd.read_csv("configs/rsr_universe_42.csv")
    rsr42    = {row["symbol"]: row["sector"] for _, row in rsr42_df.iterrows()}

    # スナップショット利用可能銘柄
    if DATA_VERSION:
        snap_dir  = SNAPSHOT_BASE / DATA_VERSION
        snap_syms = {f.stem for f in snap_dir.glob("*.parquet")}
    else:
        snap_syms = set(TOPIX100_TICKERS.keys())

    # RSR76: TOPIX100_TICKERS ∩ snapshot のみ（補完なし）
    rsr76_ctx = {
        sym: sec for sym, sec in TOPIX100_TICKERS.items()
        if sym in snap_syms
    }  # ≈ 76 銘柄

    # RSR91: snapshot 全銘柄
    rsr91_ctx = dict(rsr76_ctx)
    for sym in snap_syms:
        if sym not in rsr91_ctx:
            rsr91_ctx[sym] = TOPIX100_TICKERS.get(sym, "その他")
    # RSR42 固有銘柄も RSR91 コンテキストに含める（rsr42 に非TOPIX100がある場合）
    for sym, sec in rsr42.items():
        if sym not in rsr91_ctx:
            rsr91_ctx[sym] = sec

    return {
        f"RSR_CTX{len(rsr42)}":      rsr42,       # コンテキスト=42, トレード=42
        f"RSR_CTX{len(rsr76_ctx)}":  rsr76_ctx,   # コンテキスト≈76, トレード=42
        f"RSR_CTX{len(rsr91_ctx)}":  rsr91_ctx,   # コンテキスト≈91, トレード=42
    }


# ------------------------------------------------------------------ #
# RSRランク固着度（HHI ベース）
# ------------------------------------------------------------------ #
def _calc_top5_concentration(rsr_df: pd.DataFrame, top_k: int = 5) -> float:
    """
    上位 top_k の RSR ランキングの固着度を HHI で測る。

    各日の上位 top_k に登場した回数を銘柄ごとにカウント →
    出現頻度を正規化して HHI を計算。

    HHI = 0 : 全銘柄が均等に上位に入れ替わる（理想）
    HHI = 1 : 特定銘柄が常に上位を独占（固着）
    """
    top_counts: dict[str, int] = {}
    total_days = 0
    for _, row in rsr_df.iterrows():
        valid = row.dropna()
        if len(valid) < top_k:
            continue
        top_syms = valid.nlargest(top_k).index.tolist()
        for sym in top_syms:
            top_counts[sym] = top_counts.get(sym, 0) + 1
        total_days += 1

    if total_days == 0 or not top_counts:
        return float("nan")

    freq = np.array(list(top_counts.values()), dtype=float)
    freq /= freq.sum()
    hhi = float(np.sum(freq ** 2))
    return round(hhi, 4)


def _calc_rsr_rank_turnover(rsr_df: pd.DataFrame, top_k: int = 5) -> float:
    """
    隣接日の上位 top_k メンバーの入れ替え率を計算する。

    turnover = 1 - |前日top_k ∩ 当日top_k| / top_k
    高いほどランキングが流動的。
    """
    prev_top: set | None = None
    turnovers = []
    for _, row in rsr_df.iterrows():
        valid = row.dropna()
        if len(valid) < top_k:
            prev_top = None
            continue
        cur_top = set(valid.nlargest(top_k).index.tolist())
        if prev_top is not None:
            overlap = len(prev_top & cur_top)
            turnovers.append(1.0 - overlap / top_k)
        prev_top = cur_top

    return round(float(np.mean(turnovers)), 4) if turnovers else float("nan")


# ------------------------------------------------------------------ #
# 平均エントリー遅延（Step2: RSR lag分析）
# ------------------------------------------------------------------ #
def _calc_entry_lag(trades: list[dict], universe_raw: dict) -> dict:
    """
    trade ごとに entry_date → price peak の遅延を計算する。

    遅延 > 0 : エントリー後に価格ピークがある（遅れてエントリー）
    遅延 < 0 : エントリー前にピークがあった（逆張り的）
    遅延 ≈ 0 : エントリー日付近がピーク（理想）
    """
    lag_days = []
    for t in trades:
        sym        = t.get("symbol")
        entry_idx  = t.get("entry_idx")
        exit_idx   = t.get("exit_idx")
        if sym is None or entry_idx is None or exit_idx is None:
            continue
        if sym not in universe_raw:
            continue
        df = universe_raw[sym]["df"]
        dates = df.index
        if entry_idx >= len(dates) or exit_idx >= len(dates):
            continue
        # 保有期間中の Close
        hold_close = df["Close"].iloc[entry_idx: exit_idx + 1]
        if hold_close.empty:
            continue
        peak_pos    = hold_close.values.argmax()
        lag_days.append(int(peak_pos))   # 0=エントリー当日がピーク

    if not lag_days:
        return {"avg_lag_days": float("nan"), "pct_lag_gt20": float("nan")}

    arr = np.array(lag_days, dtype=float)
    return {
        "avg_lag_days":  round(float(arr.mean()), 1),
        "median_lag":    round(float(np.median(arr)), 1),
        "pct_lag_gt20":  round(float((arr > 20).mean() * 100), 1),
        "pct_lag_le5":   round(float((arr <= 5).mean() * 100), 1),
    }


# ------------------------------------------------------------------ #
# BASELINE バックテスト（composite_alpha_bt から最小限移植）
# ------------------------------------------------------------------ #
@dataclass
class _Pos:
    symbol:     str
    sector:     str
    qty:        int
    entry_price: float
    entry_idx:  int


def _run_baseline(
    universe_raw: dict,
    rsr_df:       pd.DataFrame,
    trade_syms:   dict[str, str],
    start:        str = START,
    end:          str = END,
    capital:      float = CAPITAL,
) -> dict:
    """BASELINE シナリオのみ実行（RSR降順ランク / 均等ウェイト）。"""

    # ---- 戦略初期化 ----
    strats: dict[str, object] = {}
    for sym, sector in trade_syms.items():
        if sym not in universe_raw:
            continue
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule  = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_rsr=MIN_RSR, min_sepa=MIN_SEPA,
                mom_period=MOM_PERIOD,
                turtle_entry=TURTLE_ENTRY, turtle_exit=TURTLE_EXIT,
                use_turtle_entry=True,
            )

    # ---- 共通取引日 ----
    common_dates = None
    for sym in trade_syms:
        if sym not in universe_raw:
            continue
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    if common_dates is None or len(common_dates) == 0:
        return {}
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(start)) & (common_dates <= pd.Timestamp(end))
    ]

    # ---- ポートフォリオ変数 ----
    cash         = float(capital)
    positions: dict[str, _Pos] = {}
    equity_curve = []
    pos_list     = []
    exposure_list= []
    trades: list[dict] = []
    peak_equity  = float(capital)
    reentry_ban: dict[str, int] = {}

    for i, date in enumerate(common_dates):
        invested = sum(
            pos.qty * float(universe_raw[sym]["df"].loc[date, "Close"])
            for sym, pos in positions.items()
            if sym in universe_raw and date in universe_raw[sym]["df"].index
        )
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        pos_list.append(len(positions))
        exposure_list.append(invested / max(1.0, cur_equity))
        if cur_equity > peak_equity:
            peak_equity = cur_equity

        next_i    = i + 1
        next_date = common_dates[next_i] if next_i < len(common_dates) else None

        rsr_row = rsr_df.loc[date] if date in rsr_df.index else pd.Series(dtype=float)

        sell_signals:   list[tuple] = []
        buy_candidates: list[tuple] = []

        for sym in trade_syms:
            if sym not in universe_raw or sym not in strats:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue

            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_row.get(sym, 0.0)) if sym in rsr_row.index else 0.0

            if is_holding and hold_idx > MAX_HOLD_DAYS:
                sell_signals.append((sym, "TIME_STOP"))
                continue
            if is_holding and rsr_val < MIN_RSR and hold_idx >= 0:
                sell_signals.append((sym, "RSR_EXIT"))
                continue

            past = df_sym.loc[:date]
            if len(past) < 30:
                continue
            sig = strats[sym].generate_signal(past)  # +1=BUY, -1=SELL, 0=hold

            if sig == -1 and is_holding:
                sell_signals.append((sym, "STRATEGY"))
            elif sig == 1 and not is_holding:
                ban_until = reentry_ban.get(sym, 0)
                if i >= ban_until and rsr_val >= MIN_RSR:
                    buy_candidates.append((rsr_val, sym))

        # ---- SELL ----
        for sym, reason in sell_signals:
            if sym not in positions or next_date is None:
                continue
            df_sym    = universe_raw[sym]["df"]
            if next_date not in df_sym.index:
                continue
            sell_px   = float(df_sym.loc[next_date, "Open"])
            pos       = positions[sym]
            proceeds  = pos.qty * sell_px * (1 - COST_ONE_WAY)
            cash     += proceeds
            entry_idx = pos.entry_idx
            trades.append({
                "symbol": sym, "side": "SELL",
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "entry_idx": entry_idx, "exit_idx": next_i,
                "reason": reason,
            })
            reentry_ban[sym] = i + 5
            del positions[sym]

        # ---- BUY ----
        buy_candidates.sort(key=lambda x: -x[0])  # RSR降順
        for rsr_val, sym in buy_candidates:
            open_slots = MAX_POSITIONS - len(positions)
            if open_slots <= 0:
                break
            if next_date is None:
                break
            df_sym = universe_raw[sym]["df"]
            if next_date not in df_sym.index:
                continue
            buy_px = float(df_sym.loc[next_date, "Open"])
            if buy_px <= 0:
                continue
            n_rem  = sum(1 for _, s in buy_candidates if s not in positions)
            slots  = min(open_slots, max(1, n_rem))
            alloc  = min(cash / slots, capital * 0.25)
            qty    = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                continue
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            positions[sym] = _Pos(
                symbol=sym, sector=trade_syms.get(sym, "不明"),
                qty=qty, entry_price=buy_px, entry_idx=next_i,
            )
            trades.append({
                "symbol": sym, "side": "BUY",
                "entry": buy_px, "qty": qty, "entry_idx": next_i,
            })

    # ---- 指標計算 ----
    eq = pd.Series(equity_curve, index=common_dates)
    sells = [t for t in trades if t["side"] == "SELL"]
    years = (common_dates[-1] - common_dates[0]).days / 365.25

    cagr  = (eq.iloc[-1] / capital) ** (1 / max(years, 0.01)) - 1
    ret_d = eq.pct_change().dropna()
    sharpe = float(ret_d.mean() / ret_d.std() * np.sqrt(252)) if ret_d.std() > 0 else 0.0
    running_max = eq.cummax()
    drawdown    = (eq - running_max) / running_max
    max_dd      = float(drawdown.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    win_trades  = [t for t in sells if t["exit"] > t["entry"]]
    win_rate    = len(win_trades) / max(len(sells), 1)

    annual: dict[str, float] = {}
    for yr, grp in eq.groupby(eq.index.year):
        annual[str(yr)] = round(float(grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2)

    lag_stats = _calc_entry_lag(sells, universe_raw)

    return {
        "cagr":           round(cagr * 100, 2),
        "sharpe":         round(sharpe, 3),
        "max_dd":         round(max_dd * 100, 2),
        "calmar":         round(calmar, 3),
        "n_trades_yr":    round(len(sells) / max(years, 0.01), 1),
        "win_rate":       round(win_rate * 100, 1),
        "avg_holdings":   round(float(np.mean(pos_list)), 2),
        "avg_exposure":   round(float(np.mean(exposure_list)) * 100, 1),
        "annual_returns": annual,
        "_sells": sells,
        **lag_stats,
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> int:
    print("=" * 70)
    print("  RSRユニバースサイズ感度テスト（RSR42 / RSR76 / RSR91）")
    print(f"  期間: {START} 〜 {END}  /  DATA_VERSION={DATA_VERSION or 'live'}")
    print("=" * 70)

    uni_configs = _build_universe_configs()
    labels      = sorted(uni_configs.keys(), key=lambda k: int(k[7:]))  # "RSR_CTX42" → 42

    # トレードユニバースは常に RSR42 固定
    rsr42_df = pd.read_csv("configs/rsr_universe_42.csv")
    trade_syms_fixed = {row["symbol"]: row["sector"]
                        for _, row in rsr42_df.iterrows()}

    print("\n[ユニバース構成]  ← トレード対象は常にRSR42固定, RSRコンテキストのみ拡大")
    for label, syms in uni_configs.items():
        print(f"  {label}: RSRコンテキスト={len(syms)}銘柄, トレード={len(trade_syms_fixed)}銘柄")
    print()

    # ---- 全ユニバース銘柄を一括ダウンロード（最大ユニバースで取得）----
    all_syms: dict[str, str] = {}
    for syms in uni_configs.values():
        all_syms.update(syms)
    print(f"[データ取得] {len(all_syms)} 銘柄...")
    universe_raw = download_universe(all_syms, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe_raw)} 銘柄\n")

    results: dict[str, dict] = {}
    rsr_dfs: dict[str, pd.DataFrame] = {}

    for label in labels:
        ctx_syms = uni_configs[label]
        print(f"─── {label} ─────────────────────────────────────────────────")

        # RSR計算（コンテキストユニバース全体で横断ランク）
        rsr_prices = {
            sym: universe_raw[sym]["df"]["Close"]
            for sym in ctx_syms if sym in universe_raw
        }
        rsr_df = calc_universe_rsr(rsr_prices)
        rsr_df = rsr_df.shift(1)    # 先読み防止
        rsr_dfs[label] = rsr_df

        # 固着度・ターンオーバー（RSR42のトレード銘柄だけで計算）
        trade_rsr_df = rsr_df[[s for s in trade_syms_fixed if s in rsr_df.columns]]
        top5_conc  = _calc_top5_concentration(trade_rsr_df, top_k=5)
        rank_turn  = _calc_rsr_rank_turnover(trade_rsr_df, top_k=5)

        # BASELINE 実行（トレードはRSR42固定、RSR判断は拡大コンテキスト）
        res = _run_baseline(
            universe_raw=universe_raw,
            rsr_df=rsr_df,
            trade_syms=trade_syms_fixed,   # 常にRSR42の42銘柄
        )
        if not res:
            print("  ⚠ バックテスト失敗\n")
            continue

        res["top5_concentration"] = top5_conc
        res["rsr_rank_turnover"]  = rank_turn
        res["universe_size"]      = len(syms)
        results[label] = res

        print(f"  CAGR={res['cagr']:+.1f}%  Sharpe={res['sharpe']:.3f}  "
              f"MaxDD={res['max_dd']:.1f}%  Calmar={res['calmar']:.3f}")
        print(f"  取引数/年={res['n_trades_yr']:.0f}  勝率={res['win_rate']:.1f}%")
        print()
        print(f"  [追加メトリクス]")
        print(f"  avg_holdings       : {res['avg_holdings']:.2f}  "
              f"{'✅ 正常(5〜8)' if 5 <= res['avg_holdings'] <= 8 else '⚠ 危険(<4)' if res['avg_holdings'] < 4 else '⚠ フィルター弱'}")
        print(f"  entries_per_year   : {res['n_trades_yr']:.1f}")
        print(f"  top5_concentration : {top5_conc:.4f}  "
              f"{'(固着傾向)' if top5_conc > 0.10 else '(正常な流動性)'}")
        print(f"  rsr_rank_turnover  : {rank_turn:.4f}  "
              f"({'流動的' if rank_turn > 0.3 else '固着'})")
        print(f"  universe_size      : {len(syms)}")
        print()
        print(f"  [RSR遅延分析 Step2]")
        print(f"  avg_lag_days       : {res.get('avg_lag_days', 'N/A')}  "
              f"{'理想(-5〜+5)' if abs(res.get('avg_lag_days', 99)) <= 5 else '⚠ RSR遅延あり(+20以上は危険)' if res.get('avg_lag_days', 0) > 20 else ''}")
        print(f"  median_lag         : {res.get('median_lag', 'N/A')}")
        print(f"  lag>20日 の割合    : {res.get('pct_lag_gt20', 'N/A')}%")
        print(f"  lag≤5日 の割合     : {res.get('pct_lag_le5', 'N/A')}%")
        print()

    # ---- 比較サマリー ----
    if not results:
        print("結果なし")
        return 1

    print("=" * 70)
    print("  比較サマリー")
    print("=" * 70)
    col_w = 10
    header = f"  {'指標':<24}" + "".join(f" {lb:>{col_w}}" for lb in labels)
    print(header)
    print(f"  {'-'*24}" + f" {'-'*col_w}" * len(labels))

    metrics_def = [
        ("universe_size",      "ユニバースサイズ",    "{:.0f}"),
        ("avg_holdings",       "avgHoldings",         "{:.2f}"),
        ("entries_per_year",   "エントリー/年",       "{:.1f}"),
        ("n_trades_yr",        "取引数/年",           "{:.0f}"),
        ("top5_concentration", "top5固着度(HHI)",     "{:.4f}"),
        ("rsr_rank_turnover",  "ランクターンオーバー", "{:.4f}"),
        ("cagr",               "CAGR(%)",             "{:+.1f}"),
        ("sharpe",             "Sharpe",              "{:.3f}"),
        ("max_dd",             "MaxDD(%)",            "{:.1f}"),
        ("calmar",             "Calmar",              "{:.3f}"),
        ("avg_lag_days",       "avgLag(日)",          "{:.1f}"),
        ("pct_lag_gt20",       "lag>20日(%)",         "{:.1f}"),
        ("pct_lag_le5",        "lag≤5日(%)",          "{:.1f}"),
    ]
    for key, label, fmt in metrics_def:
        row = f"  {label:<24}"
        for lb in labels:
            v = results.get(lb, {}).get(key, float("nan"))
            try:
                row += f" {fmt.format(v):>{col_w}}"
            except (ValueError, TypeError):
                row += f" {'N/A':>{col_w}}"
        print(row)

    # entries_per_year は n_trades_yr と同じなので追加計算不要
    print()
    print("  ── 年次リターン ────────────────────────────────────────────")
    all_years = sorted(set(
        yr for r in results.values() for yr in r.get("annual_returns", {})
    ))
    print(f"  {'年':<6}" + "".join(f" {lb:>{col_w}}" for lb in labels))
    for yr in all_years:
        row = f"  {yr:<6}"
        for lb in labels:
            v = results.get(lb, {}).get("annual_returns", {}).get(str(yr), float("nan"))
            try:
                row += f" {v:>+9.1f}%"
            except (ValueError, TypeError):
                row += f" {'N/A':>{col_w}}"
        print(row)

    print()
    print("  ── 判定 ───────────────────────────────────────────────────")
    print("  avgHoldings 理想: 5〜8 / 危険: <4 / フィルター弱: >10")
    for lb in labels:
        r = results.get(lb, {})
        h = r.get("avg_holdings", 0)
        c = r.get("top5_concentration", 0)
        verdict = "✅ 正常" if 5 <= h <= 8 else ("⚠ 保有数危険" if h < 4 else "⚠ フィルター弱")
        fixation = "⚠ ランク固着" if c > 0.10 else "✅ 流動的"
        print(f"  {lb}: avgHoldings={h:.2f} {verdict}  top5HHI={c:.4f} {fixation}")

    # ---- 結果保存 ----
    out_path = Path("results") / f"rsr_universe_sweep_{pd.Timestamp.now().strftime('%Y-%m-%d')}.json"
    out_data = {
        lb: {k: v for k, v in r.items() if k != "_sells"}
        for lb, r in results.items()
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
