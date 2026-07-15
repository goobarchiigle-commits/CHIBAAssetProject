"""
src/backtest/study76_clenow_benchmark_wf.py
Study76 — Clenow純正ベンチマークWF（複雑性の対価定量化）

正典: reports/study76_execution_plan.md / study76_checklist.md / study76_dependency_matrix.md
      （矛盾時はこれらの正典を優先。本ファイルは正典の実装であり、簡略化・独自解釈は行わない）

目的:
  現行Production（D_ATR_EQ）の複雑性（レジーム5機構・Exit複数系統・Addon/Sizing機構）が
  実際にリターンを稼いでいるかを、単一パラダイム（Clenowモメンタム・週次回転・
  単一レジームフィルター・ランク脱落回転Exit）の最簡構成と直接比較して定量化する。

比較対象（3系列・Study75規則ユニバース上で全てfresh run）:
  Arm 1: Clenow純正構成 + Turtleブレイクアウト・トリガー有
  Arm 2: Clenow純正構成 + Turtleブレイクアウト・トリガー無
  Baseline: D_ATR_EQ（現行Production全アーキテクチャ・Study75 Universeで再測定）

固定パラメータ（<10個・事前固定・スイープ禁止）:
  1. capital              = 3,000,000円（PARAMS_LOCKED）
  2. slippage             = 0.001（PARAMS_LOCKED）
  3. commission           = 0.00055（PARAMS_LOCKED）
  4. max_positions (top-N)= 3（PARAMS_LOCKED）
  5. rank_lookback_days   = 90（Clenow標準・slope90d）
  6. regime_ma_period     = 200（TOPIX > MA200 単一フィルター）
  7. rebalance_freq       = 週次（各週の最初の取引日）
  8. turtle_entry_lookback= strategy.yaml `fujiko.turtle_entry`（既存ロック値・Arm1のみ使用）
  9. arms                 = 2（Turtleトリガー 有/無）

削除（D_ATR_EQ → Clenow純正構成、正典§2.1）:
  - レジーム5機構（dynamic_universe/bear_universe_filter/gross_exposure/shock_exit_mode/fraction）
    → TOPIX > MA200 単一フィルターに置換
  - Exit複数系統（RSR_EXIT/ATR_TRAIL/ATR_TRAILING/RUNNER_TRAIL_EXIT/TIME_STOP/TURTLE_EXIT/
    FIXED_PCT_TRAILING/MARKET_SHOCK_EXIT等）→ ランク脱落回転Exitのみに置換
  - Entry複合スコア（RSR42+composite alpha+entry_timing boost）→ (slope90d×R²) ランクのみに置換
  - Addon/Sizing機構（eq_scale_addon/position_sizing/adaptive_growth/quality_replacement）
    → 均等ウェイト固定に置換
  - Capital Scaling層 → 不使用（固定資本）

Universe/データ入力（Study75完了後に確定するインターフェース。本ファイルは以下の
契約に従うアダプタとして機能する。Study75の実出力形式が上記と異なる場合は、
本ファイルではなく変換アダプタ側を直す）:
  - 規則ユニバース: STUDY75_UNIVERSE_FILE
      {"monthly_universe": {"YYYY-MM-DD": ["1301.T", ...], ...}}
      key = 各月の再適用日（第1営業日）。銘柄はTOPIX500∩流動性∩lot制約フィルター適用済み。
  - 価格データ: data/jquants/processed/{symbol}.parquet（src/jquants/ 取得基盤の出力）
      survivorship-free（上場廃止銘柄を含む）。yfinance/RSR42ベースの旧データは使用禁止。
  - TOPIX: data/jquants/processed/TOPIX.parquet（Close列。無ければ fail-closed）

Decision（正典§2.4・変更禁止）:
  各Armのを D_ATR_EQ と比較して ΔCAGR を算出:
    PASS         : ΔCAGR ≥ -2pp
    FAIL         : ΔCAGR <  -4pp
    INCONCLUSIVE : -4pp ≤ ΔCAGR < -2pp（ユーザー決裁・裁量判断禁止）
  Study76全体の結論 = 2アームのうち良い方（勝者構成）の判定を採用する
  （正典Phase5「勝者構成（純正 or Turtleトリガー有無いずれか）」の文言に基づく）。
  両アームともFAILの場合のみ全体FAIL。いずれかがPASSならば全体PASS（勝者はそのArm）。
  それ以外（少なくとも一方がINCONCLUSIVE、PASSなし）は全体INCONCLUSIVE。

実行前提（正典・禁止事項）:
  - Study75未完了段階でのfresh run実行は禁止（本ファイルはコードとして完成させるのみ）。
  - パラメータ追加・アーム追加・閾値スイープは禁止。
  - キャッシュ値・旧Universe公式値との比較によるProduction判定は禁止（fresh_run_required=true）。

使い方（Study75完了後）:
  python -m src.backtest.study76_clenow_benchmark_wf --dry-run-check
  python -m src.backtest.study76_clenow_benchmark_wf --confirm --start 2018-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.config_loader import load_strategy_config
from src.paths import JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, STUDY75_UNIVERSE_FILE
import src.backtest.composite_alpha_bt as cab
from src.backtest.capital_allocation_abc import calc_metrics

_JST = timezone(timedelta(hours=9))

# ── PARAMS_LOCKED（CLAUDE.md/正典準拠。本スクリプトから変更しない）───────────────
CAPITAL       = 3_000_000
SLIPPAGE      = 0.001
COMMISSION    = 0.00055
COST_ONE_WAY  = SLIPPAGE + COMMISSION
LOT           = 100
MAX_POSITIONS = 3

# ── Study76固有パラメータ（<10個・全て事前固定・正典§2.2）────────────────────────
RANK_LOOKBACK_DAYS = 90     # slope90d
REGIME_MA_PERIOD   = 200    # TOPIX > MA200 単一フィルター
CLENOW_SCALE       = 10_000.0  # archive/clenow_momentum.py と同一スケーリング

DEFAULT_START = "2018-01-01"

PASS_THRESHOLD_PP = -2.0
FAIL_THRESHOLD_PP = -4.0

TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"


# ======================================================================
# 1. Universe読込（Study75規則ユニバース・月次再適用）
# ======================================================================
def load_study75_rule_universe(path: Path) -> dict[str, list[str]]:
    """
    {"monthly_universe": {"YYYY-MM-DD": ["1301.T", ...]}} を読み込む。
    Study75完了前は存在しない（fail-closed・ダミー生成はしない）。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Study75規則ユニバースファイルが存在しません: {path}\n"
            "  Study75（Survivorship-free規則ユニバース生成）完了後、"
            "生成物をこのパスに配置するか --universe-file で指定してください。\n"
            "  スキーマ: {\"monthly_universe\": {\"YYYY-MM-DD\": [\"1301.T\", ...]}}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    monthly = data.get("monthly_universe")
    if not isinstance(monthly, dict) or not monthly:
        raise ValueError(f"monthly_universe が空/不正です: {path}")
    return monthly


def universe_symbols(monthly_universe: dict[str, list[str]]) -> list[str]:
    """
    正規化済み monthly_universe（dict[str, list[str]]）から通期の全銘柄集合を導出する。
    Study75の生JSON構造には非依存（load_study75_rule_universe() が返す正規化済み型のみに依存）。
    全呼び出し元（main の dry-run-check / 本実行 / run_clenow_engine）はこの関数を共通で使うこと。
    重複実装を作らないこと（Study75スキーマ変更時の修正箇所を1か所に保つため）。
    """
    return sorted({s for members in monthly_universe.values() for s in members})


def active_universe_for_date(monthly_universe: dict[str, list[str]], as_of: pd.Timestamp) -> list[str]:
    """as_of 時点で適用される規則ユニバース（直近の月次再適用日以下の最新キー）を返す。"""
    keys = sorted(pd.Timestamp(k) for k in monthly_universe)
    applicable = [k for k in keys if k <= as_of]
    if not applicable:
        return []
    return monthly_universe[applicable[-1].strftime("%Y-%m-%d")]


def build_daily_active_matrix(
    monthly_universe: dict[str, list[str]],
    all_symbols: list[str],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    月次規則ユニバースを日次 0/1 活性行列へ展開する。
    D_ATR_EQ側の sym_active_df とClenow側の対象銘柄フィルタの両方に使う
    （同一Universeを両アーキテクチャで共有し、Universe差とArchitecture差を分離する）。
    """
    keys = sorted(pd.Timestamp(k) for k in monthly_universe)
    mat = pd.DataFrame(0, index=date_index, columns=all_symbols, dtype=np.int8)
    for i, k in enumerate(keys):
        end = keys[i + 1] if i + 1 < len(keys) else date_index[-1] + pd.Timedelta(days=1)
        members = [s for s in monthly_universe[k.strftime("%Y-%m-%d")] if s in mat.columns]
        mask = (date_index >= k) & (date_index < end)
        mat.loc[mask, members] = 1
    return mat


# ======================================================================
# 2. 価格データ読込（J-Quants processed。survivorship-free）
# ======================================================================
def load_jquants_panel(symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    data/jquants/processed/{symbol}.parquet から読み込む。
    yfinance/RSR42ベースの旧データ（cab.download_universe）は使用しない
    （survivorship-free性を壊すため・正典で明示的に禁止）。

    Returns: (panel, missing_symbols)
    """
    panel: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)
    for sym in symbols:
        path = JQUANTS_PROCESSED_DIR / f"{sym}.parquet"
        if not path.exists():
            missing.append(sym)
            continue
        df = pd.read_parquet(path)
        df = df.loc[(df.index >= ts_start) & (df.index <= ts_end)]
        if df.empty:
            missing.append(sym)
            continue
        panel[sym] = df
    return panel, missing


def load_jquants_topix(start: str, end: str) -> pd.Series:
    if not TOPIX_PARQUET.exists():
        raise FileNotFoundError(
            f"TOPIX価格データが存在しません: {TOPIX_PARQUET}\n"
            "  J-Quants /indices/topix（または該当エンドポイント）からの取得・"
            "processed/TOPIX.parquet への保存が必要です。"
        )
    df = pd.read_parquet(TOPIX_PARQUET)
    close = df["Close"].dropna()
    return close.loc[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]


# ======================================================================
# 3. Clenowモメンタムスコア（(slope90d × R²)。archive/clenow_momentum.py と同一定義）
# ======================================================================
def calc_clenow_score(closes: np.ndarray) -> float:
    """
    slope × R² × CLENOW_SCALE。
    過去 lookback 日の log(close) を時間に対して線形回帰し、
    傾き(slope)と当てはまり度(R²)の積をスコアとする。
    """
    if len(closes) < 2:
        return float("nan")
    closes = closes.astype(float)
    if np.any(closes <= 0) or np.any(np.isnan(closes)):
        return float("nan")

    y = np.log(closes)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, min(1.0, r2))

    return slope * r2 * CLENOW_SCALE


def compute_momentum_matrix(
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    lookback: int = RANK_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """全銘柄・全日付のClenowスコア行列（先読みなし・lookback日分のみ使用）。"""
    scores: dict[str, pd.Series] = {}
    for sym in symbols:
        if sym not in panel:
            continue
        close = panel[sym]["Close"].dropna()
        scores[sym] = close.rolling(lookback, min_periods=lookback).apply(
            lambda w: calc_clenow_score(w.values), raw=False
        )
    return pd.DataFrame(scores)


def compute_turtle_breakout_mask(
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    lookback: int,
) -> pd.DataFrame:
    """Close >= 直近lookback日高値(shift(1)・先読み防止) → ブレイクアウト成立。"""
    masks: dict[str, pd.Series] = {}
    for sym in symbols:
        if sym not in panel:
            continue
        high = panel[sym]["High"]
        close = panel[sym]["Close"]
        rolling_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
        masks[sym] = (close >= rolling_high).fillna(False)
    return pd.DataFrame(masks).fillna(False)


def weekly_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """各週(ISO week)の最初の取引日をリバランス日とする。"""
    s = pd.Series(index, index=index)
    firsts = s.groupby(index.isocalendar().year.astype(str) + "-W" + index.isocalendar().week.astype(str).str.zfill(2)).first()
    return sorted(firsts.tolist())


# ======================================================================
# 4. Clenow純正構成エンジン（週次回転・ランク脱落回転Exit・均等ウェイト）
# ======================================================================
def run_clenow_engine(
    panel: dict[str, pd.DataFrame],
    monthly_universe: dict[str, list[str]],
    topix_close: pd.Series,
    turtle_arm: bool,
    start: str,
    end: str,
    capital: float = CAPITAL,
    max_positions: int = MAX_POSITIONS,
    rank_lookback: int = RANK_LOOKBACK_DAYS,
    turtle_lookback: int = 55,
) -> dict:
    """
    週次リバランス・ランク脱落回転Exitのみ・均等ウェイト・単一TOPIX>MA200レジームフィルター。
    Returns: {"equity_curve", "dates", "trades", "exposure_list", "data_gaps"}
    """
    all_symbols = universe_symbols(monthly_universe)
    calendar = topix_close.loc[(topix_close.index >= pd.Timestamp(start)) & (topix_close.index <= pd.Timestamp(end))].index
    if len(calendar) == 0:
        raise ValueError("指定期間のTOPIXカレンダーが空です")

    momentum = compute_momentum_matrix(panel, all_symbols, lookback=rank_lookback)
    breakout = compute_turtle_breakout_mask(panel, all_symbols, lookback=turtle_lookback) if turtle_arm else None
    regime = cab._calc_regime(topix_close)
    bullish = ~regime["risk_off"]

    rebalance_dates = set(weekly_rebalance_dates(calendar))

    cash = float(capital)
    holdings: dict[str, dict] = {}   # symbol -> {"qty", "entry_price", "entry_date"}
    pending_orders: list[dict] = []  # 翌日寄付執行待ち
    trades: list[dict] = []
    equity_curve: list[float] = []
    exposure_list: list[float] = []
    dates_out: list[pd.Timestamp] = []
    data_gaps: list[dict] = []

    def _close_on(sym: str, dt: pd.Timestamp) -> float | None:
        df = panel.get(sym)
        if df is None or dt not in df.index:
            return None
        return float(df.loc[dt, "Close"])

    def _open_on(sym: str, dt: pd.Timestamp) -> float | None:
        df = panel.get(sym)
        if df is None or dt not in df.index:
            return None
        return float(df.loc[dt, "Open"])

    for i, dt in enumerate(calendar):
        # ── (a) 前日決定分の約定処理（翌日寄付執行・shift(1)パターン踏襲） ──
        if pending_orders:
            still_pending = []
            for order in pending_orders:
                px = _open_on(order["symbol"], dt)
                if px is None:
                    data_gaps.append({"date": dt.strftime("%Y-%m-%d"), "symbol": order["symbol"], "reason": "no_open_price"})
                    still_pending.append(order)
                    continue
                if order["side"] == "SELL":
                    h = holdings.pop(order["symbol"], None)
                    if h:
                        proceeds = px * (1 - COST_ONE_WAY) * h["qty"]
                        cash += proceeds
                        pnl = proceeds - h["entry_price"] * (1 + COST_ONE_WAY) * h["qty"]
                        trades.append({
                            "side": "SELL", "symbol": order["symbol"], "date": dt.strftime("%Y-%m-%d"),
                            "entry": h["entry_price"], "exit": px, "qty": h["qty"], "pnl": pnl,
                            "entry_idx": h["entry_idx"], "exit_idx": i,
                        })
                else:  # BUY
                    notional = order["target_notional"]
                    qty = int((notional / (px * (1 + COST_ONE_WAY))) // LOT * LOT)
                    if qty <= 0:
                        continue
                    cost = px * (1 + COST_ONE_WAY) * qty
                    if cost > cash:
                        continue
                    cash -= cost
                    holdings[order["symbol"]] = {"qty": qty, "entry_price": px, "entry_date": dt, "entry_idx": i}
                    trades.append({
                        "side": "BUY", "symbol": order["symbol"], "date": dt.strftime("%Y-%m-%d"),
                        "entry": px, "qty": qty, "entry_idx": i,
                    })
            pending_orders = [o for o in still_pending]

        # ── (b) 週次リバランス判定（当日Closeまでのデータで決定・翌営業日執行） ──
        if dt in rebalance_dates and i + 1 < len(calendar):
            active_syms = [s for s in active_universe_for_date(monthly_universe, dt) if s in panel]
            is_bullish = bool(bullish.get(dt, False)) if dt in bullish.index else False

            candidates = []
            for sym in active_syms:
                if sym not in momentum.columns or dt not in momentum.index:
                    continue
                score = momentum.loc[dt, sym]
                if pd.isna(score):
                    continue
                if turtle_arm:
                    if breakout is None or sym not in breakout.columns or not bool(breakout.loc[dt, sym]):
                        continue
                candidates.append((sym, float(score)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            ranked_syms = [s for s, _ in candidates]

            if is_bullish:
                target = ranked_syms[:max_positions]
            else:
                # 弱気: 新規建玉なし。既存保有のうち規則ユニバース上位に残るものだけ継続
                target = [s for s in holdings if s in ranked_syms[:max_positions]]

            # ランク脱落回転Exit: target外の保有銘柄を手仕舞い
            for sym in list(holdings.keys()):
                if sym not in target:
                    pending_orders.append({"side": "SELL", "symbol": sym})

            if is_bullish:
                open_slots = max_positions - len([s for s in target if s in holdings])
                new_entries = [s for s in target if s not in holdings][:max_positions]
                if new_entries:
                    per_slot_notional = (cash + sum(
                        (_close_on(s, dt) or 0) * h["qty"] for s, h in holdings.items()
                    )) / max_positions
                    for sym in new_entries:
                        pending_orders.append({
                            "side": "BUY", "symbol": sym, "target_notional": per_slot_notional,
                        })

        # ── (c) 日次NAVマーキング ──
        holdings_value = 0.0
        for sym, h in holdings.items():
            px = _close_on(sym, dt)
            if px is None:
                data_gaps.append({"date": dt.strftime("%Y-%m-%d"), "symbol": sym, "reason": "no_close_price"})
                px = h["entry_price"]
            holdings_value += px * h["qty"]
        nav = cash + holdings_value
        equity_curve.append(nav)
        exposure_list.append(holdings_value / nav if nav > 0 else 0.0)
        dates_out.append(dt)

    return {
        "equity_curve": equity_curve, "dates": dates_out, "trades": trades,
        "exposure_list": exposure_list, "data_gaps": data_gaps,
    }


def extract_engine_metrics(engine_result: dict, capital: float) -> dict:
    m = calc_metrics(
        engine_result["equity_curve"], engine_result["trades"],
        engine_result["exposure_list"], capital, engine_result["dates"],
    )
    return {
        "cagr": m.get("cagr", 0.0), "sharpe": m.get("sharpe", 0.0), "sortino": m.get("sortino", 0.0),
        "max_dd": m.get("max_dd", 0.0), "calmar": m.get("calmar", 0.0),
        "n_trades": m.get("n_trades", 0), "avg_exposure": m.get("avg_exposure", 0.0),
        "data_gap_count": len(engine_result["data_gaps"]),
    }


# ======================================================================
# 5. D_ATR_EQ（現行Production・全アーキテクチャ）を同一Universeでfresh run再測定
# ======================================================================
def build_d_atr_eq_dataset(panel: dict[str, pd.DataFrame], all_symbols: list[str], sym_active_df: pd.DataFrame,
                            topix_close: pd.Series, start: str, end: str) -> dict:
    """
    D_ATR_EQのアーキテクチャ（RSR_EXIT/ATRトレーリング/多層RSR/EQ Scale addon等）は一切変更せず、
    Universeのみを Study75 規則ユニバース（sym_active_df）に差し替える。
    dynamic_universe/bear_universe_filter は不使用（正典§2.1: 規則ユニバース固定リストに置換）。
    """
    trade_syms = {s: "不明" for s in all_symbols}
    prices = {sym: panel[sym]["Close"] for sym in all_symbols if sym in panel}
    rsr_df = cab.calc_universe_rsr(prices)
    regime_df = cab._calc_regime(topix_close)
    tech_matrices = cab._precompute_tech_matrices(panel, all_symbols)
    breadth_series = cab._calc_breadth(panel)
    base_cfg = load_strategy_config()

    return dict(
        base_cfg=base_cfg, trade_syms=trade_syms, rsr_syms=trade_syms,
        universe_raw=panel, rsr_df=rsr_df, topix_close=topix_close,
        sym_active_df=sym_active_df, regime_df=regime_df,
        tech_matrices=tech_matrices, breadth_series=breadth_series,
    )


def run_d_atr_eq(ds: dict, start: str, end: str, rsr_exit_threshold: float) -> dict:
    raw = cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"], start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=ds["base_cfg"].risk.min_hold_days,
        topix_close=ds["topix_close"],
        market_shock_mode="composite",
        rsr_exit_threshold=rsr_exit_threshold,
        sym_active_df=ds["sym_active_df"],   # Study75規則ユニバース（dynamic_universeは不使用）
        enable_simple_rsr_exit=True,
        enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True,
        enable_atr_risk_sizing=False,
        enable_mtf_filter=False,
        sizing_mode="existing",
        exit_policy="A",
        addon_policy="D",
        addon_size_frac=0.25,
        addon_atr_mult=1.0,
    )
    return {
        "cagr": round(float(raw.get("cagr", 0.0)), 2), "sharpe": round(float(raw.get("sharpe", 0.0)), 3),
        "sortino": round(float(raw.get("sortino", 0.0)), 3), "max_dd": round(float(raw.get("max_dd", 0.0)), 2),
        "calmar": round(float(raw.get("calmar", 0.0)), 3), "n_trades": int(raw.get("n_trades", 0) or 0),
        "avg_exposure": round(float(raw.get("avg_exposure", raw.get("avg_exp", 0)) or 0), 1),
    }


# ======================================================================
# 6. Decision
# ======================================================================
def decide_arm(delta_cagr_pp: float) -> str:
    if delta_cagr_pp >= PASS_THRESHOLD_PP:
        return "PASS"
    if delta_cagr_pp < FAIL_THRESHOLD_PP:
        return "FAIL"
    return "INCONCLUSIVE"


def decide_overall(arm_decisions: dict[str, str]) -> tuple[str, str | None]:
    """
    正典Phase5「勝者構成（純正 or Turtleトリガー有無いずれか）」に基づく統合判定。
    いずれかのArmがPASSなら全体PASS（そのArmが勝者）。両方FAILなら全体FAIL。
    それ以外はINCONCLUSIVE（ユーザー決裁）。
    """
    passing = [k for k, v in arm_decisions.items() if v == "PASS"]
    if passing:
        return "PASS", passing[0]
    if all(v == "FAIL" for v in arm_decisions.values()):
        return "FAIL", None
    return "INCONCLUSIVE", None


# ======================================================================
# main
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Study76 — Clenow純正ベンチマークWF")
    parser.add_argument("--universe-file", default=str(STUDY75_UNIVERSE_FILE))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--confirm", action="store_true", help="ASK_FIRST確認済み（ユーザー承認後のみ付与）")
    parser.add_argument("--dry-run-check", action="store_true", help="fresh runを実行せず配線確認のみ行う")
    args = parser.parse_args()

    universe_path = Path(args.universe_file)
    universe_exists = universe_path.exists()

    if args.dry_run_check:
        print("[DRY_RUN_CHECK] fresh runは実行しません（配線確認のみ）")
        print(f"[DRY_RUN_CHECK] universe_file={universe_path} exists={universe_exists}")
        if universe_exists:
            monthly = load_study75_rule_universe(universe_path)
            all_syms = universe_symbols(monthly)
            print(f"[DRY_RUN_CHECK] monthly_keys={len(monthly)} total_symbols={len(all_syms)}")
        cfg = load_strategy_config()
        print(f"[DRY_RUN_CHECK] rsr_exit_threshold(from strategy.yaml)={cfg.fujiko.rsr_exit}")
        print(f"[DRY_RUN_CHECK] turtle_entry(from strategy.yaml)={cfg.fujiko.turtle_entry}")
        print(f"[DRY_RUN_CHECK] TOPIX parquet exists={TOPIX_PARQUET.exists()} path={TOPIX_PARQUET}")
        print("[DRY_RUN_CHECK] OK — スクリプトは正常にビルド・実行可能です")
        return 0

    if not args.confirm:
        print("=" * 70)
        print("  ASK_FIRST — Study76 fresh run 実行前確認")
        print("=" * 70)
        print(f"  Universeファイル: {universe_path}")
        print(f"  存在確認        : {'OK' if universe_exists else '未生成（Study75未完了）'}")
        print("  正典: study76_execution_plan.md §5 Universe統制ポリシー確認 + ")
        print("  新規スクリプト実行のASK_FIRST承認をユーザーから得てから --confirm を付与すること。")
        print("=" * 70)
        return 2

    monthly_universe = load_study75_rule_universe(universe_path)  # Study75未完了ならfail-closed
    all_symbols = universe_symbols(monthly_universe)
    cfg = load_strategy_config()

    print(f"Universe: {len(all_symbols)}銘柄・{len(monthly_universe)}月次キー  期間: {args.start} 〜 {args.end}")
    panel, missing = load_jquants_panel(all_symbols, args.start, args.end)
    if missing:
        print(f"[WARN] J-Quants processed データ欠落: {len(missing)}銘柄 (該当日は除外・data_gapsに記録)")
    topix_close = load_jquants_topix(args.start, args.end)

    date_index = topix_close.index
    sym_active_df = build_daily_active_matrix(monthly_universe, all_symbols, date_index)

    results: dict[str, dict] = {}
    for arm_key, turtle_arm in (("TURTLE_ON", True), ("TURTLE_OFF", False)):
        print(f"[RUN] Clenow純正構成 ({arm_key}) 実行中...")
        engine_result = run_clenow_engine(
            panel, monthly_universe, topix_close, turtle_arm=turtle_arm,
            start=args.start, end=args.end, capital=CAPITAL, max_positions=MAX_POSITIONS,
            rank_lookback=RANK_LOOKBACK_DAYS, turtle_lookback=cfg.fujiko.turtle_entry,
        )
        results[arm_key] = extract_engine_metrics(engine_result, CAPITAL)
        print(f"  {arm_key}: {results[arm_key]}")

    print("[RUN] D_ATR_EQ（Study75 Universe上でfresh run再測定）実行中...")
    ds = build_d_atr_eq_dataset(panel, all_symbols, sym_active_df, topix_close, args.start, args.end)
    results["D_ATR_EQ"] = run_d_atr_eq(ds, args.start, args.end, rsr_exit_threshold=cfg.fujiko.rsr_exit)
    print(f"  D_ATR_EQ: {results['D_ATR_EQ']}")

    arm_deltas: dict[str, float] = {}
    arm_decisions: dict[str, str] = {}
    for arm_key in ("TURTLE_ON", "TURTLE_OFF"):
        delta = round(results[arm_key]["cagr"] - results["D_ATR_EQ"]["cagr"], 2)
        arm_deltas[arm_key] = delta
        arm_decisions[arm_key] = decide_arm(delta)

    overall_decision, winner_arm = decide_overall(arm_decisions)

    report = {
        "study": "Study76",
        "generated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "universe_file": str(universe_path),
        "universe_symbol_count": len(all_symbols),
        "monthly_universe_keys": len(monthly_universe),
        "period": {"start": args.start, "end": args.end},
        "capital": CAPITAL,
        "params": {
            "rank_lookback_days": RANK_LOOKBACK_DAYS, "regime_ma_period": REGIME_MA_PERIOD,
            "max_positions": MAX_POSITIONS, "turtle_entry_lookback": cfg.fujiko.turtle_entry,
            "rsr_exit_threshold": cfg.fujiko.rsr_exit, "slippage": SLIPPAGE, "commission": COMMISSION,
        },
        "results": results,
        "comparison": {
            "arm_delta_cagr_pp": arm_deltas,
            "arm_decisions": arm_decisions,
            "overall_decision": overall_decision,
            "winner_arm": winner_arm,
            "decision_rule": f"PASS: ΔCAGR>={PASS_THRESHOLD_PP}pp / FAIL: ΔCAGR<{FAIL_THRESHOLD_PP}pp / else INCONCLUSIVE; overall=best-arm",
        },
        "data_quality": {"missing_symbols": missing},
        "provenance": {
            "same_universe_all_arms": True,
            "compared_against_production_universe": False,
            "data_source": "jquants_processed (survivorship-free)",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y-%m-%d")
    json_path = RESULTS_DIR / f"study76_clenow_benchmark_wf_{today_str}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "study76_clenow_benchmark_wf.md"
    md_lines = [
        f"# Study76 — Clenow純正ベンチマークWF（{today_str}）",
        "",
        f"Universe: `{universe_path}`（{len(all_symbols)}銘柄・{len(monthly_universe)}月次キー・Study75規則ユニバース）",
        f"期間: {args.start} 〜 {args.end}　資本: ¥{CAPITAL:,}",
        "",
        "## 結果",
        "",
        "| 構成 | CAGR | Sharpe | Calmar | MaxDD | n_trades | ΔCAGR(vs D_ATR_EQ) | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm_key in ("TURTLE_ON", "TURTLE_OFF"):
        r = results[arm_key]
        md_lines.append(
            f"| {arm_key} | {r['cagr']} | {r['sharpe']} | {r['calmar']} | {r['max_dd']} | {r['n_trades']} "
            f"| {arm_deltas[arm_key]} | {arm_decisions[arm_key]} |"
        )
    r = results["D_ATR_EQ"]
    md_lines.append(f"| D_ATR_EQ (baseline) | {r['cagr']} | {r['sharpe']} | {r['calmar']} | {r['max_dd']} | {r['n_trades']} | - | - |")
    md_lines += [
        "",
        f"## Overall Decision: **{overall_decision}**" + (f"（勝者構成: {winner_arm}）" if winner_arm else ""),
        "",
        "- 判定ルール: いずれかのArmがPASS→全体PASS／両方FAIL→全体FAIL／それ以外→INCONCLUSIVE",
        "- INCONCLUSIVEの場合は裁量判断せずユーザー決裁を仰ぐこと（正典§2.4）",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 60)
    print(f"  Overall Decision: {overall_decision}" + (f" (winner={winner_arm})" if winner_arm else ""))
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
