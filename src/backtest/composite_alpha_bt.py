"""
backtest/composite_alpha_bt.py

3ステップ改善バックテスト（比較実験）

BASELINE : RSR≥75 → RSR降順ランキング → top-3 均等ウェイト（現行）
STEP1    : RSR≥75 → (slope×r2)² × RSR 複合スコアランキング → top-3 均等ウェイト
           ※ alpha² × RSR: alphaを主役にしてRSRは乗数として機能させる
STEP2    : STEP1  + 2段階マクロレジームフィルター
            A（Risk Off）: TOPIX < MA200 → 新規BUY停止
            B（Crash）   : MA50 < MA200  → ポジションサイズ半減
           + CB修正: 30営業日タイムアウト解除 + CB時35%スケール（完全停止廃止）
STEP3    : STEP2  + αウェイトポジションサイジング（均等→αランク比例）

実行:
  python -m backtest.composite_alpha_bt
  python -m backtest.composite_alpha_bt --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, field
from datetime import date as date_type

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import (
    ALLOW_YFINANCE_NETWORK,
    BACKTEST_DATASET_DIR,
    CONFIGS_DIR,
    DEFAULT_DATA_VERSION,
    RESULTS_DIR,
    UNIVERSE_DIR,
)

import numpy as np
import pandas as pd
import yfinance as yf

from src.config_loader import load_strategy_config
from src.backtest.rsr                     import calc_universe_rsr, calc_composite_alpha_matrix
from src.backtest.fujiko_strategy         import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.universe_builder        import download_universe
from src.utils.memory import log_optimization, optimize_backtest_df
from src.strategy.universe import build_dyn_rsr42_active, precompute_dynamic_caps
from src.strategy.cluster import CLUSTER_MAP_DEFAULT, get_cluster, calc_cluster_weight


# ------------------------------------------------------------------ #
# 定数（live_equivalent.py 確定値と揃える）
# ------------------------------------------------------------------ #
RSR_UNIVERSE_CSV      = CONFIGS_DIR / "rsr_universe_42.csv"
TRADING_UNIVERSE_JSON = UNIVERSE_DIR / "2026Q1_temporal24.json"

START          = "2018-01-01"
END            = "2024-12-31"
SMOKE_LOOKBACK_DAYS = 180
REENTRY_COOL   = 5
LOT            = 100
SLIPPAGE       = 0.001
COMMISSION     = 0.00055
COST_ONE_WAY   = SLIPPAGE + COMMISSION
COMP_ALPHA_WINDOW = 90   # Composite Alpha 計算ウィンドウ

# Step3: αウェイト上限（ユーザー指定 0.40）
MAX_POS_WEIGHT = 0.40

# CBデッドロック防止
CB_UNLOCK_DAYS = 30    # CB発動後30営業日で強制解除（市場回復の可能性を担保）
CB_SCALE       = 0.35  # CB時のポジションスケール（完全停止→縮小に変更）

# STEP4: ブレイクアウトスコアボーナス
BREAKOUT_LOOKBACK = 200   # 200日高値ブレイク判定期間（フィルターではなく加点）
BREAKOUT_BONUS    = 1.25  # ブレイクアウト時のスコア乗数
# STEP5: トレーリングストップ
TRAIL_PERIOD      = 50    # トレーリング参照期間（50日最高終値）
TRAIL_ATR_MULT    = 3.0   # トレーリングストップのATR倍率
# STEP6: Market Breadthレジーム
BREADTH_STOP      = 0.25  # breadth < 25%: 新規BUY停止
BREADTH_REDUCE    = 0.15  # breadth < 15%: ポジション縮小（regime_step=0.5）

# ── PROD_FAITHFUL: 本番ロジック忠実再現パラメータ（signal_bridge.py file:line 確認済み）──
PROD_ATR_TRAIL_MULT  = 3.0    # signal_bridge.py:2081 (highest_close - 3.0*ATR20)
PROD_RISK_PCT        = 0.0125 # signal_bridge.py:3515 (capital × 1.25% risk per trade)
PROD_ML_SPEED_WINDOW = 20     # signal_bridge.py:199 compute_multilayer_rsr_exit
PROD_ML_PEAK_WINDOW  = 5      # signal_bridge.py:200
PROD_ML_SCALE        = 50.0   # signal_bridge.py:201
PROD_ML_HOLD_DAYS    = 5      # signal_bridge.py:233 exit_4 hold_days>=5

SECTOR_STRATEGY: dict[str, str] = {
    "海運":     "fujiko", "機械":     "fujiko", "電機精密": "fujiko",
    "商社":     "fujiko", "電機":     "fujiko", "ゲーム":   "fujiko",
    "レジャー": "fujiko", "食品":     "fujiko",
    "ガス":     "mean_rev","鉄鋼":    "mean_rev","銀行":     "mean_rev",
    "保険":     "mean_rev","輸送機器":"mean_rev","化学":     "mean_rev",
    "小売":     "mean_rev",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

OUTPUT_DIR = RESULTS_DIR
TOPIX_SYMBOL = "1306.T"   # NEXT FUNDS TOPIX ETF（TOPIXプロキシ）


# ------------------------------------------------------------------ #
# データ構造
# ------------------------------------------------------------------ #
@dataclass
class Position:
    symbol:      str
    sector:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    alpha_score: float = 0.0   # エントリー時のcomposite alpha（Step3用）
    highest_close: float = 0.0  # PROD_FAITHFUL: ATRトレーリング用の保有来高値（Close基準）


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def _load_rsr_universe(verbose: bool = False) -> dict[str, str]:
    raw_df = pd.read_csv(RSR_UNIVERSE_CSV)
    df = optimize_backtest_df(raw_df)
    log_optimization(raw_df, df, label="RSR universe CSV", verbose=verbose)
    if "sector" in df.columns:
        return dict(zip(df["symbol"], df["sector"]))
    return {sym: "不明" for sym in df["symbol"]}


def _load_trading_universe() -> dict[str, str]:
    data = json.loads(TRADING_UNIVERSE_JSON.read_text(encoding="utf-8"))
    return data["symbols"]


def _download_topix(start: str, end: str, verbose: bool = False) -> pd.Series:
    """TOPIX ETF (1306.T) を取得して日次終値を返す。"""
    if DEFAULT_DATA_VERSION:
        snap = BACKTEST_DATASET_DIR / DEFAULT_DATA_VERSION / f"{TOPIX_SYMBOL}.parquet"
        if snap.exists():
            raw_df = pd.read_parquet(snap)
            df = optimize_backtest_df(raw_df)
            log_optimization(raw_df, df, label="TOPIX parquet", verbose=verbose)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            close = df[close_col].dropna()
            close = close.loc[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
            if not close.empty:
                return close
    if not ALLOW_YFINANCE_NETWORK:
        print("  ⚠ TOPIXローカルデータ不在 → レジームフィルター無効化")
        return pd.Series(dtype=float)
    try:
        df = yf.download(TOPIX_SYMBOL, start=start, end=end, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        return close
    except Exception as e:
        print(f"  ⚠ TOPIX取得失敗（{e}）→ レジームフィルター無効化")
        return pd.Series(dtype=float)


def _calc_regime(topix: pd.Series) -> pd.DataFrame:
    """
    2段階マクロレジームフィルターを計算する。

    Returns DataFrame with columns:
      risk_off : True = TOPIX < MA200（新規BUY停止）
      crash    : True = MA50 < MA200（ポジションサイズ半減）
    """
    if topix.empty:
        idx = pd.date_range(START, END, freq="B")
        return pd.DataFrame({"risk_off": False, "crash": False}, index=idx)

    ma200 = topix.rolling(200, min_periods=100).mean()
    ma50  = topix.rolling(50,  min_periods=25).mean()

    risk_off = topix < ma200
    crash    = ma50  < ma200

    regime = pd.DataFrame({"risk_off": risk_off, "crash": crash})
    regime = regime.fillna(False)
    return regime


def _calc_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """numpy ベースの ATR 計算（pd.concat より約6倍高速）。
    index 0 は c_prev=c[0] として tr[0]=H-L とする（pandas skipna 挙動と一致）。
    """
    h = df["High"].values.astype(np.float32)
    l = df["Low"].values.astype(np.float32)
    c = df["Close"].values.astype(np.float32)
    c_prev = np.concatenate([[c[0]], c[:-1]])   # index 0: prev=current → H-L only
    tr = np.maximum(
        np.maximum(h - l, np.abs(h - c_prev)),
        np.abs(l - c_prev),
    )
    return pd.Series(tr, index=df.index).rolling(period, min_periods=period // 2).mean().astype(np.float32)


def _precompute_tech_matrices(universe_raw: dict, symbols: list) -> dict:
    """
    STEP4/5用テクニカル指標をまとめて事前計算。

    Returns
    -------
    dict with keys:
      atr20        : ATR20（先読み防止なし・当日値）
      atr20_med90  : ATR20の90日中央値（shift(1)済み）
      high200      : 200日最高値 High（shift(1)済み）- ブレイクアウトボーナス用
      high50_close : 50日最高終値 Close（shift(1)なし）- トレーリング用
    """
    atr20_d        = {}
    atr20_med90_d  = {}
    high200_d      = {}
    high50_close_d = {}

    for sym in symbols:
        if sym not in universe_raw:
            continue
        df = universe_raw[sym]["df"]
        h = df["High"]
        c = df["Close"]

        atr20 = _calc_atr(df, period=20)

        atr20_d[sym]        = atr20
        atr20_med90_d[sym]  = atr20.rolling(90, min_periods=45).median().shift(1)
        high200_d[sym]      = h.rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK // 2).max().shift(1)
        high50_close_d[sym] = c.rolling(TRAIL_PERIOD, min_periods=TRAIL_PERIOD // 2).max()

    return {
        "atr20":        pd.DataFrame(atr20_d),
        "atr20_med90":  pd.DataFrame(atr20_med90_d),
        "high200":      pd.DataFrame(high200_d),
        "high50_close": pd.DataFrame(high50_close_d),
    }


def _precompute_multilayer_rsr_matrices(
    rsr_df: pd.DataFrame,
    speed_std_window: int = PROD_ML_SPEED_WINDOW,
    peak_window: int = PROD_ML_PEAK_WINDOW,
    scale: float = PROD_ML_SCALE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    多層RSRエグジット（signal_bridge.py:196-243 compute_multilayer_rsr_exit）のベクトル化再現。

    exit_1/exit_2/exit_3 は hold_days に依存しないため事前計算できる（本関数の戻り値）。
    exit_4（hold_days>=5 かつ z<1.2）は保有日数がランタイム状態のため、
    本関数が返す z_df をランタイム側で参照して別途判定する（run_scenario内）。
    先読みなし: rolling/diff は全て当日まで（iloc[-1]相当）の後方参照のみ。
    """
    rsr_z = rsr_df / scale
    rsr_speed = rsr_z.diff()
    rsr_speed_std = rsr_speed.rolling(speed_std_window, min_periods=5).std()
    rsr_peak = rsr_z.rolling(peak_window, min_periods=1).max()

    exit_1 = rsr_z < 1.1
    exit_2 = (rsr_speed_std > 0) & (rsr_speed < -1.5 * rsr_speed_std) & (rsr_z < 1.3)
    exit_3 = ((rsr_peak - rsr_z) > 0.6) & (rsr_z < 1.6)
    base_exit = (exit_1 | exit_2 | exit_3).fillna(False)
    return rsr_z.astype(np.float32), base_exit


def _precompute_weekly_mtf_matrices(
    universe_raw: dict,
    rsr_syms: dict[str, str],
    common_dates: pd.DatetimeIndex,
    active_syms: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    MTFフィルター（signal_bridge.py:1519-1613「_mtf_state」, :2254-2280適用箇所）の再現。

    本番は毎朝「今日までのデータ」で週足 resample("W-FRI").last() を再計算するため、
    進行中の週（当日が属する週）の終値は当日Closeそのものになる
    （まだその週の金曜日のデータが存在しないため）。
    本関数はこれを「lag0=当日Close、lag13/26/39/52=確定済み過去週の終値」として
    週次ラグ参照テーブルから再現し、先読みを防止する。

    週足RSR = (0.4*wr3 + 0.2*wr6 + 0.2*wr9 + 0.2*wr12) のRSR42内パーセンタイルランク×100
      wr3=lag0/lag13-1, wr6=lag13/lag26-1, wr9=lag26/lag39-1, wr12=lag39/lag52-1
    週足MA20判定 = 当日Close > mean(lag0, lag1, ..., lag19)（直近20週、進行中週含む）
      有効な週次ラグが20本未満なら False（データ不足=通過させない、本番のfail-closedと一致）
    """
    rsr_universe_syms = [s for s in rsr_syms if s in universe_raw]
    all_syms = sorted(set(rsr_universe_syms) | set(active_syms))

    day_period_ord = common_dates.to_period("W-FRI").asi8

    lag_lookup: dict[str, dict[int, float]] = {}
    close_asof: dict[str, np.ndarray] = {}
    for sym in all_syms:
        if sym not in universe_raw:
            continue
        close_full = universe_raw[sym]["df"]["Close"]
        wc = close_full.resample("W-FRI").last().dropna()
        ords = wc.index.to_period("W-FRI").asi8
        lag_lookup[sym] = dict(zip(ords.tolist(), wc.to_numpy(dtype=np.float64)))
        close_asof[sym] = close_full.reindex(common_dates).ffill().to_numpy(dtype=np.float64)

    def _lag(sym: str, lag: int) -> np.ndarray:
        lut = lag_lookup.get(sym, {})
        if not lut:
            return np.full(len(day_period_ord), np.nan)
        return np.array([lut.get(o - lag, np.nan) for o in day_period_ord], dtype=np.float64)

    comp_by_sym: dict[str, np.ndarray] = {}
    for sym in rsr_universe_syms:
        if sym not in close_asof:
            continue
        lag13, lag26, lag39, lag52 = _lag(sym, 13), _lag(sym, 26), _lag(sym, 39), _lag(sym, 52)
        with np.errstate(invalid="ignore", divide="ignore"):
            wr3  = close_asof[sym] / lag13 - 1.0
            wr6  = lag13 / lag26 - 1.0
            wr9  = lag26 / lag39 - 1.0
            wr12 = lag39 / lag52 - 1.0
        comp_by_sym[sym] = 0.4 * wr3 + 0.2 * wr6 + 0.2 * wr9 + 0.2 * wr12

    n_dates = len(common_dates)
    n_syms = len(active_syms)
    weekly_rsr_mat = np.full((n_dates, n_syms), 0.0, dtype=np.float32)
    weekly_ma_ok_mat = np.zeros((n_dates, n_syms), dtype=bool)

    if comp_by_sym:
        comp_syms = list(comp_by_sym.keys())
        comp_mat = np.vstack([comp_by_sym[s] for s in comp_syms]).T  # n_dates x n_rsr_syms
        weekly_rsr_full = pd.DataFrame(comp_mat, columns=comp_syms).rank(axis=1, pct=True) * 100.0
        weekly_rsr_full = weekly_rsr_full.clip(0, 100)
        for j, sym in enumerate(active_syms):
            if sym in weekly_rsr_full.columns:
                weekly_rsr_mat[:, j] = weekly_rsr_full[sym].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    for j, sym in enumerate(active_syms):
        if sym not in close_asof:
            continue
        lag_stack = [close_asof[sym]] + [_lag(sym, lag) for lag in range(1, 20)]
        stacked = np.vstack(lag_stack)  # 20 x n_dates
        valid_counts = np.sum(~np.isnan(stacked), axis=0)
        with np.errstate(invalid="ignore"):
            wma20 = np.nanmean(stacked, axis=0)
            ma_ok = (valid_counts >= 20) & (close_asof[sym] > wma20)
        weekly_ma_ok_mat[:, j] = np.nan_to_num(ma_ok, nan=False).astype(bool)

    return weekly_rsr_mat, weekly_ma_ok_mat


def _calc_breadth(universe_raw: dict, ma_period: int = 200) -> pd.Series:
    """
    Market Breadth = ユニバース内で price > MA{ma_period} の銘柄比率（日次）。

    2026-03-31: RSRパーセンタイル方式（常に0.262固定の dead code）を廃止。
    price > MA200 方式に変更。shift(1) で先読み防止済み。
    """
    above_ma = {}
    for sym, info in universe_raw.items():
        close = info["df"]["Close"]
        ma    = close.rolling(ma_period, min_periods=ma_period // 2).mean()
        above_ma[sym] = (close > ma).astype(float)

    if not above_ma:
        return pd.Series(dtype=float)

    df    = pd.DataFrame(above_ma)
    ratio = df.mean(axis=1)          # 各日の「上回り銘柄比率」
    return ratio.shift(1).fillna(0.0)


def _precompute_entry_filter_frames(
    universe_raw: dict,
    trade_syms: dict[str, str],
    topix_close: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Entry filter用の時系列を銘柄ごとに事前計算する。"""
    topix_ma50 = None
    if topix_close is not None and not topix_close.empty:
        topix_ma50 = topix_close.rolling(50, min_periods=50).mean()

    frames: dict[str, pd.DataFrame] = {}
    for sym in trade_syms:
        if sym not in universe_raw:
            continue
        df = universe_raw[sym]["df"]

        atr_5_s  = _calc_atr(df, period=5)
        atr_20_s = _calc_atr(df, period=20)

        volume_src = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
        frame = pd.DataFrame({
            "atr_5": atr_5_s,
            "atr_20": atr_20_s,
            "atr20_med90": atr_20_s.rolling(90, min_periods=45).median().shift(1),
            "volume": volume_src,
            "volume_ma20": volume_src.rolling(20, min_periods=20).mean(),
        }, index=df.index)
        if topix_close is not None and not topix_close.empty:
            frame["topix_close"] = topix_close.reindex(df.index).ffill()
            frame["topix_ma50"] = topix_ma50.reindex(df.index).ffill() if topix_ma50 is not None else np.nan
        else:
            frame["topix_close"] = np.nan
            frame["topix_ma50"] = np.nan
        frames[sym] = frame
    return frames


def _alpha_weights(candidates: list[tuple], max_weight: float = MAX_POS_WEIGHT) -> dict[str, float]:
    """
    候補銘柄のαスコアに比例したウェイトを計算する。

    Parameters
    ----------
    candidates : [(alpha_score, sym), ...]  ← alphaは正値のみ渡す想定
    max_weight : 1銘柄の最大ウェイト上限

    Returns
    -------
    {sym: weight}  ← 合計=1.0, 各 ≤ max_weight
    """
    if not candidates:
        return {}
    scores = np.array([max(0.0, a) for a, _ in candidates])
    syms   = [s for _, s in candidates]

    if scores.sum() < 1e-8:
        # 全スコアゼロ → 均等ウェイト
        w = np.ones(len(syms)) / len(syms)
    else:
        w = scores / scores.sum()
        w = np.clip(w, 0, max_weight)
        w = w / w.sum()  # 正規化（clip後）

    return {sym: float(wt) for sym, wt in zip(syms, w)}


def _take_series_to_array(
    series: pd.Series,
    target_index: pd.Index,
    *,
    dtype: np.dtype = np.float32,
    fill_value: float | bool = np.nan,
) -> np.ndarray:
    out = np.full(len(target_index), fill_value, dtype=dtype)
    if series.empty or len(target_index) == 0:
        return out

    idx = series.index.get_indexer(target_index)
    valid = idx >= 0
    if np.any(valid):
        values = series.to_numpy(dtype=dtype, copy=False)
        out[valid] = values[idx[valid]]
    return out


def _take_frame_to_matrix(
    frame: pd.DataFrame,
    target_index: pd.Index,
    target_columns: list[str],
    *,
    dtype: np.dtype = np.float32,
    fill_value: float = np.nan,
) -> np.ndarray:
    out = np.full((len(target_index), len(target_columns)), fill_value, dtype=dtype)
    if frame.empty or len(target_index) == 0 or len(target_columns) == 0:
        return out

    row_idx = frame.index.get_indexer(target_index)
    col_idx = frame.columns.get_indexer(target_columns)
    valid_rows = row_idx >= 0
    valid_cols = col_idx >= 0
    if np.any(valid_rows) and np.any(valid_cols):
        values = frame.to_numpy(dtype=dtype, copy=False)
        out[np.ix_(valid_rows, valid_cols)] = values[np.ix_(row_idx[valid_rows], col_idx[valid_cols])]
    return out


# ------------------------------------------------------------------ #
# 汎用バックテスト
# ------------------------------------------------------------------ #
def run_scenario(
    scenario:       str,
    universe_raw:   dict,
    rsr_df:         pd.DataFrame,
    alpha_df:       pd.DataFrame,  # composite alpha matrix（BASELINE時は未使用）
    regime_df:      pd.DataFrame,
    trade_syms:     dict[str, str],
    rsr_syms:       dict[str, str],
    cfg,
    start:          str = START,
    end:            str = END,
    capital:        float | None = None,
    verbose:        bool = True,
    tech_matrices:  dict | None = None,
    breadth_series: pd.Series | None = None,
    breadth_stop:        float = BREADTH_STOP,
    breadth_reduce:      float = BREADTH_REDUCE,
    min_hold:            int   = 0,
    rsr_exit_threshold:  float | None = None,
    use_phase_exit:      bool  = False,
    phase_breakeven_pct: float = 0.05,
    phase_trail_start:   float = 0.10,
    phase_trail_mult:    float = 2.0,
    vol_z_matrix:        pd.DataFrame | None = None,
    vol_zscore_min:      float | None = None,
    enable_filters:      bool  = False,
    volatility_threshold: float = 1.15,
    volume_multiplier:    float = 1.3,
    enable_volatility_filter: bool = True,
    enable_atr_filter:    bool  = True,
    enable_volume_filter: bool  = True,
    enable_market_filter: bool  = True,
    topix_close:         pd.Series | None = None,
    precompute_signals:  bool  = True,
    trail_atr_mult:      float = TRAIL_ATR_MULT,
    blackout_dates:      set   | None = None,
    market_shock_mode:   str   = "full_exit",
    composite_market_thr: float = -0.05,  # composite: TOPIX 閾値（デフォルト -5%）
    composite_sym_thr:    float = -0.08,  # composite: 個別株 閾値（デフォルト -8%）
    regime_scale_series: "pd.Series | None" = None,
    sym_regime_df:       "pd.DataFrame | None" = None,
    # ── サスペンション制御 ──────────────────────────────────────────
    enable_suspension:        bool  = False,
    suspension_pnl_threshold: float = -100_000.0,  # 90日間PnLがこれ以下で発動
    suspension_win_threshold: float = 0.40,          # 勝率がこれ以下で発動（AND条件）
    suspension_lookback_days: int   = 90,            # 直近N営業日のPnLを集計
    suspension_period_days:   int   = 60,            # 停止期間（営業日）
    suspension_min_trades:    int   = 2,             # 最低取引数（統計的意義）
    # ── 動的ユニバース制御 ─────────────────────────────────────────
    sym_active_df:            "pd.DataFrame | None" = None,  # 0/1 で銘柄活性状態
    # ── サイジングモード ──────────────────────────────────────────
    sizing_mode:              str   = "existing",  # "existing"=従来のキャッシュ均等 / "equal"=capital/max_positions固定
    # ── PROD_FAITHFUL: 本番ロジック忠実再現フラグ ──────────────────────
    enable_atr_trailing_prod: bool  = False,  # signal_bridge.py:2064-2082 ATRトレーリング（3×ATR20）
    enable_multilayer_rsr:    bool  = False,  # signal_bridge.py:196-243 多層RSR z-score Exit（OR結合）
    enable_atr_risk_sizing:   bool  = False,  # signal_bridge.py:3510-3589 ATRリスクベース・サイジング
    enable_mtf_filter:        bool  = False,  # signal_bridge.py:1519-1613,2254-2280 MTFフィルター
    risk_sizing_pct:          float = PROD_RISK_PCT,
    atr_sizing_exponent:      float = 1.0,   # position ∝ 1/ATR^p （1.0=現行, 0.5=ATR_SQRT, 0.25=ATR_025）
    atr_sizing_no_risk:       bool  = False, # True: qty_risk段を無効化しqty=qty_capのみ（No ATR）
    # ── SNAPSHOT ARCHAEOLOGY (2026-06-25): 過去スナップショット忠実再現用フラグ ──
    # 推測禁止のため、各フラグは strategy_snapshot_manifest.md の評価日時点で
    # 実際に存在した/存在しなかったことが file:line で確認された機構のみを切替する。
    enable_simple_rsr_exit:   bool  = True,   # False: 絶対閾値RSR Exit無効化（2026-06-05より前の状態。当時はz-score exit_1のみ）
    use_fixed_pct_trail:      bool  = False,  # 2026-04-12確定の固定%トレイリング（HWMから-2.5%）。strategy_review_2026-04-13.md §2.3
    fixed_trail_pct:          float = 0.025,
    ml_exit_single_only:      bool  = False,  # True: 多層RSR4条件のうちexit_1(z<1.1)のみ使用（4/13当時相当）。enable_multilayer_rsr=Trueと併用
    # ── Study38 CB Forensic: サーキットブレーカー完全バイパス（計測専用）──
    bypass_cb:                bool  = False,  # True: cb_active=Falseに固定、CB_SCALEを適用しない
    # ── Study40 Exit Continuation Policy ─────────────────────────────────
    exit_policy:              str   = "NONE", # A/B/C/D/E/F or NONE=変更なし
    exit_policy_rsr_floor:    float = 65.0,   # B/F: RSR persistence floor
    exit_policy_atr_mult:     float = 1.0,    # A/F: ATR extension multiplier
    exit_policy_defer_days:   int   = 5,      # A/B/C/F: max deferral days
    exit_policy_partial_frac: float = 0.5,    # D: 残すrunner比率
    exit_policy_profit_thr:   float = 0.05,   # C/E: continuation profit threshold
    exit_policy_time_extend:  int   = 15,     # E: TIME_STOP extension days (max 2x)
    # ── Study41 Position Cap Variants ─────────────────────────────────────
    max_positions_override:   "int | None" = None,   # 静的 max_positions 上書き（stepwise elasticity用）
    max_positions_ts:         "pd.Series | None" = None,  # 動的 max_positions 系列（vol-adjusted用）
    capital_mode:             str   = "fixed",     # "fixed" | "equity_linked"
    equity_linked_step:       float = 0.10,        # 資本成長 step per additional slot (10%)
    equity_linked_max:        int   = 7,           # equity-linked 上限スロット数
    # ── Study42 Capital Constraint Archaeology ───────────────────────────────
    lot_size:                 int   = LOT,         # 売買単位（100=通常、1=単元未満/フラクション研究用）
    # ── Study44 Lot Cost Ratio Walk-Forward ──────────────────────────────────
    max_lot_cost_ratio:       "float | None" = None,  # None=現行動作（qty=0→skip）
                                                       # 0.30/0.35/0.40/0.45: lot_cost≤capital*ratio かつ cash>=lot_cost なら1ロット強制入場
    # ── Study45 Addon Expansion Walk-Forward ─────────────────────────────────
    addon_policy:             str   = "NONE",  # NONE/B/C/D/E/F
    addon_atr_mult:           float = 1.0,     # B/C/E/F: stage1 trigger (unrealized >= mult × ATR)
    addon_stage2_mult:        float = 2.0,     # C: stage2 trigger
    addon_max_per_pos:        int   = 1,       # C: max addon stages per position (1=single, 2=pyramid)
    addon_size_frac:          float = 0.25,    # D: addon size as fraction of cash (equity-scaled)
    # ── Study57 Dynamic Portfolio Optimization ─────────────────────────────
    quality_exit_pairs:  "frozenset | None" = None,  # frozenset{(sym, day_idx)} force Quality Exit
    quality_forced_exits: "dict | None"     = None,  # {day_idx: sym} → force exit this holding (swap)
    cluster_cap_override: "float | None"    = None,  # override cluster_cap_base (Phase3)
) -> dict:
    """
    汎用バックテストループ。

    scenario別の挙動:
      BASELINE      : RSR降順ランキング / 均等ウェイト / レジームなし
      STEP1         : alpha複合スコアランキング / 均等ウェイト / レジームなし
      STEP2         : alpha複合スコアランキング / 均等ウェイト / 2段階レジームフィルター
      STEP3         : alpha複合スコアランキング / αウェイト / 2段階レジームフィルター
      PROD_FAITHFUL : BASELINEのランキング/サイジング基盤 + enable_* フラグで本番Exit/Sizing/MTFを追加再現
    """
    use_alpha_rank      = (scenario not in ("BASELINE", "PROD_FAITHFUL"))
    use_regime          = (scenario in ("STEP2", "STEP3"))
    use_alpha_weight    = (scenario == "STEP3")
    use_breakout_filter = (scenario in ("STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_trail_exit      = (scenario in ("STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_breadth_regime  = (scenario in ("STEP6", "STEP6A", "STEP6B"))
    capital = float(cfg.portfolio.capital if capital is None else capital)
    rsr_exit_threshold = cfg.fujiko.rsr_exit if rsr_exit_threshold is None else rsr_exit_threshold
    filter_frames = (
        _precompute_entry_filter_frames(universe_raw, trade_syms, topix_close=topix_close)
        if enable_filters else {}
    )

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
                min_rsr=cfg.fujiko.min_rsr,
                turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s,
                min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period,
                turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
                enable_filters=enable_filters,
                volatility_threshold=volatility_threshold,
                volume_multiplier=volume_multiplier,
                enable_volatility_filter=enable_volatility_filter,
                enable_atr_filter=enable_atr_filter,
                enable_volume_filter=enable_volume_filter,
                enable_market_regime_filter=enable_market_filter,
                topix_series=topix_close,
                filter_frame=filter_frames.get(sym),
            )

    active_syms = [sym for sym in trade_syms if sym in universe_raw and sym in strats]
    if not active_syms:
        return {}

    # [2026-06-28 INTEGRITY FIX] union instead of intersection.
    # intersection forces backtest start to the latest-IPO symbol's first date
    # (e.g. 4055.T IPO 2020-08-11 truncates a 2018-2024 run to 2020-2024).
    # With union, each symbol participates only when it has data; pre-IPO dates
    # use NaN prices, so no buy is ever generated for that symbol before listing.
    common_dates: pd.DatetimeIndex | None = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.union(idx)
    if common_dates is None or len(common_dates) == 0:
        return {}
    common_dates = common_dates.sort_values()
    ts_f = pd.Timestamp(start)
    te_f = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts_f) & (common_dates <= te_f)]
    if len(common_dates) == 0:
        return {}

    n_dates = len(common_dates)
    n_syms = len(active_syms)
    need_fallback_frames = not precompute_signals
    df_by_sym: dict[str, pd.DataFrame] = {} if need_fallback_frames else {}
    open_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    high_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    signal_mat = np.zeros((n_dates, n_syms), dtype=np.int8)
    signal_ready = np.zeros(n_syms, dtype=bool)
    momentum_exit_mat = np.zeros((n_dates, n_syms), dtype=bool)
    turtle_exit_mat = np.zeros((n_dates, n_syms), dtype=bool)
    sym_to_idx = {sym: idx for idx, sym in enumerate(active_syms)}
    signal_calls = 0

    for sym_idx, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        valid = row_idx >= 0
        if not np.any(valid):
            continue  # symbol has zero overlap with requested period

        open_vals = df_src["Open"].to_numpy(dtype=np.float32, copy=False)
        close_vals = df_src["Close"].to_numpy(dtype=np.float32, copy=False)
        high_vals = df_src["High"].to_numpy(dtype=np.float32, copy=False)
        valid_pos = np.where(valid)[0]
        open_mat[valid_pos, sym_idx] = open_vals[row_idx[valid]]
        close_mat[valid_pos, sym_idx] = close_vals[row_idx[valid]]
        high_mat[valid_pos, sym_idx] = high_vals[row_idx[valid]]

        if need_fallback_frames:
            df_by_sym[sym] = df_src.take(row_idx[valid])

        strat = strats[sym]
        if precompute_signals and hasattr(strat, "precompute_signals"):
            # full dataset を渡して look-ahead-free なシグナルを計算し、
            # OOS/IS 期間のインデックス位置だけを切り出す。
            # ★修正前: df_src.take(row_idx) → OOS 243行のみ → min_bars=275 未達 → 全ゼロ
            # ★修正後: df_src 全体(1967行) → min_bars 充足 → 正常シグナル
            required_bars = 252 + getattr(strat, "mom_period", 21) + 2
            if len(df_src) < required_bars:
                raise ValueError(
                    f"{sym}: insufficient bars {len(df_src)} < {required_bars}"
                )
            sig_series_full = strat.precompute_signals(df_src)
            sig_np = sig_series_full.to_numpy(dtype=np.int8, copy=False)
            signal_mat[valid_pos, sym_idx] = sig_np[row_idx[valid]]
            signal_ready[sym_idx] = True
            signal_calls += int(np.sum(valid))

            # PROD_FAITHFUL: Exit内訳集計用（turtle vs RSRモメンタムfallback）
            _mom_mask_full = getattr(strat, "exit_momentum_mask_full", None)
            _turtle_mask_full = getattr(strat, "exit_turtle_mask_full", None)
            if _mom_mask_full is not None:
                momentum_exit_mat[valid_pos, sym_idx] = _mom_mask_full[row_idx[valid]]
            if _turtle_mask_full is not None:
                turtle_exit_mat[valid_pos, sym_idx] = _turtle_mask_full[row_idx[valid]]

    rsr_mat = np.nan_to_num(
        _take_frame_to_matrix(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )
    alpha_mat = (
        np.nan_to_num(
            _take_frame_to_matrix(alpha_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
            nan=0.0,
        )
        if use_alpha_rank else np.zeros((n_dates, n_syms), dtype=np.float32)
    )

    risk_off_arr = np.zeros(n_dates, dtype=bool)
    crash_arr = np.zeros(n_dates, dtype=bool)
    if use_regime:
        risk_off_arr = _take_series_to_array(regime_df["risk_off"], common_dates, dtype=bool, fill_value=False)
        crash_arr = _take_series_to_array(regime_df["crash"], common_dates, dtype=bool, fill_value=False)

    breadth_arr = None
    if use_breadth_regime and breadth_series is not None:
        breadth_arr = _take_series_to_array(breadth_series, common_dates, dtype=np.float32, fill_value=np.nan)

    atr20_mat = high200_mat = high50_close_mat = vol_z_mat = None
    if tech_matrices is not None:
        if "atr20" in tech_matrices:
            atr20_mat = _take_frame_to_matrix(tech_matrices["atr20"], common_dates, active_syms, dtype=np.float32, fill_value=np.nan)
        if "high200" in tech_matrices:
            high200_mat = _take_frame_to_matrix(tech_matrices["high200"], common_dates, active_syms, dtype=np.float32, fill_value=np.nan)
        if "high50_close" in tech_matrices:
            high50_close_mat = _take_frame_to_matrix(tech_matrices["high50_close"], common_dates, active_syms, dtype=np.float32, fill_value=np.nan)
    if vol_z_matrix is not None:
        vol_z_mat = _take_frame_to_matrix(vol_z_matrix, common_dates, active_syms, dtype=np.float32, fill_value=np.nan)

    _atr_ref_for_sizing = None
    if enable_atr_risk_sizing and not atr_sizing_no_risk and atr_sizing_exponent != 1.0 and atr20_mat is not None:
        # ATR^p の単位を揃えるための正規化基準値（全期間・全銘柄のATR中央値、固定スカラー）
        _atr_ref_for_sizing = float(np.nanmedian(atr20_mat[atr20_mat > 0])) if np.any(atr20_mat > 0) else None

    # ── PROD_FAITHFUL: 多層RSR Exit（hold_days非依存部分）の事前計算 ──────────
    ml_z_mat: "np.ndarray | None" = None
    ml_base_exit_mat: "np.ndarray | None" = None
    if enable_multilayer_rsr:
        _ml_z_df, _ml_base_exit_df = _precompute_multilayer_rsr_matrices(rsr_df)
        ml_z_mat = _take_frame_to_matrix(_ml_z_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan)
        ml_base_exit_mat = _take_frame_to_matrix(
            _ml_base_exit_df.astype(np.float32), common_dates, active_syms, dtype=np.float32, fill_value=0.0
        ) > 0.5

    # ── PROD_FAITHFUL: MTFフィルター（週足RSR + 週足MA20）の事前計算 ─────────
    weekly_rsr_mat: "np.ndarray | None" = None
    weekly_ma_ok_mat: "np.ndarray | None" = None
    if enable_mtf_filter:
        weekly_rsr_mat, weekly_ma_ok_mat = _precompute_weekly_mtf_matrices(
            universe_raw, rsr_syms, common_dates, active_syms
        )

    # sym_regime_df: 個別銘柄モメンタム連動サイジング行列（n_dates × n_syms）
    sym_regime_mat: np.ndarray | None = None
    if sym_regime_df is not None and not sym_regime_df.empty:
        sym_regime_mat = _take_frame_to_matrix(sym_regime_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)

    # sym_active_df: 動的ユニバース活性マトリクス（1=活性/エントリー可, 0=非活性/新規禁止）
    sym_active_mat: np.ndarray | None = None
    if sym_active_df is not None and not sym_active_df.empty:
        sym_active_mat = _take_frame_to_matrix(sym_active_df, common_dates, active_syms, dtype=np.float32, fill_value=1.0)

    # ── セクター / 銘柄 集中キャップ ──────────────────────────────────
    # sym_active_df が指定されている場合のみ適用（動的ユニバースと組み合わせて使用）
    _enable_conc_caps = sym_active_df is not None
    # feature flag: risk_controls.dynamic_cap が False の場合は固定 cap を使用
    _rc = getattr(cfg, "risk_controls", None)
    _use_dynamic_cap = bool(_rc.dynamic_cap) if _rc is not None else False
    MAX_SECTOR_WEIGHT = float(_rc.sector_cap) if _rc is not None else 0.25
    MAX_SYMBOL_WEIGHT = float(_rc.symbol_cap) if _rc is not None else 0.08

    # ── 動的 cap 事前計算 ──────────────────────────────────────────
    # _enable_conc_caps かつ _use_dynamic_cap の場合のみ有効。
    # dynamic_cap: false（production default）では固定 cap を使用し、この計算をスキップ。
    _sym_cap_df: "pd.DataFrame | None" = None
    _sec_cap_df: "pd.DataFrame | None" = None
    if _enable_conc_caps and _use_dynamic_cap:
        _close_all = pd.DataFrame({
            s: universe_raw[s]["df"]["Close"]
            for s in active_syms if s in universe_raw
        }).sort_index()
        _sym_cap_df, _sec_cap_df = precompute_dynamic_caps(_close_all, trade_syms)

    # ── Cluster cap / Bear adaptive cap 初期化 ──────────────────────
    _cluster_cap_base  = float(getattr(_rc, "cluster_cap",      0.35)) if _rc else 0.35
    _bear_sector_cap   = float(getattr(_rc, "bear_sector_cap",  0.18)) if _rc else 0.18
    _bear_cluster_cap  = float(getattr(_rc, "bear_cluster_cap", 0.25)) if _rc else 0.25

    # ── Gross Exposure cap 初期化（2026-04-07 追加）─────────────────
    _gross_exposure_enabled = bool(getattr(_rc, "gross_exposure_enabled", True))  if _rc else True
    _gross_cap_normal       = float(getattr(_rc, "gross_cap_normal",       1.0))  if _rc else 1.0
    _gross_cap_dd5          = float(getattr(_rc, "gross_cap_drawdown_5pct", 0.6)) if _rc else 0.6
    _gross_cap_dd8          = float(getattr(_rc, "gross_cap_drawdown_8pct", 0.4)) if _rc else 0.4
    _gross_cap_active_days  = 0

    # TOPIX MA200 を使って Bear 日を事前フラグ化（_bear_arr）
    # True = TOPIX が MA200 を下回る日 → bear_sector_cap / bear_cluster_cap を使用
    _bear_arr: "np.ndarray | None" = None
    if topix_close is not None and not topix_close.empty:
        _topix_ma200 = topix_close.rolling(200, min_periods=100).mean()
        _bear_s = (topix_close < _topix_ma200).reindex(
            pd.Index(common_dates), method="ffill"
        ).fillna(False)
        _bear_arr = _bear_s.values.astype(bool)

    cash = float(capital)
    positions: dict[str, Position] = {}
    equity_curve = []
    long_notional_list: list[float] = []   # 日次ロングノーショナル（実投資額）
    exposure_list = []
    pos_list = []
    cand_list = []
    trades: list[dict] = []
    exit_reason_counts: dict[str, int] = {}
    peak_equity = float(capital)
    max_hold_limit = getattr(cfg.risk, "max_hold_days", None)
    cb_active = False
    cb_days = 0
    # Study40 exit policy state (empty when exit_policy="NONE" → no overhead)
    _ep_deferred:    dict[str, int]   = {}   # sym → deadline day-index (A/B/C/F)
    _ep_runner_stop: dict[str, float] = {}   # sym → runner trail stop price (D)
    _ep_time_ext:    dict[str, int]   = {}   # sym → extensions used (E)
    # Study48 instrumentation: ATR Extension firing counters
    _atr_ext_defer_count:   int  = 0   # number of times ATR Extension deferred an exit
    _atr_ext_detail: list        = []  # per-deferral records (capped at 2000)
    reentry_ban: dict[str, int] = {}
    # ── サスペンション（構造的ルーザー銘柄の一時停止）──────────────
    suspended_until: dict[str, int] = {}   # sym -> 停止解除インデックス
    sym_trade_log:   dict[str, list] = {}  # sym -> [(exit_idx, pnl), ...]

    # ── スキップ統計（cap起因のBUY候補除外カウント）─────────────────
    skip_stats: dict[str, int] = {"sector_cap": 0, "cluster_cap": 0, "bear_adaptive": 0, "gross_exposure": 0}

    # ── Study38 CB計測変数（bypass_cb=Falseの時のみ意味を持つ）──────
    _cb_trigger_count   = 0    # cb_active が True に切り替わった回数
    _cb_active_days     = 0    # 日次ループ内で cb_active=True だった日数
    _cb_scaled_entries  = 0    # CB発動中に実行されたBUYエントリー数
    _capital_suppressed = 0.0  # CB_SCALEにより縮小された分の累積配分額（円）

    # ── Study41 Capital Utilization 計測変数 ──────────────────────────
    _initial_capital    = capital          # 期首資本（equity-linked成長率計算の基準）
    _idle_cash_total    = 0.0             # 日次アイドルキャッシュ累計（対initial_capital比）
    _days_at_max        = 0              # ポジション数がeff_max_posに達した日数
    _missed_by_cap      = 0             # ポジション上限でエントリーできなかったBUY候補数
    _missed_cands: list[dict] = []      # 詳細（後処理でfwd_return計算用）

    # ── Study42 Lot-Constraint Rejection 計測変数 ──────────────────────
    _rejected_by_lot      = 0             # qty=0(lot切り捨て)でスキップされたBUY候補数
    _rejected_by_lot_detail: list[dict] = []  # 詳細（後処理でfwd_return計算用）

    # ── Study53 Opportunity Loss 計測変数 ────────────────────────────────
    _skip_detail: list[dict] = []  # sector/cluster/gross per-candidate rejection log

    # ── Study44 Lot Cost Ratio 計測変数 ─────────────────────────────────
    _admitted_by_ratio_count  = 0            # ratio救済で1ロット強制入場した件数
    _admitted_by_ratio_detail: list[dict] = []  # 詳細（後処理でfwd_return計算用）

    # ── Study45 Q1-Q3 Attribution 計測変数 ───────────────────────────────
    _q_days_total        = 0               # ループした全日数
    _q_idle_days         = 0               # cash > 0 の日数
    _q_idle_cash_sum     = 0.0             # idle日のcash/capital累計
    _q_winner_idle_days  = 0               # idle + 1×ATR超えwinnerがある日数
    _q_deployable_sum    = 0.0             # その日のcash/capital（winner有り日のみ）

    # ── Study45 Addon State ───────────────────────────────────────────────
    _addon_done: dict[str, int] = {}       # sym → 達成済みstage数 (0=未, 1=stage1済, 2=stage2済)
    _addon_count  = 0
    _addon_detail: list[dict] = []

    # ── PROD_FAITHFUL: Entry Analysis集計用カウンタ ─────────────────
    mtf_excluded_count = 0
    dyn_universe_excluded_count = 0
    fujiko_entry_count = 0
    meanrev_entry_count = 0

    # ── Market shock detection: 1306.T 日次リターン -5%以下で翌寄り全決済 ──
    market_ret_arr = np.zeros(n_dates, dtype=np.float32)
    topix_ret_20d_arr: "np.ndarray | None" = None
    topix_ret_60d_arr: "np.ndarray | None" = None
    if topix_close is not None and not topix_close.empty:
        topix_ret = topix_close.pct_change()
        market_ret_arr = _take_series_to_array(
            topix_ret, common_dates, dtype=np.float32, fill_value=0.0
        )
        # gross exposure cap 用: 20日・60日リターン配列
        topix_ret_20d_arr = _take_series_to_array(
            topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0
        )
        topix_ret_60d_arr = _take_series_to_array(
            topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0
        )

    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_idx[sym]]) for sym, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        long_notional_list.append(invested)          # ロングノーショナル（実投資額）
        exposure_list.append(invested / max(1.0, cur_equity))
        pos_list.append(len(positions))

        # ── Study41: effective max positions（日次動的決定）──────────────
        if max_positions_override is not None:
            _eff_max_pos = max_positions_override
        elif max_positions_ts is not None and date in max_positions_ts.index:
            _eff_max_pos = int(max_positions_ts[date])
        elif capital_mode == "equity_linked":
            _growth = max(0.0, cur_equity / _initial_capital - 1.0)
            _eff_max_pos = min(equity_linked_max,
                               cfg.portfolio.max_positions + int(_growth / equity_linked_step))
        else:
            _eff_max_pos = cfg.portfolio.max_positions
        # 資本稼働率追跡
        _idle_cash_total += cash / max(1.0, _initial_capital)
        if len(positions) >= _eff_max_pos:
            _days_at_max += 1

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        if not bypass_cb:
            if not cb_active:
                if dd <= -cfg.portfolio.max_dd_limit:
                    cb_active = True
                    cb_days = 0
                    _cb_trigger_count += 1
            else:
                cb_days += 1
                if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                    cb_active = False
                    cb_days = 0
        if cb_active:
            _cb_active_days += 1

        risk_off = bool(risk_off_arr[i]) if use_regime else False
        crash = bool(crash_arr[i]) if use_regime else False
        regime_step = 0.5 if crash else 1.0

        if use_breadth_regime and breadth_arr is not None and i < len(breadth_arr):
            bval = float(breadth_arr[i])
            if not np.isnan(bval):
                if bval < breadth_stop:
                    risk_off = True
                if bval < breadth_reduce:
                    regime_step = min(regime_step, 0.5)

        sell_signals: list[tuple] = []
        buy_candidates: list[tuple] = []

        # Study57: Quality Swap — pre-scheduled force exits for today
        if quality_forced_exits and i in quality_forced_exits:
            _qfe_sym = quality_forced_exits[i]
            if _qfe_sym in positions:
                sell_signals.append((_qfe_sym, "QUALITY_SWAP_EXIT"))

        # Market shock detection: TOPIX が composite_market_thr 以下
        market_shock = bool(float(market_ret_arr[i]) <= composite_market_thr)
        if market_shock:
            if market_shock_mode == "full_exit":
                # 現行: 全ポジション強制全量決済
                for pos_sym in list(positions.keys()):
                    sell_signals.append((pos_sym, "MARKET_SHOCK_EXIT"))
            elif market_shock_mode == "partial_50":
                # 50%部分決済: 翌寄り半量売却、残り保持
                for pos_sym in list(positions.keys()):
                    sell_signals.append((pos_sym, "MARKET_SHOCK_PARTIAL"))
            # "composite": 個別銘柄ループ内で sym_ret <= -8% のものだけ決済

        for sym in active_syms:
            if market_shock and market_shock_mode in ("full_exit", "partial_50"):
                continue  # 市場ショック日: 決済処理済み、新規買いも停止

            sym_idx = sym_to_idx[sym]
            strat = strats[sym]
            is_holding = sym in positions
            hold_idx = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val = float(rsr_mat[i, sym_idx])
            close_today = float(close_mat[i, sym_idx])

            # composite mode: 市場ショック日に個別銘柄リターン -8% 以下のみ決済、新規買い禁止
            if market_shock and market_shock_mode == "composite":
                if not is_holding:
                    continue  # 新規買い禁止
                if i > 0:
                    prev_c = float(close_mat[i - 1, sym_idx])
                    if prev_c > 0:
                        sym_ret_i = close_today / prev_c - 1.0
                        if sym_ret_i <= composite_sym_thr:
                            sell_signals.append((sym, "MARKET_SHOCK_EXIT"))
                            continue
                # 個別株が-8%未満なら通常エグジットルールに続ける

            # Study57: Quality Exit Override
            if quality_exit_pairs and is_holding and (sym, i) in quality_exit_pairs and hold_idx >= min_hold:
                sell_signals.append((sym, "QUALITY_EXIT"))
                continue

            # ── PROD_FAITHFUL: ATRトレーリングストップ（signal_bridge.py:2064-2082）──
            # 優先度: 時間ストップより先（本番と同順）。保有来高値(Close基準)-3×ATR20。
            if enable_atr_trailing_prod and is_holding:
                _prev_highest = positions[sym].highest_close
                _highest = max(_prev_highest, close_today)
                positions[sym].highest_close = _highest
                _atr_for_trail = float(atr20_mat[i, sym_idx]) if atr20_mat is not None else np.nan
                if not np.isnan(_atr_for_trail) and _atr_for_trail > 0:
                    if exit_policy == "D" and sym in _ep_runner_stop:
                        _ep_runner_stop[sym] = max(_ep_runner_stop[sym], _highest - 2.0 * _atr_for_trail)
                    _trail_stop_price = _highest - PROD_ATR_TRAIL_MULT * _atr_for_trail
                    if close_today < _trail_stop_price:
                        sell_signals.append((sym, "ATR_TRAILING"))
                        continue

            # Policy D: runner trail stop check
            if exit_policy == "D" and is_holding and sym in _ep_runner_stop:
                if close_today < _ep_runner_stop.get(sym, -1e18):
                    sell_signals.append((sym, "RUNNER_TRAIL_EXIT"))
                    del _ep_runner_stop[sym]
                    continue

            # Policy A/B/C/F: deferred exit deadline
            if exit_policy in ("A", "B", "C", "F") and is_holding and sym in _ep_deferred:
                if i >= _ep_deferred[sym]:
                    sell_signals.append((sym, "DEFERRED_EXIT"))
                    del _ep_deferred[sym]
                    continue

            if is_holding and max_hold_limit is not None:
                _eff_hold_limit = max_hold_limit
                if exit_policy == "E" and sym in _ep_time_ext:
                    _eff_hold_limit += _ep_time_ext[sym] * exit_policy_time_extend
                if hold_idx > _eff_hold_limit:
                    if exit_policy == "E" and _ep_time_ext.get(sym, 0) < 2 and positions[sym].entry_price > 0:
                        _profit_e = (close_today - positions[sym].entry_price) / positions[sym].entry_price
                        if _profit_e >= exit_policy_profit_thr:
                            _ep_time_ext[sym] = _ep_time_ext.get(sym, 0) + 1
                        else:
                            sell_signals.append((sym, "TIME_STOP"))
                            continue
                    else:
                        sell_signals.append((sym, "TIME_STOP"))
                        continue

            # ── RSR Exit: シンプル閾値（本番常時） + 多層z-score（PROD_FAITHFUL時OR結合）──
            # SNAPSHOT ARCHAEOLOGY: enable_simple_rsr_exit=False で絶対閾値を無効化（2026-06-05以前の再現）
            _simple_rsr_exit = (rsr_val < rsr_exit_threshold) if enable_simple_rsr_exit else False
            _ml_rsr_exit = False
            if enable_multilayer_rsr and is_holding and ml_z_mat is not None and ml_base_exit_mat is not None:
                _z_now = float(ml_z_mat[i, sym_idx])
                if ml_exit_single_only:
                    # 4/13当時相当: exit_1(z<1.1)のみ（exit_2/3/4・多層構造は未導入時点の再現）
                    _ml_rsr_exit = (not np.isnan(_z_now)) and (_z_now < 1.1)
                else:
                    _ml_base = bool(ml_base_exit_mat[i, sym_idx])
                    _ml_exit_4 = (hold_idx >= PROD_ML_HOLD_DAYS) and (not np.isnan(_z_now)) and (_z_now < 1.2)
                    _ml_rsr_exit = _ml_base or _ml_exit_4
            if is_holding and (_simple_rsr_exit or _ml_rsr_exit) and hold_idx >= min_hold:
                if exit_policy in ("A", "B", "C", "F") and sym not in _ep_deferred:
                    _ep_entry = positions[sym].entry_price
                    _pnow = (close_today - _ep_entry) / _ep_entry if _ep_entry > 0 else 0.0
                    _atr_ep = float(atr20_mat[i, sym_idx]) if atr20_mat is not None else np.nan
                    _defer = False
                    if exit_policy == "A":
                        if _pnow > 0 and not np.isnan(_atr_ep) and _atr_ep > 0:
                            _defer = close_today > (positions[sym].highest_close - exit_policy_atr_mult * _atr_ep)
                    elif exit_policy == "B":
                        _defer = rsr_val >= exit_policy_rsr_floor
                    elif exit_policy == "C":
                        _defer = _pnow >= 0.0
                    elif exit_policy == "F":
                        if _pnow > 0 and rsr_val >= exit_policy_rsr_floor and not np.isnan(_atr_ep) and _atr_ep > 0:
                            _defer = close_today > (positions[sym].highest_close - exit_policy_atr_mult * _atr_ep)
                    if _defer:
                        _ep_deferred[sym] = i + exit_policy_defer_days
                        _atr_ext_defer_count += 1
                        if len(_atr_ext_detail) < 2000:
                            _atr_ext_detail.append({
                                "date":        str(date.date()) if hasattr(date, "date") else str(date),
                                "symbol":      sym,
                                "close":       round(float(close_today), 2),
                                "highest":     round(float(positions[sym].highest_close), 2),
                                "atr":         round(float(_atr_ep), 2),
                                "threshold":   round(float(positions[sym].highest_close - exit_policy_atr_mult * _atr_ep), 2),
                                "pnl_pct":     round(float(_pnow) * 100, 2),
                                "expire_idx":  i + exit_policy_defer_days,
                            })
                    else:
                        sell_signals.append((sym, "RSR_EXIT"))
                else:
                    sell_signals.append((sym, "RSR_EXIT"))
                continue

            # ── SNAPSHOT ARCHAEOLOGY: 固定%トレイリング HWM更新のみ（判定は優先順位末尾、後述） ──
            # strategy_review_2026-04-13.md §2.3 の優先順位(1)turtle (2)RSR momentum (3)trailing に
            # 整合させるため、トリガー判定はターtle/RSRモメンタム判定(sig)より後で行う。
            if use_fixed_pct_trail and is_holding:
                positions[sym].highest_close = max(positions[sym].highest_close, close_today)

            if use_phase_exit and is_holding and hold_idx >= min_hold and atr20_mat is not None:
                entry_px = positions[sym].entry_price
                profit = (close_today - entry_px) / entry_px if entry_px > 0 else 0.0
                if profit >= phase_breakeven_pct and close_today <= entry_px:
                    sell_signals.append((sym, "BREAKEVEN_STOP"))
                    continue
                if profit >= phase_trail_start:
                    atr_val = float(atr20_mat[i, sym_idx])
                    if not (np.isnan(atr_val) or atr_val <= 0):
                        entry_idx = positions[sym].entry_idx
                        high_since = float(np.nanmax(high_mat[entry_idx:i + 1, sym_idx]))
                        trail_px = high_since - phase_trail_mult * atr_val
                        if close_today < trail_px:
                            sell_signals.append((sym, "ATR_TRAIL"))
                            continue

            if use_trail_exit and is_holding and atr20_mat is not None and high50_close_mat is not None:
                h50c = float(high50_close_mat[i, sym_idx])
                atr20 = float(atr20_mat[i, sym_idx])
                if not (np.isnan(h50c) or np.isnan(atr20) or atr20 <= 0):
                    trail_stop = h50c - trail_atr_mult * atr20
                    if close_today < trail_stop:
                        sell_signals.append((sym, "TRAIL_EXIT"))
                        continue

            if not signal_ready[sym_idx]:
                sig = strats[sym].generate_signal(df_by_sym[sym].iloc[: i + 1])
                signal_calls += 1
            else:
                sig = int(signal_mat[i, sym_idx])

            if sig == -1 and is_holding and hold_idx >= min_hold:
                # Exit内訳タグ付け（fujiko_strategy.py exit_momentum/turtle_mask、優先: momentum>turtle）
                _exit_tag = "STRATEGY_EXIT"
                if bool(momentum_exit_mat[i, sym_idx]):
                    _exit_tag = "RSR_MOMENTUM_EXIT"
                elif bool(turtle_exit_mat[i, sym_idx]):
                    _exit_tag = "TURTLE_EXIT"
                # Policy C: TURTLE_EXIT を defer（まだ利益圏なら5日延命）
                if exit_policy == "C" and _exit_tag == "TURTLE_EXIT" and sym not in _ep_deferred:
                    _ep_c = positions[sym].entry_price
                    _pc = (close_today - _ep_c) / _ep_c if _ep_c > 0 else 0.0
                    if _pc >= exit_policy_profit_thr:
                        _ep_deferred[sym] = i + exit_policy_defer_days
                    else:
                        sell_signals.append((sym, _exit_tag))
                else:
                    sell_signals.append((sym, _exit_tag))
            elif (use_fixed_pct_trail and is_holding and hold_idx >= min_hold
                  and close_today < positions[sym].highest_close * (1.0 - fixed_trail_pct)):
                # SNAPSHOT ARCHAEOLOGY: 固定%トレイリング判定（優先順位3、turtle/RSRモメンタムの後）
                sell_signals.append((sym, "FIXED_PCT_TRAILING"))
            elif sig == 1 and not is_holding:
                ban_until = reentry_ban.get(sym, -1)
                if i < ban_until:
                    continue
                # サスペンション中はエントリー禁止
                if enable_suspension and i < suspended_until.get(sym, 0):
                    continue
                # 動的ユニバース: 非活性銘柄は新規エントリー禁止（保有中は継続）
                if sym_active_mat is not None:
                    active_val = float(sym_active_mat[i, sym_to_idx[sym]])
                    if active_val < 0.5:
                        dyn_universe_excluded_count += 1
                        continue

                # ── PROD_FAITHFUL: MTFフィルター（週足RSR≥min_rsr かつ週足Close>週足MA20）──
                # signal_bridge.py:2257「rsr_now >= min_rsr_threshold」と同じ事前ゲート後に評価
                if enable_mtf_filter and weekly_rsr_mat is not None and weekly_ma_ok_mat is not None:
                    if rsr_val >= cfg.fujiko.min_rsr:
                        _wrsr_ok = float(weekly_rsr_mat[i, sym_idx]) >= cfg.fujiko.min_rsr
                        _wma_ok = bool(weekly_ma_ok_mat[i, sym_idx])
                        if not (_wrsr_ok and _wma_ok):
                            mtf_excluded_count += 1
                            continue

                breakout_mult = 1.0
                if use_breakout_filter and high200_mat is not None:
                    h200 = float(high200_mat[i, sym_idx])
                    if not np.isnan(h200) and close_today >= h200:
                        breakout_mult = BREAKOUT_BONUS

                a_val = float(alpha_mat[i, sym_idx])
                rank_score = (max(0.0, a_val) ** 2) * rsr_val * breakout_mult if use_alpha_rank else rsr_val

                if vol_zscore_min is not None and vol_z_mat is not None:
                    vz = float(vol_z_mat[i, sym_idx])
                    if np.isnan(vz) or vz < vol_zscore_min:
                        continue

                buy_candidates.append((rank_score, a_val, rsr_val, sym))

        cand_list.append(len(buy_candidates))

        if i + 1 >= len(common_dates):
            break

        next_i = i + 1
        for sym, reason in sell_signals:
            if sym not in positions:
                continue
            pos = positions[sym]
            sell_px = float(open_mat[next_i, sym_to_idx[sym]])

            if reason == "MARKET_SHOCK_PARTIAL":
                # 50%部分決済: 半量売却、残り保持
                half_qty = (pos.qty // 2 // lot_size) * lot_size
                qty_to_sell = half_qty if half_qty > 0 else pos.qty  # 1ロットしかない場合は全量
                proceeds = qty_to_sell * sell_px * (1 - COST_ONE_WAY)
                pnl = (sell_px - pos.entry_price) * qty_to_sell
                cash += proceeds
                trades.append({
                    "symbol": sym, "side": "SELL", "sector": pos.sector,
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": qty_to_sell, "pnl": pnl, "reason": reason,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                })
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                positions[sym].qty -= qty_to_sell
                if positions[sym].qty <= 0:
                    del positions[sym]
                    _ep_deferred.pop(sym, None)
                    _ep_runner_stop.pop(sym, None)
                    _ep_time_ext.pop(sym, None)
            elif (exit_policy == "D" and sym not in _ep_runner_stop
                  and reason not in ("RUNNER_TRAIL_EXIT", "DEFERRED_EXIT", "ATR_TRAILING", "MARKET_SHOCK_EXIT")):
                # Policy D: 初回exit signal → partial_frac分だけ売却 + runner設定
                _d_sell = (int(pos.qty * exit_policy_partial_frac) // lot_size) * lot_size
                if 0 < _d_sell < pos.qty:
                    _d_proc = _d_sell * sell_px * (1 - COST_ONE_WAY)
                    _d_pnl  = (sell_px - pos.entry_price) * _d_sell
                    cash += _d_proc
                    trades.append({
                        "symbol": sym, "side": "SELL", "sector": pos.sector,
                        "entry": pos.entry_price, "exit": sell_px,
                        "qty": _d_sell, "pnl": _d_pnl, "reason": reason + "_D_PARTIAL",
                        "entry_idx": pos.entry_idx, "exit_idx": i,
                    })
                    exit_reason_counts[reason + "_D_PARTIAL"] = exit_reason_counts.get(reason + "_D_PARTIAL", 0) + 1
                    positions[sym].qty -= _d_sell
                    _atr_d = float(atr20_mat[next_i, sym_to_idx[sym]]) if atr20_mat is not None else np.nan
                    if not np.isnan(_atr_d) and _atr_d > 0:
                        _ep_runner_stop[sym] = positions[sym].highest_close - 2.0 * _atr_d
                    else:
                        _ep_runner_stop[sym] = sell_px * 0.95
                else:
                    # qty too small for partial → full exit
                    proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
                    pnl = (sell_px - pos.entry_price) * pos.qty
                    cash += proceeds
                    trades.append({
                        "symbol": sym, "side": "SELL", "sector": pos.sector,
                        "entry": pos.entry_price, "exit": sell_px,
                        "qty": pos.qty, "pnl": pnl, "reason": reason,
                        "entry_idx": pos.entry_idx, "exit_idx": i,
                    })
                    exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                    del positions[sym]
                    _ep_deferred.pop(sym, None)
                    _ep_runner_stop.pop(sym, None)
                    _ep_time_ext.pop(sym, None)
                    if reason == "TIME_STOP":
                        reentry_ban[sym] = i + 1 + REENTRY_COOL
            else:
                # 全量決済（通常）
                proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
                pnl = (sell_px - pos.entry_price) * pos.qty
                cash += proceeds
                trades.append({
                    "symbol": sym, "side": "SELL", "sector": pos.sector,
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": pos.qty, "pnl": pnl, "reason": reason,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                })
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                del positions[sym]
                _ep_deferred.pop(sym, None)
                _ep_runner_stop.pop(sym, None)
                _ep_time_ext.pop(sym, None)
                if reason == "TIME_STOP":
                    reentry_ban[sym] = i + 1 + REENTRY_COOL
                # サスペンション判定: 全量決済後に rolling PnL チェック
                if enable_suspension:
                    if sym not in sym_trade_log:
                        sym_trade_log[sym] = []
                    sym_trade_log[sym].append((i, pnl))
                    # lookback 外の古いエントリーを削除
                    cutoff = i - suspension_lookback_days
                    sym_trade_log[sym] = [
                        (idx, p) for idx, p in sym_trade_log[sym] if idx >= cutoff
                    ]
                    recent = sym_trade_log[sym]
                    if len(recent) >= suspension_min_trades:
                        rolling_pnl = sum(p for _, p in recent)
                        win_rate_r = sum(1 for _, p in recent if p > 0) / len(recent)
                        if (rolling_pnl < suspension_pnl_threshold
                                and win_rate_r < suspension_win_threshold):
                            suspended_until[sym] = i + suspension_period_days

        # 外部レジームスケール（regime_scale_series から当日のスケール係数を取得）
        ext_scale = 1.0
        if regime_scale_series is not None:
            rs_val = regime_scale_series.get(date, None)
            if rs_val is not None and not np.isnan(float(rs_val)):
                ext_scale = float(rs_val)
            # MA200下（scale=0.25）のとき新規買いをブロック
            if ext_scale <= 0.0:
                ext_scale = 0.0

        buy_blocked = (use_regime and risk_off) or (blackout_dates is not None and date in blackout_dates)
        # regime_scale_series が全停止指定（scale=0）なら買いブロック
        if ext_scale <= 0.0:
            buy_blocked = True

        if not buy_blocked and buy_candidates:
            buy_candidates.sort(key=lambda x: -x[0])

            if use_alpha_weight:
                top_cands = buy_candidates[:_eff_max_pos]
                wt_map = _alpha_weights([(a, s) for _, a, _, s in top_cands], max_weight=MAX_POS_WEIGHT)
            else:
                wt_map = {}

            cb_scale = 1.0 if (bypass_cb or not cb_active) else CB_SCALE

            # ── Bear adaptive cap（TOPIX < MA200 の日は tight cap を適用）────
            _is_bear = bool(_bear_arr[i]) if _bear_arr is not None else False
            _sec_cap_eff = _bear_sector_cap  if _is_bear else MAX_SECTOR_WEIGHT
            _cls_base_eff = cluster_cap_override if cluster_cap_override is not None else _cluster_cap_base
            _cls_cap_eff = _bear_cluster_cap if _is_bear else _cls_base_eff

            # ── 日次 gross exposure cap 計算（2026-04-07 追加）──────────────
            _gross_cap = _gross_cap_normal
            if _gross_exposure_enabled:
                _r20 = float(topix_ret_20d_arr[i]) if topix_ret_20d_arr is not None else 0.0
                _r60 = float(topix_ret_60d_arr[i]) if topix_ret_60d_arr is not None else 0.0
                if _r20 < -0.05:
                    _gross_cap = _gross_cap_dd5
                elif _r60 < -0.08:
                    _gross_cap = _gross_cap_dd8
                if _gross_cap < _gross_cap_normal:
                    _gross_cap_active_days += 1

            for _rank_idx, (rank_score, a_val, rsr_val, sym) in enumerate(buy_candidates):
                # Study53: per-candidate atr_pct (close-based, current day)
                _c53_sidx = sym_to_idx.get(sym, -1)
                if _c53_sidx >= 0 and atr20_mat is not None:
                    _c53_c = float(close_mat[i, _c53_sidx])
                    _c53_a = float(atr20_mat[i, _c53_sidx])
                    _c53_atr_pct = round(_c53_a / _c53_c * 100, 2) if _c53_c > 0 and not np.isnan(_c53_a) else 0.0
                else:
                    _c53_atr_pct = 0.0

                open_slots = _eff_max_pos - len(positions)
                if open_slots <= 0:
                    # Study41: 上限起因の見逃しエントリーを記録
                    _missed_by_cap += 1
                    if len(_missed_cands) < 5000:
                        _missed_cands.append({
                            "date": str(date.date()) if hasattr(date, "date") else str(date),
                            "symbol": sym, "rsr": round(float(rsr_val), 1),
                            "composite_score": round(float(rank_score), 4),
                            "alpha": round(float(a_val), 4),
                            "atr_pct": _c53_atr_pct,
                            "rank": _rank_idx,
                        })
                    continue  # breakからcontinueへ: 全見逃し候補をカウント

                buy_px = float(open_mat[next_i, sym_to_idx[sym]])
                if buy_px <= 0:
                    continue

                # ── セクター集中キャップ（bear adaptive / 動的 / 固定）────────
                if _enable_conc_caps:
                    _sector_sym = trade_syms.get(sym, "不明")
                    # bear adaptive 優先。dynamic_cap=True なら更に tight な方を使用
                    _eff_sec = _sec_cap_eff
                    if _use_dynamic_cap and _sec_cap_df is not None:
                        _eval_date = common_dates[i]
                        if _eval_date in _sec_cap_df.index and _sector_sym in _sec_cap_df.columns:
                            _eff_sec = min(_eff_sec, float(_sec_cap_df.loc[_eval_date, _sector_sym]))
                    _sector_val = sum(
                        p.qty * float(close_mat[i, sym_to_idx[p.symbol]])
                        for p in positions.values()
                        if trade_syms.get(p.symbol, "不明") == _sector_sym
                        and p.symbol in sym_to_idx
                    )
                    if _sector_val / max(1.0, capital) >= _eff_sec:
                        skip_stats["sector_cap"] += 1
                        if len(_skip_detail) < 5000:
                            _skip_detail.append({
                                "date": str(date.date()) if hasattr(date, "date") else str(date),
                                "symbol": sym, "rsr": round(float(rsr_val), 1),
                                "composite_score": round(float(rank_score), 4),
                                "alpha": round(float(a_val), 4),
                                "atr_pct": _c53_atr_pct, "rank": _rank_idx,
                                "reason": "SECTOR_CAP",
                            })
                        continue  # セクター上限超過: このセクターは追加 BUY 禁止

                # ── クラスター集中キャップ（cluster_cap / bear_cluster_cap）────
                if _enable_conc_caps:
                    _next_cluster = get_cluster(sym, trade_syms, CLUSTER_MAP_DEFAULT)
                    if _next_cluster != "other":
                        _pos_weights = {
                            p.symbol: p.qty * float(close_mat[i, sym_to_idx[p.symbol]]) / max(1.0, capital)
                            for p in positions.values()
                            if p.symbol in sym_to_idx
                        }
                        _cw = calc_cluster_weight(_pos_weights, trade_syms, CLUSTER_MAP_DEFAULT)
                        _min_order_w = buy_px * lot_size / max(1.0, capital)
                        _cw_val = _cw.get(_next_cluster, 0.0)
                        if _cw_val + _min_order_w > _cls_cap_eff:
                            skip_stats["cluster_cap"] += 1
                            # bear adaptive: 通常cap(cluster_cap_base)なら通過できた場合をカウント
                            if _is_bear and _cw_val + _min_order_w <= _cluster_cap_base:
                                skip_stats["bear_adaptive"] += 1
                            if len(_skip_detail) < 5000:
                                _skip_detail.append({
                                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                                    "symbol": sym, "rsr": round(float(rsr_val), 1),
                                    "composite_score": round(float(rank_score), 4),
                                    "alpha": round(float(a_val), 4),
                                    "atr_pct": _c53_atr_pct, "rank": _rank_idx,
                                    "reason": "CLUSTER_CAP",
                                })
                            continue  # クラスター上限超過: この銘柄は追加 BUY 禁止

                # ── gross exposure チェック（2026-04-07 追加）──────────────
                if _gross_exposure_enabled:
                    _cur_gross = sum(
                        p.qty * float(close_mat[i, sym_to_idx[p.symbol]])
                        for p in positions.values()
                        if p.symbol in sym_to_idx
                    ) / max(1.0, capital)
                    _order_w = buy_px * lot_size / max(1.0, capital)
                    if _cur_gross + _order_w > _gross_cap:
                        skip_stats["gross_exposure"] += 1
                        if len(_skip_detail) < 5000:
                            _skip_detail.append({
                                "date": str(date.date()) if hasattr(date, "date") else str(date),
                                "symbol": sym, "rsr": round(float(rsr_val), 1),
                                "composite_score": round(float(rank_score), 4),
                                "alpha": round(float(a_val), 4),
                                "atr_pct": _c53_atr_pct, "rank": _rank_idx,
                                "reason": "GROSS_EXPOSURE",
                            })
                        continue  # gross exposure 上限超過: 新規 BUY 禁止

                # 個別銘柄モメンタム連動スケール（sym_regime_mat が設定されている場合）
                sym_scale = 1.0
                if sym_regime_mat is not None:
                    sv = float(sym_regime_mat[next_i, sym_to_idx[sym]])
                    if not np.isnan(sv):
                        sym_scale = sv

                if enable_atr_risk_sizing and atr_sizing_no_risk:
                    # ── Sizing Ablation E: No ATR（qty_riskを使わずqty_capのみ）──
                    _effective_alloc_cap = capital * cfg.portfolio.max_single_weight * regime_step * ext_scale * sym_scale
                    qty_cap = int(_effective_alloc_cap // (buy_px * lot_size)) * lot_size
                    qty = (int(qty_cap * cb_scale) // lot_size) * lot_size
                    if qty <= 0:
                        continue
                elif enable_atr_risk_sizing:
                    # ── PROD_FAITHFUL: ATRリスクベース・サイジング（signal_bridge.py:3510-3589）──
                    # qty_risk = risk_per_trade(capital×risk_sizing_pct) / ATR20^p → 100株丸め（0なら100株フォールバック）
                    # qty_cap  = capital×max_single_weight / (price×100) → 100株丸め
                    # qty      = min(qty_risk, qty_cap) → CB時のみ追加でcb_scaleを乗算（BT固有の安全機構、既知の差分）
                    # atr_sizing_exponent=1.0が現行式。p<1.0はATR^pに正規化基準値(_atr_ref_for_sizing)で
                    # 単位（円）を揃えたeffective_atrを用い、ATR感度を弱めたサイジング感度分析用。
                    _atr_for_size = float(atr20_mat[i, sym_idx]) if atr20_mat is not None else np.nan
                    _risk_per_trade = capital * risk_sizing_pct
                    if not np.isnan(_atr_for_size) and _atr_for_size > 0:
                        if atr_sizing_exponent != 1.0 and _atr_ref_for_sizing:
                            _effective_atr = (_atr_ref_for_sizing ** (1.0 - atr_sizing_exponent)) * (_atr_for_size ** atr_sizing_exponent)
                        else:
                            _effective_atr = _atr_for_size
                        _qty_raw = int(_risk_per_trade / _effective_atr)
                        qty_risk = (_qty_raw // lot_size) * lot_size
                        if qty_risk == 0:
                            qty_risk = lot_size
                    else:
                        n_remaining = sum(1 for _, _, _, s in buy_candidates if s not in positions)
                        effective_slots = min(open_slots, max(1, n_remaining))
                        _fallback_alloc = cash / effective_slots
                        qty_risk = int(min(_fallback_alloc, capital * cfg.portfolio.max_single_weight) // (buy_px * lot_size)) * lot_size
                    qty_risk = (int(qty_risk * cb_scale) // lot_size) * lot_size
                    _effective_alloc_cap = capital * cfg.portfolio.max_single_weight * regime_step * ext_scale * sym_scale
                    qty_cap = int(_effective_alloc_cap // (buy_px * lot_size)) * lot_size
                    qty = min(qty_risk, qty_cap)
                    if qty <= 0:
                        continue
                else:
                    if use_alpha_weight and sym in wt_map:
                        target_weight = wt_map[sym] * regime_step * ext_scale * sym_scale
                        alloc = capital * target_weight * cb_scale
                    elif sizing_mode == "equal":
                        # equal-weight: capital / eff_max_pos を固定ターゲット
                        # max_single_weight キャップは適用しない（equal sizing が意図した weight のため）
                        alloc = (capital / _eff_max_pos) * regime_step * cb_scale * ext_scale * sym_scale
                    else:
                        n_remaining = sum(1 for _, _, _, s in buy_candidates if s not in positions)
                        effective_slots = min(open_slots, max(1, n_remaining))
                        alloc = (cash / effective_slots) * regime_step * cb_scale * ext_scale * sym_scale

                    if sizing_mode != "equal":
                        alloc = min(alloc, capital * cfg.portfolio.max_single_weight)
                    # 単一銘柄キャップ（動的 or 固定 8%上限）: 動的ユニバース有効時のみ適用
                    # ただし 1 ロット以上買える場合のみ（高値株で qty=0 を防ぐ）
                    if _enable_conc_caps:
                        _dyn_sym_cap_r = MAX_SYMBOL_WEIGHT
                        if _sym_cap_df is not None:
                            _eval_date = common_dates[i]
                            if _eval_date in _sym_cap_df.index and sym in _sym_cap_df.columns:
                                _dyn_sym_cap_r = float(_sym_cap_df.loc[_eval_date, sym])
                        _sym_cap = capital * _dyn_sym_cap_r
                        if _sym_cap >= buy_px * lot_size:
                            alloc = min(alloc, _sym_cap)
                    qty = int(alloc / buy_px / lot_size) * lot_size
                    if qty <= 0:
                        # Study44: max_lot_cost_ratio 救済判定
                        _lot_cost44 = buy_px * lot_size
                        _admitted_by_ratio = False
                        if max_lot_cost_ratio is not None:
                            if _lot_cost44 <= capital * max_lot_cost_ratio and cash >= _lot_cost44:
                                # lot_cost が ratio 上限以内かつ現金残高が確保できる → 1ロット強制入場
                                qty = lot_size
                                _admitted_by_ratio = True
                                _admitted_by_ratio_count += 1
                                if len(_admitted_by_ratio_detail) < 2000:
                                    _admitted_by_ratio_detail.append({
                                        "date": str(date.date()) if hasattr(date, "date") else str(date),
                                        "symbol": sym, "lot_cost": round(_lot_cost44),
                                        "buy_px": round(buy_px, 1), "rsr": round(float(rsr_val), 1),
                                        "lot_cost_ratio": round(_lot_cost44 / max(1.0, capital), 4),
                                    })
                        if not _admitted_by_ratio:
                            # Study42: lot切り捨てによる約定不可を計測
                            _rejected_by_lot += 1
                            if len(_rejected_by_lot_detail) < 2000:
                                _rejected_by_lot_detail.append({
                                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                                    "symbol": sym, "alloc": round(alloc), "buy_px": round(buy_px, 1),
                                    "rsr": round(float(rsr_val), 1),
                                    "composite_score": round(float(rank_score), 4),
                                    "alpha": round(float(a_val), 4),
                                    "atr_pct": _c53_atr_pct,
                                    "rank": _rank_idx,
                                })
                            continue

                if cb_active and not bypass_cb:
                    _cb_scaled_entries += 1
                    _capital_suppressed += alloc * (1.0 - CB_SCALE)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                _entry_sector = trade_syms.get(sym, "不明")
                positions[sym] = Position(
                    symbol=sym,
                    sector=_entry_sector,
                    qty=qty,
                    entry_price=buy_px,
                    entry_idx=next_i,
                    alpha_score=a_val,
                    highest_close=buy_px,
                )
                if SECTOR_STRATEGY.get(_entry_sector, "fujiko") == "mean_rev":
                    meanrev_entry_count += 1
                else:
                    fujiko_entry_count += 1
                trades.append({
                    "symbol": sym, "side": "BUY", "sector": _entry_sector,
                    "entry": buy_px, "exit": None,
                    "qty": qty, "pnl": None,
                    "reason": f"alpha={a_val:.4f}/RSR={rsr_val:.1f}",
                })

        # ── Study45: Q1-Q3 Idle Cash Attribution ─────────────────────────────
        _q_days_total += 1
        if cash > 0:
            _q_idle_days += 1
            _cur_cash_pct = cash / max(1.0, capital)
            _q_idle_cash_sum += _cur_cash_pct
            # 保有ポジション中に unrealized > 1×ATR の winner があるか
            _q_has_winner = False
            for _qsym, _qpos in positions.items():
                _qidx = sym_to_idx.get(_qsym, -1)
                if _qidx < 0 or atr20_mat is None:
                    continue
                _qatr = float(atr20_mat[i, _qidx])
                _qcls = float(close_mat[i, _qidx])
                if _qatr > 0 and not np.isnan(_qatr) and not np.isnan(_qcls):
                    if _qcls > _qpos.entry_price + _qatr:
                        _q_has_winner = True
                        break
            if _q_has_winner:
                _q_winner_idle_days += 1
                _q_deployable_sum += _cur_cash_pct

        # ── Study45: Addon Expansion ──────────────────────────────────────────
        if addon_policy != "NONE" and atr20_mat is not None and positions:
            # 退場済みシンボルの addon_done をリセット
            for _ad_sym in list(_addon_done.keys()):
                if _ad_sym not in positions:
                    del _addon_done[_ad_sym]

            # Vol gate: Policy E/F は calm 日のみ実行
            _addon_vol_ok = True
            if addon_policy in ("E", "F"):
                if max_positions_ts is not None and date in max_positions_ts.index:
                    _addon_vol_ok = (int(max_positions_ts[date]) > cfg.portfolio.max_positions)

            if _addon_vol_ok and cash > 0:
                # winner を gain/ATR降順でランク付け
                _addon_cands: list[tuple] = []
                for _asym, _apos in positions.items():
                    _aidx = sym_to_idx.get(_asym, -1)
                    if _aidx < 0:
                        continue
                    _acls = float(close_mat[i, _aidx])
                    _aatr = float(atr20_mat[i, _aidx])
                    if np.isnan(_acls) or np.isnan(_aatr) or _aatr <= 0:
                        continue
                    _gain_atr = (_acls - _apos.entry_price) / _aatr
                    _addon_cands.append((_gain_atr, _asym, _aidx, _acls, _aatr))
                _addon_cands.sort(key=lambda x: -x[0])

                for _gain_atr, _asym, _aidx, _acls, _aatr in _addon_cands:
                    _stage = _addon_done.get(_asym, 0)

                    # Threshold check（Policy Fは1.5×ATR）
                    _thresh1 = addon_atr_mult if addon_policy != "F" else 1.5
                    _trig1   = _gain_atr >= _thresh1
                    _trig2   = _gain_atr >= addon_stage2_mult

                    if _stage == 0 and _trig1:
                        _next_stage = 1
                    elif _stage == 1 and addon_policy == "C" and addon_max_per_pos >= 2 and _trig2:
                        _next_stage = 2
                    else:
                        continue

                    # 実行価格 = 翌日 close（新規 BUY と同じ）
                    if next_i >= len(common_dates):
                        continue
                    _addon_px = float(close_mat[next_i, _aidx])
                    if np.isnan(_addon_px) or _addon_px <= 0:
                        continue

                    # Policy D: equity-scaled size（cash × addon_size_frac）
                    if addon_policy == "D":
                        _d_alloc  = cash * addon_size_frac
                        _addon_qty = int(_d_alloc / _addon_px / lot_size) * lot_size
                        if _addon_qty <= 0:
                            _addon_qty = lot_size
                    else:
                        _addon_qty = lot_size  # B/C/E/F: 1 lot固定

                    _addon_cost = _addon_qty * _addon_px * (1 + COST_ONE_WAY)
                    if cash < _addon_cost:
                        continue

                    # max_single_weight × 1.5 キャップ（addon専用余裕）
                    _cur_val = positions[_asym].qty * _addon_px
                    _max_val = capital * cfg.portfolio.max_single_weight * 1.5
                    if _cur_val + _addon_qty * _addon_px > _max_val:
                        continue

                    # 実行: blended entry price で Position を更新
                    cash -= _addon_cost
                    _total_qty  = positions[_asym].qty + _addon_qty
                    _blended_px = (positions[_asym].qty * positions[_asym].entry_price
                                   + _addon_qty * _addon_px) / _total_qty
                    positions[_asym].qty         = _total_qty
                    positions[_asym].entry_price = _blended_px
                    _addon_done[_asym] = _next_stage
                    _addon_count += 1
                    if len(_addon_detail) < 2000:
                        _addon_detail.append({
                            "date": str(date.date()) if hasattr(date, "date") else str(date),
                            "symbol": _asym, "stage": _next_stage,
                            "addon_qty": _addon_qty, "addon_px": round(_addon_px, 1),
                            "gain_atr": round(_gain_atr, 2),
                        })
                    trades.append({
                        "symbol": _asym, "side": "BUY_ADDON",
                        "sector": trade_syms.get(_asym, "不明"),
                        "entry": _addon_px, "exit": None,
                        "qty": _addon_qty, "pnl": None,
                        "reason": f"addon_stage={_next_stage}/gain_atr={_gain_atr:.2f}",
                    })

    n_days = len(equity_curve)
    if n_days == 0:
        return {}
    eq = pd.Series(equity_curve, index=common_dates[:n_days])
    long_notional_s = pd.Series(long_notional_list, index=common_dates[:n_days])
    if eq.empty:
        return {}
    years = n_days / 252
    cagr = (eq.iloc[-1] / capital) ** (1 / max(years, 0.01)) - 1

    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    roll_max = eq.expanding().max()
    dd_ser = (eq - roll_max) / roll_max
    max_dd = float(dd_ser.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    # ── DD推移サマリー（Phase4レポート用）────────────────────────────────
    _max_dd_date = str(dd_ser.idxmin().date()) if not dd_ser.empty else ""
    _underwater = dd_ser < -0.001
    _time_underwater_pct = round(float(_underwater.mean()) * 100, 1) if len(dd_ser) else 0.0
    # 最大DD回復までの日数（max_dd以降、初めてdd>=-0.1%に戻った日）
    _recovery_days = None
    if not dd_ser.empty:
        _peak_idx = dd_ser.index.get_loc(dd_ser.idxmin())
        _after = dd_ser.iloc[_peak_idx:]
        _rec = _after[_after >= -0.001]
        if len(_rec) > 0:
            _recovery_days = int(dd_ser.index.get_loc(_rec.index[0]) - _peak_idx)
    drawdown_summary = {
        "max_dd_date": _max_dd_date,
        "time_underwater_pct": _time_underwater_pct,
        "max_dd_recovery_days": _recovery_days,
    }

    sells = [t for t in trades if t["side"] == "SELL" and t["pnl"] is not None]
    win_rate = sum(1 for t in sells if t["pnl"] > 0) / max(1, len(sells))
    win_trades = [t for t in sells if t["pnl"] > 0]
    lose_trades = [t for t in sells if t["pnl"] <= 0]
    avg_win_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in win_trades])) if win_trades else 0.0
    avg_lose_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in lose_trades])) if lose_trades else 0.0
    r_multiple = abs(avg_win_pct / avg_lose_pct) if avg_lose_pct != 0 else float("inf")

    # ── Sortino / Profit Factor / Expectancy（Phase3 KPI拡張）──────────────
    downside_dr = dr[dr < 0]
    sortino = (
        float(dr.mean() / downside_dr.std() * np.sqrt(252))
        if len(downside_dr) > 1 and downside_dr.std() > 0 else 0.0
    )
    gross_profit = float(sum(t["pnl"] for t in win_trades))
    gross_loss = float(abs(sum(t["pnl"] for t in lose_trades)))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy_pct = win_rate * avg_win_pct + (1 - win_rate) * avg_lose_pct

    annual = {}
    for yr, grp in eq.groupby(eq.index.year):
        y_ret = float(grp.iloc[-1] / grp.iloc[0] - 1)
        annual[str(yr)] = round(y_ret * 100, 2)

    monthly = {}
    for ym, grp in eq.groupby(eq.index.to_period("M")):
        m_ret = float(grp.iloc[-1] / grp.iloc[0] - 1)
        monthly[str(ym)] = round(m_ret * 100, 2)

    # ── セクター分散（SELLトレードのセクター別件数シェア）────────────────
    sector_counts: dict[str, int] = {}
    for t in sells:
        _sec = t.get("sector", "不明")
        sector_counts[_sec] = sector_counts.get(_sec, 0) + 1
    sector_diversification = {
        sec: round(100.0 * n / max(1, len(sells)), 1) for sec, n in sector_counts.items()
    }

    import os as _os
    _dataset_version = _os.environ.get("DATA_VERSION", "").strip() or DEFAULT_DATA_VERSION or "live_yfinance"

    if verbose:
        print(f"signal_calls={signal_calls}")

    return {
        "scenario": scenario,
        "dataset_version": _dataset_version,
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
        "expectancy_pct": round(expectancy_pct, 3),
        "n_trades": len(sells),
        "n_trades_yr": round(len(sells) / max(years, 0.01), 1),
        "avg_hold_days": round(float(np.mean([t["exit_idx"] - t["entry_idx"] for t in trades if t["side"] == "SELL" and "exit_idx" in t and "entry_idx" in t])) if sells else 0, 1),
        "win_rate": round(win_rate * 100, 1),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_lose_pct": round(avg_lose_pct, 2),
        "r_multiple": round(r_multiple, 2),
        "avg_exposure": round(float(np.mean(exposure_list)) * 100, 1),
        "avg_candidates": round(float(np.mean(cand_list)), 2),
        "avg_simultaneous_holdings": round(float(np.mean(pos_list)), 2),
        "pos_series": pd.Series(pos_list, index=common_dates[:n_days]),
        "cand_series": pd.Series(cand_list, index=common_dates[:n_days]),
        "annual_returns": annual,
        "monthly_returns": monthly,
        "sector_diversification_pct": sector_diversification,
        "drawdown_summary": drawdown_summary,
        "equity_curve": eq,
        "drawdown_curve": dd_ser,
        "long_notional": long_notional_s,
        "daily_pnl": eq.diff().fillna(0),
        "exit_reason_counts": exit_reason_counts,
        "signal_calls": int(signal_calls),
        "skip_stats": skip_stats,
        "gross_cap_active_days": _gross_cap_active_days,
        "cb_trigger_count": _cb_trigger_count,
        "cb_active_days": _cb_active_days,
        "cb_scaled_entries": _cb_scaled_entries,
        "capital_suppressed": round(_capital_suppressed),
        "fujiko_entry_count": fujiko_entry_count,
        "meanrev_entry_count": meanrev_entry_count,
        "dyn_universe_excluded_count": dyn_universe_excluded_count,
        "mtf_excluded_count": mtf_excluded_count,
        "_trades": [t for t in trades if t["side"] == "SELL"],
        # ── Study41 Capital Utilization ──────────────────────────────────
        "avg_idle_cash_ratio_pct": round(_idle_cash_total / max(1, n_days) * 100, 1),
        "days_at_max_positions": _days_at_max,
        "cap_saturation_rate_pct": round(_days_at_max / max(1, n_days) * 100, 1),
        "missed_by_cap_count": _missed_by_cap,
        "_missed_cands": _missed_cands,
        # ── Study42 Lot-Constraint Archaeology ───────────────────────────────
        "rejected_by_lot_count": _rejected_by_lot,
        "_rejected_by_lot_detail": _rejected_by_lot_detail,
        # ── Study44 Lot Cost Ratio Walk-Forward ──────────────────────────────
        "admitted_by_ratio_count": _admitted_by_ratio_count,
        "_admitted_by_ratio_detail": _admitted_by_ratio_detail,
        # ── Study53 Opportunity Loss ──────────────────────────────────────────
        "_skip_detail": _skip_detail,
        # ── Study45 Q1-Q3 Attribution ─────────────────────────────────────
        "q_days_total":         _q_days_total,
        "q_idle_days":          _q_idle_days,
        "q_idle_cash_avg_pct":  round(_q_idle_cash_sum / max(1, _q_idle_days) * 100, 1),
        "q1_idle_when_winner_pct": round(_q_deployable_sum / max(1e-9, _q_idle_cash_sum) * 100, 1)
                                   if _q_idle_days > 0 else None,
        "q2_idle_days_with_winner_pct": round(_q_winner_idle_days / max(1, _q_idle_days) * 100, 1),
        "q3_deployable_idle_cash_avg_pct": round(_q_deployable_sum / max(1, _q_days_total) * 100, 1),
        # ── Study45 Addon Expansion ────────────────────────────────────────
        "addon_count":   _addon_count,
        "_addon_detail": _addon_detail,
        # ── Study48 ATR Extension instrumentation ─────────────────────────
        "atr_ext_defer_count": _atr_ext_defer_count,
        "_atr_ext_detail":     _atr_ext_detail,
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> int:
    cfg = load_strategy_config()
    p = argparse.ArgumentParser()
    p.add_argument("--start",       default=START)
    p.add_argument("--end",         default=END)
    p.add_argument("--full-history", action="store_true", help="2018-2024 のフル期間で実行する")
    p.add_argument("--full-scenarios", action="store_true", help="全シナリオを実行する")
    p.add_argument(
        "--prod-faithful", action="store_true",
        help="BASELINE + PROD_FAITHFUL（ATRトレーリング/多層RSR/ATRリスクサイジング/MTFフィルター追加再現）を実行",
    )
    args = p.parse_args()

    effective_start = args.start
    effective_end = args.end
    if not args.full_history:
        if args.end == END and DEFAULT_DATA_VERSION:
            effective_end = DEFAULT_DATA_VERSION
        if args.start == START:
            effective_start = (
                pd.Timestamp(effective_end) - pd.Timedelta(days=SMOKE_LOOKBACK_DAYS)
            ).strftime("%Y-%m-%d")

    print("=" * 70)
    print("  Composite Alpha × Regime Filter 3ステップ改善バックテスト")
    print(f"  期間: {effective_start} 〜 {effective_end}  /  初期資本: ¥{cfg.portfolio.capital:,}")
    print("=" * 70)

    # ---- 1. ユニバース読み込み ----
    # 2026-03-30: RSR42に統一（TEMPORAL24廃止）
    rsr_syms   = _load_rsr_universe(verbose=True)
    trade_syms = rsr_syms   # RSR42を取引・RSR両方のユニバースとして使用
    print(f"\n[1/6] ユニバース: RSR42統一（取引・RSR共通 {len(trade_syms)}銘柄）")

    # ---- 再現性ログ（Step1 基準確認用） ----
    import os as _os
    _dataset_version = _os.environ.get("DATA_VERSION", "").strip() or DEFAULT_DATA_VERSION or "live_yfinance"
    _dataset_hash    = "unknown"
    _meta_path       = BACKTEST_DATASET_DIR / _dataset_version / "_meta.json"
    if _meta_path.exists():
        import json as _json
        with open(_meta_path, encoding="utf-8") as _f:
            _dataset_hash = _json.load(_f).get("snapshot_hash", "unknown")
    print(f"\nDATASET_VERSION  {_dataset_version}")
    print(f"DATASET_HASH     {_dataset_hash}")
    print(f"CAPITAL          {cfg.portfolio.capital:,}")
    print(f"UNIVERSE         {len(trade_syms)}")
    print()

    # ---- 2. 価格データ取得 ----
    all_syms = {**rsr_syms, **trade_syms}   # RSR42 trade時は重複するが問題なし
    print(f"[2/6] 価格データ取得中（{len(all_syms)}銘柄 + TOPIX ETF）...")
    _min_days = 90 if not args.full_history else 500
    universe_raw = download_universe(
        all_syms,
        start=effective_start,
        end=effective_end,
        min_days=_min_days,
        verbose=False,
    )
    topix_close  = _download_topix(effective_start, effective_end, verbose=True)
    print(f"  取得完了: {len(universe_raw)}銘柄, TOPIX={len(topix_close)}日")
    if not universe_raw:
        raise RuntimeError(
            "価格データを 1 銘柄も取得できませんでした。"
            f" DATA_VERSION={_dataset_version} / "
            f"期間={effective_start}〜{effective_end} / min_days={_min_days}"
        )

    # ---- 3. RSR 計算 ----
    print("[3/6] RSR計算中（42銘柄コンテキスト）...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df = calc_universe_rsr(rsr42_prices)
    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")

    # ---- 3b. 動的ユニバース: dyn_rsr42_bear_rs0 ----
    # Bull Top30 / Bear Top20 (rs>0) の sym_active_df を構築して run_scenario に渡す。
    # 先読み防止: build_dyn_rsr42_active 内部で月 T の選択に月 T-1 末データを使用。
    print("  [dyn] build_sym_active_df（dyn_rsr42_bear_rs0）計算中...")
    _bear_filter_cfg = cfg.risk_controls.bear_universe_filter
    _bear_exclude = (
        list(_bear_filter_cfg.excluded_sectors)
        if _bear_filter_cfg.enabled
        else None
    )
    sym_active_df = build_dyn_rsr42_active(
        universe_raw         = universe_raw,
        topix_close          = topix_close,
        rsr_df               = rsr_df,
        all_syms             = list(trade_syms.keys()),
        start                = effective_start,
        end                  = effective_end,
        bear_exclude_sectors = _bear_exclude,
        sym_sector_map       = dict(trade_syms) if _bear_exclude else None,
    )
    _active_frac = float(sym_active_df.mean().mean()) if not sym_active_df.empty else 0.0
    print(f"  [dyn] sym_active_df: {sym_active_df.shape} / 平均活性率={_active_frac:.2f}")

    # ---- 4. Composite Alpha 計算 ----
    print(f"[4/6] Composite Alpha計算中（window={COMP_ALPHA_WINDOW}日）...")
    trade_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms if sym in universe_raw
    }
    alpha_df = calc_composite_alpha_matrix(trade_prices, window=COMP_ALPHA_WINDOW)
    # 先読み防止: shift(1) して前日データを当日判定に使用
    alpha_df = alpha_df.shift(1)
    print(f"  Alpha matrix: {alpha_df.shape[1]}銘柄 × {alpha_df.shape[0]}日")

    # ---- レジームフィルター計算 ----
    regime_df = _calc_regime(topix_close)
    risk_off_days = int(regime_df["risk_off"].sum())
    crash_days    = int(regime_df["crash"].sum())
    total_days    = len(regime_df)
    print(f"  レジーム: risk_off={risk_off_days}日({risk_off_days/total_days:.0%}), "
          f"crash={crash_days}日({crash_days/total_days:.0%})")

    # ---- 5. STEP4/5/6用テクニカル指標を事前計算 ----
    print("[5/6] テクニカル指標事前計算中（STEP4/5/6用）...")
    tech_matrices = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth_series = _calc_breadth(universe_raw)   # 2026-03-31: price>MA200方式に修正
    print(f"  ATR20: {tech_matrices['atr20'].shape[1]}銘柄  "
          f"High200: {tech_matrices['high200'].shape[1]}銘柄  "
          f"Breadth中央値: {breadth_series.median():.2f}")

    if args.prod_faithful:
        scenario_order = ("BASELINE", "PROD_FAITHFUL")
    elif args.full_scenarios:
        scenario_order = ("BASELINE", "STEP1", "STEP2", "STEP3", "STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B")
    else:
        scenario_order = ("BASELINE",)

    # ---- 6. 7シナリオ実行 ----
    print(f"[6/6] {len(scenario_order)}シナリオ実行中...\n")

    # STEP6バリアント: breadth閾値の感度テスト
    _breadth_params = {
        "STEP6":  (BREADTH_STOP,  BREADTH_REDUCE),   # 0.25 / 0.15（デフォルト）
        "STEP6A": (0.30,          0.20),              # 0.30 / 0.20
        "STEP6B": (0.35,          0.25),              # 0.35 / 0.25
    }

    # ---- risk_controls feature flags（strategy.yaml から読み込み）----
    _rc = cfg.risk_controls
    _shock_exit_mode = _rc.shock_exit_mode    # "full_exit" | "partial_50" | "composite"
    _regime_sizing   = _rc.regime_sizing      # "none" | "regime_2" | "regime_4"
    _bear_scale      = _rc.bear_scale         # TOPIX MA200下のサイズ係数

    # regime_scale_series 生成（none 以外の場合）
    _regime_scale_series: "pd.Series | None" = None
    if _regime_sizing != "none" and topix_close is not None and not topix_close.empty:
        _ma200      = topix_close.rolling(200, min_periods=1).mean()
        _vol20      = topix_close.pct_change().rolling(20, min_periods=10).std()
        _vol_median = float(_vol20.dropna().median())
        _above      = topix_close >= _ma200
        _hi_vol     = _vol20 >= _vol_median
        _regime_scale_series = pd.Series(index=topix_close.index, dtype=float)
        _regime_scale_series[_above & ~_hi_vol] = 1.0               # 上昇安定
        _regime_scale_series[_above &  _hi_vol] = (0.5 if _regime_sizing == "regime_4" else 1.0)  # 上昇高ボラ
        _regime_scale_series[~_above]            = _bear_scale       # 下落相場
        print(f"  risk_controls: shock={_shock_exit_mode}  regime={_regime_sizing}  bear_scale={_bear_scale}")

    results = {}
    for scenario in scenario_order:
        print(f"  ─── {scenario} ───────────────────────────")
        b_stop, b_reduce = _breadth_params.get(scenario, (BREADTH_STOP, BREADTH_REDUCE))
        _is_prod_faithful = (scenario == "PROD_FAITHFUL")
        res = run_scenario(
            scenario             = scenario,
            universe_raw         = universe_raw,
            rsr_df               = rsr_df,
            alpha_df             = alpha_df,
            regime_df            = regime_df,
            trade_syms           = trade_syms,
            rsr_syms             = rsr_syms,
            cfg                  = cfg,
            start                = effective_start,
            end                  = effective_end,
            verbose              = False,
            tech_matrices        = tech_matrices,
            breadth_series       = breadth_series,
            breadth_stop         = b_stop,
            breadth_reduce       = b_reduce,
            capital              = cfg.portfolio.capital,
            min_hold             = cfg.risk.min_hold_days,
            topix_close          = topix_close,
            market_shock_mode    = _shock_exit_mode,
            regime_scale_series  = _regime_scale_series,
            sym_active_df        = sym_active_df,   # dyn_rsr42_bear_rs0
            enable_atr_trailing_prod = _is_prod_faithful,
            enable_multilayer_rsr    = _is_prod_faithful,
            enable_atr_risk_sizing   = _is_prod_faithful,
            enable_mtf_filter        = _is_prod_faithful,
        )
        results[scenario] = res

        print(f"    CAGR={res['cagr']:+.1f}%  Sharpe={res['sharpe']:.3f}  "
              f"MaxDD={res['max_dd']:.1f}%  Calmar={res['calmar']:.3f}")
        print(f"    取引={res['n_trades']}回({res['n_trades_yr']:.0f}/年)  "
              f"勝率={res['win_rate']:.1f}%  R倍率={res['r_multiple']:.2f}x")
        print(f"    avgExp={res['avg_exposure']:.1f}%  "
              f"avgWin={res['avg_win_pct']:+.2f}%  avgLose={res['avg_lose_pct']:+.2f}%")

    # ---- 集計表示 ----
    all_scenarios = scenario_order
    col_w = 9

    print("\n" + "=" * 80)
    print(f"  改善ステップ比較サマリー（{len(all_scenarios)}シナリオ）")
    print("=" * 80)
    hdr = f"  {'指標':<20}" + "".join(f" {sc:>{col_w}}" for sc in all_scenarios)
    print(hdr)
    print(f"  {'-'*20}" + f" {'-'*col_w}" * len(all_scenarios))

    metrics = [
        ("CAGR (%)",   "cagr",         "{:+.1f}"),
        ("Sharpe",     "sharpe",        "{:.3f}"),
        ("MaxDD (%)",  "max_dd",        "{:.1f}"),
        ("Calmar",     "calmar",        "{:.3f}"),
        ("勝率 (%)",   "win_rate",      "{:.1f}"),
        ("R倍率",      "r_multiple",    "{:.2f}"),
        ("avgExp (%)", "avg_exposure",  "{:.1f}"),
        ("取引数/年",  "n_trades_yr",   "{:.0f}"),
        ("avgHoldings", "avg_simultaneous_holdings", "{:.2f}"),
    ]
    for label, key, fmt in metrics:
        row = f"  {label:<20}"
        for sc in all_scenarios:
            v = results[sc].get(key, 0)
            row += f" {fmt.format(v):>{col_w}}"
        print(row)

    print("\n  ── 年次リターン ───────────────────────────────────────────────────")
    all_years = sorted(set(yr for r in results.values()
                           for yr in r.get("annual_returns", {})))
    print(f"  {'年':<6}" + "".join(f" {sc:>{col_w}}" for sc in all_scenarios))
    for yr in all_years:
        row = f"  {yr:<6}"
        for sc in all_scenarios:
            v = results[sc].get("annual_returns", {}).get(str(yr), 0.0)
            row += f" {v:>+8.1f}%"
        print(row)

    # ---- 判定 ----
    print("\n  ── 判定 ──────────────────────────────────────────────────────────")
    baseline_sharpe = results["BASELINE"]["sharpe"]
    for sc in all_scenarios:
        if sc == "BASELINE":
            continue
        diff_sharpe = results[sc]["sharpe"] - baseline_sharpe
        diff_cagr   = results[sc]["cagr"]   - results["BASELINE"]["cagr"]
        diff_dd     = results[sc]["max_dd"]  - results["BASELINE"]["max_dd"]
        verdict = "✅ 採用推奨" if diff_sharpe > 0.05 and results[sc]["max_dd"] > -20 else \
                  "△ 条件付き" if diff_sharpe > 0 else "❌ 逆効果"
        print(f"  {sc}: CAGR{diff_cagr:+.1f}pp  Sharpe{diff_sharpe:+.3f}  "
              f"MaxDD{diff_dd:+.1f}pp  → {verdict}")

    # ---- 保存 ----
    mode_tag = (
        "rsr42_prod_faithful" if args.prod_faithful
        else "rsr42_full" if args.full_scenarios
        else "rsr42_smoke"
    )
    save_path = OUTPUT_DIR / f"composite_alpha_bt_{mode_tag}_{date_type.today()}.json"
    _series_keys = {"equity_curve", "drawdown_curve", "long_notional", "daily_pnl"}
    save_data = {}
    for sc, res in results.items():
        d = {k: v for k, v in res.items() if k not in _series_keys}
        save_data[sc] = d
    save_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {save_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
