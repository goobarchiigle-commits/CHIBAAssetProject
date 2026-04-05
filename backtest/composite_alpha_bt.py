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
from src.strategy.universe import build_dyn_rsr42_active


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
) -> dict:
    """
    汎用バックテストループ。

    scenario別の挙動:
      BASELINE : RSR降順ランキング / 均等ウェイト / レジームなし
      STEP1    : alpha複合スコアランキング / 均等ウェイト / レジームなし
      STEP2    : alpha複合スコアランキング / 均等ウェイト / 2段階レジームフィルター
      STEP3    : alpha複合スコアランキング / αウェイト / 2段階レジームフィルター
    """
    use_alpha_rank      = (scenario != "BASELINE")
    use_regime          = (scenario in ("STEP2", "STEP3"))
    use_alpha_weight    = (scenario == "STEP3")
    use_breakout_filter = (scenario in ("STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_trail_exit      = (scenario in ("STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_breadth_regime  = (scenario in ("STEP6", "STEP6A", "STEP6B"))
    capital = float(cfg.portfolio.capital if capital is None else capital)
    rsr_exit_threshold = cfg.fujiko.min_rsr if rsr_exit_threshold is None else rsr_exit_threshold
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

    common_dates: pd.DatetimeIndex | None = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
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
    sym_to_idx = {sym: idx for idx, sym in enumerate(active_syms)}
    signal_calls = 0

    for sym_idx, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue

        open_vals = df_src["Open"].to_numpy(dtype=np.float32, copy=False)
        close_vals = df_src["Close"].to_numpy(dtype=np.float32, copy=False)
        high_vals = df_src["High"].to_numpy(dtype=np.float32, copy=False)
        open_mat[:, sym_idx] = open_vals[row_idx]
        close_mat[:, sym_idx] = close_vals[row_idx]
        high_mat[:, sym_idx] = high_vals[row_idx]

        if need_fallback_frames:
            df_by_sym[sym] = df_src.take(row_idx)

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
            signal_mat[:, sym_idx] = sig_series_full.to_numpy(dtype=np.int8, copy=False)[row_idx]
            signal_ready[sym_idx] = True
            signal_calls += len(row_idx)

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
    MAX_SECTOR_WEIGHT = 0.25   # 同一セクターは総資産の25%上限
    MAX_SYMBOL_WEIGHT = 0.08   # 単一銘柄は総資産の8%上限

    cash = float(capital)
    positions: dict[str, Position] = {}
    equity_curve = []
    exposure_list = []
    pos_list = []
    cand_list = []
    trades: list[dict] = []
    exit_reason_counts: dict[str, int] = {}
    peak_equity = float(capital)
    max_hold_limit = getattr(cfg.risk, "max_hold_days", None)
    cb_active = False
    cb_days = 0
    reentry_ban: dict[str, int] = {}
    # ── サスペンション（構造的ルーザー銘柄の一時停止）──────────────
    suspended_until: dict[str, int] = {}   # sym -> 停止解除インデックス
    sym_trade_log:   dict[str, list] = {}  # sym -> [(exit_idx, pnl), ...]

    # ── Market shock detection: 1306.T 日次リターン -5%以下で翌寄り全決済 ──
    market_ret_arr = np.zeros(n_dates, dtype=np.float32)
    if topix_close is not None and not topix_close.empty:
        topix_ret = topix_close.pct_change()
        market_ret_arr = _take_series_to_array(
            topix_ret, common_dates, dtype=np.float32, fill_value=0.0
        )

    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_idx[sym]]) for sym, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        pos_list.append(len(positions))

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        if not cb_active:
            if dd <= -cfg.portfolio.max_dd_limit:
                cb_active = True
                cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False
                cb_days = 0

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

            if is_holding and max_hold_limit is not None and hold_idx > max_hold_limit:
                sell_signals.append((sym, "TIME_STOP"))
                continue

            if is_holding and rsr_val < rsr_exit_threshold and hold_idx >= min_hold:
                sell_signals.append((sym, "RSR_EXIT"))
                continue

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
                sell_signals.append((sym, "STRATEGY_EXIT"))
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
                half_qty = (pos.qty // 2 // LOT) * LOT
                qty_to_sell = half_qty if half_qty > 0 else pos.qty  # 1ロットしかない場合は全量
                proceeds = qty_to_sell * sell_px * (1 - COST_ONE_WAY)
                pnl = (sell_px - pos.entry_price) * qty_to_sell
                cash += proceeds
                trades.append({
                    "symbol": sym, "side": "SELL",
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": qty_to_sell, "pnl": pnl, "reason": reason,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                })
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                positions[sym].qty -= qty_to_sell
                if positions[sym].qty <= 0:
                    del positions[sym]
            else:
                # 全量決済（通常）
                proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
                pnl = (sell_px - pos.entry_price) * pos.qty
                cash += proceeds
                trades.append({
                    "symbol": sym, "side": "SELL",
                    "entry": pos.entry_price, "exit": sell_px,
                    "qty": pos.qty, "pnl": pnl, "reason": reason,
                    "entry_idx": pos.entry_idx, "exit_idx": i,
                })
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                del positions[sym]
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
                top_cands = buy_candidates[:cfg.portfolio.max_positions]
                wt_map = _alpha_weights([(a, s) for _, a, _, s in top_cands], max_weight=MAX_POS_WEIGHT)
            else:
                wt_map = {}

            cb_scale = CB_SCALE if cb_active else 1.0

            for rank_score, a_val, rsr_val, sym in buy_candidates:
                open_slots = cfg.portfolio.max_positions - len(positions)
                if open_slots <= 0:
                    break

                buy_px = float(open_mat[next_i, sym_to_idx[sym]])
                if buy_px <= 0:
                    continue

                # ── セクター集中キャップ（25%上限）───────────────────────
                if _enable_conc_caps:
                    _sector_sym = trade_syms.get(sym, "不明")
                    _sector_val = sum(
                        p.qty * float(close_mat[i, sym_to_idx[p.symbol]])
                        for p in positions.values()
                        if trade_syms.get(p.symbol, "不明") == _sector_sym
                        and p.symbol in sym_to_idx
                    )
                    if _sector_val / max(1.0, capital) >= MAX_SECTOR_WEIGHT:
                        continue  # セクター上限超過: このセクターは追加 BUY 禁止

                # 個別銘柄モメンタム連動スケール（sym_regime_mat が設定されている場合）
                sym_scale = 1.0
                if sym_regime_mat is not None:
                    sv = float(sym_regime_mat[next_i, sym_to_idx[sym]])
                    if not np.isnan(sv):
                        sym_scale = sv

                if use_alpha_weight and sym in wt_map:
                    target_weight = wt_map[sym] * regime_step * ext_scale * sym_scale
                    alloc = capital * target_weight * cb_scale
                else:
                    n_remaining = sum(1 for _, _, _, s in buy_candidates if s not in positions)
                    effective_slots = min(open_slots, max(1, n_remaining))
                    alloc = (cash / effective_slots) * regime_step * cb_scale * ext_scale * sym_scale

                alloc = min(alloc, capital * cfg.portfolio.max_single_weight)
                # 単一銘柄キャップ（8%上限）: 動的ユニバース有効時のみ適用
                # ただし 1 ロット以上買える場合のみ（高値株で qty=0 を防ぐ）
                if _enable_conc_caps:
                    _sym_cap = capital * MAX_SYMBOL_WEIGHT
                    if _sym_cap >= buy_px * LOT:
                        alloc = min(alloc, _sym_cap)
                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue

                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                positions[sym] = Position(
                    symbol=sym,
                    sector=trade_syms.get(sym, "不明"),
                    qty=qty,
                    entry_price=buy_px,
                    entry_idx=next_i,
                    alpha_score=a_val,
                )
                trades.append({
                    "symbol": sym, "side": "BUY",
                    "entry": buy_px, "exit": None,
                    "qty": qty, "pnl": None,
                    "reason": f"alpha={a_val:.4f}/RSR={rsr_val:.1f}",
                })

    n_days = len(equity_curve)
    if n_days == 0:
        return {}
    eq = pd.Series(equity_curve, index=common_dates[:n_days])
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

    sells = [t for t in trades if t["side"] == "SELL" and t["pnl"] is not None]
    win_rate = sum(1 for t in sells if t["pnl"] > 0) / max(1, len(sells))
    win_trades = [t for t in sells if t["pnl"] > 0]
    lose_trades = [t for t in sells if t["pnl"] <= 0]
    avg_win_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in win_trades])) if win_trades else 0.0
    avg_lose_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in lose_trades])) if lose_trades else 0.0
    r_multiple = abs(avg_win_pct / avg_lose_pct) if avg_lose_pct != 0 else float("inf")

    annual = {}
    for yr, grp in eq.groupby(eq.index.year):
        y_ret = float(grp.iloc[-1] / grp.iloc[0] - 1)
        annual[str(yr)] = round(y_ret * 100, 2)

    import os as _os
    _dataset_version = _os.environ.get("DATA_VERSION", "").strip() or DEFAULT_DATA_VERSION or "live_yfinance"

    if verbose:
        print(f"signal_calls={signal_calls}")

    return {
        "scenario": scenario,
        "dataset_version": _dataset_version,
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
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
        "annual_returns": annual,
        "equity_curve": eq,
        "exit_reason_counts": exit_reason_counts,
        "signal_calls": int(signal_calls),
        "_trades": [t for t in trades if t["side"] == "SELL"],
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
    sym_active_df = build_dyn_rsr42_active(
        universe_raw = universe_raw,
        topix_close  = topix_close,
        rsr_df       = rsr_df,
        all_syms     = list(trade_syms.keys()),
        start        = effective_start,
        end          = effective_end,
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

    scenario_order = (
        ("BASELINE", "STEP1", "STEP2", "STEP3", "STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B")
        if args.full_scenarios
        else ("BASELINE",)
    )

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
    mode_tag  = "rsr42_full" if args.full_scenarios else "rsr42_smoke"
    save_path = OUTPUT_DIR / f"composite_alpha_bt_{mode_tag}_{date_type.today()}.json"
    save_data = {}
    for sc, res in results.items():
        d = {k: v for k, v in res.items() if k != "equity_curve"}
        save_data[sc] = d
    save_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {save_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
