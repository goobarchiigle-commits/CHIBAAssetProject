"""
backtest/fujiko_strategy.py
フジコ投資法 ver.4.0 — Python バックテスト実装

【戦略の構造】
  銘柄選定: SEPA 8条件（6条件以上でエース、8条件でキング）
  エントリー: RSRモメンタムがプラスかつ上昇 + タートルズS1ブレイクアウト確認
  エグジット: RSRモメンタムがマイナスかつ下降 または タートルズS1下限割れ

【先読みリーク防止（CLAUDE.md ルール1）】
  engine.py から渡される data は prices.iloc[:i+1] に切り取り済み。
  本クラスでは iloc[-1]（現在足）と過去データのみ参照する。

【参考元】
  株おじさん note「シン・フジコ投資法（ver.4.0）」
  https://note.com/kabu_ojisan/n/nd7688198c814
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.backtest.strategy import BaseStrategy
from src.backtest.rsr import (
    calc_universe_rsr,
    calc_rsr_vs_benchmark,
    calc_rsr_momentum,
    calc_sepa,
    calc_starc_band,
)
from src.strategy.filters import (
    market_regime_filter,
    volatility_filter,
    volume_filter,
)


# ------------------------------------------------------------------ #
# フジコ戦略（単一銘柄 / portfolio_engine 対応）
# ------------------------------------------------------------------ #
class FujikoStrategy(BaseStrategy):
    """
    フジコ投資法 ver.4.0 戦略クラス

    Args:
        rsr_series:      事前計算済みRSR時系列（ユニバース内ランク）
                         None の場合はベンチマーク比較を使用
        benchmark_prices: ベンチマーク価格（rsr_series=None 時に使用）
        min_sepa:        エントリー最低SEPA条件数（デフォルト6=エース以上）
        min_rsr:         エントリー最低RSR
        mom_period:      RSRモメンタム計算期間（デフォルト21日）
        turtle_entry:    タートルズS1エントリー期間（デフォルト20日）
        turtle_exit:     タートルズS1エグジット期間
        use_turtle_entry: Trueにするとブレイクアウトもエントリー条件に加える
    """

    def __init__(
        self,
        min_rsr:          float,
        turtle_exit:      int,
        rsr_series:       pd.Series  = None,
        benchmark_prices: pd.Series  = None,
        min_sepa:         int        = 6,
        mom_period:       int        = 21,
        turtle_entry:     int        = 20,
        use_turtle_entry: bool       = True,
        enable_filters:   bool       = False,
        volatility_threshold: float  = 1.15,
        volume_multiplier:   float   = 1.3,
        enable_volatility_filter: bool = True,
        enable_atr_filter: bool = True,
        enable_volume_filter: bool = True,
        enable_market_regime_filter: bool = True,
        topix_series:     pd.Series | None = None,
        filter_frame:     pd.DataFrame | None = None,
    ) -> None:
        self.rsr_series       = rsr_series
        self.benchmark_prices = benchmark_prices
        self.min_sepa         = min_sepa
        self.min_rsr          = min_rsr
        self.mom_period       = mom_period
        self.turtle_entry     = turtle_entry
        self.turtle_exit      = turtle_exit
        self.use_turtle_entry = use_turtle_entry
        self.enable_filters   = enable_filters
        self.volatility_threshold = volatility_threshold
        self.volume_multiplier = volume_multiplier
        self.enable_volatility_filter = enable_volatility_filter
        self.enable_atr_filter = enable_atr_filter
        self.enable_volume_filter = enable_volume_filter
        self.enable_market_regime_filter = enable_market_regime_filter
        self.topix_series      = topix_series
        self.filter_frame      = filter_frame
        self.signal_calls      = 0
        # PROD_FAITHFUL: precompute_signals実行後に設定されるExit内訳サブマスク
        self.exit_momentum_mask_full: np.ndarray | None = None
        self.exit_turtle_mask_full:   np.ndarray | None = None

    @property
    def name(self) -> str:
        base = (
            f"Fujiko(sepa>={self.min_sepa}, rsr>={self.min_rsr:.0f}, "
            f"turtle={self.turtle_entry}/{self.turtle_exit})"
        )
        return f"{base}[filters]" if self.enable_filters else base

    def _build_filter_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.filter_frame is not None:
            return self.filter_frame.reindex(data.index)

        filt = pd.DataFrame(index=data.index)

        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        close_prev = close.shift(1)
        tr = pd.concat(
            [high - low, (high - close_prev).abs(), (low - close_prev).abs()],
            axis=1,
        ).max(axis=1)
        filt["atr_5"] = tr.rolling(5, min_periods=5).mean()
        filt["atr_20"] = tr.rolling(20, min_periods=20).mean()
        filt["atr20_med90"] = filt["atr_20"].rolling(90, min_periods=45).median().shift(1)

        volume_col = "volume" if "volume" in data.columns else "Volume"
        if volume_col in data.columns:
            filt["volume"] = pd.to_numeric(data[volume_col], errors="coerce")
            filt["volume_ma20"] = filt["volume"].rolling(20, min_periods=20).mean()
        else:
            filt["volume"] = np.nan
            filt["volume_ma20"] = np.nan

        if self.topix_series is not None:
            topix = self.topix_series.reindex(data.index).ffill()
            filt["topix_close"] = topix
            filt["topix_ma50"] = topix.rolling(50, min_periods=50).mean()
        else:
            filt["topix_close"] = np.nan
            filt["topix_ma50"] = np.nan

        return filt

    # ------------------------------------------------------------------ #
    # RSR取得（事前計算済み or リアルタイム計算）
    # ------------------------------------------------------------------ #
    def _get_rsr(self, data: pd.DataFrame) -> pd.Series:
        if self.rsr_series is not None:
            # ユニバース内RSR（事前計算済み）を使用
            aligned = self.rsr_series.reindex(data.index)
            return aligned.ffill()
        if self.benchmark_prices is not None:
            bench = self.benchmark_prices.reindex(data.index).ffill()
            return calc_rsr_vs_benchmark(data["Close"], bench)
        # フォールバック: 全て50（ベンチマーク比較なし）
        return pd.Series(50.0, index=data.index)

    @staticmethod
    def _rolling_mean_array(arr: np.ndarray, window: int) -> np.ndarray:
        arr64 = np.asarray(arr, dtype=np.float64)
        n = arr64.shape[0]
        out = np.full(n, np.nan, dtype=np.float32)
        if window <= 0 or n < window:
            return out

        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(arr64, out=csum[1:])
        sums = csum[window:] - csum[:-window]
        out[window - 1:] = (sums / window).astype(np.float32, copy=False)
        return out

    @staticmethod
    def _rolling_prev_extreme_array(arr: np.ndarray, window: int, is_max: bool) -> np.ndarray:
        arr32 = np.asarray(arr, dtype=np.float32)
        n = arr32.shape[0]
        out = np.full(n, np.nan, dtype=np.float32)
        if window <= 0 or n <= window:
            return out

        windows = sliding_window_view(arr32[:-1], window_shape=window)
        vals = windows.max(axis=1) if is_max else windows.min(axis=1)
        out[window:] = vals.astype(np.float32, copy=False)
        return out

    @staticmethod
    def _rolling_extreme_min_periods_array(
        arr: np.ndarray,
        window: int,
        min_periods: int,
        is_max: bool,
    ) -> np.ndarray:
        arr32 = np.asarray(arr, dtype=np.float32)
        n = arr32.shape[0]
        out = np.full(n, np.nan, dtype=np.float32)
        if n == 0 or window <= 0 or min_periods <= 0:
            return out

        prefix_stop = min(n, window - 1)
        if prefix_stop > 0:
            running = arr32[0]
            for i in range(prefix_stop):
                val = arr32[i]
                running = max(running, val) if is_max else min(running, val)
                if i >= (min_periods - 1):
                    out[i] = running

        if n >= window:
            windows = sliding_window_view(arr32, window_shape=window)
            vals = windows.max(axis=1) if is_max else windows.min(axis=1)
            out[window - 1:] = vals.astype(np.float32, copy=False)

        return out

    def _slice_series_to_array(
        self,
        series: pd.Series | None,
        target_index: pd.Index,
        *,
        fill_method: str | None = None,
        default: float | None = None,
    ) -> np.ndarray:
        if series is None:
            if default is None:
                return np.full(len(target_index), np.nan, dtype=np.float32)
            return np.full(len(target_index), np.float32(default), dtype=np.float32)

        src_index = series.index
        if len(target_index) > 0 and src_index.is_monotonic_increasing:
            start = src_index.searchsorted(target_index[0], side="left")
            stop = start + len(target_index)
            if stop <= len(series):
                sliced = series.iloc[start:stop]
                if sliced.index.equals(target_index):
                    if fill_method == "ffill":
                        sliced = sliced.ffill()
                    return sliced.to_numpy(dtype=np.float32, copy=False)

        aligned = series.reindex(target_index)
        if fill_method == "ffill":
            aligned = aligned.ffill()
        return aligned.to_numpy(dtype=np.float32, copy=False)

    def _calc_sepa_score_array(self, close_arr: np.ndarray, rsr_arr: np.ndarray) -> np.ndarray:
        ma50_arr = self._rolling_mean_array(close_arr, 50)
        ma150_arr = self._rolling_mean_array(close_arr, 150)
        ma200_arr = self._rolling_mean_array(close_arr, 200)
        high52_arr = self._rolling_extreme_min_periods_array(close_arr, 252, 126, is_max=True)
        low52_arr = self._rolling_extreme_min_periods_array(close_arr, 252, 126, is_max=False)

        ma200_prev21 = np.roll(ma200_arr, 21)
        ma200_prev21[:21] = np.nan
        ma50_prev1 = np.roll(ma50_arr, 1)
        ma50_prev1[0] = np.nan

        score = np.zeros(len(close_arr), dtype=np.int16)
        score += ((close_arr > ma150_arr) & (close_arr > ma200_arr)).astype(np.int16)
        score += (ma150_arr > ma200_arr).astype(np.int16)
        score += (ma200_arr > ma200_prev21).astype(np.int16)
        score += (ma50_arr > ma50_prev1).astype(np.int16)
        score += (close_arr > ma50_arr).astype(np.int16)
        score += (close_arr >= (low52_arr * np.float32(1.30))).astype(np.int16)
        score += (close_arr >= (high52_arr * np.float32(0.75))).astype(np.int16)
        score += (rsr_arr >= np.float32(70.0)).astype(np.int16)
        return score

    def precompute_signals(self, data: pd.DataFrame) -> pd.Series:
        n = len(data)
        min_bars = 252 + self.mom_period + 2
        signals_arr = np.zeros(n, dtype=np.int8)
        if n < min_bars:
            self.signal_calls += n
            return pd.Series(signals_arr, index=data.index, dtype=np.int8)

        close_arr = data["Close"].to_numpy(dtype=np.float32, copy=False)
        rsr_arr = self._slice_series_to_array(self.rsr_series, data.index, fill_method="ffill", default=50.0)
        if self.rsr_series is None and self.benchmark_prices is not None:
            rsr_arr = self._get_rsr(data).to_numpy(dtype=np.float32, copy=False)
        mom_arr = (rsr_arr - np.roll(rsr_arr, self.mom_period)).astype(np.float32, copy=False)
        mom_arr[:self.mom_period] = np.nan
        mom_prev_arr = np.roll(mom_arr, 1)
        mom_prev_arr[0] = np.nan

        sepa_score_arr = self._calc_sepa_score_array(close_arr, rsr_arr)
        turtle_high_arr = self._rolling_prev_extreme_array(close_arr, self.turtle_entry, is_max=True)
        turtle_low_arr = self._rolling_prev_extreme_array(close_arr, self.turtle_exit, is_max=False)

        idx_arr = np.arange(n)
        valid = (
            np.isfinite(rsr_arr)
            & np.isfinite(mom_arr)
            & np.isfinite(mom_prev_arr)
            & (idx_arr >= (min_bars - 1))
        )

        momentum_exit_mask = valid & (mom_arr < 0) & (mom_arr < mom_prev_arr)
        turtle_exit_mask = valid & (close_arr < turtle_low_arr) & ~momentum_exit_mask
        exit_mask = momentum_exit_mask | turtle_exit_mask
        # PROD_FAITHFUL: Exit内訳集計用（momentum優先、turtleはmomentum不成立時のみ集計）
        self.exit_momentum_mask_full = momentum_exit_mask
        self.exit_turtle_mask_full = turtle_exit_mask
        entry_mask = valid.copy()
        entry_mask &= sepa_score_arr >= self.min_sepa
        entry_mask &= rsr_arr >= self.min_rsr
        entry_mask &= (mom_arr > 0) & (mom_arr > mom_prev_arr)
        if self.use_turtle_entry:
            entry_mask &= close_arr > turtle_high_arr

        if self.enable_filters:
            filt = self._build_filter_frame(data)
            atr5_arr = filt["atr_5"].to_numpy(dtype=np.float32, copy=False)
            atr20_arr = filt["atr_20"].to_numpy(dtype=np.float32, copy=False)
            atr20_med90_arr = filt["atr20_med90"].to_numpy(dtype=np.float32, copy=False)
            vol_arr = filt["volume"].to_numpy(dtype=np.float32, copy=False)
            vma20_arr = filt["volume_ma20"].to_numpy(dtype=np.float32, copy=False)
            vol_ok = (atr5_arr / atr20_arr) > self.volatility_threshold
            atr_ok = atr20_arr > atr20_med90_arr
            volume_ok = vol_arr > (vma20_arr * np.float32(self.volume_multiplier))
            if {"topix_close", "topix_ma50"}.issubset(filt.columns):
                topix_close_arr = filt["topix_close"].to_numpy(dtype=np.float32, copy=False)
                topix_ma50_arr = filt["topix_ma50"].to_numpy(dtype=np.float32, copy=False)
                market_has = np.isfinite(topix_close_arr) & np.isfinite(topix_ma50_arr)
                market_ok = (~market_has) | (topix_close_arr > topix_ma50_arr)
            else:
                market_ok = np.ones(n, dtype=bool)
            if not self.enable_market_regime_filter:
                market_ok = np.ones(n, dtype=bool)
            if self.enable_volatility_filter:
                entry_mask &= np.nan_to_num(vol_ok, nan=False)
            if self.enable_atr_filter:
                entry_mask &= np.nan_to_num(atr_ok, nan=False)
            if self.enable_volume_filter:
                entry_mask &= np.nan_to_num(volume_ok, nan=False)
            entry_mask &= np.nan_to_num(market_ok, nan=True)

        signals_arr[exit_mask] = -1
        signals_arr[entry_mask & ~exit_mask] = 1
        self.signal_calls += n
        return pd.Series(signals_arr, index=data.index, dtype=np.int8)

    # ------------------------------------------------------------------ #
    # シグナル生成
    # ------------------------------------------------------------------ #
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        フジコ投資法シグナルを生成する。

        エントリー（+1）:
          SEPA条件 >= min_sepa
          かつ RSR >= min_rsr
          かつ RSRモメンタム > 0 かつ 上昇中
          かつ（use_turtle_entry=True の場合）20日高値ブレイクアウト

        エグジット（-1）:
          RSRモメンタム < 0 かつ 下降中（フジコ法の「必ず売り」）
          または タートルズS1 10日安値割れ

        Returns:
            +1: 買い / -1: 売り / 0: 何もしない
        """
        self.signal_calls += 1
        min_bars = 252 + self.mom_period + 2
        if len(data) < min_bars:
            return 0

        close = data["Close"]

        # --- RSR + RSRモメンタム ---
        rsr = self._get_rsr(data)
        mom = calc_rsr_momentum(rsr, self.mom_period)

        rsr_now  = rsr.iloc[-1]
        mom_now  = mom.iloc[-1]
        mom_prev = mom.iloc[-2]

        if pd.isna(rsr_now) or pd.isna(mom_now) or pd.isna(mom_prev):
            return 0

        # --- SEPA 8条件 ---
        sepa = calc_sepa(data, rsr)
        sepa_score = sepa["sepa_score"].iloc[-1]

        # --- タートルズS1 ---
        turtle_high = close.iloc[-(self.turtle_entry + 1):-1].max()  # 前日まで
        turtle_low  = close.iloc[-(self.turtle_exit  + 1):-1].min()
        price_now   = close.iloc[-1]

        # ==================== エグジット判定 ====================
        # 優先度: エグジット > エントリー

        # フジコ法の「必ず売り」: RSRモメンタム負かつ下降
        if mom_now < 0 and mom_now < mom_prev:
            return -1

        # タートルズS1 エグジット: 10日安値割れ
        if price_now < turtle_low:
            return -1

        # ==================== エントリー判定 ====================
        # 1. SEPA条件数チェック（銘柄選定）
        if sepa_score < self.min_sepa:
            return 0

        # 2. RSRチェック（相対強度）
        if rsr_now < self.min_rsr:
            return 0

        # 3. RSRモメンタム: プラスかつ上昇
        if not (mom_now > 0 and mom_now > mom_prev):
            return 0

        # 4. タートルズS1 エントリー確認（オプション）
        if self.use_turtle_entry and price_now <= turtle_high:
            return 0

        if self.enable_filters:
            filt = self._build_filter_frame(data)
            if self.enable_volatility_filter and not volatility_filter(filt, threshold=self.volatility_threshold):
                return 0
            latest = filt.iloc[-1]
            if self.enable_atr_filter:
                atr20 = latest.get("atr_20")
                atr20_med90 = latest.get("atr20_med90")
                if pd.isna(atr20) or pd.isna(atr20_med90) or float(atr20) <= float(atr20_med90):
                    return 0
            if self.enable_volume_filter and not volume_filter(filt, multiplier=self.volume_multiplier):
                return 0
            has_market_data = (
                {"topix_close", "topix_ma50"}.issubset(filt.columns)
                and not filt[["topix_close", "topix_ma50"]].iloc[-1].isna().any()
            )
            if self.enable_market_regime_filter and has_market_data and not market_regime_filter(filt):
                return 0

        return 1


# ------------------------------------------------------------------ #
# バックテスト実行スクリプト（単体実行用）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    import matplotlib
    matplotlib.use("Agg")

    import yfinance as yf
    from src.backtest.engine import TradeCost

    # ---------------------------------------------------------------- #
    # データ取得
    # ---------------------------------------------------------------- #
    UNIVERSE_DEF = {
        "7203.T": "輸送機器",
        "6758.T": "電機",
        "9984.T": "情報通信",
        "8306.T": "銀行",
        "4502.T": "医薬品",
        "3382.T": "小売",
        "6861.T": "電機精密",   # キーエンス
        "4063.T": "化学",       # 信越化学
        "8035.T": "電機精密",   # 東京エレクトロン
        "9432.T": "情報通信",   # NTT
    }
    BENCHMARK_TICKER = "1306.T"  # TOPIXのETF（ベンチマーク代替）

    print("データ取得中...")
    universe_raw = {}
    for sym, sector in UNIVERSE_DEF.items():
        df = yf.download(sym, start="2018-01-01", end="2024-12-31", progress=False)
        if not df.empty:
            df = df.droplevel(1, axis=1)
            universe_raw[sym] = {"df": df, "sector": sector}
            print(f"  {sym} ({sector}): {len(df)} 日")

    bench_df = yf.download(BENCHMARK_TICKER, start="2018-01-01", end="2024-12-31", progress=False)
    bench_df = bench_df.droplevel(1, axis=1)
    print(f"  ベンチマーク(TOPIX ETF): {len(bench_df)} 日")

    # ---------------------------------------------------------------- #
    # RSR事前計算（ユニバース内ランク）
    # ---------------------------------------------------------------- #
    print("\nRSR計算中...")
    universe_prices = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
    rsr_universe    = calc_universe_rsr(universe_prices)

    # ---------------------------------------------------------------- #
    # フジコ戦略（エース以上: SEPA >= 6）
    # ---------------------------------------------------------------- #
    print("\n--- フジコ戦略（エース: SEPA >= 6）---")
    universe_fujiko = {}
    for sym, info in universe_raw.items():
        strat = FujikoStrategy(
            rsr_series       = rsr_universe[sym] if sym in rsr_universe else None,
            benchmark_prices = bench_df["Close"],
            min_sepa         = 6,
            min_rsr          = 65.0,    # ユニバースが小さいので65に緩和
            mom_period       = 21,
            turtle_entry     = 20,
            turtle_exit      = 10,
            use_turtle_entry = True,
        )
        universe_fujiko[sym] = {"df": info["df"], "sector": info["sector"]}

    # 全銘柄に同じRSRを渡すため、PortfolioEngineをカスタム呼び出し
    # （各銘柄のRSRを個別に設定）
    results_by_symbol = {}
    for sym, info in universe_raw.items():
        strat  = FujikoStrategy(
            rsr_series       = rsr_universe[sym] if sym in rsr_universe.columns else None,
            benchmark_prices = bench_df["Close"],
            min_sepa         = 6,
            min_rsr          = 65.0,
            mom_period       = 21,
            turtle_entry     = 20,
            turtle_exit      = 10,
            use_turtle_entry = True,
        )
        # 単一銘柄バックテスト
        from src.backtest.engine import BacktestEngine
        eng = BacktestEngine(
            prices   = info["df"],
            strategy = strat,
            capital  = 2_000_000,
            cost     = TradeCost(),
            symbol   = sym,
        )
        results_by_symbol[sym] = eng.run()

    # ---------------------------------------------------------------- #
    # 結果表示
    # ---------------------------------------------------------------- #
    print("\n=== フジコ戦略 単一銘柄バックテスト比較 ===")
    print(f"{'銘柄':<10} {'リターン':>8} {'シャープ':>8} {'最大DD':>8} {'取引':>6} {'勝率':>6}")
    print("-" * 52)
    for sym, r in results_by_symbol.items():
        sector = universe_raw[sym]["sector"]
        print(
            f"{sym:<10} {r.total_return*100:>+7.1f}%"
            f" {r.sharpe_ratio:>8.3f}"
            f" {r.max_drawdown*100:>+7.1f}%"
            f" {r.num_trades:>5}回"
            f" {r.win_rate*100:>5.1f}%"
        )

    # ---------------------------------------------------------------- #
    # MAバンド戦略との比較グラフ
    # ---------------------------------------------------------------- #
    print("\nグラフ生成中...")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import platform

    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "MS Gothic"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 全銘柄の資産推移を重ねて表示
    ax = axes[0]
    colors = plt.cm.tab10.colors
    for i, (sym, r) in enumerate(results_by_symbol.items()):
        eq = r.equity_curve / 10_000
        ax.plot(eq.index, eq.values, linewidth=1.2, color=colors[i],
                label=f"{sym} {r.total_return*100:+.1f}%")
    ax.axhline(200, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("資産（万円）")
    ax.set_title("フジコ戦略 銘柄別バックテスト（2018-2024 / 初期資本200万円）")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:.0f}万"))

    # RSR推移（代表銘柄）
    ax = axes[1]
    rep_sym = "7203.T"
    rsr_s   = rsr_universe[rep_sym]
    mom_s   = calc_rsr_momentum(rsr_s)
    ax2     = ax.twinx()

    ax.plot(rsr_s.index, rsr_s.values, color="royalblue", linewidth=1.0, label="RSR")
    ax.axhline(70, color="red",   linestyle="--", linewidth=0.8, label="RSR=70")
    ax.axhline(50, color="gray",  linestyle="--", linewidth=0.6)
    ax.set_ylabel("RSR")
    ax.set_ylim(0, 100)

    mom_pos = mom_s.clip(lower=0)
    mom_neg = mom_s.clip(upper=0)
    ax2.bar(mom_s.index, mom_pos.values, color="deeppink",  alpha=0.6, width=1, label="RSRモメンタム(+)")
    ax2.bar(mom_s.index, mom_neg.values, color="steelblue", alpha=0.6, width=1, label="RSRモメンタム(-)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("RSRモメンタム")

    ax.set_title(f"RSR + RSRモメンタム  ({rep_sym})")
    ax.legend(loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("data/fujiko_backtest.png", dpi=150, bbox_inches="tight")
    print("グラフ保存: data/fujiko_backtest.png")
