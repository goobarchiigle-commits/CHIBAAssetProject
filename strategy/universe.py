"""
strategy/universe.py
動的ユニバース選択ロジック — backtest / batch / live 共通モジュール

確定設定: dyn_rsr42_bear_rs0 (採用 2026-04-05)
  Bull: RSR42 Top30, score = mom_63(0.40) + rsr(0.35) + log_vol(0.25)
  Bear: RSR42 Top20, score = rs_topix(0.50) + rsr(0.30) + log_vol(0.20), rs>0 filter
  判定: TOPIX < MA200 かつ 直近60日のうち40日以上MA200下 = 持続 Bear

使用例:
  from src.strategy.universe import build_dyn_rsr42_active
  sym_active_df = build_dyn_rsr42_active(universe_raw, topix_close, rsr_df, syms, start, end)
  # → run_scenario(..., sym_active_df=sym_active_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 確定パラメータ（dyn_rsr42_bear_rs0）
# ------------------------------------------------------------------ #
MA200_PERIOD        = 200   # TOPIX レジーム判定: MA200 期間
MOM_PERIOD          = 63    # 3ヶ月モメンタム計算期間
VOL_PERIOD          = 20    # 流動性計算期間
BULL_ACTIVE_N       = 30    # Bull 時の最大選択銘柄数
BEAR_ACTIVE_N       = 20    # Bear 時の最大選択銘柄数
SUSTAINED_BEAR_DAYS = 40    # 直近 LOOKBACK_BEAR_CHECK 日のうち何日以上 MA200 下なら持続 Bear
LOOKBACK_BEAR_CHECK = 60    # 持続 Bear 判定のルックバック期間


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def _zscore(s: pd.Series) -> pd.Series:
    """NaN を無視してクロスセクション zscore を計算する。"""
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(0.0, index=s.index)
    z = (s - valid.mean()) / (valid.std() + 1e-9)
    return z.fillna(0.0)


# ------------------------------------------------------------------ #
# メイン関数
# ------------------------------------------------------------------ #
def build_sym_active_df(
    universe_raw: dict,
    topix_close: pd.Series,
    all_syms: list[str],
    active_n: int,
    start: str,
    end: str,
    rsr_df_full: "pd.DataFrame | None" = None,
    bear_n: int | None = None,
    bear_pool: "list[str] | str | None" = None,
) -> pd.DataFrame:
    """
    月次レジームアウェアスコアリングで銘柄活性 DataFrame (0/1) を構築する。

    Bull regime（TOPIX >= MA200 持続）:
        score = 0.40*mom_63 + 0.35*rsr + 0.25*log_vol
        → Top active_n 銘柄

    Bear regime（TOPIX < MA200 かつ 持続 Bear 判定）:
        score = 0.50*rs_topix + 0.30*rsr + 0.20*log_vol
        → rs_topix > 0 フィルター後 Top bear_n 銘柄

    先読み防止: 月 T の選択は月 T-1 末データで計算。

    Parameters
    ----------
    universe_raw : {symbol: {"df": OHLCV DataFrame, ...}}
    topix_close  : TOPIX (or ETF proxy) の日次終値 Series
    all_syms     : スコアリング対象銘柄リスト
    active_n     : Bull 時の選択銘柄数上限
    start, end   : sym_active_df の対象期間
    rsr_df_full  : 事前計算済み RSR DataFrame（列=銘柄）。None の場合 RSR スコアは 0
    bear_n       : Bear 時の選択銘柄数上限（None → max(20, active_n-10)）
    bear_pool    : Bear 時の候補プール
                   None          → all_syms を使用（rs>0 フィルター付き）
                   list[str]     → その銘柄のみ候補（rs>0 フィルター付き）
                   "bull_score_only" → Bull スコアのまま Top bear_n に絞るだけ

    Returns
    -------
    pd.DataFrame : index=取引日、columns=銘柄、値は 1.0（活性）/ 0.0（非活性）
    """
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    if bear_n is None:
        bear_n = max(20, active_n - 10)

    # 全銘柄の Close と Volume を横断 DataFrame に整理
    close_all = pd.DataFrame({
        s: universe_raw[s]["df"]["Close"]
        for s in all_syms if s in universe_raw
    })
    vol_all = pd.DataFrame({
        s: universe_raw[s]["df"]["Volume"]
        for s in all_syms if s in universe_raw
    })
    close_all = close_all.sort_index()
    vol_all   = vol_all.sort_index()

    valid_syms = list(close_all.columns)

    # 対象期間の全取引日
    all_dates = close_all.index[
        (close_all.index >= ts_start) & (close_all.index <= ts_end)
    ]
    if len(all_dates) == 0:
        return pd.DataFrame(1.0, index=all_dates, columns=valid_syms)

    # 各月の最初の取引日を特定
    monthly_first: dict[str, pd.Timestamp] = {}
    for dt in all_dates:
        key = dt.strftime("%Y-%m")
        if key not in monthly_first:
            monthly_first[key] = dt

    # 月ごとの活性銘柄セットを構築
    monthly_active: dict[str, set] = {}
    monthly_regime: dict[str, str] = {}   # "bull" | "sustained_bear"

    for key, first_dt in sorted(monthly_first.items()):
        pos = close_all.index.searchsorted(first_dt)
        if pos == 0:
            monthly_active[key] = set(valid_syms[:min(active_n, len(valid_syms))])
            monthly_regime[key] = "bull"
            continue
        eval_dt = close_all.index[pos - 1]

        hist_close = close_all.loc[:eval_dt]
        hist_vol   = vol_all.loc[:eval_dt]

        if len(hist_close) < MOM_PERIOD + 5:
            monthly_active[key] = set(valid_syms[:min(active_n, len(valid_syms))])
            monthly_regime[key] = "bull"
            continue

        # ── レジーム判定: TOPIX vs MA200（持続型 Bear のみ切り替え）──
        # 短期 crash（COVID型, 2ヶ月以内）では Bear scoring を適用しない。
        # 持続型 Bear（2022型, 60営業日以上）でのみ Bear scoring を使う。
        topix_hist = topix_close.loc[:eval_dt]
        topix_last = float(topix_hist.iloc[-1]) if len(topix_hist) > 0 else np.nan
        if len(topix_hist) >= MA200_PERIOD:
            topix_ma200 = float(topix_hist.iloc[-MA200_PERIOD:].mean())
        else:
            topix_ma200 = float(topix_hist.mean()) if len(topix_hist) > 0 else np.nan

        is_sustained_bear = False
        if (not np.isnan(topix_last) and not np.isnan(topix_ma200)
                and topix_last < topix_ma200):
            lb = min(LOOKBACK_BEAR_CHECK, len(topix_hist))
            if lb >= LOOKBACK_BEAR_CHECK:
                topix_lb     = topix_hist.iloc[-lb:]
                rolling_ma200 = topix_hist.rolling(MA200_PERIOD).mean().loc[:eval_dt]
                recent_ma200  = rolling_ma200.iloc[-lb:] if len(rolling_ma200) >= lb else rolling_ma200
                days_below    = int((topix_lb.values < recent_ma200.values).sum())
                is_sustained_bear = days_below >= SUSTAINED_BEAR_DAYS

        is_bull = not is_sustained_bear
        regime_label = "bull" if is_bull else "sustained_bear"
        monthly_regime[key] = regime_label

        # ── 共通ファクター ────────────────────────────────────────────
        rsr_score = pd.Series(np.nan, index=hist_close.columns)
        if rsr_df_full is not None:
            rsr_hist = rsr_df_full.loc[rsr_df_full.index <= eval_dt]
            if not rsr_hist.empty:
                last_rsr = rsr_hist.iloc[-1]
                for sym in hist_close.columns:
                    if sym in last_rsr.index:
                        rsr_score[sym] = last_rsr[sym]

        if len(hist_vol) >= VOL_PERIOD:
            avg_vol = hist_vol.iloc[-VOL_PERIOD:].mean()
        else:
            avg_vol = hist_vol.mean()
        log_vol = np.log1p(avg_vol.replace(0, np.nan).fillna(0))

        if len(hist_close) > MOM_PERIOD:
            mom63 = hist_close.iloc[-1] / hist_close.iloc[-1 - MOM_PERIOD] - 1
        else:
            mom63 = pd.Series(0.0, index=hist_close.columns)

        # ── loss_penalty: 直近90日の損失ペナルティ ──────────────────────
        # trailing_pnl_90d の代理として 90日リターンを使用。
        # 負のリターン（損失）銘柄のスコアを下げることで構造的ルーザーを排除する。
        # 係数 0.10 = 小さく始めて OOS 改善幅を見てから調整する。
        LOSS_PENALTY_COEF = 0.10
        LOSS_PERIOD       = 90
        loss_penalty_z    = pd.Series(0.0, index=hist_close.columns)
        if len(hist_close) > LOSS_PERIOD:
            pnl_90d = hist_close.iloc[-1] / hist_close.iloc[-1 - LOSS_PERIOD] - 1
            # 損失のみを対象（正のリターンは 0 に clamp）
            loss_only = pnl_90d.clip(upper=0.0).abs()
            loss_penalty_z = _zscore(loss_only)

        if is_bull:
            # ── Bull: モメンタム重視 ──────────────────────────────────
            score = (
                0.40 * _zscore(mom63)
                + 0.35 * _zscore(rsr_score)
                + 0.25 * _zscore(log_vol)
                - LOSS_PENALTY_COEF * loss_penalty_z
            )
            score_valid = score.dropna()
            if len(score_valid) == 0:
                score_valid = _zscore(mom63).dropna()
            n_select = min(active_n, len(score_valid))

        else:
            # ── Bear モード ───────────────────────────────────────────
            if bear_pool == "bull_score_only":
                # Bull スコアのまま Bear 時は Top bear_n に絞るだけ（rs>0 フィルターなし）
                score = (
                    0.40 * _zscore(mom63)
                    + 0.35 * _zscore(rsr_score)
                    + 0.20 * _zscore(log_vol)
                    - LOSS_PENALTY_COEF * loss_penalty_z
                )
                score_valid = score.dropna()
                n_select = min(bear_n, len(score_valid))
            else:
                # RS vs TOPIX 重視 + rs>0 フィルター
                if len(topix_hist) > MOM_PERIOD:
                    topix_mom = float(
                        topix_hist.iloc[-1] / topix_hist.iloc[-1 - MOM_PERIOD] - 1
                    )
                else:
                    topix_mom = 0.0

                if len(hist_close) > MOM_PERIOD:
                    sym_mom = hist_close.iloc[-1] / hist_close.iloc[-1 - MOM_PERIOD] - 1
                else:
                    sym_mom = pd.Series(0.0, index=hist_close.columns)
                rs_topix = sym_mom - topix_mom

                score = (
                    0.50 * _zscore(rs_topix)
                    + 0.30 * _zscore(rsr_score)
                    + 0.20 * _zscore(log_vol)
                    - LOSS_PENALTY_COEF * loss_penalty_z
                )

                if bear_pool is not None and isinstance(bear_pool, list):
                    bear_mask  = score.index.isin(bear_pool)
                    score_bear = score[bear_mask]
                    rs_bear    = rs_topix[bear_mask]
                else:
                    score_bear = score
                    rs_bear    = rs_topix

                score_valid = score_bear[rs_bear > 0].dropna()
                if len(score_valid) == 0:
                    score_valid = score_bear.dropna()
                n_select = min(bear_n, max(20, len(score_valid)))

        top_syms = set(score_valid.nlargest(n_select).index.tolist())
        monthly_active[key] = top_syms

    # ── 日次 sym_active_df 構築 ────────────────────────────────────────
    active_df = pd.DataFrame(0.0, index=all_dates, columns=valid_syms)
    for dt in all_dates:
        key = dt.strftime("%Y-%m")
        active_set = monthly_active.get(key, set())
        for sym in active_set:
            if sym in active_df.columns:
                active_df.loc[dt, sym] = 1.0

    return active_df


# ------------------------------------------------------------------ #
# dyn_rsr42_bear_rs0 確定設定 ラッパー
# ------------------------------------------------------------------ #
def build_dyn_rsr42_active(
    universe_raw: dict,
    topix_close: pd.Series,
    rsr_df: pd.DataFrame,
    all_syms: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    dyn_rsr42_bear_rs0 の確定設定で sym_active_df を構築する。

    Parameters
    ----------
    universe_raw : {symbol: {"df": OHLCV DataFrame, ...}}
    topix_close  : TOPIX ETF 等の日次終値 Series
    rsr_df       : RSR42 RSR DataFrame（columns=銘柄）
    all_syms     : RSR42 銘柄リスト（42銘柄）
    start, end   : sym_active_df の対象期間

    Returns
    -------
    pd.DataFrame : index=取引日、columns=銘柄、値は 1.0/0.0
    """
    return build_sym_active_df(
        universe_raw = universe_raw,
        topix_close  = topix_close,
        all_syms     = all_syms,
        active_n     = BULL_ACTIVE_N,
        start        = start,
        end          = end,
        rsr_df_full  = rsr_df,
        bear_n       = BEAR_ACTIVE_N,
        bear_pool    = None,   # None → all_syms (RSR42) を使用、rs>0 フィルター付き
    )


def get_today_active_syms(
    universe_raw: dict,
    topix_close: pd.Series,
    rsr_df: pd.DataFrame,
    all_syms: list[str],
    end: str,
) -> set[str]:
    """
    今日（end 日）時点で活性状態の銘柄セットを返す。
    live シグナル生成で BUY 候補フィルターに使用する。

    データが不足している場合は空セットを返す（フォールバック: フィルターなし）。
    """
    if topix_close is None or topix_close.empty or not universe_raw:
        return set()

    # 十分な履歴がない場合はフォールバック
    if len(topix_close) < MOM_PERIOD + 5:
        return set()

    # end の1ヶ月前をスタートにして月次計算を走らせる
    ts_end   = pd.Timestamp(end)
    ts_start = ts_end - pd.Timedelta(days=45)   # 1〜2ヶ月分あれば月次計算に十分
    start_str = ts_start.strftime("%Y-%m-%d")

    try:
        active_df = build_dyn_rsr42_active(
            universe_raw = universe_raw,
            topix_close  = topix_close,
            rsr_df       = rsr_df,
            all_syms     = all_syms,
            start        = start_str,
            end          = end,
        )
    except Exception:
        return set()

    if active_df.empty:
        return set()

    last_row = active_df.iloc[-1]
    return set(last_row[last_row == 1.0].index.tolist())
