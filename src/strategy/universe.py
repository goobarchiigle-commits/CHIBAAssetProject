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

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
# Bear レジーム判定ユーティリティ
# ------------------------------------------------------------------ #
def is_bear_regime(
    topix_close: pd.Series,
    lookback_days: int = 60,
    ma200_below_days: int = 40,
) -> bool:
    """
    直近 lookback_days 日中 ma200_below_days 日以上 MA200 割れ → Bear 判定。
    単日ではなく rolling sum で判定することで過敏性を排除。
    """
    if len(topix_close) < MA200_PERIOD:
        return False
    ma200 = topix_close.rolling(MA200_PERIOD).mean()
    below = (topix_close < ma200).rolling(lookback_days).sum()
    latest_below = below.iloc[-1] if len(below) > 0 else 0
    return bool(latest_below >= ma200_below_days)


def apply_bear_universe_filter(
    universe_df: pd.DataFrame,
    is_bear: bool,
    excluded_sectors: "list[str] | tuple[str, ...]",
    date_str: str = "",
) -> pd.DataFrame:
    """
    Bear regime 時にユニバース最上流で cyclical セクターを除外する。

    Parameters
    ----------
    universe_df      : index=symbol, columns に "sector" を含む DataFrame
    is_bear          : Bear レジーム判定結果
    excluded_sectors : Bear 時に除外するセクター名リスト
    date_str         : ログ用日付文字列

    Returns
    -------
    pd.DataFrame : Bear 時は excluded_sectors を除外した DataFrame、それ以外は原本
    """
    if not is_bear:
        return universe_df

    original_count = len(universe_df)
    if "sector" in universe_df.columns:
        filtered_df = universe_df[~universe_df["sector"].isin(excluded_sectors)]
        excluded_mask = universe_df["sector"].isin(excluded_sectors)
        excluded_symbols = universe_df[excluded_mask].index.tolist()
        sector_dist = filtered_df["sector"].value_counts().to_dict()
    else:
        filtered_df = universe_df
        excluded_symbols = []
        sector_dist = {}

    excluded_count = original_count - len(filtered_df)

    logger.info(
        "bear_universe_filter: date=%s excluded=%d/%d symbols=%s active=%d",
        date_str, excluded_count, original_count,
        excluded_symbols[:10], len(filtered_df),
    )

    try:
        from src.utils.bear_monitor import log_bear_event
        from datetime import date as _date
        import pandas as _pd
        _dt = _pd.Timestamp(date_str).date() if date_str else _date.today()
        log_bear_event(
            dt=_dt,
            is_bear=True,
            excluded_count=excluded_count,
            excluded_symbols=excluded_symbols,
            active_count=len(filtered_df),
            sector_dist=sector_dist,
        )
    except Exception:
        pass  # ログ失敗で本番が落ちない

    return filtered_df


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
    # ── Bear スコアリング v2 ──────────────────────────────────────
    bear_scoring_version: str = "v1",           # "v1"=既存 / "v2"=low_beta+defensive追加
    bear_exclude_sectors: "list[str] | None" = None,  # Bear時に除外するセクターリスト
    sym_sector_map: "dict[str, str] | None" = None,   # {symbol: sector} Bear除外・防御スコア用
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

            # ── Bear ユニバースフィルター: cyclical セクター事前除外 ──────
            # bear_exclude_sectors が指定されている場合、スコアリング前に除外する
            _bear_hist_close = hist_close
            if bear_exclude_sectors and sym_sector_map:
                _uni_df = pd.DataFrame(
                    {"sector": {s: sym_sector_map.get(s, "") for s in hist_close.columns}}
                )
                _filtered_uni = apply_bear_universe_filter(
                    _uni_df,
                    is_bear=True,
                    excluded_sectors=list(bear_exclude_sectors),
                    date_str=key,
                )
                _keep = list(_filtered_uni.index)
                _bear_hist_close = hist_close[_keep] if _keep else hist_close

            if bear_pool == "bull_score_only":
                # Bull スコアのまま Bear 時は Top bear_n に絞るだけ（rs>0 フィルターなし）
                score = (
                    0.40 * _zscore(mom63.reindex(_bear_hist_close.columns).fillna(0))
                    + 0.35 * _zscore(rsr_score.reindex(_bear_hist_close.columns).fillna(0))
                    + 0.20 * _zscore(log_vol.reindex(_bear_hist_close.columns).fillna(0))
                    - LOSS_PENALTY_COEF * loss_penalty_z.reindex(_bear_hist_close.columns).fillna(0)
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

                if len(_bear_hist_close) > MOM_PERIOD:
                    sym_mom = _bear_hist_close.iloc[-1] / _bear_hist_close.iloc[-1 - MOM_PERIOD] - 1
                else:
                    sym_mom = pd.Series(0.0, index=_bear_hist_close.columns)
                rs_topix = sym_mom - topix_mom

                # ── Bear スコアリング v1/v2 分岐 ─────────────────────────
                BETA_PERIOD = 60  # rolling beta 計算期間
                if bear_scoring_version == "v2":
                    # v2: low_beta_score + defensive_sector_score 追加
                    # low_beta_score: rolling 60日ベータ（対TOPIX）の負 zscore
                    if len(_bear_hist_close) >= BETA_PERIOD and len(topix_hist) >= BETA_PERIOD:
                        sym_ret_60  = _bear_hist_close.iloc[-BETA_PERIOD:].pct_change().dropna()
                        topix_ret_60 = topix_hist.iloc[-BETA_PERIOD:].pct_change().dropna()
                        combined_idx = sym_ret_60.index.intersection(topix_ret_60.index)
                        if len(combined_idx) >= 10:
                            sr60 = sym_ret_60.loc[combined_idx]
                            tr60 = topix_ret_60.loc[combined_idx]
                            var_t = float(tr60.var())
                            betas = {
                                s: float(sr60[s].cov(tr60)) / var_t if var_t > 1e-10 else 1.0
                                for s in sr60.columns
                            }
                            beta_s = pd.Series(betas, dtype=float)
                        else:
                            beta_s = pd.Series(1.0, index=_bear_hist_close.columns)
                    else:
                        beta_s = pd.Series(1.0, index=_bear_hist_close.columns)
                    low_beta_z = -_zscore(beta_s.reindex(_bear_hist_close.columns).fillna(1.0))

                    # defensive_sector_score: 防御セクターなら +1
                    DEFENSIVE_SECTORS_V2 = {"食品", "ガス", "情報通信"}
                    if sym_sector_map is not None:
                        def_flag = pd.Series(
                            {s: 1.0 if sym_sector_map.get(s, "") in DEFENSIVE_SECTORS_V2 else 0.0
                             for s in _bear_hist_close.columns}, dtype=float
                        )
                    else:
                        def_flag = pd.Series(0.0, index=_bear_hist_close.columns)
                    defensive_z = _zscore(def_flag)

                    score = (
                        0.45 * _zscore(rs_topix)
                        + 0.25 * _zscore(rsr_score.reindex(_bear_hist_close.columns).fillna(0))
                        + 0.20 * low_beta_z
                        + 0.10 * defensive_z
                        - LOSS_PENALTY_COEF * loss_penalty_z.reindex(_bear_hist_close.columns).fillna(0)
                    )
                else:
                    # v1 (既存): RS重視
                    score = (
                        0.50 * _zscore(rs_topix)
                        + 0.30 * _zscore(rsr_score.reindex(_bear_hist_close.columns).fillna(0))
                        + 0.20 * _zscore(log_vol.reindex(_bear_hist_close.columns).fillna(0))
                        - LOSS_PENALTY_COEF * loss_penalty_z.reindex(_bear_hist_close.columns).fillna(0)
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
    bear_scoring_version: str = "v1",
    bear_exclude_sectors: "list[str] | None" = None,
    sym_sector_map: "dict[str, str] | None" = None,
) -> pd.DataFrame:
    """
    dyn_rsr42_bear_rs0 の確定設定で sym_active_df を構築する。

    Parameters
    ----------
    universe_raw         : {symbol: {"df": OHLCV DataFrame, ...}}
    topix_close          : TOPIX ETF 等の日次終値 Series
    rsr_df               : RSR42 RSR DataFrame（columns=銘柄）
    all_syms             : RSR42 銘柄リスト（42銘柄）
    start, end           : sym_active_df の対象期間
    bear_scoring_version : "v1"=既存 / "v2"=low_beta+defensive追加
    bear_exclude_sectors : Bear時に除外するセクターリスト（None=除外なし）
    sym_sector_map       : {symbol: sector} Bear除外・防御スコア用

    Returns
    -------
    pd.DataFrame : index=取引日、columns=銘柄、値は 1.0/0.0
    """
    return build_sym_active_df(
        universe_raw         = universe_raw,
        topix_close          = topix_close,
        all_syms             = all_syms,
        active_n             = BULL_ACTIVE_N,
        start                = start,
        end                  = end,
        rsr_df_full          = rsr_df,
        bear_n               = BEAR_ACTIVE_N,
        bear_pool            = None,   # None → all_syms (RSR42) を使用、rs>0 フィルター付き
        bear_scoring_version = bear_scoring_version,
        bear_exclude_sectors = bear_exclude_sectors,
        sym_sector_map       = sym_sector_map,
    )


# ------------------------------------------------------------------ #
# 動的セクター除外（research ブランチ / production=false）
# ------------------------------------------------------------------ #
def apply_dynamic_bear_filter(
    universe_df: pd.DataFrame,
    sector_returns_63d: pd.Series,
    topix_return_63d: float,
    is_bear: bool,
    date_str: str = "",
) -> pd.DataFrame:
    """
    固定リストではなく、相対弱度（sector_rs_63）ベースでセクターを自動除外。

    sector_rs_63 = sector_return_63d - topix_return_63d
    exclude = sector_rs_63 < 0  （TOPIX より弱いセクターを除外）

    Bear 時のみ適用。TOPIX より強いセクターは残す。

    メリット:
    - 2022: 銀行・機械を切る（金利ショック）
    - 別の Bear では情報通信を切る（利益率低下局面）
    - 市場適応的、固定リストの硬直性を排除

    **research only** — strategy.yaml の bear_dynamic_filter.enabled が
    false の間は呼び出さないこと。

    Parameters
    ----------
    universe_df        : index=symbol, columns に "sector" を含む DataFrame
    sector_returns_63d : セクター別 63日リターン（index=sector 名）
    topix_return_63d   : TOPIX 63日リターン（float）
    is_bear            : Bear レジーム判定結果
    date_str           : ログ用日付文字列

    Returns
    -------
    pd.DataFrame : Bear 時は rs<0 セクターを除外した DataFrame、それ以外は原本

    sector_returns_63d の計算例（実際のデータ構造に合わせること）:
    ---------------------------------------------------------------
    # price_df: symbol × date の価格 DataFrame（columns=symbol, index=date）
    # sector_map: {symbol: sector_name}
    #
    # symbol_ret_63 = price_df.pct_change(63).iloc[-1]       # 各銘柄の63日リターン
    # sector_map_s  = pd.Series(sector_map)                  # symbol → sector
    # sector_rets   = symbol_ret_63.groupby(sector_map_s).mean()  # セクター平均
    # topix_ret_63  = topix_close.pct_change(63).iloc[-1]
    """
    if not is_bear:
        return universe_df

    if sector_returns_63d.empty:
        logger.warning("apply_dynamic_bear_filter: sector_returns_63d が空 → フィルターをスキップ")
        return universe_df

    # sector_rs_63 = セクターリターン - TOPIX リターン
    sector_rs = sector_returns_63d - topix_return_63d
    exclude_sectors = sector_rs[sector_rs < 0].index.tolist()

    if not exclude_sectors:
        logger.info(
            "apply_dynamic_bear_filter: date=%s rs<0 セクターなし → 除外なし",
            date_str,
        )
        return universe_df

    if "sector" not in universe_df.columns:
        logger.warning("apply_dynamic_bear_filter: 'sector' 列なし → フィルターをスキップ")
        return universe_df

    original_count = len(universe_df)
    filtered_df    = universe_df[~universe_df["sector"].isin(exclude_sectors)]
    excluded_count = original_count - len(filtered_df)

    logger.info(
        {
            "event":             "dynamic_bear_filter_applied",
            "date":              date_str,
            "exclude_sectors":   exclude_sectors,
            "excluded_count":    excluded_count,
            "active_count":      len(filtered_df),
            "sector_rs_summary": sector_rs.sort_values().round(4).to_dict(),
        }
    )

    return filtered_df


TARGET_VOL     = 0.15    # 年率ターゲットボラ（15%）
VOL_CAP_WINDOW = 20      # ボラ計算ウィンドウ（営業日）
SYM_CAP_MIN    = 0.02    # 銘柄 cap 下限
SYM_CAP_MAX    = 0.08    # 銘柄 cap 上限（固定 cap と揃える）
SEC_CAP_MIN    = 0.10    # セクター cap 下限
SEC_CAP_MAX    = 0.25    # セクター cap 上限（固定 cap と揃える）


def dynamic_caps(
    symbol: str,
    sector: str,
    close_prices: "pd.DataFrame",       # 全銘柄 Close (columns=symbol)
    sector_members: "dict[str, list]",  # {sector: [sym, ...]}
    eval_date_idx: int,
    window: int = VOL_CAP_WINDOW,
) -> "tuple[float, float]":
    """
    ボラティリティ連動動的 cap を返す。

    Parameters
    ----------
    symbol        : 対象銘柄
    sector        : 対象セクター名
    close_prices  : 全銘柄終値 DataFrame (index=日付, columns=銘柄コード)
    sector_members: {sector: [銘柄リスト]}  ─ セクター equal-weight 指数計算に使用
    eval_date_idx : 評価日インデックス（スライス上限）
    window        : ボラ計算ウィンドウ（営業日）

    Returns
    -------
    (symbol_cap, sector_cap) : 資本比率（0.0〜1.0）
    """
    start_idx = max(0, eval_date_idx - window)

    # ── 銘柄ボラ ──────────────────────────────────────────────────
    sym_cap = SYM_CAP_MAX
    if symbol in close_prices.columns and eval_date_idx > start_idx:
        sym_ret = close_prices[symbol].iloc[start_idx:eval_date_idx].pct_change().dropna()
        if len(sym_ret) >= 5:
            sym_vol = float(sym_ret.std()) * np.sqrt(252)
            if sym_vol > 0:
                sym_cap = max(SYM_CAP_MIN, min(SYM_CAP_MAX, 0.5 * TARGET_VOL / sym_vol))

    # ── セクターボラ ───────────────────────────────────────────────
    sec_cap = SEC_CAP_MAX
    members = sector_members.get(sector, [])
    valid_members = [s for s in members if s in close_prices.columns]
    if valid_members and eval_date_idx > start_idx:
        sec_close = close_prices[valid_members].iloc[start_idx:eval_date_idx]
        sec_idx = sec_close.mean(axis=1)
        sec_ret = sec_idx.pct_change().dropna()
        if len(sec_ret) >= 5:
            sec_vol = float(sec_ret.std()) * np.sqrt(252)
            if sec_vol > 0:
                sec_cap = max(SEC_CAP_MIN, min(SEC_CAP_MAX, 1.2 * TARGET_VOL / sec_vol))

    return sym_cap, sec_cap


def precompute_dynamic_caps(
    close_prices: "pd.DataFrame",
    trade_syms: "dict[str, str]",
    window: int = VOL_CAP_WINDOW,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    事前に全日付 × 全銘柄の symbol_cap と sector_cap を計算して返す。

    Returns
    -------
    sym_cap_df   : DataFrame (index=日付, columns=銘柄) ─ 各日の symbol cap
    sec_cap_df   : DataFrame (index=日付, columns=セクター) ─ 各日の sector cap
    """
    # セクター別メンバー
    sector_members: dict[str, list] = {}
    for sym, sec in trade_syms.items():
        sector_members.setdefault(sec, []).append(sym)

    dates = close_prices.index
    n     = len(dates)

    # ── 銘柄ボラ → symbol cap ────────────────────────────────────
    pct_ch = close_prices.pct_change()
    roll_std = pct_ch.rolling(window, min_periods=5).std() * np.sqrt(252)
    # cap = clip(0.5 * TARGET_VOL / vol, SYM_CAP_MIN, SYM_CAP_MAX)
    raw_sym_cap = (0.5 * TARGET_VOL) / roll_std.replace(0, np.nan)
    sym_cap_df  = raw_sym_cap.clip(lower=SYM_CAP_MIN, upper=SYM_CAP_MAX)
    sym_cap_df  = sym_cap_df.fillna(SYM_CAP_MAX)   # ボラ不足時は上限

    # ── セクターボラ → sector cap ────────────────────────────────
    sec_cap_records: dict[str, pd.Series] = {}
    for sec, members in sector_members.items():
        valid = [s for s in members if s in close_prices.columns]
        if not valid:
            sec_cap_records[sec] = pd.Series(SEC_CAP_MAX, index=dates)
            continue
        sec_idx = close_prices[valid].mean(axis=1)
        sec_ret = sec_idx.pct_change()
        sec_vol = sec_ret.rolling(window, min_periods=5).std() * np.sqrt(252)
        raw_cap = (1.2 * TARGET_VOL) / sec_vol.replace(0, np.nan)
        sec_cap = raw_cap.clip(lower=SEC_CAP_MIN, upper=SEC_CAP_MAX).fillna(SEC_CAP_MAX)
        sec_cap_records[sec] = sec_cap

    sec_cap_df = pd.DataFrame(sec_cap_records, index=dates)

    return sym_cap_df, sec_cap_df


def get_today_active_syms(
    universe_raw: dict,
    topix_close: pd.Series,
    rsr_df: pd.DataFrame,
    all_syms: list[str],
    end: str,
    bear_exclude_sectors: "list[str] | None" = None,
    sym_sector_map: "dict[str, str] | None" = None,
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
            universe_raw         = universe_raw,
            topix_close          = topix_close,
            rsr_df               = rsr_df,
            all_syms             = all_syms,
            start                = start_str,
            end                  = end,
            bear_exclude_sectors = bear_exclude_sectors,
            sym_sector_map       = sym_sector_map,
        )
    except Exception:
        return set()

    if active_df.empty:
        return set()

    last_row = active_df.iloc[-1]
    active_set = set(last_row[last_row == 1.0].index.tolist())

    # ── 日次レジームログ ────────────────────────────────────────────
    # Bear 日かつ bear_exclude_sectors が有効な場合は apply_bear_universe_filter 内で詳細ログ済み。
    # Bull 日、またはフィルター無効な Bear 日はここでログ（ファイル空を防ぐ）。
    try:
        _is_bear_live = is_bear_regime(topix_close)
        _need_log = (not _is_bear_live) or (bear_exclude_sectors is None)
        if _need_log:
            from src.utils.bear_monitor import log_bear_event
            log_bear_event(
                dt=pd.Timestamp(end).date(),
                is_bear=_is_bear_live,
                excluded_count=0,
                excluded_symbols=[],
                active_count=len(active_set),
                sector_dist={},
            )
    except Exception:
        pass

    return active_set
