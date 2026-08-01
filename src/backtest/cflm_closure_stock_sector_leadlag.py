"""
src/backtest/cflm_closure_stock_sector_leadlag.py
CFLM Closure Audit — Stock<->Sector Lead-Lag Structural Diagnostic（一回限り・非Study）

正典: docs/research/cflm_closure_audit_2026-08-01.md
     ユーザー承認スコープ（2026-08-01・AskUserQuestion回答）。

目的: Stock->Sector / Sector->Stock / Same-day のいずれの構造が支配的かを診断する。
      forward-return targetは一切使用しない（新alpha探索ではない）。相関は
      lead-lag associationとして扱い、因果関係として解釈しない。

禁止事項（本スクリプトのスコープ）:
  新規Study番号なし・production/Scheduler変更なし・BT/Composite Score/戦略シグナル一切なし・
  閾値/パラメータ最適化なし（分類規則は事前固定・結果を見て変更しない）・
  新規forward-return targetなし・個別銘柄Layerへの拡張なし。

再利用（import・無改変）: src.backtest.study111_cflm_sector_capital_flow
  load_master_calendar / load_sector_map / load_pit_universe_intervals / load_ohlcv_panel
  build_sector_daily_series / accel_signal_from_level / breadth_expansion_signal
  build_official_rank_signals / DISCOVERY / VALIDATION / MA_SHORT / MA_LONG / ACCEL_LAG
src.backtest.study98_sector_momentum_persistence.calc_regime（年度別チェック用・未使用なら省略可）

新規処理（Study111に無いもの）:
  1. 個別銘柄側accel signal（TurnoverValue/Volume、accel_signal_from_levelと同一式をper-Code適用）
  2. Stock-side participation breadth（sector-day別「個別accel>0の銘柄比率」）
  3. Cross-correlation（lag=-10..+10営業日、sector別z-score後pooled・discovery/validation別・年度別）
  4. Stock->Sector / Sector->Stock / Same-day 分類（事前固定規則: argmax|corr|のlag<=-2→Sector->Stock、
     >=+2→Stock->Sector、-1..+1→Same-day）

出力: backtests/cflm_closure_stock_sector_leadlag.json（1個のみ）
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json

import numpy as np
import pandas as pd

from src.backtest.study111_cflm_sector_capital_flow import (
    ACCEL_LAG,
    DISCOVERY,
    MA_LONG,
    MA_SHORT,
    VALIDATION,
    accel_signal_from_level,
    breadth_expansion_signal,
    build_official_rank_signals,
    build_sector_daily_series,
    load_master_calendar,
    load_ohlcv_panel,
)
from src.database.index_prices import SECTOR17_NAMES
from src.paths import RESULTS_DIR

_JST = timezone(timedelta(hours=9))

LAGS = list(range(-10, 11))  # 営業日。負=sectorが先行（stock(t) vs sector(t+lag), lag<0はsector(t-|lag|)）
DEAD_ZONE = 1  # |lag|<=1 -> Same-day（事前固定）
OUT_JSON = RESULTS_DIR / "cflm_closure_stock_sector_leadlag.json"


# ======================================================================
# 1. Stock-side proxy（Study111 accel_signal_from_levelと同一式・per-Code適用）
# ======================================================================
def build_stock_side_accel(panel: pd.DataFrame) -> pd.DataFrame:
    """panel（Date,Code,Sector17Code,Close,Volume,TurnoverValue・PIT filtered済み）から
    個別銘柄ごとにMA_SHORT/MA_LONG比のACCEL_LAG日変化（Study111 H1/H2と同一式）を計算し、
    'turnover_accel_pos'/'volume_accel_pos'（正なら1・負/NaNなら0）を付与して返す。"""
    p = panel.sort_values(["Code", "Date"]).copy()

    def _accel(s: pd.Series) -> pd.Series:
        ma_short = s.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
        ma_long = s.rolling(MA_LONG, min_periods=MA_LONG).mean()
        ratio = ma_short / ma_long
        return ratio - ratio.shift(ACCEL_LAG)

    p["turnover_accel"] = p.groupby("Code")["TurnoverValue"].transform(_accel)
    p["volume_accel"] = p.groupby("Code")["Volume"].transform(_accel)
    p["turnover_accel_pos"] = (p["turnover_accel"] > 0).astype(float)
    p["volume_accel_pos"] = (p["volume_accel"] > 0).astype(float)
    p.loc[p["turnover_accel"].isna(), "turnover_accel_pos"] = np.nan
    p.loc[p["volume_accel"].isna(), "volume_accel_pos"] = np.nan
    return p


def build_stock_participation(panel_with_accel: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """sector-day別「個別accel>0の銘柄比率」（Stock-side participation breadth）。
    Study111 H5（price-MA breadth）とは異なる次元（capital-flow参加率）。"""
    agg = panel_with_accel.groupby(["Date", "Sector17Code"]).agg(
        turnover_participation=("turnover_accel_pos", "mean"),
        volume_participation=("volume_accel_pos", "mean"),
        n_valid=("turnover_accel_pos", "count"),
    ).reset_index()

    def _pivot(col: str) -> pd.DataFrame:
        wide = agg.pivot(index="Date", columns="Sector17Code", values=col)
        return wide.reindex(calendar)

    return {
        "turnover_participation": _pivot("turnover_participation"),
        "volume_participation": _pivot("volume_participation"),
    }


# ======================================================================
# 2. Cross-correlation（sector別z-score後pooled）
# ======================================================================
def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0)) / df.std(axis=0)


def _pooled_lag_corr(stock_side: pd.DataFrame, sector_side: pd.DataFrame, dates_mask: pd.Series, lags: list[int]) -> dict[int, dict]:
    """stock_side(t) vs sector_side(t+lag) のPearson相関。sector列別z-score後、
    共通sector列・共通日付でpoolして1本のlag-corr系列にする。"""
    common_cols = stock_side.columns.intersection(sector_side.columns)
    ss = _zscore(stock_side[common_cols])
    sc = _zscore(sector_side[common_cols])

    out: dict[int, dict] = {}
    for lag in lags:
        sc_shift = sc.shift(-lag)  # lag>0: sectorの未来値を今日のstockと突き合わせる = stock->sector検出
        x_frames, y_frames = [], []
        for col in common_cols:
            x = ss[col].where(dates_mask)
            y = sc_shift[col].where(dates_mask)
            both = pd.concat([x, y], axis=1).dropna()
            if len(both) < 30:
                continue
            x_frames.append(both.iloc[:, 0])
            y_frames.append(both.iloc[:, 1])
        if not x_frames:
            out[lag] = {"corr": None, "n": 0}
            continue
        x_all = pd.concat(x_frames, ignore_index=True)
        y_all = pd.concat(y_frames, ignore_index=True)
        corr = float(x_all.corr(y_all))
        out[lag] = {"corr": None if np.isnan(corr) else corr, "n": int(len(x_all))}
    return out


def classify_direction(lag_corr: dict[int, dict]) -> dict:
    valid = {lag: v["corr"] for lag, v in lag_corr.items() if v["corr"] is not None}
    if not valid:
        return {"argmax_lag": None, "max_abs_corr": None, "direction": "INSUFFICIENT_DATA"}
    argmax_lag = max(valid, key=lambda k: abs(valid[k]))
    max_corr = valid[argmax_lag]
    if argmax_lag <= -2:
        direction = "Sector->Stock"
    elif argmax_lag >= 2:
        direction = "Stock->Sector"
    else:
        direction = "Same-day"
    return {"argmax_lag": argmax_lag, "max_abs_corr": max_corr, "direction": direction}


def period_mask(calendar: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    return pd.Series((calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end)), index=calendar)


def year_masks(calendar: pd.DatetimeIndex) -> dict[str, pd.Series]:
    years = sorted(set(calendar.year))
    return {str(y): pd.Series(calendar.year == y, index=calendar) for y in years}


# ======================================================================
# 3. メイン
# ======================================================================
def main() -> int:
    print("[1/6] マスターカレンダー構築...")
    topix = load_master_calendar()
    calendar = topix.index

    print("[2/6] OHLCV読込・PIT filter（Study111既存関数）...")
    panel = load_ohlcv_panel()
    print(f"  panel rows={len(panel):,}  codes={panel['Code'].nunique():,}")

    print("[3/6] Sector-side signal構築（Study111既存関数・再利用のみ）...")
    sector_series = build_sector_daily_series(panel, calendar)
    h1_sector = accel_signal_from_level(sector_series["turnover"])
    h2_sector = accel_signal_from_level(sector_series["volume"])
    h5_sector = breadth_expansion_signal(sector_series["breadth"])
    official = build_official_rank_signals(calendar)
    h3_sector = official["h3_rank_change"]
    h4_sector = official["h4_rank_acceleration"]

    print("[4/6] Stock-side proxy構築（新規・per-Code accel + sector-day participation）...")
    panel_accel = build_stock_side_accel(panel)
    stock_participation = build_stock_participation(panel_accel, calendar)
    turnover_participation = stock_participation["turnover_participation"]
    volume_participation = stock_participation["volume_participation"]

    pairs = {
        "turnover_participation_vs_sector_H1_turnover_accel": (turnover_participation, h1_sector),
        "volume_participation_vs_sector_H2_volume_accel": (volume_participation, h2_sector),
        "turnover_participation_vs_sector_H5_breadth_expansion": (turnover_participation, h5_sector),
        "volume_participation_vs_sector_H5_breadth_expansion": (volume_participation, h5_sector),
        "turnover_participation_vs_sector_H3_rank_change": (turnover_participation, h3_sector),
        "turnover_participation_vs_sector_H4_rank_acceleration": (turnover_participation, h4_sector),
    }

    print("[5/6] Cross-correlation計算（pooled/discovery/validation/年度別）...")
    disc_mask = period_mask(calendar, *DISCOVERY)
    val_mask = period_mask(calendar, *VALIDATION)
    full_mask = pd.Series(True, index=calendar)
    yr_masks = year_masks(calendar)

    results: dict = {}
    for pair_name, (stock_side, sector_side) in pairs.items():
        print(f"  {pair_name} ...")
        pooled = _pooled_lag_corr(stock_side, sector_side, full_mask, LAGS)
        disc = _pooled_lag_corr(stock_side, sector_side, disc_mask, LAGS)
        val = _pooled_lag_corr(stock_side, sector_side, val_mask, LAGS)
        yby = {yr: _pooled_lag_corr(stock_side, sector_side, m, LAGS) for yr, m in yr_masks.items()}

        results[pair_name] = {
            "pooled": {"lag_corr": pooled, "classification": classify_direction(pooled)},
            "discovery": {"lag_corr": disc, "classification": classify_direction(disc)},
            "validation": {"lag_corr": val, "classification": classify_direction(val)},
            "year_by_year": {yr: {"lag_corr": v, "classification": classify_direction(v)} for yr, v in yby.items()},
        }
        print(f"    pooled direction={results[pair_name]['pooled']['classification']['direction']}"
              f"  argmax_lag={results[pair_name]['pooled']['classification']['argmax_lag']}")

    print("[6/6] 出力生成...")
    write_output(results, panel, calendar)
    print(f"  JSON: {OUT_JSON}")
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return str(o)


def write_output(results: dict, panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = {
        "study": "CFLM_Closure_Audit_Stock_Sector_Leadlag",
        "title": "Stock<->Sector Lead-Lag Structural Diagnostic（探索的・forward-return target不使用・非Study）",
        "generated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope_note": "CFLM Sector Layer Gate=CLOSED（Study111）は変更しない。本診断は独立の構造確認。"
                      "相関はlead-lag associationとして扱い、因果関係として解釈しない。"
                      "classification.parquet non-PIT制約を継承（Study111 H1/H2/H5と同一・MEASUREMENT_LIMITATION相当）。",
        "params": {
            "MA_SHORT": MA_SHORT, "MA_LONG": MA_LONG, "ACCEL_LAG": ACCEL_LAG,
            "LAGS": LAGS, "DEAD_ZONE": DEAD_ZONE,
            "classification_rule": "argmax|corr| lag<=-2 -> Sector->Stock, >=2 -> Stock->Sector, -1..1 -> Same-day（事前固定）",
            "DISCOVERY": DISCOVERY, "VALIDATION": VALIDATION,
        },
        "sector17_names": SECTOR17_NAMES,
        "n_ohlcv_panel_rows": int(len(panel)),
        "n_ohlcv_codes": int(panel["Code"].nunique()),
        "prohibited_confirmed": {
            "bt_engine_used": False, "strategy_backtest": False, "new_jquants_calls": False,
            "forward_return_target_used": False, "threshold_optimization": False,
            "new_study_number_assigned": False, "production_code_changed": False,
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(full, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
