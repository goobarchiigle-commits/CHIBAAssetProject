"""
src/jquants/study75_adapter.py
Study76互換Adapter（Study75本体は未実装のため変更しない。既に実装済みで待機中の
src/backtest/study76_clenow_benchmark_wf.py が要求する形式へ変換するだけの薄い層）。

Study76が固定で読む形式:
  data/jquants/processed/{symbol}.parquet  … Open/High/Low/Close/Volume/AdjustmentFactor 列・
                                              DatetimeIndex・Open/High/Low/Close は「調整済み値」という前提
                                              （provider.py の既存コメント通り）。

正規化データセット（processed/daily_bars_{year}.parquet、normalize.py の固定スキーマ）から
対象銘柄だけ抽出し、AdjustmentOpen/High/Low/Close/Volume を Open/High/Low/Close/Volume に
リネームして書き出す。processed/ 配下には「正規化フルスキーマ」と「本Adapterが生成する
銘柄別互換ファイル」の2種類が共存する（ファイル名が異なるため衝突しない）。
"""
from __future__ import annotations

import logging

import pandas as pd

from src.paths import JQUANTS_PROCESSED_DIR

logger = logging.getLogger(__name__)

_ADJUSTMENT_TO_OHLCV = {
    "AdjustmentOpen":   "Open",
    "AdjustmentHigh":   "High",
    "AdjustmentLow":    "Low",
    "AdjustmentClose":  "Close",
    "AdjustmentVolume": "Volume",
}
_READ_COLUMNS = ["Date", "Code", "AdjustmentFactor", *_ADJUSTMENT_TO_OHLCV.keys()]
_OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor"]


def _symbol_to_code(symbol: str) -> str:
    return symbol.split(".")[0]


def _load_normalized_dataset() -> pd.DataFrame:
    year_files = sorted(JQUANTS_PROCESSED_DIR.glob("daily_bars_*.parquet"))
    if not year_files:
        return pd.DataFrame()
    frames = []
    for f in year_files:
        try:
            frames.append(pd.read_parquet(f, columns=_READ_COLUMNS))
        except (OSError, ValueError, KeyError) as e:
            logger.warning("[STUDY75_ADAPTER] %s 読込失敗（スキップ）: %s", f, e)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["Code"] = combined["Code"].astype(str)
    return combined


def materialize_symbol_panels(symbols: list[str]) -> dict:
    """
    正規化データセットから対象銘柄だけ抽出し、Study76互換の processed/{symbol}.parquet を書き出す。
    Returns: {"materialized": [...], "missing": [...]}
    """
    JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _load_normalized_dataset()
    materialized: list[str] = []
    missing: list[str] = []

    if dataset.empty:
        return {"materialized": [], "missing": list(symbols)}

    for symbol in symbols:
        code = _symbol_to_code(symbol)
        sub = dataset.loc[dataset["Code"] == code]
        if sub.empty:
            # J-Quantsのコード桁数変更（4桁→5桁）への簡易フォールバック。
            sub = dataset.loc[dataset["Code"] == f"{code}0"]
        if sub.empty:
            missing.append(symbol)
            continue

        out = sub.rename(columns=_ADJUSTMENT_TO_OHLCV)
        out = out[["Date", *_OUTPUT_COLUMNS]].set_index("Date").sort_index()
        out.to_parquet(JQUANTS_PROCESSED_DIR / f"{symbol}.parquet")
        materialized.append(symbol)

    if missing:
        logger.warning("[STUDY75_ADAPTER] 正規化データセットに存在しない銘柄: %s", missing)
    return {"materialized": materialized, "missing": missing}
