"""
src/database/sources/jpx_official.py
JPX（日本取引所グループ）公式サイトが提供する機械取得可能CSV（automation配下・固定URL）を
SourceAdapter として実装する。J-Quants APIには存在しない指数構成銘柄データを補完する
（database/market 指数構成銘柄フラグ調査 Priority 2 に対応。plan参照）。

実装済み: JPXプライム150（安定した automation パスのCSV・月次更新）。
JPX日経400は構成銘柄CSVの添付URLが定期見直しごとに変わるため v1 では未実装
（classifications.parquet の IsJPX400 は NULL 固定。次段階で HTML再発見ロジックを追加して昇格）。
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

JPX_PRIME150_WEIGHT_CSV_URL = (
    "https://www.jpx.co.jp/automation/markets/indices/jpx-prime150/files/jpxprime150weight_j.csv"
)
_REQUEST_TIMEOUT_SEC = 30.0


def _normalize_code(raw_code: str) -> str:
    """
    JPX公式CSVの証券コードは4桁（例: "1878"）。J-Quantsは普通株式に5桁コード
    （4桁+"0"、例: "18780"）を使うため、突合できるよう正規化する。
    """
    code = str(raw_code).strip()
    if len(code) == 4:
        return code + "0"
    return code


class JPXOfficialSource:
    """JPX公式automation CSVを取得するSourceAdapter実装。"""

    name = "jpx_official_csv"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._as_of: datetime | None = None

    def fetch_ohlcv(self, start: str, end: str) -> pd.DataFrame:  # SourceAdapterプロトコル（非対応）
        raise NotImplementedError("JPXOfficialSource はOHLCVを提供しない（指数構成銘柄専用）。")

    def fetch_jpx_prime150_constituents(self) -> pd.DataFrame:
        """
        JPXプライム150の最新構成銘柄一覧を取得する。
        戻り値列: Code（J-Quants 5桁形式に正規化済み）, CompanyName, Sector, Weight, AsOfDate
        """
        resp = self._session.get(JPX_PRIME150_WEIGHT_CSV_URL, timeout=_REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content), encoding="cp932", dtype=str)
        df.columns = ["AsOfDate", "CompanyName", "Code", "Sector", "Weight"]
        df = df.dropna(subset=["Code"]).reset_index(drop=True)  # 末尾の空行・注記行を除外
        df["Code"] = df["Code"].map(_normalize_code)
        df["AsOfDate"] = pd.to_datetime(df["AsOfDate"], format="%Y%m%d", errors="coerce")
        self._as_of = datetime.now(_JST)
        logger.info("[JPX_OFFICIAL] jpx_prime150 constituents=%d as_of=%s", len(df), self._as_of)
        return df

    def fetch_master(self, kind: str) -> pd.DataFrame:
        if kind == "classifications_prime150":
            return self.fetch_jpx_prime150_constituents()
        raise ValueError(f"JPXOfficialSource.fetch_master: 未対応kind={kind}")

    def as_of(self) -> datetime:
        return self._as_of or datetime.now(_JST)
