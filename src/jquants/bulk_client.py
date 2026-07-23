"""
src/jquants/bulk_client.py
J-Quants Bulk API（/v2/bulk/list, /v2/bulk/get）低レベルクライアント。

背景（2026-07-23 実測確認）: 分足・Tickは銘柄別REST逐次取得（daily_bars_for_date等と同パターン）
だと数万リクエスト規模になり非現実的。正しい経路はBulk API — 1リクエストで日次/月次の
「全銘柄まとめてCSV.GZ」を取得できる。同じ経路で equities/bars/daily・equities/master・
fins/summary・indices/bars/daily・markets/margin-*・markets/short-* も取得可能（実測確認済み）。

ファイル粒度（実測確認済み）:
  直近約2-3週間  : {endpoint}/live/{endpoint_flat}_{YYYYMMDD}.csv.gz       （日次・全銘柄）
  それより過去   : {endpoint}/historical/{year}/{endpoint_flat}_{YYYYMM}.csv.gz （月次・全銘柄）
  endpoint_flat = endpoint の "/" を "_" に置換したもの。

責務: Bulk APIのlist/get呼び出し（JQuantsClientのスロットリング・リトライを再利用）と、
署名付きURLからのストリーミングダウンロード（SHA256計算込み・archive/への永久保存）のみ。
Parquet化・スキーマ検証は呼び出し側（src/database/minute_bars.py等）の責務。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from src.jquants.client import JQuantsClient
from src.jquants.exceptions import JQuantsAPIError

logger = logging.getLogger(__name__)

BULK_LIST_PATH = "/v2/bulk/list"
BULK_GET_PATH = "/v2/bulk/get"

_DOWNLOAD_TIMEOUT_SEC = 300.0
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB
_DOWNLOAD_RETRY_MAX = 3

# 実測確認済みエンドポイント一覧（2026-07-23・bulk/list実測）。
ENDPOINT_EQUITIES_TRADES = "equities/trades"
ENDPOINT_EQUITIES_BARS_MINUTE = "equities/bars/minute"
ENDPOINT_EQUITIES_BARS_DAILY = "equities/bars/daily"
ENDPOINT_EQUITIES_MASTER = "equities/master"
ENDPOINT_EQUITIES_INVESTOR_TYPES = "equities/investor-types"
ENDPOINT_FINS_SUMMARY = "fins/summary"
ENDPOINT_INDICES_BARS_DAILY = "indices/bars/daily"
ENDPOINT_INDICES_BARS_DAILY_TOPIX = "indices/bars/daily/topix"
ENDPOINT_MARKETS_MARGIN_ALERT = "markets/margin-alert"
ENDPOINT_MARKETS_MARGIN_INTEREST = "markets/margin-interest"
ENDPOINT_MARKETS_SHORT_RATIO = "markets/short-ratio"
ENDPOINT_MARKETS_SHORT_SALE_REPORT = "markets/short-sale-report"


@dataclass(frozen=True)
class BulkFileRef:
    """Bulk APIが返す1ファイルの参照情報（bulk/listの1レコードに対応）。"""
    key: str
    size_bytes: int
    last_modified: str


def _endpoint_flat(endpoint: str) -> str:
    return endpoint.replace("/", "_")


def historical_monthly_key(endpoint: str, year: int, yyyymm: str) -> str:
    """月次アーカイブのKeyを構築する（例: equities/trades/historical/2026/equities_trades_202606.csv.gz）。"""
    return f"{endpoint}/historical/{year}/{_endpoint_flat(endpoint)}_{yyyymm}.csv.gz"


def live_daily_key(endpoint: str, yyyymmdd: str) -> str:
    """直近日次ファイルのKeyを構築する（例: equities/trades/live/equities_trades_20260722.csv.gz）。"""
    return f"{endpoint}/live/{_endpoint_flat(endpoint)}_{yyyymmdd}.csv.gz"


class JQuantsBulkClient:
    """
    使い方:
        client = JQuantsBulkClient()
        files = client.list_files("20260710")  # その日に存在する全Bulkファイル一覧
        ref = client.get_file_ref("equities/trades", month_key="202606")
        client.download_to_file(ref.key, dest_path, expected_size=ref.size_bytes)
    """

    def __init__(self, client: JQuantsClient | None = None) -> None:
        self.client = client or JQuantsClient()

    # ------------------------------------------------------------------ #
    def list_files(self, date: str) -> list[BulkFileRef]:
        """
        指定日（"YYYYMMDD"）に存在するBulkファイル一覧を返す（全endpoint横断・1リクエスト）。
        list結果はそのまま `key` の実在を保証しない場合がある（実測: get時にのみ真の存在確認可能な
        ケースを確認済み）。存在確認は download_to_file() 側の404/400ハンドリングに委ねる。
        """
        data = self.client.get(BULK_LIST_PATH, {"date": date})
        return [
            BulkFileRef(key=x["Key"], size_bytes=int(x.get("Size", 0)), last_modified=x.get("LastModified", ""))
            for x in data.get("data", [])
        ]

    def get_download_url(self, key: str) -> str:
        """指定Keyの署名付きダウンロードURLを取得する。存在しないKeyは JQuantsAPIError を送出する。"""
        data = self.client.get(BULK_GET_PATH, {"key": key})
        url = data.get("url", "")
        if not url:
            raise JQuantsAPIError(f"bulk/get: url が応答に含まれない key={key}")
        return url

    def file_exists(self, key: str) -> bool:
        """指定Keyがダウンロード可能か確認する（bulk/getが成功するかで判定・軽量な存在確認）。"""
        try:
            self.get_download_url(key)
            return True
        except JQuantsAPIError:
            return False

    # ------------------------------------------------------------------ #
    def download_to_file(self, key: str, dest_path: Path) -> tuple[int, str]:
        """
        Bulk APIのKeyをdest_pathへストリーミングダウンロードする（メモリに全展開しない）。
        署名付きURL先（Cloudflare R2）はJ-Quants本体のレート制限対象外だが、大容量転送のため
        独自にリトライする（J-QuantsのAPIキー呼び出しはbulk/get1回のみ・都度URLを再発行する）。

        Returns: (ダウンロードバイト数, SHA256ハッシュ)
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        last_error: Exception | None = None
        for attempt in range(_DOWNLOAD_RETRY_MAX):
            url = self.get_download_url(key)  # 都度再発行（署名付きURLは短時間で失効するため）
            sha256 = hashlib.sha256()
            total = 0
            try:
                with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                            if not chunk:
                                continue
                            f.write(chunk)
                            sha256.update(chunk)
                            total += len(chunk)
                tmp_path.replace(dest_path)
                logger.info("[BULK_CLIENT] downloaded key=%s bytes=%d sha256=%s", key, total, sha256.hexdigest()[:12])
                return total, sha256.hexdigest()
            except (requests.RequestException, OSError) as e:
                last_error = e
                logger.warning(
                    "[BULK_CLIENT] download_error key=%s attempt=%d/%d error=%s",
                    key, attempt + 1, _DOWNLOAD_RETRY_MAX, e,
                )
                tmp_path.unlink(missing_ok=True)

        raise JQuantsAPIError(f"Bulkダウンロード失敗（リトライ上限到達）: key={key} error={last_error}")

    # ------------------------------------------------------------------ #
    def resolve_month_source(
        self, endpoint: str, year: int, month: int, trading_days: list[str],
    ) -> dict:
        """
        指定月のデータ取得方法を解決する。まず月次historicalファイルの存在を確認し、
        あればそれを使う（1ファイル=全銘柄・当該月）。なければ（直近・未ロールアップの月）
        trading_days（"YYYYMMDD"のリスト・呼び出し側がJPXCalendar等で計算する）の日次liveファイルに
        フォールバックする。

        Returns: {"granularity": "monthly", "key": str} または
                 {"granularity": "daily", "keys": list[str]}
        """
        yyyymm = f"{year}{month:02d}"
        monthly_key = historical_monthly_key(endpoint, year, yyyymm)
        if self.file_exists(monthly_key):
            return {"granularity": "monthly", "key": monthly_key}

        # 月次historicalが未ロールアップ（直近月）の場合、日次liveにフォールバックする。
        # 個々の日の存在確認はしない（休場日等で正規に無い日もあるため）。呼び出し側の
        # ダウンロードループが1日ずつ試行し、404/400は「その日は無い」として警告ログのみでスキップする。
        daily_keys = [live_daily_key(endpoint, d) for d in trading_days]
        return {"granularity": "daily", "keys": daily_keys}
