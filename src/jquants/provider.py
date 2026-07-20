"""
src/jquants/provider.py
J-Quants エンドポイントの意味的ラッパー。client.py の上に薄く被せるだけで、
HTTP の詳細（Retry/Backoff/Rate Limit/Token）は client.py に委譲する。

raw/processed 責務分離の原則（2026-07-09改訂）:
  本モジュールが返すのは常に「APIレスポンスをできる限りそのまま」保った形。
  列のリネーム・選別・調整値での上書きは行わない（それは normalize.py の責務）。
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.jquants.client import JQuantsClient
from src.jquants.exceptions import JQuantsAPIError

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

# 2026-07-09実測: 契約データ提供開始日は「今日から遡って約10年」のローリング窓と推定される
# （範囲外リクエストへの400応答 "Your subscription covers the following dates: YYYY-MM-DD ~ ."
#  が、問い合わせ当日からちょうど10年前の日付を返した）。固定始点ではないため、
# ハードコードされた日付定数は翌日には既に不正確になる（実際に2026-07-09→07-10で1日ズレを確認済み）。
SUBSCRIPTION_ROLLING_WINDOW_YEARS = 10
_SUBSCRIPTION_FLOOR_PATTERN = re.compile(r"covers the following dates:\s*(\d{4}-\d{2}-\d{2})")


def estimate_subscription_floor(today: date | None = None) -> str:
    """
    契約データ提供開始日の推定値（APIを呼ばない日付計算のみ・ローリング10年窓と仮定）。
    真の値を確認するには JQuantsProvider.detect_subscription_floor()（1リクエスト）を使うこと。
    """
    today = today or datetime.now(_JST).date()
    try:
        floor = today.replace(year=today.year - SUBSCRIPTION_ROLLING_WINDOW_YEARS)
    except ValueError:
        # today が 2/29 で floor年がうるう年でない場合のフォールバック
        floor = today.replace(year=today.year - SUBSCRIPTION_ROLLING_WINDOW_YEARS, day=28)
    return floor.strftime("%Y-%m-%d")


def raw_records_to_frame(records: list[dict]) -> pd.DataFrame:
    """
    daily_quotes/indices 生レコード列 → DataFrame（列名リネームなし・Dateは列のまま）。
    raw/daily_bars_{year}.parquet・staging の両方がこの形をそのまま使う。
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        sort_cols = [c for c in ("Code", "Date") if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ── エンドポイントパス（v2）─────────────────────────────────────────────
# 全て確定（2026-07-09 ASK_FIRST②実APIスモークテスト・各1リクエストのみで実測確認）:
#   応答の list_key は共通で "data"。コードは5桁（例: "86970"）。
#   レコード列は略記（daily_quotes/topix: O/H/L/C/UL/LL/Vo/Va/AdjFactor/AdjO/AdjH/AdjL/AdjC/AdjVo,
#   listed_info: CoName/CoNameEn/S17/S17Nm/S33/S33Nm/ScaleCat/Mkt/MktNm/Mrgn/MrgnNm/ProdCat）
#   → 標準名への変換は src/jquants/schema.py（normalize.py・validator.py・universe.pyが使用）。
# Download Strategy Validation（2026-07-09）で追加確認済み:
#   from/to範囲は動作する（code+from/toで約10年2,442件を1リクエスト・pagination_keyなし）。
#   ただしサブスクリプションのデータ提供開始日は「今日から遡って約10年」のローリング窓
#   （2026-07-09時点で2016-07-09〜。固定始点ではない可能性が高い・要監視）。
#   date単独（code省略）でも動作し、その日の全銘柄（約4,437件）を1リクエストで返す
#   → Strategy C（日次イテレーション）採用済み・src/jquants/study75_downloader.py参照。
DAILY_BARS_PATH = "/v2/equities/bars/daily"   # 確認済み: code+date/code+from-to/date単独 いずれも200
TOPIX_PATH = "/v2/indices/bars/daily"          # 確認済み: code="0000"+date/from-to で200
LISTED_INFO_PATH = "/v2/equities/master"       # 確認済み: code指定で200・data配下1件（2026-07-09追加確認）
# Study82 Phase0.1（2026-07-20公式ドキュメント調査・reports/study82_phase0_1_result.md）で存在確認。
# 実APIパスはv1想定の "/fins/statements" ではなくv2体系のため未実測 — 以下は候補。
# get_fins_summary() 初回呼び出し時に実パスをスモークテストで確認すること（Study82 Phase0本審査の一部）。
FINS_SUMMARY_PATH_CANDIDATES = ("/v2/fins/summary", "/fins/summary")
_LIST_KEY = "data"


class JQuantsProvider:
    def __init__(self, client: JQuantsClient | None = None) -> None:
        self.client = client or JQuantsClient()

    def listed_info(self, code: str = "", date: str = "") -> list[dict]:
        """上場銘柄一覧（Universe構築用）。code/date省略時は全銘柄最新時点。"""
        params: dict[str, str] = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        return list(self.client.get_paginated(LISTED_INFO_PATH, params, list_key=_LIST_KEY))

    def daily_quotes_raw(self, code: str, from_date: str, to_date: str) -> list[dict]:
        """1銘柄の日次OHLCV生レコード（J-Quantsの列名のまま・略記フィールド）。"""
        params = {"code": code, "from": from_date, "to": to_date}
        return list(self.client.get_paginated(DAILY_BARS_PATH, params, list_key=_LIST_KEY))

    def daily_quotes_df(self, code: str, from_date: str, to_date: str) -> pd.DataFrame:
        """1銘柄のOHLCV生レコードをDataFrame化（列名リネームなし）。"""
        records = self.daily_quotes_raw(code, from_date, to_date)
        return raw_records_to_frame(records)

    def daily_bars_for_date(self, date: str) -> list[dict]:
        """
        指定日の全銘柄日次OHLCV生レコード（code省略・Strategy C: 1営業日=1リクエスト）。
        date は確認済みフォーマット "YYYYMMDD"（Download Strategy Validation・2026-07-09実測）。
        """
        params = {"date": date}
        return list(self.client.get_paginated(DAILY_BARS_PATH, params, list_key=_LIST_KEY))

    def topix_raw(self, from_date: str, to_date: str) -> list[dict]:
        """TOPIX指数の日次OHLC生レコード（列名リネームなし）。"""
        params = {"from": from_date, "to": to_date, "code": "0000"}
        return list(self.client.get_paginated(TOPIX_PATH, params, list_key=_LIST_KEY))

    def topix_df(self, from_date: str, to_date: str) -> pd.DataFrame:
        """TOPIXをDataFrame化（列名リネームなし）。"""
        records = self.topix_raw(from_date, to_date)
        return raw_records_to_frame(records)

    def get_fins_summary(
        self, code: str = "", date_from: str = "", date_to: str = "", date: str = "",
    ) -> tuple[list[dict], str]:
        """
        決算短信サマリ（発表日時・業績数値）生レコード取得（Study82 Phase0監査専用・列名リネームなし）。
        code と (date_from/date_to または date) の少なくとも一方の組み合わせを指定すること
        （J-Quants API共通の慣例に倣う・詳細はv2実APIスモークテストで確認）。

        戻り値: (records, resolved_path)。resolved_path はどの候補パスで疎通できたかを示す
        （FINS_SUMMARY_PATH_CANDIDATES を順に試し、200が返った最初のパスを採用・キャッシュしない
        — Phase0監査は都度実測することが目的のため）。

        監査専用メソッド: raw responseをそのまま返す（正規化・列選別なし・raw/processed分離原則）。
        Production判定・アルファ計算には使用しないこと（本Study82の禁止事項）。
        """
        params: dict[str, str] = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        else:
            if date_from:
                params["from"] = date_from
            if date_to:
                params["to"] = date_to

        last_error: JQuantsAPIError | None = None
        for path in FINS_SUMMARY_PATH_CANDIDATES:
            try:
                records = list(self.client.get_paginated(path, params, list_key=_LIST_KEY))
                return records, path
            except JQuantsAPIError as e:
                last_error = e
                logger.info("[JQUANTS_PROVIDER] get_fins_summary: path=%s 疎通失敗 (%s)・次の候補を試行", path, e)
                continue
        raise JQuantsAPIError(
            f"get_fins_summary: 候補パス全滅（{FINS_SUMMARY_PATH_CANDIDATES}）: {last_error}"
        )

    def detect_subscription_floor(self) -> str | None:
        """
        契約データ提供開始日を実測する（1リクエストのみ）。意図的に範囲外（1900年〜）を
        リクエストし、返ってくる400エラー本文
        "Your subscription covers the following dates: YYYY-MM-DD ~ ." から実際の開始日を抽出する。
        文言が変わった場合や200が返った場合は None を返す（fail-open・呼び出し側で
        estimate_subscription_floor() にフォールバックすること）。
        """
        try:
            self.client.get(DAILY_BARS_PATH, {"code": "86970", "from": "1900-01-01", "to": "1900-01-02"})
        except JQuantsAPIError as e:
            body = e.body or str(e)
            match = _SUBSCRIPTION_FLOOR_PATTERN.search(body)
            if match:
                return match.group(1)
            logger.warning("[JQUANTS_PROVIDER] detect_subscription_floor: 応答形式が想定外・パース失敗 body=%s", body[:200])
            return None
        logger.warning("[JQUANTS_PROVIDER] detect_subscription_floor: 期待した400エラーが返らなかった（1900年データが実在？）")
        return None
