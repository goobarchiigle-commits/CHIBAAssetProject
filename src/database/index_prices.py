"""
src/database/index_prices.py
TOPIX-17業種別指数（公式・J-Quants /v2/indices/bars/daily）の取得・保存。

database/market/index/README.md で「設計のみ・未実装」とされていた領域を実装する
（Study77で「TOPIX17セクターETF価格データは既存パイプラインに存在せず」と記録された
ギャップの解消）。DIY等ウェイトproxyではなく公式指数を使うことで、セクターモメンタム
研究（Study98等）の精度を向上させる。

指数コード対応（2026-07-23 訂正・実測確認済み）:
  2026-07-15時点の実装は index_code = f"{39 + sector17_code:04d}"（0040-0056）としていたが、
  これは誤り（0040-0056は東証33業種指数側のコード帯・TOPIX-17ではない）。J-Quants公式
  配信対象指数コード一覧（https://jpx-jquants.com/ja/spec/idx-bars-daily/indexcodes）および
  実データでの相関検証（Sector17Code別バスケットとの日次リターン相関・銀行/医薬品で対照実験）
  により、正しいコード帯は 0080-0090（16進表記）であることを実測確認した
  （0080=食品(1) 〜 0090=不動産(17)）。companies.parquet の Sector17Code（1〜17）との対応は
  index_code = format(0x80 + sector17_code - 1, "04X")。TOPIX全体は code="0000"。
  旧コード帯(0040-0056)は東証33業種指数（別データセット・v1では未実装）。

保存先: database/market/index/prices/{IndexCode}.parquet（README想定スキーマに準拠）。
書き込み境界: data/jquants/ には一切書き込まない（database/market 専用・既存の
WriteBoundary原則を踏襲。src/database/sources/jquants_source.py の設計思想と同一）。

実行:
    python -m src.database.index_prices --start 2016-07-01 --end 2026-07-15
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.jquants.provider import JQuantsProvider, TOPIX_PATH, raw_records_to_frame
from src.jquants.schema import DAILY_BARS_RAW_TO_STANDARD, rename_to_standard
from src.paths import DATABASE_MARKET_DIR

logger = logging.getLogger(__name__)

INDEX_PRICES_DIR = DATABASE_MARKET_DIR / "index" / "prices"

# Sector17Code（companies.parquet） → 名称（参照用・実際の取得はコード変換のみで行う）
SECTOR17_NAMES: dict[int, str] = {
    1: "食品", 2: "エネルギー資源", 3: "建設・資材", 4: "素材・化学", 5: "医薬品",
    6: "自動車・輸送機", 7: "鉄鋼・非鉄", 8: "機械", 9: "電機・精密",
    10: "情報通信・サービスその他", 11: "電気・ガス", 12: "運輸・物流",
    13: "商社・卸売", 14: "小売", 15: "銀行", 16: "金融（除く銀行）", 17: "不動産",
}

# その他主要指数（2026-07-23実測確認・Standardプランで取得可能）。
# 日経225はJ-Quants非提供（Nikkei Inc独自ライセンスのため。公式配信対象指数コード一覧に
# 存在しないことを確認済み）。JPX日経400は配当込み・終値のみのB507が唯一の入手可能な系列。
OTHER_INDEX_CODES: dict[str, str] = {
    "0070": "東証グロース市場250指数（旧東証マザーズ指数）",
    "0075": "東証REIT指数",
    "0500": "東証プライム市場指数",
    "0501": "東証スタンダード市場指数",
    "0502": "東証グロース市場指数",
    "0503": "JPXプライム150指数",
    "0504": "JPXスタートアップ急成長100指数",
    "B507": "配当込みJPX日経インデックス400（終値のみ）",
}


def sector17_to_index_code(sector17_code: int) -> str:
    """companies.parquet の Sector17Code（1-17）→ J-Quants TOPIX-17指数コード（'0080'-'0090'・16進）。"""
    if not (1 <= sector17_code <= 17):
        raise ValueError(f"Sector17Code must be 1-17 (TOPIX-17 official sectors only): {sector17_code}")
    return f"{0x80 + sector17_code - 1:04X}"


def fetch_index_series(index_code: str, start: str, end: str,
                        provider: "JQuantsProvider | None" = None) -> pd.DataFrame:
    """1指数の日次OHLC（Date/Open/High/Low/Close）を取得する。"""
    provider = provider or JQuantsProvider()
    params = {"from": start, "to": end, "code": index_code}
    records = list(provider.client.get_paginated(TOPIX_PATH, params, list_key="data"))
    if not records:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
    df = raw_records_to_frame(records)
    df = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").set_index("Date")
    return df[["Open", "High", "Low", "Close"]]


def fetch_and_save_all_topix17(start: str, end: str, *, include_topix: bool = True) -> dict[str, int]:
    """
    ⚠ KNOWN_ISSUES.md ISSUE-001: 既存Parquetを無条件上書きする（狭いstartで実行すると
    既存履歴が消失する）。新規コードから呼び出さないこと。安全な差分更新には
    fetch_and_save_all_topix17_safe() を使う。後方互換のため残置のみ。

    TOPIX-17全17業種指数（+ 任意でTOPIX本体）を取得し
    database/market/index/prices/{IndexCode}.parquet へ保存する。

    Returns: {index_code: n_rows}
    """
    INDEX_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = JQuantsProvider()
    result: dict[str, int] = {}

    codes = [sector17_to_index_code(n) for n in range(1, 18)]
    if include_topix:
        codes = ["0000"] + codes

    for code in codes:
        df = fetch_index_series(code, start, end, provider=provider)
        out_path = INDEX_PRICES_DIR / f"{code}.parquet"
        df.to_parquet(out_path)
        result[code] = len(df)
        logger.info("[INDEX_PRICES] code=%s rows=%d saved=%s", code, len(df), out_path)

    return result


def fetch_and_save_other_indices(start: str, end: str) -> dict[str, int]:
    """
    ⚠ KNOWN_ISSUES.md ISSUE-001: 既存Parquetを無条件上書きする。新規コードから呼び出さないこと。
    安全な差分更新には fetch_and_save_other_indices_safe() を使う。後方互換のため残置のみ。

    OTHER_INDEX_CODES（Growth250/REIT/Prime/Standard/Growth市場/JPXプライム150等）を取得・保存する。
    """
    INDEX_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = JQuantsProvider()
    result: dict[str, int] = {}
    for code in OTHER_INDEX_CODES:
        df = fetch_index_series(code, start, end, provider=provider)
        out_path = INDEX_PRICES_DIR / f"{code}.parquet"
        df.to_parquet(out_path)
        result[code] = len(df)
        logger.info("[INDEX_PRICES] code=%s (%s) rows=%d saved=%s", code, OTHER_INDEX_CODES[code], len(df), out_path)
    return result


class IndexPricesShrinkError(RuntimeError):
    """新規取得後のマージ結果が既存データより縮小する場合に送出する（fail-closed）。"""


def fetch_and_save_index_safe(
    code: str, start: str | None, end: str,
    *, provider: "JQuantsProvider | None" = None, label: str = "",
) -> dict:
    """
    ISSUE-001の恒久対策版: 既存Parquetを必ず読み込み、新規分をappendしてから重複除去して
    保存する（既存ファイルへの単純上書きは一切行わない）。

    - startを省略した場合、既存ファイルの最終日+1日を自動的に開始日とする（真の差分更新）。
      既存ファイルが無い場合のみ、呼び出し元が渡したstart（省略時は全期間相当のデフォルト）を使う。
    - マージ後の行数が既存より減った場合、またはdate_minが既存より後退した場合は
      IndexPricesShrinkErrorを送出し、書き込みを一切行わない（fail-closed）。

    Returns: {"code", "status", "existing_rows", "new_rows", "combined_rows",
              "date_min", "date_max", "fetch_start"}
    """
    provider = provider or JQuantsProvider()
    out_path = INDEX_PRICES_DIR / f"{code}.parquet"

    existing = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    existing.index.name = "Date"
    if out_path.exists():
        existing = pd.read_parquet(out_path)

    existing_rows = len(existing)
    existing_date_min = existing.index.min() if existing_rows else None
    existing_date_max = existing.index.max() if existing_rows else None

    if existing_rows and start is None:
        fetch_start = (existing_date_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        fetch_start = start or "2016-07-01"

    if pd.Timestamp(fetch_start) > pd.Timestamp(end):
        return {
            "code": code, "label": label, "status": "up_to_date",
            "existing_rows": existing_rows, "new_rows": 0, "combined_rows": existing_rows,
            "date_min": str(existing_date_min) if existing_date_min is not None else None,
            "date_max": str(existing_date_max) if existing_date_max is not None else None,
            "fetch_start": fetch_start,
        }

    new_df = fetch_index_series(code, fetch_start, end, provider=provider)

    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if len(combined) < existing_rows:
        raise IndexPricesShrinkError(
            f"code={code}: マージ後の行数({len(combined)})が既存({existing_rows})より少ない・書き込み中止"
        )
    if existing_date_min is not None and (
        combined.index.min() is None or combined.index.min() > existing_date_min
    ):
        raise IndexPricesShrinkError(
            f"code={code}: マージ後のdate_min({combined.index.min()})が既存({existing_date_min})より後退・書き込み中止"
        )

    combined.to_parquet(out_path)
    logger.info(
        "[INDEX_PRICES_SAFE] code=%s(%s) existing=%d new=%d combined=%d fetch_start=%s date_range=%s..%s",
        code, label, existing_rows, len(new_df), len(combined), fetch_start,
        combined.index.min(), combined.index.max(),
    )
    return {
        "code": code, "label": label, "status": "ok",
        "existing_rows": existing_rows, "new_rows": len(new_df), "combined_rows": len(combined),
        "date_min": str(combined.index.min()), "date_max": str(combined.index.max()),
        "fetch_start": fetch_start,
    }


def fetch_and_save_all_topix17_safe(end: str, *, start: str | None = None, include_topix: bool = True) -> list[dict]:
    """TOPIX本体+TOPIX-17業種指数を安全な差分マージ方式で更新する（ISSUE-001対策版）。"""
    INDEX_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = JQuantsProvider()
    codes = [(sector17_to_index_code(n), SECTOR17_NAMES[n]) for n in range(1, 18)]
    if include_topix:
        codes = [("0000", "TOPIX")] + codes
    return [fetch_and_save_index_safe(code, start, end, provider=provider, label=name) for code, name in codes]


def fetch_and_save_other_indices_safe(end: str, *, start: str | None = None) -> list[dict]:
    """OTHER_INDEX_CODESを安全な差分マージ方式で更新する（ISSUE-001対策版）。"""
    INDEX_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = JQuantsProvider()
    return [
        fetch_and_save_index_safe(code, start, end, provider=provider, label=name)
        for code, name in OTHER_INDEX_CODES.items()
    ]


def load_index_series(index_code: str) -> pd.DataFrame:
    """保存済みの指数価格系列を読み込む（Date index・Open/High/Low/Close列）。"""
    path = INDEX_PRICES_DIR / f"{index_code}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"指数価格データが存在しません: {path}\n"
            "  python -m src.database.index_prices --start ... --end ... で取得してください。"
        )
    return pd.read_parquet(path)


def load_all_topix17_series() -> dict[int, pd.DataFrame]:
    """Sector17Code(1-17) → 価格DataFrame の辞書を返す。"""
    return {n: load_index_series(sector17_to_index_code(n)) for n in range(1, 18)}


def main() -> int:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="TOPIX-17業種別指数の取得・保存")
    parser.add_argument("--start", default=None, help="省略時・--safe指定時は既存ファイルの最終日+1日から自動差分取得")
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--safe", action="store_true",
        help="ISSUE-001対策版（既存ファイルを読み込みappend→重複除去→保存・縮小時はfail-closed）を使う。新規実行では常にこちらを推奨",
    )
    parser.add_argument("--other", action="store_true", help="OTHER_INDEX_CODES（Growth250/REIT等）も併せて処理する")
    args = parser.parse_args()

    if args.safe:
        results = fetch_and_save_all_topix17_safe(args.end, start=args.start)
        if args.other:
            results += fetch_and_save_other_indices_safe(args.end, start=args.start)
        for r in results:
            print(f"  {r['code']} ({r['label']}): status={r['status']} existing={r['existing_rows']} "
                  f"new={r['new_rows']} combined={r['combined_rows']} range={r['date_min']}..{r['date_max']}")
        return 0

    if args.start is None:
        args.start = "2016-07-01"
    result = fetch_and_save_all_topix17(args.start, args.end)
    for code, n in result.items():
        sector_n = int(code, 16) - 0x80 + 1 if code != "0000" else 0
        name = "TOPIX" if code == "0000" else SECTOR17_NAMES.get(sector_n, "?")
        print(f"  {code} ({name}): {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
