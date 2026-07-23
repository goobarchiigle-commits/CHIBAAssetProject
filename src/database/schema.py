"""
src/database/schema.py
database/market/ 配下の全テーブルの列名・dtype種別・nullable可否・データソースを一元定義する
単一の翻訳表（metadata/schema.json はこのモジュールから生成する。手動同期しない）。

dtype_kind は緩い分類（"string"/"float"/"int"/"datetime"/"boolean"）とし、pandasの実dtype
（float32 vs float64等）まではここで固定しない（optimize_dtypes()の裁量に委ねる）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.database.exceptions import SchemaValidationError

# database/market スキーマバージョン（破壊的変更時のみ手動bump。bulk_ingest_state.parquet等の
# トレーサビリティ台帳に記録し、生成物がどのスキーマ世代のコードで作られたか追跡できるようにする）。
DATABASE_SCHEMA_VERSION = "1.1.0"
JQUANTS_API_VERSION = "v2"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype_kind: str  # "string" | "float" | "int" | "datetime" | "boolean"
    nullable: bool
    description: str
    source: str = "jquants_api"


@dataclass(frozen=True)
class TableSpec:
    table_name: str
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if not c.nullable]

    @property
    def all_columns(self) -> list[str]:
        return [c.name for c in self.columns]


# ── ohlcv/{year}.parquet ──────────────────────────────────────────────────
OHLCV_SCHEMA = TableSpec(
    table_name="ohlcv",
    columns=[
        ColumnSpec("Date", "datetime", False, "取引日"),
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("Open", "float", True, "始値（未調整）"),
        ColumnSpec("High", "float", True, "高値（未調整）"),
        ColumnSpec("Low", "float", True, "安値（未調整）"),
        ColumnSpec("Close", "float", True, "終値（未調整）"),
        ColumnSpec("Volume", "float", True, "出来高（未調整）"),
        ColumnSpec("AdjustmentFactor", "float", True, "調整係数"),
        ColumnSpec("AdjustmentOpen", "float", True, "調整済み始値"),
        ColumnSpec("AdjustmentHigh", "float", True, "調整済み高値"),
        ColumnSpec("AdjustmentLow", "float", True, "調整済み安値"),
        ColumnSpec("AdjustmentClose", "float", True, "調整済み終値"),
        ColumnSpec("AdjustmentVolume", "float", True, "調整済み出来高"),
        ColumnSpec("UpperLimit", "string", True, "値幅制限上限フラグ（J-Quants raw仕様上str型）"),
        ColumnSpec("LowerLimit", "string", True, "値幅制限下限フラグ（J-Quants raw仕様上str型）"),
        ColumnSpec("TurnoverValue", "float", True, "売買代金"),
    ],
)

# ── master/companies.parquet ──────────────────────────────────────────────
COMPANIES_SCHEMA = TableSpec(
    table_name="companies",
    columns=[
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("CompanyName", "string", True, "会社名（日本語）"),
        ColumnSpec("CompanyNameEnglish", "string", True, "会社名（英語）"),
        ColumnSpec("MarketCode", "string", True, "市場区分コード"),
        ColumnSpec("MarketCodeName", "string", True, "市場区分名"),
        ColumnSpec("Sector17Code", "string", True, "17業種コード"),
        ColumnSpec("Sector17CodeName", "string", True, "17業種名"),
        ColumnSpec("Sector33Code", "string", True, "33業種コード"),
        ColumnSpec("Sector33CodeName", "string", True, "33業種名"),
        ColumnSpec("ScaleCategory", "string", True, "規模区分（TOPIX Core30等）"),
        ColumnSpec("MarginCode", "string", True, "信用区分コード"),
        ColumnSpec("MarginCodeName", "string", True, "信用区分名"),
        ColumnSpec("ProductCategory", "string", True, "商品区分"),
        ColumnSpec("ListingDate", "datetime", True, "上場日（初出現日から推定）"),
        ColumnSpec("DelistingDate", "datetime", True, "上場廃止日（最終出現日・現存銘柄はNULL）"),
        ColumnSpec("IsCurrentlyListed", "boolean", True, "現在上場中か"),
    ],
)

# ── master/classifications.parquet ────────────────────────────────────────
CLASSIFICATIONS_SCHEMA = TableSpec(
    table_name="classifications",
    columns=[
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("MarketCode", "string", True, "市場区分コード"),
        ColumnSpec("MarketCodeName", "string", True, "市場区分名"),
        ColumnSpec("Sector17Code", "string", True, "17業種コード"),
        ColumnSpec("Sector17CodeName", "string", True, "17業種名"),
        ColumnSpec("Sector33Code", "string", True, "33業種コード"),
        ColumnSpec("Sector33CodeName", "string", True, "33業種名"),
        ColumnSpec("ScaleCategory", "string", True, "規模区分 raw値（TOPIX Core30/Large70/Mid400/Small 1/Small 2/-）"),
        ColumnSpec("IsTOPIXCore30", "boolean", True, "ScaleCategoryから導出"),
        ColumnSpec("IsTOPIXLarge70", "boolean", True, "ScaleCategoryから導出"),
        ColumnSpec("IsTOPIXMid400", "boolean", True, "ScaleCategoryから導出"),
        ColumnSpec("IsTOPIXSmall", "boolean", True, "ScaleCategoryから導出（Small 1/Small 2両方）"),
        ColumnSpec("topix_scale_source", "string", True, "TOPIX規模区分の取得元", source="jquants_api"),
        ColumnSpec("topix_scale_last_updated", "datetime", True, "TOPIX規模区分の最終更新日時"),
        ColumnSpec("IsJPXPrime150", "boolean", True, "JPXプライム150構成銘柄か（NULL=未取得/False=非採用確定）", source="jpx_official_csv"),
        ColumnSpec("jpx_prime150_source", "string", True, "取得元"),
        ColumnSpec("jpx_prime150_last_updated", "datetime", True, "最終更新日時"),
        ColumnSpec("IsJPX400", "boolean", True, "JPX日経400構成銘柄か（v1未実装のためNULL固定）", source=""),
        ColumnSpec("jpx400_source", "string", True, "取得元（v1未実装のためNULL）"),
        ColumnSpec("jpx400_last_updated", "datetime", True, "最終更新日時（v1未実装のためNULL）"),
        ColumnSpec("IsNikkei225", "boolean", True, "日経225構成銘柄か（データソース未確定のためNULL固定）", source=""),
        ColumnSpec("nikkei225_source", "string", True, "取得元（未確定のためNULL）"),
        ColumnSpec("nikkei225_last_updated", "datetime", True, "最終更新日時（未確定のためNULL）"),
    ],
)

# ── master/universe.parquet ───────────────────────────────────────────────
UNIVERSE_SCHEMA = TableSpec(
    table_name="universe",
    columns=[
        ColumnSpec("UniverseName", "string", False, "ユニバース名（例: TSE_ALL, RSR42）"),
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("FromDate", "datetime", False, "メンバーシップ開始日"),
        ColumnSpec("ToDate", "datetime", True, "メンバーシップ終了日（現在有効ならNULL）"),
        ColumnSpec("IsTradable", "boolean", True, "区間内で実売買可能だったか"),
        ColumnSpec("IsDelisted", "boolean", True, "Codeが最終的に上場廃止済みか"),
    ],
)

# ── minute/{year}/{yyyymm}.parquet（J-Quants Bulk API equities/bars/minute） ──
MINUTE_BARS_SCHEMA = TableSpec(
    table_name="minute_bars",
    columns=[
        ColumnSpec("Date", "datetime", False, "取引日"),
        ColumnSpec("Time", "string", False, "分足時刻（HH:MM・取引所ローカル）"),
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("Open", "float", True, "分足始値"),
        ColumnSpec("High", "float", True, "分足高値"),
        ColumnSpec("Low", "float", True, "分足安値"),
        ColumnSpec("Close", "float", True, "分足終値"),
        ColumnSpec("Volume", "float", True, "分足出来高"),
        ColumnSpec("TurnoverValue", "float", True, "分足売買代金"),
    ],
)

# ── tick/{year}/{yyyymm}.parquet（J-Quants Bulk API equities/trades） ────────
TICK_SCHEMA = TableSpec(
    table_name="tick",
    columns=[
        ColumnSpec("Date", "datetime", False, "取引日"),
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("Time", "string", False, "約定時刻（HH:MM:SS.ffffff・マイクロ秒精度・文字列で精度保持）"),
        ColumnSpec("SessionDistinction", "string", True, "セッション区分（前場/後場等の原コード）"),
        ColumnSpec("Price", "float", True, "約定価格"),
        ColumnSpec("TradingVolume", "float", True, "約定出来高"),
        ColumnSpec("TransactionId", "int", True, "約定ID（当日内での識別子）"),
    ],
)

# ── fundamentals/summary/{year}.parquet（J-Quants Bulk API fins/summary） ────
# 実測列数=約90列（決算短信サマリの全項目）。全列を手動定義せず、識別に必須の列のみ
# nullable=Falseで固定し、残りはBulk CSVの列をそのまま透過させる（schema.pyのvalidate_schemaは
# 未知の追加列をエラーにしない設計のため・数値実務項目の追加/変更に追随しやすくする狙い）。
FUNDAMENTALS_SUMMARY_SCHEMA = TableSpec(
    table_name="fundamentals_summary",
    columns=[
        ColumnSpec("DiscDate", "datetime", False, "開示日"),
        ColumnSpec("DiscTime", "string", True, "開示時刻"),
        ColumnSpec("Code", "string", False, "銘柄コード"),
        ColumnSpec("DiscNo", "string", False, "開示番号（一意識別子）"),
        ColumnSpec("DocType", "string", True, "開示書類種別"),
    ],
)

# ── master/indices.parquet ────────────────────────────────────────────────
INDICES_SCHEMA = TableSpec(
    table_name="indices",
    columns=[
        ColumnSpec("IndexCode", "string", False, "指数コード"),
        ColumnSpec("IndexName", "string", True, "指数名"),
        ColumnSpec("Category", "string", True, "指数カテゴリ（TOPIX/TOPIX_SECTOR/REIT/GROWTH/JPX_MARKET_SEGMENT/JPX等）"),
        ColumnSpec("Provider", "string", True, "算出主体（JPX/日本経済新聞社等）"),
        ColumnSpec("AssetClass", "string", True, "資産クラス"),
        ColumnSpec("Description", "string", True, "説明"),
        ColumnSpec("ConstituentSource", "string", True, "構成銘柄データの取得元"),
        ColumnSpec("UpdateFrequency", "string", True, "更新頻度"),
        ColumnSpec("FirstDate", "datetime", True, "価格データ開始日"),
        ColumnSpec("LastDate", "datetime", True, "価格データ最終日"),
        ColumnSpec("Remarks", "string", True, "備考"),
    ],
)

TABLE_SCHEMAS: dict[str, TableSpec] = {
    "ohlcv": OHLCV_SCHEMA,
    "companies": COMPANIES_SCHEMA,
    "classifications": CLASSIFICATIONS_SCHEMA,
    "universe": UNIVERSE_SCHEMA,
    "indices": INDICES_SCHEMA,
    "minute_bars": MINUTE_BARS_SCHEMA,
    "tick": TICK_SCHEMA,
    "fundamentals_summary": FUNDAMENTALS_SUMMARY_SCHEMA,
}


def validate_schema(df, table_name: str) -> None:
    """
    df の列が table_name のスキーマ定義と整合するか検証する。
    必須列（nullable=False）の欠落があれば SchemaValidationError を送出する（fail-closed）。
    未知の追加列はエラーにしない（将来の列拡張を許容する設計）。
    """
    if table_name not in TABLE_SCHEMAS:
        raise SchemaValidationError(f"未知のテーブル名: {table_name}")
    spec = TABLE_SCHEMAS[table_name]
    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"{table_name}: 必須列が欠落しています: {missing}")


def to_schema_json() -> dict:
    """schema.json 出力用の辞書表現を生成する。"""
    return {
        table_name: {
            "columns": [
                {
                    "name": c.name,
                    "dtype_kind": c.dtype_kind,
                    "nullable": c.nullable,
                    "description": c.description,
                    "source": c.source,
                }
                for c in spec.columns
            ]
        }
        for table_name, spec in TABLE_SCHEMAS.items()
    }
