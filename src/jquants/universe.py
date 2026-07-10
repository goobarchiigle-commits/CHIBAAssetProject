"""
src/jquants/universe.py
上場銘柄ユニバースのイベントソーシング復元（上場廃止銘柄を含む全銘柄コードの発見）。

設計:
  正本は data/jquants/metadata/universe_events.parquet（ADD/REMOVEイベントログ・追記専用）。
  「任意日時点のUniverse」は reconstruct_universe_asof() でイベントをリプレイして求める
  （読み出し側はイベントログを都度再生するだけで、別途「現在状態」を二重管理しない）。

Option A vs B 評価（2026-07-10・ASK_FIRST③着手前）:
  A（listed/master日次ポーリング・旧正本）: rebuild_universe_events()。
    営業日ごとに /v2/equities/master を叩き前日との差分を取る。専用リクエストが
    対象期間の全営業日数だけ必要（約2,440件）。会社名・セクター等の記述情報を
    ADD時点で同時取得できる利点があるが、Study75の価格ダウンロード（Strategy C・
    study75_downloader.py）と完全に重複した営業日ループを二重に持つ非効率がある。
  B（daily bars日次スナップショット由来・現行正本）: rebuild_universe_events_from_daily_bars() /
    rebuild_universe_events_from_staged_bars()。study75_downloader.pyが価格取得のために
    既に取得済みの /v2/equities/bars/daily?date=X 応答のCode列からUniverse ADD/REMOVEを導出する。
    追加の専用リクエストが不要（価格ダウンロードと同時なら限界コストゼロ）。さらに
    rebuild_universe_events_from_staged_bars() はローカルにステージング済みの
    cache/daily/day_*.parquet だけから完全オフラインで再構築でき、何度でも
    決定論的に再現できる（listed/masterという外部APIの応答が将来変わっても影響を受けない）。
    比較の詳細は reports/study75_universe_event_source_comparison.md 参照。
  結論: Bがリクエスト数・再現性で優位（Aの詳細な会社情報は日次では持てないという
  トレードオフはあるが、それはenrich_universe_reference_with_listed_info()で
  「コード単位・1回だけ」補う設計にすることで解消する）→ B を正本として採用。
  A（rebuild_universe_events）はレガシー・比較用として残置（削除しない）。

  営業日判定は src/market/jpx_calendar.JPXCalendar を再利用し、休日にAPIを叩かない。
  初回フル復元はチェックポイント（metadata/universe_reconstruction_state.json に
  最終処理済み営業日を保持）により中断・再開が安全にできる設計。以後の更新は
  sync_latest_snapshot_diff()（A方式）または日次ダウンロードに同梱する形（B方式）で足りる。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.jquants import study75_downloader
from src.jquants.provider import JQuantsProvider, raw_records_to_frame
from src.market.jpx_calendar import JPXCalendar
from src.paths import JQUANTS_METADATA_DIR, JQUANTS_PROCESSED_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

EVENTS_FILE: Path = JQUANTS_METADATA_DIR / "universe_events.parquet"
STATE_FILE: Path = JQUANTS_METADATA_DIR / "universe_reconstruction_state.json"
UNIVERSE_REFERENCE_FILE: Path = JQUANTS_PROCESSED_DIR / "universe.parquet"

EVENT_COLUMNS = [
    "event_date", "code", "event_type",
    "company_name", "sector_33_code", "sector_33_name",
    "market_code", "market_code_name",
]

# J-Quants API v2 /v2/equities/master の略記フィールド名（2026-07-09実測確認済み・src/jquants/schema.py参照）。
_INFO_FIELDS = {
    "CoName": "company_name",
    "S33":    "sector_33_code",
    "S33Nm":  "sector_33_name",
    "Mkt":    "market_code",
    "MktNm":  "market_code_name",
}

DEFAULT_FLUSH_INTERVAL_DAYS = 20


# ────────────────────────────────────────────────────────────────────────── #
# スナップショット取得
# ────────────────────────────────────────────────────────────────────────── #
def fetch_listed_snapshot(provider: JQuantsProvider, date: str = "") -> dict[str, dict]:
    """指定日（省略時は最新）の上場銘柄一覧を {code: info_dict} で返す。"""
    records = provider.listed_info(date=date)
    snapshot: dict[str, dict] = {}
    for rec in records:
        code = str(rec.get("Code", "")).strip()
        if not code:
            continue
        snapshot[code] = {dst: rec.get(src, "") for src, dst in _INFO_FIELDS.items()}
    return snapshot


# ────────────────────────────────────────────────────────────────────────── #
# イベントログ I/O
# ────────────────────────────────────────────────────────────────────────── #
def load_universe_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_parquet(EVENTS_FILE)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def save_universe_events(df: pd.DataFrame) -> None:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["event_date", "code", "event_type"])
    df = df.sort_values(["event_date", "code"]).reset_index(drop=True)
    df.to_parquet(EVENTS_FILE)


def append_universe_events(new_events: list[dict]) -> None:
    if not new_events:
        return
    existing = load_universe_events()
    combined = pd.concat([existing, pd.DataFrame(new_events, columns=EVENT_COLUMNS)], ignore_index=True)
    save_universe_events(combined)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[JQUANTS_UNIVERSE] state読込失敗（%s）→ 空状態から再開", e)
        return {}


def _save_state(state: dict) -> None:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────── #
# リプレイ（任意日時点のUniverse復元）
# ────────────────────────────────────────────────────────────────────────── #
def reconstruct_universe_asof(date: str | pd.Timestamp) -> pd.DataFrame:
    """
    universe_events.parquet を date までリプレイし、その時点で上場中のコード一覧を返す。
    列: code, company_name, sector_33_code, sector_33_name, market_code, market_code_name
    （各コードの最終イベントが ADD であるものだけを残す）。
    """
    events = load_universe_events()
    if events.empty:
        return pd.DataFrame(columns=["code"] + list(_INFO_FIELDS.values()))

    cutoff = pd.Timestamp(date)
    subset = events.loc[events["event_date"] <= cutoff].sort_values(["code", "event_date"])
    if subset.empty:
        return pd.DataFrame(columns=["code"] + list(_INFO_FIELDS.values()))

    last_per_code = subset.groupby("code", as_index=False).last()
    listed = last_per_code.loc[last_per_code["event_type"] == "ADD"]
    return listed.drop(columns=["event_type"]).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────── #
# フル復元（初回・チェックポイント/再開対応）
# ────────────────────────────────────────────────────────────────────────── #
def _diff_events(event_date: pd.Timestamp, prev_codes: set[str], curr_snapshot: dict[str, dict],
                  known_info: dict[str, dict]) -> list[dict]:
    curr_codes = set(curr_snapshot.keys())
    events: list[dict] = []

    for code in sorted(curr_codes - prev_codes):
        info = curr_snapshot[code]
        events.append({"event_date": event_date, "code": code, "event_type": "ADD", **info})

    for code in sorted(prev_codes - curr_codes):
        info = known_info.get(code, {dst: "" for dst in _INFO_FIELDS.values()})
        events.append({"event_date": event_date, "code": code, "event_type": "REMOVE", **info})

    return events


def rebuild_universe_events(
    provider: JQuantsProvider,
    start: str,
    end: str,
    flush_interval_days: int = DEFAULT_FLUSH_INTERVAL_DAYS,
    force: bool = False,
) -> dict:
    """
    [LEGACY・Option A] listed/master日次ポーリング版。2026-07-10のOption B採用決定により
    正本ではなくなった（rebuild_universe_events_from_staged_bars が正本）。比較・検証用に
    削除せず残置。会社情報をADD時点で同時取得できる点はB方式に対する唯一の利点。

    start〜end の全営業日を1日ずつ処理し、ADD/REMOVEイベントを復元する。
    中断しても metadata/universe_reconstruction_state.json のチェックポイントから再開できる。
    force=True で最初からやり直す（既存イベントログ・状態を破棄）。

    Returns: {"processed_days": int, "events_written": int, "last_processed_date": str}
    """
    calendar = JPXCalendar()
    trading_days = [
        d for d in pd.bdate_range(start, end) if calendar.is_trading_day(d)
    ]
    if not trading_days:
        return {"processed_days": 0, "events_written": 0, "last_processed_date": None}

    if force:
        save_universe_events(pd.DataFrame(columns=EVENT_COLUMNS))
        _save_state({})

    state = _load_state()
    resume_from = state.get("last_processed_date")

    if resume_from:
        prev_codes = set(reconstruct_universe_asof(resume_from)["code"])
        remaining_days = [d for d in trading_days if d > pd.Timestamp(resume_from)]
    else:
        prev_codes = set()
        remaining_days = trading_days

    known_info: dict[str, dict] = {
        row["code"]: {k: row[k] for k in _INFO_FIELDS.values()}
        for _, row in reconstruct_universe_asof(resume_from or trading_days[0]).iterrows()
    } if resume_from else {}

    buffer: list[dict] = []
    events_written = 0
    last_flushed_date: pd.Timestamp | None = None

    for i, day in enumerate(remaining_days, start=1):
        day_str = day.strftime("%Y-%m-%d")
        snapshot = fetch_listed_snapshot(provider, date=day_str)
        buffer.extend(_diff_events(day, prev_codes, snapshot, known_info))

        for code, info in snapshot.items():
            known_info[code] = info
        prev_codes = set(snapshot.keys())

        is_flush_point = (i % flush_interval_days == 0) or (i == len(remaining_days))
        if is_flush_point:
            append_universe_events(buffer)
            events_written += len(buffer)
            buffer = []
            last_flushed_date = day
            _save_state({"last_processed_date": day_str})
            logger.info(
                "[JQUANTS_UNIVERSE] progress=%d/%d date=%s cumulative_codes=%d",
                i, len(remaining_days), day_str, len(prev_codes),
            )

    return {
        "processed_days": len(remaining_days),
        "events_written": events_written,
        "last_processed_date": last_flushed_date.strftime("%Y-%m-%d") if last_flushed_date is not None else resume_from,
    }


# ────────────────────────────────────────────────────────────────────────── #
# Option B（daily bars由来のUniverse復元）— 2026-07-10採用・正本
# ────────────────────────────────────────────────────────────────────────── #
_EMPTY_INFO: dict[str, str] = {dst: "" for dst in _INFO_FIELDS.values()}


def derive_codes_from_daily_bars(day_df: pd.DataFrame) -> dict[str, dict]:
    """
    1営業日ぶんの日次OHLCVデータ（study75_downloader.fetch_and_stage_day の返り値、または
    provider.daily_bars_for_date の応答を raw_records_to_frame() した形）から、その日
    取引されたコード集合を抽出する。listed/master への追加リクエストは不要
    （価格取得のために既に取得済みのデータをそのまま再利用する・Option Bの核心）。
    会社名・セクター等の記述情報は持たない（空文字で埋める）→
    enrich_universe_reference_with_listed_info() で別途・コード単位に補完する。
    """
    if day_df.empty or "Code" not in day_df.columns:
        return {}
    codes = day_df["Code"].astype(str).unique()
    return {code: dict(_EMPTY_INFO) for code in codes}


def rebuild_universe_events_from_daily_bars(
    provider: JQuantsProvider,
    start: str,
    end: str,
    flush_interval_days: int = DEFAULT_FLUSH_INTERVAL_DAYS,
    force: bool = False,
) -> dict:
    """
    [Option B・ライブAPI版] daily bars（/v2/equities/bars/daily?date=X）を日毎に取得して
    Universe ADD/REMOVEを導出する。study75_downloader.download_full_market_by_day と
    同じAPI呼び出しパターンのため、価格ダウンロードと同時に実行すれば実質的に
    追加コストゼロになる（別々に実行する場合はrebuild_universe_events(A)と同じ
    リクエスト数がかかる点に注意 — オフラインで済ませたい場合は
    rebuild_universe_events_from_staged_bars() を使うこと）。

    Returns: {"processed_days": int, "events_written": int, "last_processed_date": str}
    """
    calendar = JPXCalendar()
    trading_days = [d for d in pd.bdate_range(start, end) if calendar.is_trading_day(d)]
    if not trading_days:
        return {"processed_days": 0, "events_written": 0, "last_processed_date": None}

    if force:
        save_universe_events(pd.DataFrame(columns=EVENT_COLUMNS))
        _save_state({})

    state = _load_state()
    resume_from = state.get("last_processed_date")
    if resume_from:
        prev_codes = set(reconstruct_universe_asof(resume_from)["code"])
        remaining_days = [d for d in trading_days if d > pd.Timestamp(resume_from)]
    else:
        prev_codes = set()
        remaining_days = trading_days

    buffer: list[dict] = []
    events_written = 0
    last_flushed_date: pd.Timestamp | None = None

    for i, day in enumerate(remaining_days, start=1):
        day_str = day.strftime("%Y-%m-%d")
        records = provider.daily_bars_for_date(day_str.replace("-", ""))
        day_df = raw_records_to_frame(records)
        snapshot = derive_codes_from_daily_bars(day_df)

        buffer.extend(_diff_events(day, prev_codes, snapshot, {}))
        prev_codes = set(snapshot.keys())

        is_flush_point = (i % flush_interval_days == 0) or (i == len(remaining_days))
        if is_flush_point:
            append_universe_events(buffer)
            events_written += len(buffer)
            buffer = []
            last_flushed_date = day
            _save_state({"last_processed_date": day_str})
            logger.info(
                "[JQUANTS_UNIVERSE] (option-B live) progress=%d/%d date=%s cumulative_codes=%d",
                i, len(remaining_days), day_str, len(prev_codes),
            )

    return {
        "processed_days": len(remaining_days),
        "events_written": events_written,
        "last_processed_date": last_flushed_date.strftime("%Y-%m-%d") if last_flushed_date is not None else resume_from,
    }


def rebuild_universe_events_from_staged_bars(
    start: str | None = None,
    end: str | None = None,
    flush_interval_days: int = DEFAULT_FLUSH_INTERVAL_DAYS,
    force: bool = False,
) -> dict:
    """
    [Option B・完全オフライン版・正本] 既にステージング済み・検証済みの
    data/jquants/cache/daily/day_YYYY-MM-DD.parquet からADD/REMOVEイベントを導出する。
    API通信ゼロ。study75_downloader.download_full_market_by_day() 実行後ならいつでも
    決定論的に再実行できる（Option Bの再現性の核心 — listed/masterという外部APIへの
    依存を完全に断ち切る）。

    「検証済み」の判定は study75_downloader.daily_completed_dates.json への記録有無を使う
    （cache/daily/ に物理ファイルがあるだけでは不十分 — 0行等でvalidate_staged_dayがerror
    判定した日はそこに記録されないため、誤って「その日は全銘柄が消えた」と解釈しない）。

    start/end 省略時はステージング済みデータの全期間が対象。
    ステージング未済の営業日に到達したら、そこで処理を打ち切る（連続性のないギャップを
    挟むと以降のADD/REMOVE判定が不正確になるため）。次回、当該日がステージングされてから
    再実行すれば続きから再開する。

    Returns: {"processed_days": int, "stopped_at_missing_day": str | None,
              "events_written": int, "last_processed_date": str | None}
    """
    completed = study75_downloader.load_completed_dates()
    if not completed:
        return {"processed_days": 0, "stopped_at_missing_day": None, "events_written": 0, "last_processed_date": None}

    available_dates = sorted(pd.Timestamp(d) for d in completed.keys())
    calendar = JPXCalendar()
    range_start = pd.Timestamp(start) if start else available_dates[0]
    range_end = pd.Timestamp(end) if end else available_dates[-1]
    trading_days = [d for d in pd.bdate_range(range_start, range_end) if calendar.is_trading_day(d)]

    if force:
        save_universe_events(pd.DataFrame(columns=EVENT_COLUMNS))
        _save_state({})

    state = _load_state()
    resume_from = state.get("last_processed_date")
    if resume_from:
        prev_codes = set(reconstruct_universe_asof(resume_from)["code"])
        remaining_days = [d for d in trading_days if d > pd.Timestamp(resume_from)]
    else:
        prev_codes = set()
        remaining_days = trading_days

    buffer: list[dict] = []
    events_written = 0
    processed = 0
    stopped_at_missing_day: str | None = None
    last_flushed_date: pd.Timestamp | None = None
    last_processed_day: pd.Timestamp | None = None
    available_set = set(d.strftime("%Y-%m-%d") for d in available_dates)

    for i, day in enumerate(remaining_days, start=1):
        day_str = day.strftime("%Y-%m-%d")
        if day_str not in available_set:
            stopped_at_missing_day = day_str
            break

        day_df = pd.read_parquet(study75_downloader.daily_staging_path(day_str))
        snapshot = derive_codes_from_daily_bars(day_df)
        buffer.extend(_diff_events(day, prev_codes, snapshot, {}))
        prev_codes = set(snapshot.keys())
        processed += 1
        last_processed_day = day

        is_flush_point = (processed % flush_interval_days == 0) or (i == len(remaining_days))
        if is_flush_point:
            append_universe_events(buffer)
            events_written += len(buffer)
            buffer = []
            last_flushed_date = day
            _save_state({"last_processed_date": day_str})
            logger.info(
                "[JQUANTS_UNIVERSE] (option-B offline) progress=%d date=%s cumulative_codes=%d",
                processed, day_str, len(prev_codes),
            )

    # ギャップで早期break（またはその他の理由）でループを抜けた際、flush_interval_days境界に
    # 満たない未flushの buffer が残っていれば、失わずに書き出す（さもないと最大flush_interval_days-1日分の
    # 導出済みイベントが握りつぶされ、次回resumeで再計算が必要になる＝非効率だが実害はdedupで防げる。
    # とはいえ正しくは毎回確実にflushすべき）。
    if buffer:
        append_universe_events(buffer)
        events_written += len(buffer)
        last_flushed_date = last_processed_day
        _save_state({"last_processed_date": last_processed_day.strftime("%Y-%m-%d")})
        logger.info(
            "[JQUANTS_UNIVERSE] (option-B offline) 終端flush date=%s",
            last_processed_day.strftime("%Y-%m-%d"),
        )

    return {
        "processed_days": processed,
        "stopped_at_missing_day": stopped_at_missing_day,
        "events_written": events_written,
        "last_processed_date": last_flushed_date.strftime("%Y-%m-%d") if last_flushed_date is not None else resume_from,
    }


def enrich_universe_reference_with_listed_info(
    provider: JQuantsProvider,
    codes: list[str] | None = None,
    only_missing: bool = True,
) -> pd.DataFrame:
    """
    [listed/master = メタデータ補完専用・Option B採用に伴う役割変更]
    Option Bで導出したUniverseイベント（ADD/REMOVEのみ・会社情報を持たない）に対し、
    processed/universe.parquet の記述列（会社名・セクター・市場区分）を
    「コード単位で1回だけ」補完する。日次スナップショットではなく銘柄単位の
    ピンポイント参照のため、Aのような全営業日リクエストは発生しない
    （codes省略時は現在のUniverse参照テーブルの全コード数だけ・最大でも銘柄数分のリクエスト）。

    only_missing=True（既定）: company_name が未設定（空文字）のコードのみ問い合わせる
    （2回目以降の実行は新規コードのみが対象になり、リクエスト数がさらに減る）。

    イベントログ（universe_events.parquet）自体は書き換えない（ADD/REMOVEの事実のみを保持する
    という設計原則を維持）。更新するのはマテリアライズドビュー（processed/universe.parquet）のみ。

    Returns: 更新後の参照テーブル（processed/universe.parquet に保存済み）。
    """
    ref = pd.read_parquet(UNIVERSE_REFERENCE_FILE) if UNIVERSE_REFERENCE_FILE.exists() else materialize_universe_reference()

    if codes is not None:
        target_codes = codes
    elif only_missing:
        target_codes = ref.loc[ref["company_name"] == "", "code"].tolist()
    else:
        target_codes = ref["code"].tolist()

    for code in target_codes:
        records = provider.listed_info(code=code)
        if not records:
            continue
        rec = records[-1]  # 最新レコードを採用
        info = {dst: str(rec.get(src, "")) for src, dst in _INFO_FIELDS.items()}
        mask = ref["code"] == code
        for col, val in info.items():
            ref.loc[mask, col] = val

    JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(UNIVERSE_REFERENCE_FILE)
    return ref


def sync_latest_snapshot_diff(provider: JQuantsProvider) -> dict:
    """
    以後の日次/週次更新用: 最新スナップショット1回のみ取得し、直近イベント状態との差分だけ追記する。
    フル営業日走査はしない。
    """
    state = _load_state()
    resume_from = state.get("last_processed_date")
    events = load_universe_events()

    if resume_from:
        prev_codes = set(reconstruct_universe_asof(resume_from)["code"])
        prev_ref = reconstruct_universe_asof(resume_from)
        known_info = {row["code"]: {k: row[k] for k in _INFO_FIELDS.values()} for _, row in prev_ref.iterrows()}
    elif not events.empty:
        latest_known = events["event_date"].max()
        prev_codes = set(reconstruct_universe_asof(latest_known)["code"])
        prev_ref = reconstruct_universe_asof(latest_known)
        known_info = {row["code"]: {k: row[k] for k in _INFO_FIELDS.values()} for _, row in prev_ref.iterrows()}
    else:
        prev_codes = set()
        known_info = {}

    today = pd.Timestamp.now(tz=_JST).normalize().tz_localize(None)
    snapshot = fetch_listed_snapshot(provider, date="")
    new_events = _diff_events(today, prev_codes, snapshot, known_info)
    append_universe_events(new_events)
    _save_state({"last_processed_date": today.strftime("%Y-%m-%d")})

    return {"events_written": len(new_events), "as_of": today.strftime("%Y-%m-%d")}


# ────────────────────────────────────────────────────────────────────────── #
# processed/universe.parquet マテリアライズドビュー
# ────────────────────────────────────────────────────────────────────────── #
def materialize_universe_reference() -> pd.DataFrame:
    """
    universe_events.parquet から派生する現在形の参照テーブルを
    processed/universe.parquet に書き出す（再生成可能なキャッシュ・イベントログが正本）。
    列: code, company_name, sector_33_code, sector_33_name, market_code, market_code_name,
        first_seen_date, last_seen_date, is_currently_listed
    """
    events = load_universe_events()
    if events.empty:
        empty = pd.DataFrame(columns=[
            "code", *_INFO_FIELDS.values(), "first_seen_date", "last_seen_date", "is_currently_listed",
        ])
        JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(UNIVERSE_REFERENCE_FILE)
        return empty

    events = events.sort_values(["code", "event_date"])
    rows = []
    for code, grp in events.groupby("code"):
        adds = grp.loc[grp["event_type"] == "ADD"]
        first_seen = adds["event_date"].min() if not adds.empty else pd.NaT
        last_row = grp.iloc[-1]
        is_listed = bool(last_row["event_type"] == "ADD")
        last_seen = grp["event_date"].max()
        info_source = last_row if last_row["company_name"] else adds.iloc[-1] if not adds.empty else last_row
        rows.append({
            "code": code,
            "company_name": info_source.get("company_name", ""),
            "sector_33_code": info_source.get("sector_33_code", ""),
            "sector_33_name": info_source.get("sector_33_name", ""),
            "market_code": info_source.get("market_code", ""),
            "market_code_name": info_source.get("market_code_name", ""),
            "first_seen_date": first_seen,
            "last_seen_date": last_seen,
            "is_currently_listed": is_listed,
        })

    ref = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(UNIVERSE_REFERENCE_FILE)
    return ref
