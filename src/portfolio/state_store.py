"""
src/portfolio/state_store.py
portfolio_state.json への読み書きの唯一の経路。

RULE:
  禁止: json.dump / write_text / open("w") on portfolio_state.json
  代替: save_portfolio_state() / commit_broker_snapshot() を必ず使う。

  禁止: state["available_cash"] だけ更新
  禁止: state["position_qtys"] だけ更新
  必須: broker snapshot 全取得成功時のみ commit_broker_snapshot() で一括 commit

atomic write: tmp (同一 volume) → flush → fsync → os.replace()
  Windows では同一ドライブ内の os.replace() は原子的。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_BASE_DIR                  = Path(os.environ.get("AI_TRADING_HOME", str(Path(__file__).resolve().parents[2])))
_DEFAULT_PATH              = _BASE_DIR / "runtime" / "portfolio_state.json"
_BACKUP_DIR                = _BASE_DIR / "runtime" / "state_backups"
_RECON_LOG_DIR             = _BASE_DIR / "logs"    / "reconciliation"
_GENERATION_WATERMARK_PATH = _BASE_DIR / "runtime" / "state_generation.txt"

SCHEMA_VERSION      = 3
STALE_WARN_SECONDS  = 900    # 15 分
STALE_HARD_SECONDS  = 3600   # 60 分
INITIAL_CAPITAL     = 3_000_000.0


# ── Schema 定義 ──────────────────────────────────────────────────────────────

SCHEMA_V2_REQUIRED: frozenset[str] = frozenset({
    "schema_version", "updated_at", "data_source",
    "cb_state", "equity_peak", "available_cash",
    "position_entry_dates", "position_entry_prices",
    "position_qtys", "positions_count", "last_equity",
    "generation_id",
})

V2_SAFE_DEFAULTS: dict[str, Any] = {
    "schema_version":           SCHEMA_VERSION,
    "updated_at":               None,
    "data_source":              "unknown",
    "cb_state":                 "NORMAL",
    "equity_peak":              INITIAL_CAPITAL,
    "available_cash":           0.0,
    "position_entry_dates":     {},
    "position_entry_prices":    {},
    "position_qtys":            {},
    "positions_count":          0,
    "last_equity":              0.0,
    "safe_warn_count":          0,
    "cb_cooldown_end_date":     None,
    "recovery_threshold":       None,
    # EQUITY_PEAK_HARDENING (2026-07-03): +10%以上のpeakジャンプは即時採用せず
    # ここへ一時格納し、翌営業日の再確認後に確定する。
    # {"value": float, "staged_date": "YYYY-MM-DD", "reason": str,
    #  "current_equity_at_stage": float} | None
    "candidate_peak":           None,
    "position_fill_dates":      {},
    "position_entry_atrs":      {},
    "position_highest_closes":  {},
    "position_current_prices":  {},
    "position_unrealized_pnl":  {},
    "position_unrealized_pct":  {},
    "position_strategy_types":  {},
    "shadow_virtual_positions": {},
    "shadow_positions":         {},
    "reentry_blocked":          {},
    "last_updated":             None,
    # v2.1 additions
    "generation_id":            0,
    "snapshot_hash":            None,
    "snapshot_ts":              None,
    "snapshot_avg_costs":       {},   # broker avg costs at last commit (for hash recomputation)
    # v3 addition — Quality Replacement Engine (Study57/58A)
    "position_entry_rsrs":      {},   # {sym: rsr_at_buy} populated by update_state_after_execution
    # GHOST_POSITION_FIX (2026-07-03): {sym: consecutive_missing_run_count}
    "position_missing_streak":  {},
}


# ── 資産系フィールドの読み取りガード (Broker-as-Sole-SSOT, 2026-07-19) ──────────
# load_portfolio_state() は state_store.py の宣言された公開読み取りAPIである。
# その戻り値からcash/positions/market_value/equityの"現在値"を読み出せると、
# 呼び出し元が誤ってこれを資産計算・売買判断へ使ってしまう
# （2026-07-15〜17 equity_peak異常値インシデントの根本原因）。
# ここでは load_portfolio_state() の戻り値を _GuardedStateDict でラップし、
# 許可モジュール以外からの読み取りを RuntimeError で即座に検知する。
#
# 注意: SignalBridge._load_portfolio_state() は json.loads() を直接呼ぶ独自経路
# であり、この場合 state は plain dict のまま（本ガードは適用されない）。
# signal_bridge.py内の残存する資産系フィールド読み取りは診断ログ専用であることを
# 個別レビュー済み（scripts/check_no_state_asset_read.py の ALLOWED_FILES 参照）。
_ASSET_VALUE_KEYS = frozenset({
    "available_cash", "last_equity", "position_qtys", "positions_count",
    "position_current_prices", "snapshot_avg_costs",
})
_ASSET_VALUE_ALLOWED_MODULES = frozenset({
    "src.portfolio.state_store",
    "src.portfolio.equity",
    "src.portfolio.broker_source",
    # Broker-as-Sole-SSOTリファクタ (2026-07-18/19) で fetch_broker_snapshot()
    # 経由へ移行済み。残る読み取りは診断ログ・CB決済ラグ補償（Step8で再設計予定）・
    # 直前state比較によるクリーンアップ用途など、個別に検証済み
    # （scripts/check_no_state_asset_read.py の ALLOWED_FILES と同一の集合）。
    "src.kabusapi.signal_bridge",
    "src.startup_check",
    "src.scripts.sync_positions",
    "src.run_live_signal",
})


class AssetFieldReadForbiddenError(RuntimeError):
    """load_portfolio_state() の戻り値から資産系フィールドを許可外モジュールが
    読み取ろうとした場合の異常。cash/positions/market_value/equityは
    fetch_broker_snapshot() 経由のBrokerSnapshotのみを入力とすること。"""


def _guarded_caller_module() -> str:
    frame = sys._getframe(2)  # caller of get()/__getitem__ (skip this helper + the dict method)
    return frame.f_globals.get("__name__", "")


def _is_guard_exempt_module(module_name: str) -> bool:
    """テストモジュールは資産系フィールドを自由に検証できる（本番コードではない）。
    src.foo.test_bar / foo.test_bar / foo.bar_test いずれの import 経路でも
    module_name の末尾コンポーネントで判定する（sys.path構成によりsrc.接頭辞の
    有無が変わるため）。"""
    last = module_name.rsplit(".", 1)[-1]
    return last.startswith("test_") or last.endswith("_test")


class _GuardedStateDict(dict):
    """load_portfolio_state() が返す state dict のラッパー。
    equity_peak / cb_state / candidate_peak 等の状態管理フィールドは無制限に
    読み取れるが、_ASSET_VALUE_KEYS は _ASSET_VALUE_ALLOWED_MODULES 以外からの
    読み取りを禁止する。"""

    def __getitem__(self, key):
        if key in _ASSET_VALUE_KEYS:
            caller = _guarded_caller_module()
            if caller not in _ASSET_VALUE_ALLOWED_MODULES and not _is_guard_exempt_module(caller):
                raise AssetFieldReadForbiddenError(
                    f"'{key}' はload_portfolio_state()経由で読み取れません "
                    f"(呼び出し元: {caller or 'unknown'})。"
                    "fetch_broker_snapshot()経由のBrokerSnapshotを使用してください。"
                )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in _ASSET_VALUE_KEYS:
            caller = _guarded_caller_module()
            if caller not in _ASSET_VALUE_ALLOWED_MODULES and not _is_guard_exempt_module(caller):
                raise AssetFieldReadForbiddenError(
                    f"'{key}' はload_portfolio_state()経由で読み取れません "
                    f"(呼び出し元: {caller or 'unknown'})。"
                    "fetch_broker_snapshot()経由のBrokerSnapshotを使用してください。"
                )
        return super().get(key, default)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok:                   bool
    hard_fails:           list[str] = field(default_factory=list)
    warnings:             list[str] = field(default_factory=list)
    healed:               list[str] = field(default_factory=list)
    snapshot_age_seconds: float | None = None

    @property
    def is_stale(self) -> bool:
        return (self.snapshot_age_seconds or 0) > STALE_WARN_SECONDS

    @property
    def is_very_stale(self) -> bool:
        return (self.snapshot_age_seconds or 0) > STALE_HARD_SECONDS


@dataclass
class BrokerSnapshot:
    """
    broker API から取得した完全スナップショット。

    commit_broker_snapshot() に渡す前に validate_broker_snapshot() でバリデーションされる。
    partial な broker データ（cash のみ, positions のみ）での commit は禁止。
    cash / positions / avg_costs / market_values がすべて揃っている場合のみ生成すること。
    """
    cash:           float
    positions:      dict[str, int]         # symbol → qty (live, qty > 0 のみ)
    avg_costs:      dict[str, float]       # symbol → avg_price (取得単価)
    market_values:  dict[str, float]       # symbol → current_price (時価)
    equity:         float                  # compute_live_equity() 呼出後に設定
    ts:             str                    # ISO8601+TZ
    source:         str                    # "broker" | "broker_error" | "virtual"
    api_health:     dict                   # {"positions_ok": bool, "wallet_ok": bool}
    generation_id:  int   = 0             # from portfolio_state generation_id at snapshot time
    exposure:       float = 0.0           # sum(qty * market_price) / equity
    snapshot_hash:  str   = ""            # sha256[:16] computed after commit


class SnapshotValidationError(ValueError):
    """BrokerSnapshot のバリデーション失敗。commit_broker_snapshot() が raise する。"""


# ── Generation watermark ──────────────────────────────────────────────────────

def _read_generation_watermark() -> int:
    """runtime/state_generation.txt から generation high-watermark を読む。"""
    try:
        return int(_GENERATION_WATERMARK_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def _write_generation_watermark(gen_id: int) -> None:
    try:
        _GENERATION_WATERMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GENERATION_WATERMARK_PATH.write_text(str(gen_id), encoding="utf-8")
    except Exception as e:
        logger.warning("[STATE] generation watermark write failed: %s", e)


# ── Snapshot hash ─────────────────────────────────────────────────────────────

def _compute_snapshot_hash(snapshot: BrokerSnapshot) -> str:
    """
    sha256(cash + sorted(symbols/qtys/costs)) の先頭 16 文字。

    state ファイルの integrity check と rollback 検知に使用。
    avg_costs を含めることで positions + cash のセットを一意に識別する。
    """
    symbols = sorted(snapshot.positions.keys())
    payload = json.dumps(
        {
            "cash":    round(snapshot.cash, 0),
            "symbols": symbols,
            "qtys":    [int(snapshot.positions[s]) for s in symbols],
            "costs":   [round(float(snapshot.avg_costs.get(s, 0.0)), 4) for s in symbols],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _recompute_hash_from_state(state: dict) -> str:
    """
    state dict の現在フィールドから snapshot_hash を再計算する。

    validate_state() 内で stored hash との整合性チェックに使用。
    snapshot_avg_costs を使うことで position_entry_prices との混同を避ける。
    """
    symbols = sorted(state.get("position_qtys", {}).keys())
    costs   = state.get("snapshot_avg_costs", {})
    payload = json.dumps(
        {
            "cash":    round(float(state.get("available_cash", 0)), 0),
            "symbols": symbols,
            "qtys":    [int(state.get("position_qtys", {})[s]) for s in symbols],
            "costs":   [round(float(costs.get(s, 0.0)), 4) for s in symbols],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Snapshot validation ───────────────────────────────────────────────────────

def validate_broker_snapshot(snapshot: BrokerSnapshot) -> list[str]:
    """
    BrokerSnapshot のバリデーション。空リストなら合格。

    検証項目:
      - cash: 存在 / NaN/Inf 禁止 / 負値禁止
      - symbols alignment: positions / avg_costs / market_values キー一致
      - qty count == cost count
      - 重複シンボル禁止
      - qty / cost 値の NaN/Inf 禁止
      - ts 存在
      - equity: NaN/Inf 禁止 (0 は許容)
    """
    errors: list[str] = []

    # cash
    if snapshot.cash is None:
        errors.append("cash is None")
    elif math.isnan(snapshot.cash) or math.isinf(snapshot.cash):
        errors.append(f"cash is NaN/Inf: {snapshot.cash}")
    elif snapshot.cash < 0:
        errors.append(f"cash is negative: {snapshot.cash}")

    # symbol alignment
    pos_syms  = set(snapshot.positions.keys())
    cost_syms = set(snapshot.avg_costs.keys())
    val_syms  = set(snapshot.market_values.keys())
    if pos_syms != cost_syms:
        errors.append(
            f"positions/avg_costs symbol mismatch: "
            f"extra={pos_syms - cost_syms} missing={cost_syms - pos_syms}"
        )
    if pos_syms != val_syms:
        errors.append(
            f"positions/market_values symbol mismatch: "
            f"extra={pos_syms - val_syms} missing={val_syms - pos_syms}"
        )

    # qty count == cost count
    if len(snapshot.positions) != len(snapshot.avg_costs):
        errors.append(
            f"qty count ({len(snapshot.positions)}) != cost count ({len(snapshot.avg_costs)})"
        )

    # duplicate symbols
    pos_keys = list(snapshot.positions.keys())
    if len(pos_keys) != len(set(pos_keys)):
        errors.append(f"duplicate symbols in positions: {pos_keys}")

    # qty values
    for sym, qty in snapshot.positions.items():
        if qty is None:
            errors.append(f"positions[{sym}] is None")
        elif isinstance(qty, float) and (math.isnan(qty) or math.isinf(qty)):
            errors.append(f"positions[{sym}] is NaN/Inf")

    # cost values
    for sym, cost in snapshot.avg_costs.items():
        if cost is None:
            errors.append(f"avg_costs[{sym}] is None")
        elif math.isnan(cost) or math.isinf(cost):
            errors.append(f"avg_costs[{sym}] is NaN/Inf: {cost}")

    # ts
    if not snapshot.ts:
        errors.append("ts is empty/None")

    # equity
    if snapshot.equity is None:
        errors.append("equity is None")
    elif math.isnan(snapshot.equity) or math.isinf(snapshot.equity):
        errors.append(f"equity is NaN/Inf: {snapshot.equity}")

    return errors


# ── Snapshot commit ───────────────────────────────────────────────────────────

def commit_broker_snapshot(state: dict, snapshot: BrokerSnapshot) -> dict:
    """
    完全な BrokerSnapshot をバリデーション後に state dict へ一括 commit する。

    FORBIDDEN: partial commit (cash のみ / positions のみ)
    REQUIRED:  positions_ok AND wallet_ok の両 API 成功時のみ呼び出すこと。

    バリデーション失敗時は SnapshotValidationError を raise し state を一切変更しない。
    例外が発生した場合も旧 state が保持される (partial write なし)。
    """
    errors = validate_broker_snapshot(snapshot)
    if errors:
        raise SnapshotValidationError(
            f"BrokerSnapshot validation failed ({len(errors)} errors): {errors}"
        )

    snap_hash = _compute_snapshot_hash(snapshot)

    old_hash = state.get("snapshot_hash")
    if old_hash and old_hash != snap_hash:
        logger.info("[SNAPSHOT] hash changed %s → %s", old_hash, snap_hash)

    # ── 一括 commit ──────────────────────────────────────────────────────────
    live_pos = {sym: int(q) for sym, q in snapshot.positions.items() if int(q) > 0}
    state["available_cash"]          = round(snapshot.cash, 0)
    state["position_qtys"]           = live_pos
    state["positions_count"]         = len(live_pos)
    # position_entry_prices: original entry price を保護し、新規銘柄のみ avg_cost で初期化
    ep = state.setdefault("position_entry_prices", {})
    for sym, cost in snapshot.avg_costs.items():
        if sym not in ep:
            ep[sym] = float(cost)
    state["position_current_prices"] = {sym: float(v) for sym, v in snapshot.market_values.items()}
    state["last_equity"]             = round(snapshot.equity, 0)
    state["snapshot_ts"]             = snapshot.ts
    state["snapshot_hash"]           = snap_hash
    state["snapshot_avg_costs"]      = {sym: float(c) for sym, c in snapshot.avg_costs.items()}

    logger.info(
        "[SNAPSHOT] committed: source=%s cash=¥%s positions=%d equity=¥%s hash=%s",
        snapshot.source,
        f"{snapshot.cash:,.0f}",
        len(live_pos),
        f"{snapshot.equity:,.0f}",
        snap_hash,
    )

    # ── Settlement lag 検出 ────────────────────────────────────────────────
    # position_entry_dates に存在するが broker snapshot に含まれない銘柄を警告する。
    # これは同日購入後の未着金 (settlement lag) または state ファイル不整合を示す。
    entry_syms  = set(state.get("position_entry_dates", {}).keys())
    broker_syms = set(live_pos.keys())
    orphaned    = entry_syms - broker_syms
    if orphaned:
        orphaned_val = sum(
            float(state.get("position_entry_prices", {}).get(sym, 0))
            * int(state.get("position_qtys", {}).get(sym, 0) or 0)
            for sym in orphaned
        )
        logger.warning(
            "[SNAPSHOT] SETTLEMENT_LAG_DETECTED: %d 銘柄が entry_dates にあるが "
            "broker snapshot に未反映 → %s (推定額=¥%s) "
            "翌日 commit 後に自動解消予定。"
            "CB_GUARD V2 が補償値として考慮します。",
            len(orphaned), sorted(orphaned), f"{orphaned_val:,.0f}",
        )

    _cleanup_orphaned_entry_metadata(state, live_pos)

    return state


# broker上で保有0になった銘柄について、position_qtys以外に残存するper-symbol
# metadataを削除する対象フィールド一覧（2026-07-19、entry_metadata cleanup統合）。
# position_entry_prices は対象外（reentry判定・損益分析での過去建値参照用途があり、
# 削除要否は別途要検討のため今回は変更しない）。
_ORPHANED_METADATA_FIELDS: tuple[str, ...] = (
    "position_entry_dates",
    "position_entry_atrs",
    "position_highest_closes",
    "position_unrealized_pnl",
    "position_unrealized_pct",
    "reentry_blocked",
    "position_entry_rsrs",
    "entry_metadata_missing",
)


def _cleanup_orphaned_entry_metadata(state: dict, live_pos: dict) -> None:
    """
    broker snapshot（live_pos）に存在しない銘柄の per-symbol metadata を
    _ORPHANED_METADATA_FIELDS から削除する（Broker-as-Sole-SSOT, 2026-07-19）。

    commit_broker_snapshot() が position_qtys/positions_count を broker 値で
    正しく上書きする一方、entry_dates/highest_closes/reentry_blocked 等の
    per-symbol metadataは銘柄が売却されても削除されず残存し続けていた
    （実機検証で2802.T/5301.T/6506.T/6981.Tが数日〜1週間以上残存する事例を確認）。

    対象は _ORPHANED_METADATA_FIELDS のいずれかに存在するがlive_posに存在しない
    symbol（position_entry_dates基準ではなく、8フィールド全ての和集合を基準にする
    — entry_date記録が欠落したまま他フィールドにのみ残るケース（2802.T型）も
    確実に捕捉するため）。position_qtys/position_entry_pricesは対象外
    （前者はこの直前で既にbroker値へ正しく上書き済み、後者は意図的に対象外）。
    """
    live_syms: set = set(live_pos.keys())
    stale_syms: set = set()
    for field in _ORPHANED_METADATA_FIELDS:
        d = state.get(field)
        if isinstance(d, dict):
            stale_syms |= (set(d.keys()) - live_syms)

    if not stale_syms:
        return

    removed_by_field: dict[str, list[str]] = {}
    for field in _ORPHANED_METADATA_FIELDS:
        d = state.get(field)
        if not isinstance(d, dict):
            continue
        removed = [sym for sym in stale_syms if sym in d]
        for sym in removed:
            del d[sym]
        if removed:
            removed_by_field[field] = sorted(removed)

    logger.info(
        "[ENTRY_METADATA_CLEANUP] removed=%d symbols=%s fields=%s",
        len(stale_syms), sorted(stale_syms), removed_by_field,
    )


# ── アトミック書き込み ────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    """
    tmp → flush → fsync → os.replace() の順でアトミック書き込みする。

    Windows では同一ドライブ内の os.replace() は原子的なので
    書き込み途中のクラッシュでファイルが壊れない。
    tmpfile を path と同じディレクトリに置くことで同一 volume を保証する。

    Windows 同時書き込み対応: os.replace() が WinError 5 (AccessDenied) を
    返す場合は短いバックオフ後にリトライする（最大 5 回）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return
            except OSError as e:
                if attempt < 4 and getattr(e, "winerror", None) == 5:
                    import time as _time
                    _time.sleep(0.01 * (attempt + 1))
                else:
                    raise
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── バックアップ ──────────────────────────────────────────────────────────────

def _backup_state(path: Path) -> Path | None:
    if not path.exists():
        return None
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    dst = _BACKUP_DIR / f"portfolio_state.bak.{ts}.json"
    try:
        shutil.copy2(path, dst)
        logger.info("[STATE] バックアップ作成: %s", dst)
        return dst
    except Exception as e:
        logger.warning("[STATE] バックアップ失敗 (非致命的): %s", e)
        return None


# ── スナップショット年齢 ──────────────────────────────────────────────────────

def _compute_snapshot_age(state: dict) -> float | None:
    raw = state.get("updated_at") or state.get("last_updated")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
        return (datetime.now(JST) - dt).total_seconds()
    except ValueError:
        pass
    try:
        dt = datetime.strptime(str(raw)[:10], "%Y-%m-%d").replace(tzinfo=JST)
        return (datetime.now(JST) - dt).total_seconds()
    except ValueError:
        return None


# ── バリデーション ────────────────────────────────────────────────────────────

def _is_nan_or_inf(v: Any) -> bool:
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def validate_state(state: dict) -> ValidationResult:
    """
    schema v2 に対してフル検証を行う。

    Hard fails: equity<0, cash<0, qty<0, NaN/Inf, schema 完全崩壊
    Warnings:   stale, peak anomaly, 欠損キー, snapshot_hash mismatch
    """
    hard_fails: list[str] = []
    warnings:   list[str] = []

    if not isinstance(state, dict):
        hard_fails.append(f"state が dict でない: {type(state)}")
        return ValidationResult(ok=False, hard_fails=hard_fails)

    # ── 必須キー ──
    missing = SCHEMA_V2_REQUIRED - set(state.keys())
    if missing:
        warnings.append(f"必須キー不足 (self-heal 済み): {sorted(missing)}")

    # ── 数値フィールド検証 ──
    equity = state.get("last_equity", 0)
    cash   = state.get("available_cash", 0)
    peak   = state.get("equity_peak", INITIAL_CAPITAL)

    if _is_nan_or_inf(equity):
        hard_fails.append(f"last_equity が NaN/Inf: {equity}")
    elif float(equity) < 0:
        hard_fails.append(f"last_equity が負値: {equity}")
    elif float(equity) == 0:
        warnings.append("last_equity=0 — 初回起動または未計算")

    if _is_nan_or_inf(cash):
        hard_fails.append(f"available_cash が NaN/Inf: {cash}")
    elif float(cash) < 0:
        hard_fails.append(f"available_cash が負値: {cash}")

    if _is_nan_or_inf(peak):
        hard_fails.append(f"equity_peak が NaN/Inf: {peak}")
    elif float(peak) <= 0:
        hard_fails.append(f"equity_peak が非正値: {peak}")

    # ── qty 検証 ──
    for sym, qty in state.get("position_qtys", {}).items():
        if _is_nan_or_inf(qty):
            hard_fails.append(f"position_qtys[{sym}] が NaN/Inf: {qty}")
        elif int(qty) < 0:
            hard_fails.append(f"position_qtys[{sym}] が負値: {qty}")

    # ── 重複シンボルチェック ──
    qtys_keys = list(state.get("position_qtys", {}).keys())
    if len(qtys_keys) != len(set(qtys_keys)):
        hard_fails.append(f"position_qtys に重複シンボル: {qtys_keys}")

    # ── スナップショット年齢 ──
    age = _compute_snapshot_age(state)
    if age is not None:
        if age > STALE_HARD_SECONDS:
            warnings.append(
                f"[STATE] stale snapshot age={age:.0f}s "
                f"> {STALE_HARD_SECONDS}s — using persisted broker snapshot"
            )
        elif age > STALE_WARN_SECONDS:
            warnings.append(
                f"[STATE] stale snapshot age={age:.0f}s "
                f"> {STALE_WARN_SECONDS}s — using persisted broker snapshot"
            )

    # ── cb_state 値域 ──
    valid_cb = {"NORMAL", "SAFE_WARN", "CB_ACTIVE", "RECOVERY"}
    cb = state.get("cb_state", "NORMAL")
    if cb not in valid_cb:
        warnings.append(f"cb_state 不正値 '{cb}' → NORMAL に補正")

    # ── schema_version ──
    if state.get("schema_version") != SCHEMA_VERSION:
        warnings.append(
            f"schema_version={state.get('schema_version')} → v{SCHEMA_VERSION} に移行"
        )

    # ── positions_count の一貫性 ──
    actual_count = sum(1 for q in state.get("position_qtys", {}).values() if int(q) > 0)
    if state.get("positions_count", actual_count) != actual_count:
        warnings.append(
            f"positions_count={state.get('positions_count')} が "
            f"position_qtys ({actual_count}件) と不一致 → 修正"
        )

    # ── snapshot_hash integrity ──
    stored_hash = state.get("snapshot_hash")
    if stored_hash:
        try:
            recomputed = _recompute_hash_from_state(state)
            if stored_hash != recomputed:
                warnings.append(
                    f"snapshot_hash mismatch: stored={stored_hash} recomputed={recomputed} "
                    f"— state may have been modified outside state_store"
                )
        except Exception as e:
            warnings.append(f"snapshot_hash recomputation failed: {e}")

    # ── candidate_peak 形式検証 (EQUITY_PEAK_HARDENING) ──
    cand = state.get("candidate_peak")
    if cand is not None:
        if not isinstance(cand, dict) or "value" not in cand or "staged_date" not in cand:
            warnings.append(f"candidate_peak 形式不正 (self-heal 対象): {cand!r}")
        elif _is_nan_or_inf(cand.get("value")) or float(cand.get("value", 0)) <= 0:
            warnings.append(f"candidate_peak.value が非正値/NaN (self-heal 対象): {cand.get('value')}")

    return ValidationResult(
        ok                   = len(hard_fails) == 0,
        hard_fails           = hard_fails,
        warnings             = warnings,
        snapshot_age_seconds = age,
    )


# ── Self-heal ────────────────────────────────────────────────────────────────

def _self_heal(state: dict, validation: ValidationResult) -> tuple[dict, list[str]]:
    """修復可能な問題を自動修正する。hard_fails のうち修正可能なものも処置する。"""
    healed: list[str] = []

    for k, default in V2_SAFE_DEFAULTS.items():
        if k not in state:
            state[k] = default
            healed.append(f"欠損キー '{k}' を default={default!r} で補填")

    for key in ("last_equity", "available_cash"):
        if _is_nan_or_inf(state.get(key, 0)):
            state[key] = 0.0
            healed.append(f"{key}: NaN/Inf → 0.0")

    if float(state.get("available_cash", 0)) < 0:
        state["available_cash"] = 0.0
        healed.append("available_cash: 負値 → 0.0")

    if _is_nan_or_inf(state.get("equity_peak", 0)) or float(state.get("equity_peak", 0)) <= 0:
        _old_peak_for_audit = state.get("equity_peak", 0)
        state["equity_peak"] = INITIAL_CAPITAL
        healed.append(f"equity_peak: 非正値 → {INITIAL_CAPITAL}")
        # Study96 EquityPeak SSOT Root Cause Audit (2026-07-17): この経路は
        # _commit_equity_peak() のSSOT外だが破損値修復として正当な例外的書込み。
        # 監査の完全性のため equity_peak_audit.jsonl にも記録する（durable trace）。
        try:
            from src.portfolio.equity import append_peak_audit as _append_peak_audit
            _append_peak_audit(
                action="SELF_HEAL", old_peak=float(_old_peak_for_audit) if not _is_nan_or_inf(_old_peak_for_audit) else -1.0,
                new_peak=INITIAL_CAPITAL, current_equity=float(state.get("last_equity", 0) or 0),
                broker_equity=None, caller="state_store._self_heal", reason="corrupted_peak_repair",
                diag=f"raw equity_peak was NaN/Inf/non-positive: {_old_peak_for_audit!r}",
                trading_date=datetime.now(JST).strftime("%Y-%m-%d"), mode="self_heal",
                pid=os.getpid(), run_id="N/A",
            )
        except Exception as _audit_err:
            logger.warning("[EQUITY_PEAK_AUDIT] self_heal audit記録失敗 (非致命的): %s", _audit_err)

    _cand = state.get("candidate_peak")
    if _cand is not None and (
        not isinstance(_cand, dict)
        or "value" not in _cand
        or "staged_date" not in _cand
        or _is_nan_or_inf(_cand.get("value", 0))
        or float(_cand.get("value", 0)) <= 0
    ):
        state["candidate_peak"] = None
        healed.append("candidate_peak: 不正値 → None にクリア")

    if state.get("cb_state") not in {"NORMAL", "SAFE_WARN", "CB_ACTIVE", "RECOVERY"}:
        state["cb_state"] = "NORMAL"
        healed.append("cb_state: 不正値 → NORMAL")

    actual_count = sum(1 for q in state.get("position_qtys", {}).values() if int(q) > 0)
    if state.get("positions_count", 0) != actual_count:
        state["positions_count"] = actual_count
        healed.append(f"positions_count: {state.get('positions_count')} → {actual_count}")

    if state.get("schema_version") != SCHEMA_VERSION:
        old = state.get("schema_version", "なし")
        state["schema_version"] = SCHEMA_VERSION
        healed.append(f"schema_version: {old} → {SCHEMA_VERSION}")

    return state, healed


# ── ロード ────────────────────────────────────────────────────────────────────

def load_portfolio_state(path: Path | None = None) -> tuple[dict, ValidationResult]:
    """
    portfolio_state.json を読み込み、検証・自動修復して返す。

    generation_id が watermark より小さい場合は rollback を警告する。
    エラーがあっても必ず使用可能な state を返す（crash しない）。
    """
    target = Path(path) if path else _DEFAULT_PATH
    state: dict = dict(V2_SAFE_DEFAULTS)

    if not target.exists():
        vr = ValidationResult(
            ok         = False,
            hard_fails = [f"portfolio_state.json が存在しない: {target}"],
            healed     = ["safe defaults を使用"],
        )
        logger.warning("[STATE] %s", vr.hard_fails[0])
        return _GuardedStateDict(state), vr

    try:
        raw    = target.read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"JSON root is not a dict: {type(loaded)}")
        state.update(loaded)
    except Exception as exc:
        logger.error("[STATE] JSON 読み込み失敗: %s — safe defaults を使用", exc)
        _backup_state(target)
        vr = ValidationResult(
            ok         = False,
            hard_fails = [f"JSON 読み込み失敗: {exc}"],
            healed     = ["safe defaults を使用"],
        )
        return _GuardedStateDict(state), vr

    # ── generation rollback 検知 ──
    loaded_gen = int(state.get("generation_id", 0))
    watermark  = _read_generation_watermark()
    if loaded_gen < watermark:
        logger.warning(
            "[STATE] generation rollback detected: loaded=%d < watermark=%d "
            "— state may have been replaced with an older backup",
            loaded_gen, watermark,
        )

    # ── バリデーション ──
    vr = validate_state(state)

    # ── self-heal ──
    if vr.warnings or not vr.ok:
        _backup_state(target)
        state, healed = _self_heal(state, vr)
        vr.healed.extend(healed)
        if healed:
            logger.warning("[STATE] self-heal 適用 (%d 修正): %s", len(healed), healed)

    for w in vr.warnings:
        logger.warning("[STATE] %s", w)
    for f in vr.hard_fails:
        logger.error("[STATE] HARD FAIL: %s", f)

    return _GuardedStateDict(state), vr


# ── セーブ ────────────────────────────────────────────────────────────────────

def save_portfolio_state(
    state:       dict,
    path:        Path | None = None,
    data_source: str         = "internal",
) -> None:
    """
    portfolio_state.json への唯一の書き込み経路。

    - generation_id をインクリメントし watermark を更新
    - schema_version / updated_at / data_source を自動付与
    - snapshot_hash を必ず再計算してから永続化する（2026-07-15 SSOT修正）
    - atomic write (tmp → fsync → os.replace)

    snapshot_hash の意味論（2026-07-15 確定）:
      「broker snapshotとの一致証明」ではなく「永続化されるcash/qtys/costs
      フィールド自体の自己整合性チェックサム」。commit_broker_snapshot() を
      経由したか否かに関わらず、save_portfolio_state() を呼べば必ず
      その時点のcash/position_qtys/snapshot_avg_costsと一致する値になる。
      validate_state() の mismatch 警告は「state_storeを経由しない外部からの
      直接編集」のみを検知する意味に純化される（本来の意図通り）。
      snapshot_hash はいかなる発注可否判定・authority/deployment gatingにも
      使われていないことを2026-07-15 SSOT監査で確認済み（純粋診断用）。
    """
    target  = Path(path) if path else _DEFAULT_PATH
    now_iso = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

    # generation_id monotonically increment
    new_gen = int(state.get("generation_id", 0)) + 1
    state["generation_id"]  = new_gen
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"]     = now_iso
    state["data_source"]    = data_source
    state["last_updated"]   = now_iso[:10]

    state["positions_count"] = sum(
        1 for q in state.get("position_qtys", {}).values() if int(q) > 0
    )
    state["snapshot_hash"] = _recompute_hash_from_state(state)

    atomic_write_json(target, state)

    # update watermark if new_gen is higher
    wm = _read_generation_watermark()
    if new_gen > wm:
        _write_generation_watermark(new_gen)

    logger.info(
        "[STATE] 保存完了: gen=%d source=%s path=%s",
        new_gen, data_source, target,
    )

    try:
        _caller_frame = sys._getframe(1)
        _caller = f"{_caller_frame.f_code.co_filename}:{_caller_frame.f_code.co_name}:{_caller_frame.f_lineno}"
    except Exception:
        _caller = "unknown"
    _last_equity = float(state.get("last_equity", 0))
    _cash        = float(state.get("available_cash", 0))
    logger.info(
        "[EQUITY_PEAK_SAVE] equity_peak=¥%s equity=¥%s cash=¥%s market_value=¥%s caller=%s",
        f"{float(state.get('equity_peak', 0)):,.0f}",
        f"{_last_equity:,.0f}",
        f"{_cash:,.0f}",
        f"{_last_equity - _cash:,.0f}",
        _caller,
    )


# ── Broker snapshot 永続化 (legacy / DRY-mode internal use) ──────────────────

def update_portfolio_state_from_broker(
    state:           dict,
    *,
    available_cash:  float | None            = None,
    position_qtys:   dict[str, int] | None   = None,
    avg_costs:       dict[str, float] | None  = None,
    market_values:   dict[str, float] | None  = None,
    current_equity:  float | None            = None,
) -> dict:
    """
    [INTERNAL] broker API 取得値を state に反映する (partial update)。

    WARN: 全取得成功時は commit_broker_snapshot() を使うこと。
    このメソッドは partial snapshot (equity のみ更新など) の用途に限定する。
    呼び出し後に save_portfolio_state() でファイルに書き出すこと。
    """
    if available_cash is not None:
        state["available_cash"] = round(float(available_cash), 0)

    if position_qtys is not None:
        state["position_qtys"]   = {sym: int(q) for sym, q in position_qtys.items() if int(q) > 0}
        state["positions_count"] = len(state["position_qtys"])

    if avg_costs is not None:
        state.setdefault("position_entry_prices", {}).update(
            {sym: float(p) for sym, p in avg_costs.items()}
        )

    if market_values is not None:
        state["position_current_prices"] = {sym: float(v) for sym, v in market_values.items()}

    if current_equity is not None:
        state["last_equity"] = round(float(current_equity), 0)

    return state


# ── 観測性: 起動ログ行 ────────────────────────────────────────────────────────

def log_startup_state_line(
    state:      dict,
    validation: ValidationResult,
    log:        logging.Logger | None = None,
) -> None:
    """起動時に必ず1行の構造化 [STATE] ログを出す。"""
    _log    = log or logger
    age     = validation.snapshot_age_seconds
    age_str = f"{age:.0f}s" if age is not None else "unknown"

    _log.info(
        "[STATE] source=%s cash=¥%s positions=%d equity=¥%s peak=¥%s "
        "snapshot_age=%s schema=v%s cb=%s gen=%d",
        state.get("data_source", "?"),
        f"{float(state.get('available_cash', 0)):,.0f}",
        int(state.get("positions_count", 0)),
        f"{float(state.get('last_equity', 0)):,.0f}",
        f"{float(state.get('equity_peak', 0)):,.0f}",
        age_str,
        state.get("schema_version", "?"),
        state.get("cb_state", "?"),
        int(state.get("generation_id", 0)),
    )

    if validation.is_very_stale:
        _log.warning("[STATE] stale snapshot age=%s using persisted broker snapshot", age_str)
    elif validation.is_stale:
        _log.warning("[STATE] stale snapshot age=%s using persisted broker snapshot", age_str)

    for w in validation.warnings:
        _log.warning("[STATE] %s", w)
    for h in validation.hard_fails:
        _log.error("[STATE] HARD FAIL: %s", h)


# ── リコンシリエーション ログ ────────────────────────────────────────────────

def write_reconciliation_log(
    mode:            str,
    broker_cash:     float | None,
    state_cash:      float,
    broker_equity:   float | None,
    computed_equity: float,
    positions_match: bool,
    log_dir:         Path | None = None,
) -> None:
    """毎 run の照合結果を logs/reconciliation/YYYYMMDD.jsonl に追記する。"""
    target_dir = Path(log_dir) if log_dir else _RECON_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    now   = datetime.now(JST)
    date  = now.strftime("%Y%m%d")
    ts    = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    delta = (broker_equity - computed_equity) if broker_equity is not None else None

    record = {
        "ts":              ts,
        "mode":            mode,
        "broker_cash":     round(broker_cash,     0) if broker_cash     is not None else None,
        "state_cash":      round(state_cash,      0),
        "broker_equity":   round(broker_equity,   0) if broker_equity   is not None else None,
        "computed_equity": round(computed_equity, 0),
        "delta":           round(delta,           0) if delta           is not None else None,
        "positions_match": positions_match,
    }

    log_path = target_dir / f"{date}.jsonl"
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[RECON] ログ書き込み失敗 (非致命的): %s", e)
        return

    if delta is not None and abs(delta) > 1:
        logger.warning(
            "[RECON] delta=¥%.0f > 1円 — broker_equity=¥%.0f computed=¥%.0f",
            delta, broker_equity, computed_equity,
        )
    else:
        logger.info("[RECON] OK delta=%s positions_match=%s", delta, positions_match)
