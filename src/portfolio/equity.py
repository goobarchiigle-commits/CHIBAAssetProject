"""
src/portfolio/equity.py
ポートフォリオ時価総額の唯一の計算源。

RULE: このモジュール以外で cash + market_value を直接計算することを禁止する。
すべてのパス（startup / watchdog / CB / dry-run / reconcile / risk / DD）は
compute_live_equity() を呼び出すこと。

FORBIDDEN in callers:
    available_cash + market_value  ← inline 計算禁止
    margin_balance                 ← 信用評価額を equity に含めない
    shadow_value                   ← 仮想ポジションを real equity に含めない
    allocation_capacity            ← 余力は equity ではない
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.portfolio.state_store import BrokerSnapshot

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_BASE_DIR      = Path(os.environ.get("AI_TRADING_HOME", str(Path(__file__).resolve().parents[2])))
_SNAPSHOT_FILE = _BASE_DIR / "logs" / "equity_snapshots.jsonl"

EQUITY_SOURCE_TAG = "canonical_compute_live_equity"

# Peak anomaly: peak が current_equity のこの倍率を超えたら SAFE_WARN 候補
# 通常 CB は -15% (ratio=1.176) で発動するため、それを超える ratio は state 異常を疑う
PEAK_ANOMALY_RATIO: float = 1.25

# SAFE_WARN: N 連続確認後に CB_ACTIVE へ昇格
SAFE_WARN_CONFIRM_REQUIRED: int = 3

# 入出金推定の閾値: cash 変化のうち market_value 変化で説明できない残差がこの額を超えたら
# 入出金イベントと推定する。
# Study26 (2026-06-24): 300_000 → 50_000 に縮小。2026-06-23 の ¥100,000 未説明残差
# (settlement 起因と推定) を旧閾値では検知できなかったため。
CASH_EVENT_THRESHOLD: float = 50_000.0

# equity_peak 更新前の broker 生値 vs cache 併用計算値 整合性チェック閾値
# (EQUITY_PEAK_HARDENING, 2026-07-03: sync_positions.py 迂回書込み + ohlcv_cache
#  起因の誤peak混入事故を受けて追加)。絶対額・比率の両方が閾値を超えた場合のみ
# EQUITY_PEAK_REJECT とする（AND条件。小口口座での誤rejectを避けつつ、
# 2026-06-23 型の ohlcv_cache 起因スパイクを確実に捕捉する）。
# run() 内の [EQUITY_DIVERGENCE] 警告と同値を踏襲。
PEAK_CONSISTENCY_ABS_THRESHOLD: float = 300_000.0
PEAK_CONSISTENCY_PCT_THRESHOLD: float = 0.05

_PEAK_AUDIT_FILE = _BASE_DIR / "logs" / "equity_peak_audit.jsonl"


@dataclass
class EquitySnapshot:
    """equity_snapshots.jsonl に書き出す構造化スナップショット。"""
    timestamp:       str
    equity_source:   str
    cash:            float
    market_value:    float
    equity:          float
    equity_peak:     float
    positions_count: int
    mode:            str = "unknown"


def compute_live_equity(
    *,
    snapshot: "BrokerSnapshot",
    mode: str = "unknown",
    equity_peak: float = 0.0,
    persist_snapshot: bool = True,
) -> float:
    """
    ポートフォリオ時価総額の唯一の計算源（Broker-as-Sole-SSOT, 2026-07-18）。

    入力は BrokerSnapshot のみ。price は snapshot.market_values
    （broker `/positions` 応答の CurrentPrice、欠損時のみ同応答内の Price へ
    フォールバック — いずれも broker 由来）を使う。
    OHLCV キャッシュ・state・その他ローカル再計算には一切アクセスしない
    （2026-07-15〜17 equity_peak異常値インシデントの根本原因: 複数の資産取得
    経路が存在し、微妙に異なる値を返していたこと。単一入力・単一計算式に
    統一することでこのバグクラス自体を排除する）。

    Args:
        snapshot:         BrokerSnapshot（必須）
        mode:             ログ用コンテキスト ("live" / "dry" / "startup" / "reconcile" 等)
        equity_peak:      現在の peak 値（snapshot ログ用のみ）
        persist_snapshot: True なら logs/equity_snapshots.jsonl に追記

    Returns:
        float: cash + sum(qty_i * market_value_i)
    """
    _cash = snapshot.cash
    market_value = 0.0
    for sym, qty in snapshot.positions.items():
        if int(qty) <= 0:
            continue
        price = float(snapshot.market_values.get(sym, snapshot.avg_costs.get(sym, 0.0)))
        market_value += int(qty) * price
    equity = _cash + market_value

    n_pos = sum(1 for q in snapshot.positions.values() if int(q) > 0)
    snap  = EquitySnapshot(
        timestamp       = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        equity_source   = EQUITY_SOURCE_TAG,
        cash            = round(_cash,        0),
        market_value    = round(market_value, 0),
        equity          = round(equity,       0),
        equity_peak     = round(equity_peak,  0),
        positions_count = n_pos,
        mode            = mode,
    )

    logger.info(
        "[EQUITY] source=%s mode=%s cash=¥%.0f mktval=¥%.0f equity=¥%.0f peak=¥%.0f pos=%d",
        EQUITY_SOURCE_TAG, mode,
        _cash, market_value, equity, equity_peak, n_pos,
    )

    if persist_snapshot:
        _append_snapshot(snap)

    return equity


def check_peak_anomaly(
    equity_peak:     float,
    current_equity:  float,
    safe_warn_count: int = 0,
) -> tuple[bool, int, str]:
    """
    persisted equity_peak が current_equity より異常に高い場合を検出する。

    通常の CB は -15% DD で発動する。
    PEAK_ANOMALY_RATIO (1.25) を超える ratio は状態ファイルの破損を疑う。

    Returns:
        (is_anomaly, new_safe_warn_count, diagnostic_message)
    """
    if equity_peak <= 0 or current_equity <= 0:
        return False, safe_warn_count, "peak または equity が非正値 — チェックスキップ"

    ratio = equity_peak / current_equity
    if ratio > PEAK_ANOMALY_RATIO:
        new_count = safe_warn_count + 1
        msg = (
            f"PEAK_ANOMALY: peak=¥{equity_peak:,.0f} / equity=¥{current_equity:,.0f} "
            f"ratio={ratio:.3f} > threshold={PEAK_ANOMALY_RATIO:.2f} "
            f"confirm={new_count}/{SAFE_WARN_CONFIRM_REQUIRED}"
        )
        return True, new_count, msg

    # 比率が正常範囲に戻った → カウンターリセット
    return False, 0, f"peak ratio={ratio:.3f} — 正常範囲"


def check_broker_consistency(
    current_equity:  float,
    broker_snapshot: "BrokerSnapshot | None",
) -> tuple[bool, float | None, str]:
    """
    equity_peak 更新前の整合性チェック（EQUITY_PEAK_HARDENING）。

    broker_equity = snapshot.cash + Σ qty × (market_values.get(sym) or avg_costs.get(sym))
    — OHLCV キャッシュに依存しない純 broker 生値。
    current_equity（compute_live_equity() の cache 併用計算値、peak 更新判定に
    使われる値）との乖離が閾値超なら不整合と判定する。

    broker_snapshot が None（API 部分失敗等で snapshot 未構築）の場合は
    detect_cash_event() / rebuild_equity_peak() と同じ FAIL_OPEN 方針で
    整合とみなす（チェックできないことを理由に正当な peak 更新を止めない）。

    Returns:
        (is_consistent, broker_equity_or_None, diagnostic_message)
    """
    if broker_snapshot is None:
        return True, None, "broker_snapshot 未提供 — チェックスキップ (FAIL_OPEN)"

    broker_equity = float(broker_snapshot.cash)
    for sym, qty in broker_snapshot.positions.items():
        if int(qty) <= 0:
            continue
        price = float(
            broker_snapshot.market_values.get(sym, broker_snapshot.avg_costs.get(sym, 0.0))
        )
        broker_equity += int(qty) * price

    diff_abs = abs(current_equity - broker_equity)
    diff_pct = diff_abs / broker_equity if broker_equity > 0 else 0.0

    if diff_abs >= PEAK_CONSISTENCY_ABS_THRESHOLD and diff_pct >= PEAK_CONSISTENCY_PCT_THRESHOLD:
        msg = (
            f"EQUITY_PEAK_REJECT: broker_equity=¥{broker_equity:,.0f} "
            f"current_equity=¥{current_equity:,.0f} diff=¥{diff_abs:,.0f} "
            f"({diff_pct * 100:.1f}%) > threshold=¥{PEAK_CONSISTENCY_ABS_THRESHOLD:,.0f}/"
            f"{PEAK_CONSISTENCY_PCT_THRESHOLD * 100:.0f}%"
        )
        return False, broker_equity, msg

    return True, broker_equity, (
        f"consistent: diff=¥{diff_abs:,.0f} ({diff_pct * 100:.1f}%)"
    )


def append_peak_audit(
    *,
    action:         str,
    old_peak:       float,
    new_peak:       float,
    current_equity: float,
    broker_equity:  float | None,
    caller:         str,
    reason:         str,
    diag:           str,
    trading_date:   str,
    mode:           str,
    pid:            int,
    run_id:         str,
) -> None:
    """
    logs/equity_peak_audit.jsonl に equity_peak の全変更試行を追記する durable シンク
    （EQUITY_PEAK_HARDENING）。

    _log_equity_peak_update() の logger.warning() は watchdog_runner.py の
    stderr[-2000:] 切り詰めにより実運用ログに永続化されない問題が確認されたため、
    stderr/stdout パイプに依存しない独立ファイルへ fsync 付きで書き込む。

    action: "APPLIED" | "REJECTED" | "STAGED" | "CONFIRMED" | "DISCARDED"
    mode:   "live" | "dry" | "manual"（"dry" は実ファイル未反映、集計時は
            mode=="live" でフィルタすること）
    """
    record = {
        "timestamp":      datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "action":         action,
        "old_peak":       round(float(old_peak), 0),
        "new_peak":       round(float(new_peak), 0),
        "current_equity": round(float(current_equity), 0),
        "broker_equity":  round(float(broker_equity), 0) if broker_equity is not None else None,
        "caller":         caller,
        "reason":         reason,
        "diagnostic":     diag,
        "trading_date":   trading_date,
        "mode":           mode,
        "pid":            pid,
        "run_id":         run_id,
    }
    try:
        _PEAK_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _PEAK_AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.warning("[EQUITY_PEAK_AUDIT] 追記失敗 (非致命的): %s record=%s", e, record)


def rebuild_equity_peak(
    current_equity: float, n_entries: int = 30, snapshot_file: Path | None = None,
) -> float:
    """
    equity_snapshots.jsonl の直近 n_entries 件から峰値を再構築する。

    PEAK_ANOMALY 時に呼び出し、異常 peak の代替値を提供する。
    サンプル不足 / IO エラー時は current_equity をそのまま返す（FAIL_OPEN）。

    Args:
        snapshot_file: 読み込み元 JSONL（既定 _SNAPSHOT_FILE）。
                       repair_equity_peak.py が任意ファイルを指すために使用。

    Returns:
        float: 再構築された equity_peak (>= current_equity を保証しない)
    """
    _snap_file = snapshot_file or _SNAPSHOT_FILE
    try:
        if not _snap_file.exists():
            return current_equity
        lines = _snap_file.read_text(encoding="utf-8").splitlines()
        equities: list[float] = []
        for raw in lines[-n_entries:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                snap_d = json.loads(raw)
                eq = float(snap_d.get("equity", 0))
                if eq > 0:
                    equities.append(eq)
            except Exception:
                pass
        if len(equities) < 2:
            return current_equity
        # Use median to filter outlier spikes (e.g., stale-run inflated values)
        equities.sort()
        mid = len(equities) // 2
        median_eq = (equities[mid] + equities[mid - 1]) / 2 if len(equities) % 2 == 0 else equities[mid]
        rebuilt = max(median_eq, current_equity)
        logger.info(
            "[PEAK_REBUILD] n=%d median=¥%.0f current=¥%.0f → rebuilt_peak=¥%.0f",
            len(equities), median_eq, current_equity, rebuilt,
        )
        return rebuilt
    except Exception as exc:
        logger.warning("[PEAK_REBUILD] failed (FAIL_OPEN): %s", exc)
        return current_equity


def detect_cash_event(
    prev_cash:         float,
    new_cash:          float,
    prev_market_value: float,
    new_market_value:  float,
) -> dict | None:
    """
    cash 変化が market_value 変化で説明不能な場合、入出金イベントを推定し記録する。

    検出のみ。state の自動修復・equity_peak の補正は行わない（FAIL_OPEN/観測専用）。

    ロジック:
        explained_delta   = -market_value_delta  （売買による cash<->mv の振替分）
        unexplained_delta = cash_delta - explained_delta
        |unexplained_delta| > CASH_EVENT_THRESHOLD → deposit（+）/ withdrawal（-）と推定

    Returns:
        dict（イベント検出時）/ None（閾値未満・通常運用とみなす）
    """
    cash_delta       = new_cash - prev_cash
    mv_delta         = new_market_value - prev_market_value
    explained_delta  = -mv_delta
    unexplained_delta = cash_delta - explained_delta

    if abs(unexplained_delta) <= CASH_EVENT_THRESHOLD:
        return None

    event_type = "deposit" if unexplained_delta > 0 else "withdrawal"
    logger.warning(
        "[EQUITY_CASH_EVENT] old_cash=¥%s new_cash=¥%s delta=¥%s "
        "unexplained=¥%s event_type=%s",
        f"{prev_cash:,.0f}", f"{new_cash:,.0f}", f"{cash_delta:,.0f}",
        f"{unexplained_delta:,.0f}", event_type,
    )
    return {
        "old_cash":           prev_cash,
        "new_cash":           new_cash,
        "delta":              cash_delta,
        "unexplained_delta":  unexplained_delta,
        "event_type":         event_type,
    }


def _append_snapshot(snap: EquitySnapshot) -> None:
    """equity_snapshots.jsonl に構造化スナップショットを追記する。"""
    try:
        _SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _SNAPSHOT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snap), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("equity snapshot 追記失敗 (非致命的): %s", e)
