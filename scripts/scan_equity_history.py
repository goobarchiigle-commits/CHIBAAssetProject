"""
scripts/scan_equity_history.py
equity_snapshots.jsonl と scheduler ログから equity 異常を法医学的に検出する。

使い方:
    cd C:/ai-trading
    python scripts/scan_equity_history.py

出力:
    疑わしいイベントを stdout に報告する。終了コード 0 = クリーン / 1 = 異常あり。

検出項目:
    [A] 単一ステップで >8% の equity 急変
    [B] equity == cash (market_value=0 なのにポジションあり → 二重計算バグ疑い)
    [C] equity_peak > equity * 1.25 (fake peak)
    [D] equity_peak > capital * 1.5 (初期資本比で非現実的な peak)
    [E] equity の連続2ステップで急騰→急落 (一時的な重複加算の signature)
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JST      = timezone(timedelta(hours=9))
_ROOT    = Path(__file__).resolve().parents[1]
_CAPITAL = 3_000_000   # CLAUDE.md capital=3_000_000

SNAPSHOT_FILE  = _ROOT / "logs"    / "equity_snapshots.jsonl"
STATE_FILE     = _ROOT / "runtime" / "portfolio_state.json"
SCHEDULER_DIR  = _ROOT / "logs"    / "scheduler"
CB_EVENTS_DIR  = _ROOT / "logs"    / "cb_events"

# 閾値
SPIKE_THRESHOLD   = 0.08   # 8% 単一ステップ jump
FAKE_PEAK_RATIO   = 1.25
IMPOSSIBLE_RATIO  = 1.50   # capital 比


@dataclass
class Finding:
    rule:      str
    timestamp: str
    message:   str
    severity:  str  # "WARN" / "CRITICAL"


def _load_snapshots() -> list[dict]:
    if not SNAPSHOT_FILE.exists():
        return []
    records = []
    with SNAPSHOT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def _load_cb_events() -> list[dict]:
    events: list[dict] = []
    if not CB_EVENTS_DIR.exists():
        return events
    for fpath in sorted(CB_EVENTS_DIR.glob("cb_*.jsonl")):
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


# ── スケジューラーログからの equity 抽出 ───────────────────────────────────────
# 例: "2026-05-07 08:43:12 INFO CB 状態: NORMAL / equity=¥3,000,000 / peak=¥3,670,000 / DD=-18.2%"
_LOG_PATTERN = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"equity=¥(?P<equity>[\d,]+).*peak=¥(?P<peak>[\d,]+).*DD=(?P<dd>[-\d.]+)%"
)


def _load_log_equity_records() -> list[dict]:
    records = []
    if not SCHEDULER_DIR.exists():
        return records
    for fpath in sorted(SCHEDULER_DIR.glob("*.log")):
        with fpath.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_PATTERN.search(line)
                if m:
                    records.append({
                        "timestamp": m.group("ts"),
                        "equity":    float(m.group("equity").replace(",", "")),
                        "peak":      float(m.group("peak").replace(",", "")),
                        "dd":        float(m.group("dd")),
                        "source":    str(fpath.name),
                    })
    records.sort(key=lambda r: r["timestamp"])
    return records


# ── 検出ルール ──────────────────────────────────────────────────────────────────

def _check_single_step_spike(records: list[dict], equity_key: str = "equity") -> list[Finding]:
    findings: list[Finding] = []
    for i in range(1, len(records)):
        prev = float(records[i - 1].get(equity_key, 0))
        curr = float(records[i].get(equity_key, 0))
        if prev <= 0:
            continue
        change = (curr - prev) / prev
        if abs(change) > SPIKE_THRESHOLD:
            ts  = records[i].get("timestamp", "unknown")
            src = records[i].get("source",    records[i].get("mode", "?"))
            findings.append(Finding(
                rule      = "[A] single-step-spike",
                timestamp = ts,
                message   = (
                    f"equity {prev:,.0f} → {curr:,.0f} "
                    f"({change*100:+.1f}%) > {SPIKE_THRESHOLD*100:.0f}% 閾値  source={src}"
                ),
                severity  = "CRITICAL" if abs(change) > 0.20 else "WARN",
            ))
    return findings


def _check_market_value_zero(records: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    for r in records:
        mkt    = float(r.get("market_value", -1))
        n_pos  = int(r.get("positions_count", 0))
        if mkt == 0.0 and n_pos > 0:
            findings.append(Finding(
                rule      = "[B] market-value-zero",
                timestamp = r.get("timestamp", "unknown"),
                message   = (
                    f"market_value=0 だが positions_count={n_pos} — "
                    f"ポジションが計上されていない可能性 (二重計算バグ疑い)"
                ),
                severity  = "WARN",
            ))
    return findings


def _check_fake_peak(records: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    for r in records:
        equity = float(r.get("equity", 0) or r.get("equity_val", 0))
        peak   = float(r.get("equity_peak", 0) or r.get("peak", 0))
        if equity <= 0 or peak <= 0:
            continue
        ratio = peak / equity
        if ratio > FAKE_PEAK_RATIO:
            findings.append(Finding(
                rule      = "[C] fake-peak",
                timestamp = r.get("timestamp", "unknown"),
                message   = (
                    f"peak=¥{peak:,.0f} / equity=¥{equity:,.0f} "
                    f"ratio={ratio:.3f} > {FAKE_PEAK_RATIO:.2f}"
                ),
                severity  = "CRITICAL",
            ))
    return findings


def _check_impossible_peak(records: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    for r in records:
        peak = float(r.get("equity_peak", 0) or r.get("peak", 0))
        if peak > _CAPITAL * IMPOSSIBLE_RATIO:
            findings.append(Finding(
                rule      = "[D] impossible-peak",
                timestamp = r.get("timestamp", "unknown"),
                message   = (
                    f"peak=¥{peak:,.0f} > capital×{IMPOSSIBLE_RATIO:.1f}=¥{_CAPITAL*IMPOSSIBLE_RATIO:,.0f} — "
                    f"初期資本比で非現実的"
                ),
                severity  = "CRITICAL",
            ))
    return findings


def _check_pump_dump(records: list[dict], equity_key: str = "equity") -> list[Finding]:
    """2 ステップで急騰→急落するパターン（二重加算の典型的 signature）"""
    findings: list[Finding] = []
    for i in range(1, len(records) - 1):
        prev = float(records[i - 1].get(equity_key, 0))
        curr = float(records[i].get(equity_key, 0))
        nxt  = float(records[i + 1].get(equity_key, 0))
        if prev <= 0 or curr <= 0 or nxt <= 0:
            continue
        up   = (curr - prev) / prev
        down = (nxt  - curr) / curr
        if up > 0.06 and down < -0.06:
            findings.append(Finding(
                rule      = "[E] pump-dump-signature",
                timestamp = records[i].get("timestamp", "unknown"),
                message   = (
                    f"{prev:,.0f} → {curr:,.0f} ({up*100:+.1f}%) → {nxt:,.0f} ({down*100:+.1f}%) — "
                    f"一時的な二重加算の疑い"
                ),
                severity  = "CRITICAL",
            ))
    return findings


def _check_current_state() -> list[Finding]:
    """現在の portfolio_state.json の peak 異常を直接チェックする"""
    findings: list[Finding] = []
    if not STATE_FILE.exists():
        return findings
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return findings
    ep = float(state.get("equity_peak", 0))
    le = float(state.get("last_equity", 0))
    ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

    if ep > _CAPITAL * IMPOSSIBLE_RATIO:
        findings.append(Finding(
            rule      = "[D] impossible-peak (current state)",
            timestamp = ts,
            message   = f"現在の state: equity_peak=¥{ep:,.0f} > capital×{IMPOSSIBLE_RATIO:.1f}",
            severity  = "CRITICAL",
        ))

    if le > 0 and ep > 0 and (ep / le) > FAKE_PEAK_RATIO:
        findings.append(Finding(
            rule      = "[C] fake-peak (current state)",
            timestamp = ts,
            message   = (
                f"現在の state: peak=¥{ep:,.0f} / last_equity=¥{le:,.0f} "
                f"ratio={ep/le:.3f} > {FAKE_PEAK_RATIO:.2f}"
            ),
            severity  = "CRITICAL",
        ))

    cb = state.get("cb_state", "NORMAL")
    swc = int(state.get("safe_warn_count", 0))
    if cb == "SAFE_WARN":
        findings.append(Finding(
            rule      = "[SAFE_WARN] active",
            timestamp = ts,
            message   = f"cb_state=SAFE_WARN confirm_count={swc}/3 — peak 異常を検出中",
            severity  = "WARN",
        ))
    return findings


# ── メイン ─────────────────────────────────────────────────────────────────────

def main() -> int:
    all_findings: list[Finding] = []

    # ── snapshots (新フォーマット) ─────────────────────────────────────────
    snaps = _load_snapshots()
    print(f"equity_snapshots: {len(snaps)} 件読み込み")
    if snaps:
        all_findings += _check_single_step_spike(snaps)
        all_findings += _check_market_value_zero(snaps)
        all_findings += _check_fake_peak(snaps)
        all_findings += _check_impossible_peak(snaps)
        all_findings += _check_pump_dump(snaps)

    # ── scheduler ログ（旧フォーマット）────────────────────────────────────
    log_records = _load_log_equity_records()
    print(f"scheduler logs  : {len(log_records)} equity 行読み込み")
    if log_records:
        # log_records に "equity_peak" キーが入っている
        all_findings += _check_single_step_spike(log_records)
        all_findings += _check_fake_peak(log_records)
        all_findings += _check_impossible_peak(log_records)
        all_findings += _check_pump_dump(log_records)

    # ── 現在の state ───────────────────────────────────────────────────────
    all_findings += _check_current_state()

    # ── 結果出力 ───────────────────────────────────────────────────────────
    print()
    if not all_findings:
        print("✅ 異常なし — equity 履歴はクリーンです。")
        return 0

    criticals = [f for f in all_findings if f.severity == "CRITICAL"]
    warns     = [f for f in all_findings if f.severity == "WARN"]

    print(f"⚠️  FINDINGS: {len(all_findings)} 件  "
          f"(CRITICAL={len(criticals)}  WARN={len(warns)})")
    print("=" * 70)
    for f in sorted(all_findings, key=lambda x: x.timestamp):
        icon = "🔴" if f.severity == "CRITICAL" else "🟡"
        print(f"{icon} [{f.severity}] {f.rule}")
        print(f"   ts      : {f.timestamp}")
        print(f"   message : {f.message}")
        print()

    return 1 if criticals else 0


if __name__ == "__main__":
    sys.exit(main())
