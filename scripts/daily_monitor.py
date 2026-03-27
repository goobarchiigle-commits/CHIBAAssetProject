"""
scripts/daily_monitor.py
日次監視スクリプト（モメンタム崩壊・DD急変・発注リスクの早期検知）

週次PDCAだけだと崩壊検知が遅れる。毎日ルールベースでチェックして
問題が起きたらアラートを出す。AIレビューなし（高速・軽量）。

実行:
  python scripts/daily_monitor.py              # 平日のみ自動実行
  python scripts/daily_monitor.py --force      # 曜日に関係なく実行（テスト用）
  python scripts/daily_monitor.py --quiet      # 異常時のみ出力

Windowsタスクスケジューラー:
  毎日 16:00 → スクリプト内で土日を自動スキップ
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

JST      = timezone(timedelta(hours=9))
BASE_DIR = Path(__file__).parent.parent

METRICS_PATH  = BASE_DIR / "logs" / "diagnostics" / "metrics.jsonl"
PHASE2_PATH   = BASE_DIR / "logs" / "phase2_live_metrics.jsonl"
TRADES_PATH   = BASE_DIR / "logs" / "trades.jsonl"
CB_STATE_PATH = BASE_DIR / "runtime" / "cb_state.json"


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def _load_jsonl(path: Path, tail: int = 30) -> list[dict]:
    if not path.exists():
        return []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    for l in lines[-tail:]:
        try:
            results.append(json.loads(l))
        except Exception:
            pass
    return results


# ------------------------------------------------------------------ #
# チェック関数（各チェックは alerts リストにメッセージを追記する）
# ------------------------------------------------------------------ #
def check_drawdown(alerts: list[str]) -> dict:
    """DD監視: -10%警告 / -13%危険ライン"""
    rows = _load_jsonl(PHASE2_PATH, tail=5)
    if not rows:
        return {"equity": None, "drawdown": None}
    latest = rows[-1]
    dd     = float(latest.get("drawdown", 0.0))
    equity = float(latest.get("equity", 0.0))
    if dd <= -0.13:
        alerts.append(f"[CRITICAL] DD危険: {dd:.1%}  -- CB発動(-15%)まであと{abs(dd+0.15):.1%}")
    elif dd <= -0.10:
        alerts.append(f"[WARNING]  DD警告: {dd:.1%}  -- -10%ライン突破")
    return {"equity": equity, "drawdown": dd}


def check_signal_drought(alerts: list[str]) -> dict:
    """シグナル枯渇: BUY候補ゼロが何日連続しているか"""
    rows = _load_jsonl(METRICS_PATH, tail=10)
    if not rows:
        return {"drought_days": None}
    zero_streak = 0
    for row in reversed(rows):
        cnt = int(row.get("candidate_count", 0) or 0)
        if cnt == 0:
            zero_streak += 1
        else:
            break
    if zero_streak >= 5:
        alerts.append(
            f"[WARNING]シグナル枯渇: {zero_streak}日連続 BUY候補ゼロ"
            " → 戦略停止状態（相場停滞 or パラメータ過剰）"
        )
    return {"drought_days": zero_streak}


def check_regime_change(alerts: list[str]) -> dict:
    """レジーム変化: trend_strength の急低下を検知"""
    rows = _load_jsonl(METRICS_PATH, tail=5)
    if len(rows) < 3:
        return {}
    latest  = rows[-1]
    ref     = rows[0]   # 5日前
    regime  = str(latest.get("trend_market", "unknown") or "unknown")
    ts_now  = float(latest.get("trend_strength", 0) or 0)
    ts_prev = float(ref.get("trend_strength", 0) or 0)
    delta   = ts_now - ts_prev

    if delta <= -0.04:
        alerts.append(
            f"[WARNING]レジーム変化の兆候: trend_strength {ts_prev:+.3f} → {ts_now:+.3f}"
            f" ({delta:+.3f}/5日）→ MTF / CB に注意"
        )
    if regime == "bear" and ts_now < -0.02:
        alerts.append(
            f"[CRITICAL] bear相場確認: trend_strength={ts_now:+.3f}"
            " → 新規BUYは慎重に（RSR閾値引き上げを検討）"
        )
    return {"regime": regime, "trend_strength": ts_now, "ts_delta_5d": round(delta, 4)}


def check_rsr_momentum_decay(alerts: list[str]) -> dict:
    """モメンタム減衰: RSRリーダー半減期の急低下"""
    rows = _load_jsonl(METRICS_PATH, tail=7)
    hl_values = []
    for r in rows:
        v = r.get("rsr_leader_half_life")
        if v is not None:
            fv = float(v)
            if fv < 99.0:   # 99.0 = フォールバック上限値なので除外
                hl_values.append(fv)
    if len(hl_values) < 2:
        return {}
    hl_latest = hl_values[-1]
    hl_avg    = sum(hl_values) / len(hl_values)
    if hl_latest < 8:
        alerts.append(
            f"[WARNING]モメンタム減衰: RSRリーダー半減期={hl_latest:.1f}日 < 8日"
            " → 回転相場（戦略効果低下。翌週のシャープ比に注意）"
        )
    elif hl_latest < hl_avg * 0.65:
        alerts.append(
            f"[NOTICE]半減期急低下: 平均{hl_avg:.1f}日 → {hl_latest:.1f}日"
            f" (-{(1 - hl_latest/hl_avg):.0%}) → トレンド持続力が落ちている可能性"
        )
    return {"hl_latest": hl_latest, "hl_avg_7d": round(hl_avg, 1)}


def check_order_risk(alerts: list[str]) -> dict:
    """発注リスク監査: 過剰発注頻度・流動性インパクト・シグナルクラスタリング
    JPX過剰発注規制対策として日次でチェックする
    """
    rows = _load_jsonl(METRICS_PATH, tail=5)
    if not rows:
        return {}

    # クラスタリング: 直近5日で breakout_cluster_fired（同日BUY≥3）が何日あったか
    cluster_days = sum(1 for r in rows if r.get("breakout_cluster_fired", False))
    if cluster_days >= 3:
        alerts.append(
            f"[WARNING]シグナルクラスタリング: 直近5日中{cluster_days}日で同日BUY≥3"
            " → 過剰発注リスク要確認（JPX規制対応）"
        )

    # 機会損失: missed_breakout が多い → max_positions 引き上げ候補
    missed_avg = sum(float(r.get("missed_breakout_count", 0) or 0) for r in rows) / max(1, len(rows))
    if missed_avg >= 2.5:
        alerts.append(
            f"[NOTICE]機会損失: missed_breakout 平均{missed_avg:.1f}/日"
            " → max_positions 引き上げを検討（要バックテスト確認）"
        )

    # 流動性ブロック: rsp_pass_tradeable_ratio が低い → 強い銘柄が買えない状態
    latest = rows[-1]
    tr_ratio = latest.get("rsr_pass_tradeable_ratio")
    if tr_ratio is not None and float(tr_ratio) < 0.4:
        alerts.append(
            f"[NOTICE]流動性ブロック: rsr_pass_tradeable_ratio={float(tr_ratio):.0%}"
            " < 40% → 強い銘柄が価格制約で買えない（capital拡大 or 宇宙見直し）"
        )

    return {
        "cluster_days_5d": cluster_days,
        "missed_avg":      round(missed_avg, 1),
    }


def check_cb_state(alerts: list[str]) -> dict:
    """サーキットブレーカー状態確認"""
    if not CB_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(CB_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    state = data.get("cb_state", "NORMAL")
    if state == "CB_ACTIVE":
        cooldown_end = data.get("cb_cooldown_end_date", "不明")
        alerts.append(f"[CRITICAL] CB発動中: サーキットブレーカーACTIVE → BUY停止中（解除予定: {cooldown_end}）")
    elif state == "RECOVERY":
        recovery_thr = data.get("recovery_threshold", "不明")
        alerts.append(f"[WARNING]CB回復中: RECOVERY状態 → BUY再開には equity ≥ {recovery_thr} が必要")
    return {"cb_state": state}


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main(quiet: bool = False, force: bool = False) -> int:
    now  = datetime.now(JST)
    today_weekday = now.weekday()  # 0=月 ... 4=金 / 5=土 6=日

    # 土日はスキップ（--force で無視）
    if not force and today_weekday >= 5:
        return 0

    today = now.strftime("%Y-%m-%d")
    alerts: list[str] = []

    dd_info      = check_drawdown(alerts)
    drought_info = check_signal_drought(alerts)
    regime_info  = check_regime_change(alerts)
    decay_info   = check_rsr_momentum_decay(alerts)
    order_info   = check_order_risk(alerts)
    cb_info      = check_cb_state(alerts)

    has_critical = any("[CRITICAL]" in a for a in alerts)
    has_warning  = any("[WARNING]" in a for a in alerts)

    if alerts or not quiet:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  日次監視レポート  {today}  ({['月','火','水','木','金','土','日'][today_weekday]})")
        print(sep)

        equity_str = f"¥{dd_info['equity']:,.0f}" if dd_info.get("equity") else "N/A"
        dd_val     = dd_info.get("drawdown")
        dd_str     = f"{dd_val:.1%}" if dd_val is not None else "N/A"
        hl_str     = f"{decay_info.get('hl_latest', 'N/A')}日"
        regime_str = regime_info.get("regime", "N/A")
        ts_str     = f"{regime_info.get('trend_strength', 'N/A')}" if isinstance(regime_info.get('trend_strength'), float) else "N/A"
        cb_str     = cb_info.get("cb_state", "N/A")

        print(f"  資産       : {equity_str}  |  DD: {dd_str}")
        print(f"  Regime     : {regime_str}  |  TrendStr: {ts_str}")
        print(f"  RSR半減期  : {hl_str}  |  CB: {cb_str}")
        print(f"  枯渇日数   : {drought_info.get('drought_days', 'N/A')}日連続ゼロ")

        if alerts:
            status = "[CRITICAL] 要対応" if has_critical else ("[WARNING]要注意" if has_warning else "△ 注意")
            print(f"\n  【アラート {len(alerts)}件】  {status}")
            for a in alerts:
                print(f"    {a}")
        else:
            print("\n  異常なし [OK]")

        print(f"{sep}\n")

    # 終了コード: [CRITICAL] があれば 1（タスクスケジューラーで失敗検知可能）
    return 1 if has_critical else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="日次監視スクリプト")
    p.add_argument("--quiet", action="store_true", help="異常時のみ出力")
    p.add_argument("--force", action="store_true", help="土日もスキップせず実行")
    args = p.parse_args()
    sys.exit(main(quiet=args.quiet, force=args.force))
