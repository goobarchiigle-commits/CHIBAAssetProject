"""
bear_universe_filter の監視・週次レポート生成。
ログから bear 発動日数・除外銘柄・Sharpe/DD 推移を集計。
"""
import json
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

BEAR_LOG_PATH = Path("runtime/bear_filter_log.jsonl")


def log_bear_event(
    dt: date,
    is_bear: bool,
    excluded_count: int,
    excluded_symbols: list,
    active_count: int,
    sector_dist: dict,
) -> None:
    """bear_filter 発動イベントを jsonl に追記。ログ失敗で本番が落ちないよう try/except 済み。"""
    record = {
        "date": str(dt),
        "bear_regime": is_bear,
        "excluded_count": excluded_count,
        "excluded_symbols": excluded_symbols[:10],
        "active_count": active_count,
        "sector_distribution": sector_dist,
    }
    try:
        BEAR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BEAR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def weekly_bear_report(lookback_weeks: int = 4) -> dict:
    """
    直近 N 週分の bear_filter ログを集計してレポート出力。

    出力:
    - bear_days: bear 発動日数
    - bear_rate_pct: bear 発動率（%）
    - avg_excluded_per_bear_day: 平均除外銘柄数（bear 発動日のみ）
    - avg_active_per_bear_day: bear 時の平均 active 銘柄数
    """
    if not BEAR_LOG_PATH.exists():
        return {"error": "no bear filter log found"}

    records = []
    try:
        with open(BEAR_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        return {"error": str(e)}

    if not records:
        return {"error": "no records in log"}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=lookback_weeks)
    df = df[df["date"] >= cutoff]

    if df.empty:
        return {"error": "no data in period"}

    bear_days = int(df["bear_regime"].sum())
    total_days = len(df)
    bear_df = df[df["bear_regime"]]
    avg_excluded = float(bear_df["excluded_count"].mean()) if bear_days > 0 else 0.0
    avg_active = float(bear_df["active_count"].mean()) if bear_days > 0 else 0.0

    report = {
        "period_days": total_days,
        "bear_days": bear_days,
        "bear_rate_pct": round(bear_days / total_days * 100, 1) if total_days > 0 else 0,
        "avg_excluded_per_bear_day": round(avg_excluded, 1),
        "avg_active_per_bear_day": round(avg_active, 1),
    }

    print(f"\n=== Bear Filter 週次レポート（直近{lookback_weeks}週） ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


IDEAL_RANGES: dict[str, tuple] = {
    "bear_rate_pct":        (10.0, 25.0),   # 10-25%
    "avg_excluded_per_day": (5.0,  12.0),   # 5-12 銘柄
    "avg_active_per_day":   (12.0, None),   # 12以上
    "bear_dd_vs_baseline":  (None, 0.0),    # baseline 以下（負の値）
}

SMOKE_LOG_PATH = Path("runtime/smoke_test_log.jsonl")


def weekly_kpi_report(lookback_weeks: int = 4, equity_curve: "pd.Series | None" = None) -> dict:
    """
    4つの固定 KPI を毎週出力。ideal range を外れた項目に ⚠️ を付ける。

    1. bear 発動率（ideal: 10-25%）
    2. 平均除外銘柄数（bear 発動日のみ、ideal: 5-12）
    3. 全日平均 active_count（ideal: 12以上）
    4. bear 期間 DD（equity_curve が渡された場合のみ計算）

    Parameters
    ----------
    lookback_weeks : 集計対象の直近週数
    equity_curve   : 日次エクイティ Series（index=日付）。None の場合 bear_dd は N/A

    Returns
    -------
    dict : KPI 名 → 値
    """
    if not BEAR_LOG_PATH.exists():
        print("⚠️  bear_filter_log.jsonl が存在しません")
        return {"error": "no bear filter log found"}

    records = []
    try:
        with open(BEAR_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
    except Exception as e:
        return {"error": str(e)}

    if not records:
        return {"error": "no records in log"}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=lookback_weeks)
    df = df[df["date"] >= cutoff]

    if df.empty:
        return {"error": "no data in period"}

    total_days   = len(df)
    bear_days    = int(df["bear_regime"].sum())
    bear_df      = df[df["bear_regime"]]
    bull_df      = df[~df["bear_regime"]]

    bear_rate    = round(bear_days / total_days * 100, 1) if total_days > 0 else 0.0
    # 平均除外数（bear 日のみ）
    avg_excluded = float(bear_df["excluded_count"].mean()) if bear_days > 0 else 0.0
    # 全日の平均 active_count（bear 日は actual active_count、bull 日は active_count を使用）
    avg_active   = float(df["active_count"].mean()) if total_days > 0 else 0.0

    # bear_dd_vs_baseline: equity_curve が渡された場合のみ
    bear_dd_val  = None
    if equity_curve is not None and not equity_curve.empty and bear_days > 0:
        try:
            bear_dates   = pd.DatetimeIndex(bear_df["date"])
            eq_bear      = equity_curve.reindex(bear_dates, method="nearest").dropna()
            if len(eq_bear) >= 2:
                bear_dd_val = float(
                    (eq_bear / eq_bear.cummax() - 1).min() * 100
                )
        except Exception:
            bear_dd_val = None

    # ── KPI テーブル構築 ────────────────────────────────────────────
    def _in_range(val, lo, hi) -> bool:
        if val is None:
            return True  # N/A は判定しない
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    kpi_items = [
        ("bear_rate_pct",        bear_rate,   *IDEAL_RANGES["bear_rate_pct"]),
        ("avg_excluded_per_day", avg_excluded, *IDEAL_RANGES["avg_excluded_per_day"]),
        ("avg_active_per_day",   avg_active,   *IDEAL_RANGES["avg_active_per_day"]),
        ("bear_dd_vs_baseline",  bear_dd_val,  *IDEAL_RANGES["bear_dd_vs_baseline"]),
    ]

    print(f"\n=== 週次 KPI レポート（直近{lookback_weeks}週 / {total_days}日） ===")
    print(f"{'KPI':<28} {'値':>10} {'理想レンジ':>18} {'判定':>6}")
    print("-" * 66)
    for name, val, lo, hi in kpi_items:
        judgment = _in_range(val, lo, hi)
        status   = "✅" if judgment else "⚠️"
        if val is None:
            val_str   = "N/A"
            range_str = f"{lo or '-'} ~ {hi or '-'}"
        else:
            val_str   = f"{val:.1f}"
            lo_str    = str(lo) if lo is not None else "-"
            hi_str    = str(hi) if hi is not None else "-"
            range_str = f"{lo_str} ~ {hi_str}"
        print(f"{name:<28} {val_str:>10} {range_str:>18} {status:>6}")

    # 追加警告
    if bear_rate > 40:
        print("⚠️  発動率 > 40%: bear_filter の閾値（lookback/ma200_below_days）を緩める検討を")
    elif bear_rate < 5:
        print("⚠️  発動率 < 5%: bear_filter が実質無効。市場環境かパラメータを確認")

    kpi_dict = {
        "period_days":           total_days,
        "bear_days":             bear_days,
        "bear_rate_pct":         bear_rate,
        "avg_excluded_per_day":  round(avg_excluded, 1),
        "avg_active_per_day":    round(avg_active, 1),
        "bear_dd_pct":           round(bear_dd_val, 2) if bear_dd_val is not None else None,
    }
    return kpi_dict


def weekly_smoke_summary(lookback_weeks: int = 4) -> dict:
    """
    smoke_test_log.jsonl から警告（passed=False）の発生回数を集計して出力。

    Returns
    -------
    dict with keys: total_days, warning_count, recent_warnings
    """
    if not SMOKE_LOG_PATH.exists():
        print("  smoke_test_log.jsonl が存在しません（smoke test 未実行）")
        return {"error": "no smoke log found"}

    records = []
    try:
        with open(SMOKE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
    except Exception as e:
        return {"error": str(e)}

    if not records:
        print("  smoke_test_log.jsonl にレコードがありません")
        return {"total_days": 0, "warning_count": 0, "recent_warnings": []}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=lookback_weeks)
    df = df[df["date"] >= cutoff]

    total_days    = len(df)
    warning_count = int((~df["passed"]).sum()) if "passed" in df.columns else 0
    recent_warn   = (
        df[~df["passed"]]["date"].dt.strftime("%Y-%m-%d").tolist()[-3:]
        if "passed" in df.columns else []
    )

    print(f"\n--- Smoke Test サマリー（直近{lookback_weeks}週） ---")
    print(f"  smoke_test 実行日数: {total_days} 日")
    print(f"  警告回数（passed=False）: {warning_count} / {total_days} 日")
    if recent_warn:
        print(f"  最近の警告日: {recent_warn}")

    return {
        "total_days":      total_days,
        "warning_count":   warning_count,
        "recent_warnings": recent_warn,
    }


if __name__ == "__main__":
    weekly_bear_report(lookback_weeks=4)
    print()
    weekly_kpi_report(lookback_weeks=4)
    weekly_smoke_summary(lookback_weeks=4)
