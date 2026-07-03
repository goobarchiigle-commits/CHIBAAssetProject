from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.market.jpx_calendar import JPXCalendar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "logs" / "diagnostics" / "metrics.jsonl"
SIGNALS_DIR = PROJECT_ROOT / "data" / "signals"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "signal_funnel_60d.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "signal_funnel_summary.txt"
LOOKBACK_DAYS = 60


def _safe_rate(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _load_latest_metrics_by_date(path: Path) -> dict[str, dict]:
    by_date: dict[str, dict] = {}
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        date = row.get("date")
        run_at = row.get("run_at", "")
        if not date:
            continue
        if date not in by_date or run_at > by_date[date].get("run_at", ""):
            by_date[date] = row
    return by_date


def _count_executed_buys(data: dict) -> int:
    send_results = data.get("send_results", [])
    if send_results:
        return sum(
            1
            for row in send_results
            if row.get("side") == "BUY" and bool(row.get("success"))
        )

    orders = data.get("orders", [])
    return sum(1 for row in orders if row.get("side") == "BUY")


def _load_executed_counts(signal_dir: Path) -> dict[str, int]:
    by_date: dict[str, tuple[str, int]] = {}
    if not signal_dir.exists():
        return {}

    for path in sorted(signal_dir.glob("signal_*_executed.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        date = data.get("data_as_of")
        generated_at = data.get("generated_at", "")
        if not date:
            continue

        executed_count = _count_executed_buys(data)
        if date not in by_date or generated_at > by_date[date][0]:
            by_date[date] = (generated_at, executed_count)

    return {date: count for date, (_, count) in by_date.items()}


def _select_last_business_dates(
    available_dates: list[str],
    lookback_days: int,
) -> list[str]:
    if not available_dates:
        return []

    cal = JPXCalendar()
    latest = pd.Timestamp(max(available_dates))
    selected: list[str] = []
    cur = latest
    seen = set(available_dates)

    while len(selected) < lookback_days:
        cur_str = cur.date().isoformat()
        if cal.is_trading_day(cur) and cur_str in seen:
            selected.append(cur_str)
        prev = cal.prev_trading_day(cur)
        if prev >= cur:
            break
        cur = prev
        if cur.year < 2015:
            break

    return sorted(selected)


def build_funnel_frame(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    metrics_by_date = _load_latest_metrics_by_date(METRICS_PATH)
    executed_counts = _load_executed_counts(SIGNALS_DIR)
    selected_dates = _select_last_business_dates(sorted(metrics_by_date), lookback_days)

    rows: list[dict] = []
    for date in selected_dates:
        row = metrics_by_date[date]
        universe_count = int(row.get("universe_size", 0) or 0)
        filter_pass_count = int(row.get("rsr_pass_count", 0) or 0)
        buy_signal_count = int(row.get("candidate_count", 0) or 0)
        executed_count = int(executed_counts.get(date, 0) or 0)

        rows.append(
            {
                "date": date,
                "universe_count": universe_count,
                "filter_pass_count": filter_pass_count,
                "buy_signal_count": buy_signal_count,
                "executed_count": executed_count,
                "filter_pass_rate": round(_safe_rate(filter_pass_count, universe_count), 6),
                "signal_rate": round(_safe_rate(buy_signal_count, filter_pass_count), 6),
                "execution_rate": round(_safe_rate(executed_count, buy_signal_count), 6),
            }
        )

    return pd.DataFrame(rows)


def _fmt_num(value: float) -> str:
    return f"{value:.2f}"


def _fmt_rate(value: float) -> str:
    return f"{value:.2%}"


def build_summary(df: pd.DataFrame, requested_days: int = LOOKBACK_DAYS) -> str:
    if df.empty:
        return (
            "signal funnel summary\n"
            "no rows were generated from logs/diagnostics/metrics.jsonl\n"
        )

    available_days = len(df)
    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]

    avg_rates = {
        "filter_pass_rate": float(df["filter_pass_rate"].mean()),
        "signal_rate": float(df["signal_rate"].mean()),
        "execution_rate": float(df["execution_rate"].mean()),
    }
    bottleneck_stage = min(avg_rates, key=avg_rates.get)
    pre_execution_rates = {
        "filter_pass_rate": avg_rates["filter_pass_rate"],
        "signal_rate": avg_rates["signal_rate"],
    }
    pre_execution_bottleneck_stage = min(pre_execution_rates, key=pre_execution_rates.get)

    lines = [
        "signal funnel summary",
        f"requested_business_days: {requested_days}",
        f"available_business_days: {available_days}",
        f"date_range: {start_date} to {end_date}",
        "",
        "counts",
    ]

    count_cols = [
        "universe_count",
        "filter_pass_count",
        "buy_signal_count",
        "executed_count",
    ]
    for col in count_cols:
        lines.append(
            f"{col}: avg={_fmt_num(float(df[col].mean()))}, "
            f"min={int(df[col].min())}, max={int(df[col].max())}"
        )

    lines.extend(["", "rates"])
    rate_cols = ["filter_pass_rate", "signal_rate", "execution_rate"]
    for col in rate_cols:
        lines.append(
            f"{col}: avg={_fmt_rate(float(df[col].mean()))}, "
            f"min={_fmt_rate(float(df[col].min()))}, max={_fmt_rate(float(df[col].max()))}"
        )

    lines.extend(
        [
            "",
            f"bottleneck_stage: {bottleneck_stage}",
            f"bottleneck_avg_rate: {_fmt_rate(avg_rates[bottleneck_stage])}",
            f"pre_execution_bottleneck_stage: {pre_execution_bottleneck_stage}",
            f"pre_execution_bottleneck_avg_rate: {_fmt_rate(pre_execution_rates[pre_execution_bottleneck_stage])}",
        ]
    )

    if available_days < requested_days:
        lines.extend(
            [
                "",
                "note: fewer than 60 business days were available in metrics.jsonl,",
                "so the report uses all available observed business days only.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_funnel_frame(LOOKBACK_DAYS)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_SUMMARY.write_text(build_summary(df, LOOKBACK_DAYS), encoding="utf-8")
    print(f"saved: {OUTPUT_CSV}")
    print(f"saved: {OUTPUT_SUMMARY}")
    print(f"rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
