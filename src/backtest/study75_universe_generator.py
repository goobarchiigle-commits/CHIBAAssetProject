"""
src/backtest/study75_universe_generator.py
Study75A — Point-in-time Universe Generator。

PIT設計（reports/study75_pit_audit.md 準拠）:
  rebalance_date（月次第1営業日・ユニバース発効日）と snapshot_date（rebalance_dateのT-1営業日・
  ユニバース判定に使うデータの基準日）を明確に分離する。ADV20・lot feasibility・listing_age は
  すべて snapshot_date までのデータのみを参照し、rebalance_date当日の情報は一切使わない。

Universe C ルール:
  - ADV20（snapshot_date時点のtrailing 20営業日平均売買代金） >= ¥300,000,000
  - lot feasible（1単元コスト <= 有効資本 × max_lot_cost_ratio）
  - listing_age >= min_history_days(60営業日)（左検閲銘柄=データセット開始日から存在する銘柄は
    真の上場日が不明なため本フィルターの対象外とする。§IPO handling注記参照）

Lot feasibility の既知の限界（reports/study75_pit_audit.md §3・PARTIAL判定）:
  単元株数（trading unit size）は本データセットに存在しない。LOT_SIZE=100を全期間一律で
  適用するが、2018-10以前（東証単元株数統一キャンペーン中）はこの仮定が一部銘柄で
  不正確である可能性がある。診断ファイルに lot_size_verified フラグとして記録する。

出力:
  backtests/study75_rule_universe.json      {"monthly_universe": {"YYYY-MM-DD": ["13010", ...]}}
  backtests/study75_universe_diagnostics.parquet
    列: date, code, adv20, listing_age, lot_feasible, first_seen, last_seen, excluded_reason,
        lot_size_verified
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json

import pandas as pd

from src.market.jpx_calendar import JPXCalendar
from src.paths import JQUANTS_PROCESSED_DIR, RESULTS_DIR

LOT_SIZE = 100
CAPITAL = 3_000_000              # PARAMS_LOCKED capital（strategy.yaml）
MAX_LOT_COST_RATIO = 0.30        # strategy.yaml max_lot_cost_ratio
LOT_COST_THRESHOLD = CAPITAL * MAX_LOT_COST_RATIO  # 900,000円
ADV20_THRESHOLD_JPY = 300_000_000  # 本タスク仕様の明示値（Universe Design記述分析の3,000万円とは別値）
MIN_HISTORY_DAYS = 60             # 営業日
ADV20_WINDOW_TRADING_DAYS = 20
UNIT_UNIFICATION_DATE = pd.Timestamp("2018-10-01")  # 東証単元株数統一完了（周知の市場構造上の事実）

RULE_UNIVERSE_FILE = RESULTS_DIR / "study75_rule_universe.json"
DIAGNOSTICS_FILE = RESULTS_DIR / "study75_universe_diagnostics.parquet"


def _load_price_panel() -> pd.DataFrame:
    year_files = sorted(JQUANTS_PROCESSED_DIR.glob("daily_bars_*.parquet"))
    frames = [pd.read_parquet(f, columns=["Date", "Code", "Close", "Volume"]) for f in year_files]
    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["TradedValue"] = panel["Close"] * panel["Volume"]
    return panel.sort_values(["Code", "Date"]).reset_index(drop=True)


def _load_universe_reference() -> pd.DataFrame:
    """
    processed/universe.parquet（universe.materialize_universe_reference()の成果物）を読む。
    first_seen_date はイベントログのADD日でありUniverse Cの判定（listing_age）にそのまま使える。

    注意: universe.parquet の last_seen_date は「最後に記録されたイベントの日付」であり、
    継続上場中（REMOVEイベントが一度も無い）銘柄では実質的に first_seen_date と同じ値になる
    （イベントソーシングはADD/REMOVE等の状態遷移のみを記録し、「今日も引き続き上場中」を
    毎日記録する設計ではないため）。診断用途としては誤解を招くため、本モジュールでは
    価格パネル側の実際の最終出現日で上書きする（下記 _attach_actual_last_seen 参照）。
    last_seen はどの判定ロジックにも使わない診断専用列であり、データセット全期間
    （snapshot_dateより未来の日付を含む）から算出する事後監査情報として明示的に扱う
    （Universe C自体の合否判定には一切使わないためlookahead biasは生じない）。
    """
    ref = pd.read_parquet(JQUANTS_PROCESSED_DIR / "universe.parquet")
    ref["first_seen_date"] = pd.to_datetime(ref["first_seen_date"])
    ref["last_seen_date"] = pd.to_datetime(ref["last_seen_date"])
    return ref


def _actual_last_seen_map(panel: pd.DataFrame) -> dict:
    """価格パネル全体（snapshot_dateより未来を含む）から銘柄別の実際の最終出現日を求める。
    診断専用（事後監査情報）。Universe C の合否判定には使わない。"""
    return panel.groupby("Code")["Date"].max().to_dict()


def _monthly_rebalance_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """データに実在する各月の最初の営業日を月次リバランス日として使う（暦上の計算ではなく実データ基準）。"""
    dates = panel["Date"].drop_duplicates().sort_values()
    ym = dates.dt.to_period("M")
    first_of_month = dates.groupby(ym).min()
    return sorted(first_of_month.tolist())


def build_universe_c(panel: pd.DataFrame, universe_ref: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    calendar = JPXCalendar()
    dataset_start = panel["Date"].min()
    # データセット開始日から数営業日以内に存在した銘柄＝真の上場日が不明な左検閲銘柄。
    # min_history_daysの対象外とする（さもないと全既存銘柄が一斉に「新規上場扱い」される誤り）。
    left_censor_cutoff = calendar.add_trading_days(dataset_start, 3)
    first_seen_map = universe_ref.set_index("code")["first_seen_date"].to_dict()
    last_seen_map = _actual_last_seen_map(panel)  # 診断専用・価格パネル実測（事後監査情報）

    rebalance_dates = _monthly_rebalance_dates(panel)
    monthly_universe: dict[str, list[str]] = {}
    diag_rows: list[dict] = []

    for rebalance_date in rebalance_dates:
        snapshot_date = calendar.prev_trading_day(rebalance_date)
        if snapshot_date < dataset_start:
            continue  # 初月はT-1データが存在しないためスキップ

        window_start = snapshot_date - pd.Timedelta(days=45)  # 20営業日を確保する余裕枠
        window = panel.loc[(panel["Date"] > window_start) & (panel["Date"] <= snapshot_date)]
        adv = window.groupby("Code").agg(
            adv20=("TradedValue", "mean"),
            n_days=("Date", "count"),
            last_close=("Close", "last"),
        )
        adv = adv.loc[adv["n_days"] >= 15]  # 直近20営業日ウィンドウ内で最低15日分のデータを要求
        adv["lot_cost"] = adv["last_close"] * LOT_SIZE
        adv["lot_feasible"] = adv["lot_cost"] <= LOT_COST_THRESHOLD

        rebalance_date_str = rebalance_date.strftime("%Y-%m-%d")
        lot_size_verified = bool(snapshot_date >= UNIT_UNIFICATION_DATE)
        passing: list[str] = []

        for code, row in adv.iterrows():
            first_seen = first_seen_map.get(code, pd.NaT)
            last_seen = last_seen_map.get(code, pd.NaT)

            is_left_censored = pd.notna(first_seen) and first_seen <= left_censor_cutoff
            if is_left_censored:
                listing_age = None  # 真の上場日不明・フィルター対象外
            else:
                listing_age = (snapshot_date - first_seen).days if pd.notna(first_seen) else None

            excluded_reason = ""
            if not is_left_censored and (listing_age is None or listing_age < MIN_HISTORY_DAYS):
                excluded_reason = "insufficient_history"
            elif not bool(row["lot_feasible"]):
                excluded_reason = "lot_infeasible"
            elif row["adv20"] < ADV20_THRESHOLD_JPY:
                excluded_reason = "liquidity_insufficient"

            if not excluded_reason:
                passing.append(code)

            diag_rows.append({
                "date": rebalance_date_str,
                "code": code,
                "adv20": float(row["adv20"]),
                "listing_age": listing_age,
                "lot_feasible": bool(row["lot_feasible"]),
                "first_seen": first_seen.strftime("%Y-%m-%d") if pd.notna(first_seen) else None,
                "last_seen": last_seen.strftime("%Y-%m-%d") if pd.notna(last_seen) else None,
                "excluded_reason": excluded_reason,
                "lot_size_verified": lot_size_verified,
            })

        monthly_universe[rebalance_date_str] = sorted(passing)

    diagnostics = pd.DataFrame(diag_rows)
    return monthly_universe, diagnostics


def main() -> int:
    print("[1/4] Loading price panel...")
    panel = _load_price_panel()
    print(f"  rows={len(panel):,} symbols={panel['Code'].nunique():,}")

    print("[2/4] Loading universe reference (first_seen/last_seen)...")
    universe_ref = _load_universe_reference()
    print(f"  reference_rows={len(universe_ref):,}")

    print("[3/4] Building Universe C (monthly rebalance, T-1 snapshot)...")
    monthly_universe, diagnostics = build_universe_c(panel, universe_ref)
    n_months = len(monthly_universe)
    avg_size = sum(len(v) for v in monthly_universe.values()) / max(1, n_months)
    print(f"  months={n_months} avg_universe_size={avg_size:.1f}")

    print("[4/4] Writing outputs...")
    RULE_UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RULE_UNIVERSE_FILE.write_text(
        json.dumps({"monthly_universe": monthly_universe}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diagnostics.to_parquet(DIAGNOSTICS_FILE)
    print(f"  Saved: {RULE_UNIVERSE_FILE}")
    print(f"  Saved: {DIAGNOSTICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
