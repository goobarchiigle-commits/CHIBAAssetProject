"""
build_enriched_signals.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purpose
    OHLCV parquet → signals CSV に実価格・リターン・z-score を付与する。
    live_pipeline.py が要求する z / z_regime の前提データを生成する。

Input A  signals CSV  (date, symbol, position)
         default: data/latest_signals.csv
Input B  OHLCV parquet per symbol
         default: data/backtest_dataset/2026-03-28/{symbol}.parquet
         columns (case-insensitive): open, high, low, close, volume
         index: Date (datetime)

Output   enriched CSV (date, symbol, open, position, ret, z, z_regime)
         default: data/signals_enriched.csv

Modes
─────
live      signal date が OHLCV 末端を超える場合、最終利用可能 OHLCV 日を
          使用してフォールバック。ギャップ > GAP_WARN_DAYS なら警告。
backtest  OHLCV 範囲内の signal 日を全て処理する。

No-lookahead guarantee
──────────────────────
z at date T = cross-sectional z-score of 1-period open return at T.
OHLCV は end_date 以前のデータのみ読み込む（T+1 以降を使わない）。

Usage
─────
python src/scripts/build_enriched_signals.py
python src/scripts/build_enriched_signals.py --mode backtest --lookback 120
python src/scripts/build_enriched_signals.py --inplace
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# project root を sys.path に追加（直接実行時）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.strategy.score import enrich_with_z  # noqa: E402

# ── デフォルトパス ─────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_SIGNALS  = _ROOT / "data" / "latest_signals.csv"
DEFAULT_OHLCV    = _ROOT / "data" / "backtest_dataset" / "2026-03-28"
DEFAULT_OUTPUT   = _ROOT / "data" / "signals_enriched.csv"
DEFAULT_LOOKBACK = 60    # z 計算に使う過去日数（カレンダー日）
GAP_WARN_DAYS    = 30    # signal 日と OHLCV 末端の許容ギャップ

# ── スキーマ ───────────────────────────────────────────────────────────────────
SIGNALS_REQUIRED = {"date", "symbol", "position"}
OUTPUT_COLS      = ["date", "symbol", "open", "position", "ret", "z", "z_regime"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loader: signals
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(path: Path) -> pd.DataFrame:
    """
    signals CSV を読み込み、スキーマを検証して返す。

    Returns
    -------
    DataFrame: date (datetime64), symbol (str), position (numeric), ...
    """
    if not path.exists():
        raise FileNotFoundError(f"signals file not found: {path}")

    df = pd.read_csv(path)

    missing = SIGNALS_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {missing}")

    df["date"]   = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loader: OHLCV panel
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv_panel(
    symbols:    list[str],
    ohlcv_dir:  Path,
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
) -> pd.DataFrame:
    """
    symbols × date の OHLCV パネルを構築する。

    Parameters
    ----------
    symbols    : 対象銘柄リスト
    ohlcv_dir  : parquet ファイルが格納されたディレクトリ
    start_date : 読み込み開始日（lookback を含む）
    end_date   : 読み込み終了日（signal 日または OHLCV 末端）

    Returns
    -------
    DataFrame: date (datetime64), symbol (str), open, close, volume
    欠損銘柄はスキップして警告。
    """
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for sym in symbols:
        path = ohlcv_dir / f"{sym}.parquet"
        if not path.exists():
            skipped.append(sym)
            continue

        try:
            raw = pd.read_parquet(path)
        except Exception as exc:
            warnings.warn(f"[SKIP] {sym}: parquet read failed: {exc}")
            skipped.append(sym)
            continue

        # 列名を小文字に統一（parquet は "Open"/"Close" 等）
        raw.columns = raw.columns.str.lower()
        raw.index   = pd.to_datetime(raw.index)

        if "open" not in raw.columns:
            warnings.warn(f"[SKIP] {sym}: 'open' column missing")
            skipped.append(sym)
            continue

        # 日付フィルタ（先読みなし）
        raw = raw.loc[start_date:end_date, ["open", "close"]]
        if raw.empty:
            warnings.warn(f"[SKIP] {sym}: no OHLCV data in range [{start_date.date()}, {end_date.date()}]")
            skipped.append(sym)
            continue

        raw = raw.reset_index().rename(columns={"index": "date", "Date": "date", "date": "date"})
        raw["symbol"] = sym
        frames.append(raw[["date", "symbol", "open", "close"]])

    if skipped:
        warnings.warn(f"[WARN] {len(skipped)} symbol(s) skipped (no parquet): {skipped}")

    if not frames:
        raise ValueError("No OHLCV data loaded — check ohlcv_dir and symbol list")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core: enrich
# ─────────────────────────────────────────────────────────────────────────────

def enrich(
    signals:      pd.DataFrame,
    ohlcv_panel:  pd.DataFrame,
    effective_date: pd.Timestamp,
    signal_date:    pd.Timestamp,
) -> pd.DataFrame:
    """
    OHLCV パネルから z / z_regime を計算し、signals に結合する。

    Parameters
    ----------
    signals        : 元の signals DataFrame
    ohlcv_panel    : load_ohlcv_panel の出力
    effective_date : z 計算・価格取得に使う実際の OHLCV 日付
                     (signal_date と異なる場合は live モードのフォールバック)
    signal_date    : 元の signal 日付（出力に保持）

    Returns
    -------
    DataFrame: OUTPUT_COLS
    """
    # enrich_with_z: ohlcv_panel 全体に ret / z / z_regime を付与
    enriched = enrich_with_z(ohlcv_panel)

    # effective_date の行を抽出
    snap = enriched[enriched["date"] == effective_date].copy()
    if snap.empty:
        raise ValueError(
            f"effective_date {effective_date.date()} not found in enriched OHLCV. "
            "Check date range and OHLCV coverage."
        )

    # signals とマージ（symbol 単位）
    # signals に既存 open 列があれば削除（OHLCV の実価格で上書きする）
    snap = snap.rename(columns={"date": "ohlcv_date"})
    signals_clean = signals.drop(columns=["open"], errors="ignore")
    merged = signals_clean.merge(
        snap[["symbol", "open", "ret", "z", "z_regime"]],
        on="symbol",
        how="left",
    )

    # open が取れなかった銘柄を警告
    missing_open = merged[merged["open"].isna()]["symbol"].tolist()
    if missing_open:
        warnings.warn(f"[WARN] open price missing for {len(missing_open)} symbol(s): {missing_open}")

    # signal_date を使用（effective_date で上書きしない）
    merged["date"] = signal_date

    return merged[OUTPUT_COLS].dropna(subset=["open"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Effective date resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_effective_date(
    signal_date: pd.Timestamp,
    ohlcv_max:   pd.Timestamp,
    mode:        str,
) -> pd.Timestamp:
    """
    signal_date が OHLCV 末端を超えている場合の解決ロジック。

    live    : ohlcv_max にフォールバック（ギャップ > GAP_WARN_DAYS で警告）
    backtest: signal_date が ohlcv_max を超えていたら ValueError
    """
    if signal_date <= ohlcv_max:
        return signal_date

    gap = (signal_date - ohlcv_max).days
    if mode == "backtest":
        raise ValueError(
            f"backtest mode: signal_date {signal_date.date()} "
            f"exceeds OHLCV end {ohlcv_max.date()} by {gap} days. "
            "Extend OHLCV dataset or switch to --mode live."
        )

    if gap > GAP_WARN_DAYS:
        warnings.warn(
            f"[WARN] signal_date {signal_date.date()} is {gap} days beyond "
            f"OHLCV end {ohlcv_max.date()} (threshold: {GAP_WARN_DAYS}). "
            "Price and z-score may be stale.",
            stacklevel=2,
        )
    else:
        print(
            f"[INFO] signal_date {signal_date.date()} beyond OHLCV end "
            f"{ohlcv_max.date()} ({gap}d gap) — using OHLCV end date."
        )

    return ohlcv_max


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def build(
    signals_path: Path,
    ohlcv_dir:    Path,
    output_path:  Path,
    lookback:     int,
    mode:         str,
) -> pd.DataFrame:
    """
    enriched signals を構築して output_path に書き込む。

    Returns
    -------
    enriched DataFrame (OUTPUT_COLS)
    """
    # ── Load signals ──────────────────────────────────────────────────────────
    signals = load_signals(signals_path)
    symbols      = signals["symbol"].unique().tolist()
    signal_dates = sorted(signals["date"].unique())

    print(f"[INFO] signals  : {len(signals)} rows, {len(symbols)} symbols, "
          f"{len(signal_dates)} date(s)")
    print(f"[INFO] ohlcv_dir: {ohlcv_dir}")

    # ── Determine date range for OHLCV load ───────────────────────────────────
    max_signal_date = max(signal_dates)
    min_signal_date = min(signal_dates)
    start_date = min_signal_date - pd.Timedelta(days=lookback)

    # ── Load OHLCV panel ──────────────────────────────────────────────────────
    ohlcv = load_ohlcv_panel(symbols, ohlcv_dir, start_date, max_signal_date)
    ohlcv_max = ohlcv["date"].max()

    print(f"[INFO] OHLCV panel: {len(ohlcv)} rows, "
          f"date range [{ohlcv['date'].min().date()}, {ohlcv_max.date()}]")

    # ── Process each signal date ──────────────────────────────────────────────
    all_enriched: list[pd.DataFrame] = []

    for sig_date in signal_dates:
        eff_date = resolve_effective_date(sig_date, ohlcv_max, mode)

        # signals for this date only
        signals_on_date = signals[signals["date"] == sig_date].copy()

        try:
            enriched_slice = enrich(signals_on_date, ohlcv, eff_date, sig_date)
        except ValueError as exc:
            warnings.warn(f"[SKIP] date {sig_date.date()}: {exc}")
            continue

        all_enriched.append(enriched_slice)

        # ── Inject prev trading day ───────────────────────────────────────────
        # Ensures output has ≥2 dates so live_pipeline never needs to concat
        # with signals_history.csv (which may have placeholder open prices).
        ohlcv_dates = sorted(ohlcv["date"].unique())
        eff_idx     = ohlcv_dates.index(eff_date) if eff_date in ohlcv_dates else -1
        if eff_idx > 0:
            prev_date = ohlcv_dates[eff_idx - 1]
            try:
                prev_slice = enrich(signals_on_date, ohlcv, prev_date, prev_date)
                all_enriched.insert(0, prev_slice)
            except ValueError:
                pass  # prev day not available — tolerate

    if not all_enriched:
        raise RuntimeError("No signal dates could be enriched — check OHLCV coverage.")

    result = pd.concat(all_enriched, ignore_index=True)

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[INFO] output   : {output_path}  ({len(result)} rows)")
    _print_summary(result)

    return result


def _print_summary(df: pd.DataFrame) -> None:
    """z / z_regime / score の分布サマリを標準出力する。"""
    print("\n── Score summary ──────────────────────────────────")
    print(f"  rows           : {len(df)}")
    print(f"  z  mean={df['z'].mean():.4f}  std={df['z'].std():.4f}  "
          f"min={df['z'].min():.4f}  max={df['z'].max():.4f}")
    print(f"  z_regime dist  :\n{df['z_regime'].value_counts().to_string()}")
    print(f"  open  mean={df['open'].mean():.2f}  "
          f"min={df['open'].min():.2f}  max={df['open'].max():.2f}")
    print("────────────────────────────────────────────────────\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OHLCV + signals → enriched signals (open / ret / z / z_regime)"
    )
    ap.add_argument("--signals",  type=Path, default=DEFAULT_SIGNALS,
                    help="input signals CSV (date, symbol, position)")
    ap.add_argument("--ohlcv-dir", type=Path, default=DEFAULT_OHLCV,
                    help="directory containing {symbol}.parquet files")
    ap.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,
                    help="output enriched CSV path")
    ap.add_argument("--lookback", type=int,  default=DEFAULT_LOOKBACK,
                    help="calendar days of OHLCV history for z computation")
    ap.add_argument("--mode",     choices=["live", "backtest"], default="live",
                    help="live: tolerate signal date beyond OHLCV end; "
                         "backtest: error if signal date > OHLCV end")
    ap.add_argument("--inplace",  action="store_true",
                    help="overwrite --signals file instead of writing to --output")
    args = ap.parse_args()

    output = args.signals if args.inplace else args.output

    try:
        build(
            signals_path=args.signals,
            ohlcv_dir=args.ohlcv_dir,
            output_path=output,
            lookback=args.lookback,
            mode=args.mode,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(f"[FATAL] {exc}")


if __name__ == "__main__":
    main()
