"""
verify_dataset.py
スナップショットデータの整合性検証スクリプト

使い方:
  python verify_dataset.py                      # デフォルトバージョン
  DATA_VERSION=2026-03-28 python verify_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_VERSION: str  = os.environ.get("DATA_VERSION", "2026-03-28")
SNAPSHOT_DIR: Path = Path("data/backtest_dataset") / DATA_VERSION

MIN_ROWS      = 2000      # 最低行数（2018-01以降のデータで約1500日以上）
MAX_NAN_RATIO = 0.005     # Close列のNaN許容率（0.5%以下）
CHECK_COLS    = ["Open", "High", "Low", "Close", "Volume"]


def load_meta() -> dict:
    meta_path = SNAPSHOT_DIR / "_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def verify_all() -> None:
    print("=" * 60)
    print(f"  データ整合性検証")
    print(f"  バージョン: {DATA_VERSION}")
    print(f"  検証先:     {SNAPSHOT_DIR}")
    print("=" * 60)

    if not SNAPSHOT_DIR.exists():
        print(f"\n❌ スナップショットディレクトリが存在しません: {SNAPSHOT_DIR}")
        print(f"  先に実行: python build_dataset_snapshot.py")
        sys.exit(1)

    meta = load_meta()
    if meta:
        print(f"\n  メタ情報: saved_ok={meta.get('saved_ok')} / "
              f"total={meta.get('total_symbols')} / "
              f"hash={meta.get('snapshot_hash')}")

    parquet_files = sorted(SNAPSHOT_DIR.glob("*.parquet"))
    print(f"\n  ファイル数: {len(parquet_files)}\n")

    ok_list  = []
    ng_list  = []

    for path in parquet_files:
        sym = path.stem
        if sym.startswith("_"):
            continue

        issues = []

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            ng_list.append((sym, f"READ_ERROR: {e}"))
            print(f"  ❌ {sym:<10} READ_ERROR: {e}")
            continue

        # ① 行数チェック
        if len(df) < MIN_ROWS:
            issues.append(f"short_history={len(df)}")

        # ② インデックス順序
        if not df.index.is_monotonic_increasing:
            issues.append("index_not_sorted")

        # ③ 必須列の存在
        missing_cols = [c for c in CHECK_COLS if c not in df.columns]
        if missing_cols:
            issues.append(f"missing_cols={missing_cols}")

        # ④ NaN率チェック（Closeのみ）
        if "Close" in df.columns:
            nan_rate = df["Close"].isna().mean()
            if nan_rate > MAX_NAN_RATIO:
                issues.append(f"high_nan_close={nan_rate:.2%}")

        # ⑤ ゼロ価格チェック（データ破損の典型）
        if "Close" in df.columns:
            zero_rows = (df["Close"] == 0).sum()
            if zero_rows > 0:
                issues.append(f"zero_price={zero_rows}行")

        # ⑥ 日付範囲
        date_from = df.index[0].date() if len(df) > 0 else "N/A"
        date_to   = df.index[-1].date() if len(df) > 0 else "N/A"

        if issues:
            ng_list.append((sym, "; ".join(issues)))
            print(f"  ⚠ {sym:<10} {'; '.join(issues)}")
        else:
            ok_list.append(sym)
            print(f"  ✓ {sym:<10} rows={len(df):,}  {date_from} → {date_to}")

    # サマリー
    print("\n" + "=" * 60)
    print(f"  OK:  {len(ok_list):3d} 銘柄")
    print(f"  NG:  {len(ng_list):3d} 銘柄")

    if ng_list:
        print(f"\n  問題銘柄:")
        for sym, reason in ng_list:
            print(f"    {sym:<10} {reason}")
        print(f"\n  再取得コマンド:")
        ng_syms = " ".join(s for s, _ in ng_list)
        print(f"    python build_dataset_snapshot.py --force --symbols {ng_syms}")
        sys.exit(1)
    else:
        print(f"\n  ✅ 全銘柄OK。このスナップショットでバックテストを実行できます。")
        print(f"\n  実行例:")
        print(f"    DATA_VERSION={DATA_VERSION} python -m backtest.composite_alpha_bt")


if __name__ == "__main__":
    verify_all()
