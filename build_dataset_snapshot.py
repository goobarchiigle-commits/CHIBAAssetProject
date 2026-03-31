"""
build_dataset_snapshot.py
バックテスト用データスナップショット作成スクリプト

【目的】
  yfinance はセッション差・企業アクション後調整により毎回異なる価格を返す。
  研究の再現性を保証するため、データを特定バージョンに凍結する。

【設計原則】
  - バックテスト用 (immutable) と ライブ用 (更新) を完全分離
  - auto_adjust=False で生データを保存（調整は自前で制御可能）
  - 1銘柄1ファイル = 部分更新・並列化・デバッグが容易

【使い方】
  # 初回（デフォルトバージョン 2026-03-28 で作成）
  python build_dataset_snapshot.py

  # 新バージョン作成（既存は変更しない）
  DATA_VERSION=2026-06-30 python build_dataset_snapshot.py

  # 特定銘柄だけ再取得（強制上書き）
  python build_dataset_snapshot.py --force --symbols 6981.T 8035.T

【出力】
  data/backtest_dataset/{DATA_VERSION}/
    {symbol}.parquet        # OHLCV + Dividends + Stock Splits
    _meta.json              # バージョンメタ情報（ハッシュ付き）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from paths import BACKTEST_DATASET_DIR, RSR_UNIVERSE_FILE  # .env 読み込み込み

# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #
DATA_VERSION: str = os.environ.get("DATA_VERSION", "2026-03-28")
START:        str = "2010-01-01"   # 252日RSR計算には252日の先行データが必要
END:          str = "2026-04-01"   # 今日以降は取得されない

SNAPSHOT_DIR: Path = BACKTEST_DATASET_DIR / DATA_VERSION
TOPIX_SYMBOL: str  = "^TOPIX"     # TOPIX指数

# ------------------------------------------------------------------ #
# ユニバース収集（全スクリプトで使う銘柄を網羅）
# ------------------------------------------------------------------ #
def _collect_all_symbols() -> dict[str, str]:
    """バックテスト・拡張検証で使う全銘柄を収集する。"""
    all_syms: dict[str, str] = {}

    # ① RSR42ユニバース（メインバックテスト）
    rsr_csv = RSR_UNIVERSE_FILE
    if rsr_csv.exists():
        df = pd.read_csv(rsr_csv)
        for _, row in df.iterrows():
            all_syms[row["symbol"]] = row.get("sector", "不明")

    # ② TOPIX100拡張ユニバース
    try:
        from backtest.universe_builder import TOPIX100_TICKERS
        all_syms.update(TOPIX100_TICKERS)
    except ImportError:
        pass

    return all_syms


# ------------------------------------------------------------------ #
# ダウンロード関数
# ------------------------------------------------------------------ #
def download_symbol(sym: str, start: str, end: str, force: bool = False) -> bool:
    """
    1銘柄をダウンロードして保存。スナップショットが既にあればスキップ。

    Parameters
    ----------
    sym   : ticker symbol (例 "8035.T")
    start : 開始日 "YYYY-MM-DD"
    end   : 終了日 "YYYY-MM-DD"
    force : True なら既存スナップショットを上書き

    Returns
    -------
    True: 保存成功 or スキップ、False: 失敗
    """
    path = SNAPSHOT_DIR / f"{sym}.parquet"

    if path.exists() and not force:
        print(f"  SKIP {sym:<8} (already exists)")
        return True

    try:
        warnings.filterwarnings("ignore")
        df = yf.download(
            sym,
            start=start,
            end=end,
            auto_adjust=False,  # 生データ保存（再現性最優先）
            actions=True,       # Dividends + Stock Splits も保存
            progress=False,
            threads=False,      # 並列DLはタイミング差を生む → 直列固定
        )

        if df.empty:
            print(f"  FAIL {sym:<8} (empty data)")
            return False

        # MultiIndex（yfinance v0.2+）→ 1階層
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)

        df.to_parquet(path)
        rows = len(df)
        print(f"  SAVE {sym:<8} rows={rows:,}  {df.index[0].date()} → {df.index[-1].date()}")
        return True

    except Exception as e:
        print(f"  FAIL {sym:<8} error={e}")
        return False


# ------------------------------------------------------------------ #
# データ整合性チェック
# ------------------------------------------------------------------ #
def verify_snapshot(symbols: list[str]) -> dict:
    """保存済みスナップショットの整合性を確認する。"""
    ok = []
    ng = []

    for sym in symbols:
        path = SNAPSHOT_DIR / f"{sym}.parquet"
        if not path.exists():
            ng.append((sym, "missing"))
            continue

        try:
            df = pd.read_parquet(path)

            nan_count = int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())
            if nan_count > 0:
                ng.append((sym, f"NaN={nan_count}"))
                continue

            if not df.index.is_monotonic_increasing:
                ng.append((sym, "index_not_sorted"))
                continue

            if len(df) < 2000:
                ng.append((sym, f"short_history={len(df)}"))
                continue

            ok.append(sym)

        except Exception as e:
            ng.append((sym, f"read_error={e}"))

    return {"ok": ok, "ng": ng}


# ------------------------------------------------------------------ #
# ハッシュ計算
# ------------------------------------------------------------------ #
def compute_snapshot_hash(symbols: list[str]) -> str:
    """全銘柄のCloseデータからMD5ハッシュを計算する（再現性チェック用）。"""
    frames = []
    for sym in sorted(symbols):
        path = SNAPSHOT_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "Adj Close" in df.columns:
            frames.append(df["Adj Close"].rename(sym))
        elif "Close" in df.columns:
            frames.append(df["Close"].rename(sym))

    if not frames:
        return "empty"

    combined = pd.concat(frames, axis=1).round(4)
    raw_bytes = pd.util.hash_pandas_object(combined, index=True).values.tobytes()
    return hashlib.md5(raw_bytes).hexdigest()[:16]


# ------------------------------------------------------------------ #
# メタ情報保存
# ------------------------------------------------------------------ #
def save_meta(all_syms: dict[str, str], ok: list[str], ng: list, snapshot_hash: str) -> None:
    meta = {
        "dataset_version": DATA_VERSION,
        "start":           START,
        "end":             END,
        "total_symbols":   len(all_syms),
        "saved_ok":        len(ok),
        "failed":          [(s, r) for s, r in ng],
        "snapshot_hash":   snapshot_hash,
        "auto_adjust":     False,
        "actions":         True,
    }
    meta_path = SNAPSHOT_DIR / "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n  メタ情報保存: {meta_path}")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="バックテスト用データスナップショット作成")
    parser.add_argument("--force",   action="store_true", help="既存ファイルを上書き")
    parser.add_argument("--symbols", nargs="*",           help="対象銘柄（省略時は全銘柄）")
    parser.add_argument("--topix",   action="store_true", help="TOPIX指数も取得")
    args = parser.parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  データスナップショット作成")
    print(f"  バージョン : {DATA_VERSION}")
    print(f"  期間       : {START} 〜 {END}")
    print(f"  保存先     : {SNAPSHOT_DIR}")
    print(f"  auto_adjust: False（生データ保存）")
    print("=" * 60)

    # ユニバース決定
    all_syms = _collect_all_symbols()
    if args.symbols:
        target = {s: all_syms.get(s, "不明") for s in args.symbols}
    else:
        target = all_syms

    print(f"\n対象銘柄: {len(target)}銘柄\n")

    # ① TOPIX指数（オプション）
    if args.topix:
        print("TOPIX指数取得中...")
        download_symbol(TOPIX_SYMBOL, START, END, force=args.force)
        print()

    # ② 個別銘柄
    ok_syms, failed = [], []
    for i, (sym, sector) in enumerate(target.items(), 1):
        sys.stdout.write(f"[{i:3d}/{len(target)}] ")
        success = download_symbol(sym, START, END, force=args.force)
        if success and (SNAPSHOT_DIR / f"{sym}.parquet").exists():
            ok_syms.append(sym)
        else:
            failed.append((sym, "download_failed"))
        time.sleep(0.3)  # レート制限対策（Yahoo Finance）

    # ③ 整合性チェック
    print("\n整合性チェック中...")
    verify = verify_snapshot(ok_syms)
    ok_verified = verify["ok"]
    ng_verify   = verify["ng"]

    # ④ ハッシュ計算
    print("スナップショットハッシュ計算中...")
    snap_hash = compute_snapshot_hash(ok_verified)

    # ⑤ メタ情報保存
    all_ng = failed + ng_verify
    save_meta(all_syms, ok_verified, all_ng, snap_hash)

    # ⑥ 結果サマリー
    print("\n" + "=" * 60)
    print(f"  完了: {len(ok_verified)}/{len(target)} 銘柄")
    print(f"  DATASET_HASH={snap_hash}")
    if all_ng:
        print(f"\n  問題あり ({len(all_ng)}件):")
        for sym, reason in all_ng:
            print(f"    {sym}: {reason}")
    print(f"\n  次のステップ:")
    print(f"    export DATA_VERSION={DATA_VERSION}")
    print(f"    python -m backtest.composite_alpha_bt  # スナップショット使用")


if __name__ == "__main__":
    main()
