"""
diagnostics/rsr_universe_test.py
RSR母集団テスト: 現行42銘柄 vs TOPIX100 vs TOPIX300プロキシ

目的:
  RSRスコアが低い原因が「ユニバースバイアス」か「市場弱気」かを切り分ける。

  仮説: 42銘柄（強銘柄選定済み）内でRSRを計算すると、
        全員が相対的に似た強さなのでパーセンタイルが圧縮される。
        → TOPIX500内で同じ銘柄を測ると、弱い銘柄との比較でRSR>75になる可能性。

実行:
  cd asset_simulation
  python diagnostics/rsr_universe_test.py
"""
import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from paths import UNIVERSE_DIR, LOGS_DIR

import pandas as pd
import numpy as np
import yfinance as yf

from backtest.rsr import calc_universe_rsr

JST = timezone(timedelta(hours=9))
TODAY = datetime.now(JST).strftime("%Y-%m-%d")

# ------------------------------------------------------------------ #
# ユニバース定義
# ------------------------------------------------------------------ #

# 現行 RSR context（42銘柄）
RSR_42 = [
    "8035.T","6645.T","6702.T","6501.T","6762.T","6920.T","7203.T","7201.T",
    "9432.T","8306.T","8411.T","8309.T","7182.T","8725.T","8058.T","8053.T",
    "8002.T","8001.T","4021.T","2914.T","7011.T","3382.T","5401.T","5411.T",
    "9531.T","9101.T","9104.T","6857.T","6146.T","6981.T","6479.T","6506.T",
    "7012.T","7013.T","9107.T","8015.T","6869.T","6594.T","8354.T","5706.T",
    "3197.T","4055.T",
]

# TOPIX300プロキシ（各セクターの流動性上位）
# 現行42銘柄 + 追加銘柄で幅広い市場を代表
TOPIX300_PROXY = RSR_42 + [
    # 電機・精密（追加）
    "6723.T","6752.T","6758.T","6861.T","6902.T","6954.T","6963.T","7751.T","7974.T",
    # 情報通信
    "3659.T","4689.T","4755.T","9433.T","9437.T","9984.T",
    # 小売
    "2651.T","2802.T","3086.T","7453.T","8267.T","8273.T","9983.T",
    # 不動産
    "3003.T","3289.T","8830.T","8801.T","8802.T",
    # 建設
    "1801.T","1802.T","1803.T","1925.T","1928.T",
    # 医薬品
    "4502.T","4503.T","4519.T","4523.T","4543.T","4568.T","4578.T",
    # 化学
    "4004.T","4183.T","4452.T","4901.T","4911.T","6988.T",
    # 輸送・倉庫
    "9020.T","9021.T","9022.T","9064.T","9143.T","9147.T",
    # 保険・証券
    "8253.T","8303.T","8316.T","8601.T","8604.T","8630.T","8750.T",
    # エネルギー
    "1605.T","5020.T","5019.T",
    # 食品・飲料
    "2503.T","2897.T","2267.T","2871.T",
    # レジャー・サービス
    "4661.T","9602.T","9766.T",
    # 機械（追加）
    "6098.T","6301.T","6326.T","6367.T","7270.T","7272.T",
]
# 重複除去
TOPIX300_PROXY = list(dict.fromkeys(TOPIX300_PROXY))


# ------------------------------------------------------------------ #
# データ取得
# ------------------------------------------------------------------ #
def fetch_prices(tickers: list[str], lookback_days: int = 650) -> dict[str, pd.Series]:
    """yfinanceで終値を取得する（IBD式RSRにはshift(252)が必要なので2年以上）。"""
    from datetime import datetime, timezone, timedelta
    end   = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    start = (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    print(f"  [{len(tickers)}銘柄] データ取得中 ({start}〜{end})...")
    raw = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker")
    prices: dict[str, pd.Series] = {}
    for sym in tickers:
        try:
            s = raw[sym]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
            if len(s) >= 300:
                prices[sym] = s
        except (KeyError, TypeError):
            pass
    print(f"  取得成功: {len(prices)}銘柄")
    return prices


# ------------------------------------------------------------------ #
# RSR計算・比較
# ------------------------------------------------------------------ #
def compute_rsr_latest(prices: dict[str, pd.Series]) -> dict[str, float]:
    """直近のRSRスコアを返す。"""
    rsr_df = calc_universe_rsr(prices)
    return {sym: float(rsr_df[sym].dropna().iloc[-1])
            for sym in rsr_df.columns
            if not rsr_df[sym].dropna().empty}


def run_comparison(trade_symbols: list[str]) -> dict:
    """
    3種のユニバースでRSRを計算し、取引ユニバース銘柄のスコアを比較する。
    """
    print("\n=== RSR母集団テスト ===")
    print(f"  取引ユニバース: {len(trade_symbols)}銘柄")

    results = {}

    # --- ケース1: 現行42銘柄コンテキスト ---
    print("\n[Case A] 現行42銘柄コンテキスト")
    prices_42 = fetch_prices(RSR_42)
    rsr_42 = compute_rsr_latest(prices_42)
    trade_rsr_42 = {s: rsr_42.get(s, 0.0) for s in trade_symbols if s in rsr_42}
    results["case_A_rsr42"] = {
        "context_size": len(prices_42),
        "threshold_75_count": sum(1 for v in trade_rsr_42.values() if v >= 75),
        "threshold_70_count": sum(1 for v in trade_rsr_42.values() if v >= 70),
        "top10": sorted(
            [{"symbol": s, "rsr": round(v, 1)} for s, v in trade_rsr_42.items()],
            key=lambda x: x["rsr"], reverse=True
        )[:10],
        "max_rsr": round(max(trade_rsr_42.values(), default=0.0), 1),
    }

    # --- ケース2: TOPIX300プロキシコンテキスト ---
    print("\n[Case B] TOPIX300プロキシコンテキスト")
    prices_300 = fetch_prices(TOPIX300_PROXY)
    rsr_300 = compute_rsr_latest(prices_300)
    trade_rsr_300 = {s: rsr_300.get(s, 0.0) for s in trade_symbols if s in rsr_300}
    results["case_B_rsr300"] = {
        "context_size": len(prices_300),
        "threshold_75_count": sum(1 for v in trade_rsr_300.values() if v >= 75),
        "threshold_70_count": sum(1 for v in trade_rsr_300.values() if v >= 70),
        "top10": sorted(
            [{"symbol": s, "rsr": round(v, 1)} for s, v in trade_rsr_300.items()],
            key=lambda x: x["rsr"], reverse=True
        )[:10],
        "max_rsr": round(max(trade_rsr_300.values(), default=0.0), 1),
    }

    # --- ケース3: 42銘柄コンテキスト内での全銘柄RSR分布（現状把握） ---
    rsr_42_all_sorted = sorted(
        [{"symbol": s, "rsr": round(v, 1)} for s, v in rsr_42.items()],
        key=lambda x: x["rsr"], reverse=True
    )
    results["all_context_distribution_42"] = rsr_42_all_sorted[:15]

    return results, rsr_42, rsr_300


# ------------------------------------------------------------------ #
# サマリー表示
# ------------------------------------------------------------------ #
def print_summary(results: dict) -> None:
    a = results["case_A_rsr42"]
    b = results["case_B_rsr300"]

    print("\n" + "=" * 60)
    print("  RSR母集団テスト結果サマリー")
    print("=" * 60)
    print(f"  {'指標':<30} {'Case A: 42銘柄':>15} {'Case B: 300銘柄':>15}")
    print(f"  {'-'*58}")
    print(f"  {'RSRコンテキスト銘柄数':<30} {a['context_size']:>15} {b['context_size']:>15}")
    print(f"  {'RSR≥75 通過銘柄数':<30} {a['threshold_75_count']:>15} {b['threshold_75_count']:>15}")
    print(f"  {'RSR≥70 通過銘柄数':<30} {a['threshold_70_count']:>15} {b['threshold_70_count']:>15}")
    print(f"  {'最高RSR':<30} {a['max_rsr']:>15.1f} {b['max_rsr']:>15.1f}")
    print()

    print("  [Case A] 取引ユニバース RSR上位10（42銘柄コンテキスト）:")
    for e in a["top10"]:
        bar = "█" * int(e["rsr"] / 5)
        flag = " ← ★" if e["rsr"] >= 75 else (" ← △" if e["rsr"] >= 70 else "")
        print(f"    {e['symbol']:<8} {e['rsr']:>5.1f} {bar}{flag}")

    print()
    print("  [Case B] 取引ユニバース RSR上位10（300銘柄コンテキスト）:")
    for e in b["top10"]:
        bar = "█" * int(e["rsr"] / 5)
        flag = " ← ★" if e["rsr"] >= 75 else (" ← △" if e["rsr"] >= 70 else "")
        print(f"    {e['symbol']:<8} {e['rsr']:>5.1f} {bar}{flag}")

    print()
    # 診断
    diff_75 = b["threshold_75_count"] - a["threshold_75_count"]
    if diff_75 > 0:
        print(f"  ✅ 診断: RSR≥75 が Case A→B で +{diff_75}銘柄増加")
        print("     → ユニバースバイアスが主因。ranking universe 拡張が有効。")
    elif b["max_rsr"] < 50:
        print("  ⚠  診断: Case B でも最高RSRが50未満")
        print("     → 市場全体の弱気局面が主因。レジームフィルターを検討。")
    else:
        print(f"  ℹ  診断: Case B 最高RSR={b['max_rsr']:.1f}。閾値調整(75→70)を検討。")
    print("=" * 60)


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    # 取引ユニバースを temporal24.json から読み込む
    universe_file = UNIVERSE_DIR / "2026Q1_temporal24.json"
    data = json.loads(universe_file.read_text(encoding="utf-8"))
    trade_symbols = list(data["symbols"].keys())
    print(f"取引ユニバース: {len(trade_symbols)}銘柄")

    results, rsr_42, rsr_300 = run_comparison(trade_symbols)
    print_summary(results)

    # ログ保存
    log_dir = LOGS_DIR / "diagnostics"
    log_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "date": TODAY,
        "trade_universe_size": len(trade_symbols),
        "rsr42_context_size": len(rsr_42),
        "rsr300_context_size": len(rsr_300),
        **results,
    }
    out_path = log_dir / f"rsr_universe_test_{TODAY}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 結果保存: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
