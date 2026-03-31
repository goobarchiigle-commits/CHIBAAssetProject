"""
backtest/rolling_universe.py

ローリング選定ユニバース生成（look-ahead bias 除去版）

設計:
  Train window: 3年
  Test window : 1年（OOS）

  スクリーニング基準（train 期間のみで評価）:
    Sharpe > 0.3  AND  MaxDD < 30%

フォールド:
  2015-2017 → 2018 OOS
  2016-2018 → 2019 OOS
  2017-2019 → 2020 OOS
  2018-2020 → 2021 OOS
  2019-2021 → 2022 OOS
  2020-2022 → 2023 OOS
  2021-2023 → 2024 OOS

出力:
  configs/rolling_universe.json
    {"2018": {"7203.T": "輸送機器", ...}, "2019": {...}, ...}

実行:
  python -m backtest.rolling_universe
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from paths import CONFIGS_DIR

import numpy as np
import pandas as pd

from backtest.universe_builder  import TOPIX100_TICKERS, download_universe
from backtest.rsr               import calc_universe_rsr
from backtest.fujiko_strategy   import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine            import BacktestEngine, TradeCost

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
OUTPUT_JSON = CONFIGS_DIR / "rolling_universe.json"

FOLDS = [
    {"train_start": "2015-01-01", "train_end": "2017-12-31", "test_year": 2018},
    {"train_start": "2016-01-01", "train_end": "2018-12-31", "test_year": 2019},
    {"train_start": "2017-01-01", "train_end": "2019-12-31", "test_year": 2020},
    {"train_start": "2018-01-01", "train_end": "2020-12-31", "test_year": 2021},
    {"train_start": "2019-01-01", "train_end": "2021-12-31", "test_year": 2022},
    {"train_start": "2020-01-01", "train_end": "2022-12-31", "test_year": 2023},
    {"train_start": "2021-01-01", "train_end": "2023-12-31", "test_year": 2024},
]

MIN_SHARPE   = 0.30
MAX_DD       = -0.30
MIN_BARS     = 400   # train 期間の最低取引日数

# セクター → 戦略マッピング（signal_bridge / portfolio_v2 と共通）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":     "fujiko", "機械":     "fujiko", "電機精密": "fujiko",
    "商社":     "fujiko", "電機":     "fujiko", "ゲーム":   "fujiko",
    "レジャー": "fujiko", "食品":     "fujiko",
    "ガス":     "mean_rev","鉄鋼":    "mean_rev","銀行":     "mean_rev",
    "保険":     "mean_rev","輸送機器":"mean_rev","化学":     "mean_rev",
    "小売":     "mean_rev",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


# ------------------------------------------------------------------ #
# スクリーニング関数
# ------------------------------------------------------------------ #
def screen_on_train(
    universe_raw:  dict,
    rsr_train:     pd.DataFrame,   # train 期間の RSR DataFrame
    train_start:   str,
    train_end:     str,
    min_sharpe:    float = MIN_SHARPE,
    max_dd:        float = MAX_DD,
) -> dict[str, str]:
    """
    train 期間のバックテスト結果で銘柄をスクリーニングし、
    {symbol: sector} を返す。

    先読みリーク防止:
      - RSR は train 期間内のみで計算（rsr_train を使用）
      - test 期間のデータは一切参照しない
    """
    cost   = TradeCost()
    passed: dict[str, str] = {}

    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)

    for sym, info in universe_raw.items():
        df     = info["df"]
        sector = info["sector"]

        # train 期間に切り出し
        train_df = df.loc[ts:te]
        if len(train_df) < MIN_BARS:
            continue

        # RSR series（train 期間内のみ）
        rsr_s = rsr_train[sym] if sym in rsr_train.columns else None

        # 戦略選択
        rule = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strat = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat = FujikoStrategy(
                rsr_series       = rsr_s,
                min_sepa         = 6,
                min_rsr          = 70.0,   # スクリーニング時は 70 で緩め（選定幅を確保）
                mom_period       = 21,
                turtle_entry     = 20,
                turtle_exit      = 20,
                use_turtle_entry = True,
            )

        eng = BacktestEngine(
            prices  = train_df,
            strategy = strat,
            capital  = 1_000_000,
            cost     = cost,
            symbol   = sym,
        )
        r = eng.run()

        if r.sharpe_ratio > min_sharpe and r.max_drawdown > max_dd:
            passed[sym] = sector

    return passed


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> int:
    print("=" * 68)
    print("  ローリング選定ユニバース生成（look-ahead bias 除去）")
    print(f"  フォールド数: {len(FOLDS)}  /  スクリーニング: Sharpe>{MIN_SHARPE} MaxDD>{MAX_DD}")
    print("=" * 68)

    # ---- 1. 全期間データを一括取得（2015〜2024） ----
    print("\n[1/2] TOPIX100 データ取得中（2015-2024）...")
    universe_raw = download_universe(
        TOPIX100_TICKERS,
        start="2015-01-01",
        end="2024-12-31",
        verbose=False,
    )
    print(f"  取得完了: {len(universe_raw)} 銘柄")

    # ---- 2. フォールドごとにスクリーニング ----
    print("\n[2/2] フォールド別スクリーニング...")
    rolling: dict[str, dict[str, str]] = {}   # {year_str: {sym: sector}}

    for fold in FOLDS:
        ts  = fold["train_start"]
        te  = fold["train_end"]
        yr  = fold["test_year"]

        # train 期間のみで RSR を計算（テストデータ混入なし）
        prices_train = {}
        for sym, info in universe_raw.items():
            s = info["df"].loc[pd.Timestamp(ts):pd.Timestamp(te)]["Close"]
            if len(s) >= MIN_BARS:
                prices_train[sym] = s
        rsr_train = calc_universe_rsr(prices_train)

        passed = screen_on_train(universe_raw, rsr_train, ts, te)

        rolling[str(yr)] = passed

        print(
            f"  {ts[:4]}-{te[:4]} (train) → {yr} OOS: "
            f"{len(passed)}銘柄  [{', '.join(sorted(passed.keys())[:5])}{'...' if len(passed)>5 else ''}]"
        )

    # ---- 3. 統計 ----
    sizes    = [len(v) for v in rolling.values()]
    all_syms = set(sym for d in rolling.values() for sym in d)

    print(f"\n  選定銘柄数: 平均={np.mean(sizes):.1f}  min={min(sizes)}  max={max(sizes)}")
    print(f"  延べユニーク銘柄: {len(all_syms)}")
    print(f"\n  年別ユニバース:")
    for yr, syms in rolling.items():
        print(f"    {yr}: {len(syms)}銘柄")

    # ---- 4. 保存 ----
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(rolling, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  保存完了: {OUTPUT_JSON}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
