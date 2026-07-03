"""
backtest/debug_alpha_ranking.py

STEP1 = BASELINE 問題の原因特定

チェック項目:
  A. alpha_df の値分布（NaN率・ゼロ率・有効値範囲）
  B. 複数候補日数（re-rankingが意味を持つ日数）
  C. ランキング変化日数（BASELINEとSTEP1で順位が変わる日）
  D. RSR×alpha vs RSR: 順位逆転が起きる候補ペア数

実行:
  python -m backtest.debug_alpha_ranking
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import CONFIGS_DIR, UNIVERSE_DIR

import json
import numpy as np
import pandas as pd
from pathlib import Path

from backtest.rsr                     import calc_universe_rsr, calc_composite_alpha_matrix
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.universe_builder        import download_universe

RSR_UNIVERSE_CSV      = CONFIGS_DIR / "rsr_universe_42.csv"
TRADING_UNIVERSE_JSON = UNIVERSE_DIR / "2026Q1_temporal24.json"
START = "2018-01-01"
END   = "2024-12-31"
COMP_ALPHA_WINDOW = 90
MIN_RSR  = 75.0
MIN_SEPA = 6

SECTOR_STRATEGY: dict[str, str] = {
    "海運": "fujiko", "機械": "fujiko", "電機精密": "fujiko",
    "商社": "fujiko", "電機": "fujiko",  "ゲーム":   "fujiko",
    "レジャー": "fujiko", "食品": "fujiko",
    "ガス": "mean_rev", "鉄鋼": "mean_rev", "銀行": "mean_rev",
    "保険": "mean_rev", "輸送機器": "mean_rev", "化学": "mean_rev",
    "小売": "mean_rev",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


def main() -> int:
    rsr_syms   = {row["symbol"]: row.get("sector","不明")
                  for _, row in pd.read_csv(RSR_UNIVERSE_CSV).iterrows()}
    trade_syms = json.loads(TRADING_UNIVERSE_JSON.read_text(encoding="utf-8"))["symbols"]

    print(f"RSU={len(rsr_syms)} TRADE={len(trade_syms)}")

    # ---- データ取得 ----
    all_syms = {**rsr_syms, **trade_syms}
    print("価格データ取得中...")
    universe_raw = download_universe(all_syms, start=START, end=END, verbose=False)

    # ---- RSR計算 ----
    rsr42_prices = {s: universe_raw[s]["df"]["Close"]
                    for s in rsr_syms if s in universe_raw}
    rsr_df = calc_universe_rsr(rsr42_prices)

    # ---- Alpha計算 ----
    trade_prices = {s: universe_raw[s]["df"]["Close"]
                    for s in trade_syms if s in universe_raw}
    alpha_df = calc_composite_alpha_matrix(trade_prices, window=COMP_ALPHA_WINDOW)
    alpha_df_shifted = alpha_df.shift(1)  # 先読み防止

    print("\n" + "="*60)
    print("  A. alpha_df 値分布チェック")
    print("="*60)
    flat = alpha_df_shifted.values.flatten()
    nan_count  = np.sum(np.isnan(flat))
    zero_count = np.sum(flat == 0.0)
    neg_count  = np.sum(flat < 0)
    pos_count  = np.sum(flat > 0)
    total      = len(flat)

    print(f"  総セル数 : {total:,}")
    print(f"  NaN      : {nan_count:,} ({nan_count/total:.1%})")
    print(f"  ゼロ     : {zero_count:,} ({zero_count/total:.1%})")
    print(f"  負値     : {neg_count:,} ({neg_count/total:.1%})")
    print(f"  正値     : {pos_count:,} ({pos_count/total:.1%})")
    valid = flat[~np.isnan(flat) & (flat != 0)]
    if len(valid):
        print(f"  有効値範囲: [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"  有効値中央: {np.median(valid):.6f}")

    # ---- 戦略初期化 ----
    strats = {}
    for sym, sector in trade_syms.items():
        if sym not in universe_raw:
            continue
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        if SECTOR_STRATEGY.get(sector, "fujiko") == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=MIN_SEPA, min_rsr=MIN_RSR,
                mom_period=21, turtle_entry=20, turtle_exit=20, use_turtle_entry=True,
            )

    # ---- 共通取引日 ----
    common_dates = None
    for sym in trade_syms:
        if sym not in universe_raw:
            continue
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    common_dates = common_dates[
        (common_dates >= pd.Timestamp(START)) & (common_dates <= pd.Timestamp(END))
    ]

    # ---- B/C/D: 候補日・ランキング変化日・逆転ペア ----
    print("\n" + "="*60)
    print("  B. BUY候補数の分布 & C. ランキング変化 & D. NaN状況")
    print("="*60)

    days_0cand      = 0
    days_1cand      = 0
    days_2pcand     = 0
    days_rerank     = 0     # BASELINE vs STEP1 で1位が変わった日
    days_nan_alpha  = 0     # alphaがNaNの候補があった日
    days_slot_limit = 0     # 複数候補 & スロット不足（rankinが実際に結果に影響する日）
    n_positions_on_multi = []  # 複数候補日の保有数
    sample_multi    = []    # 複数候補の日のサンプル
    # スロット競合シミュレーション用（簡易）
    held_positions = 0  # 単純カウンタ（保有銘柄数の近似）

    for i, date in enumerate(common_dates):
        rsr_row   = rsr_df.loc[date]       if date in rsr_df.index   else pd.Series(dtype=float)
        alpha_row = alpha_df_shifted.loc[date] if date in alpha_df_shifted.index else pd.Series(dtype=float)

        base_cands  = []   # (rsr, sym)
        step1_cands = []   # (rsr*alpha, sym)
        has_nan_alpha = False

        for sym in trade_syms:
            if sym not in universe_raw or sym not in strats:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue
            past = df_sym.loc[:date]
            if len(past) < 30:
                continue
            sig = strats[sym].generate_signal(past)
            if sig != 1:
                continue

            rsr_val = float(rsr_row.get(sym, 0.0)) if sym in rsr_row.index else 0.0

            # alpha値の取得
            a_raw = alpha_row.get(sym, np.nan) if sym in alpha_row.index else np.nan
            if np.isnan(a_raw):
                has_nan_alpha = True
                a_val = 0.0
            else:
                a_val = float(a_raw)

            base_cands.append((rsr_val, sym))
            rank_score = rsr_val * max(0.0, a_val)
            step1_cands.append((rank_score, a_val, sym))

        n = len(base_cands)
        if has_nan_alpha:
            days_nan_alpha += 1

        if n == 0:
            days_0cand += 1
        elif n == 1:
            days_1cand += 1
        else:
            days_2pcand += 1
            n_positions_on_multi.append(held_positions)
            # スロット競合チェック: 候補数 > 空きスロット → rankingが実際に影響する
            free_slots = max(0, 3 - held_positions)
            if n > free_slots:
                days_slot_limit += 1
            # ランキング比較
            base_top  = sorted(base_cands,  key=lambda x: -x[0])[0][1]
            step1_top = sorted(step1_cands, key=lambda x: -x[0])[0][1]
            if base_top != step1_top:
                days_rerank += 1
        # 簡易ポジション更新（BUY候補があれば保有増、売りは無視）
        if n > 0:
            buy_n  = min(n, max(0, 3 - held_positions))
            held_positions = min(3, held_positions + buy_n)
        # 簡易エグジット（17日平均保有 → 毎日1/17 の確率で1銘柄終了）
        import random
        if held_positions > 0 and random.random() < held_positions / 17:
            held_positions -= 1
            if len(sample_multi) < 5:
                cands_info = []
                for r, s in base_cands:
                    a = next((a for rs, a, ss in step1_cands if ss == s), None)
                    cands_info.append({"sym": s, "rsr": r, "alpha": a})
                sample_multi.append({
                    "date": str(date.date()),
                    "n": n,
                    "base_top": base_top,
                    "step1_top": step1_top,
                    "cands": cands_info,
                })

    total_days = len(common_dates)
    print(f"  総取引日数 : {total_days}")
    print(f"  BUY候補=0  : {days_0cand} ({days_0cand/total_days:.1%})  ← rankingが不要")
    print(f"  BUY候補=1  : {days_1cand} ({days_1cand/total_days:.1%})  ← ranking効果なし")
    print(f"  BUY候補≥2  : {days_2pcand} ({days_2pcand/total_days:.1%})  ← rankingが意味を持つ")
    print(f"  alpha=NaN日 : {days_nan_alpha} ({days_nan_alpha/total_days:.1%})")
    if days_2pcand > 0:
        print(f"  1位逆転日   : {days_rerank} / {days_2pcand} ({days_rerank/max(days_2pcand,1):.1%})  ← ランキングで1位が変わる")
    print(f"  スロット競合日: {days_slot_limit} / {days_2pcand} ({days_slot_limit/max(days_2pcand,1):.1%})  ← 競合がないとrankingは無意味")
    if n_positions_on_multi:
        print(f"  複数候補日の平均保有数: {np.mean(n_positions_on_multi):.2f} / 最大3")

    if sample_multi:
        print("\n  複数候補日サンプル:")
        for s in sample_multi:
            print(f"    {s['date']}  n={s['n']}  BASE_top={s['base_top']}  STEP1_top={s['step1_top']}")
            for c in s["cands"]:
                a = c["alpha"] if c["alpha"] is not None else float("nan")
                rank_score = c["rsr"] * max(0.0, a) if not np.isnan(a) else 0.0
                a_str = f"{a:.6f}" if not np.isnan(a) else "NaN"
                print(f"      {c['sym']}  RSR={c['rsr']:.1f}  alpha={a_str}  "
                      f"combined={rank_score:.4f}")

    print("\n" + "="*60)
    print("  診断結論")
    print("="*60)
    if days_2pcand == 0:
        print("  ❌ 複数候補日がゼロ → ranking層は動作するが機能しない")
        print("     根本原因: TEMPORAL24ユニバースの供給不足")
        print("     対策: RSR42を取引ユニバースとしても使用する")
    elif days_rerank == 0:
        print("  ❌ 複数候補日あり・但し1位が一度も変わらない")
        print("     alphaとRSRが相関しすぎて再ランク順変化なし")
    elif days_rerank > 0:
        pct = days_rerank / max(days_2pcand, 1)
        print(f"  ✅ 複数候補日{days_2pcand}日中{days_rerank}日({pct:.0%})で1位が変わる")
        print(f"     → ranking層は正しく動いている")
        print(f"     → STEP1=BASELINE は別原因（サンプル不足 or 最終選択が同じ）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
