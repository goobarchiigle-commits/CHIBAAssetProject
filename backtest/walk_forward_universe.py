"""
backtest/walk_forward_universe.py
Walk-Forward Universe Test — 銘柄スクリーニングの過学習検証

【検証設計】

  問い: 「現行G27のスクリーニングは過学習か、それとも真の銘柄選択力か？」

  Train期間: 2018-01-01〜2021-12-31（4年）
    → Ex-ante全銘柄でフジコ法個別バックテストを実行
    → パフォーマンス上位N銘柄を選定（Calmar基準）

  Test期間: 2022-01-01〜2024-12-31（3年）
    → Train期間で選んだ銘柄のみ（固定）でポートフォリオバックテスト
    → 銘柄の再選定は一切しない（真のOOS評価）

  比較軸:
    (1) 現行G27 — Testのみ (2022-2024)
        Train期間の結果で選ばれた27銘柄（in-sample selection）のTest成績
    (2) WF-Top27 — Train→Test
        Walk-Forward手順で選ばれた27銘柄のTest成績
    (3) 全銘柄ベンチマーク — Testのみ (2022-2024)
        スクリーニングなし全銘柄のTest成績

  判定基準:
    WF-Top27(Test) が G27(Test) より有意に優れる
      → Train期間の「銘柄選択力」がOOSで維持されている → 過学習でない
    WF-Top27(Test) が G27(Test) と同等か劣る
      → 銘柄選択が特定期間への過学習 → バイアスの可能性あり
"""

from __future__ import annotations
import os, sys, warnings
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder        import TOPIX100_TICKERS, download_universe
from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.portfolio_engine        import PortfolioEngine
from backtest.engine                  import BacktestEngine, TradeCost

# ── 期間設定 ──────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-12-31"
FULL_START  = "2018-01-01"
FULL_END    = "2024-12-31"

CAPITAL     = 2_000_000
TOP_N       = 27          # Trainで選ぶ銘柄数
OUTPUT      = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

# ── セクター→戦略 ──────────────────────────────────────────────────
SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "フジコ法", "機械":    "フジコ法", "電機精密": "フジコ法",
    "商社":    "フジコ法", "電機":    "フジコ法", "食品":    "フジコ法",
    "レジャー":"フジコ法", "ゲーム":  "フジコ法", "サービス": "フジコ法",
    "医薬品":  "フジコ法", "不動産":  "フジコ法", "情報通信": "フジコ法",
    "小売":    "フジコ法",
    "ガス":    "平均回帰", "鉄鋼":    "平均回帰", "銀行":    "平均回帰",
    "保険":    "平均回帰", "輸送機器": "平均回帰", "化学":    "平均回帰",
    "陸運":    "平均回帰", "石油":    "平均回帰",
    "空運":    "平均回帰", "建設":    "平均回帰", "証券":    "平均回帰",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

# ── 現行G27 ───────────────────────────────────────────────────────
G27_SCREENED: dict[str, str] = {
    "8035.T": "電機精密", "6645.T": "電機精密", "6702.T": "電機",
    "6501.T": "電機",     "6762.T": "電機精密", "6920.T": "電機精密",
    "7203.T": "輸送機器", "7201.T": "輸送機器", "9432.T": "情報通信",
    "8306.T": "銀行",     "8411.T": "銀行",     "8309.T": "銀行",
    "7182.T": "銀行",     "8725.T": "保険",     "8058.T": "商社",
    "8053.T": "商社",     "8002.T": "商社",     "8001.T": "商社",
    "4021.T": "化学",     "2914.T": "食品",     "7011.T": "機械",
    "3382.T": "小売",     "5401.T": "鉄鋼",     "5411.T": "鉄鋼",
    "9531.T": "ガス",     "9101.T": "海運",     "9104.T": "海運",
    # V2追加
    "6857.T": "電機精密", "6594.T": "電機精密",
}

# Ex-ante Universeも含む全候補
EX_ANTE_ALL: dict[str, str] = {**TOPIX100_TICKERS, **{
    "6857.T": "電機精密", "6594.T": "電機精密", "6146.T": "電機精密",
    "6981.T": "電機精密", "6479.T": "電機精密", "6506.T": "機械",
    "7012.T": "機械",     "7013.T": "機械",     "9107.T": "海運",
    "8015.T": "商社",     "6869.T": "電機精密", "8316.T": "銀行",
    "5706.T": "鉄鋼",     "5713.T": "鉄鋼",     "8766.T": "保険",
    "8750.T": "保険",     "4519.T": "医薬品",   "4568.T": "医薬品",
    "9532.T": "ガス",     "9202.T": "空運",     "9201.T": "空運",
    "1801.T": "建設",     "1802.T": "建設",     "1803.T": "建設",
    "1812.T": "建設",     "8604.T": "証券",     "8601.T": "証券",
    "5020.T": "石油",
}}


# ================================================================
# ヘルパー
# ================================================================
def _calmar(cagr: float, mdd: float) -> float:
    return cagr / abs(mdd) if mdd != 0 else 0.0


def _slice_universe(universe_full: dict, start: str, end: str, min_days: int = 60) -> dict:
    """期間スライス済みユニバースを返す（データ不足銘柄を除外）"""
    result = {}
    for sym, info in universe_full.items():
        df_slice = info["df"].loc[start:end]
        if len(df_slice) >= min_days:
            result[sym] = {"df": df_slice, "sector": info["sector"]}
    return result


def _run_single_backtest(sym: str, info: dict, rsr_s: pd.Series) -> dict | None:
    """単一銘柄バックテストを実行してメトリクスを返す"""
    sector = info["sector"]
    rule   = SECTOR_STRATEGY.get(sector, "フジコ法")

    if rule == "平均回帰":
        strat = MeanReversionStrategy(**MR_PARAMS)
    else:
        strat = FujikoStrategy(
            rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
            mom_period=21, turtle_entry=20, turtle_exit=10,
            use_turtle_entry=True,
        )

    try:
        eng = BacktestEngine(
            prices=info["df"], strategy=strat,
            capital=1_000_000, cost=TradeCost(), symbol=sym,
        )
        r = eng.run()
        if r.num_trades < 2:   # 取引少なすぎ → 統計的に無意味
            return None
        # CAGR計算（BacktestResultにはtotal_returnのみ、cagrは自前で計算）
        eq = r.equity_curve
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr  = (1 + r.total_return) ** (1.0 / max(years, 0.1)) - 1 if years > 0 else r.total_return
        mdd   = r.max_drawdown
        cal   = _calmar(cagr, mdd)
        return {
            "symbol":  sym,
            "sector":  sector,
            "cagr":    cagr,
            "sharpe":  r.sharpe_ratio,
            "mdd":     mdd,
            "calmar":  cal,
            "trades":  r.num_trades,
            "win_rate": r.win_rate,
        }
    except Exception:
        return None


def _run_portfolio(data_dict: dict, rsr_uni: pd.DataFrame, max_positions: int = 3) -> object:
    """PortfolioEngineを実行"""
    strat_dict = {}
    for sym, info in data_dict.items():
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "フジコ法")
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        if rule == "平均回帰":
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )
    return PortfolioEngine(
        universe=data_dict, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1, max_positions=max_positions,
        cost=TradeCost(), vol_target=0.0,
        max_single_weight=min(0.25, 1.0 / max(1, max_positions)),
        use_idm=False,
    ).run()


def _port_metrics(r, label: str, period: str) -> dict:
    avg_pos = float(r.n_positions.mean())
    return dict(
        label=label, period=period,
        cagr=r.cagr, sharpe=r.sharpe_ratio,
        mdd=r.max_drawdown, calmar=_calmar(r.cagr, r.max_drawdown),
        trades=r.n_trades, win_rate=r.win_rate,
        avg_pos=avg_pos,
    )


def _print_port(m: dict):
    print(
        f"  {m['label']:<38} {m['period']:<12} "
        f"CAGR={m['cagr']*100:>+7.1f}%  Sharpe={m['sharpe']:>6.3f}  "
        f"MaxDD={m['mdd']*100:>7.1f}%  Calmar={m['calmar']:>6.3f}  "
        f"取引={m['trades']:>4}  勝率={m['win_rate']*100:>5.1f}%  "
        f"平均保有={m['avg_pos']:>4.2f}"
    )


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  walk_forward_universe.py — 銘柄スクリーニング過学習検証")
    print(f"  Train: {TRAIN_START}〜{TRAIN_END} / Test: {TEST_START}〜{TEST_END}")
    print("=" * 80)

    # ── [0] データ取得（フル期間）──────────────────────────────────
    print(f"\n[0] データ取得中（{len(EX_ANTE_ALL)}銘柄候補）...")
    universe_full = download_universe(EX_ANTE_ALL, start=FULL_START, end=FULL_END, verbose=False)
    print(f"  取得成功: {len(universe_full)} 銘柄")

    # ── [1] STEP1: Train期間（2018-2021）で個別バックテスト ────────
    print(f"\n[1] STEP1: Train期間バックテスト（{TRAIN_START}〜{TRAIN_END}）")
    print("  全銘柄を個別にバックテストし、上位27銘柄を選定します...")

    univ_train = _slice_universe(universe_full, TRAIN_START, TRAIN_END, min_days=200)
    print(f"  対象銘柄: {len(univ_train)} 銘柄")

    # RSR計算（Train期間のみで計算 — 先読みリーク防止）
    prices_train = {s: i["df"]["Close"] for s, i in univ_train.items()}
    rsr_train    = calc_universe_rsr(prices_train)

    # 個別バックテスト
    train_results = []
    for sym, info in univ_train.items():
        rsr_s = rsr_train[sym] if sym in rsr_train.columns else None
        m = _run_single_backtest(sym, info, rsr_s)
        if m is not None:
            train_results.append(m)

    df_train = pd.DataFrame(train_results)
    print(f"  有効結果: {len(df_train)} 銘柄")

    # ── 選定基準: Calmar降順でTop-N ──
    # Calmar = CAGR / |MaxDD|: リターンとリスクのバランスが良い銘柄を優先
    # ただしCAGRがマイナスの銘柄は除外
    df_train_pos = df_train[df_train["cagr"] > 0].copy()
    df_train_sorted = df_train_pos.sort_values("calmar", ascending=False)

    wf_top_n = min(TOP_N, len(df_train_sorted))
    selected = df_train_sorted.head(wf_top_n)
    wf_symbols = set(selected["symbol"].tolist())

    print(f"\n  【Train期間 Calmar上位{wf_top_n}銘柄 (WF選定)】")
    print(f"  {'銘柄':<10} {'セクター':<10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'取引':>5}")
    print("  " + "-" * 62)
    for _, row in selected.iterrows():
        print(f"  {row['symbol']:<10} {row['sector']:<10} "
              f"{row['cagr']*100:>+7.1f}% {row['sharpe']:>7.3f} "
              f"{row['mdd']*100:>7.1f}% {row['calmar']:>7.3f} {int(row['trades']):>5}")

    # ── [2] STEP2: Test期間（2022-2024）でポートフォリオ比較 ───────
    print(f"\n[2] STEP2: Test期間バックテスト（{TEST_START}〜{TEST_END}）")
    print("  4種類のシナリオを比較します")

    # Test期間データ
    univ_test_full = _slice_universe(universe_full, TEST_START, TEST_END, min_days=60)

    # RSR計算（Test期間のみで計算 — Train期間のRSRはリーク防止のため使わない）
    prices_test = {s: i["df"]["Close"] for s, i in univ_test_full.items()}
    rsr_test    = calc_universe_rsr(prices_test)

    # ── シナリオA: WF選定27銘柄 (Test期間のみ) ──
    wf_univ_test = {s: univ_test_full[s] for s in wf_symbols if s in univ_test_full}
    print(f"\n  A: WF選定27銘柄 / Test期間のみ（{len(wf_univ_test)}銘柄利用可能）")
    r_wf = _run_portfolio(wf_univ_test, rsr_test, max_positions=3)
    m_wf = _port_metrics(r_wf, "WF-Top27 [OOS Test]", f"{TEST_START}〜{TEST_END}")
    _print_port(m_wf)

    # ── シナリオB: 現行G27 (Test期間のみ) ──
    g27_univ_test = {s: univ_test_full[s] for s in G27_SCREENED if s in univ_test_full}
    print(f"\n  B: 現行G27 / Test期間のみ（{len(g27_univ_test)}銘柄利用可能）")
    r_g27_test = _run_portfolio(g27_univ_test, rsr_test, max_positions=3)
    m_g27_test = _port_metrics(r_g27_test, "現行G27 [Test期間のみ]", f"{TEST_START}〜{TEST_END}")
    _print_port(m_g27_test)

    # ── シナリオC: 現行G27 (フル期間: 2018-2024) ──
    univ_full_g27 = _slice_universe(universe_full, FULL_START, FULL_END)
    g27_univ_full = {s: univ_full_g27[s] for s in G27_SCREENED if s in univ_full_g27}
    prices_full = {s: i["df"]["Close"] for s, i in g27_univ_full.items()}
    rsr_full    = calc_universe_rsr(prices_full)
    print(f"\n  C: 現行G27 / フル期間（{len(g27_univ_full)}銘柄）")
    r_g27_full = _run_portfolio(g27_univ_full, rsr_full, max_positions=3)
    m_g27_full = _port_metrics(r_g27_full, "現行G27 [フル期間]", f"{FULL_START}〜{FULL_END}")
    _print_port(m_g27_full)

    # ── シナリオD: 全銘柄ベンチマーク (Test期間のみ) ──
    print(f"\n  D: 全銘柄ノースクリーニング / Test期間（{len(univ_test_full)}銘柄）")
    r_all_test = _run_portfolio(univ_test_full, rsr_test, max_positions=3)
    m_all_test = _port_metrics(r_all_test, "全銘柄ノースクリーニング [Test]", f"{TEST_START}〜{TEST_END}")
    _print_port(m_all_test)

    # ── [3] 追加: Train期間とTest期間のG27パフォーマンス比較 ──────
    print(f"\n[3] G27 Train/Test パフォーマンス比較（時期分割）")

    univ_train_g27 = {s: univ_train[s] for s in G27_SCREENED if s in univ_train}
    prices_g27_train = {s: i["df"]["Close"] for s, i in univ_train_g27.items()}
    rsr_g27_train    = calc_universe_rsr(prices_g27_train)
    r_g27_train = _run_portfolio(univ_train_g27, rsr_g27_train, max_positions=3)
    m_g27_train = _port_metrics(r_g27_train, "現行G27 [Train期間のみ]", f"{TRAIN_START}〜{TRAIN_END}")
    _print_port(m_g27_train)
    _print_port(m_g27_test)

    # ── [4] 結果サマリー ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【最終比較サマリー】")
    print("=" * 80)

    hdr = (f"  {'シナリオ':<38} {'期間':<12} "
           f"{'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'平均保有':>7}")
    print(hdr)
    print("  " + "-" * 90)

    all_m = [m_g27_full, m_g27_train, m_g27_test, m_wf, m_all_test]
    for m in all_m:
        print(f"  {m['label']:<38} {m['period']:<12} "
              f"{m['cagr']*100:>+8.1f}% {m['sharpe']:>7.3f} "
              f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} {m['avg_pos']:>7.2f}")

    # ── [5] 過学習判定 ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【過学習判定】")
    print("=" * 80)

    print(f"""
  【比較軸①】G27 Train vs Test（同じ銘柄の時期別差）
    Train期間 CAGR: {m_g27_train['cagr']*100:+.1f}%  Sharpe: {m_g27_train['sharpe']:.3f}
    Test期間  CAGR: {m_g27_test['cagr']*100:+.1f}%  Sharpe: {m_g27_test['sharpe']:.3f}
    →差分: ΔCAGR={m_g27_test['cagr']*100 - m_g27_train['cagr']*100:+.1f}%  ΔSharpe={m_g27_test['sharpe'] - m_g27_train['sharpe']:+.3f}""")

    print(f"""
  【比較軸②】WF-Top27 vs 現行G27（同期間・別銘柄選定）
    WF-Top27  (Test) CAGR: {m_wf['cagr']*100:+.1f}%  Sharpe: {m_wf['sharpe']:.3f}  Calmar: {m_wf['calmar']:.3f}
    現行G27   (Test) CAGR: {m_g27_test['cagr']*100:+.1f}%  Sharpe: {m_g27_test['sharpe']:.3f}  Calmar: {m_g27_test['calmar']:.3f}
    →差分: ΔCAGR={m_wf['cagr']*100 - m_g27_test['cagr']*100:+.1f}%  ΔSharpe={m_wf['sharpe'] - m_g27_test['sharpe']:+.3f}""")

    print(f"""
  【比較軸③】WF-Top27 vs 全銘柄ベンチマーク（スクリーニングの有無）
    WF-Top27  (Test) CAGR: {m_wf['cagr']*100:+.1f}%  Sharpe: {m_wf['sharpe']:.3f}
    全銘柄    (Test) CAGR: {m_all_test['cagr']*100:+.1f}%  Sharpe: {m_all_test['sharpe']:.3f}
    →差分: ΔCAGR={m_wf['cagr']*100 - m_all_test['cagr']*100:+.1f}%  ΔSharpe={m_wf['sharpe'] - m_all_test['sharpe']:+.3f}""")

    # 判定ロジック
    wf_vs_all_sharpe  = m_wf['sharpe']  - m_all_test['sharpe']
    wf_vs_all_cagr    = m_wf['cagr']    - m_all_test['cagr']
    train_test_decay  = m_g27_test['sharpe'] - m_g27_train['sharpe']
    wf_vs_g27_sharpe  = m_wf['sharpe']  - m_g27_test['sharpe']

    print("\n  【判定】")

    # 判定1: WF選定がベンチマーク（全銘柄）に勝てるか → 銘柄選択にAlphaがあるか
    if wf_vs_all_sharpe > 0.1:
        j1 = "✅ WFスクリーニングに有効なAlphaがある（選択力はOOSで機能する）"
    elif wf_vs_all_sharpe > -0.1:
        j1 = "△ WFスクリーニングの効果は中立的（誤差範囲内）"
    else:
        j1 = "⚠️  WFスクリーニングがベンチマーク以下（選択力はOOSで消失）"
    print(f"  銘柄選択のOOS有効性: {j1}")

    # 判定2: G27のTrain→Test劣化率 → 過学習の指標
    if train_test_decay > -0.3:
        j2 = "✅ 軽微（過学習は小さい）"
    elif train_test_decay > -0.7:
        j2 = "△ 中程度の劣化（部分的な過学習の可能性）"
    else:
        j2 = "⚠️  大きな劣化（過学習の可能性が高い）"
    print(f"  G27 Train→Test劣化: {j2}（ΔSharpe={train_test_decay:+.3f}）")

    # 判定3: WF vs 現行G27 → より公正な方法で選んだ銘柄の比較
    if wf_vs_g27_sharpe > 0.1:
        j3 = "✅ WF選定 > 現行G27（OOS設計の方が優れた銘柄を発見）"
    elif wf_vs_g27_sharpe > -0.1:
        j3 = "△ 同等（どちらの選定方法も大差なし）"
    else:
        j3 = "⚠️  WF選定 < 現行G27（in-sampleの全期間スクリーニングが偶然に強い）"
    print(f"  WF選定 vs 現行G27: {j3}（ΔSharpe={wf_vs_g27_sharpe:+.3f}）")

    # 総合判定
    print("\n  【総合判定】")
    overfitting_score = 0
    if wf_vs_all_sharpe < 0:   overfitting_score += 1
    if train_test_decay < -0.3: overfitting_score += 1
    if wf_vs_g27_sharpe < -0.1: overfitting_score += 1

    if overfitting_score == 0:
        verdict = "過学習の証拠なし ✅ — 銘柄スクリーニングはOOSで機能する"
        detail  = "Train期間の選定力がTest期間でも維持されており、現行設計は妥当。"
    elif overfitting_score == 1:
        verdict = "軽度の過学習の可能性 △ — バイアスは限定的"
        detail  = "一部指標で劣化が見られるが、戦略の有効性は概ね維持されている。"
    elif overfitting_score == 2:
        verdict = "中程度の過学習 ⚠️  — 銘柄選定方法の見直しを推奨"
        detail  = "銘柄選定がTrain期間に最適化されており、OOSでの安定性に懸念がある。"
    else:
        verdict = "強い過学習の証拠 ❌ — スクリーニングの設計を根本的に見直すべき"
        detail  = "Train期間の選定銘柄がTest期間でも全銘柄ベンチマークに勝てない。"

    print(f"\n  スコア: {overfitting_score}/3")
    print(f"  判定: {verdict}")
    print(f"  詳細: {detail}")

    # ── [6] WF選定銘柄と現行G27の重複分析 ────────────────────────
    print("\n" + "=" * 80)
    print("【銘柄重複分析】")
    print("=" * 80)

    g27_set = set(G27_SCREENED.keys())
    overlap = wf_symbols & g27_set
    only_wf = wf_symbols - g27_set
    only_g27 = g27_set - wf_symbols

    print(f"\n  現行G27銘柄数: {len(g27_set)}")
    print(f"  WF選定銘柄数: {len(wf_symbols)}")
    print(f"  重複銘柄: {len(overlap)}銘柄 ({len(overlap)/len(g27_set)*100:.0f}%)")
    print(f"\n  重複銘柄（両方に含まれる）: {sorted(overlap)}")
    print(f"\n  WF選定のみ（G27に含まれない）: {sorted(only_wf)}")
    print(f"\n  G27のみ（WFで選ばれなかった）: {sorted(only_g27)}")

    # ── [7] CSV保存 ────────────────────────────────────────────────
    rows = []
    for m in all_m:
        rows.append({
            "シナリオ": m["label"], "期間": m["period"],
            "CAGR%": round(m["cagr"]*100, 2), "Sharpe": round(m["sharpe"], 3),
            "MaxDD%": round(m["mdd"]*100, 2), "Calmar": round(m["calmar"], 3),
            "取引数": m["trades"], "勝率%": round(m["win_rate"]*100, 1),
            "平均保有銘柄": round(m["avg_pos"], 2),
        })
    df_result = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT, "walk_forward_universe.csv")
    df_result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # WF選定銘柄リスト
    df_wf_selected = selected.copy()
    df_wf_selected["in_G27"] = df_wf_selected["symbol"].isin(g27_set)
    df_wf_selected.to_csv(os.path.join(OUTPUT, "wf_selected_stocks.csv"),
                          index=False, encoding="utf-8-sig")

    print(f"\n  結果CSV: {out_csv}")
    print(f"  WF選定銘柄CSV: {os.path.join(OUTPUT, 'wf_selected_stocks.csv')}")
    print("\n完了。")

    return {
        "m_g27_full": m_g27_full, "m_g27_train": m_g27_train,
        "m_g27_test": m_g27_test, "m_wf": m_wf, "m_all_test": m_all_test,
        "wf_symbols": wf_symbols, "df_train": df_train,
        "overfitting_score": overfitting_score, "verdict": verdict,
    }


if __name__ == "__main__":
    main()
