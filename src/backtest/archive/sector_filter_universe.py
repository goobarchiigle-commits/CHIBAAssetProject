"""
backtest/sector_filter_universe.py
Sector Filter Universe Test — 構造フィルターベースのUniverse設計検証

【目的】
  銘柄固定バイアスを除去し、「セクター構造フィルター」のみでUniverseを設計する。
  個別銘柄の選定結果（Sharpe等）は一切使わない。

【Universe設計ルール】
  1. 対象市場: TSEプライム
  2. 流動性フィルター: 2018年前半の平均売買代金 > 5億円/日
  3. セクターフィルター: 以下7セクターのみ許可
       電機精密 / 電機 / 海運 / 鉄鋼 / 機械 / 商社 / 輸送機器
  4. 除外セクター: 小売 / 食品
  ※ 銘柄は固定しない。条件を満たす銘柄を自動的にUniverseに含める。

【比較対象】
  1. 現行G27（個別銘柄スクリーニング済み）
  2. 全銘柄ノースクリーニング（前回検証の再現）
  3. Sector Filter Universe（本検証）

【戦略】
  既存のフジコ法ロジック（SECTOR_STRATEGYマッピング準拠）を使用。
  ※ 分析ポイントとして、7セクター全銘柄をフジコ法一本で回す
    「フジコ法オンリー版」も追加比較する。
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
from backtest.engine                  import TradeCost

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
OUTPUT  = "C:/Users/owner/.claude/レポート"
os.makedirs(OUTPUT, exist_ok=True)

DAILY_TURNOVER_THRESHOLD = 500_000_000  # 5億円/日

# ── 許可セクター（7種） ────────────────────────────────────────────
ALLOWED_SECTORS = {
    "電機精密", "電機", "海運", "鉄鋼", "機械", "商社", "輸送機器"
}
EXCLUDED_SECTORS = {"小売", "食品"}

# ── セクター→戦略マッピング（既存ロジック準拠） ────────────────────
SECTOR_STRATEGY_MIXED: dict[str, str] = {
    # フジコ法適合セクター
    "海運":    "フジコ法", "機械":    "フジコ法", "電機精密": "フジコ法",
    "商社":    "フジコ法", "電機":    "フジコ法",
    # 平均回帰適合セクター（既存ロジック準拠）
    "輸送機器": "平均回帰", "鉄鋼":    "平均回帰",
}
# フジコ法オンリー（全7セクターをフジコ法で処理）
SECTOR_STRATEGY_FUJIKO_ONLY: dict[str, str] = {
    sec: "フジコ法" for sec in ALLOWED_SECTORS
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

# ── 現行G27（比較用） ─────────────────────────────────────────────
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
    "6857.T": "電機精密", "6594.T": "電機精密",
}

# ── TSEプライム広範囲候補（7セクター + その他含む） ─────────────────
# 「構造フィルター」の候補プール。個別銘柄の事前評価なし。
TSE_PRIME_CANDIDATES: dict[str, str] = {
    # ── 電機精密 ──
    "6758.T": "電機",       # ソニーグループ
    "6861.T": "電機精密",   # キーエンス
    "8035.T": "電機精密",   # 東京エレクトロン
    "6954.T": "電機精密",   # ファナック
    "6645.T": "電機精密",   # オムロン
    "6702.T": "電機",       # 富士通
    "6501.T": "電機",       # 日立製作所
    "6503.T": "電機",       # 三菱電機
    "6752.T": "電機",       # パナソニックHD
    "6762.T": "電機精密",   # TDK
    "6971.T": "電機精密",   # 京セラ
    "6723.T": "電機精密",   # ルネサスエレクトロニクス
    "7741.T": "電機精密",   # HOYA
    "6920.T": "電機精密",   # レーザーテック
    "6902.T": "電機",       # デンソー
    "7751.T": "電機精密",   # キヤノン
    "6857.T": "電機精密",   # アドバンテスト
    "6594.T": "電機精密",   # ニデック
    "6506.T": "機械",       # 安川電機
    "6146.T": "電機精密",   # ディスコ
    "6981.T": "電機精密",   # 村田製作所
    "6479.T": "電機精密",   # ミネベアミツミ
    "6869.T": "電機精密",   # シスメックス
    # ── 輸送機器 ──
    "7203.T": "輸送機器",   # トヨタ自動車
    "7267.T": "輸送機器",   # 本田技研工業
    "7201.T": "輸送機器",   # 日産自動車
    "7270.T": "輸送機器",   # SUBARU
    "7261.T": "輸送機器",   # マツダ
    "6201.T": "機械",       # 豊田自動織機
    # ── 情報通信（非対象だが候補に含め、フィルターで除外） ──
    "9984.T": "情報通信",   # ソフトバンクグループ
    "9432.T": "情報通信",   # NTT
    "9433.T": "情報通信",   # KDDI
    "4307.T": "情報通信",   # 野村総合研究所
    # ── 銀行（非対象） ──
    "8306.T": "銀行",
    "8316.T": "銀行",
    "8411.T": "銀行",
    "8309.T": "銀行",
    "7182.T": "銀行",
    # ── 保険（非対象） ──
    "8766.T": "保険",
    "8725.T": "保険",
    "8750.T": "保険",
    # ── 商社 ──
    "8058.T": "商社",       # 三菱商事
    "8031.T": "商社",       # 三井物産
    "8053.T": "商社",       # 住友商事
    "8002.T": "商社",       # 丸紅
    "8001.T": "商社",       # 伊藤忠商事
    "8015.T": "商社",       # 豊田通商
    # ── 医薬品（非対象） ──
    "4502.T": "医薬品",
    "4519.T": "医薬品",
    "4568.T": "医薬品",
    "4543.T": "医薬品",
    # ── 化学（非対象） ──
    "4063.T": "化学",
    "4021.T": "化学",
    "4901.T": "化学",
    # ── 食品（除外セクター） ──
    "2802.T": "食品",
    "2914.T": "食品",
    "2503.T": "食品",
    "2502.T": "食品",
    # ── 機械 ──
    "6367.T": "機械",       # ダイキン工業
    "6301.T": "機械",       # 小松製作所
    "6326.T": "機械",       # クボタ
    "7011.T": "機械",       # 三菱重工業
    "7012.T": "機械",       # 川崎重工業
    "7013.T": "機械",       # IHI
    # ── 不動産（非対象） ──
    "8801.T": "不動産",
    "8802.T": "不動産",
    # ── 小売（除外セクター） ──
    "3382.T": "小売",
    "9983.T": "小売",
    # ── レジャー（非対象） ──
    "4661.T": "レジャー",
    "6098.T": "サービス",
    "7974.T": "ゲーム",
    # ── 鉄鋼 ──
    "5401.T": "鉄鋼",       # 日本製鉄
    "5411.T": "鉄鋼",       # JFEスチール
    "5713.T": "鉄鋼",       # 住友金属鉱山
    "5706.T": "鉄鋼",       # 三菱マテリアル
    # ── 石油・ガス（非対象） ──
    "5020.T": "石油",
    "9531.T": "ガス",
    "9532.T": "ガス",
    # ── 海運 ──
    "9101.T": "海運",       # 日本郵船
    "9104.T": "海運",       # 商船三井
    "9107.T": "海運",       # 川崎汽船
    # ── 陸運（非対象） ──
    "9020.T": "陸運",
    "9022.T": "陸運",
    # ── 建設（非対象） ──
    "1801.T": "建設",
    "1812.T": "建設",
    # ── 証券（非対象） ──
    "8604.T": "証券",
}


# ================================================================
# ヘルパー
# ================================================================
def _calmar(cagr: float, mdd: float) -> float:
    return cagr / abs(mdd) if mdd != 0 else 0.0


def _apply_sector_and_liquidity_filter(
    universe_all: dict,
    allowed_sectors: set,
    min_turnover: float = DAILY_TURNOVER_THRESHOLD,
) -> dict:
    """
    セクターフィルター + 流動性フィルターを適用する。
    バックテスト結果は一切使わない（純粋な構造フィルター）。
    """
    passed = {}
    rejected_sector = []
    rejected_liquidity = []

    for sym, info in universe_all.items():
        sector = info["sector"]

        # セクターフィルター
        if sector not in allowed_sectors:
            rejected_sector.append((sym, sector))
            continue

        # 流動性フィルター（2018年前半の売買代金）
        df = info["df"]
        df_2018 = df.loc["2018-01-01":"2018-06-30"]
        if len(df_2018) < 60:
            rejected_liquidity.append((sym, "データ不足"))
            continue
        turnover = (df_2018["Close"] * df_2018["Volume"]).mean()
        if turnover < min_turnover:
            rejected_liquidity.append((sym, f"{turnover/1e8:.1f}億"))
            continue

        passed[sym] = info

    return passed, rejected_sector, rejected_liquidity


def _build_strategies(
    data_dict: dict,
    rsr_uni: pd.DataFrame,
    strategy_map: dict,
    min_rsr: float = 70.0,
    min_sepa: int = 6,
) -> dict:
    strat_dict = {}
    for sym, info in data_dict.items():
        sector = info["sector"]
        rule   = strategy_map.get(sector, "フジコ法")
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        if rule == "平均回帰":
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=min_sepa, min_rsr=min_rsr,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )
    return strat_dict


def _run_portfolio(
    data_dict: dict,
    strat_dict: dict,
    max_positions: int = 3,
) -> object:
    return PortfolioEngine(
        universe=data_dict,
        strategy=strat_dict,
        capital=CAPITAL,
        max_dd_limit=0.15,
        min_sectors=1,
        max_positions=max_positions,
        cost=TradeCost(),
        vol_target=0.0,
        max_single_weight=min(0.25, 1.0 / max(1, max_positions)),
        use_idm=False,
    ).run()


def _metrics(r, label: str, n_stocks: int) -> dict:
    avg_pos = float(r.n_positions.mean())
    return dict(
        label=label, n_stocks=n_stocks,
        cagr=r.cagr, sharpe=r.sharpe_ratio,
        mdd=r.max_drawdown, calmar=_calmar(r.cagr, r.max_drawdown),
        trades=r.n_trades, win_rate=r.win_rate, avg_pos=avg_pos,
        _res=r,
    )


def _count_annual_signals(data_dict: dict, strat_dict: dict) -> float:
    """全銘柄の年間BUYシグナル合計数を概算する（サンプリング版）"""
    total_buy = 0
    for sym, info in data_dict.items():
        strat = strat_dict.get(sym)
        if strat is None:
            continue
        df = info["df"]
        # ウォームアップ後の全日付をスキャン
        warmup = 252 + 21 + 2
        for i in range(warmup, len(df)):
            past = df.iloc[:i + 1]
            try:
                sig = strat.generate_signal(past)
                if sig == 1:
                    total_buy += 1
            except Exception:
                pass
    years = 7.0  # 2018-2024
    return total_buy / years


def _sector_summary(r) -> pd.DataFrame:
    """trades からセクター別サマリーを生成"""
    if r.trades.empty:
        return pd.DataFrame()
    sells = r.trades[r.trades["side"].str.startswith("SELL")].copy()
    if sells.empty:
        return pd.DataFrame()
    rows = []
    for sec in sells["sector"].unique():
        pnl = sells[sells["sector"] == sec]["pnl"].values
        rows.append({
            "セクター": sec,
            "取引数": len(pnl),
            "勝率%": round(float((pnl > 0).mean()) * 100, 1),
            "平均PnL(万円)": round(float(pnl.mean()) / 10000, 2),
            "合計PnL(万円)": round(float(pnl.sum()) / 10000, 1),
        })
    return pd.DataFrame(rows).sort_values("合計PnL(万円)", ascending=False)


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  sector_filter_universe.py — セクターフィルターUniverse検証")
    print("  期間: 2018〜2024  /  初期資本: ¥2,000,000")
    print(f"  許可セクター: {', '.join(sorted(ALLOWED_SECTORS))}")
    print("=" * 80)

    # ── [0] データ取得（全候補） ─────────────────────────────────
    print(f"\n[0] データ取得中（{len(TSE_PRIME_CANDIDATES)}銘柄候補）...")
    universe_all = download_universe(
        TSE_PRIME_CANDIDATES, start=START, end=END, verbose=False
    )
    print(f"  取得成功: {len(universe_all)} 銘柄")

    # ── [1] Sector Filter Universe 構築 ─────────────────────────
    print("\n[1] Sector Filter Universe 構築")
    print(f"  フィルター: セクター={sorted(ALLOWED_SECTORS)}")
    print(f"  流動性: 2018年前半 平均売買代金 > {DAILY_TURNOVER_THRESHOLD/1e8:.0f}億円/日")

    univ_sector, rej_sector, rej_liq = _apply_sector_and_liquidity_filter(
        universe_all, ALLOWED_SECTORS
    )

    print(f"\n  【フィルター結果】")
    print(f"    全候補:       {len(universe_all)} 銘柄")
    print(f"    セクター除外: {len(rej_sector)} 銘柄")
    print(f"    流動性除外:   {len(rej_liq)} 銘柄")
    print(f"    ✅ 通過:       {len(univ_sector)} 銘柄")

    print(f"\n  【Sector Filter Universe — 銘柄リスト】")
    sec_count: dict[str, list] = {}
    for sym, info in univ_sector.items():
        sec = info["sector"]
        sec_count.setdefault(sec, []).append(sym)
    for sec in sorted(sec_count.keys()):
        syms = sec_count[sec]
        print(f"    {sec:<10} ({len(syms)}銘柄): {', '.join(sorted(syms))}")

    # ── [2] RSR計算 ───────────────────────────────────────────────
    print(f"\n[2] RSR計算中...")

    # 現行G27 RSR（G27銘柄のみで計算）
    g27_in_universe = {s: universe_all[s] for s in G27_SCREENED if s in universe_all}
    prices_g27   = {s: i["df"]["Close"] for s, i in g27_in_universe.items()}
    rsr_g27      = calc_universe_rsr(prices_g27)

    # 全銘柄 RSR
    prices_all   = {s: i["df"]["Close"] for s, i in universe_all.items()}
    rsr_all      = calc_universe_rsr(prices_all)

    # Sector Filter RSR（Sector Universe内でのみ計算）
    prices_sec   = {s: i["df"]["Close"] for s, i in univ_sector.items()}
    rsr_sec      = calc_universe_rsr(prices_sec)

    print(f"  G27: {rsr_g27.shape[1]}銘柄  全銘柄: {rsr_all.shape[1]}銘柄  "
          f"Sector: {rsr_sec.shape[1]}銘柄")

    # ── [3] バックテスト実行 ──────────────────────────────────────
    print("\n[3] バックテスト実行（4シナリオ）")

    # ─ A: 現行G27（既存SECTOR_STRATEGYマッピング）
    print("  [A] 現行G27（スクリーニング済み）...")
    strat_g27 = _build_strategies(g27_in_universe, rsr_g27, SECTOR_STRATEGY_MIXED)
    r_g27     = _run_portfolio(g27_in_universe, strat_g27)
    m_g27     = _metrics(r_g27, f"A: 現行G27({len(g27_in_universe)}銘柄)", len(g27_in_universe))

    # ─ B: 全銘柄ノースクリーニング
    print(f"  [B] 全銘柄ノースクリーニング（{len(universe_all)}銘柄）...")
    # 全セクターに対応する戦略マップ（許可外セクターも含む）
    ALL_SECTOR_MAP = {**SECTOR_STRATEGY_MIXED,
                      "銀行":"平均回帰","保険":"平均回帰","ガス":"平均回帰",
                      "化学":"平均回帰","陸運":"平均回帰","石油":"平均回帰",
                      "医薬品":"フジコ法","情報通信":"フジコ法","不動産":"フジコ法",
                      "レジャー":"フジコ法","ゲーム":"フジコ法","サービス":"フジコ法",
                      "小売":"フジコ法","食品":"フジコ法","建設":"平均回帰",
                      "証券":"平均回帰"}
    strat_all = _build_strategies(universe_all, rsr_all, ALL_SECTOR_MAP)
    r_all     = _run_portfolio(universe_all, strat_all)
    m_all     = _metrics(r_all, f"B: 全銘柄ノースクリーニング({len(universe_all)}銘柄)", len(universe_all))

    # ─ C: Sector Filter Universe（既存戦略マッピング）
    print(f"  [C] Sector Filter Universe / 既存戦略マッピング（{len(univ_sector)}銘柄）...")
    strat_sec_mixed = _build_strategies(univ_sector, rsr_sec, SECTOR_STRATEGY_MIXED)
    r_sec_mixed     = _run_portfolio(univ_sector, strat_sec_mixed)
    m_sec_mixed     = _metrics(r_sec_mixed, f"C: Sector Filter / 既存マッピング({len(univ_sector)}銘柄)", len(univ_sector))

    # ─ D: Sector Filter Universe（フジコ法オンリー）
    print(f"  [D] Sector Filter Universe / フジコ法オンリー（{len(univ_sector)}銘柄）...")
    strat_sec_fj = _build_strategies(univ_sector, rsr_sec, SECTOR_STRATEGY_FUJIKO_ONLY)
    r_sec_fj     = _run_portfolio(univ_sector, strat_sec_fj)
    m_sec_fj     = _metrics(r_sec_fj, f"D: Sector Filter / フジコ法オンリー({len(univ_sector)}銘柄)", len(univ_sector))

    all_m = [m_g27, m_all, m_sec_mixed, m_sec_fj]

    # ── [4] シグナル頻度分析（Sector Filter）───────────────────────
    print("\n[4] シグナル頻度分析（Sector Filter Universe / フジコ法オンリー）")
    print("  ※ 全日付スキャンのため時間がかかります...")

    total_buy_signals = 0
    for sym, info in univ_sector.items():
        strat = strat_sec_fj.get(sym)
        if strat is None:
            continue
        df = info["df"]
        warmup = 252 + 21 + 2
        for i in range(warmup, len(df)):
            past = df.iloc[:i + 1]
            try:
                if strat.generate_signal(past) == 1:
                    total_buy_signals += 1
            except Exception:
                pass
    annual_signals = total_buy_signals / 7.0

    print(f"  年間BUYシグナル数（Sector Filter / フジコ法）: {annual_signals:.1f} 件/年")
    print(f"  銘柄あたり年平均: {annual_signals/max(1, len(univ_sector)):.2f} 件/年")

    # ── [5] 結果サマリー ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【最終比較サマリー】")
    print("=" * 80)

    hdr = (f"\n  {'シナリオ':<45} {'銘柄':>5} "
           f"{'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
           f"{'取引':>5} {'勝率':>6} {'平均保有':>7}")
    print(hdr)
    print("  " + "-" * 100)
    for m in all_m:
        print(
            f"  {m['label']:<45} {m['n_stocks']:>5} "
            f"{m['cagr']*100:>+8.1f}% {m['sharpe']:>7.3f} "
            f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} "
            f"{m['trades']:>5} {m['win_rate']*100:>6.1f}% {m['avg_pos']:>7.2f}"
        )

    # ── [6] 分析ポイント解答 ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("【分析ポイント解答】")
    print("=" * 80)

    print(f"""
  ① Sector Universe の Sharpe が全銘柄Universe (Sharpe≈{m_all['sharpe']:.3f}) を上回るか？
     既存マッピング版: Sharpe={m_sec_mixed['sharpe']:.3f}  →  {'✅ 上回る' if m_sec_mixed['sharpe'] > m_all['sharpe'] else '❌ 下回る' if m_sec_mixed['sharpe'] < m_all['sharpe'] else '△ 同等'}
     フジコ法オンリー: Sharpe={m_sec_fj['sharpe']:.3f}  →  {'✅ 上回る' if m_sec_fj['sharpe'] > m_all['sharpe'] else '❌ 下回る' if m_sec_fj['sharpe'] < m_all['sharpe'] else '△ 同等'}
  """)

    print(f"""  ② シグナル頻度と平均保有銘柄数の変化（vs G27）
     現行G27:            平均保有={m_g27['avg_pos']:.2f}銘柄  取引数={m_g27['trades']}件/7年
     Sector(混合):        平均保有={m_sec_mixed['avg_pos']:.2f}銘柄  取引数={m_sec_mixed['trades']}件/7年
     Sector(フジコ):      平均保有={m_sec_fj['avg_pos']:.2f}銘柄  取引数={m_sec_fj['trades']}件/7年
     年間BUYシグナル:      {annual_signals:.1f}件/年（Sector Filter / フジコ法）
  """)

    # フジコ法が景気循環セクター特化かの判定
    sec_fj_vs_all_sharpe = m_sec_fj['sharpe'] - m_all['sharpe']
    sec_fj_vs_all_cagr   = m_sec_fj['cagr']   - m_all['cagr']

    if sec_fj_vs_all_sharpe > 0.2:
        cyclical_verdict = "強く支持される ✅"
        cyclical_detail  = "景気循環7セクターに絞ることでSharpeが大幅改善。フジコ法は景気循環戦略。"
    elif sec_fj_vs_all_sharpe > 0:
        cyclical_verdict = "支持される △"
        cyclical_detail  = "セクター絞り込みにより緩やかな改善。フジコ法は景気循環系に向く傾向あり。"
    else:
        cyclical_verdict = "支持されない ❌"
        cyclical_detail  = "セクター絞り込みの効果が見られない。他の要因（個別銘柄の質）が支配的。"

    print(f"""  ③ フジコ法が「景気循環セクター特化戦略」である可能性
     Sector(フジコ) vs 全銘柄: ΔSharpe={sec_fj_vs_all_sharpe:+.3f}  ΔCAGR={sec_fj_vs_all_cagr*100:+.1f}%
     仮説の支持度: {cyclical_verdict}
     詳細: {cyclical_detail}
  """)

    # ── [7] セクター別パフォーマンス（Sector Filter / フジコ法）─────
    print("=" * 80)
    print("【Sector Filter / フジコ法オンリー — セクター別パフォーマンス】")
    print("=" * 80)
    df_sec = _sector_summary(r_sec_fj)
    if not df_sec.empty:
        print(f"\n  {'セクター':<12} {'取引数':>6} {'勝率%':>7} {'平均PnL万':>9} {'合計PnL万':>9}")
        print("  " + "-" * 48)
        for _, row in df_sec.iterrows():
            print(f"  {row['セクター']:<12} {row['取引数']:>6} {row['勝率%']:>7.1f} "
                  f"{row['平均PnL(万円)']:>9.2f} {row['合計PnL(万円)']:>9.1f}")

    print("\n【G27 — セクター別パフォーマンス（比較用）】")
    df_sec_g27 = _sector_summary(r_g27)
    if not df_sec_g27.empty:
        print(f"\n  {'セクター':<12} {'取引数':>6} {'勝率%':>7} {'平均PnL万':>9} {'合計PnL万':>9}")
        print("  " + "-" * 48)
        for _, row in df_sec_g27.iterrows():
            print(f"  {row['セクター']:<12} {row['取引数']:>6} {row['勝率%']:>7.1f} "
                  f"{row['平均PnL(万円)']:>9.2f} {row['合計PnL(万円)']:>9.1f}")

    # ── [8] 年次内訳（Sector Filter / フジコ法オンリー）─────────────
    print("\n" + "=" * 80)
    print("【Sector Filter フジコ法オンリー — 年次内訳】")
    print("=" * 80)
    eq = r_sec_fj.equity_curve
    dd = r_sec_fj.dd_series
    print(f"\n  {'年':>4}  {'リターン':>9}  {'MaxDD':>8}  {'Calmar':>7}")
    print("  " + "-" * 35)
    for yr in range(2018, 2025):
        ye = eq[eq.index.year == yr]
        yd = dd[dd.index.year == yr]
        if ye.empty:
            continue
        ret = ye.iloc[-1] / ye.iloc[0] - 1
        mdd_yr = float(yd.min())
        cal_yr = _calmar(ret, mdd_yr) if mdd_yr < 0 else float("inf")
        cal_str = f"{cal_yr:.2f}" if cal_yr != float("inf") else "∞"
        print(f"  {yr:>4}  {ret*100:>+9.1f}%  {mdd_yr*100:>8.1f}%  {cal_str:>7}")

    # ── [9] CSV・結論出力 ─────────────────────────────────────────
    rows = []
    for m in all_m:
        rows.append({
            "シナリオ": m["label"], "銘柄数": m["n_stocks"],
            "CAGR%": round(m["cagr"]*100, 2), "Sharpe": round(m["sharpe"], 3),
            "MaxDD%": round(m["mdd"]*100, 2), "Calmar": round(m["calmar"], 3),
            "取引数": m["trades"], "勝率%": round(m["win_rate"]*100, 1),
            "平均保有銘柄": round(m["avg_pos"], 2),
        })
    df_result = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT, "sector_filter_universe.csv")
    df_result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Universe銘柄リスト
    univ_rows = [{"銘柄": s, "セクター": i["sector"]} for s, i in univ_sector.items()]
    pd.DataFrame(univ_rows).to_csv(
        os.path.join(OUTPUT, "sector_filter_stocks.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\n  結果CSV: {out_csv}")

    print("\n" + "=" * 80)
    print("【Universe設計に関する結論】")
    print("=" * 80)

    best = max(all_m, key=lambda m: m["calmar"])
    print(f"""
  ① 構造フィルター（セクター絞り込み）の有効性
     Calmar最大シナリオ: {best['label']}
     CAGR={best['cagr']*100:+.1f}%  Sharpe={best['sharpe']:.3f}  MaxDD={best['mdd']*100:.1f}%

  ② スクリーニング廃止の影響（前回検証からの継続）
     全銘柄（スクリーニングなし）: Sharpe={m_all['sharpe']:.3f}
     Sector Filter（構造のみ）:  Sharpe={max(m_sec_mixed['sharpe'], m_sec_fj['sharpe']):.3f}
     →セクター構造フィルターによる改善: ΔSharpe={max(m_sec_mixed['sharpe'], m_sec_fj['sharpe'])-m_all['sharpe']:+.3f}

  ③ 個別銘柄スクリーニング vs 構造フィルター
     現行G27（個別選定）: Sharpe={m_g27['sharpe']:.3f}  Calmar={m_g27['calmar']:.3f}
     Sector Filter best : Sharpe={max(m_sec_mixed['sharpe'], m_sec_fj['sharpe']):.3f}  Calmar={max(m_sec_mixed['calmar'], m_sec_fj['calmar'], key=lambda x: x):.3f}
     →{'G27が優位（個別銘柄選定の知恵が活きている）' if m_g27['sharpe'] > max(m_sec_mixed['sharpe'], m_sec_fj['sharpe']) else 'Sector Filterが優位（構造フィルターの方が汎化性が高い）'}
    """)

    print("完了。")
    return {
        "univ_sector": univ_sector,
        "metrics": all_m,
        "annual_signals": annual_signals,
    }


if __name__ == "__main__":
    main()
