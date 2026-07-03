"""
ユニバース拡張 × スクリーニング廃止 検証スクリプト

【検証設計】
  銘柄事前スクリーニング（Sharpe>0.3）を廃止し、
  シグナル（RSR≥70・SEPA≥6）に銘柄選択を委ねる

  比較軸:
    - ユニバースサイズ: 27銘柄(現行) / 74銘柄(全量) / 拡張銘柄
    - max_positions: 3 / 5 / 8
    - スクリーニング: あり(現行) / なし(提案)

  主要指標:
    - CAGR / Sharpe / MaxDD / Calmar
    - 平均保有銘柄数（avg_positions）
    - 資本稼働率（capital_utilization = 1 - 平均キャッシュ比率）
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
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

# ── 現行27銘柄（スクリーニング済み）──────────────────────────────
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
}

# ── TOPIX100_TICKERSに含まれない追加銘柄（拡張用）─────────────────
EXTRA_TICKERS: dict[str, str] = {
    # 電機・精密
    "6367.T": "電機精密",   # ダイキン工業
    "6503.T": "電機",       # 三菱電機
    "6752.T": "電機",       # パナソニックHD
    "6902.T": "電機精密",   # デンソー
    "6954.T": "電機精密",   # ファナック
    "6857.T": "電機精密",   # アドバンテスト
    "6594.T": "電機精密",   # ニデック
    "6758.T": "電機",       # ソニーグループ
    "6861.T": "電機精密",   # キーエンス
    # 情報通信
    "9433.T": "情報通信",   # KDDI
    "9984.T": "情報通信",   # ソフトバンクグループ
    "4307.T": "情報通信",   # 野村総合研究所
    # 医薬品
    "4519.T": "医薬品",     # 中外製薬
    "4523.T": "医薬品",     # エーザイ
    "4568.T": "医薬品",     # 第一三共
    "4502.T": "医薬品",     # 武田薬品
    # 商社・小売
    "8031.T": "商社",       # 三井物産
    "3382.T": "小売",       # セブン&アイ（重複）
    "2802.T": "食品",       # 味の素
    # 機械・輸送
    "7267.T": "輸送機器",   # ホンダ
    "7261.T": "輸送機器",   # マツダ
    "7011.T": "機械",       # 三菱重工（重複）
    "6201.T": "機械",       # 豊田自動織機
    # 不動産
    "8802.T": "不動産",     # 三菱地所
    "8830.T": "不動産",     # 住友不動産
    "3003.T": "不動産",     # ヒューリック
    # 海運・陸運
    "9107.T": "海運",       # 川崎汽船
    "9020.T": "陸運",       # JR東日本
    "9022.T": "陸運",       # JR東海
    # 保険・銀行
    "8750.T": "保険",       # 第一生命HD
    "8316.T": "銀行",       # 三井住友FG
    # 石油・化学
    "5020.T": "石油",       # ENEOSホールディングス
    "4063.T": "化学",       # 信越化学
}

SECTOR_STRATEGY = {
    "海運":    "フジコ法", "機械":    "フジコ法", "電機精密": "フジコ法",
    "商社":    "フジコ法", "電機":    "フジコ法", "食品":    "フジコ法",
    "ガス":    "平均回帰", "鉄鋼":    "平均回帰", "銀行":    "平均回帰",
    "保険":    "平均回帰", "輸送機器": "平均回帰", "化学":    "平均回帰",
    "情報通信": "フジコ法", "小売":    "フジコ法",
    "医薬品":  "フジコ法", "不動産":  "フジコ法", "陸運":    "平均回帰",
    "石油":    "平均回帰",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10,
)


def build_and_run(data_dict: dict, max_positions: int = 3, min_rsr: float = 70.0) -> object:
    """スクリーニングなしでPortfolioEngineを構築・実行"""
    price_df = pd.DataFrame({s: i["df"]["Close"] for s, i in data_dict.items()})
    rsr_uni  = calc_universe_rsr(price_df)

    strat_dict = {}
    for sym, info in data_dict.items():
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "フジコ法")
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        if rule == "平均回帰":
            strat_dict[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat_dict[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=min_rsr,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )

    return PortfolioEngine(
        universe=data_dict, strategy=strat_dict, capital=CAPITAL,
        max_dd_limit=0.15, min_sectors=1,
        max_positions=max_positions,
        cost=TradeCost(), vol_target=0.0,
        max_single_weight=min(0.25, 1.0 / max_positions),
        use_idm=False,
    ).run()


def metrics(r, max_positions: int) -> dict:
    sharpe = r.sharpe_ratio
    mdd    = r.max_drawdown
    cal    = r.cagr / abs(mdd) if mdd < 0 else 0.0
    avg_pos  = float(r.n_positions.mean()) if hasattr(r, "n_positions") else float("nan")
    cap_util = avg_pos / max_positions if max_positions > 0 else float("nan")
    return dict(
        cagr=r.cagr, sharpe=sharpe, mdd=mdd, calmar=cal,
        trades=r.n_trades, avg_pos=avg_pos, cap_util=cap_util,
        max_positions=max_positions,
    )


def print_row(label: str, m: dict, max_pos: int, n_stocks: int):
    print(
        f"{label:<28} {n_stocks:>6} {max_pos:>6} "
        f"{m['cagr']*100:>+7.1f}% {m['sharpe']:>7.3f} "
        f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} "
        f"{m['avg_pos']:>7.2f} {m['cap_util']*100:>8.1f}% "
        f"{m['trades']:>7}"
    )


# ──────────────────────────────────────────────────────────────────
# 0. データ取得
# ──────────────────────────────────────────────────────────────────
print("=" * 90)
print("【0】データ取得")
print("=" * 90)

# 74銘柄（TOPIX100_TICKERS）
print("TOPIX100_TICKERS 取得中...")
data_74 = download_universe(TOPIX100_TICKERS, start=START, end=END, verbose=False)
print(f"  74銘柄定義 → 取得成功: {len(data_74)} 銘柄")

# 拡張銘柄（重複除去）
extra_unique = {k: v for k, v in EXTRA_TICKERS.items() if k not in data_74}
print(f"拡張銘柄 取得中（{len(extra_unique)}銘柄）...")
data_extra = download_universe(extra_unique, start=START, end=END, verbose=False)
print(f"  取得成功: {len(data_extra)} 銘柄")

# 現行27銘柄
data_27 = {k: data_74[k] for k in G27_SCREENED if k in data_74}
print(f"\n現行27銘柄（スクリーニング済み）: {len(data_27)} 銘柄確認")
print(f"TOPIX100全量（スクリーニングなし）: {len(data_74)} 銘柄")

# 拡張ユニバース
data_expanded = {**data_74, **data_extra}
print(f"拡張ユニバース（74+追加）: {len(data_expanded)} 銘柄")


# ──────────────────────────────────────────────────────────────────
# 1. メイン比較テーブル
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("【1】スクリーニングあり vs なし × max_positions 比較")
print("=" * 90)

HDR = (f"{'シナリオ':<28} {'銘柄数':>6} {'maxP':>6} "
       f"{'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
       f"{'平均保有':>7} {'稼働率':>8} {'取引数':>7}")
print(HDR)
print("-" * 90)

results = {}

# A: 現行（スクリーニングあり, 27銘柄, max_pos=3）
label = "A: 現行(27銘柄/スクリーニング済)"
r = build_and_run(data_27, max_positions=3)
m = metrics(r, 3)
results["A_current"] = (m, r)
print_row(label, m, 3, len(data_27))

print()

# B: 74銘柄スクリーニングなし × max_positions 3/5/8
for mp in [3, 5, 8]:
    label = f"B: 74銘柄/スクリーニングなし/p={mp}"
    r = build_and_run(data_74, max_positions=mp)
    m = metrics(r, mp)
    results[f"B_74_p{mp}"] = (m, r)
    print_row(label, m, mp, len(data_74))

print()

# C: 拡張ユニバース × max_positions 3/5/8
for mp in [3, 5, 8]:
    label = f"C: 拡張({len(data_expanded)}銘柄)/スクリーニングなし/p={mp}"
    r = build_and_run(data_expanded, max_positions=mp)
    m = metrics(r, mp)
    results[f"C_exp_p{mp}"] = (m, r)
    print_row(label, m, mp, len(data_expanded))


# ──────────────────────────────────────────────────────────────────
# 2. 資本稼働率の詳細分析
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("【2】資本稼働率と平均保有銘柄数の詳細")
print("=" * 90)

print(f"\n{'シナリオ':<40} {'平均保有':>8} {'最大':>6} {'0銘柄日%':>10} {'稼働率':>8}")
print("-" * 75)

for key, (m, r) in results.items():
    np_series = r.n_positions
    zero_days_pct = (np_series == 0).mean() * 100
    max_pos = np_series.max()
    label_map = {
        "A_current":   "A: 現行(27銘柄/スクリーニング済)/p=3",
        "B_74_p3":     "B: 74銘柄/スクリーニングなし/p=3",
        "B_74_p5":     "B: 74銘柄/スクリーニングなし/p=5",
        "B_74_p8":     "B: 74銘柄/スクリーニングなし/p=8",
        f"C_exp_p3":   f"C: 拡張/スクリーニングなし/p=3",
        f"C_exp_p5":   f"C: 拡張/スクリーニングなし/p=5",
        f"C_exp_p8":   f"C: 拡張/スクリーニングなし/p=8",
    }
    lbl = label_map.get(key, key)
    print(f"{lbl:<40} {m['avg_pos']:>8.2f} {max_pos:>6} {zero_days_pct:>9.1f}% {m['cap_util']*100:>8.1f}%")


# ──────────────────────────────────────────────────────────────────
# 3. セクター別シグナル頻度（銘柄選択がシグナル任せになるか確認）
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("【3】74銘柄/p=5 のセクター別エントリー回数（シグナル選別の確認）")
print("=" * 90)

_, r_b5 = results["B_74_p5"]
if not r_b5.trades.empty:
    buys = r_b5.trades[r_b5.trades["side"] == "BUY"].copy()
    if "sector" in buys.columns:
        sector_cnt = buys.groupby("sector").size().sort_values(ascending=False)
        print(f"\n{'セクター':<15} {'BUY回数':>8}")
        print("-" * 25)
        for sec, cnt in sector_cnt.items():
            print(f"{sec:<15} {cnt:>8}")
    else:
        # symbolからセクター逆引き
        sym_to_sector = {s: i["sector"] for s, i in data_74.items()}
        buys["sector"] = buys["symbol"].map(sym_to_sector)
        sector_cnt = buys.groupby("sector").size().sort_values(ascending=False)
        print(f"\n{'セクター':<15} {'BUY回数':>8}")
        print("-" * 25)
        for sec, cnt in sector_cnt.items():
            print(f"{sec:<15} {cnt:>8}")
else:
    print("取引なし")


# ──────────────────────────────────────────────────────────────────
# 4. ウォークフォワード（74銘柄/p=5 の過学習チェック）
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("【4】ウォークフォワード（74銘柄/スクリーニングなし/p=5）")
print("=" * 90)

windows = [
    ("2018-2019", "2020"),
    ("2019-2020", "2021"),
    ("2020-2021", "2022"),
    ("2021-2022", "2023"),
    ("2022-2023", "2024"),
]

print(f"\n{'IS期間':<14} {'OOS期間':<10} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'比率':>7}")
print("-" * 56)

oos_is_ratios = []
for is_label, oos_year in windows:
    is_start_yr, is_end_yr = is_label.split("-")
    is_start = f"{is_start_yr}-01-01"
    is_end   = f"{is_end_yr}-12-31"
    oos_start = f"{oos_year}-01-01"
    oos_end   = f"{oos_year}-12-31"

    # IS用データ
    data_is = {s: {"df": i["df"].loc[is_start:is_end], "sector": i["sector"]}
               for s, i in data_74.items()
               if len(i["df"].loc[is_start:is_end]) >= 100}
    # OOS用データ（RSR計算のためIS+OOSが必要）
    data_oos = {s: {"df": i["df"].loc[oos_start:oos_end], "sector": i["sector"]}
                for s, i in data_74.items()
                if len(i["df"].loc[oos_start:oos_end]) >= 50}

    try:
        r_is  = build_and_run(data_is,  max_positions=5)
        r_oos = build_and_run(data_oos, max_positions=5)
        ratio = r_oos.sharpe_ratio / r_is.sharpe_ratio if r_is.sharpe_ratio != 0 else float("nan")
        oos_is_ratios.append(ratio)
        flag = "⚠" if ratio < 0 else ""
        print(f"{is_label:<14} {oos_year:<10} {r_is.sharpe_ratio:>10.3f} {r_oos.sharpe_ratio:>11.3f} {ratio:>7.2f} {flag}")
    except Exception as e:
        print(f"{is_label:<14} {oos_year:<10} エラー: {e}")

if oos_is_ratios:
    valid = [x for x in oos_is_ratios if not np.isnan(x)]
    print(f"{'平均':<14} {'':<10} {'':<10} {'':<11} {np.mean(valid):>7.2f}")
    print(f"\n→ OOS/IS平均={np.mean(valid):.2f}  {'✅ 過学習なし' if np.mean(valid) > 0.7 else '⚠ 過学習の可能性'}")


# ──────────────────────────────────────────────────────────────────
# 5. 総括
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("【5】総括：スクリーニング廃止の効果")
print("=" * 90)

m_a, _  = results["A_current"]
m_b5, _ = results["B_74_p5"]
m_c5, _ = results.get("C_exp_p5", results["B_74_p5"])

print(f"""
                        A: 現行        B: 74銘柄/p=5  C: 拡張/p=5
  スクリーニング         : あり          なし           なし
  銘柄数                : {len(data_27):>4}           {len(data_74):>4}           {len(data_expanded):>4}
  max_positions         :    3              5              5
  CAGR                  : {m_a['cagr']*100:>+7.1f}%      {m_b5['cagr']*100:>+7.1f}%      {m_c5['cagr']*100:>+7.1f}%
  Sharpe                : {m_a['sharpe']:>7.3f}        {m_b5['sharpe']:>7.3f}        {m_c5['sharpe']:>7.3f}
  MaxDD                 : {m_a['mdd']*100:>7.1f}%      {m_b5['mdd']*100:>7.1f}%      {m_c5['mdd']*100:>7.1f}%
  Calmar                : {m_a['calmar']:>7.3f}        {m_b5['calmar']:>7.3f}        {m_c5['calmar']:>7.3f}
  平均保有銘柄数         : {m_a['avg_pos']:>7.2f}        {m_b5['avg_pos']:>7.2f}        {m_c5['avg_pos']:>7.2f}
  資本稼働率            : {m_a['cap_util']*100:>7.1f}%      {m_b5['cap_util']*100:>7.1f}%      {m_c5['cap_util']*100:>7.1f}%

【仮説検証】
  「CAGRのボトルネックは戦略の質ではなく資本稼働率」
  → 上記の平均保有銘柄数・資本稼働率の変化で判定
""")

print("=" * 90)
print("検証完了")
print("=" * 90)
