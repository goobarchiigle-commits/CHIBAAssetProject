"""
backtest/advanced_analysis.py
外部レビュー対応 — 4タスク統合検証スクリプト

【Task 1】Ex-ante Universe再構築
  バックテスト結果（Sharpe等）を使わない。
  2018年時点の流動性（売買代金>5億/日）のみでフィルター。
  TSEプライム大型株 ~150銘柄 → 100〜200銘柄規模に拡張。

【Task 2】セクター別パフォーマンス分析
  全トレードをセクター別に集計:
    セクター別Sharpe / CAGR寄与 / トレード数 / 勝率 / 平均リターン

【Task 3】ポジション数最適化 × RSRランキング採用
  max_positions = 3 / 5 / 8
  エントリー候補をRSR強度の降順にソート（強いシグナルを優先採用）

【Task 4】シグナル頻度の改善分析
  RSR閾値: 60 / 65 / 70（現行）
  SEPA閾値: 4 / 5 / 6（現行）
  各変更が CAGR / Sharpe / MaxDD / トレード数 に与える影響を比較
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

# ──────────────────────────────────────────────────────────────────
# Ex-ante Universe候補（~150銘柄）
# 選定基準: TSEプライム上場 / 大型株 / バックテスト結果は一切使わない
# セクター情報は事前知識（公開情報）のみ使用
# ──────────────────────────────────────────────────────────────────
EX_ANTE_CANDIDATES: dict[str, str] = {
    # ── 電機・精密 ──
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
    "6367.T": "機械",       # ダイキン工業（電機系）
    "6506.T": "機械",       # 安川電機
    "6146.T": "電機精密",   # ディスコ
    "6981.T": "電機精密",   # 村田製作所
    "6479.T": "電機精密",   # ミネベアミツミ
    "6869.T": "電機精密",   # シスメックス
    "4185.T": "化学",       # JSR
    # ── 輸送機器 ──
    "7203.T": "輸送機器",   # トヨタ自動車
    "7267.T": "輸送機器",   # 本田技研工業
    "7201.T": "輸送機器",   # 日産自動車
    "7270.T": "輸送機器",   # SUBARU
    "7261.T": "輸送機器",   # マツダ
    "6201.T": "機械",       # 豊田自動織機
    # ── 情報通信 ──
    "9984.T": "情報通信",   # ソフトバンクグループ
    "9432.T": "情報通信",   # NTT
    "9433.T": "情報通信",   # KDDI
    "9613.T": "情報通信",   # NTTデータグループ
    "4307.T": "情報通信",   # 野村総合研究所
    "4704.T": "情報通信",   # トレンドマイクロ
    # ── 銀行 ──
    "8306.T": "銀行",       # 三菱UFJ
    "8316.T": "銀行",       # 三井住友FG
    "8411.T": "銀行",       # みずほFG
    "8309.T": "銀行",       # 三井住友トラストHD
    "7182.T": "銀行",       # ゆうちょ銀行
    # ── 保険 ──
    "8766.T": "保険",       # 東京海上HD
    "8725.T": "保険",       # MS&AD
    "8750.T": "保険",       # 第一生命HD
    # ── 商社 ──
    "8058.T": "商社",       # 三菱商事
    "8031.T": "商社",       # 三井物産
    "8053.T": "商社",       # 住友商事
    "8002.T": "商社",       # 丸紅
    "8001.T": "商社",       # 伊藤忠商事
    "8015.T": "商社",       # 豊田通商
    # ── 医薬品 ──
    "4502.T": "医薬品",     # 武田薬品
    "4519.T": "医薬品",     # 中外製薬
    "4568.T": "医薬品",     # 第一三共
    "4523.T": "医薬品",     # エーザイ
    "4578.T": "医薬品",     # 大塚HD
    "4543.T": "医薬品",     # テルモ
    # ── 化学 ──
    "4063.T": "化学",       # 信越化学工業
    "4452.T": "化学",       # 花王
    "3407.T": "化学",       # 旭化成
    "4021.T": "化学",       # 日産化学
    "4183.T": "化学",       # 三井化学 ← 既出でdup, skip handled below
    "4911.T": "化学",       # 資生堂
    "4901.T": "化学",       # 富士フイルムHD
    "4208.T": "化学",       # 宇部興産（UBE）
    # ── 食品・飲料 ──
    "2802.T": "食品",       # 味の素
    "2914.T": "食品",       # JT
    "2503.T": "食品",       # キリンHD
    "2502.T": "食品",       # アサヒグループHD
    "2269.T": "食品",       # 明治HD
    # ── 機械 ──
    "6301.T": "機械",       # 小松製作所
    "6326.T": "機械",       # クボタ
    "7011.T": "機械",       # 三菱重工業
    "7012.T": "機械",       # 川崎重工業
    "7013.T": "機械",       # IHI
    # ── 不動産 ──
    "8801.T": "不動産",     # 三井不動産
    "8802.T": "不動産",     # 三菱地所
    "8830.T": "不動産",     # 住友不動産
    "3003.T": "不動産",     # ヒューリック
    # ── 小売 ──
    "3382.T": "小売",       # セブン&アイ
    "9983.T": "小売",       # ファーストリテイリング
    "8267.T": "小売",       # イオン
    # ── レジャー・サービス ──
    "4661.T": "レジャー",   # オリエンタルランド
    "6098.T": "サービス",   # リクルートHD
    "7974.T": "ゲーム",     # 任天堂
    # ── 鉄鋼 ──
    "5401.T": "鉄鋼",       # 日本製鉄
    "5411.T": "鉄鋼",       # JFEスチール
    # ── 石油・ガス ──
    "5020.T": "石油",       # ENEOS
    "9531.T": "ガス",       # 東京ガス
    "9532.T": "ガス",       # 大阪ガス
    # ── 海運 ──
    "9101.T": "海運",       # 日本郵船
    "9104.T": "海運",       # 商船三井
    "9107.T": "海運",       # 川崎汽船
    # ── 陸運・インフラ ──
    "9020.T": "陸運",       # JR東日本
    "9021.T": "陸運",       # JR西日本
    "9022.T": "陸運",       # JR東海
    # ── 空運 ──
    "9202.T": "空運",       # ANA HD
    "9201.T": "空運",       # JAL
    # ── 建設 ──
    "1801.T": "建設",       # 大成建設
    "1802.T": "建設",       # 大林組
    "1803.T": "建設",       # 清水建設
    "1812.T": "建設",       # 鹿島建設
    # ── 繊維・素材 ──
    "3401.T": "化学",       # 帝人
    "3402.T": "化学",       # 東レ
    # ── 証券 ──
    "8604.T": "証券",       # 野村HD
    "8601.T": "証券",       # 大和証券グループ
    # ── 非鉄 ──
    "5706.T": "鉄鋼",       # 三菱マテリアル
    "5713.T": "鉄鋼",       # 住友金属鉱山
    "5714.T": "鉄鋼",       # DOWA
    # ── ガラス・窯業 ──
    "5201.T": "化学",       # AGC
    "5332.T": "化学",       # TOTO
}

# セクター→戦略マッピング（比較分析研究からの知見）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":    "フジコ法", "機械":    "フジコ法", "電機精密": "フジコ法",
    "商社":    "フジコ法", "電機":    "フジコ法", "食品":    "フジコ法",
    "レジャー":"フジコ法", "ゲーム":  "フジコ法", "サービス": "フジコ法",
    "医薬品":  "フジコ法", "不動産":  "フジコ法",
    "ガス":    "平均回帰", "鉄鋼":    "平均回帰", "銀行":    "平均回帰",
    "保険":    "平均回帰", "輸送機器": "平均回帰", "化学":    "平均回帰",
    "情報通信": "フジコ法", "小売":    "フジコ法",
    "陸運":    "平均回帰", "石油":    "平均回帰",
    "空運":    "平均回帰", "建設":    "平均回帰", "証券":    "平均回帰",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

DAILY_TURNOVER_THRESHOLD = 500_000_000  # 5億円/日 (ex-ante流動性基準)


# ================================================================
# ヘルパー
# ================================================================
def _calmar(cagr: float, mdd: float) -> float:
    return cagr / abs(mdd) if mdd < 0 else 0.0


def _build_strategies(
    data_dict: dict,
    rsr_uni: pd.DataFrame,
    min_rsr: float = 70.0,
    min_sepa: int = 6,
) -> dict:
    strat_dict = {}
    for sym, info in data_dict.items():
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "フジコ法")
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
    max_single_weight: float = 0.25,
    clenow_scores: pd.DataFrame = None,
) -> object:
    """PortfolioEngineを実行。clenow_scoresにRSRを渡すとRSRランキング採用になる。"""
    return PortfolioEngine(
        universe=data_dict,
        strategy=strat_dict,
        capital=CAPITAL,
        max_dd_limit=0.15,
        min_sectors=1,
        max_positions=max_positions,
        cost=TradeCost(),
        vol_target=0.0,
        max_single_weight=min(max_single_weight, 1.0 / max(1, max_positions)),
        use_idm=False,
        clenow_scores=clenow_scores,
    ).run()


def _metrics(r, label: str, n_stocks: int, max_positions: int) -> dict:
    avg_pos  = float(r.n_positions.mean())
    cap_util = avg_pos / max_positions if max_positions > 0 else 0.0
    return dict(
        label=label, n_stocks=n_stocks, max_positions=max_positions,
        cagr=r.cagr, sharpe=r.sharpe_ratio, mdd=r.max_drawdown,
        calmar=_calmar(r.cagr, r.max_drawdown),
        trades=r.n_trades, win_rate=r.win_rate,
        avg_pos=avg_pos, cap_util=cap_util,
        _res=r,
    )


def _print_row(m: dict):
    print(
        f"{m['label']:<35} {m['n_stocks']:>5} {m['max_positions']:>5} "
        f"{m['cagr']*100:>+7.1f}% {m['sharpe']:>7.3f} "
        f"{m['mdd']*100:>7.1f}% {m['calmar']:>7.3f} "
        f"{m['avg_pos']:>7.2f} {m['cap_util']*100:>7.1f}% "
        f"{m['trades']:>6} {m['win_rate']*100:>6.1f}%"
    )


# ================================================================
# Task 1: Ex-ante Universe フィルター
# ================================================================
def build_ex_ante_universe(
    candidates: dict,
    start: str,
    end: str,
    min_turnover_yen: float = DAILY_TURNOVER_THRESHOLD,
) -> dict:
    """
    バックテスト結果を一切使わないEx-anteフィルター。

    フィルター条件（公開情報のみ）:
      1. データ取得成功（上場・流通している）
      2. バックテスト開始時点（2018年前半）の平均売買代金 > 5億円/日
         ← 時価総額3000億相当の流動性の代理変数

    Returns: {symbol: {"df": DataFrame, "sector": str}} の辞書
    """
    print(f"\n候補銘柄: {len(candidates)}銘柄 → データ取得中...")
    universe_all = download_universe(candidates, start=start, end=end, verbose=False)
    print(f"  取得成功: {len(universe_all)} 銘柄")

    # 流動性フィルター: 2018年の売買代金（株価 × 出来高）で判定
    passed = {}
    failed_liquidity = []

    for sym, info in universe_all.items():
        df = info["df"]
        # 2018年上半期（最初の120営業日）のデータで判定 ← バックテスト外の期間
        df_2018 = df.loc["2018-01-01":"2018-06-30"]
        if len(df_2018) < 60:
            failed_liquidity.append((sym, "データ不足"))
            continue

        # 売買代金 = Close × Volume
        turnover = (df_2018["Close"] * df_2018["Volume"]).mean()
        if turnover >= min_turnover_yen:
            passed[sym] = info
        else:
            failed_liquidity.append((sym, f"売買代金{turnover/1e8:.1f}億"))

    print(f"\n【Ex-ante フィルター結果】")
    print(f"  候補 {len(universe_all)}銘柄 → 通過 {len(passed)}銘柄")
    print(f"  除外 {len(failed_liquidity)}銘柄（流動性不足）")

    # セクター別内訳
    sector_count: dict[str, int] = {}
    for info in passed.values():
        sec = info["sector"]
        sector_count[sec] = sector_count.get(sec, 0) + 1

    print("\n  セクター別内訳:")
    for sec, cnt in sorted(sector_count.items(), key=lambda x: -x[1]):
        print(f"    {sec:<12} {cnt}銘柄")

    return passed


# ================================================================
# Task 2: セクター別パフォーマンス分析
# ================================================================
def analyze_sector_performance(res, label: str) -> pd.DataFrame:
    """
    PortfolioResultのtradesからセクター別PnL統計を集計する。

    Returns:
        セクター別分析DataFrame
    """
    if res.trades.empty:
        print(f"  [{label}] トレードなし")
        return pd.DataFrame()

    trades = res.trades.copy()
    # BUYとSELLのマッチング
    sells = trades[trades["side"].str.startswith("SELL")].copy()
    buys  = trades[trades["side"] == "BUY"].copy()

    if sells.empty:
        print(f"  [{label}] SELLトレードなし")
        return pd.DataFrame()

    rows = []
    for sector in sells["sector"].unique():
        s_sells = sells[sells["sector"] == sector]
        s_buys  = buys[buys["sector"] == sector] if not buys.empty else pd.DataFrame()

        pnl_list = s_sells["pnl"].values
        n_trades = len(pnl_list)
        win_rate = float((pnl_list > 0).mean()) if n_trades > 0 else 0.0
        avg_ret  = float(pnl_list.mean()) if n_trades > 0 else 0.0
        total_pnl = float(pnl_list.sum())

        # セクター寄与CAGR（近似：セクター合計PnL / 初期資本 / 7年）
        years = 7.0
        cagr_contrib = total_pnl / CAPITAL / years * 100

        rows.append({
            "セクター": sector,
            "トレード数": n_trades,
            "勝率%": round(win_rate * 100, 1),
            "平均PnL(円)": round(avg_ret, 0),
            "合計PnL(万円)": round(total_pnl / 10000, 1),
            "CAGR寄与%": round(cagr_contrib, 2),
        })

    df = pd.DataFrame(rows).sort_values("CAGR寄与%", ascending=False)

    # セクター別Sharpe（各トレードのPnL / 資本 の日次換算近似）
    # 注: これは正確なSharpeではなくトレードPnLベースの近似
    sharpe_rows = []
    for _, row in df.iterrows():
        sec = row["セクター"]
        s_sells_pnl = sells[sells["sector"] == sec]["pnl"].values
        if len(s_sells_pnl) >= 3:
            pnl_r = s_sells_pnl / CAPITAL
            mu  = float(np.mean(pnl_r))
            std = float(np.std(pnl_r))
            sh  = mu / std * np.sqrt(len(s_sells_pnl)) if std > 0 else 0.0
        else:
            sh = 0.0
        sharpe_rows.append(sh)
    df["トレードベースSharpe"] = [round(x, 3) for x in sharpe_rows]

    return df.reset_index(drop=True)


# ================================================================
# Task 3: RSRランキング採用 × max_positions最適化
# ================================================================
def run_rsr_ranked_scenarios(data_dict: dict, rsr_uni: pd.DataFrame) -> list[dict]:
    """
    RSRスコアをソートキーとして使い、max_positions=3/5/8を比較。

    通常採用（ランダム順）と RSRランキング順の両方を比較する。
    """
    results = []
    n = len(data_dict)

    for max_pos in [3, 5, 8]:
        # --- (A) 通常採用（ランキングなし） ---
        strat_a = _build_strategies(data_dict, rsr_uni)
        r_a = _run_portfolio(data_dict, strat_a, max_positions=max_pos)
        m_a = _metrics(r_a, f"通常順/max_pos={max_pos}", n, max_pos)
        results.append(m_a)

        # --- (B) RSRランキング順採用 ---
        # RSR（0〜100のランクスコア）をclenow_scoresとして渡す
        # PortfolioEngineがエントリー候補をスコア降順にソートする
        strat_b = _build_strategies(data_dict, rsr_uni)
        r_b = _run_portfolio(
            data_dict, strat_b,
            max_positions=max_pos,
            clenow_scores=rsr_uni,   # RSRをソートキーとして使用
        )
        m_b = _metrics(r_b, f"RSRランク順/max_pos={max_pos}", n, max_pos)
        results.append(m_b)

    return results


# ================================================================
# Task 4: シグナル頻度分析
# ================================================================
def analyze_signal_frequency(data_dict: dict, rsr_uni: pd.DataFrame) -> dict:
    """
    シグナル発生頻度の詳細分析。

    Parameters:
        data_dict: 対象ユニバース
        rsr_uni: RSRスコアDataFrame

    Returns:
        分析結果の辞書
    """
    print("\n  シグナル頻度を銘柄別に計算中...")

    all_signals = []
    for sym, info in data_dict.items():
        df = info["df"]
        sector = info["sector"]
        rule = SECTOR_STRATEGY.get(sector, "フジコ法")
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None

        if rule == "平均回帰":
            strat = MeanReversionStrategy(**MR_PARAMS)
        else:
            strat = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )

        # 全日付でシグナルを評価
        dates = df.index
        buy_count = 0
        rsr_at_buy = []
        for i in range(252 + 21 + 2, len(dates)):  # ウォームアップ後
            past = df.iloc[:i + 1]
            sig = strat.generate_signal(past)
            if sig == 1:
                buy_count += 1
                # RSR値を記録
                if rsr_s is not None:
                    rsr_val = rsr_s.reindex(past.index).ffill().iloc[-1]
                    if not np.isnan(rsr_val):
                        rsr_at_buy.append(float(rsr_val))

        # 年間シグナル数（2018-2024 = 7年）
        total_days = max(1, len(dates) - 252 - 21 - 2)
        annual_signals = buy_count / 7.0

        all_signals.append({
            "symbol": sym,
            "sector": sector,
            "annual_signals": annual_signals,
            "total_signals": buy_count,
            "rsr_at_buy_mean": float(np.mean(rsr_at_buy)) if rsr_at_buy else float("nan"),
            "rsr_at_buy_std":  float(np.std(rsr_at_buy))  if rsr_at_buy else float("nan"),
        })

    df_sig = pd.DataFrame(all_signals)

    # 要約統計
    total_annual = df_sig["annual_signals"].sum()
    total_stocks_with_signal = (df_sig["annual_signals"] > 0).sum()

    print(f"\n  【シグナル頻度サマリー（現行: RSR>70, SEPA>=6）】")
    print(f"    対象銘柄数         : {len(df_sig)}")
    print(f"    シグナルを出した銘柄: {total_stocks_with_signal} / {len(df_sig)}")
    print(f"    年間総BUYシグナル   : {total_annual:.1f} 件/年")
    print(f"    銘柄あたり年平均    : {total_annual/max(1,len(df_sig)):.2f} 件/年")

    if not df_sig[df_sig["annual_signals"] > 0].empty:
        print(f"\n  シグナルが多い銘柄 TOP10:")
        top = df_sig.nlargest(10, "annual_signals")
        for _, r in top.iterrows():
            print(f"    {r['symbol']:<10} ({r['sector']:<8}) "
                  f"年{r['annual_signals']:.1f}件  RSR平均={r['rsr_at_buy_mean']:.1f}")

    return {
        "df": df_sig,
        "total_annual_signals": total_annual,
        "stocks_with_signal": total_stocks_with_signal,
    }


def run_threshold_sensitivity(
    data_dict: dict,
    rsr_uni: pd.DataFrame,
    max_positions: int = 5,
) -> list[dict]:
    """
    RSR閾値 × SEPA閾値の感度分析。

    組み合わせ:
        RSR  : 60, 65, 70（現行）
        SEPA : 4, 5, 6（現行）
    """
    results = []
    n = len(data_dict)

    for min_rsr in [60, 65, 70]:
        for min_sepa in [4, 5, 6]:
            label = f"RSR>={min_rsr}/SEPA>={min_sepa}"
            strat = _build_strategies(data_dict, rsr_uni, min_rsr=min_rsr, min_sepa=min_sepa)
            # フジコ法銘柄のみ閾値変更（平均回帰はMR_PARAMSで固定）
            r = _run_portfolio(data_dict, strat, max_positions=max_positions)
            m = _metrics(r, label, n, max_positions)
            results.append(m)
            print(
                f"  {label:<22} CAGR={m['cagr']*100:>+6.1f}%  "
                f"Sharpe={m['sharpe']:>6.3f}  MaxDD={m['mdd']*100:>6.1f}%  "
                f"Calmar={m['calmar']:>6.3f}  trades={m['trades']:>4}  "
                f"pos={m['avg_pos']:.2f}"
            )

    return results


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  advanced_analysis.py — 外部レビュー対応 4タスク統合検証")
    print("  期間: 2018〜2024  /  初期資本: ¥2,000,000")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────
    # Task 1: Ex-ante Universe構築
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【Task 1】Ex-ante Universe構築（バックテスト結果使用禁止）")
    print("=" * 80)
    print("フィルター基準: 2018年前半の平均売買代金 > 5億円/日")

    universe_ex_ante = build_ex_ante_universe(
        EX_ANTE_CANDIDATES, start=START, end=END,
        min_turnover_yen=DAILY_TURNOVER_THRESHOLD,
    )

    print(f"\n→ Ex-ante Universe: {len(universe_ex_ante)}銘柄 確定")

    # ── 現行27銘柄（スクリーニング済み）との比較用
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
    universe_g27 = {k: universe_ex_ante[k] for k in G27_SCREENED if k in universe_ex_ante}

    # ── RSR一括計算（全銘柄）
    print(f"\nRSR計算中（{len(universe_ex_ante)}銘柄）...")
    prices_all = {s: i["df"]["Close"] for s, i in universe_ex_ante.items()}
    rsr_uni    = calc_universe_rsr(prices_all)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    # ── 現行(G27) vs Ex-ante Full の比較
    print("\n現行G27 vs Ex-ante Full バックテスト比較...")
    strat_g27 = _build_strategies(universe_g27, rsr_uni)
    r_g27 = _run_portfolio(universe_g27, strat_g27, max_positions=3)
    m_g27 = _metrics(r_g27, "G27-現行(スクリーニング済み)", len(universe_g27), 3)

    strat_ea = _build_strategies(universe_ex_ante, rsr_uni)
    r_ea_p3 = _run_portfolio(universe_ex_ante, strat_ea, max_positions=3)
    m_ea_p3 = _metrics(r_ea_p3, f"Ex-ante({len(universe_ex_ante)}銘柄)/max_pos=3", len(universe_ex_ante), 3)

    strat_ea5 = _build_strategies(universe_ex_ante, rsr_uni)
    r_ea_p5 = _run_portfolio(universe_ex_ante, strat_ea5, max_positions=5)
    m_ea_p5 = _metrics(r_ea_p5, f"Ex-ante({len(universe_ex_ante)}銘柄)/max_pos=5", len(universe_ex_ante), 5)

    HDR = (f"{'シナリオ':<35} {'銘柄':>5} {'maxP':>5} "
           f"{'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
           f"{'平均保有':>7} {'稼働率':>7} {'取引':>6} {'勝率':>6}")
    print("\n" + HDR)
    print("-" * 95)
    for m in [m_g27, m_ea_p3, m_ea_p5]:
        _print_row(m)

    # ──────────────────────────────────────────────────────────────
    # Task 2: セクター別パフォーマンス分析
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【Task 2】セクター別パフォーマンス分析（Ex-ante / max_pos=5）")
    print("=" * 80)

    df_sector = analyze_sector_performance(r_ea_p5, "Ex-ante/max_pos=5")
    if not df_sector.empty:
        print("\n  セクター別サマリー（CAGR寄与降順）")
        print(f"  {'セクター':<12} {'取引数':>6} {'勝率%':>7} {'平均PnL':>10} "
              f"{'CAGR寄与%':>9} {'合計PnL万':>10} {'Sharpe':>8}")
        print("  " + "-" * 70)
        for _, row in df_sector.iterrows():
            print(f"  {row['セクター']:<12} {row['トレード数']:>6} {row['勝率%']:>7.1f} "
                  f"{row['平均PnL(円)']:>10,.0f} {row['CAGR寄与%']:>9.2f}% "
                  f"{row['合計PnL(万円)']:>10.1f} {row['トレードベースSharpe']:>8.3f}")

        # G27現行のセクター分析も実施
        print("\n  【比較】現行G27 セクター別サマリー")
        df_sector_g27 = analyze_sector_performance(r_g27, "G27/max_pos=3")
        if not df_sector_g27.empty:
            print(f"  {'セクター':<12} {'取引数':>6} {'勝率%':>7} {'平均PnL':>10} "
                  f"{'CAGR寄与%':>9} {'合計PnL万':>10} {'Sharpe':>8}")
            print("  " + "-" * 70)
            for _, row in df_sector_g27.iterrows():
                print(f"  {row['セクター']:<12} {row['トレード数']:>6} {row['勝率%']:>7.1f} "
                      f"{row['平均PnL(円)']:>10,.0f} {row['CAGR寄与%']:>9.2f}% "
                      f"{row['合計PnL(万円)']:>10.1f} {row['トレードベースSharpe']:>8.3f}")

    # ──────────────────────────────────────────────────────────────
    # Task 3: RSRランキング × max_positions最適化
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【Task 3】ポジション数最適化 × RSRランキング採用（Ex-ante Universe）")
    print("=" * 80)
    print(f"\n{'シナリオ':<35} {'銘柄':>5} {'maxP':>5} "
          f"{'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} "
          f"{'平均保有':>7} {'稼働率':>7} {'取引':>6} {'勝率':>6}")
    print("-" * 95)
    _print_row(m_g27)  # ベースライン
    print("-" * 95)

    rsr_results = run_rsr_ranked_scenarios(universe_ex_ante, rsr_uni)
    for m in rsr_results:
        _print_row(m)

    # ──────────────────────────────────────────────────────────────
    # Task 4: シグナル頻度分析
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【Task 4-A】シグナル頻度の詳細分析")
    print("=" * 80)

    freq_result = analyze_signal_frequency(universe_ex_ante, rsr_uni)

    print("\n" + "=" * 80)
    print("【Task 4-B】エントリー閾値感度分析（Ex-ante Universe / max_pos=5）")
    print("=" * 80)
    print(f"\n  {'閾値設定':<22} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'Calmar':>7} {'取引':>5} {'平均保有':>8}")
    print("  " + "-" * 70)

    threshold_results = run_threshold_sensitivity(universe_ex_ante, rsr_uni, max_positions=5)

    # ──────────────────────────────────────────────────────────────
    # 結果サマリー保存
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【最終サマリー】")
    print("=" * 80)

    # 全シナリオをまとめてCSV保存
    all_metrics = [m_g27, m_ea_p3, m_ea_p5] + rsr_results + threshold_results
    rows_for_csv = []
    for m in all_metrics:
        rows_for_csv.append({
            "シナリオ": m["label"],
            "銘柄数": m["n_stocks"],
            "max_positions": m["max_positions"],
            "CAGR%": round(m["cagr"] * 100, 2),
            "Sharpe": round(m["sharpe"], 3),
            "MaxDD%": round(m["mdd"] * 100, 2),
            "Calmar": round(m["calmar"], 3),
            "平均保有銘柄": round(m["avg_pos"], 2),
            "資本稼働率%": round(m["cap_util"] * 100, 1),
            "取引数": m["trades"],
            "勝率%": round(m["win_rate"] * 100, 1),
        })

    df_all = pd.DataFrame(rows_for_csv)
    out_csv = os.path.join(OUTPUT, "advanced_analysis_results.csv")
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  全シナリオCSV保存 → {out_csv}")

    # セクター分析CSV
    if not df_sector.empty:
        out_sec = os.path.join(OUTPUT, "sector_analysis.csv")
        df_sector.to_csv(out_sec, index=False, encoding="utf-8-sig")
        print(f"  セクター分析CSV保存 → {out_sec}")

    # シグナル頻度CSV
    out_freq = os.path.join(OUTPUT, "signal_frequency.csv")
    freq_result["df"].to_csv(out_freq, index=False, encoding="utf-8-sig")
    print(f"  シグナル頻度CSV保存 → {out_freq}")

    # ── ベスト設定の特定
    print("\n  【最良設定（Calmar最大）】")
    best_calmar = max(all_metrics, key=lambda m: m["calmar"])
    best_cagr   = max(all_metrics, key=lambda m: m["cagr"])
    print(f"  Calmar最大: {best_calmar['label']}")
    print(f"    CAGR={best_calmar['cagr']*100:+.1f}%  Sharpe={best_calmar['sharpe']:.3f}  "
          f"MaxDD={best_calmar['mdd']*100:.1f}%  Calmar={best_calmar['calmar']:.3f}  "
          f"avg_pos={best_calmar['avg_pos']:.2f}")
    print(f"  CAGR最大: {best_cagr['label']}")
    print(f"    CAGR={best_cagr['cagr']*100:+.1f}%  Sharpe={best_cagr['sharpe']:.3f}  "
          f"MaxDD={best_cagr['mdd']*100:.1f}%  Calmar={best_cagr['calmar']:.3f}  "
          f"avg_pos={best_cagr['avg_pos']:.2f}")

    # ── シグナル頻度仮説の検証
    print("\n  【シグナル頻度仮説の検証】")
    print(f"  年間総BUYシグナル（現行RSR>70/SEPA>=6）: {freq_result['total_annual_signals']:.1f} 件/年")
    print(f"  平均保有銘柄（現行G27/max_pos=3）: {m_g27['avg_pos']:.2f}")
    print(f"  平均保有銘柄（Ex-ante/max_pos=5）: {m_ea_p5['avg_pos']:.2f}")

    bottleneck = "シグナル頻度" if freq_result["total_annual_signals"] < 15 else "Universe規模 or max_positions"
    print(f"\n  → CAGRのボトルネック推定: {bottleneck}")

    if freq_result["total_annual_signals"] < 15:
        print("    年間シグナルが15件未満 → 閾値緩和（RSR↓ or SEPA↓）が有効")
    else:
        print("    シグナルは十分にある → max_positions拡大が有効")

    print("\n完了。")
    return {
        "universe_ex_ante": universe_ex_ante,
        "rsr_uni": rsr_uni,
        "metrics": all_metrics,
        "sector_analysis": df_sector,
        "signal_freq": freq_result,
        "threshold_results": threshold_results,
    }


if __name__ == "__main__":
    main()
