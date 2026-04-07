"""
run_cap_reject_analysis.py
--------------------------
Cap Reject Missed Alpha 分析スクリプト

【目的】
2024-04-01 〜 2025-03-31 において Bear/Bull モード時に
スコア上位に入ったが max_positions=3 制約で発注されなかった銘柄の
機会損失（Missed Alpha）を定量化し、動的 cap 緩和条件を提案する。

【cap reject の定義】
  (A) max_positions reject : スコアが採用閾値以上だが N=3 で入れなかった銘柄
  (B) bear_universe_filter reject : 海運業等の固定除外セクターで弾かれた銘柄
  (C) symbol_cap structural mismatch : order_weight(≒33%) > symbol_cap(8%) の構造的問題

出力: C:/ai-trading/runtime/cap_reject_analysis.json

実行:
  cd C:/ai-trading
  python -m scripts.run_cap_reject_analysis
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ─── パス設定 ──────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

# PROJECT_ROOT = C:/ai-trading/src, BASE_DIR = C:/ai-trading
try:
    from src.paths import BASE_DIR, BACKTEST_DATASET_DIR, RSR_UNIVERSE_FILE, RUNTIME_DIR
    from src.config_loader import load_strategy_config
except ImportError:
    # フォールバック: スタンドアロン実行
    BASE_DIR = Path("C:/ai-trading")
    BACKTEST_DATASET_DIR = BASE_DIR / "data" / "backtest_dataset"
    RSR_UNIVERSE_FILE = BASE_DIR / "src" / "configs" / "rsr_universe_42.csv"
    RUNTIME_DIR = BASE_DIR / "runtime"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RUNTIME_DIR / "cap_reject_analysis.json"

# ─── 分析パラメータ ────────────────────────────────────────────
ANALYSIS_START = "2024-04-01"
ANALYSIS_END   = "2025-03-31"

# 確定パラメータ（strategy.yaml と同値）
MAX_POSITIONS    = 3
CAPITAL          = 3_000_000
SYMBOL_CAP       = 0.08          # pre_trade_risk_check の symbol_cap
MA200_PERIOD     = 200
BEAR_DAYS_THRESH = 40            # 直近60日のうち40日以上MA200割れ → 持続Bear
BEAR_LOOKBACK    = 60

BULL_ACTIVE_N    = 30
BEAR_ACTIVE_N    = 20

# ダイナミックユニバース設定（strategy.yaml 準拠）
BULL_WEIGHTS = {"mom_63": 0.40, "rsr": 0.35, "log_vol": 0.25}
BEAR_WEIGHTS = {"rs_topix": 0.50, "rsr": 0.30, "log_vol": 0.20}

# bear_universe_filter で除外されるセクター（strategy.yaml の risk_controls）
BEAR_EXCLUDE_SECTORS = {"機械", "鉄鋼", "銀行業", "保険業", "輸送用機器", "海運業", "化学",
                        # CSVでの略称も対応
                        "銀行", "保険", "輸送機器", "海運"}

SHIPPING_SYMS = {"9101.T", "9104.T", "9107.T"}

FORWARD_DAYS = 5   # 機会損失計測: N営業日後リターン

# ─── データロード ──────────────────────────────────────────────
def _load_universe_csv() -> pd.DataFrame:
    """RSR42 ユニバース CSV を読み込む（symbol, sector）"""
    if not RSR_UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"RSR universe CSV not found: {RSR_UNIVERSE_FILE}")
    return pd.read_csv(RSR_UNIVERSE_FILE)


def _find_dataset_version() -> Path:
    """利用可能な最新の backtest_dataset バージョンディレクトリを返す。"""
    if not BACKTEST_DATASET_DIR.exists():
        raise FileNotFoundError(f"backtest_dataset not found: {BACKTEST_DATASET_DIR}")
    versions = sorted(
        p for p in BACKTEST_DATASET_DIR.iterdir()
        if p.is_dir() and (p / "_meta.json").exists()
    )
    if not versions:
        raise FileNotFoundError("No versioned snapshot found in backtest_dataset/")
    return versions[-1]


def _load_close_series(sym: str, dataset_dir: Path) -> pd.Series | None:
    """指定シンボルの終値 Series を返す。存在しない場合は None。"""
    p = dataset_dir / f"{sym}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"  [WARN] {sym} 読み込み失敗: {e}")
        return None


def load_all_prices(syms: list[str], dataset_dir: Path) -> Dict[str, pd.Series]:
    """全シンボルの終値 Series を dict で返す。"""
    prices: Dict[str, pd.Series] = {}
    for sym in syms:
        s = _load_close_series(sym, dataset_dir)
        if s is not None:
            prices[sym] = s
    return prices


# ─── レジーム判定 ──────────────────────────────────────────────
def calc_bear_regime(topix: pd.Series) -> pd.Series:
    """
    各日の Bear フラグを返す。
    TOPIX < MA200 かつ 直近 BEAR_LOOKBACK 日のうち BEAR_DAYS_THRESH 日以上 MA200 割れ
    = 持続 Bear → True
    """
    ma200 = topix.rolling(MA200_PERIOD, min_periods=MA200_PERIOD).mean()
    below = (topix < ma200).astype(int)
    below_count = below.rolling(BEAR_LOOKBACK, min_periods=1).sum()
    is_bear = below_count >= BEAR_DAYS_THRESH
    return is_bear


# ─── スコアリング ──────────────────────────────────────────────
def _calc_rsr(prices_df: pd.DataFrame, period: int = 63) -> pd.DataFrame:
    """RSR（Relative Strength Rank）: 63日リターンのランク"""
    ret = prices_df.pct_change(period)
    return ret.rank(axis=1, pct=True)


def _calc_rs_topix(prices_df: pd.DataFrame, topix: pd.Series, period: int = 63) -> pd.DataFrame:
    """TOPIX 比較 RS: 銘柄リターン - TOPIX リターン"""
    topix_ret = topix.pct_change(period)
    sym_ret = prices_df.pct_change(period)
    rs = sym_ret.subtract(topix_ret, axis=0)
    return rs


def _calc_mom63(prices_df: pd.DataFrame) -> pd.DataFrame:
    """63日モメンタム（リターン）"""
    return prices_df.pct_change(63)


def _calc_log_vol(prices_df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """対数ボラティリティの逆数スコア（低ボラ = 高スコア）"""
    log_ret = np.log(prices_df / prices_df.shift(1))
    vol = log_ret.rolling(window, min_periods=5).std()
    # 低ボラ優先: -log_vol をスコア化して rank
    return (-vol).rank(axis=1, pct=True)


def calc_bull_scores(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Bull スコア: mom_63(0.40) + rsr(0.35) + log_vol(0.25)"""
    rsr    = _calc_rsr(prices_df, 63)
    mom63  = _calc_mom63(prices_df).rank(axis=1, pct=True)
    logvol = _calc_log_vol(prices_df)
    score = (
        BULL_WEIGHTS["mom_63"]  * mom63  +
        BULL_WEIGHTS["rsr"]     * rsr    +
        BULL_WEIGHTS["log_vol"] * logvol
    )
    return score


def calc_bear_scores(prices_df: pd.DataFrame, topix: pd.Series) -> pd.DataFrame:
    """Bear スコア: rs_topix(0.50) + rsr(0.30) + log_vol(0.20)"""
    rs_topix = _calc_rs_topix(prices_df, topix, 63).rank(axis=1, pct=True)
    rsr      = _calc_rsr(prices_df, 63)
    logvol   = _calc_log_vol(prices_df)
    score = (
        BEAR_WEIGHTS["rs_topix"] * rs_topix +
        BEAR_WEIGHTS["rsr"]      * rsr      +
        BEAR_WEIGHTS["log_vol"]  * logvol
    )
    return score


# ─── 先読み防止: スコアは shift(1) して前日データで判定 ─────────
def shift_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    return score_df.shift(1)


# ─── 機会損失分析 ──────────────────────────────────────────────
def calc_forward_return(prices: Dict[str, pd.Series], sym: str, date: pd.Timestamp) -> float | None:
    """date 翌営業日 → FORWARD_DAYS 営業日後の終値リターンを計算する。"""
    if sym not in prices:
        return None
    s = prices[sym]
    # date 以降のインデックス
    future = s.loc[s.index > date]
    if len(future) <= FORWARD_DAYS:
        return None
    entry_price = future.iloc[0]   # 翌営業日終値（エントリー日近似）
    exit_price  = future.iloc[FORWARD_DAYS]
    if entry_price <= 0:
        return None
    return float(exit_price / entry_price - 1.0)


def run_analysis(
    sym_sector: pd.DataFrame,
    prices: Dict[str, pd.Series],
    topix: pd.Series,
    is_bear: pd.Series,
) -> dict:
    """
    メイン分析ループ。

    Returns
    -------
    dict: 分析結果
    """
    syms = list(prices.keys())
    syms = [s for s in syms if s != "1306.T"]  # TOPIXを除外

    # sector マップ
    sector_map = dict(zip(sym_sector["symbol"], sym_sector["sector"]))

    # 全銘柄の価格行列（共通インデックスに揃える）
    all_closes = pd.DataFrame({s: prices[s] for s in syms}).sort_index()

    # 分析期間にトリム
    ts_start = pd.Timestamp(ANALYSIS_START)
    ts_end   = pd.Timestamp(ANALYSIS_END)
    period_idx = all_closes.index[(all_closes.index >= ts_start) & (all_closes.index <= ts_end)]

    # スコア行列を計算（先読み防止: shift(1)）
    bull_scores = shift_scores(calc_bull_scores(all_closes))
    bear_scores = shift_scores(calc_bear_scores(all_closes, topix))

    # bear モードでの rs_topix（rs>0 フィルター用）
    rs_topix_raw = _calc_rs_topix(all_closes, topix, 63).shift(1)

    # ────────────────────────────────────────────────────
    # 日次シミュレーション
    # ────────────────────────────────────────────────────
    records_cap_reject: List[dict] = []       # max_positions reject
    records_adopted:    List[dict] = []       # 採用された銘柄
    records_bu_filter:  List[dict] = []       # bear_universe_filter reject

    for date in period_idx:
        regime_bear = bool(is_bear.get(date, False))
        score_row   = bear_scores.loc[date] if regime_bear else bull_scores.loc[date]
        score_row   = score_row.dropna()

        if regime_bear:
            # bear_rs_filter: rs_topix > 0 のみ対象
            if date in rs_topix_raw.index:
                rs_row = rs_topix_raw.loc[date].dropna()
                valid_syms = rs_row[rs_row > 0].index.tolist()
                score_row = score_row.reindex(valid_syms).dropna()

            # bear_universe_filter（固定除外セクター）
            bu_excluded = []
            bu_kept     = []
            for sym in score_row.index:
                sec = sector_map.get(sym, "")
                if sec in BEAR_EXCLUDE_SECTORS:
                    bu_excluded.append(sym)
                else:
                    bu_kept.append(sym)

            # フィルター後スコア
            score_filtered = score_row.reindex(bu_kept).dropna()

            # bear_universe_filter で除外された銘柄の仮想リターン記録
            for sym in bu_excluded:
                ret = calc_forward_return(prices, sym, date)
                if ret is not None:
                    records_bu_filter.append({
                        "date":   str(date.date()),
                        "symbol": sym,
                        "sector": sector_map.get(sym, ""),
                        "score":  float(score_row.get(sym, np.nan)),
                        "return_5d": ret,
                        "regime": "bear",
                    })

            active_n = BEAR_ACTIVE_N
        else:
            score_filtered = score_row
            bu_excluded    = []
            active_n       = BULL_ACTIVE_N

        # スコア上位 active_n → さらに max_positions (N=3) で絞る
        ranked = score_filtered.sort_values(ascending=False)

        adopted_syms     = ranked.iloc[:MAX_POSITIONS].index.tolist()
        cap_rejected_syms = ranked.iloc[MAX_POSITIONS:active_n].index.tolist()

        # 採用銘柄のリターン
        for sym in adopted_syms:
            ret = calc_forward_return(prices, sym, date)
            if ret is not None:
                records_adopted.append({
                    "date":   str(date.date()),
                    "symbol": sym,
                    "sector": sector_map.get(sym, ""),
                    "score":  float(ranked.get(sym, np.nan)),
                    "return_5d": ret,
                    "regime": "bear" if regime_bear else "bull",
                })

        # max_positions cap reject 銘柄のリターン
        for sym in cap_rejected_syms:
            ret = calc_forward_return(prices, sym, date)
            if ret is not None:
                records_cap_reject.append({
                    "date":   str(date.date()),
                    "symbol": sym,
                    "sector": sector_map.get(sym, ""),
                    "score":  float(ranked.get(sym, np.nan)),
                    "return_5d": ret,
                    "regime": "bear" if regime_bear else "bull",
                })

    return {
        "cap_reject":  records_cap_reject,
        "adopted":     records_adopted,
        "bu_filter":   records_bu_filter,
        "sector_map":  sector_map,
    }


# ─── 集計 ──────────────────────────────────────────────────────
def aggregate(results: dict) -> dict:
    """分析結果を集計して辞書で返す。"""
    cap_rej = pd.DataFrame(results["cap_reject"])
    adopted = pd.DataFrame(results["adopted"])
    bu_filt = pd.DataFrame(results["bu_filter"])
    smap    = results["sector_map"]

    def _stats(df: pd.DataFrame, label: str) -> dict:
        if df.empty or "return_5d" not in df.columns:
            return {"count": 0, "mean": None, "median": None, "label": label}
        r = df["return_5d"]
        return {
            "count":  int(len(r)),
            "mean":   float(r.mean()),
            "median": float(r.median()),
            "label":  label,
        }

    # ── 全体集計 ────────────────────────────────────────────────
    stats_cap_rej = _stats(cap_rej, "cap_reject (max_positions)")
    stats_adopted = _stats(adopted, "adopted")

    # ── シンボル別 cap reject 頻度 ──────────────────────────────
    if not cap_rej.empty:
        sym_freq = (
            cap_rej.groupby("symbol")
            .agg(count=("return_5d", "size"), mean_ret=("return_5d", "mean"))
            .sort_values("count", ascending=False)
            .head(15)
        )
        top_syms = [
            {"symbol": idx, "count": int(row["count"]), "mean_5d": float(row["mean_ret"])}
            for idx, row in sym_freq.iterrows()
        ]
    else:
        top_syms = []

    # ── セクター別集計（cap reject）────────────────────────────
    if not cap_rej.empty:
        sec_stats = (
            cap_rej.groupby("sector")
            .agg(count=("return_5d", "size"), mean_ret=("return_5d", "mean"))
            .sort_values("count", ascending=False)
        )
        sector_summary = [
            {
                "sector":  sec,
                "count":   int(row["count"]),
                "mean_5d": float(row["mean_ret"]),
            }
            for sec, row in sec_stats.iterrows()
        ]
    else:
        sector_summary = []

    # ── 海運セクター専用分析 ────────────────────────────────────
    shipping_bu    = bu_filt[bu_filt["symbol"].isin(SHIPPING_SYMS)] if not bu_filt.empty else pd.DataFrame()
    shipping_cap   = cap_rej[cap_rej["symbol"].isin(SHIPPING_SYMS)] if not cap_rej.empty else pd.DataFrame()

    shipping_stats = {
        "bu_filter_reject": _stats(shipping_bu, "shipping bear_universe_filter reject"),
        "cap_reject":       _stats(shipping_cap, "shipping max_positions reject"),
    }

    # ── symbol_cap 構造的ミスマッチ分析 ────────────────────────
    # order_weight = capital_per_position / capital
    # = (capital / max_positions) / capital = 1/3 ≒ 33%
    # symbol_cap = 8% → 構造的に全 BUY が reject される
    order_weight_equal = 1.0 / MAX_POSITIONS  # = 0.333
    symbol_cap_val     = SYMBOL_CAP           # = 0.08
    structural_reject = order_weight_equal > symbol_cap_val

    # ── regime 別比較 ────────────────────────────────────────────
    for df_name, df in [("cap_reject", cap_rej), ("adopted", adopted)]:
        pass  # below

    regime_comparison = {}
    for regime in ["bear", "bull"]:
        cr = cap_rej[cap_rej["regime"] == regime] if not cap_rej.empty else pd.DataFrame()
        ad = adopted[adopted["regime"] == regime] if not adopted.empty else pd.DataFrame()
        regime_comparison[regime] = {
            "cap_reject": _stats(cr, f"{regime} cap_reject"),
            "adopted":    _stats(ad, f"{regime} adopted"),
        }

    return {
        "period":          {"start": ANALYSIS_START, "end": ANALYSIS_END},
        "params":          {
            "max_positions": MAX_POSITIONS,
            "symbol_cap":    SYMBOL_CAP,
            "capital":       CAPITAL,
        },
        "overall": {
            "cap_reject": stats_cap_rej,
            "adopted":    stats_adopted,
            "opportunity_loss_per_event": (
                (stats_cap_rej["mean"] or 0) - (stats_adopted["mean"] or 0)
                if stats_cap_rej["mean"] is not None and stats_adopted["mean"] is not None
                else None
            ),
        },
        "top_cap_reject_symbols":    top_syms,
        "sector_cap_reject_summary": sector_summary,
        "shipping_analysis":         shipping_stats,
        "regime_comparison":         regime_comparison,
        "structural_analysis": {
            "order_weight_equal_3pos": order_weight_equal,
            "symbol_cap":              symbol_cap_val,
            "structural_reject_flag":  structural_reject,
            "note": (
                "equal-weight(N=3)時の order_weight≒33% が symbol_cap=8% を大幅超過。"
                "現行設定では pre_trade_risk_check が実質的に全 BUY を弾く構造。"
                "symbol_cap を 0.35 以上に引き上げるか、または max_single_weight(0.25)に統一が必要。"
            ) if structural_reject else "問題なし",
        },
    }


# ─── 動的 cap 緩和条件の提案生成 ──────────────────────────────
def generate_proposals(agg: dict) -> List[dict]:
    """分析結果に基づく動的 cap 緩和条件を提案する。"""
    proposals = []

    overall_cr_mean = agg["overall"]["cap_reject"].get("mean")
    overall_ad_mean = agg["overall"]["adopted"].get("mean")

    # 提案1: max_positions 4への拡張
    if overall_cr_mean is not None and overall_cr_mean > 0.02:
        cond = f"cap_reject_alpha_5d > 2% (実測 {overall_cr_mean*100:.2f}%)"
        proposals.append({
            "id": 1,
            "title": "max_positions 3→4 への一時拡張",
            "trigger_condition": cond,
            "proposed_rule": (
                "Bear/Bull 共通: スコア4位銘柄の期待5日リターン推定値 > 2% の場合に "
                "max_positions を当日のみ 4 に拡張する。"
                "ただし max_single_weight=0.25 を維持し、4ポジション時は均等25%配分。"
            ),
            "risk_note": "DD計算・CB閾値は capital ベース維持。4ポジション時は個別リスク増。",
        })
    else:
        proposals.append({
            "id": 1,
            "title": "max_positions 3→4 への一時拡張",
            "trigger_condition": "cap_reject_alpha_5d > 2%",
            "proposed_rule": "分析期間の機会損失が閾値未満のため現状維持を推奨。",
            "risk_note": "",
        })

    # 提案2: 海運セクター temporary overweight
    ship_bu = agg["shipping_analysis"]["bu_filter_reject"]
    ship_mean = ship_bu.get("mean")
    if ship_mean is not None and ship_mean > 0.01:
        proposals.append({
            "id": 2,
            "title": "海運セクター temporary overweight 許容",
            "trigger_condition": (
                f"海運業 bear_universe_filter 除外時の平均5日リターン={ship_mean*100:.2f}% > 1%"
            ),
            "proposed_rule": (
                "Bear 時の bear_universe_filter 除外対象の海運業を以下条件で例外許容:\n"
                "  - rs_topix_63d > 0.05（TOPIX を5%以上アウトパフォーム）\n"
                "  - 直近20日リターン > 0（短期モメンタム陽転）\n"
                "  - max_positions は 3 を維持（既存3銘柄の1枠を海運で置換）\n"
                "実装: apply_bear_universe_filter() に rs_threshold 条件を追加"
            ),
            "risk_note": "海運は景気敏感。Bear 中の一時強さが反転するリスクあり。RSフィルターを厳格に。",
        })
    else:
        proposals.append({
            "id": 2,
            "title": "海運セクター temporary overweight 許容",
            "trigger_condition": "海運除外時の期待リターン > 1%",
            "proposed_rule": (
                f"海運除外時リターン={ship_mean*100:.2f}% が閾値1%未満のため現行フィルター維持を推奨。"
                if ship_mean is not None else
                "データ不足のため判断保留。海運銘柄が分析期間にフィルター除外された日が存在しない可能性。"
            ),
            "risk_note": "",
        })

    # 提案3: cluster cap regime 連動
    proposals.append({
        "id": 3,
        "title": "cluster cap の regime 連動緩和",
        "trigger_condition": "Bear regime 時",
        "proposed_rule": (
            "Bear regime では防御セクター（食品・ガス・情報通信）への集中を許容するため "
            "cluster_cap を 1/N(≒0.33) → 2/N(≒0.67) に緩和する。\n"
            "  - strategy.yaml: bear_cluster_cap: 0.67（現行0.25から変更）\n"
            "  - 適用条件: is_bear=True かつ cluster セクターが防御セクターの場合のみ\n"
            "  - Bull regime は現行 cluster_cap=0.35 を維持"
        ),
        "risk_note": "防御セクター集中はドローダウン抑制に寄与するが、回復局面でのアンダーパフォームに注意。",
    })

    # 提案4: symbol_cap の修正（構造的問題）
    if agg["structural_analysis"]["structural_reject_flag"]:
        proposals.append({
            "id": 4,
            "title": "symbol_cap 構造的ミスマッチの修正（最重要）",
            "trigger_condition": "常時",
            "proposed_rule": (
                "現行 symbol_cap=0.08(8%) は equal-weight(N=3) 時の order_weight≒33% を大幅下回り、"
                "実質的に全 BUY が pre_trade_risk_check で拒否される。\n"
                "以下いずれかを選択:\n"
                "  Option A: symbol_cap を max_single_weight(0.25) に統一 → 0.08→0.25\n"
                "  Option B: symbol_cap を (1/max_positions * 1.2) = 0.40 に設定\n"
                "  Option C: symbol_cap チェックを symbol の既存保有ウェイトのみに変更\n"
                "            （新規 order_weight のチェックは max_single_weight で代替）\n"
                "推奨: Option A（最もシンプル、既存 max_single_weight=0.25 と整合）"
            ),
            "risk_note": "変更前に WF 再検証を実施すること（CLAUDE.md 必須確認ルール）。",
        })

    return proposals


# ─── レポート出力 ──────────────────────────────────────────────
def print_report(agg: dict, proposals: List[dict]) -> None:
    """分析レポートをターミナルに出力する。"""
    sep = "=" * 60

    print(f"\n{sep}")
    print("=== Cap Reject Missed Alpha 分析レポート ===")
    print(sep)

    print(f"\n【分析期間】{agg['period']['start']} 〜 {agg['period']['end']}")
    print(f"【パラメータ】max_positions={agg['params']['max_positions']}, "
          f"symbol_cap={agg['params']['symbol_cap']}, capital={agg['params']['capital']:,}円")

    # ── 構造的問題（最優先）──────────────────────────────────────
    sa = agg["structural_analysis"]
    print(f"\n【⚠ 構造的問題: symbol_cap ミスマッチ】")
    print(f"  order_weight (equal N=3): {sa['order_weight_equal_3pos']*100:.1f}%")
    print(f"  symbol_cap              : {sa['symbol_cap']*100:.1f}%")
    print(f"  structural_reject       : {'YES（全 BUY が拒否される）' if sa['structural_reject_flag'] else 'No'}")
    if sa["structural_reject_flag"]:
        print(f"  → {sa['note']}")

    # ── Cap Reject 件数 ──────────────────────────────────────────
    cr = agg["overall"]["cap_reject"]
    ad = agg["overall"]["adopted"]
    print(f"\n【Cap Reject 件数 (max_positions 超過)】")
    print(f"  総発生件数: {cr['count']}件")

    if agg["top_cap_reject_symbols"]:
        top_str = ", ".join(
            f"{r['symbol']}({r['count']}件,{r['mean_5d']*100:+.2f}%)"
            for r in agg["top_cap_reject_symbols"][:8]
        )
        print(f"  上位シンボル: {top_str}")

    # ── 5日リターン比較 ──────────────────────────────────────────
    print(f"\n【5日リターン比較】")
    print(f"  Cap reject シンボル: 平均={cr['mean']*100:+.2f}%, 中央値={cr['median']*100:+.2f}%" if cr["mean"] is not None else "  Cap reject: データなし")
    print(f"  採用シンボル       : 平均={ad['mean']*100:+.2f}%, 中央値={ad['median']*100:+.2f}%" if ad["mean"] is not None else "  採用: データなし")

    opp = agg["overall"]["opportunity_loss_per_event"]
    if opp is not None:
        direction = "機会損失" if opp < 0 else "採用銘柄が優秀"
        print(f"  → Cap reject − 採用: {opp*100:+.2f}% ({direction})")

    # ── セクター別 ───────────────────────────────────────────────
    print(f"\n【セクター別 cap reject 頻度と平均リターン】")
    for s in agg["sector_cap_reject_summary"]:
        print(f"  {s['sector']:10s}: {s['count']:4d}件  平均5d={s['mean_5d']*100:+.2f}%")

    # ── regime 別 ────────────────────────────────────────────────
    print(f"\n【regime 別 比較】")
    for regime, rc in agg["regime_comparison"].items():
        cr_r = rc["cap_reject"]
        ad_r = rc["adopted"]
        cr_s = f"平均={cr_r['mean']*100:+.2f}%({cr_r['count']}件)" if cr_r["mean"] is not None else "データなし"
        ad_s = f"平均={ad_r['mean']*100:+.2f}%({ad_r['count']}件)" if ad_r["mean"] is not None else "データなし"
        print(f"  {regime.upper():5s}: cap_reject={cr_s}  採用={ad_s}")

    # ── 海運セクター ─────────────────────────────────────────────
    print(f"\n【海運セクター分析 (9101.T / 9104.T / 9107.T)】")
    ship = agg["shipping_analysis"]
    bu_s  = ship["bu_filter_reject"]
    cap_s = ship["cap_reject"]
    if bu_s["mean"] is not None:
        print(f"  bear_universe_filter 除外件数: {bu_s['count']}件")
        print(f"  除外時の仮想5日リターン: 平均={bu_s['mean']*100:+.2f}%, 中央値={bu_s['median']*100:+.2f}%")
    else:
        print(f"  bear_universe_filter 除外: 0件（分析期間中は Bear に入らなかった可能性）")
    if cap_s["mean"] is not None:
        print(f"  max_positions cap reject件数: {cap_s['count']}件")
        print(f"  reject時の仮想5日リターン: 平均={cap_s['mean']*100:+.2f}%, 中央値={cap_s['median']*100:+.2f}%")
    else:
        print(f"  max_positions cap reject: 0件")

    # ── 提案 ─────────────────────────────────────────────────────
    print(f"\n【動的 cap 緩和条件（提案）】")
    for p in proposals:
        print(f"\n  {p['id']}. {p['title']}")
        print(f"     発動条件: {p['trigger_condition']}")
        for line in p['proposed_rule'].split('\n'):
            print(f"     {line}")
        if p.get("risk_note"):
            print(f"     ⚠ リスク: {p['risk_note']}")

    print(f"\n{sep}")
    print(f"出力ファイル: {OUTPUT_FILE}")
    print(sep)


# ─── メイン ────────────────────────────────────────────────────
def main() -> int:
    print("=== Cap Reject Missed Alpha 分析 起動 ===")

    # 1. ユニバース CSV 読み込み
    sym_sector = _load_universe_csv()
    all_syms   = sym_sector["symbol"].tolist()
    print(f"  ユニバース: {len(all_syms)}銘柄")

    # 2. データセット特定
    try:
        dataset_dir = _find_dataset_version()
        print(f"  データセット: {dataset_dir.name}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return 1

    # 3. TOPIX ロード
    topix_raw = _load_close_series("1306.T", dataset_dir)
    if topix_raw is None:
        print("  [ERROR] TOPIX (1306.T) データが見つかりません")
        return 1
    topix = topix_raw.sort_index()

    # 4. 全銘柄ロード
    print(f"  価格データ読み込み中...")
    prices = load_all_prices(all_syms + ["1306.T"], dataset_dir)
    available = [s for s in all_syms if s in prices]
    print(f"  取得成功: {len(available)}/{len(all_syms)}銘柄")

    missing = [s for s in all_syms if s not in prices]
    if missing:
        print(f"  [WARN] 欠損銘柄: {missing}")

    # 5. レジーム判定
    is_bear = calc_bear_regime(topix)
    bear_days = int(is_bear.loc[
        (is_bear.index >= pd.Timestamp(ANALYSIS_START)) &
        (is_bear.index <= pd.Timestamp(ANALYSIS_END))
    ].sum())
    total_days = int(is_bear.loc[
        (is_bear.index >= pd.Timestamp(ANALYSIS_START)) &
        (is_bear.index <= pd.Timestamp(ANALYSIS_END))
    ].count())
    print(f"  Bear日数: {bear_days}/{total_days}日 ({bear_days/max(1,total_days)*100:.1f}%)")

    # 6. 分析実行
    print(f"  シミュレーション実行中 ({ANALYSIS_START} 〜 {ANALYSIS_END})...")
    results = run_analysis(sym_sector, prices, topix, is_bear)
    print(f"  cap reject イベント数: {len(results['cap_reject'])}件")
    print(f"  採用イベント数        : {len(results['adopted'])}件")
    print(f"  bu_filter reject件数  : {len(results['bu_filter'])}件")

    # 7. 集計
    agg = aggregate(results)

    # 8. 提案生成
    proposals = generate_proposals(agg)
    agg["proposals"] = proposals

    # 9. JSON 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  結果保存: {OUTPUT_FILE}")

    # 10. レポート出力
    print_report(agg, proposals)

    return 0


if __name__ == "__main__":
    sys.exit(main())
