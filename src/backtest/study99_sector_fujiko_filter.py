"""
src/backtest/study99_sector_fujiko_filter.py
Study99 — Sector × Fujiko filter 存在確認（factor-level・2026-07-16）

正典: ユーザー指示「Study99」（2026-07-16）。Study98（セクターモメンタム持続性・
TOPIX17公式指数）の銘柄レベル拡張。FUJIKO 2.0ロードマップ仮説2
（Market→Sector→Stock階層・Study88/89系列）のfactor-level検証を兼ねる。

目的:
  Analysis A: P(Stock(t+3M) > TOPIX | Sector(t) > TOPIX)
  Analysis B: A + RS score filter（threshold 60/70/75/80）
  Analysis C: 25MA乖離率bin別（0-5 / 5-10 / 10-15 / 15-20 / 20+ %）
  Analysis D: Combined interaction（sector条件 × RS × 25MA乖離）
  共通出力: sample count / hit ratio / excess return / t統計量 /
            subperiod（2016-2020 / 2021-2025）/ regime（TOPIX>MA200）

禁止（タスク仕様・厳守）:
  戦略BT禁止。本スクリプトはfactor-level統計のみ（発注・ポジションサイジング・
  BTエンジン・composite_alpha_bt/fujiko_strategy 一切不使用）。

統治原則3注記（percentile型パラメータ）:
  RS score = canon calc_cross_sectional_rsr（src/data/make_dataset.py）と同一の
  IBD式12M加重複合リターンを、当該月Universe Cメンバー内でパーセンタイルランク化
  （0-100）したもの。本Studyでの閾値60/70/75/80はタスクが明示要求する記述統計上の
  bin分解であり、Production採用パラメータではない。プールサイズ依存性
  （仮説4・pool=Universe C≈1,100銘柄）はレポートに明記する。

データ（全てStudy75A/95/98系譜・fresh run）:
  - backtests/study75_rule_universe.json          Universe C月次PITメンバーシップ
  - data/jquants/processed/{code}.parquet         銘柄別価格（Close=分割調整済み）
  - data/jquants/processed/TOPIX.parquet          銘柄ベンチマーク・レジーム（TOPIX>MA200）
  - database/market/index/prices/{0000,0040-0056}.parquet
                                                  公式TOPIX-17指数（セクター条件・Study98と同一）
  - database/market/master/companies.parquet      Sector17Code（スナップショット・PIT注記あり）

PIT設計（Study95と同一パターン）:
  snapshot_date = rebalance_date直前営業日。RS score・25MA乖離・セクターtrailing excess
  は全てsnapshot_date以前のデータのみで算出（lookahead無し）。
  forward return = rebalance_date終値起点の3M（63営業日）。

セクター条件:
  excess_trailing_3M(snapshot) = sector_3M_ret - topix_3M_ret > 0（公式指数・Study98定義）
  top3 = その月のexcess_trailing_3M降順で上位3セクター（Study98で唯一有意だった切り口）

統計:
  pooled: 2標本比率z（条件群 vs 全量base rate）・Welch t（条件群excess vs 補集合）
  monthly-clustered: 月次平均差系列のt（Fama-MacBeth型・クロスセクション相関補正）
  注意: 3M forwardの月次サンプリングは窓が重複（自己相関でpooled tは過大。FM tを主とする）

出力:
  backtests/study99_sector_fujiko_filter.json
  reports/study99_sector_fujiko_filter.md
  reports/study99_interaction_chart.png
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

from src.database.index_prices import SECTOR17_NAMES, load_all_topix17_series, load_index_series
from src.paths import DATABASE_MASTER_DIR, JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR

rcParams["font.family"] = "MS Gothic"

_JST = timezone(timedelta(hours=9))

# ── 固定パラメータ（事前固定・スイープ禁止） ──────────────────────────
FWD_DAYS = 63                    # 3M forward（営業日）
SECTOR_TRAILING_DAYS = 63        # セクター条件trailing（forwardと対称・Study98の3M）
REGIME_MA_PERIOD = 200           # TOPIX>MA200（canon定義・Study76/95/98と同一）
MA_DEV_PERIOD = 25               # フジコ法25MA
RS_THRESHOLDS = [60, 70, 75, 80]  # タスク指定
TOP_N = 3                        # Study98と同一
FFILL_LIMIT = 2                  # Study95と同一（長期欠損は補完しない）
MA_DEV_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, np.inf)]
SUBPERIODS: dict[str, tuple[int, int]] = {"2016-2020": (2016, 2020), "2021-2025": (2021, 2025)}

UNIVERSE_FILE = RESULTS_DIR / "study75_rule_universe.json"
TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"
COMPANIES_PARQUET = DATABASE_MASTER_DIR / "companies.parquet"

OUT_JSON = RESULTS_DIR / "study99_sector_fujiko_filter.json"
OUT_MD = REPORTS_DIR / "study99_sector_fujiko_filter.md"
OUT_CHART = REPORTS_DIR / "study99_interaction_chart.png"


# ======================================================================
# 1. データ読込
# ======================================================================
def load_universe() -> dict[str, list[str]]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe Cファイルが存在しません: {UNIVERSE_FILE}")
    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    monthly = data.get("monthly_universe")
    if not isinstance(monthly, dict) or not monthly:
        raise ValueError(f"monthly_universe が空/不正です: {UNIVERSE_FILE}")
    return {k: v for k, v in monthly.items() if v}


def load_topix_close() -> pd.Series:
    if not TOPIX_PARQUET.exists():
        raise FileNotFoundError(f"TOPIX価格データが存在しません: {TOPIX_PARQUET}")
    return pd.read_parquet(TOPIX_PARQUET)["Close"].dropna().sort_index()


def load_close_panel(codes: list[str], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """Study95 load_close_panel と同一規約（Close=分割調整済み・ffill limit 2）。"""
    series_map: dict[str, pd.Series] = {}
    missing: list[str] = []
    for code in codes:
        path = JQUANTS_PROCESSED_DIR / f"{code}.parquet"
        if not path.exists():
            missing.append(code)
            continue
        s = pd.read_parquet(path, columns=["Close"])["Close"]
        s = s[~s.index.duplicated(keep="last")]
        series_map[code] = s
    close_df = pd.DataFrame(series_map).reindex(calendar).ffill(limit=FFILL_LIMIT)
    return close_df, missing


def load_sector17_map() -> dict[str, int]:
    comp = pd.read_parquet(COMPANIES_PARQUET, columns=["Code", "Sector17Code"])
    comp = comp.drop_duplicates(subset="Code")
    out: dict[str, int] = {}
    for code, sc in zip(comp["Code"], comp["Sector17Code"]):
        try:
            sc_i = int(sc)
        except (TypeError, ValueError):
            continue
        if 1 <= sc_i <= 17:
            out[str(code)] = sc_i
    return out


def load_sector_indices(calendar: pd.DatetimeIndex) -> tuple[pd.Series, dict[int, pd.Series]]:
    """公式TOPIX-17指数（Study98と同一ソース）。共通カレンダーへreindex（補完なし）。"""
    topix_idx = load_index_series("0000")["Close"].dropna().sort_index().reindex(calendar)
    sector_series = load_all_topix17_series()
    sector_close = {k: v["Close"].dropna().sort_index().reindex(calendar) for k, v in sector_series.items()}
    return topix_idx, sector_close


# ======================================================================
# 2. Factor計算（全てsnapshot時点・lookahead無し）
# ======================================================================
def calc_composite_return_row(close_df: pd.DataFrame, snap_pos: int) -> pd.Series:
    """IBD式12M加重複合リターン（canon calc_cross_sectional_rsr と同一定義）。"""
    if snap_pos - 252 < 0:
        return pd.Series(np.nan, index=close_df.columns)
    p0 = close_df.iloc[snap_pos]
    p63 = close_df.iloc[snap_pos - 63]
    p126 = close_df.iloc[snap_pos - 126]
    p189 = close_df.iloc[snap_pos - 189]
    p252 = close_df.iloc[snap_pos - 252]
    with np.errstate(divide="ignore", invalid="ignore"):
        comp = (0.4 * (p0 / p63 - 1) + 0.2 * (p63 / p126 - 1)
                + 0.2 * (p126 / p189 - 1) + 0.2 * (p189 / p252 - 1))
    return comp.replace([np.inf, -np.inf], np.nan)


def calc_ma_dev_row(close_df: pd.DataFrame, snap_pos: int) -> pd.Series:
    """25MA乖離率（%）。window完全充足のみ（min_periods=25）。"""
    if snap_pos - MA_DEV_PERIOD + 1 < 0:
        return pd.Series(np.nan, index=close_df.columns)
    window = close_df.iloc[snap_pos - MA_DEV_PERIOD + 1: snap_pos + 1]
    ma = window.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = (close_df.iloc[snap_pos] / ma - 1.0) * 100.0
    return dev.replace([np.inf, -np.inf], np.nan)


def calc_sector_condition(
    topix_idx: pd.Series, sector_close: dict[int, pd.Series], snap_pos: int,
) -> dict[int, dict]:
    """snapshot時点のセクターexcess_trailing_3Mとtop3フラグ（Study98定義）。"""
    if snap_pos - SECTOR_TRAILING_DAYS < 0:
        return {}
    t_now, t_prev = topix_idx.iloc[snap_pos], topix_idx.iloc[snap_pos - SECTOR_TRAILING_DAYS]
    if pd.isna(t_now) or pd.isna(t_prev) or t_prev <= 0:
        return {}
    topix_ret = t_now / t_prev - 1.0
    excess: dict[int, float] = {}
    for sc, ser in sector_close.items():
        s_now, s_prev = ser.iloc[snap_pos], ser.iloc[snap_pos - SECTOR_TRAILING_DAYS]
        if pd.isna(s_now) or pd.isna(s_prev) or s_prev <= 0:
            continue
        excess[sc] = (s_now / s_prev - 1.0) - topix_ret
    if not excess:
        return {}
    ranked = sorted(excess.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {sc: i + 1 for i, (sc, _) in enumerate(ranked)}
    return {sc: {"excess_trailing": v, "rank": rank_of[sc],
                 "cond": v > 0, "top3": rank_of[sc] <= TOP_N} for sc, v in excess.items()}


# ======================================================================
# 3. パネル構築
# ======================================================================
def build_panel() -> tuple[pd.DataFrame, dict]:
    universe = load_universe()
    topix = load_topix_close()
    calendar = topix.index
    cal_pos = {d: i for i, d in enumerate(calendar)}

    all_codes = sorted({c for v in universe.values() for c in v})
    close_df, missing = load_close_panel(all_codes, calendar)
    sector_map = load_sector17_map()
    topix_idx, sector_close = load_sector_indices(calendar)

    regime_sma = topix.rolling(REGIME_MA_PERIOD, min_periods=REGIME_MA_PERIOD).mean()
    regime_bull = topix >= regime_sma

    n = len(calendar)
    rows: list[dict] = []
    skipped = {"no_rb_date": 0, "no_sector_cond": 0, "insufficient_fwd": 0}

    for rb_key in sorted(universe.keys()):
        rb_ts = pd.Timestamp(rb_key)
        # rebalance_date = キー当日（営業日でなければ直後の営業日・Study95規約）
        pos_arr = calendar.searchsorted(rb_ts)
        if pos_arr >= n:
            skipped["no_rb_date"] += 1
            continue
        rb_pos = int(pos_arr)
        rb_date = calendar[rb_pos]
        snap_pos = rb_pos - 1
        if snap_pos < 252:
            continue
        if rb_pos + FWD_DAYS >= n:
            skipped["insufficient_fwd"] += 1
            continue

        sector_cond = calc_sector_condition(topix_idx, sector_close, snap_pos)
        if not sector_cond:
            skipped["no_sector_cond"] += 1
            continue

        members = [c for c in universe[rb_key] if c in close_df.columns]
        comp = calc_composite_return_row(close_df[members], snap_pos)
        rs = (comp.rank(pct=True) * 100).clip(0, 100)  # 当月Universe C内パーセンタイル
        ma_dev = calc_ma_dev_row(close_df[members], snap_pos)

        topix_now, topix_fwd = topix.iloc[rb_pos], topix.iloc[rb_pos + FWD_DAYS]
        topix_fwd_ret = topix_fwd / topix_now - 1.0
        is_bull = bool(regime_bull.iloc[rb_pos]) if not pd.isna(regime_sma.iloc[rb_pos]) else None

        for code in members:
            sc = sector_map.get(code)
            if sc is None or sc not in sector_cond:
                continue
            p_now = close_df[code].iloc[rb_pos]
            p_fwd = close_df[code].iloc[rb_pos + FWD_DAYS]
            if pd.isna(p_now) or pd.isna(p_fwd) or p_now <= 0:
                continue
            rs_v = rs.get(code, np.nan)
            if pd.isna(rs_v):
                continue
            excess_fwd = (p_fwd / p_now - 1.0) - topix_fwd_ret
            sec = sector_cond[sc]
            rows.append({
                "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                "year": rb_date.year,
                "code": code,
                "sector17_code": sc,
                "sector_excess_trailing": sec["excess_trailing"],
                "sector_cond": bool(sec["cond"]),
                "sector_top3": bool(sec["top3"]),
                "rs": float(rs_v),
                "ma25_dev": float(ma_dev.get(code, np.nan)),
                "excess_fwd": float(excess_fwd),
                "hit": bool(excess_fwd > 0),
                "regime_bull": is_bull,
            })

    meta = {
        "n_codes_requested": len(all_codes),
        "n_codes_missing": len(missing),
        "skipped": skipped,
    }
    return pd.DataFrame(rows), meta


# ======================================================================
# 4. 統計量
# ======================================================================
def _two_prop_z(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float((p1 - p2) / se) if se > 0 else float("nan")


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def _fm_t(df: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
    """月次クラスタt: 各月の（条件群平均 - 全量平均）系列に対するt（Fama-MacBeth型）。"""
    sub = df[mask]
    if sub.empty:
        return float("nan"), 0
    m_cond = sub.groupby("rebalance_date")["excess_fwd"].mean()
    m_all = df.groupby("rebalance_date")["excess_fwd"].mean()
    diff = (m_cond - m_all.reindex(m_cond.index)).dropna()
    if len(diff) < 3:
        return float("nan"), len(diff)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return (float(diff.mean() / se) if se > 0 else float("nan")), int(len(diff))


def group_stats(df: pd.DataFrame, mask: pd.Series) -> dict | None:
    """条件群 vs 全量ベースライン（Study98と同一比較構造 + FM t追加）。"""
    sub, rest = df[mask], df[~mask]
    n = int(len(sub))
    if n == 0:
        return None
    base_rate = float(df["hit"].mean())
    hit_ratio = float(sub["hit"].mean())
    z = _two_prop_z(int(sub["hit"].sum()), n, int(df["hit"].sum()), int(len(df)))
    mean_excess = float(sub["excess_fwd"].mean())
    t_pooled = _welch_t(sub["excess_fwd"].to_numpy(), rest["excess_fwd"].to_numpy()) if len(rest) else float("nan")
    t_fm, n_months = _fm_t(df, mask)
    return {
        "n": n, "n_months": n_months,
        "hit_ratio": hit_ratio, "base_rate": base_rate, "z": z,
        "mean_excess": mean_excess,
        "mean_excess_all": float(df["excess_fwd"].mean()),
        "t_pooled": t_pooled, "t_fm": t_fm,
    }


# ======================================================================
# 5. 分析
# ======================================================================
def analyze(panel: pd.DataFrame) -> dict:
    results: dict = {}

    def run_slices(df: pd.DataFrame) -> dict:
        out: dict = {}
        # A: sector条件
        out["A_sector_gt_topix"] = group_stats(df, df["sector_cond"])
        out["A_sector_top3"] = group_stats(df, df["sector_top3"])
        # B: A × RS閾値（+ RS単独の帰属分解）
        for th in RS_THRESHOLDS:
            rs_m = df["rs"] >= th
            out[f"B_rs{th}_only"] = group_stats(df, rs_m)
            out[f"B_sector_x_rs{th}"] = group_stats(df, df["sector_cond"] & rs_m)
            out[f"B_top3_x_rs{th}"] = group_stats(df, df["sector_top3"] & rs_m)
        # C: 25MA乖離bin（単独）
        out["C_ma_below0"] = group_stats(df, df["ma25_dev"] < 0)
        for lo, hi in MA_DEV_BINS:
            label = f"C_ma_{lo:g}_{'inf' if np.isinf(hi) else f'{hi:g}'}"
            out[label] = group_stats(df, (df["ma25_dev"] >= lo) & (df["ma25_dev"] < hi))
        # D: combined interaction（sector_cond × RS75 × MA bin / top3 × RS75）
        d_base = df["sector_cond"] & (df["rs"] >= 75)
        out["D_sector_x_rs75_ma_below0"] = group_stats(df, d_base & (df["ma25_dev"] < 0))
        for lo, hi in MA_DEV_BINS:
            label = f"D_sector_x_rs75_ma_{lo:g}_{'inf' if np.isinf(hi) else f'{hi:g}'}"
            out[label] = group_stats(df, d_base & (df["ma25_dev"] >= lo) & (df["ma25_dev"] < hi))
        out["D_top3_x_rs75_ma_0_10"] = group_stats(
            df, df["sector_top3"] & (df["rs"] >= 75) & (df["ma25_dev"] >= 0) & (df["ma25_dev"] < 10))
        return out

    results["full"] = run_slices(panel)

    results["subperiods"] = {}
    for label, (y0, y1) in SUBPERIODS.items():
        sub = panel[(panel["year"] >= y0) & (panel["year"] <= y1)]
        if not sub.empty:
            results["subperiods"][label] = run_slices(sub)

    results["regime"] = {}
    for label, flag in [("bull", True), ("bear", False)]:
        sub = panel[panel["regime_bull"] == flag]
        if not sub.empty:
            results["regime"][label] = run_slices(sub)

    return results


# ======================================================================
# 6. 出力
# ======================================================================
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def _fmt_pct(v, digits=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{digits}f}%"


def _fmt(v, digits=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _row(label: str, s: dict | None) -> str:
    if s is None:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    return (f"| {label} | {s['n']} | {_fmt_pct(s['hit_ratio'])} | {_fmt_pct(s['base_rate'])} "
            f"| {_fmt(s['z'])} | {_fmt_pct(s['mean_excess'])} | {_fmt(s['t_pooled'])} | {_fmt(s['t_fm'])} |")


_HDR = ("| 区分 | N | hit ratio | base rate | z | 平均excess 3M | t(pooled) | t(FM月次) |\n"
        "|---|---|---|---|---|---|---|---|")


def write_outputs(results: dict, panel: pd.DataFrame, meta: dict) -> None:
    payload = {
        "study": "study99_sector_fujiko_filter",
        "title": "Study99 — Sector × Fujiko filter 存在確認（factor-level）",
        "generated_at": datetime.now(_JST).isoformat(),
        "data_source": {
            "universe": str(UNIVERSE_FILE),
            "stock_prices": str(JQUANTS_PROCESSED_DIR),
            "sector_indices": "database/market/index/prices（公式TOPIX-17・Study98と同一）",
            "sector_map": str(COMPANIES_PARQUET) + "（スナップショット・非PIT注記あり）",
        },
        "prohibited_confirmed": {"bt_engine_used": False, "strategy_backtest": False},
        "params": {
            "fwd_days": FWD_DAYS, "sector_trailing_days": SECTOR_TRAILING_DAYS,
            "rs_thresholds": RS_THRESHOLDS, "ma_dev_period": MA_DEV_PERIOD,
            "top_n": TOP_N, "regime_ma": REGIME_MA_PERIOD,
            "rs_definition": "IBD式12M加重複合リターンの当月Universe C内パーセンタイル（canon calc_cross_sectional_rsr同一・pool≈1100銘柄）",
        },
        "panel_meta": {
            "rows": int(len(panel)),
            "months": int(panel["rebalance_date"].nunique()),
            "period": [panel["rebalance_date"].min(), panel["rebalance_date"].max()],
            **meta,
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default),
                        encoding="utf-8")

    md: list[str] = []
    md.append("# Study99 — Sector × Fujiko filter 存在確認（factor-level・2026-07-16）\n")
    md.append("目的: P(Stock(t+3M)>TOPIX | Sector(t)>TOPIX) + RS score / 25MA乖離filterの交互作用。戦略BT不使用。\n")
    md.append(f"パネル: 行数={len(panel):,}・月数={panel['rebalance_date'].nunique()}・"
              f"期間={panel['rebalance_date'].min()}〜{panel['rebalance_date'].max()}・"
              f"Universe C（PIT）・forward=3M（63営業日）・セクター条件=公式TOPIX-17指数 3M trailing excess\n")
    md.append("統計注記: t(FM月次)=月次クラスタt（クロスセクション相関補正・主指標）。"
              "t(pooled)は銘柄間相関でN水増しのため参考値。3M forward窓は月次サンプリングで重複"
              "（自己相関により両tともやや過大）。RS=Universe C内パーセンタイル（プールサイズ依存・仮説4注記）。\n")

    full = results["full"]

    md.append("## Analysis A — セクター条件単独\n")
    md.append(_HDR)
    md.append(_row("A: Sector>TOPIX（全量）", full["A_sector_gt_topix"]))
    md.append(_row("A: Sector top3", full["A_sector_top3"]))
    md.append("")

    md.append("## Analysis B — RS score filter（60/70/75/80）\n")
    md.append(_HDR)
    for th in RS_THRESHOLDS:
        md.append(_row(f"RS≥{th} 単独", full[f"B_rs{th}_only"]))
    for th in RS_THRESHOLDS:
        md.append(_row(f"Sector>TOPIX ∧ RS≥{th}", full[f"B_sector_x_rs{th}"]))
    for th in RS_THRESHOLDS:
        md.append(_row(f"Sector top3 ∧ RS≥{th}", full[f"B_top3_x_rs{th}"]))
    md.append("")

    md.append("## Analysis C — 25MA乖離率bin（単独）\n")
    md.append(_HDR)
    md.append(_row("乖離<0%（参考）", full["C_ma_below0"]))
    for lo, hi in MA_DEV_BINS:
        label = f"C_ma_{lo:g}_{'inf' if np.isinf(hi) else f'{hi:g}'}"
        md.append(_row(f"乖離{lo:g}〜{'∞' if np.isinf(hi) else f'{hi:g}'}%", full[label]))
    md.append("")

    md.append("## Analysis D — Combined interaction（Sector>TOPIX ∧ RS≥75 × 25MA乖離bin）\n")
    md.append(_HDR)
    md.append(_row("D: ∧乖離<0%（参考）", full["D_sector_x_rs75_ma_below0"]))
    for lo, hi in MA_DEV_BINS:
        label = f"D_sector_x_rs75_ma_{lo:g}_{'inf' if np.isinf(hi) else f'{hi:g}'}"
        md.append(_row(f"D: ∧乖離{lo:g}〜{'∞' if np.isinf(hi) else f'{hi:g}'}%", full[label]))
    md.append(_row("D: top3 ∧ RS≥75 ∧ 乖離0〜10%", full["D_top3_x_rs75_ma_0_10"]))
    md.append("")

    md.append("## サブ期間（主要区分のみ）\n")
    md.append("| 期間 | 区分 | N | hit ratio | base rate | z | 平均excess 3M | t(FM月次) |")
    md.append("|---|---|---|---|---|---|---|---|")
    key_slices = [("A: Sector>TOPIX", "A_sector_gt_topix"), ("A: top3", "A_sector_top3"),
                  ("B: Sector∧RS75", "B_sector_x_rs75"), ("B: top3∧RS75", "B_top3_x_rs75"),
                  ("RS75単独", "B_rs75_only")]
    for period, res in results["subperiods"].items():
        for name, key in key_slices:
            s = res.get(key)
            if s is None:
                md.append(f"| {period} | {name} | 0 | - | - | - | - | - |")
            else:
                md.append(f"| {period} | {name} | {s['n']} | {_fmt_pct(s['hit_ratio'])} | "
                          f"{_fmt_pct(s['base_rate'])} | {_fmt(s['z'])} | "
                          f"{_fmt_pct(s['mean_excess'])} | {_fmt(s['t_fm'])} |")
    md.append("")

    md.append("## Regime別（TOPIX>MA200・主要区分のみ）\n")
    md.append("| Regime | 区分 | N | hit ratio | base rate | z | 平均excess 3M | t(FM月次) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for regime, res in results["regime"].items():
        for name, key in key_slices:
            s = res.get(key)
            if s is None:
                md.append(f"| {regime} | {name} | 0 | - | - | - | - | - |")
            else:
                md.append(f"| {regime} | {name} | {s['n']} | {_fmt_pct(s['hit_ratio'])} | "
                          f"{_fmt_pct(s['base_rate'])} | {_fmt(s['z'])} | "
                          f"{_fmt_pct(s['mean_excess'])} | {_fmt(s['t_fm'])} |")
    md.append("")

    md.append("![interaction chart](study99_interaction_chart.png)\n")
    md.append("---\n")
    md.append(f"*生成: Study99自動分析パイプライン, {datetime.now(_JST).strftime('%Y-%m-%d')}。"
              "戦略BT不使用（factor-level統計のみ）。セクター分類=companies.parquetスナップショット"
              "（非PIT・Study95と同一注記）。RS閾値はProduction採用パラメータではない（統治原則3）。*")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")


def write_chart(results: dict) -> None:
    full = results["full"]
    labels_keys = [
        ("全量base", None),
        ("Sector>TPX", "A_sector_gt_topix"), ("top3", "A_sector_top3"),
        ("RS75単独", "B_rs75_only"),
        ("Sector∧RS75", "B_sector_x_rs75"), ("top3∧RS75", "B_top3_x_rs75"),
        ("D:∧乖離0-5", "D_sector_x_rs75_ma_0_5"), ("D:∧乖離5-10", "D_sector_x_rs75_ma_5_10"),
        ("D:∧乖離10-15", "D_sector_x_rs75_ma_10_15"), ("D:∧乖離15-20", "D_sector_x_rs75_ma_15_20"),
        ("D:∧乖離20+", "D_sector_x_rs75_ma_20_inf"),
    ]
    base = full["A_sector_gt_topix"]["base_rate"]
    names, vals = [], []
    for name, key in labels_keys:
        if key is None:
            names.append(name); vals.append(base * 100)
            continue
        s = full.get(key)
        if s is None:
            continue
        names.append(f"{name}\n(N={s['n']})"); vals.append(s["hit_ratio"] * 100)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#888888"] + ["#4878b0"] * (len(names) - 1)
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(base * 100, color="#c44e52", linestyle="--", linewidth=1, label=f"無条件base rate {base*100:.1f}%")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("hit ratio（Stock 3M > TOPIX）%")
    ax.set_title("Study99 — Sector × RS × 25MA乖離 交互作用（3M forward hit ratio）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CHART, dpi=110)
    plt.close(fig)


# ======================================================================
# 7. メイン
# ======================================================================
def main() -> int:
    print("Study99 — Sector × Fujiko filter 存在確認（factor-level）")
    panel, meta = build_panel()
    if panel.empty:
        print("FATAL: パネルが空です")
        return 1
    print(f"panel rows={len(panel):,} months={panel['rebalance_date'].nunique()} "
          f"period={panel['rebalance_date'].min()}〜{panel['rebalance_date'].max()} "
          f"missing_codes={meta['n_codes_missing']} skipped={meta['skipped']}")
    results = analyze(panel)
    write_outputs(results, panel, meta)
    write_chart(results)
    a = results["full"]["A_sector_gt_topix"]
    b = results["full"]["B_top3_x_rs75"]
    print(f"A(Sector>TPX): N={a['n']} hit={a['hit_ratio']:.1%} base={a['base_rate']:.1%} "
          f"z={a['z']:.2f} t_fm={a['t_fm']:.2f}")
    if b:
        print(f"B(top3∧RS75): N={b['n']} hit={b['hit_ratio']:.1%} z={b['z']:.2f} "
              f"excess={b['mean_excess']:.2%} t_fm={b['t_fm']:.2f}")
    print(f"出力: {OUT_JSON}")
    print(f"出力: {OUT_MD}")
    print(f"出力: {OUT_CHART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
