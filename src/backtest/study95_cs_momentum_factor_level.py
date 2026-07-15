"""
src/backtest/study95_cs_momentum_factor_level.py
Study95 — Cross-Sectional Momentum Factor-Level Ground Truth（H0・最優先）

正典: reports/fujiko_r2_research_roadmap.md v2 Part5 Study95仕様 / タスク指示
      「Study95: 目的=日本株におけるCross-sectional momentumの存在をfactor-levelで検証する」

目的:
  仮説1（Cross-sectional momentumは日本で依然有効か）をエンジン非経由・factor-levelで検証する。
  Study75A Universe C（PIT・月次規則ユニバース）上で (1) 12-1モメンタム (2) Clenow slope×R²
  の2ファクターについてdecileポートフォリオ分析・IC・Q10-Q1スプレッド（t検定+bootstrap CI）・
  regime分解・sector分解・容量診断を行う。

禁止（タスク仕様・厳守）:
  - フジコ法ロジック使用禁止 → composite_alpha_bt / fujiko_strategy は一切importしない
  - RSR使用禁止 → RSRパーセンタイルランキングは使わない（本ファイルにRSR計算コードなし）
  - percentile型パラメータ使用禁止 → トレーディングルールとしてのパーセンタイル閾値
    （min_rsr等）は導入しない。decile分解自体は本タスクが明示的に要求する記述統計手法であり、
    銘柄選定ルールではない（統治原則3が禁じる「Production採用パラメータとしての新規percentile」
    には該当しない）
  - BTエンジン使用禁止 → composite_alpha_bt.run_scenario 等は一切呼ばない。
    本スクリプトは全篇、価格パネルに対する純粋なベクトル化クロスセクション統計のみで構成される

データ（Study75A系譜・fresh run）:
  - backtests/study75_rule_universe.json          Universe C月次PITメンバーシップ（Study75A）
  - data/jquants/processed/{code}.parquet         銘柄別価格（Close=分割調整済み・canon Study76と同一ソース）
  - data/jquants/processed/TOPIX.parquet          レジーム判定（TOPIX>MA200・既存canon定義）
  - database/market/master/companies.parquet      Sector17/33（現在時点スナップショット。
                                                   下記PIT注記参照）
  - backtests/study75_universe_diagnostics.parquet ADV20（Study75A既存成果物の再利用・容量診断）

PIT注記（sector）:
  companies.parquet は現在時点（2026-07-13）のセクター分類の単一スナップショットであり、
  時系列テーブルではない。業種再分類は稀（本分析はセクターの粗い持続性検証が目的であり、
  数年単位の再分類イベントが結果を覆す可能性は低いと判断）。将来Study88でTOPIX17/33の
  時系列分類が必要になった場合は別途PIT化を検討する。

PIT設計（価格・factor）:
  各月のrebalance_date（Universe Cキー）についてsnapshot_date=直前営業日を用いてfactorを算出する
  （Study75A/canon Study76と同一パターン。snapshot_date以前のデータのみ使用・lookahead無し）。
  forward returnはrebalance_date終値を起点に計測する。

出力:
  backtests/study95_cs_momentum_factor_level.json
  reports/study95_cs_momentum_factor_level.md
  reports/study95_decile_chart.png
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy import stats

from src.paths import DATABASE_MASTER_DIR, JQUANTS_PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR

rcParams["font.family"] = "MS Gothic"  # ENV_WIN準拠（日本語ラベル文字化け防止）

_JST = timezone(timedelta(hours=9))

# ── 固定パラメータ（<10個級・事前固定・スイープ禁止） ─────────────────────────
MOM_SKIP_DAYS = 21          # skip-1-month（直近1ヶ月の短期リバーサルを除外・Jegadeesh-Titman標準）
MOM_LOOKBACK_DAYS = 252     # 12-1モメンタム: P[t-21]/P[t-252]-1
CLENOW_LOOKBACK_DAYS = 90   # canon Study76（study76_clenow_benchmark_wf.py）と同一定義を再利用
CLENOW_SCALE = 10_000.0     # 同上
REGIME_MA_PERIOD = 200      # TOPIX>MA200・既存canon Market Regime定義（Study76と同一）
N_DECILES = 10
HORIZONS: dict[str, int] = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}  # 営業日換算
HORIZON_MONTHS: dict[str, int] = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
FFILL_LIMIT = 2  # 個別銘柄の単発売買停止のみ許容（長期欠損は前方補完しない=生存者バイアス回避）

UNIVERSE_FILE = RESULTS_DIR / "study75_rule_universe.json"
TOPIX_PARQUET = JQUANTS_PROCESSED_DIR / "TOPIX.parquet"
COMPANIES_PARQUET = DATABASE_MASTER_DIR / "companies.parquet"
DIAGNOSTICS_PARQUET = RESULTS_DIR / "study75_universe_diagnostics.parquet"

OUT_JSON = RESULTS_DIR / "study95_cs_momentum_factor_level.json"
OUT_MD = REPORTS_DIR / "study95_cs_momentum_factor_level.md"
OUT_CHART = REPORTS_DIR / "study95_decile_chart.png"


# ======================================================================
# 1. データ読込
# ======================================================================
def load_universe() -> dict[str, list[str]]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe Cファイルが存在しません: {UNIVERSE_FILE}（Study75A未完了）")
    data = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    monthly = data.get("monthly_universe")
    if not isinstance(monthly, dict) or not monthly:
        raise ValueError(f"monthly_universe が空/不正です: {UNIVERSE_FILE}")
    return {k: v for k, v in monthly.items() if v}  # 空月（初期化前）は除外


def load_topix_calendar() -> pd.Series:
    if not TOPIX_PARQUET.exists():
        raise FileNotFoundError(f"TOPIX価格データが存在しません: {TOPIX_PARQUET}")
    df = pd.read_parquet(TOPIX_PARQUET)
    close = df["Close"].dropna().sort_index()
    return close


def load_close_panel(codes: list[str], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """
    data/jquants/processed/{code}.parquet からClose列のみ読み込み、共通カレンダーへreindexする。
    Close列は分割調整済み（canon Study76と同一ソース・§冒頭確認済み）。
    FFILL_LIMIT=2営業日までの単発欠損のみ補完（長期欠損=上場廃止/長期停止は補完しない）。
    Returns: (close_df [calendar x codes], missing_codes)
    """
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
    close_df = pd.DataFrame(series_map)
    close_df = close_df.reindex(calendar)
    close_df = close_df.ffill(limit=FFILL_LIMIT)
    return close_df, missing


def load_sector_map() -> pd.DataFrame:
    comp = pd.read_parquet(COMPANIES_PARQUET, columns=["Code", "Sector17CodeName", "Sector33CodeName"])
    comp = comp.drop_duplicates(subset="Code").set_index("Code")
    return comp


def load_adv20_map() -> pd.DataFrame:
    """Study75A既存成果物（backtests/study75_universe_diagnostics.parquet）のADV20を再利用する。"""
    diag = pd.read_parquet(DIAGNOSTICS_PARQUET, columns=["date", "code", "adv20"])
    return diag


# ======================================================================
# 2. Factor計算（ベクトル化・cross-section一括）
# ======================================================================
def calc_momentum_row(close_df: pd.DataFrame, snap_pos: int) -> pd.Series:
    """12-1モメンタム: P[snap-21]/P[snap-252]-1（両端ともsnapshot_date以前=lookahead無し）。"""
    if snap_pos - MOM_LOOKBACK_DAYS < 0:
        return pd.Series(np.nan, index=close_df.columns)
    p_recent = close_df.iloc[snap_pos - MOM_SKIP_DAYS]
    p_distant = close_df.iloc[snap_pos - MOM_LOOKBACK_DAYS]
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = (p_recent / p_distant) - 1.0
    return mom.replace([np.inf, -np.inf], np.nan)


def calc_clenow_row(close_df: pd.DataFrame, snap_pos: int) -> pd.Series:
    """
    slope90d × R²（canon Study76 calc_clenow_score と同一定義。ベクトル化版）。
    window = [snap_pos-89, snap_pos]（90日・snapshot_date以前=lookahead無し）。
    """
    if snap_pos - (CLENOW_LOOKBACK_DAYS - 1) < 0:
        return pd.Series(np.nan, index=close_df.columns)
    window = close_df.iloc[snap_pos - CLENOW_LOOKBACK_DAYS + 1: snap_pos + 1]
    if len(window) != CLENOW_LOOKBACK_DAYS:
        return pd.Series(np.nan, index=close_df.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(window.to_numpy(dtype=float))  # (90, n_codes)
    valid_cols = ~np.isnan(y).any(axis=0)
    x = np.arange(CLENOW_LOOKBACK_DAYS, dtype=float)
    x_mean = x.mean()
    sxx = float(np.sum((x - x_mean) ** 2))

    slope = np.full(y.shape[1], np.nan)
    r2 = np.full(y.shape[1], np.nan)
    if valid_cols.any():
        yv = y[:, valid_cols]
        y_mean = yv.mean(axis=0)
        sxy = np.sum((x - x_mean)[:, None] * (yv - y_mean[None, :]), axis=0)
        sl = sxy / sxx
        intercept = y_mean - sl * x_mean
        y_pred = sl[None, :] * x[:, None] + intercept[None, :]
        ss_res = np.sum((yv - y_pred) ** 2, axis=0)
        ss_tot = np.sum((yv - y_mean[None, :]) ** 2, axis=0)
        r2v = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
        r2v = np.clip(r2v, 0.0, 1.0)
        slope[valid_cols] = sl
        r2[valid_cols] = r2v
    score = slope * r2 * CLENOW_SCALE
    return pd.Series(score, index=close_df.columns)


def calc_forward_returns_row(close_df: pd.DataFrame, base_pos: int) -> dict[str, pd.Series]:
    """rebalance_date終値を起点にHORIZONS分の前方リターンを一括計算する。"""
    out: dict[str, pd.Series] = {}
    n = len(close_df)
    p0 = close_df.iloc[base_pos]
    for label, h in HORIZONS.items():
        pos_h = base_pos + h
        if pos_h >= n:
            out[label] = pd.Series(np.nan, index=close_df.columns)
            continue
        p1 = close_df.iloc[pos_h]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (p1 / p0) - 1.0
        out[label] = ret.replace([np.inf, -np.inf], np.nan)
    return out


# ======================================================================
# 3. Decile割当・統計ユーティリティ
# ======================================================================
def assign_deciles(factor: pd.Series) -> pd.Series:
    """有効値のみを対象にqcut(10分位)。タイが多い場合はrank(method='first')で分散させる。"""
    valid = factor.dropna()
    if len(valid) < N_DECILES * 3:  # 各decileに最低3銘柄相当が無ければ全体を無効化
        return pd.Series(np.nan, index=factor.index)
    ranks = valid.rank(method="first")
    deciles = pd.qcut(ranks, N_DECILES, labels=False) + 1  # 1..10
    out = pd.Series(np.nan, index=factor.index)
    out.loc[valid.index] = deciles
    return out


def newey_west_tstat(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Newey-West HAC標準誤差によるmean(x)のt統計量。overlapping観測（月次formation×複数月horizon）向け。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mean = x.mean()
    resid = x - mean
    lag = max(0, min(lag, n - 1))
    gamma0 = float(np.sum(resid ** 2)) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)  # Bartlett kernel
        gamma_k = float(np.sum(resid[k:] * resid[:-k])) / n
        var += 2 * w * gamma_k
    se = np.sqrt(max(var, 0.0) / n)
    t = mean / se if se > 0 else float("nan")
    return float(mean), float(t)


def block_bootstrap_ci(x: np.ndarray, block_len: int, n_boot: int = N_BOOTSTRAP,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """moving block bootstrapによる平均値の95%CI（overlapping観測の自己相関を保持）。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < max(5, block_len + 1):
        return float("nan"), float("nan")
    block_len = max(1, min(block_len, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts_pool = np.arange(0, n - block_len + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([x[s: s + block_len] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def annualize(mean_ret: float, months: int) -> float:
    if np.isnan(mean_ret):
        return float("nan")
    return float((1.0 + mean_ret) ** (12.0 / months) - 1.0)


# ======================================================================
# 4. メイン分析パイプライン
# ======================================================================
def main() -> int:
    print("[1/7] Universe C読込...")
    monthly_universe = load_universe()
    rebalance_dates = sorted(pd.Timestamp(k) for k in monthly_universe)
    print(f"  months={len(rebalance_dates)}  first={rebalance_dates[0].date()}  last={rebalance_dates[-1].date()}")

    print("[2/7] TOPIXカレンダー・レジーム読込...")
    topix = load_topix_calendar()
    calendar = topix.index
    topix_sma200 = topix.rolling(REGIME_MA_PERIOD, min_periods=REGIME_MA_PERIOD).mean()
    regime_bull = topix >= topix_sma200  # NaN(初期200日)はFalse扱い→後段でdropna

    print("[3/7] 価格パネル読込（Universe C全期間・銘柄数=%d）..." % len(
        {c for v in monthly_universe.values() for c in v}
    ))
    all_codes = sorted({c for v in monthly_universe.values() for c in v})
    close_df, missing_codes = load_close_panel(all_codes, calendar)
    print(f"  loaded={close_df.shape[1]}  missing={len(missing_codes)}")

    print("[4/7] Sector/ADV20読込...")
    sector_map = load_sector_map()
    adv20_df = load_adv20_map()
    adv20_df["date"] = adv20_df["date"].astype(str)

    print("[5/7] 月次factor・forward return計算（PIT: snapshot=T-1営業日）...")
    records: list[dict] = []
    calendar_pos = {d: i for i, d in enumerate(calendar)}

    for rb_date in rebalance_dates:
        rb_str = rb_date.strftime("%Y-%m-%d")
        universe_codes = monthly_universe[rb_str]
        if rb_date not in calendar_pos:
            # rebalance_dateがTOPIXカレンダーに無い場合は直後の営業日で代替
            later = calendar[calendar >= rb_date]
            if len(later) == 0:
                continue
            rb_date_eff = later[0]
        else:
            rb_date_eff = rb_date
        base_pos = calendar_pos[rb_date_eff]
        snap_pos = base_pos - 1  # snapshot_date = T-1営業日（Study75Aと同一PIT規約）
        if snap_pos < 0:
            continue

        universe_mask = pd.Series(False, index=close_df.columns)
        universe_mask.loc[[c for c in universe_codes if c in close_df.columns]] = True

        mom = calc_momentum_row(close_df, snap_pos).where(universe_mask)
        clenow = calc_clenow_row(close_df, snap_pos).where(universe_mask)
        fwd = calc_forward_returns_row(close_df, base_pos)

        mom_decile = assign_deciles(mom)
        clenow_decile = assign_deciles(clenow)

        snap_date = calendar[snap_pos]
        is_bull = bool(regime_bull.get(snap_date, False)) if snap_date in regime_bull.index else False
        regime_defined = snap_date in topix_sma200.dropna().index

        adv_lookup = adv20_df.loc[adv20_df["date"] == rb_str].set_index("code")["adv20"] \
            if (adv20_df["date"] == rb_str).any() else pd.Series(dtype=float)

        for code in universe_codes:
            if code not in close_df.columns:
                continue
            row = {
                "rebalance_date": rb_str,
                "code": code,
                "mom_value": float(mom.get(code, np.nan)),
                "mom_decile": mom_decile.get(code, np.nan),
                "clenow_value": float(clenow.get(code, np.nan)),
                "clenow_decile": clenow_decile.get(code, np.nan),
                "regime_bull": is_bull if regime_defined else None,
                "sector17": sector_map["Sector17CodeName"].get(code, "UNKNOWN"),
                "sector33": sector_map["Sector33CodeName"].get(code, "UNKNOWN"),
                "adv20": float(adv_lookup.get(code, np.nan)),
            }
            for label in HORIZONS:
                row[f"fwd_{label}"] = float(fwd[label].get(code, np.nan))
            records.append(row)

    panel = pd.DataFrame(records)
    print(f"  panel rows={len(panel):,}")

    print("[6/7] 集計（decile table / IC / Q10-Q1 spread / monotonicity / regime / sector / capacity / persistence）...")
    results = aggregate(panel, rebalance_dates)

    print("[7/7] 出力生成...")
    write_outputs(results, panel, missing_codes, len(rebalance_dates))
    print(f"  JSON: {OUT_JSON}")
    print(f"  Markdown: {OUT_MD}")
    print(f"  Chart: {OUT_CHART}")
    return 0


# ======================================================================
# 5. 集計ロジック
# ======================================================================
def _decile_table(panel: pd.DataFrame, factor_prefix: str) -> dict:
    decile_col = f"{factor_prefix}_decile"
    table = {}
    for label in HORIZONS:
        g = panel.dropna(subset=[decile_col, f"fwd_{label}"]).groupby(decile_col)[f"fwd_{label}"]
        means = g.mean()
        counts = g.count()
        table[label] = {
            int(d): {"mean_fwd_return": float(means.get(d, np.nan)), "n": int(counts.get(d, 0))}
            for d in range(1, N_DECILES + 1)
        }
    return table


def _monotonicity(decile_table: dict) -> dict:
    out = {}
    for label, deciles in decile_table.items():
        xs = list(range(1, N_DECILES + 1))
        ys = [deciles[d]["mean_fwd_return"] for d in xs]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y)]
        if len(valid) < 3:
            out[label] = {"spearman_rho": None, "spearman_p": None, "adjacent_violations": None}
            continue
        xv, yv = zip(*valid)
        rho, p = stats.spearmanr(xv, yv)
        violations = sum(1 for i in range(len(yv) - 1) if yv[i + 1] < yv[i])
        out[label] = {
            "spearman_rho": float(rho), "spearman_p": float(p),
            "adjacent_violations": int(violations), "adjacent_pairs": len(yv) - 1,
        }
    return out


def _ic(panel: pd.DataFrame, factor_col: str) -> dict:
    out = {}
    for label in HORIZONS:
        sub = panel.dropna(subset=[factor_col, f"fwd_{label}"])
        ics = []
        for rb_date, g in sub.groupby("rebalance_date"):
            if len(g) < 10:
                continue
            rho, _ = stats.spearmanr(g[factor_col], g[f"fwd_{label}"])
            if not np.isnan(rho):
                ics.append(rho)
        ics_arr = np.array(ics)
        if len(ics_arr) < 3:
            out[label] = {"mean_ic": None, "std_ic": None, "t_stat": None, "hit_rate": None, "n_periods": len(ics_arr)}
            continue
        mean_ic = float(ics_arr.mean())
        std_ic = float(ics_arr.std(ddof=1))
        t_stat = float(mean_ic / (std_ic / np.sqrt(len(ics_arr)))) if std_ic > 0 else float("nan")
        hit_rate = float((ics_arr > 0).mean())
        out[label] = {
            "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat,
            "hit_rate": hit_rate, "n_periods": int(len(ics_arr)),
        }
    return out


def _q10_q1_spread(panel: pd.DataFrame, decile_col: str, regime_filter: str | None = None) -> dict:
    sub = panel
    if regime_filter == "bull":
        sub = panel[panel["regime_bull"] == True]  # noqa: E712
    elif regime_filter == "bear":
        sub = panel[panel["regime_bull"] == False]  # noqa: E712

    out = {}
    for label, h in HORIZONS.items():
        months = HORIZON_MONTHS[label]
        spreads = []
        for rb_date, g in sub.groupby("rebalance_date"):
            g = g.dropna(subset=[decile_col, f"fwd_{label}"])
            q10 = g.loc[g[decile_col] == N_DECILES, f"fwd_{label}"]
            q1 = g.loc[g[decile_col] == 1, f"fwd_{label}"]
            if len(q10) == 0 or len(q1) == 0:
                continue
            spreads.append(float(q10.mean() - q1.mean()))
        spreads_arr = np.array(spreads)
        if len(spreads_arr) < 5:
            out[label] = {
                "n_periods": int(len(spreads_arr)), "mean_spread": None, "annualized_spread": None,
                "newey_west_t": None, "bootstrap_ci_95": [None, None],
            }
            continue
        mean_spread, t_stat = newey_west_tstat(spreads_arr, lag=max(0, months - 1))
        ann = annualize(mean_spread, months)
        block_len = max(1, months)
        ci_lo, ci_hi = block_bootstrap_ci(spreads_arr, block_len=block_len)
        ci_lo_ann = annualize(ci_lo, months) if not np.isnan(ci_lo) else None
        ci_hi_ann = annualize(ci_hi, months) if not np.isnan(ci_hi) else None
        out[label] = {
            "n_periods": int(len(spreads_arr)),
            "mean_spread": mean_spread,
            "annualized_spread": ann,
            "newey_west_t": t_stat,
            "bootstrap_ci_95": [ci_lo_ann, ci_hi_ann],
        }
    return out


def _sector_ic(panel: pd.DataFrame, factor_col: str, sector_col: str, horizon: str = "1M") -> dict:
    out = {}
    for sector, g in panel.dropna(subset=[factor_col, f"fwd_{horizon}"]).groupby(sector_col):
        if len(g) < 30:
            out[sector] = {"ic": None, "n": int(len(g)), "note": "insufficient_n"}
            continue
        rho, p = stats.spearmanr(g[factor_col], g[f"fwd_{horizon}"])
        out[sector] = {"ic": float(rho), "p_value": float(p), "n": int(len(g))}
    return out


def _sector_neutral_spread(panel: pd.DataFrame, factor_value_col: str) -> dict:
    """TOPIX17内でfactorをdemeanしてからdecile再割当 → Q10-Q1スプレッド（sector bet除去後）。"""
    df = panel.copy()
    df["_demeaned"] = df.groupby(["rebalance_date", "sector17"])[factor_value_col].transform(
        lambda s: s - s.mean() if len(s) >= 3 else np.nan
    )
    demeaned_decile = "_demeaned_decile"
    parts = []
    for rb_date, g in df.groupby("rebalance_date"):
        d = assign_deciles(g.set_index("code")["_demeaned"])
        parts.append(pd.DataFrame({"code": d.index, "rebalance_date": rb_date, demeaned_decile: d.values}))
    dmap = pd.concat(parts, ignore_index=True)
    df = df.merge(dmap, on=["rebalance_date", "code"], how="left")
    return _q10_q1_spread(df, demeaned_decile)


def _capacity(panel: pd.DataFrame, decile_col: str) -> dict:
    q1 = panel[panel[decile_col] == 1]["adv20"].dropna()
    q10 = panel[panel[decile_col] == N_DECILES]["adv20"].dropna()

    def _pct(s: pd.Series) -> dict:
        if len(s) == 0:
            return {"p10": None, "median": None, "p90": None, "n": 0}
        return {
            "p10": float(np.percentile(s, 10)), "median": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)), "n": int(len(s)),
        }

    # Q10メンバーシップの月次turnover
    q10_sets = {}
    for rb_date, g in panel[panel[decile_col] == N_DECILES].groupby("rebalance_date"):
        q10_sets[rb_date] = set(g["code"])
    dates_sorted = sorted(q10_sets.keys())
    turnovers = []
    for i in range(1, len(dates_sorted)):
        prev, cur = q10_sets[dates_sorted[i - 1]], q10_sets[dates_sorted[i]]
        if len(cur) == 0:
            continue
        changed = len(cur - prev)
        turnovers.append(changed / len(cur))

    return {
        "adv20_jpy": {"q1_bottom_decile": _pct(q1), "q10_top_decile": _pct(q10)},
        "q10_monthly_turnover_mean": float(np.mean(turnovers)) if turnovers else None,
        "q10_monthly_turnover_median": float(np.median(turnovers)) if turnovers else None,
    }


def _persistence(panel: pd.DataFrame, factor_col: str) -> dict:
    """factor値のrank自己相関（1/3/6/12ヶ月ラグ）。"""
    dates_sorted = sorted(panel["rebalance_date"].unique())
    date_idx = {d: i for i, d in enumerate(dates_sorted)}
    wide = panel.pivot_table(index="rebalance_date", columns="code", values=factor_col, aggfunc="first")
    wide = wide.reindex(dates_sorted)

    out = {}
    for lag_months, key in ((1, "1M"), (3, "3M"), (6, "6M"), (12, "12M")):
        corrs = []
        for i in range(len(dates_sorted) - lag_months):
            a = wide.iloc[i]
            b = wide.iloc[i + lag_months]
            both = pd.concat([a, b], axis=1).dropna()
            if len(both) < 30:
                continue
            rho, _ = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])
            if not np.isnan(rho):
                corrs.append(rho)
        out[key] = {
            "mean_rank_autocorr": float(np.mean(corrs)) if corrs else None,
            "n_periods": len(corrs),
        }
    return out


def _verdict(mom_spread: dict, mom_ic: dict, mom_regime: dict) -> dict:
    """ユーザー指定の判定基準を機械適用する。"""
    ann_12m = mom_spread.get("12M", {}).get("annualized_spread")
    t_12m = mom_spread.get("12M", {}).get("newey_west_t")
    signs = []
    consistent_horizons = 0
    for label in HORIZONS:
        ann = mom_spread.get(label, {}).get("annualized_spread")
        if ann is not None:
            signs.append(np.sign(ann))
            if ann > 0:
                consistent_horizons += 1

    success = (
        ann_12m is not None and ann_12m > 0.05
        and t_12m is not None and abs(t_12m) > 2.0
        and consistent_horizons >= 3
    )

    bull_ann = mom_regime.get("bull", {}).get("12M", {}).get("annualized_spread")
    bear_ann = mom_regime.get("bear", {}).get("12M", {}).get("annualized_spread")
    regime_gated = bool(
        (bull_ann is not None and bull_ann > 0.05) and not (bear_ann is not None and bear_ann > 0.02)
    )

    if success:
        verdict = "SUCCESS"
    elif regime_gated:
        verdict = "SUCCESS_REGIME_GATED"
    elif ann_12m is not None and abs(ann_12m) < 0.02:
        verdict = "FAIL_ZERO_SPREAD"
    elif ann_12m is not None and ann_12m < 0:
        verdict = "FAIL_REVERSAL"
    else:
        verdict = "FAIL_UNSTABLE"

    kill = verdict.startswith("FAIL")
    return {
        "verdict": verdict,
        "kill_triggered": kill,
        "recommendation": (
            "Candidate A-E全凍結。旧正典ARCH系（PEAD/TSMOM）への転進をユーザーへ提起。"
            if kill else
            "regime-gated型（Bull限定）でCandidate設計を継続" if verdict == "SUCCESS_REGIME_GATED" else
            "H0合格。Study96・canon Study76へ進行可"
        ),
        "criteria": {
            "success_rule": "12M annualized Q10-Q1 > +5% AND |t|>2 AND >=3/4 horizons positive",
            "fail_rule": "spread within +-2% (zero) OR negative (reversal) OR criteria unmet (unstable)",
            "ann_12m_spread": ann_12m, "t_12m": t_12m, "positive_horizons": consistent_horizons,
        },
    }


def aggregate(panel: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> dict:
    results: dict = {"n_rebalance_dates": len(rebalance_dates), "n_panel_rows": len(panel)}

    for factor_prefix in ("mom", "clenow"):
        decile_col = f"{factor_prefix}_decile"
        value_col = f"{factor_prefix}_value"
        decile_table = _decile_table(panel, factor_prefix)
        spread = _q10_q1_spread(panel, decile_col)
        regime = {
            "bull": _q10_q1_spread(panel, decile_col, regime_filter="bull"),
            "bear": _q10_q1_spread(panel, decile_col, regime_filter="bear"),
        }
        sector17_ic = _sector_ic(panel, value_col, "sector17", horizon="1M")
        sector33_ic = _sector_ic(panel, value_col, "sector33", horizon="1M")
        sector_neutral = _sector_neutral_spread(panel, value_col)

        block = {
            "decile_table": decile_table,
            "monotonicity": _monotonicity(decile_table),
            "information_coefficient": _ic(panel, value_col),
            "q10_q1_spread": spread,
            "regime_decomposition": regime,
            "sector17_ic": sector17_ic,
            "sector33_ic": sector33_ic,
            "sector17_neutral_q10_q1_spread": sector_neutral,
            "capacity": _capacity(panel, decile_col),
            "persistence": _persistence(panel, value_col),
        }
        results[factor_prefix] = block

    results["verdict"] = _verdict(
        results["mom"]["q10_q1_spread"], results["mom"]["information_coefficient"],
        results["mom"]["regime_decomposition"],
    )
    results["verdict_clenow"] = _verdict(
        results["clenow"]["q10_q1_spread"], results["clenow"]["information_coefficient"],
        results["clenow"]["regime_decomposition"],
    )
    return results


# ======================================================================
# 6. 出力生成
# ======================================================================
def write_outputs(results: dict, panel: pd.DataFrame, missing_codes: list[str], n_months: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    full = {
        "study": "Study95",
        "title": "Cross-Sectional Momentum Factor-Level Ground Truth（H0）",
        "generated_at": datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prohibited_confirmed": {
            "fujiko_logic_used": False, "rsr_used": False,
            "percentile_trading_param_used": False, "bt_engine_used": False,
        },
        "data_quality": {"missing_price_files": len(missing_codes), "n_rebalance_months": n_months},
        "params": {
            "mom_skip_days": MOM_SKIP_DAYS, "mom_lookback_days": MOM_LOOKBACK_DAYS,
            "clenow_lookback_days": CLENOW_LOOKBACK_DAYS, "regime_ma_period": REGIME_MA_PERIOD,
            "n_deciles": N_DECILES, "horizons_trading_days": HORIZONS,
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(full, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    _write_chart(results)
    _write_markdown(full, panel)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write_chart(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, factor_prefix, title in (
        (axes[0], "mom", "12-1モメンタム"), (axes[1], "clenow", "Clenow slope×R²"),
    ):
        table = results[factor_prefix]["decile_table"]["1M"]
        xs = list(range(1, N_DECILES + 1))
        ys = [table[d]["mean_fwd_return"] * 100 if table[d]["mean_fwd_return"] is not None else np.nan for d in xs]
        colors = ["#c0392b" if (not np.isnan(y) and y < 0) else "#2980b9" for y in ys]
        ax.bar(xs, ys, color=colors)
        ax.set_title(f"{title}: Decile別 1ヶ月先リターン")
        ax.set_xlabel("Decile（1=最低 → 10=最高）")
        ax.set_ylabel("平均前方リターン（%）")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
    fig.suptitle("Study95 — Decile Monotonicity Chart（1M forward return）")
    fig.tight_layout()
    fig.savefig(OUT_CHART, dpi=140)
    plt.close(fig)


def _fmt_pct(v, digits=2):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{digits}f}%"


def _fmt(v, digits=3):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _write_markdown(full: dict, panel: pd.DataFrame) -> None:
    r = full["results"]
    today_str = date.today().strftime("%Y-%m-%d")
    lines = [
        f"# Study95 — Cross-Sectional Momentum Factor-Level Ground Truth（{today_str}）",
        "",
        "正典: `reports/fujiko_r2_research_roadmap.md` v2 Part5 Study95仕様（H0・最優先）。",
        "禁止事項（フジコ法ロジック/RSR/percentile型トレーディングパラメータ/BTエンジン）は全て不使用"
        "（decile分解は本タスク仕様が明示的に要求する記述統計手法であり、選定ルールではない）。",
        "",
        f"データ: Universe C（Study75A）{full['data_quality']['n_rebalance_months']}ヶ月・"
        f"パネル行数={len(panel):,}・価格ファイル欠落={full['data_quality']['missing_price_files']}銘柄",
        "",
        "## サマリー判定",
        "",
        f"- **12-1モメンタム: {r['verdict']['verdict']}**"
        f"（12M年率スプレッド={_fmt_pct(r['verdict']['criteria']['ann_12m_spread'])}・"
        f"NW-t={_fmt(r['verdict']['criteria']['t_12m'])}・"
        f"正のhorizon数={r['verdict']['criteria']['positive_horizons']}/4）",
        f"  - 推奨: {r['verdict']['recommendation']}",
        f"- **Clenow slope×R²: {r['verdict_clenow']['verdict']}**"
        f"（12M年率スプレッド={_fmt_pct(r['verdict_clenow']['criteria']['ann_12m_spread'])}・"
        f"NW-t={_fmt(r['verdict_clenow']['criteria']['t_12m'])}・"
        f"正のhorizon数={r['verdict_clenow']['criteria']['positive_horizons']}/4）",
        f"  - 推奨: {r['verdict_clenow']['recommendation']}",
        "",
        "判定基準（タスク仕様）: SUCCESS=Q10-Q1年率>+5%∧|t|>2∧複数期間一貫 / "
        "FAIL=spread≈0・逆転・不安定 / KILL=CS momentum否定→FUJIKO-R2停止→PEAD/TSMOM再検討",
        "",
    ]

    for factor_prefix, title in (("mom", "12-1モメンタム"), ("clenow", "Clenow slope×R²")):
        b = r[factor_prefix]
        lines += [f"## {title}", "", "### Decile Table（平均前方リターン・%）", "",
                  "| Decile | 1M | 3M | 6M | 12M | N(1M) |", "|---|---|---|---|---|---|"]
        for d in range(1, N_DECILES + 1):
            row = [f"{d}"]
            for label in HORIZONS:
                v = b["decile_table"][label][d]["mean_fwd_return"]
                row.append(_fmt_pct(v))
            row.append(str(b["decile_table"]["1M"][d]["n"]))
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "### Q10-Q1 Spread（年率・Newey-West t・Bootstrap 95%CI）", "",
                  "| Horizon | N期間 | 平均spread(生) | 年率spread | NW-t | Bootstrap 95%CI(年率) |",
                  "|---|---|---|---|---|---|"]
        for label in HORIZONS:
            s = b["q10_q1_spread"][label]
            ci = s["bootstrap_ci_95"]
            ci_str = f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]" if ci[0] is not None else "N/A"
            lines.append(
                f"| {label} | {s['n_periods']} | {_fmt_pct(s['mean_spread'])} | "
                f"{_fmt_pct(s['annualized_spread'])} | {_fmt(s['newey_west_t'])} | {ci_str} |"
            )
        lines += ["", "### Information Coefficient", "",
                  "| Horizon | mean IC | std IC | t-stat | hit rate | N期間 |", "|---|---|---|---|---|---|"]
        for label in HORIZONS:
            ic = b["information_coefficient"][label]
            lines.append(
                f"| {label} | {_fmt(ic['mean_ic'])} | {_fmt(ic['std_ic'])} | {_fmt(ic['t_stat'])} | "
                f"{_fmt_pct(ic['hit_rate']) if ic['hit_rate'] is not None else 'N/A'} | {ic['n_periods']} |"
            )
        lines += ["", "### Monotonicity（Spearman: decile順位 vs 平均リターン）", "",
                  "| Horizon | Spearman ρ | p値 | 隣接decile逆転数(/9) |", "|---|---|---|---|"]
        for label in HORIZONS:
            m = b["monotonicity"][label]
            lines.append(
                f"| {label} | {_fmt(m['spearman_rho'])} | {_fmt(m['spearman_p'])} | "
                f"{m['adjacent_violations']}/{m['adjacent_pairs']} |" if m["spearman_rho"] is not None
                else f"| {label} | N/A | N/A | N/A |"
            )
        lines += ["", "### Regime分解（TOPIX>MA200・12M horizon年率spread）", "",
                  "| Regime | N期間 | 年率spread | NW-t |", "|---|---|---|---|"]
        for regime_key in ("bull", "bear"):
            s = b["regime_decomposition"][regime_key]["12M"]
            lines.append(f"| {regime_key} | {s['n_periods']} | {_fmt_pct(s['annualized_spread'])} | {_fmt(s['newey_west_t'])} |")
        lines += ["", "### Sector Neutral（TOPIX17内demean後・Q10-Q1 spread）", "",
                  "| Horizon | N期間 | 年率spread(sector-neutral) | NW-t |", "|---|---|---|---|"]
        for label in HORIZONS:
            s = b["sector17_neutral_q10_q1_spread"][label]
            lines.append(f"| {label} | {s['n_periods']} | {_fmt_pct(s['annualized_spread'])} | {_fmt(s['newey_west_t'])} |")
        lines += ["", "### 容量診断（ADV20・Q10 turnover）", ""]
        cap = b["capacity"]
        lines += [
            f"- Q1(最低decile) ADV20: median=¥{cap['adv20_jpy']['q1_bottom_decile']['median']:,.0f}"
            if cap["adv20_jpy"]["q1_bottom_decile"]["median"] else "- Q1 ADV20: N/A",
            f"- Q10(最高decile) ADV20: median=¥{cap['adv20_jpy']['q10_top_decile']['median']:,.0f}"
            if cap["adv20_jpy"]["q10_top_decile"]["median"] else "- Q10 ADV20: N/A",
            f"- Q10月次turnover: mean={_fmt_pct(cap['q10_monthly_turnover_mean'])} "
            f"median={_fmt_pct(cap['q10_monthly_turnover_median'])}",
            "",
            "### Factor Persistence（rank自己相関）", "",
            "| Lag | mean rank autocorr | N期間 |", "|---|---|---|",
        ]
        for lag_key in ("1M", "3M", "6M", "12M"):
            p = b["persistence"][lag_key]
            lines.append(f"| {lag_key} | {_fmt(p['mean_rank_autocorr'])} | {p['n_periods']} |")
        lines.append("")

    lines += [
        "## Sector別 IC（TOPIX17・1M horizon・N<30は insufficient_n）",
        "",
        "| Sector17 | mom IC | mom N | Clenow IC | Clenow N |",
        "|---|---|---|---|---|",
    ]
    mom17 = r["mom"]["sector17_ic"]
    cl17 = r["clenow"]["sector17_ic"]
    for sector in sorted(set(mom17.keys()) | set(cl17.keys())):
        m = mom17.get(sector, {})
        c = cl17.get(sector, {})
        lines.append(
            f"| {sector} | {_fmt(m.get('ic')) if m.get('ic') is not None else 'N/A(n<30)'} | {m.get('n', 0)} | "
            f"{_fmt(c.get('ic')) if c.get('ic') is not None else 'N/A(n<30)'} | {c.get('n', 0)} |"
        )

    lines += [
        "",
        "## 結論",
        "",
        f"- 仮説1（Cross-sectional momentum有効性）: **{r['verdict']['verdict']}**"
        f"（主判定=12-1モメンタム。Clenow slope×R²は補助判定=**{r['verdict_clenow']['verdict']}**）",
        f"- Kill基準発動: **{r['verdict']['kill_triggered']}**",
        f"- 次アクション: {r['verdict']['recommendation']}",
        "",
        "![decile chart](study95_decile_chart.png)",
        "",
        "---",
        "",
        f"*生成: Study95自動分析パイプライン, {today_str}。"
        "新規BT実行なし（BTエンジン不使用・純粋クロスセクション統計）。fresh_run_required準拠"
        "（本分析自体が初回実行・キャッシュ値不使用）。*",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
