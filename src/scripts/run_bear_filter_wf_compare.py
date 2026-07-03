"""
scripts/run_bear_filter_wf_compare.py

Step 2: bear_universe_filter の WF比較
  パターン A: 現行固定除外（7セクター、Bear時のみ）
  パターン B: sector_rs_63 < threshold で動的除外
              Bull時 threshold=-0.05 / Bear時 threshold=-0.02

WF 5分割（OOS: 2020/2021/2022/2023/2024）+ True OOS 2025

採用基準:
  2022 MaxDD 改善 >= 2〜3pt  AND  2025 OOS Sharpe 劣化 <= 0.10〜0.15

実行:
  cd C:/ai-trading
  python src/scripts/run_bear_filter_wf_compare.py
"""

from __future__ import annotations
import os, sys, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest.composite_alpha_bt as _bt
from backtest.wf_dynamic_universe import (
    load_data,
    BULL_ACTIVE_N, BEAR_ACTIVE_N,
    WF_SEGS, TRUE_OOS, IS_FULL, OUTPUT_DIR,
)
from src.config_loader import load_strategy_config
from src.strategy.universe import (
    build_dyn_rsr42_active,
    _zscore,
    MA200_PERIOD,
    MOM_PERIOD,
    VOL_PERIOD,
    SUSTAINED_BEAR_DAYS,
    LOOKBACK_BEAR_CHECK,
    apply_bear_universe_filter,
)

os.environ.pop("DATA_VERSION", None)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR, f"wf_bear_filter_compare_{time.strftime('%Y-%m-%d')}.json"
)

SHOCK_MODE = "composite"

# ---- パターン定義 --------------------------------------------------------
# Pattern A: 固定7セクター（現行本番）
FIXED_SECTORS = ["機械", "鉄鋼", "銀行業", "保険業", "輸送用機器", "海運業", "化学"]

# Pattern B: 動的RS閾値
B_BULL_THRESH = -0.05   # Bull時: rs_63 < -0.05 で除外（緩め）
B_BEAR_THRESH = -0.02   # Bear時: rs_63 < -0.02 で除外（厳しめ）
RS_LOOKBACK   = 63      # セクターRS計算期間（63営業日≒3ヶ月）


# ---- Pattern B 用: 動的セクター除外 sym_active_df ビルダー ---------------

def _compute_sector_rs(
    hist_close: pd.DataFrame,
    topix_hist: pd.Series,
    sym_sector_map: dict[str, str],
    lookback: int = RS_LOOKBACK,
) -> dict[str, float]:
    """
    eval_dt 時点での sector_rs_63 = sector_63d_return - topix_63d_return を返す。

    先読み防止: hist_close / topix_hist はすでに eval_dt 以前のデータ。

    Returns
    -------
    {sector_name: rs_value}  ※ データ不足の場合は空 dict
    """
    if len(hist_close) <= lookback or len(topix_hist) <= lookback:
        return {}

    try:
        sym_ret_63 = hist_close.iloc[-1] / hist_close.iloc[-1 - lookback] - 1
        topix_ret_63 = float(topix_hist.iloc[-1] / topix_hist.iloc[-1 - lookback] - 1)
    except Exception:
        return {}

    sector_rs: dict[str, float] = {}
    for sym, sec in sym_sector_map.items():
        if sym in sym_ret_63.index and not np.isnan(sym_ret_63[sym]):
            if sec not in sector_rs:
                sector_rs[sec] = []
            sector_rs[sec].append(sym_ret_63[sym])  # type: ignore[assignment]

    return {
        sec: float(np.mean(rets)) - topix_ret_63
        for sec, rets in sector_rs.items()  # type: ignore[attr-defined]
        if rets
    }


def _dynamic_excluded_sectors(
    sector_rs: dict[str, float],
    is_bear: bool,
    bull_thresh: float = B_BULL_THRESH,
    bear_thresh: float = B_BEAR_THRESH,
) -> list[str]:
    """sector_rs < threshold のセクター一覧を返す。"""
    thresh = bear_thresh if is_bear else bull_thresh
    return [sec for sec, rs in sector_rs.items() if rs < thresh]


def _build_active_dynamic_rs(
    universe_raw: dict,
    topix_close: pd.Series,
    all_syms: list[str],
    active_n: int,
    data_start: str,
    end: str,
    rsr_df: pd.DataFrame | None = None,
    bear_n: int | None = None,
    sym_sector_map: dict[str, str] | None = None,
    bull_thresh: float = B_BULL_THRESH,
    bear_thresh: float = B_BEAR_THRESH,
) -> pd.DataFrame:
    """
    Pattern B 用 sym_active_df ビルダー。

    build_sym_active_df と同じ月次ループだが、セクター除外リストを
    sector_rs_63 から動的に決定する。

    Bear時: rs_63 < bear_thresh のセクターを除外
    Bull時: rs_63 < bull_thresh のセクターを除外（通常ほぼなし）

    先読み防止: 月 T の選択 = 月 T-1 末データで計算
    """
    ts_start = pd.Timestamp(data_start)
    ts_end   = pd.Timestamp(end)

    if bear_n is None:
        bear_n = BEAR_ACTIVE_N

    close_all = pd.DataFrame({
        s: universe_raw[s]["df"]["Close"]
        for s in all_syms if s in universe_raw
    }).sort_index()
    vol_all = pd.DataFrame({
        s: universe_raw[s]["df"]["Volume"]
        for s in all_syms if s in universe_raw
    }).sort_index()

    valid_syms = list(close_all.columns)

    all_dates = close_all.index[
        (close_all.index >= ts_start) & (close_all.index <= ts_end)
    ]
    if len(all_dates) == 0:
        return pd.DataFrame(1.0, index=all_dates, columns=valid_syms)

    monthly_first: dict[str, pd.Timestamp] = {}
    for dt in all_dates:
        key = dt.strftime("%Y-%m")
        if key not in monthly_first:
            monthly_first[key] = dt

    monthly_active: dict[str, set] = {}
    LOSS_PENALTY_COEF = 0.10
    LOSS_PERIOD       = 90

    for key, first_dt in sorted(monthly_first.items()):
        pos = close_all.index.searchsorted(first_dt)
        if pos == 0:
            monthly_active[key] = set(valid_syms[:min(active_n, len(valid_syms))])
            continue
        eval_dt    = close_all.index[pos - 1]
        hist_close = close_all.loc[:eval_dt]
        hist_vol   = vol_all.loc[:eval_dt]

        if len(hist_close) < MOM_PERIOD + 5:
            monthly_active[key] = set(valid_syms[:min(active_n, len(valid_syms))])
            continue

        # ── レジーム判定 ───────────────────────────────────────────────
        topix_hist = topix_close.loc[:eval_dt]
        topix_last = float(topix_hist.iloc[-1]) if len(topix_hist) > 0 else np.nan
        if len(topix_hist) >= MA200_PERIOD:
            topix_ma200 = float(topix_hist.iloc[-MA200_PERIOD:].mean())
        else:
            topix_ma200 = float(topix_hist.mean()) if len(topix_hist) > 0 else np.nan

        is_sustained_bear = False
        if not np.isnan(topix_last) and not np.isnan(topix_ma200) and topix_last < topix_ma200:
            lb = min(LOOKBACK_BEAR_CHECK, len(topix_hist))
            if lb >= LOOKBACK_BEAR_CHECK:
                topix_lb      = topix_hist.iloc[-lb:]
                rolling_ma200 = topix_hist.rolling(MA200_PERIOD).mean().loc[:eval_dt]
                recent_ma200  = rolling_ma200.iloc[-lb:] if len(rolling_ma200) >= lb else rolling_ma200
                if len(topix_lb) == len(recent_ma200):
                    days_below        = int((topix_lb.values < recent_ma200.values).sum())
                    is_sustained_bear = days_below >= SUSTAINED_BEAR_DAYS

        is_bull = not is_sustained_bear

        # ── 動的セクター除外（Pattern B のコア） ─────────────────────────
        sector_rs = {}
        if sym_sector_map:
            sector_rs = _compute_sector_rs(hist_close, topix_hist, sym_sector_map)

        excluded_dyn = _dynamic_excluded_sectors(sector_rs, is_bear=not is_bull,
                                                  bull_thresh=bull_thresh,
                                                  bear_thresh=bear_thresh)

        # 除外後の close
        if excluded_dyn and sym_sector_map:
            keep_syms = [
                s for s in hist_close.columns
                if sym_sector_map.get(s, "") not in excluded_dyn
            ]
            work_close = hist_close[keep_syms] if keep_syms else hist_close
        else:
            work_close = hist_close

        # ── 共通ファクター ────────────────────────────────────────────
        rsr_score = pd.Series(np.nan, index=work_close.columns)
        if rsr_df is not None:
            rsr_hist = rsr_df.loc[rsr_df.index <= eval_dt]
            if not rsr_hist.empty:
                last_rsr = rsr_hist.iloc[-1]
                for sym in work_close.columns:
                    if sym in last_rsr.index:
                        rsr_score[sym] = last_rsr[sym]

        if len(hist_vol) >= VOL_PERIOD:
            avg_vol = hist_vol[list(work_close.columns)].iloc[-VOL_PERIOD:].mean()
        else:
            avg_vol = hist_vol[list(work_close.columns)].mean()
        log_vol = np.log1p(avg_vol.replace(0, np.nan).fillna(0))

        if len(work_close) > MOM_PERIOD:
            mom63 = work_close.iloc[-1] / work_close.iloc[-1 - MOM_PERIOD] - 1
        else:
            mom63 = pd.Series(0.0, index=work_close.columns)

        loss_penalty_z = pd.Series(0.0, index=work_close.columns)
        if len(work_close) > LOSS_PERIOD:
            pnl_90d    = work_close.iloc[-1] / work_close.iloc[-1 - LOSS_PERIOD] - 1
            loss_only  = pnl_90d.clip(upper=0.0).abs()
            loss_penalty_z = _zscore(loss_only)

        if is_bull:
            score       = (
                0.40 * _zscore(mom63)
                + 0.35 * _zscore(rsr_score)
                + 0.25 * _zscore(log_vol)
                - LOSS_PENALTY_COEF * loss_penalty_z
            )
            score_valid = score.dropna()
            n_select    = min(active_n, len(score_valid))
        else:
            if len(topix_hist) > MOM_PERIOD:
                topix_mom = float(topix_hist.iloc[-1] / topix_hist.iloc[-1 - MOM_PERIOD] - 1)
            else:
                topix_mom = 0.0

            if len(work_close) > MOM_PERIOD:
                sym_mom  = work_close.iloc[-1] / work_close.iloc[-1 - MOM_PERIOD] - 1
            else:
                sym_mom  = pd.Series(0.0, index=work_close.columns)
            rs_topix = sym_mom - topix_mom

            score = (
                0.50 * _zscore(rs_topix)
                + 0.30 * _zscore(rsr_score.reindex(work_close.columns).fillna(0))
                + 0.20 * _zscore(log_vol.reindex(work_close.columns).fillna(0))
                - LOSS_PENALTY_COEF * loss_penalty_z.reindex(work_close.columns).fillna(0)
            )
            score_valid = score[rs_topix > 0].dropna()
            if len(score_valid) == 0:
                score_valid = score.dropna()
            n_select = min(bear_n, max(20, len(score_valid)))

        top_syms = set(score_valid.nlargest(n_select).index.tolist())
        monthly_active[key] = top_syms

    active_df = pd.DataFrame(0.0, index=all_dates, columns=valid_syms)
    for dt in all_dates:
        key = dt.strftime("%Y-%m")
        active_set = monthly_active.get(key, set())
        for sym in active_set:
            if sym in active_df.columns:
                active_df.loc[dt, sym] = 1.0

    return active_df


# ---- セグメント実行ヘルパー -----------------------------------------------

def run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
            tech_matrices, topix_close, cfg,
            start: str, end: str,
            sym_active_df: pd.DataFrame | None = None) -> dict:
    return _bt.run_scenario(
        scenario                 = "BASELINE",
        universe_raw             = universe_raw,
        rsr_df                   = rsr_df,
        alpha_df                 = alpha_df,
        regime_df                = regime_df,
        trade_syms               = all_syms,
        rsr_syms                 = all_syms,
        cfg                      = cfg,
        start                    = start,
        end                      = end,
        capital                  = float(cfg.portfolio.capital),
        verbose                  = False,
        tech_matrices            = tech_matrices,
        topix_close              = topix_close,
        min_hold                 = int(cfg.risk.min_hold_days),
        market_shock_mode        = SHOCK_MODE,
        enable_volume_filter     = True,
        enable_volatility_filter = False,
        enable_atr_filter        = False,
        enable_market_filter     = False,
        sym_active_df            = sym_active_df,
    )


# ---- 1パターン WF全セグ実行 -----------------------------------------------

def run_pattern(
    pattern_name: str,
    sym_active_builder,        # callable(data_start, end) -> sym_active_df
    universe_raw, rsr_df, alpha_df, regime_df, all_syms,
    tech_matrices, topix_close, cfg,
) -> dict:
    print(f"\n{'─'*60}")
    print(f"  パターン {pattern_name}")

    seg_results = []
    wf_wins     = 0

    for seg_def in WF_SEGS:
        n        = seg_def["seg"]
        is_s, is_e   = seg_def["is"]
        oos_s, oos_e = seg_def["oos"]

        active_is  = sym_active_builder("2017-01-01", is_e)
        active_oos = sym_active_builder("2017-01-01", oos_e)

        r_is  = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, is_s, is_e, active_is)
        r_oos = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                        tech_matrices, topix_close, cfg, oos_s, oos_e, active_oos)

        is_sh  = r_is.get("sharpe", 0)
        oos_sh = r_oos.get("sharpe", 0)
        oos_dd = r_oos.get("max_dd", 0)
        win    = oos_sh > 0
        if win:
            wf_wins += 1

        seg_results.append({
            "seg": n, "oos_year": oos_s[:4],
            "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
            "oos_max_dd": round(oos_dd, 2), "win": win,
        })
        mark = "OK" if win else "NG"
        print(f"    Seg{n} OOS={oos_s[:4]}  IS={is_sh:.3f}  OOS={oos_sh:.3f}"
              f"  DD={oos_dd:+.1f}%  {mark}")
        del r_is, r_oos, active_is, active_oos

    # Full IS
    active_full = sym_active_builder("2017-01-01", IS_FULL[1])
    r_is_full   = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                          tech_matrices, topix_close, cfg, IS_FULL[0], IS_FULL[1], active_full)

    # True OOS 2025
    active_2025  = sym_active_builder("2017-01-01", TRUE_OOS[1])
    r_oos_2025   = run_seg(universe_raw, rsr_df, alpha_df, regime_df, all_syms,
                           tech_matrices, topix_close, cfg, TRUE_OOS[0], TRUE_OOS[1], active_2025)

    is_sh_full    = r_is_full.get("sharpe", 0)
    is_dd_full    = r_is_full.get("max_dd", 0)
    is_cagr_full  = r_is_full.get("cagr", 0)
    oos_sh_2025   = r_oos_2025.get("sharpe", 0)
    oos_dd_2025   = r_oos_2025.get("max_dd", 0)
    oos_cagr_2025 = r_oos_2025.get("cagr", 0)

    print(f"\n    Full IS (2018-2024): CAGR={is_cagr_full:+.1f}%  Sharpe={is_sh_full:.3f}"
          f"  MaxDD={is_dd_full:.1f}%")
    print(f"    True OOS 2025:       CAGR={oos_cagr_2025:+.1f}%  Sharpe={oos_sh_2025:.3f}"
          f"  MaxDD={oos_dd_2025:.1f}%")
    print(f"    WF wins: {wf_wins}/5")

    # 2022 MaxDD（Seg3 OOS=2022）
    seg3   = next((s for s in seg_results if s["oos_year"] == "2022"), None)
    dd_2022 = seg3["oos_max_dd"] if seg3 else float("nan")
    sh_2022 = seg3["oos_sharpe"] if seg3 else float("nan")

    return {
        "pattern": pattern_name,
        "wf_wins": wf_wins,
        "seg_results": seg_results,
        "full_is": {
            "cagr":   round(is_cagr_full, 2),
            "sharpe": round(is_sh_full, 3),
            "max_dd": round(is_dd_full, 2),
        },
        "oos_2022": {
            "sharpe": round(sh_2022, 3),
            "max_dd": round(dd_2022, 2),
        },
        "oos_2025": {
            "cagr":   round(oos_cagr_2025, 2),
            "sharpe": round(oos_sh_2025, 3),
            "max_dd": round(oos_dd_2025, 2),
        },
    }


# ---- 採用判定 -------------------------------------------------------------

def judge(res_a: dict, res_b: dict) -> str:
    """
    採用基準:
      2022 MaxDD 改善 >= 2.0pt  AND  2025 OOS Sharpe 劣化 <= 0.15
    """
    dd_a  = res_a["oos_2022"]["max_dd"]   # 負値
    dd_b  = res_b["oos_2022"]["max_dd"]   # 負値
    sh_a  = res_a["oos_2025"]["sharpe"]
    sh_b  = res_b["oos_2025"]["sharpe"]

    dd_improve    = dd_a - dd_b          # B の DD が改善（小さくなる）なら正
    sh_degradation = sh_a - sh_b         # B の Sharpe が下がるなら正

    lines = [
        f"  2022 MaxDD: A={dd_a:.1f}% → B={dd_b:.1f}%  改善={dd_improve:+.1f}pt",
        f"  2025 Sharpe: A={sh_a:.3f} → B={sh_b:.3f}  劣化={sh_degradation:+.3f}",
    ]

    if dd_improve >= 2.0 and sh_degradation <= 0.15:
        verdict = "→ 採用 (2022DD改善≥2pt かつ 2025Sharpe劣化≤0.15)"
    elif dd_improve >= 1.0 and sh_degradation <= 0.10:
        verdict = "→ 条件付き採用検討 (DD改善≥1pt かつ Sharpe劣化≤0.10)"
    else:
        verdict = "→ 不採用 (採用基準未達)"

    return "\n".join(lines) + "\n" + verdict


# ---- メイン ------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  bear_universe_filter WF比較")
    print("  A: 固定7セクター / B: sector_rs_63 動的除外")
    print("=" * 65)

    t0 = time.time()

    # ─ データロード ─────────────────────────────────────────────────────
    cfg = load_strategy_config()
    (
        universe_raw, rsr_df, rsr_df_full, alpha_df, regime_df,
        rsr_syms_avail, all_syms, tech_matrices, topix_close, cfg,
    ) = load_data(cfg)

    # rsr_syms_avail = {symbol: sector} （RSR42銘柄）
    sym_sector_map = dict(rsr_syms_avail)
    trade_syms     = rsr_syms_avail

    print(f"\nデータロード完了: {len(universe_raw)}銘柄  RSR42={len(rsr_syms_avail)}銘柄")
    print(f"セクターマップ: {len(sym_sector_map)}銘柄  ユニーク={len(set(sym_sector_map.values()))}セクター")
    print(f"  → セクター一覧: {sorted(set(sym_sector_map.values()))}")

    # ─ Pattern A: 固定7セクター ─────────────────────────────────────────
    def build_a(data_start, end):
        return build_dyn_rsr42_active(
            universe_raw         = universe_raw,
            topix_close          = topix_close,
            rsr_df               = rsr_df,
            all_syms             = list(trade_syms.keys()),
            start                = data_start,
            end                  = end,
            bear_exclude_sectors = FIXED_SECTORS,
            sym_sector_map       = sym_sector_map,
        )

    # ─ Pattern B: 動的 sector_rs_63 ─────────────────────────────────────
    def build_b(data_start, end):
        return _build_active_dynamic_rs(
            universe_raw   = universe_raw,
            topix_close    = topix_close,
            all_syms       = list(trade_syms.keys()),
            active_n       = BULL_ACTIVE_N,
            data_start     = data_start,
            end            = end,
            rsr_df         = rsr_df,
            bear_n         = BEAR_ACTIVE_N,
            sym_sector_map = sym_sector_map,
            bull_thresh    = B_BULL_THRESH,
            bear_thresh    = B_BEAR_THRESH,
        )

    # ─ WF実行 ─────────────────────────────────────────────────────────
    print("\n[Pattern A] 固定7セクター除外...")
    res_a = run_pattern(
        "A（固定7セクター）", build_a,
        universe_raw, rsr_df, alpha_df, regime_df, trade_syms,
        tech_matrices, topix_close, cfg,
    )

    print("\n[Pattern B] 動的 sector_rs_63 除外...")
    res_b = run_pattern(
        f"B（動的RS: Bull<{B_BULL_THRESH} / Bear<{B_BEAR_THRESH}）", build_b,
        universe_raw, rsr_df, alpha_df, regime_df, trade_syms,
        tech_matrices, topix_close, cfg,
    )

    elapsed = time.time() - t0

    # ─ 結果出力 ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("=== Step 2: WF 検証結果 ===")
    print()

    for res, label in [(res_a, "パターン A（固定除外）"), (res_b, "パターン B（動的除外）")]:
        print(f"{label}:")
        for s in res["seg_results"]:
            mark = "OK" if s["win"] else "NG"
            print(f"  Seg{s['seg']} {s['oos_year']}: IS={s['is_sharpe']:.3f}"
                  f"  OOS={s['oos_sharpe']:.3f}  DD={s['oos_max_dd']:+.1f}%  {mark}")
        print(f"  2022 Sharpe: {res['oos_2022']['sharpe']:.3f}"
              f"  / MaxDD: {res['oos_2022']['max_dd']:.1f}%")
        print(f"  2025 OOS: Sharpe={res['oos_2025']['sharpe']:.3f}"
              f"  / CAGR={res['oos_2025']['cagr']:+.1f}%"
              f"  / MaxDD={res['oos_2025']['max_dd']:.1f}%")
        print(f"  WF wins: {res['wf_wins']}/5")
        print()

    verdict = judge(res_a, res_b)
    print(verdict)
    print()
    print(f"実行時間: {elapsed/60:.1f}分")

    # ─ JSON保存 ──────────────────────────────────────────────────────
    output = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "pattern_a": "固定7セクター除外（Bear時のみ）",
            "pattern_b": f"動的RS除外 Bull<{B_BULL_THRESH} Bear<{B_BEAR_THRESH}",
            "rs_lookback": RS_LOOKBACK,
        },
        "results": {"A": res_a, "B": res_b},
        "verdict": verdict,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"→ 保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
