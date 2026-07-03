"""
study_m1rca_oos_decomposition_2026-07-04.py
M1-RCA — OOS 2025 ΔCAGR=-2.06pp の要因分解（ユーザー追加指示 2026-07-04）

目的:
  M1(addon執行価格 close→open)適用でOOS2025のみ-2.06ppとなった理由を分解する。
  BASELINE(現行/close執行)とPATCHED(open執行)でOOS 2025のfull trade ledgerを取得し、
  (symbol, entry_idx)でマッチングして3分類する:
    - IDENTICAL_TRADE:    同一symbol/entry_idx/exit_idx/reason (価格差のみ)
    - TIMING_SHIFT:       同一symbol/entry_idxだが exit_idx or reason が相違 (Exitトリガー変化)
    - DIVERGED_PORTFOLIO: 片方にのみ存在 (保有銘柄構成そのものが変化)

  各カテゴリのΔPnL寄与率を算出し、「価格差だけ」か「Exit構造変化」かを判定する。

禁止: composite_alpha_bt.py の恒久変更なし。本スクリプト内でmonkeypatchし、実行後に自動復元。
観測専用。Production変更なし。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab

TODAY_STR = date.today().strftime("%Y-%m-%d")
OUT_FILE  = ROOT / "backtests" / f"study_m1rca_oos_decomposition_{TODAY_STR}.json"

CAPITAL   = 3_000_000
MIN_HOLD  = 3
OOS_START, OOS_END = "2025-01-01", "2025-12-31"
DATA_END = "2025-12-31"

CFG = {"exit_policy": "A", "addon_policy": "D", "rsr_exit": 70.0}  # CURRENT/D_ATR_EQ


def get_active(ds, all_syms, start, end):
    from src.config_loader import load_strategy_config
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=all_syms, start=start, end=end,
        bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds, sym_active_df, start, end) -> dict:
    return cab.run_scenario(
        scenario               = "BASELINE",
        universe_raw           = ds["universe_raw"],
        rsr_df                 = ds["rsr_df"],
        alpha_df               = None,
        regime_df              = ds["regime_df"],
        trade_syms             = ds["trade_syms"],
        rsr_syms               = ds["rsr_syms"],
        cfg                    = ds["base_cfg"],
        start                  = start, end=end, verbose=False,
        tech_matrices          = ds["tech_matrices"],
        breadth_series         = ds["breadth_series"],
        capital                = CAPITAL,
        min_hold               = MIN_HOLD,
        topix_close            = ds["topix_close"],
        market_shock_mode      = "composite",
        rsr_exit_threshold     = CFG["rsr_exit"],
        sym_active_df          = sym_active_df,
        enable_simple_rsr_exit = True,
        enable_atr_trailing_prod = True,
        enable_multilayer_rsr  = True,
        enable_atr_risk_sizing = False,
        enable_mtf_filter      = False,
        sizing_mode            = "existing",
        exit_policy            = CFG["exit_policy"],
        addon_policy           = CFG["addon_policy"],
        addon_size_frac        = 0.25,
        addon_atr_mult         = 1.0,
    )


def extract_trades(raw: dict) -> list[dict]:
    """_trades は SELL のみ。symbol/entry_idx/exit_idx/entry/exit/qty/pnl/reason を持つ。"""
    out = []
    for t in raw.get("_trades", []):
        out.append({
            "symbol": t["symbol"], "entry_idx": t.get("entry_idx"), "exit_idx": t.get("exit_idx"),
            "entry_px": round(float(t.get("entry", 0)), 2), "exit_px": round(float(t.get("exit", 0)), 2),
            "qty": int(t.get("qty", 0)), "pnl": round(float(t.get("pnl", 0)), 1),
            "reason": t.get("reason", ""), "hold_days": (t.get("exit_idx", 0) - t.get("entry_idx", 0))
                                                           if t.get("entry_idx") is not None and t.get("exit_idx") is not None else None,
        })
    return out


def key(t: dict) -> tuple:
    return (t["symbol"], t["entry_idx"])


def main():
    print("=" * 80)
    print("  M1-RCA — OOS 2025 ΔCAGR=-2.06pp 要因分解")
    print(f"  Date: {TODAY_STR}")
    print("=" * 80)

    ds = build_common_dataset(DATA_END)
    all_syms = list(ds["trade_syms"].keys())
    act = get_active(ds, all_syms, OOS_START, OOS_END)

    # ── BASELINE (close執行・現行) ──────────────────────────────────────────
    print("\n[RUN] BASELINE (close執行・現行) OOS 2025...")
    raw_base = run_bt(ds, act, OOS_START, OOS_END)
    trades_base = extract_trades(raw_base)
    addon_base  = raw_base.get("_addon_detail", [])
    print(f"  CAGR={raw_base['cagr']:+.2f}%  n_trades={len(trades_base)}  addon_count={raw_base.get('addon_count')}")

    # ── PATCHED (open執行・M1適用) — monkeypatch ────────────────────────────
    print("\n[PATCH] composite_alpha_bt.run_scenario 内 _addon_px を close_mat->open_mat へ一時変更...")
    src_path = ROOT / "src" / "backtest" / "composite_alpha_bt.py"
    original_src = src_path.read_text(encoding="utf-8")
    marker_old = "_addon_px = float(close_mat[next_i, _aidx])"
    marker_new = "_addon_px = float(open_mat[next_i, _aidx])"
    assert marker_old in original_src, "PATCH対象文字列が見つからない — grepで再確認要"
    patched_src = original_src.replace(marker_old, marker_new)
    assert patched_src != original_src

    try:
        src_path.write_text(patched_src, encoding="utf-8")
        # モジュール再ロードして反映
        import importlib
        importlib.reload(cab)

        print("[RUN] PATCHED (open執行・M1) OOS 2025...")
        raw_patch = run_bt(ds, act, OOS_START, OOS_END)
        trades_patch = extract_trades(raw_patch)
        addon_patch  = raw_patch.get("_addon_detail", [])
        print(f"  CAGR={raw_patch['cagr']:+.2f}%  n_trades={len(trades_patch)}  addon_count={raw_patch.get('addon_count')}")
    finally:
        print("\n[ROLLBACK] composite_alpha_bt.py を PATCH前へ復元...")
        src_path.write_text(original_src, encoding="utf-8")
        import importlib
        importlib.reload(cab)
        assert src_path.read_text(encoding="utf-8") == original_src
        print("  復元確認OK")

    # ── マッチング・3分類 ────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  トレード分類")
    print("─" * 80)

    base_by_key = {}
    for t in trades_base:
        base_by_key.setdefault(key(t), []).append(t)
    patch_by_key = {}
    for t in trades_patch:
        patch_by_key.setdefault(key(t), []).append(t)

    all_keys = set(base_by_key) | set(patch_by_key)

    identical, timing_shift, diverged = [], [], []
    for k in all_keys:
        b_list = base_by_key.get(k, [])
        p_list = patch_by_key.get(k, [])
        if b_list and p_list:
            # 同一(symbol, entry_idx) — 先頭同士を比較（通常1件ずつ）
            b, p = b_list[0], p_list[0]
            if b["exit_idx"] == p["exit_idx"] and b["reason"] == p["reason"]:
                identical.append({"key": k, "base": b, "patch": p, "delta_pnl": round(p["pnl"] - b["pnl"], 1)})
            else:
                timing_shift.append({"key": k, "base": b, "patch": p,
                                       "delta_pnl": round(p["pnl"] - b["pnl"], 1),
                                       "delta_hold_days": (p["hold_days"] or 0) - (b["hold_days"] or 0)})
        elif b_list and not p_list:
            diverged.append({"key": k, "side": "BASE_ONLY", "trade": b_list[0], "delta_pnl": round(-b_list[0]["pnl"], 1)})
        elif p_list and not b_list:
            diverged.append({"key": k, "side": "PATCH_ONLY", "trade": p_list[0], "delta_pnl": round(p_list[0]["pnl"], 1)})

    total_delta = sum(x["delta_pnl"] for x in identical) + sum(x["delta_pnl"] for x in timing_shift) + sum(x["delta_pnl"] for x in diverged)
    d_identical = sum(x["delta_pnl"] for x in identical)
    d_timing    = sum(x["delta_pnl"] for x in timing_shift)
    d_diverged  = sum(x["delta_pnl"] for x in diverged)

    def pct(x):
        return round(100.0 * x / total_delta, 1) if total_delta != 0 else 0.0

    print(f"\n  IDENTICAL_TRADE:    {len(identical):3d}件   ΔPnL={d_identical:+,.0f}円  ({pct(d_identical):+.1f}%)")
    print(f"  TIMING_SHIFT:       {len(timing_shift):3d}件   ΔPnL={d_timing:+,.0f}円  ({pct(d_timing):+.1f}%)")
    print(f"  DIVERGED_PORTFOLIO: {len(diverged):3d}件   ΔPnL={d_diverged:+,.0f}円  ({pct(d_diverged):+.1f}%)")
    print(f"  合計ΔPnL: {total_delta:+,.0f}円  (CAGR換算 baseline={raw_base['cagr']:+.2f}% -> patched={raw_patch['cagr']:+.2f}%  Δ={raw_patch['cagr']-raw_base['cagr']:+.2f}pp)")

    # ── 保有日数 ──────────────────────────────────────────────────────────
    hold_base  = [t["hold_days"] for t in trades_base if t["hold_days"] is not None]
    hold_patch = [t["hold_days"] for t in trades_patch if t["hold_days"] is not None]
    print(f"\n  保有日数 平均: baseline={np.mean(hold_base):.1f}d  patched={np.mean(hold_patch):.1f}d  Δ={np.mean(hold_patch)-np.mean(hold_base):+.1f}d")
    if timing_shift:
        ts_hold_deltas = [x["delta_hold_days"] for x in timing_shift]
        print(f"  TIMING_SHIFT該当トレードの保有日数差分: {ts_hold_deltas}")

    # ── 銘柄別寄与 ────────────────────────────────────────────────────────
    sym_delta: dict[str, float] = {}
    for group in (identical, timing_shift):
        for x in group:
            sym = x["key"][0]
            sym_delta[sym] = sym_delta.get(sym, 0.0) + x["delta_pnl"]
    for x in diverged:
        sym = x["key"][0]
        sym_delta[sym] = sym_delta.get(sym, 0.0) + x["delta_pnl"]
    sym_ranked = sorted(sym_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)

    print("\n  銘柄別ΔPnL寄与（上位10・絶対値降順）:")
    for sym, d in sym_ranked[:10]:
        print(f"    {sym}: {d:+,.0f}円")

    # ── addon件数比較（約定率の代理指標） ───────────────────────────────────
    print(f"\n  addon件数: baseline={len(addon_base)}件  patched={len(addon_patch)}件")

    # ── 判定 ──────────────────────────────────────────────────────────────
    price_only_pct = abs(pct(d_identical))
    print("\n" + "─" * 80)
    print("  判定（ユーザー基準: 価格差だけの寄与≥95% → 安心してM1採用可）")
    print("─" * 80)
    print(f"  IDENTICAL_TRADE(価格差のみ)の寄与率: {price_only_pct:.1f}%")
    verdict = "PRICE_ONLY_SAFE" if price_only_pct >= 95.0 else "STRUCTURAL_CHANGE_DETECTED"
    print(f"  判定: {verdict}")

    output = {
        "study": "M1_RCA_OOS_decomposition", "date": TODAY_STR,
        "period": "OOS_2025",
        "baseline_cagr": raw_base["cagr"], "patched_cagr": raw_patch["cagr"],
        "delta_cagr": round(raw_patch["cagr"] - raw_base["cagr"], 2),
        "classification": {
            "identical_trade": {"count": len(identical), "delta_pnl": d_identical, "pct": pct(d_identical), "detail": identical},
            "timing_shift":     {"count": len(timing_shift), "delta_pnl": d_timing, "pct": pct(d_timing), "detail": timing_shift},
            "diverged_portfolio": {"count": len(diverged), "delta_pnl": d_diverged, "pct": pct(d_diverged), "detail": diverged},
        },
        "hold_days": {"baseline_avg": round(float(np.mean(hold_base)), 2), "patched_avg": round(float(np.mean(hold_patch)), 2)},
        "symbol_contribution": [{"symbol": s, "delta_pnl": d} for s, d in sym_ranked],
        "addon_count": {"baseline": len(addon_base), "patched": len(addon_patch)},
        "verdict": verdict,
        "price_only_pct": price_only_pct,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUTPUT] {OUT_FILE}")


if __name__ == "__main__":
    main()
