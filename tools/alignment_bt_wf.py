"""
tools/alignment_bt_wf.py — BT/live整合化 対照BT + WF 5-fold  (research-only)

目的: replay推定(+3〜4pp)をエンジンBT内で正式検証。
     live専用3制約をBTエンジンに実装し WF 5-fold / 2022 Bear / OOS/IS を比較。

ケース:
  base        : 現行BT (制約なし / MAX_NEW=無制限 / names=weight-only / buffer=0%)
  live_proxy  : live現行相当 (MAX_NEW=1 / names=1 / buffer=10%)
  mx2         : MAX_NEW=2のみ
  sn2         : sector_names=2のみ
  cb5         : cash_buffer=5%のみ
  bundle_225  : MAX_NEW=2 + names=2 + buffer=5% (提案バンドル)

WF: 5-fold (2018-2025 CLAUDE.md標準セグメント)
判定: WF≥4/5 ∧ Seg3_2022_oos_cagr≥0 ∧ ΔIS_CAGR vs base≥-0.5pp

Run : cd C:/ai-trading && python tools/alignment_bt_wf.py
出力: backtests/alignment_bt_wf_YYYY-MM-DD.json
"""
from __future__ import annotations

import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import (
    load_data, run_period,
    IS_START, IS_END, OOS_START, OOS_END,
)

OUT_JSON = Path(f"backtests/alignment_bt_wf_{time.strftime('%Y-%m-%d')}.json")

# ─────────────────────────────────────────────────────────────────────
# WF 5-fold 標準セグメント（CLAUDE.md基準）
# ─────────────────────────────────────────────────────────────────────
WF_SEGS = [
    {"seg": 1, "is": ("2018-01-01", "2019-12-31"), "oos": ("2020-01-01", "2020-12-31")},
    {"seg": 2, "is": ("2019-01-01", "2020-12-31"), "oos": ("2021-01-01", "2021-12-31")},
    {"seg": 3, "is": ("2020-01-01", "2021-12-31"), "oos": ("2022-01-01", "2022-12-31")},  # Bear
    {"seg": 4, "is": ("2021-01-01", "2022-12-31"), "oos": ("2023-01-01", "2023-12-31")},
    {"seg": 5, "is": ("2022-01-01", "2023-12-31"), "oos": ("2024-01-01", "2024-12-31")},
]
FULL_IS  = (IS_START, IS_END)      # 2018-01-01 〜 2024-12-31
TRUE_OOS = (OOS_START, OOS_END)    # 2025-01-01 〜 2025-12-31

# WF pass: OOS CAGR > 0 をパス判定（OVERFIT_GUARD: oos_is_ratio≥0.7は参考表示）
WF_PASS_MIN = 4  # ≥4/5

# ─────────────────────────────────────────────────────────────────────
# 研究ケース定義
# ─────────────────────────────────────────────────────────────────────
CASES: dict[str, dict] = {
    "base":       dict(max_new_per_day=None, max_names_per_sector=None, cash_buffer=0.00),
    "live_proxy": dict(max_new_per_day=1,    max_names_per_sector=1,    cash_buffer=0.10),
    "mx2":        dict(max_new_per_day=2,    max_names_per_sector=None, cash_buffer=0.00),
    "sn2":        dict(max_new_per_day=None, max_names_per_sector=2,    cash_buffer=0.00),
    "cb5":        dict(max_new_per_day=None, max_names_per_sector=None, cash_buffer=0.05),
    "bundle_225": dict(max_new_per_day=2,    max_names_per_sector=2,    cash_buffer=0.05),
}

CASE_LABELS = {
    "base":       "base       (現行BT: 制約なし)",
    "live_proxy": "live_proxy (live現行: 1/1/10%)",
    "mx2":        "mx2        (MAX_NEW=2のみ)",
    "sn2":        "sn2        (sector_names=2のみ)",
    "cb5":        "cb5        (cash_buffer=5%のみ)",
    "bundle_225": "bundle_225 (2+2+5% 全適用)",
}


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _run(data_args, start, end, case_params):
    m = run_period(*data_args, start=start, end=end, pattern="A",
                   verbose=False, **case_params)
    return {
        "cagr":    round(m.get("cagr",         0.0), 2),
        "sharpe":  round(m.get("sharpe",        0.0), 3),
        "max_dd":  round(m.get("max_dd",        0.0), 2),
        "pf":      round(m.get("profit_factor", 0.0), 3),
        "wr":      round(m.get("win_rate",      0.0), 1),
        "n":       m.get("n_trades", 0),
        "exp":     round(m.get("avg_exposure",  0.0), 1),
        "annual":  m.get("annual_returns", {}),
    }


def _yearly_str(annual: dict, years=range(2018, 2026)):
    parts = []
    for y in years:
        v = annual.get(str(y))
        parts.append(f"{y}:{v:+.1f}%" if v is not None else f"{y}:N/A")
    return "  ".join(parts)


def run_wf(case_name, case_params, data_args):
    folds = []
    for seg in WF_SEGS:
        is_m  = _run(data_args, *seg["is"],  case_params)
        oos_m = _run(data_args, *seg["oos"], case_params)
        passed = oos_m["cagr"] > 0
        ois = round(oos_m["cagr"] / is_m["cagr"], 3) if is_m["cagr"] != 0 else None
        folds.append({
            "seg":       seg["seg"],
            "oos_year":  seg["oos"][0][:4],
            "is_cagr":   is_m["cagr"],
            "oos_cagr":  oos_m["cagr"],
            "oos_sharpe": oos_m["sharpe"],
            "oos_dd":    oos_m["max_dd"],
            "ois_ratio": ois,
            "pass":      passed,
        })

    pass_count = sum(1 for f in folds if f["pass"])
    seg3 = next((f for f in folds if f["seg"] == 3), None)
    seg3_oos_cagr = seg3["oos_cagr"] if seg3 else None

    return {
        "case":           case_name,
        "pass_count":     pass_count,
        "wf_pass":        pass_count >= WF_PASS_MIN,
        "seg3_2022_cagr": seg3_oos_cagr,
        "seg3_2022_ok":   (seg3_oos_cagr is not None and seg3_oos_cagr >= 0.0),
        "folds":          folds,
    }


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_strategy_config()
    print("=" * 72)
    print("  BT/live整合化 対照BT + WF 5-fold  (research-only, Pattern A)")
    print(f"  IS: {FULL_IS[0]}〜{FULL_IS[1]}  /  OOS: {TRUE_OOS[0]}〜{TRUE_OOS[1]}")
    print(f"  ケース数: {len(CASES)}  WF folds: {len(WF_SEGS)}")
    print("=" * 72 + "\n")

    t_start = time.time()
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)
    # run_period 引数順: (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df, topix_close, rsr_syms, cfg)
    data_args = (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                 topix_close, rsr_syms, cfg)
    print(f"  データロード完了 ({time.time()-t_start:.0f}s)\n")

    all_results = {}
    run_count = 0
    total_runs = len(CASES) * (len(WF_SEGS) * 2 + 2)  # WF×2 + IS + OOS

    for case_name, case_params in CASES.items():
        print(f"  ── {CASE_LABELS[case_name]}")
        t_case = time.time()

        # WF 5-fold
        wf_res = run_wf(case_name, case_params, data_args)
        run_count += len(WF_SEGS) * 2

        # Full IS
        is_m  = _run(data_args, *FULL_IS,  case_params)
        run_count += 1

        # True OOS 2025
        oos_m = _run(data_args, *TRUE_OOS, case_params)
        run_count += 1

        ois = round(oos_m["cagr"] / is_m["cagr"], 3) if is_m["cagr"] != 0 else None

        all_results[case_name] = {
            "wf":      wf_res,
            "full_is": is_m,
            "oos_25":  oos_m,
            "ois":     ois,
        }

        elapsed = time.time() - t_case
        verdict = "PASS" if (wf_res["wf_pass"] and wf_res["seg3_2022_ok"]) else "FAIL"
        print(f"     IS CAGR={is_m['cagr']:+.2f}%  OOS={oos_m['cagr']:+.2f}%  "
              f"WF {wf_res['pass_count']}/5  Seg3_22={wf_res['seg3_2022_cagr']:+.2f}%  "
              f"→ {verdict}  ({elapsed:.0f}s)")
        print(f"     年次: {_yearly_str(is_m['annual'])}")

    # ── サマリー表 ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  比較サマリー")
    print(f"  {'ケース':<28} {'IS%':>7} {'OOS%':>7} {'WF':>4} {'Seg3_22%':>9} "
          f"{'MaxDD%':>8} {'Exp%':>6} {'N':>5} {'判定':>6}")
    print("  " + "-" * 82)

    base_cagr = all_results["base"]["full_is"]["cagr"]
    for case_name in CASES:
        r = all_results[case_name]
        w = r["wf"]
        is_m = r["full_is"]
        oos_m = r["oos_25"]
        delta = is_m["cagr"] - base_cagr
        verdict = "PASS" if (w["wf_pass"] and w["seg3_2022_ok"]) else "FAIL"
        mark = "★" if verdict == "PASS" and case_name != "base" else " "
        delta_str = f"({delta:+.1f}pp)" if case_name != "base" else "     --"
        seg3_str = f"{w['seg3_2022_cagr']:+.2f}" if w['seg3_2022_cagr'] is not None else "  N/A"
        print(f"  {mark}{CASE_LABELS[case_name]:<36} "
              f"{is_m['cagr']:>+6.2f}% {delta_str} "
              f"{oos_m['cagr']:>+6.2f}%  "
              f"{w['pass_count']}/5  {seg3_str:>8}%  "
              f"{is_m['max_dd']:>7.1f}%  "
              f"{is_m['exp']:>5.1f}%  "
              f"{is_m['n']:>5}  {verdict}")

    # ── WF fold 詳細 ──────────────────────────────────────────────
    print("\n  ── WF fold詳細 (OOS CAGR%)")
    header = f"  {'ケース':<18}" + "".join(
        f"  {WF_SEGS[i]['oos'][0][:4]}({'P' if True else 'F'})" for i in range(5)
    )
    header = f"  {'ケース':<18}  2020   2021   2022   2023   2024"
    print(header)
    print("  " + "-" * 60)
    for case_name in CASES:
        folds = all_results[case_name]["wf"]["folds"]
        row = f"  {case_name:<18}"
        for f in folds:
            mark = "✓" if f["pass"] else "✗"
            row += f"  {f['oos_cagr']:>+5.1f}{mark}"
        print(row)

    # ── 2022 Bear 年次比較 ──────────────────────────────────────────
    print("\n  ── IS全期間 年次CAGR% (base比)")
    print(f"  {'ケース':<18}  " + "  ".join(f"{y}" for y in range(2018, 2025)))
    print("  " + "-" * 72)
    for case_name in CASES:
        annual = all_results[case_name]["full_is"]["annual"]
        row = f"  {case_name:<18}"
        for y in range(2018, 2025):
            v = annual.get(str(y))
            row += f"  {v:>+5.1f}%" if v is not None else "    N/A"
        print(row)

    # ── JSON保存 ──────────────────────────────────────────────
    output = {
        "date":        time.strftime("%Y-%m-%d"),
        "engine":      "capital_allocation_abc.run_period pattern=A",
        "wf_segs":     WF_SEGS,
        "cases":       CASES,
        "results":     all_results,
        "total_time_s": round(time.time() - t_start, 1),
    }
    def _jsonable(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_):       return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj).__name__}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=_jsonable),
        encoding="utf-8")
    print(f"\n  保存: {OUT_JSON}")
    print(f"  総実行時間: {time.time()-t_start:.0f}s  ({run_count}回 run_period)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
