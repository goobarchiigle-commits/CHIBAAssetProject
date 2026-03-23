"""
diagnostics/rsr_period_test.py
RSR期間感度テスト: window=63 vs 42 vs blended(21+63)

目的:
  RSR計算期間を変えることでシグナル供給が改善するか診断する。
  現在 IBD式12ヶ月加重 (実質 ~63日が中心) → 短期成分を強化すると候補が変わるか検証。

実行:
  cd asset_simulation
  python diagnostics/rsr_period_test.py
"""
import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

JST = timezone(timedelta(hours=9))
TODAY = datetime.now(JST).strftime("%Y-%m-%d")

# ------------------------------------------------------------------ #
# RSR計算バリアント
# ------------------------------------------------------------------ #

def calc_rsr_ibd(prices_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """現行: IBD式加重12ヶ月 (直近3ヶ月×40% + 残り×20%×3)"""
    comps = {}
    for sym, p in prices_dict.items():
        r1 = p / p.shift(63)  - 1
        r2 = p.shift(63)  / p.shift(126) - 1
        r3 = p.shift(126) / p.shift(189) - 1
        r4 = p.shift(189) / p.shift(252) - 1
        comps[sym] = 0.4*r1 + 0.2*r2 + 0.2*r3 + 0.2*r4
    comp_df = pd.DataFrame(comps).dropna(how="all")
    rsr = comp_df.rank(axis=1, pct=True) * 100
    return rsr


def calc_rsr_42d(prices_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """42日リターン（2ヶ月モメンタム）"""
    comps = {}
    for sym, p in prices_dict.items():
        comps[sym] = p / p.shift(42) - 1
    comp_df = pd.DataFrame(comps).dropna(how="all")
    rsr = comp_df.rank(axis=1, pct=True) * 100
    return rsr


def calc_rsr_blended(prices_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """ブレンド: IBD×70% + 21日リターン×30%（短期成分強化）"""
    ibd_comps = {}
    st_comps  = {}
    for sym, p in prices_dict.items():
        r1 = p / p.shift(63)  - 1
        r2 = p.shift(63)  / p.shift(126) - 1
        r3 = p.shift(126) / p.shift(189) - 1
        r4 = p.shift(189) / p.shift(252) - 1
        ibd_comps[sym] = 0.4*r1 + 0.2*r2 + 0.2*r3 + 0.2*r4
        st_comps[sym]  = p / p.shift(21) - 1
    ibd_df = pd.DataFrame(ibd_comps).dropna(how="all")
    st_df  = pd.DataFrame(st_comps).dropna(how="all")
    common_idx = ibd_df.index.intersection(st_df.index)
    blended = 0.7 * ibd_df.loc[common_idx] + 0.3 * st_df.loc[common_idx]
    rsr = blended.rank(axis=1, pct=True) * 100
    return rsr


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main():
    # RSRコンテキスト（現行42銘柄）でデータ取得
    cfg_path = Path("configs/rsr_universe_42.csv")
    rsr_syms = list(pd.read_csv(cfg_path)["symbol"])

    uni_path = Path("configs/universe/2026Q1_temporal24.json")
    trade_syms = list(json.loads(uni_path.read_text(encoding="utf-8"))["symbols"].keys())

    print(f"RSRコンテキスト: {len(rsr_syms)}銘柄  取引ユニバース: {len(trade_syms)}銘柄")
    # IBD式RSRはshift(252)が必要 → 2.5年分取得
    from datetime import datetime, timezone, timedelta
    _jst = timezone(timedelta(hours=9))
    end_dt   = datetime.now(_jst).strftime("%Y-%m-%d")
    start_dt = (datetime.now(_jst) - timedelta(days=650)).strftime("%Y-%m-%d")
    print(f"データ取得中 ({start_dt}〜{end_dt})...")

    all_syms = list(dict.fromkeys(rsr_syms + trade_syms))
    raw = yf.download(all_syms, start=start_dt, end=end_dt, progress=False, group_by="ticker")

    prices: dict[str, pd.Series] = {}
    for sym in all_syms:
        try:
            s = raw[sym]["Close"].dropna() if len(all_syms) > 1 else raw["Close"].dropna()
            if len(s) >= 80:
                prices[sym] = s
        except (KeyError, TypeError):
            pass

    rsr_ctx = {s: prices[s] for s in rsr_syms if s in prices}
    print(f"取得成功: {len(rsr_ctx)}銘柄（RSRコンテキスト）")

    # --- 3種類のRSRを計算 ---
    variants = {
        "IBD_63（現行）":  calc_rsr_ibd(rsr_ctx),
        "単純42日":        calc_rsr_42d(rsr_ctx),
        "ブレンド(70%IBD+30%21d)": calc_rsr_blended(rsr_ctx),
    }

    print("\n" + "=" * 70)
    print("  RSR期間感度テスト（コンテキスト: 42銘柄）")
    print("=" * 70)
    print(f"  {'銘柄':<10}", end="")
    for name in variants:
        print(f" {name:>22}", end="")
    print()
    print("  " + "-" * 68)

    results = {}
    for name, rsr_df in variants.items():
        rsr_latest: dict[str, float] = {}
        for sym in trade_syms:
            if sym in rsr_df.columns:
                s = rsr_df[sym].dropna()
                if not s.empty:
                    rsr_latest[sym] = float(s.iloc[-1])
        results[name] = {
            "rsr_values":       {s: round(v, 1) for s, v in rsr_latest.items()},
            "pass_75":          sum(1 for v in rsr_latest.values() if v >= 75),
            "pass_70":          sum(1 for v in rsr_latest.values() if v >= 70),
            "max_rsr":          round(max(rsr_latest.values(), default=0.0), 1),
            "top5":             sorted(
                [{"symbol": s, "rsr": round(v, 1)} for s, v in rsr_latest.items()],
                key=lambda x: x["rsr"], reverse=True
            )[:5],
        }

    # 銘柄別比較表
    all_trade_sym_sorted = sorted(
        trade_syms,
        key=lambda s: results["IBD_63（現行）"]["rsr_values"].get(s, 0.0),
        reverse=True
    )
    for sym in all_trade_sym_sorted[:12]:
        print(f"  {sym:<10}", end="")
        for name, r in results.items():
            v = r["rsr_values"].get(sym, 0.0)
            flag = "★" if v >= 75 else ("△" if v >= 70 else " ")
            print(f" {v:>20.1f}{flag}", end="")
        print()

    print("\n  --- 閾値通過数サマリー ---")
    print(f"  {'閾値':<12}", end="")
    for name in variants:
        print(f" {name:>22}", end="")
    print()
    for thr_name, key in [("RSR≥75", "pass_75"), ("RSR≥70", "pass_70"), ("最高RSR", "max_rsr")]:
        print(f"  {thr_name:<12}", end="")
        for name, r in results.items():
            print(f" {r[key]:>22}", end="")
        print()

    # 診断
    print("\n  --- 診断 ---")
    ibdv = results["IBD_63（現行）"]
    for name, r in results.items():
        if name == "IBD_63（現行）":
            continue
        diff = r["pass_75"] - ibdv["pass_75"]
        if diff > 0:
            print(f"  ✅ {name}: RSR≥75が +{diff}銘柄改善 → 期間調整が有効")
        elif r["max_rsr"] > ibdv["max_rsr"] + 5:
            print(f"  △  {name}: 最高RSR +{r['max_rsr']-ibdv['max_rsr']:.1f}pt（閾値70なら効果あり）")
        else:
            print(f"  ✗  {name}: 改善なし（期間変更は効かない）")

    print("=" * 70)

    # ログ保存
    log_dir = Path("logs/diagnostics")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {"date": TODAY, "variants": results}
    out_path = log_dir / f"rsr_period_test_{TODAY}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 結果保存: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
