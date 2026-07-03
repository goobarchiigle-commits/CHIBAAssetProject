"""
oos_stability_eval.py — _STABILITY_CONSTRAINT 導入前後の OOS 安定性 定量評価

usage:
    python src/tools/oos_stability_eval.py \\
        --before logs/pipeline_before.csv \\
        --after  logs/pipeline_after.csv \\
        [--plot] [--out results.json]

    # テスト用サンプルCSV生成
    python src/tools/oos_stability_eval.py --generate-sample logs/sample

input CSV 必須カラム:
    strategy_id, period, is_oos, sharpe, trades, max_dd, run_id

output:
    - strategy別評価テーブル (stdout)
    - before/after 比較テーブル (stdout)
    - rejection_summary (count + ratio)
    - smoothed rejection ratios → logs/rejection_history.json
    - 改善判定 (True/False)
    - 改善要因の仮説分解
    - 改善=False 時は自動で図を保存 (--plot で強制保存)
    - (--out) JSON 出力
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "MS Gothic"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

# ────────────────────────────────────────────────────────────
# 定数
# ────────────────────────────────────────────────────────────
REQUIRED_COLS = {"strategy_id", "period", "is_oos", "sharpe", "trades", "max_dd", "run_id"}

_BROKEN_OOS_SHARPE_MIN    = 0.0   # OOS Sharpe < 0 → broken
_BROKEN_SHARPE_STD_MAX    = 1.0   # sharpe_std > 1.0 → broken
_BROKEN_TRADES_MIN        = 30    # trades合計 < 30 → broken
_BROKEN_HIGH_SHARPE_MAX   = 2.5   # max(OOS Sharpe) > 2.5 かつ
_BROKEN_HIGH_SHARPE_MEAN  = 0.5   # OOS平均 < 0.5 → broken
_BROKEN_OOS_PERIODS_MIN   = 3     # OOS期間数 < 3 → broken


# ────────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────────
def _fatal(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


# ────────────────────────────────────────────────────────────
# Step 0: 入力検証（最優先）
# ────────────────────────────────────────────────────────────
def validate_input(df: pd.DataFrame, label: str) -> None:
    # 1. 必須カラム欠損チェック
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        _fatal(f"[ERROR] {label}: 必須カラム欠損: {sorted(missing)}")

    # 2. 空データ禁止
    if len(df) == 0:
        _fatal(f"[ERROR] {label}: データが空")

    # 3. strategy_id数 > 0
    if df["strategy_id"].nunique() == 0:
        _fatal(f"[ERROR] {label}: strategy_id 件数=0")

    # 4. OOSデータが空ならエラー
    raw = df["is_oos"]
    if pd.api.types.is_numeric_dtype(raw):
        oos_mask = raw.astype(bool)
    else:
        oos_mask = raw.astype(str).str.lower().isin(["true", "1"])
    if int(oos_mask.sum()) == 0:
        _fatal(f"[ERROR] {label}: OOS データが空 (is_oos=True の行が 0 件)")

    # 5. sharpeのNaN / Infチェック
    sharpe_num = pd.to_numeric(df["sharpe"], errors="coerce")
    nan_n = int(sharpe_num.isna().sum())
    if nan_n > 0:
        _fatal(f"[ERROR] {label}: sharpe に NaN/変換不能値: {nan_n} 件")
    inf_n = int(np.isinf(sharpe_num.fillna(0)).sum())
    if inf_n > 0:
        _fatal(f"[ERROR] {label}: sharpe に Inf: {inf_n} 件")


# ────────────────────────────────────────────────────────────
# Step 1: データ整形
# ────────────────────────────────────────────────────────────
def _parse_bool_col(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().map(
        {"true": True, "1": True, "false": False, "0": False}
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["period"]      = df["period"].astype(str)
    df["run_id"]      = df["run_id"].astype(str)
    df["is_oos"]      = _parse_bool_col(df["is_oos"])
    df["sharpe"]      = pd.to_numeric(df["sharpe"],  errors="coerce")
    df["trades"]      = pd.to_numeric(df["trades"],  errors="coerce")
    df["max_dd"]      = pd.to_numeric(df["max_dd"],  errors="coerce")
    return df


def check_period_consistency(df: pd.DataFrame, label: str) -> None:
    """strategy_id 単位でperiodセットが統一されているか確認"""
    period_sets = df.groupby("strategy_id")["period"].apply(frozenset)
    ref = period_sets.iloc[0]
    bad = period_sets[period_sets != ref]
    if len(bad) > 0:
        _fatal(
            f"[ERROR] {label}: period粒度不一致 — "
            f"{len(bad)} strategy_id が基準セット {sorted(ref)} と異なる\n"
            f"  不一致 strategy_id (先頭5件): {bad.index.tolist()[:5]}"
        )


# ────────────────────────────────────────────────────────────
# Step 2: 指標算出（strategy単位）
# ────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    is_df  = df[~df["is_oos"]]
    oos_df = df[ df["is_oos"]]

    is_sharpe        = is_df.groupby("strategy_id")["sharpe"].mean()
    oos_grp          = oos_df.groupby("strategy_id")
    oos_sharpe_mean  = oos_grp["sharpe"].mean()
    oos_sharpe_max   = oos_grp["sharpe"].max()
    sharpe_std       = oos_grp["sharpe"].std(ddof=1).fillna(0.0)
    trades_total     = oos_grp["trades"].sum()
    max_dd_worst     = oos_grp["max_dd"].min()
    oos_period_count = oos_grp["period"].nunique()   # OOS期間数（< 3 → broken）

    return pd.concat([
        is_sharpe.rename("is_sharpe"),
        oos_sharpe_mean.rename("oos_sharpe_mean"),
        oos_sharpe_max.rename("oos_sharpe_max"),
        sharpe_std.rename("sharpe_std"),
        trades_total.rename("trades_total"),
        max_dd_worst.rename("max_dd_worst"),
        oos_period_count.rename("oos_period_count"),
    ], axis=1).rename_axis("strategy_id").reset_index()


# ────────────────────────────────────────────────────────────
# Step 3: 破綻判定
# ────────────────────────────────────────────────────────────
def apply_broken(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()

    b_neg_sharpe  = agg["oos_sharpe_mean"]  < _BROKEN_OOS_SHARPE_MIN
    b_high_std    = agg["sharpe_std"]        > _BROKEN_SHARPE_STD_MAX
    b_few_trades  = agg["trades_total"]      < _BROKEN_TRADES_MIN
    b_suspicious  = (
        (agg["oos_sharpe_max"]  > _BROKEN_HIGH_SHARPE_MAX) &
        (agg["oos_sharpe_mean"] < _BROKEN_HIGH_SHARPE_MEAN)
    )
    b_few_oos     = agg["oos_period_count"]  < _BROKEN_OOS_PERIODS_MIN

    agg["broken"]   = b_neg_sharpe | b_high_std | b_few_trades | b_suspicious | b_few_oos
    agg["eligible"] = ~agg["broken"]   # eligible=False → pipeline再利用禁止

    reasons_list: list[str] = []
    for _, row in agg.iterrows():
        r: list[str] = []
        if row["oos_sharpe_mean"] < _BROKEN_OOS_SHARPE_MIN:
            r.append("oos_sharpe<0")
        if row["sharpe_std"] > _BROKEN_SHARPE_STD_MAX:
            r.append("std>1.0")
        if row["trades_total"] < _BROKEN_TRADES_MIN:
            r.append("trades<30")
        if (row["oos_sharpe_max"] > _BROKEN_HIGH_SHARPE_MAX and
                row["oos_sharpe_mean"] < _BROKEN_HIGH_SHARPE_MEAN):
            r.append("high_sharpe_low_mean")
        if row["oos_period_count"] < _BROKEN_OOS_PERIODS_MIN:
            r.append("oos_periods<3")
        reasons_list.append(",".join(r) if r else "OK")
    agg["broken_reason"] = reasons_list

    return agg


# ────────────────────────────────────────────────────────────
# rejection_summary: count + ratio
# ────────────────────────────────────────────────────────────
def build_rejection_summary(agg: pd.DataFrame) -> dict[str, dict]:
    """
    破綻条件ごとの件数と割合を集計する。
    pipeline.py / smooth_summary の入力となる。
    """
    total = len(agg)
    if total == 0:
        return {}

    def _cnt(mask: pd.Series) -> dict[str, Any]:
        n = int(mask.sum())
        return {"count": n, "ratio": round(n / total, 4)}

    return {
        "low_sharpe":               _cnt(agg["oos_sharpe_mean"]  < _BROKEN_OOS_SHARPE_MIN),
        "high_variance":            _cnt(agg["sharpe_std"]        > _BROKEN_SHARPE_STD_MAX),
        "low_trades":               _cnt(agg["trades_total"]      < _BROKEN_TRADES_MIN),
        "insufficient_oos_periods": _cnt(agg["oos_period_count"]  < _BROKEN_OOS_PERIODS_MIN),
    }


# ────────────────────────────────────────────────────────────
# フィードバック平滑化（EMA）
# ────────────────────────────────────────────────────────────
def smooth_summary(
    current: dict[str, dict],
    history: dict[str, dict],
    alpha:   float = 0.5,
) -> dict[str, float]:
    """
    rejection_summary を EMA で平滑化する。

    Args:
        current : build_rejection_summary の出力
                  {"key": {"count": int, "ratio": float}}
        history : 前回保存した smoothed ratios
                  {"key": {"ratio": float}}
        alpha   : 平滑化係数 0〜1（高いほど現在値重視）

    Returns:
        {"key": smoothed_ratio}  ← pipeline.py が参照するフォーマット
    """
    smoothed: dict[str, float] = {}
    for k, v in current.items():
        prev = history.get(k, {"ratio": 0.0})["ratio"]
        smoothed[k] = alpha * v["ratio"] + (1 - alpha) * prev
    return smoothed


# ────────────────────────────────────────────────────────────
# 排除履歴 I/O
# ────────────────────────────────────────────────────────────
def _rej_history_path() -> Path:
    from src.paths import LOGS_DIR
    return LOGS_DIR / "rejection_history.json"


def load_rejection_history(path: "Path | None" = None) -> dict[str, dict]:
    """logs/rejection_history.json を読み込む。存在しなければ空dict。"""
    p = path if path is not None else _rej_history_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_rejection_history(
    smoothed: dict[str, float],
    path:     "Path | None" = None,
) -> Path:
    """smoothed ratios を {"key": {"ratio": float}} 形式で保存する。"""
    p = path if path is not None else _rej_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    history = {k: {"ratio": round(float(v), 6)} for k, v in smoothed.items()}
    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ────────────────────────────────────────────────────────────
# Step 4: before/after 比較
# ────────────────────────────────────────────────────────────
def compare(before_agg: pd.DataFrame, after_agg: pd.DataFrame) -> dict[str, Any]:
    def _stats(agg: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_strategies":    int(len(agg)),
            "mean_oos_sharpe": round(float(agg["oos_sharpe_mean"].mean()), 4),
            "mean_sharpe_std": round(float(agg["sharpe_std"].mean()), 4),
            "survival_rate":   round(float((~agg["broken"]).mean()), 4),
            "n_broken":        int(agg["broken"].sum()),
            "n_eligible":      int(agg["eligible"].sum()),
        }
    return {"before": _stats(before_agg), "after": _stats(after_agg)}


# ────────────────────────────────────────────────────────────
# Step 5: 可視化
# ────────────────────────────────────────────────────────────
def plot_results(
    before_agg: pd.DataFrame,
    after_agg:  pd.DataFrame,
    out_dir:    Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("OOS安定性 before/after 比較 (_STABILITY_CONSTRAINT 導入効果)", fontsize=12)

    # ─ IS vs OOS Sharpe 散布図 ─────────────────────────────
    ax = axes[0]
    for label, agg, color, marker in [
        ("before", before_agg, "#4477AA", "o"),
        ("after",  after_agg,  "#EE6677", "^"),
    ]:
        valid_mask  = agg["eligible"]
        ax.scatter(agg.loc[valid_mask,  "is_sharpe"],
                   agg.loc[valid_mask,  "oos_sharpe_mean"],
                   c=color, marker=marker, alpha=0.7, label=f"{label} (eligible)", s=45)
        ax.scatter(agg.loc[~valid_mask, "is_sharpe"],
                   agg.loc[~valid_mask, "oos_sharpe_mean"],
                   c=color, marker="x",  alpha=0.45, label=f"{label} (broken)",   s=45)

    all_is  = pd.concat([before_agg["is_sharpe"],       after_agg["is_sharpe"]]).dropna()
    all_oos = pd.concat([before_agg["oos_sharpe_mean"],  after_agg["oos_sharpe_mean"]]).dropna()
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    if len(all_is) > 0 and len(all_oos) > 0:
        lo = min(float(all_is.min()), float(all_oos.min())) - 0.1
        hi = max(float(all_is.max()), float(all_oos.max())) + 0.1
        ax.plot([lo, hi], [lo, hi], color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("IS Sharpe (期間平均)")
    ax.set_ylabel("OOS Sharpe (期間平均)")
    ax.set_title("IS vs OOS Sharpe")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─ sharpe_std ヒストグラム ──────────────────────────────
    ax2 = axes[1]
    all_std = pd.concat([before_agg["sharpe_std"], after_agg["sharpe_std"]]).dropna()
    hi_std  = max(float(all_std.max()) if len(all_std) > 0 else 2.0,
                  _BROKEN_SHARPE_STD_MAX + 0.5)
    bins    = np.linspace(0.0, hi_std, 20)
    ax2.hist(before_agg["sharpe_std"].dropna(), bins=bins,
             alpha=0.55, color="#4477AA", label="before")
    ax2.hist(after_agg["sharpe_std"].dropna(),  bins=bins,
             alpha=0.55, color="#EE6677", label="after")
    ax2.axvline(_BROKEN_SHARPE_STD_MAX, color="black", linestyle="--",
                linewidth=1.2, label=f"broken閾値 ({_BROKEN_SHARPE_STD_MAX})")
    ax2.set_xlabel("OOS Sharpe std")
    ax2.set_ylabel("戦略数")
    ax2.set_title("sharpe_std 分布 (OOS)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "oos_stability_eval.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ────────────────────────────────────────────────────────────
# Step 6: 改善判定
# ────────────────────────────────────────────────────────────
def is_improved(comp: dict[str, Any]) -> bool:
    b, a = comp["before"], comp["after"]
    return (
        a["mean_sharpe_std"]  < b["mean_sharpe_std"]  and
        a["survival_rate"]    > b["survival_rate"]     and
        a["mean_oos_sharpe"] >= 0.0
    )


# ────────────────────────────────────────────────────────────
# Step 7: 改善要因の仮説分解
# ────────────────────────────────────────────────────────────
def decompose_hypothesis(
    comp:       dict[str, Any],
    before_agg: pd.DataFrame,
    after_agg:  pd.DataFrame,
) -> list[str]:
    b, a = comp["before"], comp["after"]
    hyp: list[str] = []

    std_delta = b["mean_sharpe_std"] - a["mean_sharpe_std"]
    label_e   = "改善" if std_delta > 0 else "悪化/変化なし"
    hyp.append(
        f"[entry構造] sharpe_std: {b['mean_sharpe_std']:.3f}→{a['mean_sharpe_std']:.3f} "
        f"(Δ={std_delta:+.3f}) | {label_e}: "
        f"entry+filter 2層構造が期間間安定性に{'寄与' if std_delta > 0 else '未寄与'}"
    )

    bb = int((before_agg["sharpe_std"] > _BROKEN_SHARPE_STD_MAX).sum())
    ab = int((after_agg["sharpe_std"]  > _BROKEN_SHARPE_STD_MAX).sum())
    label_f   = "削減効果あり" if bb > ab else ("変化なし" if bb == ab else "悪化")
    hyp.append(
        f"[filter条件] std>1.0 破綻: {bb}→{ab}件 (Δ={bb - ab:+d}) | {label_f}: "
        f"ボラ/トレンドフィルタが不安定戦略を{'排除' if bb > ab else '排除できていない'}"
    )

    bn    = int((before_agg["oos_sharpe_mean"] < 0).sum())
    an    = int((after_agg["oos_sharpe_mean"]  < 0).sum())
    oos_d = a["mean_oos_sharpe"] - b["mean_oos_sharpe"]
    label_h = "改善" if (bn > an or oos_d > 0) else "効果軽微"
    hyp.append(
        f"[hold_days制約] OOS Sharpe<0 破綻: {bn}→{an}件, "
        f"OOS mean Δ={oos_d:+.3f} | {label_h}: "
        f"時間exit必須化が損失期間を{'削減' if bn > an else '削減できていない'}"
    )

    return hyp


# ────────────────────────────────────────────────────────────
# 表示ユーティリティ
# ────────────────────────────────────────────────────────────
def _print_agg_table(agg: pd.DataFrame, label: str) -> None:
    _section(f"{label}: strategy別評価テーブル")
    out = agg[[
        "strategy_id", "is_sharpe", "oos_sharpe_mean", "sharpe_std",
        "oos_period_count", "trades_total", "max_dd_worst",
        "eligible", "broken", "broken_reason",
    ]].copy()
    for c in ("is_sharpe", "oos_sharpe_mean", "sharpe_std", "max_dd_worst"):
        out[c] = out[c].round(4)
    out["trades_total"]    = out["trades_total"].fillna(0).astype(int)
    out["oos_period_count"] = out["oos_period_count"].fillna(0).astype(int)
    print(out.to_string(index=False))


def _print_rejection_summary(label: str, rej: dict[str, dict]) -> None:
    _section(f"{label}: rejection_summary")
    rows = [{"condition": k, **v} for k, v in rej.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def _print_compare_table(comp: dict[str, Any]) -> None:
    _section("before/after 比較テーブル")
    keys = [
        "n_strategies", "mean_oos_sharpe", "mean_sharpe_std",
        "survival_rate", "n_broken", "n_eligible",
    ]
    rows = []
    for key in keys:
        bv = comp["before"][key]
        av = comp["after"][key]
        dv = round(av - bv, 4) if isinstance(av, (int, float)) else None
        rows.append({"指標": key, "before": bv, "after": av, "Δ": dv})
    print(pd.DataFrame(rows).to_string(index=False))


def _print_smoothed(smoothed: dict[str, float], weight: float = 0.5) -> None:
    _section(f"smoothed rejection ratios (alpha=0.5, weight={weight})")
    rows = []
    for k, v in smoothed.items():
        rows.append({
            "condition":      k,
            "smoothed_ratio": round(v, 4),
            "effective_ratio": round(weight * v, 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("  ↑ logs/rejection_history.json に保存済み（pipeline.py が参照）")


# ────────────────────────────────────────────────────────────
# テスト用サンプルデータ生成
# ────────────────────────────────────────────────────────────
def generate_sample(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n_strategies = 12
    periods = ["P1", "P2", "P3", "P4", "P5"]

    def _make(run_tag: str, stability_bonus: float) -> pd.DataFrame:
        rows = []
        for sid in range(1, n_strategies + 1):
            base        = rng.uniform(-0.3, 1.8)
            noise_scale = max(0.15, 0.55 - stability_bonus * 0.35)
            for period in periods:
                for is_oos in [False, True]:
                    noise = rng.normal(0, noise_scale)
                    s     = base + noise - (0.15 if is_oos else 0.0)
                    # S01/S02 は取引数を意図的に少なく → trades<30 broken テスト
                    trd_lo = 5  if sid <= 2 else int(15 + stability_bonus * 12)
                    trd_hi = 20 if sid <= 2 else int(55 + stability_bonus * 25)
                    trd    = int(rng.uniform(trd_lo, trd_hi))
                    dd     = round(rng.uniform(-0.35, -0.04), 4)
                    rows.append({
                        "run_id":      run_tag,
                        "strategy_id": f"S{sid:02d}",
                        "period":      period,
                        "is_oos":      is_oos,
                        "sharpe":      round(s, 4),
                        "trades":      trd,
                        "max_dd":      dd,
                    })
        return pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    _make("run_before", 0.0).to_csv(out_dir / "before.csv", index=False)
    _make("run_after",  1.0).to_csv(out_dir / "after.csv",  index=False)
    print(f"[INFO] サンプルCSV生成完了")
    print(f"  before: {out_dir / 'before.csv'}")
    print(f"  after:  {out_dir / 'after.csv'}")
    print(f"\n実行例:")
    print(f"  python src/tools/oos_stability_eval.py "
          f"--before {out_dir}/before.csv --after {out_dir}/after.csv --plot")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOS安定性 before/after 定量評価 (_STABILITY_CONSTRAINT 導入効果)"
    )
    parser.add_argument("--before",          help="before CSV パス")
    parser.add_argument("--after",           help="after CSV パス")
    parser.add_argument("--plot",            action="store_true",
                        help="改善=True 時も強制で図を保存（改善=False は自動保存）")
    parser.add_argument("--out",             default=None,
                        help="結果を JSON で保存するパス")
    parser.add_argument("--generate-sample", metavar="OUTDIR",
                        help="テスト用 before/after CSV を生成して終了")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample(Path(args.generate_sample))
        return

    if not args.before or not args.after:
        _fatal("[ERROR] --before と --after は必須 (テスト: --generate-sample OUTDIR)")

    before_path = Path(args.before)
    after_path  = Path(args.after)
    for p in (before_path, after_path):
        if not p.exists():
            _fatal(f"[ERROR] ファイルが見つかりません: {p}")

    # ── Load ────────────────────────────────────────────────
    before_raw = pd.read_csv(before_path)
    after_raw  = pd.read_csv(after_path)

    # ── Step 0: 入力検証 ────────────────────────────────────
    validate_input(before_raw, "before")
    validate_input(after_raw,  "after")

    # ── Step 1: 整形 + period粒度確認 ───────────────────────
    before_df = prepare(before_raw)
    after_df  = prepare(after_raw)
    check_period_consistency(before_df, "before")
    check_period_consistency(after_df,  "after")

    # ── Step 2 + 3: 集約 + 破綻判定 ─────────────────────────
    before_agg = apply_broken(aggregate(before_df))
    after_agg  = apply_broken(aggregate(after_df))

    # strategy別評価テーブル出力
    _print_agg_table(before_agg, "before")
    _print_agg_table(after_agg,  "after")

    # ── rejection_summary (count + ratio) ───────────────────
    before_rej = build_rejection_summary(before_agg)
    after_rej  = build_rejection_summary(after_agg)
    _print_rejection_summary("before", before_rej)
    _print_rejection_summary("after",  after_rej)

    # ── フィードバック平滑化 + 履歴保存 ────────────────────────
    rej_history = load_rejection_history()
    smoothed    = smooth_summary(after_rej, rej_history, alpha=0.5)
    hist_path   = save_rejection_history(smoothed)
    _print_smoothed(smoothed, weight=0.5)
    print(f"[INFO] rejection_history 保存: {hist_path}")

    # ── Step 4: before/after 比較 ───────────────────────────
    comp = compare(before_agg, after_agg)
    _print_compare_table(comp)

    # ── Step 6: 改善判定 ────────────────────────────────────
    improved = is_improved(comp)
    b, a = comp["before"], comp["after"]
    _section("改善判定")
    verdict = "True（改善あり）" if improved else "False（改善なし）"
    print(f"  結果: {verdict}")
    print(f"  条件チェック:")
    print(f"    mean_sharpe_std 低下: "
          f"{b['mean_sharpe_std']:.4f} → {a['mean_sharpe_std']:.4f} "
          f"= {a['mean_sharpe_std'] < b['mean_sharpe_std']}")
    print(f"    survival_rate 上昇:  "
          f"{b['survival_rate']:.4f} → {a['survival_rate']:.4f} "
          f"= {a['survival_rate'] > b['survival_rate']}")
    print(f"    mean_oos_sharpe >= 0: {a['mean_oos_sharpe']:.4f} "
          f"= {a['mean_oos_sharpe'] >= 0.0}")

    # ── Step 7: 仮説分解 ────────────────────────────────────
    hypotheses = decompose_hypothesis(comp, before_agg, after_agg)
    _section("改善要因の仮説分解")
    for h in hypotheses:
        print(f"  {h}")

    # ── Step 5: 可視化 ──────────────────────────────────────
    # 改善=False → 自動保存 / 改善=True → --plot 時のみ保存
    should_plot = (not improved) or args.plot
    plot_path: "Path | None" = None
    if should_plot:
        from src.paths import DOCS_DIR
        plot_path = plot_results(before_agg, after_agg, DOCS_DIR)
        reason = "自動保存（改善なし）" if not improved else "手動保存（--plot）"
        print(f"\n[INFO] 図{reason}: {plot_path}")

    # ── JSON 出力 ────────────────────────────────────────────
    if args.out:
        result: dict[str, Any] = {
            "comparison":          comp,
            "improved":            improved,
            "hypotheses":          hypotheses,
            "before_rejection":    before_rej,
            "after_rejection":     after_rej,
            "smoothed_rejection":  smoothed,
            "before_strategies":   before_agg.to_dict(orient="records"),
            "after_strategies":    after_agg.to_dict(orient="records"),
        }
        if plot_path:
            result["plot_path"] = str(plot_path)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[INFO] JSON保存: {out_path}")

    print(f"\n[完了] 改善判定: {improved}")


if __name__ == "__main__":
    main()
