"""
pipeline.py
data/sample.csv を読み込み、日次リターンを計算し、
trade_regime を付与して出力する。
"""
import sys
import json
import contextlib
import io
import importlib.util
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    sys.exit(f"[ERROR] pandas が見つかりません: {e}")

# ── monitoring モジュール（インポート失敗は警告のみ・実行継続） ────────────────
try:
    from src.monitoring.audit import compute_metrics as _compute_metrics
    from src.monitoring.controller import (
        apply_controller as _apply_controller,
        get_system_status as _get_system_status,
        load_state as _load_state,
        save_state as _save_state,
    )
    from src.monitoring.gate import pre_trade_gate as _pre_trade_gate
    _MON_AVAILABLE = True
except ImportError:
    _MON_AVAILABLE = False

# ── パス定義 ──────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "data" / "sample.csv"
OUT_CSV   = BASE_DIR / "data" / "trades_log.csv"
LOG_DIR   = BASE_DIR / "logs"
OUT_JSON  = LOG_DIR / "trade_regime_distribution.json"

# ── レジーム閾値（変更禁止） ────────────────────────
_UP_THRESHOLD   =  0.02
_DOWN_THRESHOLD = -0.02

# ── 執行コスト ────────────────────────────────────
_COMMISSION = 0.00055
_SLIPPAGE   = 0.0005

# ── 拡張ユニバース（Nikkei225 / TOPIX100 主要銘柄）────
_NIKKEI_EXTENDED: tuple[str, ...] = (
    "1605.T", "1721.T", "1801.T", "1802.T", "1803.T", "1812.T", "1925.T", "1928.T",
    "2002.T", "2269.T", "2282.T", "2502.T", "2503.T", "2801.T", "2802.T",
    "3099.T", "3289.T", "3402.T", "3407.T", "3861.T",
    "4004.T", "4063.T", "4188.T", "4307.T", "4452.T", "4503.T", "4507.T",
    "4519.T", "4523.T", "4543.T", "4568.T", "4578.T", "4661.T", "4901.T", "4911.T",
    "5020.T", "5108.T", "5201.T", "5333.T", "5713.T", "5714.T",
    "6098.T", "6178.T", "6201.T", "6273.T", "6301.T", "6326.T", "6367.T",
    "6471.T", "6503.T", "6504.T", "6586.T", "6701.T", "6724.T", "6752.T",
    "6758.T", "6841.T", "6902.T", "6954.T", "6963.T", "6971.T", "6981.T",
    "7011.T", "7013.T", "7202.T", "7267.T", "7270.T",
    "7731.T", "7733.T", "7741.T", "7751.T", "7752.T", "7974.T",
    "8001.T", "8002.T", "8015.T", "8031.T", "8053.T", "8058.T",
    "8113.T", "8233.T", "8267.T", "8304.T", "8308.T", "8316.T",
    "8411.T", "8601.T", "8604.T", "8630.T", "8725.T", "8750.T", "8766.T", "8795.T",
    "9020.T", "9021.T", "9022.T", "9064.T", "9101.T", "9104.T", "9107.T",
    "9202.T", "9301.T", "9433.T", "9434.T",
    "9501.T", "9502.T", "9503.T", "9531.T", "9532.T", "9613.T", "9984.T",
)


def label_trade_regime(ret: float) -> str:
    if ret > _UP_THRESHOLD:
        return "up"
    if ret < _DOWN_THRESHOLD:
        return "down"
    return "range"


def main() -> None:
    # ── monitoring: 冒頭ゲートチェック ────────────────────────────────────────
    _mon_state: dict = {}
    if _MON_AVAILABLE:
        _mon_state = _load_state()
        _prev_metrics = _mon_state.get("last_metrics") or {}
        _gate_ok, _gate_reason = _pre_trade_gate(_mon_state, _prev_metrics)
        if not _gate_ok:
            sys.exit(f"[MONITORING] GATE BLOCK: {_gate_reason}")
        print(f"[MONITORING] gate=OK  mode={_mon_state.get('mode', '?')}"
              f"  halted={_mon_state.get('halted', False)}")

    # 1. 入力ファイル確認
    if not INPUT_CSV.exists():
        sys.exit(f"[ERROR] 入力ファイルが見つかりません: {INPUT_CSV}")

    # 2. CSV 読み込み
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        sys.exit(f"[ERROR] CSV 読み込み失敗: {e}")

    # 3. スキーマ検証
    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] 必須列が不足: {missing}")

    if df.empty:
        sys.exit("[ERROR] 入力データが空です")

    # 4. 日付ソート
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        sys.exit(f"[ERROR] date 列のパース失敗: {e}")
    df = df.sort_values("date").reset_index(drop=True)

    # 5. close を数値変換
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().all():
        sys.exit("[ERROR] close 列に有効な数値がありません")

    # 5b. 正規化用ローリング標準偏差（window=20）
    df["_ret_raw"]     = df["close"].pct_change()
    df["_rolling_std"] = df["_ret_raw"].rolling(window=20).std()

    # 6. トレード生成（1期間保有: entry=current row, exit=next row）
    # filter1: abs(ret) < 0.001 はスキップ（微小トレード除外）
    # filter2: 前トレードのexit日 == 今のentry日 はスキップ（同日再エントリー禁止）
    raw: list[dict] = []
    for i in range(len(df) - 1):
        if i % 2 != 0:
            continue
        ep  = df["close"].iloc[i]
        xp  = df["close"].iloc[i + 1]
        ed  = df["date"].iloc[i]
        xd  = df["date"].iloc[i + 1]
        ret = (xp / ep) - 1
        if abs(ret) < 0.001:
            continue
        rs  = df["_rolling_std"].iloc[i]
        z   = float(ret / rs) if (pd.notna(rs) and rs > 0) else 0.0
        raw.append({"entry_date": ed, "exit_date": xd,
                    "entry_price": ep, "exit_price": xp, "ret": ret, "z": z})

    trades = pd.DataFrame(raw)

    if trades.empty:
        sys.exit("[ERROR] トレードデータが生成されませんでした")

    # 7. trade_regime 付与（既存: 絶対閾値 ±2%）
    trades["trade_regime"] = trades["ret"].apply(label_trade_regime)

    # 7b. z_regime 付与（並行: 正規化 z-score 閾値 ±1σ）
    def _label_z_regime(z: float) -> str:
        if z > 1:  return "up"
        if z < -1: return "down"
        return "range"

    trades["z_regime"] = trades["z"].apply(_label_z_regime)

    # 8. CSV 出力
    try:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        trades[["entry_date", "exit_date", "entry_price", "exit_price", "ret", "trade_regime", "z", "z_regime"]].to_csv(OUT_CSV, index=False)
        print(f"[OUTPUT] {OUT_CSV}  ({len(trades)} trades)")
    except Exception as e:
        sys.exit(f"[ERROR] CSV 書き込み失敗: {e}")

    # 9. 分布計算
    dist = trades["trade_regime"].value_counts(normalize=True)
    dist_dict = {
        "up":    round(float(dist.get("up",    0.0)), 4),
        "down":  round(float(dist.get("down",  0.0)), 4),
        "range": round(float(dist.get("range", 0.0)), 4),
    }

    # 10. JSON 出力
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(dist_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OUTPUT] {OUT_JSON}")
    except Exception as e:
        sys.exit(f"[ERROR] JSON 書き込み失敗: {e}")

    # 11. サマリー表示
    cum    = trades["ret"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    skew   = trades["ret"].skew()
    kurt   = trades["ret"].kurtosis()

    wins   = trades[trades["ret"] > 0]["ret"]
    losses = trades[trades["ret"] < 0]["ret"]
    win_rate      = len(wins) / len(trades)
    avg_win       = wins.mean()  if len(wins)   > 0 else 0.0
    avg_loss      = losses.mean() if len(losses) > 0 else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    print(f"\n[SUMMARY]")
    print(f"  trades        : {len(trades)}")
    print(f"  ret_mean      : {trades['ret'].mean():.4f}")
    print(f"  ret_std       : {trades['ret'].std():.4f}")
    print(f"  win_rate      : {win_rate:.2%}")
    print(f"  avg_win       : {avg_win:.4f}")
    print(f"  avg_loss      : {avg_loss:.4f}")
    print(f"  profit_factor : {profit_factor:.3f}")
    print(f"  skewness      : {skew:.3f}")
    print(f"  kurtosis      : {kurt:.3f}")
    print(f"  max_drawdown  : {max_dd:.4f}")
    print(f"  up            : {dist_dict['up']:.2%}")
    print(f"  down          : {dist_dict['down']:.2%}")
    print(f"  range         : {dist_dict['range']:.2%}")

    imbalance = any(v > 0.8 for v in dist_dict.values())
    if imbalance:
        print(f"\n[WARN] trade_regime 偏り検出（いずれかが80%超）")

    # 12. レジーム別グループ集計
    print(f"\n[REGIME BREAKDOWN]")
    for regime in ["up", "down", "range"]:
        g = trades[trades["trade_regime"] == regime]["ret"]
        if g.empty:
            print(f"  {regime:5s} : no trades")
            continue
        g_wins   = g[g > 0]
        g_losses = g[g < 0]
        g_wr     = len(g_wins) / len(g)
        g_avg    = g.mean()
        g_pf     = (g_wins.sum() / abs(g_losses.sum())) if g_losses.sum() != 0 else float("inf")
        print(f"  {regime:5s} : count={len(g):3d}  win_rate={g_wr:.0%}  avg_ret={g_avg:+.4f}  profit_factor={g_pf:.3f}")

    # 12b. z_regime 別集計
    z_dist = trades["z_regime"].value_counts(normalize=True)
    print(f"\n[Z_REGIME BREAKDOWN]  (z=ret/rolling_std_20, thresholds +-1)")
    for regime in ["up", "down", "range"]:
        g = trades[trades["z_regime"] == regime]["ret"]
        if g.empty:
            print(f"  {regime:5s} : no trades")
            continue
        g_wins   = g[g > 0]
        g_losses = g[g < 0]
        g_wr     = len(g_wins) / len(g)
        g_avg    = g.mean()
        g_pf     = (g_wins.sum() / abs(g_losses.sum())) if g_losses.sum() != 0 else float("inf")
        share    = float(z_dist.get(regime, 0.0))
        print(f"  {regime:5s} : count={len(g):3d}  share={share:.0%}  win_rate={g_wr:.0%}  avg_ret={g_avg:+.4f}  profit_factor={g_pf:.3f}")

    # 12c. z_regime 遷移分析
    trades["prev_z_regime"] = trades["z_regime"].shift(1)
    trans = trades.dropna(subset=["prev_z_regime"])
    tg = trans.groupby(["prev_z_regime", "z_regime"])["ret"].agg(count="count", mean_ret="mean")
    print(f"\n[Z_REGIME TRANSITIONS]  (prev -> current)")
    print(f"  {'from':5s} -> {'to':5s} : {'count':>5s}  {'mean_ret':>9s}")
    print(f"  {'-'*5} -> {'-'*5} : {'-'*5}  {'-'*9}")
    for (frm, to), row in tg.iterrows():
        print(f"  {frm:5s} -> {to:5s} : {int(row['count']):>5d}  {row['mean_ret']:>+9.4f}")

    # 12d. range PF 分解（outlier vs 真の非対称性）
    abs_rng = trades.loc[trades["trade_regime"] == "range", "ret"]
    z_rng   = trades.loc[trades["z_regime"]     == "range", "ret"]
    da, dz  = _pf_decompose(abs_rng), _pf_decompose(z_rng)
    print(f"\n[RANGE PF DECOMPOSITION]")
    print(f"  {'metric':22s} | {'abs':>9s} | {'z':>9s}")
    print(f"  {'-'*22}-+-{'-'*9}-+-{'-'*9}")
    for label, va, vz in [
        ("count",              f"{int(da['n']):>9d}",          f"{int(dz['n']):>9d}"),
        ("win_rate",           f"{da['win_rate']:>9.0%}",       f"{dz['win_rate']:>9.0%}"),
        ("avg_win",            f"{da['avg_win']:>+9.4f}",       f"{dz['avg_win']:>+9.4f}"),
        ("avg_loss",           f"{da['avg_loss']:>+9.4f}",      f"{dz['avg_loss']:>+9.4f}"),
        ("median_win",         f"{da['med_win']:>+9.4f}",       f"{dz['med_win']:>+9.4f}"),
        ("median_loss",        f"{da['med_loss']:>+9.4f}",      f"{dz['med_loss']:>+9.4f}"),
        ("top5%_profit_share", f"{da['top5_contrib']:>9.1%}",   f"{dz['top5_contrib']:>9.1%}"),
        ("bot5%_loss_share",   f"{da['bot5_contrib']:>9.1%}",   f"{dz['bot5_contrib']:>9.1%}"),
    ]:
        print(f"  {label:22s} | {va} | {vz}")

    # 12e. range win_rate 有意性検定（二項検定 + Wilson CI 95%）
    print(f"\n[WIN RATE SIGNIFICANCE]  H0: win_rate=50%  (binomial test, Wilson 95% CI)")
    print(f"  {'regime':6s} | {'wins':>5s} | {'losses':>6s} | {'win_rate':>9s} | {'p_value':>8s} | {'Wilson_95%_CI':>15s}")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*15}")
    for label, g in [("abs", abs_rng), ("z", z_rng)]:
        s   = _winrate_sig(g)
        sig = " *" if s["p_value"] < 0.05 else "  "
        print(f"  {label:6s} | {s['win_count']:>5d} | {s['loss_count']:>6d} | "
              f"{s['win_rate']:>9.1%} | {s['p_value']:>8.4f}{sig}| "
              f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")

    # 12f. モメンタム・ベースライン・シグナル（long: ret_prev>0 / short: ret_prev<0）
    trades["ret_prev"] = trades["ret"].shift(1)
    trades["sig_mom"]  = trades["ret_prev"].apply(
        lambda r: 1 if r > 0 else (-1 if r < 0 else 0))
    st            = trades[trades["sig_mom"] != 0].copy()
    st["ret_sig"] = st["sig_mom"] * st["ret"]
    st["ret_net"] = st["ret_sig"] - (_COMMISSION + _SLIPPAGE)
    import numpy as _np
    _rand_slip        = _np.random.default_rng(0).normal(_SLIPPAGE, _SLIPPAGE, size=len(st))
    st["ret_net_u"]   = st["ret_sig"] - (_COMMISSION + _rand_slip)
    _mu_ln            = _np.log(_SLIPPAGE) - 0.5 ** 2 / 2   # E[X]=_SLIPPAGE at sigma=0.5
    _adv_slip         = _np.random.default_rng(1).lognormal(_mu_ln, 0.5, size=len(st))
    st["ret_net_adv"] = st["ret_sig"] - (_COMMISSION + _adv_slip)
    # state-dependent slippage: regime flags
    _rs_roll  = st["ret"].rolling(window=10, min_periods=1).std()
    _high_vol = (_rs_roll > _rs_roll.median()).values
    _equity   = st["ret_sig"].cumsum()
    _in_dd    = (_equity < _equity.cummax()).values
    # base: LogNormal(mean=_SLIPPAGE, sigma=0.3), then regime multipliers
    _mu_st    = _np.log(_SLIPPAGE) - 0.3 ** 2 / 2
    _slip_st  = _np.random.default_rng(2).lognormal(_mu_st, 0.3, size=len(st))
    _slip_st[_high_vol] *= 1.5
    _slip_st[_in_dd]    *= 2.0
    # clustering: prev top-10% slip → current ×1.5
    _p90     = _np.percentile(_slip_st, 90)
    _prev_top = _np.concatenate([[False], _slip_st[:-1] >= _p90])
    _slip_st[_prev_top] *= 1.5
    st["ret_net_state"] = st["ret_sig"] - (_COMMISSION + _slip_st)
    _pf_f = lambda g: float(g[g > 0].sum() / abs(g[g < 0].sum())) if g[g < 0].sum() != 0 else float("inf")
    _dd_f = lambda g: float((g.cumsum() - g.cumsum().cummax()).min())
    pf_g, ws_g, ts_g = _pf_f(st["ret_sig"]),       _winrate_sig(st["ret_sig"]),       _sig(st["ret_sig"])
    pf_n, ws_n, ts_n = _pf_f(st["ret_net"]),       _winrate_sig(st["ret_net"]),       _sig(st["ret_net"])
    pf_u, ws_u, ts_u = _pf_f(st["ret_net_u"]),     _winrate_sig(st["ret_net_u"]),     _sig(st["ret_net_u"])
    pf_a, ws_a, ts_a = _pf_f(st["ret_net_adv"]),   _winrate_sig(st["ret_net_adv"]),   _sig(st["ret_net_adv"])
    pf_s, ws_s, ts_s = _pf_f(st["ret_net_state"]), _winrate_sig(st["ret_net_state"]), _sig(st["ret_net_state"])
    cost = _COMMISSION + _SLIPPAGE

    print(f"\n[MOMENTUM SIGNAL]  rule=sign(ret_prev)  n={len(st)}/{len(trades)}  cost={cost:.3%}/trade")
    print(f"  {'metric':26s} | {'gross':>12s} | {'net':>12s} | {'net+slip':>12s} | {'net+adv':>12s} | {'net+state':>12s}")
    print(f"  {'-'*26}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    mk_g = (" *" if ws_g["p_value"] < 0.05 else "  ", " *" if ts_g["ci_excl_0"] else "  ")
    mk_n = (" *" if ws_n["p_value"] < 0.05 else "  ", " *" if ts_n["ci_excl_0"] else "  ")
    mk_u = (" *" if ws_u["p_value"] < 0.05 else "  ", " *" if ts_u["ci_excl_0"] else "  ")
    mk_a = (" *" if ws_a["p_value"] < 0.05 else "  ", " *" if ts_a["ci_excl_0"] else "  ")
    mk_s = (" *" if ws_s["p_value"] < 0.05 else "  ", " *" if ts_s["ci_excl_0"] else "  ")
    print(f"  {'win / loss count':26s} | {ws_g['win_count']:>5d} / {ws_g['loss_count']:<5d} | {ws_n['win_count']:>5d} / {ws_n['loss_count']:<5d} | {ws_u['win_count']:>5d} / {ws_u['loss_count']:<5d} | {ws_a['win_count']:>5d} / {ws_a['loss_count']:<5d} | {ws_s['win_count']:>5d} / {ws_s['loss_count']:<5d}")
    print(f"  {'win_rate':26s} | {ws_g['win_rate']:>12.1%} | {ws_n['win_rate']:>12.1%} | {ws_u['win_rate']:>12.1%} | {ws_a['win_rate']:>12.1%} | {ws_s['win_rate']:>12.1%}")
    print(f"  {'p_value (binom H0=50%)':26s} | {ws_g['p_value']:>12.4f}{mk_g[0]}| {ws_n['p_value']:>12.4f}{mk_n[0]}| {ws_u['p_value']:>12.4f}{mk_u[0]}| {ws_a['p_value']:>12.4f}{mk_a[0]}| {ws_s['p_value']:>12.4f}{mk_s[0]}")
    print(f"  {'Wilson 95% CI':26s} | [{ws_g['ci_lo']:.3f},{ws_g['ci_hi']:.3f}] | [{ws_n['ci_lo']:.3f},{ws_n['ci_hi']:.3f}] | [{ws_u['ci_lo']:.3f},{ws_u['ci_hi']:.3f}] | [{ws_a['ci_lo']:.3f},{ws_a['ci_hi']:.3f}] | [{ws_s['ci_lo']:.3f},{ws_s['ci_hi']:.3f}]")
    print(f"  {'profit_factor':26s} | {pf_g:>12.3f} | {pf_n:>12.3f} | {pf_u:>12.3f} | {pf_a:>12.3f} | {pf_s:>12.3f}")
    print(f"  {'avg_ret':26s} | {st['ret_sig'].mean():>+12.4f} | {st['ret_net'].mean():>+12.4f} | {st['ret_net_u'].mean():>+12.4f} | {st['ret_net_adv'].mean():>+12.4f} | {st['ret_net_state'].mean():>+12.4f}")
    print(f"  {'t_stat':26s} | {ts_g['t_stat']:>+12.3f}{mk_g[1]}| {ts_n['t_stat']:>+12.3f}{mk_n[1]}| {ts_u['t_stat']:>+12.3f}{mk_u[1]}| {ts_a['t_stat']:>+12.3f}{mk_a[1]}| {ts_s['t_stat']:>+12.3f}{mk_s[1]}")
    print(f"  {'CI_excl_zero (boot 95%)':26s} | {'yes' if ts_g['ci_excl_0'] else 'no':>12s} | {'yes' if ts_n['ci_excl_0'] else 'no':>12s} | {'yes' if ts_u['ci_excl_0'] else 'no':>12s} | {'yes' if ts_a['ci_excl_0'] else 'no':>12s} | {'yes' if ts_s['ci_excl_0'] else 'no':>12s}")
    print(f"  {'max_drawdown':26s} | {_dd_f(st['ret_sig']):>+12.4f} | {_dd_f(st['ret_net']):>+12.4f} | {_dd_f(st['ret_net_u']):>+12.4f} | {_dd_f(st['ret_net_adv']):>+12.4f} | {_dd_f(st['ret_net_state']):>+12.4f}")

    # 13. 閾値感度分析
    print(f"\n[THRESHOLD SENSITIVITY]")
    print(f"  {'thresh':>6s} | {'up':>6s} | {'down':>6s} | {'range':>6s} | {'rng_pf':>8s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for t in [0.01, 0.02, 0.03]:
        t_up   = trades["ret"].apply(lambda r: "up"    if r >  t else ("down" if r < -t else "range"))
        cnt_u  = (t_up == "up").sum()
        cnt_d  = (t_up == "down").sum()
        cnt_r  = (t_up == "range").sum()
        rng    = trades.loc[t_up == "range", "ret"]
        rng_w  = rng[rng > 0].sum()
        rng_l  = rng[rng < 0].sum()
        rng_pf = (rng_w / abs(rng_l)) if rng_l != 0 else float("inf")
        print(f"  {'+-'+f'{t:.2f}':>6s} | {cnt_u:>6d} | {cnt_d:>6d} | {cnt_r:>6d} | {rng_pf:>8.3f}")

    # ── monitoring: 末尾メトリクス計算 → controller 適用 → state 保存 ──────────
    if _MON_AVAILABLE:
        # trades DataFrame を監視標準フォーマットへ変換
        _mon_df = trades[["ret"]].copy().rename(columns={"ret": "pnl"})
        _mon_df["expected_risk"] = _SLIPPAGE + _COMMISSION   # コストをリスク代理値として使用
        _mon_df["slippage_real"] = _SLIPPAGE
        _mon_df["slippage_est"]  = _SLIPPAGE
        _metrics = _compute_metrics(_mon_df)
        _mon_state = _apply_controller(_mon_state, _metrics)
        _save_state(_mon_state)
        _status = _get_system_status(_mon_state)
        print(
            f"\n[MONITORING] status={_status}"
            f"  trades={_metrics['n_trades']}"
            f"  max_dd={_metrics['max_dd']:.4f}"
            f"  risk_ratio={_metrics['risk_ratio']:.3f}"
            f"  slip_diff={_metrics['slip_diff']:.5f}"
            f"  global_multiplier={_mon_state.get('global_multiplier', 1.0):.3f}"
        )
        if _mon_state.get("halted"):
            print(f"[MONITORING] HALT: {_mon_state.get('halt_reason')}")


def _cap_positions(
    df: "pd.DataFrame",
    hold_n: int = 5,
    max_pos: int = 10,
) -> "pd.DataFrame":
    """Filter date-sorted trades to enforce max concurrent open positions."""
    import pandas as _pd
    _bday = _pd.tseries.offsets.BusinessDay(hold_n)
    kept: list[int] = []
    open_exits: list[_pd.Timestamp] = []
    for idx, row in df.iterrows():
        entry = _pd.Timestamp(row["date"])
        open_exits = [ex for ex in open_exits if ex > entry]
        if len(open_exits) < max_pos:
            kept.append(idx)
            open_exits.append(entry + _bday)
    return df.loc[kept].reset_index(drop=True)


def _portfolio_sim(
    dates: "pd.Series",
    rets_net: "pd.Series",
    capital: float = 3_000_000,
    fraction: float = 0.10,
) -> dict:
    import numpy as _np
    arr     = rets_net.reset_index(drop=True).values.astype(float)
    dates_s = pd.to_datetime(dates.reset_index(drop=True))
    n       = len(arr)
    if n < 2:
        nan = float("nan")
        return {"cagr": nan, "max_dd": nan, "sharpe": nan, "n_trades": n,
                "final_equity": float(capital)}
    eq    = _np.empty(n + 1)
    eq[0] = capital
    for i, r in enumerate(arr):
        eq[i + 1] = eq[i] * (1.0 + fraction * r)
    years  = (dates_s.iloc[-1] - dates_s.iloc[0]).days / 365.25
    cagr   = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    peak   = _np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    r_trd  = fraction * arr
    tpy    = n / years if years > 0 else float("nan")
    sh_den = r_trd.std(ddof=1)
    sharpe = float(r_trd.mean() / sh_den * _np.sqrt(tpy)) if sh_den > 0 else float("nan")
    return {
        "cagr": float(cagr), "max_dd": float(max_dd), "sharpe": float(sharpe),
        "n_trades": n, "trd_per_yr": float(tpy), "final_equity": float(eq[-1]),
    }


def _pf_decompose(g: "pd.Series") -> dict:
    import numpy as _np
    wins   = g[g > 0]
    losses = g[g < 0]
    n = len(g)
    if n == 0:
        return {k: float("nan") for k in
                ["n","win_rate","avg_win","avg_loss","med_win","med_loss","top5_contrib","bot5_contrib"]}
    tot_profit = wins.sum()
    tot_loss   = abs(losses.sum())
    top5_n = max(1, int(len(wins)   * 0.05))
    bot5_n = max(1, int(len(losses) * 0.05))
    top5 = (wins.nlargest(top5_n).sum()     / tot_profit) if tot_profit > 0 else float("nan")
    bot5 = (abs(losses.nsmallest(bot5_n).sum()) / tot_loss)  if tot_loss   > 0 else float("nan")
    return {
        "n":           n,
        "win_rate":    len(wins) / n,
        "avg_win":     float(wins.mean())    if len(wins)   > 0 else 0.0,
        "avg_loss":    float(losses.mean())  if len(losses) > 0 else 0.0,
        "med_win":     float(wins.median())  if len(wins)   > 0 else 0.0,
        "med_loss":    float(losses.median()) if len(losses) > 0 else 0.0,
        "top5_contrib": float(top5),
        "bot5_contrib": float(bot5),
    }


def _winrate_sig(g: "pd.Series") -> dict:
    import numpy as _np
    wins   = int((g > 0).sum())
    losses = int((g < 0).sum())
    n      = len(g)
    if n == 0:
        nan = float("nan")
        return {"win_count": 0, "loss_count": 0, "win_rate": nan,
                "p_value": nan, "ci_lo": nan, "ci_hi": nan}
    wr = wins / n
    # Wilson CI 95%
    z      = 1.96
    denom  = 1 + z ** 2 / n
    center = (wr + z ** 2 / (2 * n)) / denom
    margin = z * _np.sqrt(wr * (1 - wr) / n + z ** 2 / (4 * n ** 2)) / denom
    ci_lo, ci_hi = float(center - margin), float(center + margin)
    # 二項検定 H0: p=0.5
    try:
        from scipy.stats import binomtest as _btest
        pval = float(_btest(wins, n, p=0.5, alternative="two-sided").pvalue)
    except ImportError:
        from math import erfc, sqrt
        z_stat = (wins - 0.5 * n) / _np.sqrt(0.25 * n)
        pval   = float(erfc(abs(z_stat) / sqrt(2)))
    return {"win_count": wins, "loss_count": losses, "win_rate": wr,
            "p_value": pval, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _sig(g: "pd.Series", n_boot: int = 1000, rng_seed: int = 0) -> dict:
    import numpy as _np
    arr = g.dropna().values.astype(float)
    n   = len(arr)
    if n < 2:
        return {"t_stat": float("nan"), "ci_excl_0": False}
    t     = float(arr.mean() / (arr.std(ddof=1) / _np.sqrt(n)))
    rng   = _np.random.default_rng(rng_seed)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = float(_np.percentile(boots, 2.5)), float(_np.percentile(boots, 97.5))
    return {"t_stat": t, "ci_excl_0": bool(ci_lo > 0 or ci_hi < 0)}


_SWEEP_SEEDS = (1, 7, 13, 42, 99)


def sweep(seeds: tuple = _SWEEP_SEEDS, noise: float = 0.003) -> dict:
    import numpy as _np

    _spec = importlib.util.spec_from_file_location(
        "gen_sample_data", BASE_DIR / "scripts" / "gen_sample_data.py"
    )
    _gen = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gen)

    def _pf(g: "pd.Series") -> float:
        w, l = g[g > 0].sum(), g[g < 0].sum()
        return float(w / abs(l)) if l != 0 else float("inf")

    rows: list[dict] = []
    print(f"[SWEEP]  seeds={list(seeds)}  noise={noise:.1%}")
    print(f"  {'seed':>4s} | {'trades':>6s} | {'range_pf':>8s} | {'z_range_pf':>10s} | {'max_dd':>8s}")
    print(f"  {'----':>4s}-+-{'------':>6s}-+-{'--------':>8s}-+-{'----------':>10s}-+-{'--------':>8s}")
    for seed in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            _gen.generate(seed, noise)
            main()
        t   = pd.read_csv(OUT_CSV)
        cum = t["ret"].cumsum()
        sig_a = _sig(t.loc[t["trade_regime"] == "range", "ret"], rng_seed=seed)
        sig_z = _sig(t.loc[t["z_regime"]     == "range", "ret"], rng_seed=seed)
        r   = {
            "seed":            seed,
            "trades":          len(t),
            "range_pf":        _pf(t.loc[t["trade_regime"] == "range", "ret"]),
            "z_range_pf":      _pf(t.loc[t["z_regime"]     == "range", "ret"]),
            "max_dd":          float((cum - cum.cummax()).min()),
            "range_t":         sig_a["t_stat"],
            "range_ci_excl":   sig_a["ci_excl_0"],
            "z_range_t":       sig_z["t_stat"],
            "z_range_ci_excl": sig_z["ci_excl_0"],
        }
        rows.append(r)
        print(f"  {seed:>4d} | {r['trades']:>6d} | {r['range_pf']:>8.3f} | {r['z_range_pf']:>10.3f} | {r['max_dd']:>+8.4f}")

    metrics = ["trades", "range_pf", "z_range_pf", "max_dd"]
    print(f"\n[SWEEP SUMMARY]  n={len(seeds)}")
    print(f"  {'metric':12s} | {'mean':>8s} | {'std':>8s}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}")
    summary: dict = {}
    for m in metrics:
        vals = [r[m] for r in rows]
        summary[m] = (_np.mean(vals), _np.std(vals))
        print(f"  {m:12s} | {summary[m][0]:>+8.4f} | {summary[m][1]:>8.4f}")
    for m, ci_key in [("range_t", "range_ci_excl"), ("z_range_t", "z_range_ci_excl")]:
        vals      = [r[m] for r in rows]
        excl_frac = float(_np.mean([r[ci_key] for r in rows]))
        summary[m]      = (_np.mean(vals), _np.std(vals))
        summary[ci_key] = (excl_frac, 0.0)
        print(f"  {m:12s} | {summary[m][0]:>+8.4f} | {summary[m][1]:>8.4f}  ci_excl={excl_frac:.0%}")
    return summary


_NOISE_LEVELS = (0.002, 0.003, 0.005, 0.010)


def noise_sweep(noise_levels: tuple = _NOISE_LEVELS, seeds: tuple = _SWEEP_SEEDS) -> None:
    import numpy as _np
    rows: list[dict] = []
    for noise in noise_levels:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = sweep(seeds=seeds, noise=noise)
        rows.append({"noise": noise, **{k: v for k, (v, _) in stats.items()},
                     **{f"{k}_std": s for k, (_, s) in stats.items()}})

    print(f"[NOISE SWEEP]  seeds={list(seeds)}")
    print(f"  {'noise':>6s} | {'range_pf':>8s} {'(std)':>6s} | {'z_range_pf':>10s} {'(std)':>6s}")
    print(f"  {'------':>6s}-+-{'--------':>8s}-{'------':>6s}-+-{'----------':>10s}-{'------':>6s}")
    for r in rows:
        print(f"  {r['noise']:>6.1%} | {r['range_pf']:>8.3f} {r['range_pf_std']:>6.3f} | {r['z_range_pf']:>10.3f} {r['z_range_pf_std']:>6.3f}")

    print(f"\n[SIG]  boot=1000")
    print(f"  {'noise':>6s} | {'abs t':>6s} {'excl':>5s} | {'z t':>6s} {'excl':>5s}")
    print(f"  {'------':>6s}-+-{'------':>6s}-{'-----':>5s}-+-{'------':>6s}-{'-----':>5s}")
    for r in rows:
        print(f"  {r['noise']:>6.1%} | {r['range_t']:>6.3f} {r['range_ci_excl']:>4.0%}  | {r['z_range_t']:>6.3f} {r['z_range_ci_excl']:>4.0%}")


def _fetch_ohlcv(ohlcv_dir: "Path", target: int = 100) -> None:
    """Download missing Nikkei symbols from yfinance into ohlcv_dir as date,close CSVs."""
    try:
        import yfinance as _yf
    except ImportError:
        print("[FETCH] yfinance not available"); return
    existing = {p.stem for p in ohlcv_dir.glob("*.csv")}
    needed   = [s for s in _NIKKEI_EXTENDED if s not in existing]
    print(f"[FETCH] {len(existing)} existing  {len(needed)} to fetch  target={target}")
    n = 0
    for sym in needed:
        if len(existing) + n >= target:
            break
        try:
            raw = _yf.download(sym, start="2018-01-01", end="2024-12-31",
                               auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 200:
                continue
            close_col = raw["Close"]
            if hasattr(close_col.columns if hasattr(close_col, "columns") else [], "__len__"):
                close_vals = close_col.squeeze().values
            else:
                close_vals = close_col.values
            out = pd.DataFrame({"close": close_vals},
                               index=raw.index.strftime("%Y-%m-%d"))
            out.index.name = "date"
            out.to_csv(ohlcv_dir / f"{sym}.csv")
            n += 1
        except Exception:
            pass
    print(f"[FETCH] done  +{n} fetched  total={len(existing) + n}")


def run_real_wf() -> None:
    """Momentum walk-forward on real OHLCV data. IS≤2022 / OOS≥2023."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as _np

    OHLCV_DIR = BASE_DIR / "data" / "ohlcv"
    files = sorted(OHLCV_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[ERROR] OHLCV dir empty: {OHLCV_DIR}")

    rows: list[dict] = []
    n_loaded = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(df) < 42:
                continue
            df["_ret"]  = df["close"].pct_change()
            df["_rstd"] = df["_ret"].rolling(window=20).std()
            sym = fp.stem
            for i in range(len(df) - 1):
                if i % 2 != 0:
                    continue
                ep, xp = df["close"].iloc[i], df["close"].iloc[i + 1]
                ret = (xp / ep) - 1
                if abs(ret) < 0.001:
                    continue
                rs = df["_rstd"].iloc[i]
                z  = float(ret / rs) if (pd.notna(rs) and rs > 0) else 0.0
                rows.append({"symbol": sym, "date": df["date"].iloc[i], "ret": ret, "z": z})
            n_loaded += 1
        except Exception:
            pass

    if not rows:
        sys.exit("[ERROR] No trades from real OHLCV data")

    t = (pd.DataFrame(rows)
           .sort_values(["symbol", "date"])
           .reset_index(drop=True))
    t["ret_prev"] = t.groupby("symbol")["ret"].shift(1)
    t["sig"]      = t["ret_prev"].apply(lambda r: 1 if r > 0 else (-1 if r < 0 else 0))
    t = t[t["sig"] != 0].copy()
    t["ret_sig"]  = t["sig"] * t["ret"]
    t["ret_net"]  = t["ret_sig"] - (_COMMISSION + _SLIPPAGE)

    _pf_f = lambda g: float(g[g > 0].sum() / abs(g[g < 0].sum())) if g[g < 0].sum() != 0 else float("inf")
    _dd_f = lambda g: float((g.cumsum() - g.cumsum().cummax()).min())

    is_sl  = t["date"] <= "2022-12-31"
    oos_sl = t["date"] >= "2023-01-01"

    print(f"\n[REAL WALK-FORWARD]  symbols={n_loaded}  signal=sign(ret_prev)  cost={_COMMISSION+_SLIPPAGE:.3%}/trade")
    print(f"  {'metric':28s} | {'IS  2018-2022':>14s} | {'OOS 2023-2024':>14s}")
    print(f"  {'-'*28}-+-{'-'*14}-+-{'-'*14}")

    for tag, col in [("gross", "ret_sig"), (f"net(-{_COMMISSION+_SLIPPAGE:.3%})", "ret_net")]:
        gi, go = t.loc[is_sl, col], t.loc[oos_sl, col]
        pf_i,  pf_o  = _pf_f(gi),               _pf_f(go)
        ws_i,  ws_o  = _winrate_sig(gi),         _winrate_sig(go)
        ts_i,  ts_o  = _sig(gi, n_boot=200),     _sig(go, n_boot=200)
        mw_i = " *" if ws_i["p_value"] < 0.05 else "  "
        mw_o = " *" if ws_o["p_value"] < 0.05 else "  "
        mt_i = " *" if ts_i["ci_excl_0"]       else "  "
        mt_o = " *" if ts_o["ci_excl_0"]       else "  "
        print(f"  --- [{tag}] ---")
        print(f"  {'n_trades':28s} | {len(gi):>14,d} | {len(go):>14,d}")
        print(f"  {'win_rate':28s} | {ws_i['win_rate']:>14.1%} | {ws_o['win_rate']:>14.1%}")
        print(f"  {'p_value (binom H0=50%)':28s} | {ws_i['p_value']:>12.4f}{mw_i}| {ws_o['p_value']:>12.4f}{mw_o}")
        print(f"  {'Wilson 95% CI':28s} | [{ws_i['ci_lo']:.3f},{ws_i['ci_hi']:.3f}]      | [{ws_o['ci_lo']:.3f},{ws_o['ci_hi']:.3f}]")
        print(f"  {'profit_factor':28s} | {pf_i:>14.3f} | {pf_o:>14.3f}")
        print(f"  {'avg_ret':28s} | {gi.mean():>+14.4f} | {go.mean():>+14.4f}")
        print(f"  {'t_stat':28s} | {ts_i['t_stat']:>+12.3f}{mt_i}| {ts_o['t_stat']:>+12.3f}{mt_o}")
        print(f"  {'CI_excl_zero (boot 95%)':28s} | {'yes' if ts_i['ci_excl_0'] else 'no':>14s} | {'yes' if ts_o['ci_excl_0'] else 'no':>14s}")
        print(f"  {'max_drawdown':28s} | {_dd_f(gi):>+14.4f} | {_dd_f(go):>+14.4f}")


def run_real_wf_mom() -> None:
    """Multi-day momentum (5d+20d), low-vol filter, N=5, 1-position-per-symbol no-overlap. IS≤2022 / OOS≥2023."""
    sys.stdout.reconfigure(encoding="utf-8")

    OHLCV_DIR = BASE_DIR / "data" / "ohlcv"
    HOLD_DAYS = (5,)
    if len(list(OHLCV_DIR.glob("*.csv"))) < 100:
        _fetch_ohlcv(OHLCV_DIR)
    files = sorted(OHLCV_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[ERROR] OHLCV dir empty: {OHLCV_DIR}")

    # buckets[N] = list of per-symbol DataFrames with columns [date, ret]
    buckets: dict[int, list] = {N: [] for N in HOLD_DAYS}
    xs_rows: list = []   # (date, ret_5d) rows across all symbols → regime signal
    n_loaded = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(df) < 42:
                continue
            df["ret_1d"]  = df["close"].pct_change()
            df["ret_5d"]  = df["close"] / df["close"].shift(5) - 1
            df["ret_20d"] = df["close"] / df["close"].shift(20) - 1
            df["rstd_20"] = df["ret_1d"].rolling(20).std()
            df["low_vol"] = df["rstd_20"] < df["rstd_20"].expanding().median()
            df["sig"]     = (df["ret_5d"] > 0) & (df["ret_20d"] > 0) & df["low_vol"]
            xs_rows.append(df[["date", "ret_5d", "ret_20d"]].dropna())
            sym = fp.stem
            for N in HOLD_DAYS:
                fwd  = df["close"].shift(-N) / df["close"] - 1
                cand = df.index[(df["sig"]) & fwd.notna() & (fwd.abs() >= 0.001)].tolist()
                # max 1 position per symbol: skip signals within holding period
                kept, last_exit = [], -1
                for i in cand:
                    if i > last_exit:
                        kept.append(i)
                        last_exit = i + N
                sub = df.loc[kept, ["date"]].copy()
                sub["ret"] = fwd[kept].values
                buckets[N].append(sub)
            n_loaded += 1
        except Exception:
            pass

    # regime: xs_ret5d > -0.01  (block only strong negative regimes, lookahead-free)
    _xs          = pd.concat(xs_rows, ignore_index=True)
    _xs_g        = _xs.groupby("date")[["ret_5d", "ret_20d"]].mean()
    regime_bull  = (_xs_g["ret_5d"] > -0.01)   # date → bool

    _pf_f = lambda g: float(g[g > 0].sum() / abs(g[g < 0].sum())) if g[g < 0].sum() != 0 else float("inf")
    cost  = _COMMISSION + _SLIPPAGE

    HDR = (f"  {'N':>2} | {'per':3s} | {'n':>7} | {'win%':>6} | {'p_val':>7} | "
           f"{'PF_gr':>5} | {'avg_gr':>7} | {'t_gr':>6} | "
           f"{'PF_nt':>5} | {'avg_nt':>7} | {'t_nt':>6} | CI")
    SEP = ("  " + "-"*2 + "-+-" + "-"*3 + "-+-" + "-"*7 + "-+-" + "-"*6 + "-+-" + "-"*7 + "-+-"
           + "-"*5 + "-+-" + "-"*7 + "-+-" + "-"*6 + "-+-"
           + "-"*5 + "-+-" + "-"*7 + "-+-" + "-"*6 + "-+----")

    rgm_bull_days = int(regime_bull.sum())
    rgm_bear_days = int((~regime_bull).sum())
    print(f"\n[REAL WF - N=5 NO-OVERLAP]  symbols={n_loaded}  signal=ret5d>0&ret20d>0&low_vol  max1pos/sym  cost={cost:.3%}/trade")
    print(f"  regime: xs_ret5d>-0.01  bull={rgm_bull_days}d bear={rgm_bear_days}d  (+IS/+OS = regime-active trades only)")
    print(HDR)
    print(SEP)

    for idx, N in enumerate(HOLD_DAYS):
        dfs = buckets[N]
        if not dfs:
            for per in ("IS ", "OOS"):
                print(f"  {N:>2} | {per} | {'n/a':>7}")
            if idx < len(HOLD_DAYS) - 1:
                print(SEP)
            continue
        t = pd.concat(dfs, ignore_index=True)
        t["ret_net"] = t["ret"] - cost
        t["bull"]    = t["date"].map(regime_bull).fillna(False)
        is_sl  = t["date"] <= "2022-12-31"
        oos_sl = t["date"] >= "2023-01-01"

        def _print_row(per_label: str, g: "pd.Series", gn: "pd.Series") -> None:
            pf_g = _pf_f(g);  pf_n = _pf_f(gn)
            ws   = _winrate_sig(g)
            ts_g = _sig(g, n_boot=200)
            ts_n = _sig(gn, n_boot=200)
            mw  = "*" if ws["p_value"]    < 0.05 else " "
            mg  = "*" if ts_g["ci_excl_0"]        else " "
            mn  = "*" if ts_n["ci_excl_0"]        else " "
            ci  = "yes" if ts_n["ci_excl_0"]      else "no "
            print(f"  {N:>2} | {per_label:<3} | {len(g):>7,d} | {ws['win_rate']:>6.1%} | {ws['p_value']:>6.4f}{mw} | "
                  f"{pf_g:>5.3f} | {g.mean():>+7.4f} | {ts_g['t_stat']:>+5.2f}{mg} | "
                  f"{pf_n:>5.3f} | {gn.mean():>+7.4f} | {ts_n['t_stat']:>+5.2f}{mn} | {ci}")

        for per, sl in (("IS ", is_sl), ("OOS", oos_sl)):
            gi, gn = t.loc[sl, "ret"], t.loc[sl, "ret_net"]
            if len(gi) < 5:
                print(f"  {N:>2} | {per} | {'too few':>7}")
                continue
            _print_row(per, gi, gn)
            # regime-filtered rows (+IS / +OS)
            rsl  = sl & t["bull"]
            gr, gn_r = t.loc[rsl, "ret"], t.loc[rsl, "ret_net"]
            if len(gr) >= 5:
                _print_row("+" + per.strip()[:2], gr, gn_r)
        if idx < len(HOLD_DAYS) - 1:
            print(SEP)

    # ── portfolio simulation ───────────────────────────
    _CAPITAL       = 3_000_000
    _FRACTION      = 0.03
    _MAX_POSITIONS = 10
    _HOLD_N        = 5
    t_s   = t.sort_values("date").reset_index(drop=True)
    is_m  = t_s["date"] <= pd.Timestamp("2022-12-31")
    oos_m = t_s["date"] >= pd.Timestamp("2023-01-01")

    _HDR0 = f"  {'period':6} | {'n':>6} | {'trd/yr':>6} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7}"
    _SEP0 = f"  {'------':6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}"
    _HDR1 = _HDR0 + f" | {'final_equity':>14}"
    _SEP1 = _SEP0 + f"-+-{'-'*14}"

    print(f"\n[PORTFOLIO SIM]  capital={_CAPITAL:,}  max_pos={_MAX_POSITIONS}  cost={cost:.3%}")

    print(f"\n  [uncapped]")
    print(_HDR0)
    print(_SEP0)
    for _lbl, _msk in [("IS", is_m), ("OOS", oos_m)]:
        _sub = t_s[_msk].reset_index(drop=True)
        if len(_sub) < 2:
            print(f"  {_lbl:6} | n/a")
            continue
        _m = _portfolio_sim(_sub["date"], _sub["ret_net"], _CAPITAL, _FRACTION)
        print(f"  {_lbl:6} | {_m['n_trades']:>6,} | {_m['trd_per_yr']:>6.0f} | {_m['cagr']:>+8.1%} | {_m['max_dd']:>+8.1%} | {_m['sharpe']:>+7.3f}")

    t_cap  = _cap_positions(t_s, hold_n=_HOLD_N, max_pos=_MAX_POSITIONS)
    is_mc  = t_cap["date"] <= pd.Timestamp("2022-12-31")
    oos_mc = t_cap["date"] >= pd.Timestamp("2023-01-01")

    _FRACS = (0.03, 0.04, 0.05)
    _HDR2  = f"  {'frac':>5} | {'period':6} | {'n':>6} | {'trd/yr':>6} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'final_equity':>14}"
    _SEP2  = f"  {'-----':>5}-+-{'------':6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*14}"
    print(f"\n  [fraction sweep  max_pos={_MAX_POSITIONS}]")
    print(_HDR2)
    print(_SEP2)
    for _frac in _FRACS:
        for _lbl, _msk in [("IS", is_mc), ("OOS", oos_mc)]:
            _sub = t_cap[_msk].reset_index(drop=True)
            if len(_sub) < 2:
                print(f"  {_frac:.0%} | {_lbl:6} | n/a")
                continue
            _m = _portfolio_sim(_sub["date"], _sub["ret_net"], _CAPITAL, _frac)
            _fl = f"{_frac:.0%}" if _lbl == "IS" else "     "
            print(f"  {_fl:>5} | {_lbl:6} | {_m['n_trades']:>6,} | {_m['trd_per_yr']:>6.0f} | {_m['cagr']:>+8.1%} | {_m['max_dd']:>+8.1%} | {_m['sharpe']:>+7.3f} | {_m['final_equity']:>14,.0f}")
        print(_SEP2)


def run_diversification_eval() -> None:
    """Correlation and combined Sharpe: momentum vs 3 alternative signals (OOS)."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as _np

    OHLCV_DIR = BASE_DIR / "data" / "ohlcv"
    N, cost, MAX_POS, FRAC = 5, _COMMISSION + _SLIPPAGE, 10, 0.03

    files = sorted(OHLCV_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[ERROR] OHLCV dir empty: {OHLCV_DIR}")

    buckets: dict[str, list] = {
        "momentum": [], "mean_rev": [], "breakout": [], "vol_exp": [],
    }
    xs_rows: list = []
    n_loaded = 0

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(df) < 42:
                continue
            df["ret_1d"]  = df["close"].pct_change()
            df["ret_5d"]  = df["close"] / df["close"].shift(5) - 1
            df["ret_20d"] = df["close"] / df["close"].shift(20) - 1
            df["rstd_20"] = df["ret_1d"].rolling(20).std()
            df["low_vol"] = df["rstd_20"] < df["rstd_20"].expanding().median()
            df["hi20"]    = df["close"].rolling(20).max().shift(1)
            fwd = df["close"].shift(-N) / df["close"] - 1
            sigs: dict[str, "pd.Series"] = {
                "momentum": (df["ret_5d"] > 0) & (df["ret_20d"] > 0) & df["low_vol"],
                "mean_rev": df["ret_1d"] < -0.02,
                "breakout": df["close"] > df["hi20"],
                "vol_exp":  (df["rstd_20"] > df["rstd_20"].expanding().median()) &
                            (df["rstd_20"] > df["rstd_20"].shift(5)),
            }
            xs_rows.append(df[["date", "ret_5d"]].dropna())
            for sname, sig in sigs.items():
                cand = df.index[sig & fwd.notna() & (fwd.abs() >= 0.001)].tolist()
                kept, last_exit = [], -1
                for i in cand:
                    if i > last_exit:
                        kept.append(i)
                        last_exit = i + N
                if kept:
                    sub = df.loc[kept, ["date"]].copy()
                    sub["ret"] = fwd[kept].values
                    buckets[sname].append(sub)
            n_loaded += 1
        except Exception:
            pass

    _xs   = pd.concat(xs_rows, ignore_index=True)
    _xs_g = _xs.groupby("date")["ret_5d"].mean()
    _bull = _xs_g > -0.01

    strats: dict[str, pd.DataFrame] = {}
    for sname, dfs_list in buckets.items():
        if not dfs_list:
            continue
        t = pd.concat(dfs_list, ignore_index=True)
        t["ret_net"] = t["ret"] - cost
        if sname == "momentum":
            t = t[t["date"].map(_bull).fillna(False)].reset_index(drop=True)
        t = _cap_positions(t.sort_values("date").reset_index(drop=True),
                           hold_n=N, max_pos=MAX_POS)
        strats[sname] = t

    def _daily_ret(t: pd.DataFrame) -> "pd.Series":
        return t.groupby("date")["ret_net"].sum() * FRAC

    daily   = {k: _daily_ret(v) for k, v in strats.items()}
    names   = list(daily.keys())
    all_dt  = sorted(set().union(*[set(s.index) for s in daily.values()]))
    aligned = pd.DataFrame(
        {k: daily[k].reindex(all_dt, fill_value=0.0) for k in names},
        index=all_dt,
    )
    oos_al = aligned.loc[aligned.index >= pd.Timestamp("2023-01-01")]
    corr   = oos_al.corr()

    W = max(len(n) for n in names)
    print(f"\n[DIVERSIFICATION EVAL]  symbols={n_loaded}  N={N}  max_pos={MAX_POS}"
          f"  frac={FRAC:.0%}  cost={cost:.3%}")

    # ── correlation matrix ───────────────────────────
    print(f"\n  [CORRELATION  OOS 2023-2024  daily portfolio returns]")
    print("  " + " " * (W + 2) + "".join(f"{n:>{W + 2}}" for n in names))
    print("  " + "-" * (W + 2) + ("-" * (W + 2)) * len(names))
    for r in names:
        row = f"  {r:{W + 2}}"
        for c in names:
            v = corr.loc[r, c] if (r in corr.index and c in corr.columns) else float("nan")
            row += f"{v:>+{W + 2}.3f}"
        print(row)

    # ── signal stats ─────────────────────────────────
    def _pf_loc(g: "pd.Series") -> float:
        w, l = g[g > 0].sum(), g[g < 0].sum()
        return float(w / abs(l)) if l != 0 else float("inf")

    print(f"\n  [SIGNAL STATS  OOS 2023-2024  trade-level]")
    print(f"  {'signal':12s} | {'n':>5} | {'win%':>5} | {'PF':>5} | {'avg_ret':>8} | t_stat")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")
    for sname, t in strats.items():
        oos_t = t[t["date"] >= pd.Timestamp("2023-01-01")]
        if len(oos_t) < 5:
            print(f"  {sname:12s} | n<5")
            continue
        r  = oos_t["ret_net"]
        wr = float((r > 0).mean())
        ts = _sig(r, n_boot=200)
        mk = "*" if ts["ci_excl_0"] else " "
        print(f"  {sname:12s} | {len(r):>5,} | {wr:>5.1%} | {_pf_loc(r):>5.3f} | {r.mean():>+8.4f} | {ts['t_stat']:>+6.3f}{mk}")

    # ── combined portfolio performance ───────────────
    def _perf(s: "pd.Series") -> tuple:
        s   = s.fillna(0.0)
        eq  = (1.0 + s).cumprod()
        yrs = (oos_al.index[-1] - oos_al.index[0]).days / 365.25
        cag = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else float("nan")
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        sh  = float(s.mean() / s.std(ddof=1) * _np.sqrt(252)) if s.std(ddof=1) > 0 else float("nan")
        return cag, mdd, sh, int((s != 0).sum())

    print(f"\n  [COMBINED PORTFOLIO  OOS 2023-2024  equal weight]")
    print(f"  {'combo':28s} | {'n_act':>5} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7}")
    print(f"  {'-'*28}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    def _prnt(label: str, s: "pd.Series") -> None:
        cag, mdd, sh, na = _perf(s)
        print(f"  {label:28s} | {na:>5} | {cag:>+8.1%} | {mdd:>+8.1%} | {sh:>+7.3f}")

    mom_s   = oos_al["momentum"]
    base_sh = _perf(mom_s)[2]
    _prnt("momentum only", mom_s)

    decisions: dict[str, str] = {}
    for alt in ["mean_rev", "breakout", "vol_exp"]:
        if alt not in oos_al.columns:
            continue
        combo    = (mom_s + oos_al[alt]) / 2.0
        combo_sh = _perf(combo)[2]
        c_val    = corr.loc["momentum", alt] if alt in corr.columns else float("nan")
        _prnt(f"+{alt} (50/50)", combo)
        decisions[alt] = (
            "ADD"     if (abs(c_val) < 0.3 and combo_sh > base_sh) else
            "NEUTRAL" if  abs(c_val) < 0.3 else
            "REJECT"
        )

    if len(names) >= 2:
        _prnt("all equal (25% each)", oos_al[names].mean(axis=1))

    # ── recommendation ───────────────────────────────
    print(f"\n  [RECOMMENDATION]")
    print(f"  {'signal':12s} | {'corr_mom':>8} | {'Δsharpe':>8} | verdict")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for alt in ["mean_rev", "breakout", "vol_exp"]:
        if alt not in strats:
            continue
        c_val    = corr.loc["momentum", alt] if alt in corr.columns else float("nan")
        alt_s    = oos_al[alt] if alt in oos_al.columns else pd.Series(0.0, index=oos_al.index)
        dsh      = _perf((mom_s + alt_s) / 2.0)[2] - base_sh
        print(f"  {alt:12s} | {c_val:>+8.3f} | {dsh:>+8.3f} | {decisions.get(alt, 'N/A')}")


def run_multi_strategy(frac_cfg=None) -> dict:
    """Multi-strategy v2: score-selected, regime-sized, vol_exp-reserved, global cap."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as _np

    OHLCV_DIR  = BASE_DIR / "data" / "ohlcv"
    N          = 5
    cost       = _COMMISSION + _SLIPPAGE
    MAX_POS    = int((frac_cfg or {}).get("max_pos", 10))
    VOL_MAX    = 2           # vol_exp reserved slots
    import yaml as _yaml, math as _math
    with open(BASE_DIR / "src" / "configs" / "strategy.yaml", encoding="utf-8") as _f:
        _cfg_yaml = _yaml.safe_load(_f)
    _frac_src = frac_cfg or _cfg_yaml.get("fraction", {})
    _raw_bull = float(_frac_src.get("bull", 0.0)  or 0.0)
    _raw_bear = float(_frac_src.get("bear", 0.02) or 0.0)
    FRAC_BULL = max(0.0, min(0.05, _raw_bull)) if not _math.isnan(_raw_bull) else 0.0
    FRAC_BEAR = max(0.0, min(0.05, _raw_bear)) if not _math.isnan(_raw_bear) else 0.0
    WEIGHTS    = {"momentum": 0.40, "mean_rev": 0.40, "vol_exp": 0.20}
    CAPITAL    = 3_000_000
    IS_END     = pd.Timestamp("2022-12-31")
    OOS_START  = pd.Timestamp("2023-01-01")

    files = sorted(OHLCV_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[ERROR] OHLCV dir empty: {OHLCV_DIR}")

    buckets: dict[str, list] = {k: [] for k in WEIGHTS}
    xs_rows: list = []
    n_loaded = 0

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["date"]  = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(df) < 42:
                continue
            df["ret_1d"]  = df["close"].pct_change()
            df["ret_5d"]  = df["close"] / df["close"].shift(5) - 1
            df["ret_20d"] = df["close"] / df["close"].shift(20) - 1
            df["rstd_20"] = df["ret_1d"].rolling(20).std()
            df["low_vol"] = df["rstd_20"] < df["rstd_20"].expanding().median()
            # ① score = expanding Sharpe of ret_20d (no lookahead: ret_20d is backward)
            _mu  = df["ret_20d"].expanding(min_periods=10).mean()
            _sd  = df["ret_20d"].expanding(min_periods=10).std()
            df["score"] = (_mu / _sd).fillna(0.0)
            fwd = df["close"].shift(-N) / df["close"] - 1
            sigs: dict[str, "pd.Series"] = {
                "momentum": (df["ret_5d"] > 0) & (df["ret_20d"] > 0) & df["low_vol"],
                "mean_rev": df["ret_1d"] < -0.02,
                "vol_exp":  (df["rstd_20"] > df["rstd_20"].expanding().median()) &
                            (df["rstd_20"] > df["rstd_20"].shift(5)),
            }
            xs_rows.append(df[["date", "ret_5d"]].dropna())
            for sname, sig in sigs.items():
                cand = df.index[sig & fwd.notna() & (fwd.abs() >= 0.001)].tolist()
                kept, last_exit = [], -1
                for i in cand:
                    if i > last_exit:
                        kept.append(i)
                        last_exit = i + N
                if kept:
                    sub = df.loc[kept, ["date", "score"]].copy()
                    sub["ret"]      = fwd.loc[kept].values
                    sub["strategy"] = sname
                    sub["symbol"]   = fp.stem   # ④ for dedup
                    buckets[sname].append(sub)
            n_loaded += 1
        except Exception:
            pass

    _xs_g  = pd.concat(xs_rows, ignore_index=True).groupby("date")["ret_5d"].mean()
    _bull  = _xs_g > -0.01                                          # momentum regime gate
    _rfrac = _xs_g.apply(lambda x: FRAC_BULL if x > 0 else FRAC_BEAR)  # ② dynamic frac

    raw: dict[str, pd.DataFrame] = {}
    for sname, dfs_list in buckets.items():
        if not dfs_list:
            continue
        t = pd.concat(dfs_list, ignore_index=True)
        t["ret_net"] = t["ret"] - cost
        if sname == "momentum":
            t = t[t["date"].map(_bull).fillna(False)].reset_index(drop=True)
        raw[sname] = t.sort_values("date").reset_index(drop=True)

    # ① score-sort replaces priority-sort; ④ dedup same-day same-symbol
    merged = (
        pd.concat(list(raw.values()), ignore_index=True)
        .sort_values(["date", "symbol", "score"], ascending=[True, True, False])
        .drop_duplicates(subset=["date", "symbol"], keep="first")
        .sort_values(["date", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ③ vol_exp reservation inside local cap (does not touch _cap_positions)
    def _cap_reserved(df: pd.DataFrame) -> pd.DataFrame:
        _bday = pd.tseries.offsets.BusinessDay(N)
        kept:  list[int]   = []
        open_: list[tuple] = []   # (strategy, exit_date)
        for idx, row in df.iterrows():
            entry = pd.Timestamp(row["date"])
            strat = row["strategy"]
            open_ = [(s, ex) for s, ex in open_ if ex > entry]
            vol_n   = sum(1 for s, _ in open_ if s == "vol_exp")
            other_n = len(open_) - vol_n
            if strat == "vol_exp":
                if vol_n >= VOL_MAX or len(open_) >= MAX_POS:
                    continue
            else:
                if other_n >= (MAX_POS - VOL_MAX) or len(open_) >= MAX_POS:
                    continue
            kept.append(idx)
            open_.append((strat, entry + _bday))
        return df.loc[kept].reset_index(drop=True)

    capped = _cap_reserved(merged)

    # ② weighted port_ret with dynamic fraction
    capped = capped.copy()
    capped["frac"]     = capped["date"].map(_rfrac).fillna(FRAC_BULL)
    capped["port_ret"] = capped.apply(
        lambda r: WEIGHTS[r["strategy"]] * r["frac"] * r["ret_net"], axis=1
    )

    daily = capped.groupby("date")["port_ret"].sum()

    def _perf(s: "pd.Series") -> tuple:
        s   = s.fillna(0.0)
        eq  = (1.0 + s).cumprod()
        yrs = (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else 0.0
        cag = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else float("nan")
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        sh  = float(s.mean() / s.std(ddof=1) * _np.sqrt(252)) if s.std(ddof=1) > 0 else float("nan")
        return cag, mdd, sh, float(eq.iloc[-1] * CAPITAL)

    def _pf_loc(g: "pd.Series") -> float:
        w, l = g[g > 0].sum(), g[g < 0].sum()
        return float(w / abs(l)) if l != 0 else float("inf")

    wstr = "  ".join(f"{s[:3]}:{w:.0%}" for s, w in WEIGHTS.items())
    print(f"\n[MULTI-STRATEGY v2]  {wstr}  N={N}  max_pos={MAX_POS}"
          f"  vol_cap={VOL_MAX}  bull={FRAC_BULL:.0%}  bear={FRAC_BEAR:.0%}"
          f"  cost={cost:.3%}  symbols={n_loaded}")

    # allocation
    print(f"\n  [ALLOCATION]")
    print(f"  {'strategy':12s} | {'weight':>6s} | {'eff_bull':>8s} | {'eff_bear':>8s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for s, w in WEIGHTS.items():
        print(f"  {s:12s} | {w:>6.0%} | {w*FRAC_BULL:>8.2%} | {w*FRAC_BEAR:>8.2%}")

    # ⑤ score & selection stats (OOS)
    oos_cap  = capped[capped["date"] >= OOS_START]
    print(f"\n  [SCORE & SELECTION  OOS 2023-2024]")
    print(f"  {'strategy':12s} | {'total':>5} | {'sel':>5} | {'rate':>5} | {'score_mu':>8} | {'score_sd':>8}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}")
    for sname in WEIGHTS:
        pre = raw.get(sname, pd.DataFrame())
        pre_oos = pre[pre["date"] >= OOS_START] if len(pre) > 0 else pre
        sel     = oos_cap[oos_cap["strategy"] == sname]
        n_pre, n_sel = len(pre_oos), len(sel)
        rate  = n_sel / n_pre if n_pre > 0 else float("nan")
        sc_mu = float(sel["score"].mean()) if n_sel > 0 else float("nan")
        sc_sd = float(sel["score"].std())  if n_sel > 1 else float("nan")
        print(f"  {sname:12s} | {n_pre:>5} | {n_sel:>5} | {rate:>5.1%} | {sc_mu:>+8.3f} | {sc_sd:>8.3f}")

    # portfolio performance IS / OOS
    print(f"\n  [PORTFOLIO PERFORMANCE]  shared capital  vol_cap={VOL_MAX}/{MAX_POS}")
    print(f"  {'period':6s} | {'n_trades':>8s} | {'CAGR':>8s} | {'MaxDD':>8s} | {'Sharpe':>7s} | {'final_equity':>14s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*14}")
    metrics: dict = {}
    for lbl, d_sl, c_sl in [
        ("IS",  daily.index <= IS_END,    capped["date"] <= IS_END),
        ("OOS", daily.index >= OOS_START, capped["date"] >= OOS_START),
    ]:
        sub = daily.loc[d_sl]
        if len(sub) < 2:
            print(f"  {lbl:6s} | n/a"); continue
        cag, mdd, sh, fin = _perf(sub)
        n_tr = int(c_sl.sum())
        metrics[lbl] = {"cagr": cag, "maxdd": mdd, "sharpe": sh,
                        "n_trades": n_tr, "final_equity": fin}
        print(f"  {lbl:6s} | {n_tr:>8,} | {cag:>+8.1%} | {mdd:>+8.1%} | {sh:>+7.3f} | {fin:>14,.0f}")

    # ⑤ regime performance (OOS)
    oos_daily = daily[daily.index >= OOS_START]
    _is_bull  = _xs_g.reindex(oos_daily.index, fill_value=0.0) > 0
    print(f"\n  [REGIME PERFORMANCE  OOS 2023-2024]")
    print(f"  {'regime':6s} | {'n_days':>6} | {'n_act':>5} | {'CAGR':>8s} | {'Sharpe':>7s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")
    for rname, rmask in [("bull", _is_bull), ("bear", ~_is_bull)]:
        s = oos_daily.loc[rmask]
        if len(s) < 2:
            print(f"  {rname:6s} | n/a"); continue
        yrs = len(s) / 252
        eq  = (1.0 + s.fillna(0)).cumprod()
        cag = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else float("nan")
        sh  = float(s.mean() / s.std(ddof=1) * _np.sqrt(252)) if s.std(ddof=1) > 0 else float("nan")
        print(f"  {rname:6s} | {len(s):>6} | {int((s!=0).sum()):>5} | {cag:>+8.1%} | {sh:>+7.3f}")

    # ⑤ strategy breakdown (OOS)
    tot_pnl = oos_cap["port_ret"].sum()
    print(f"\n  [STRATEGY BREAKDOWN  OOS 2023-2024]")
    print(f"  {'strategy':12s} | {'n':>5} | {'win%':>5} | {'PF':>5} | {'avg_net':>8s} | {'contrib':>7s}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")
    for sname in WEIGHTS:
        sub = oos_cap[oos_cap["strategy"] == sname]
        if len(sub) < 5:
            print(f"  {sname:12s} | n<5"); continue
        r   = sub["ret_net"]
        pct = float(sub["port_ret"].sum() / tot_pnl * 100) if tot_pnl != 0 else float("nan")
        print(f"  {sname:12s} | {len(sub):>5,} | {float((r>0).mean()):>5.1%} | {_pf_loc(r):>5.3f}"
              f" | {r.mean():>+8.4f} | {pct:>+6.1f}%")

    return metrics


def run_variants() -> None:
    """Test base vs zero_bull fraction configs; print VARIANT SUMMARY + DIFF."""
    sys.stdout.reconfigure(encoding="utf-8")
    _variants = {
        "base":      {"bull": 0.01, "bear": 0.02},
        "zero_bull": {"bull": 0.00, "bear": 0.02},
    }
    results: dict = {}
    for name, fc in _variants.items():
        sep = "=" * 60
        print(f"\n{sep}\n  === {name}  bull={fc['bull']:.0%}  bear={fc['bear']:.0%} ===\n{sep}")
        results[name] = run_multi_strategy(frac_cfg=fc)

    print(f"\n{'='*60}\n  === VARIANT SUMMARY ===\n{'='*60}")
    print(f"  {'variant':12s} | {'period':6s} | {'CAGR':>8s} | {'MaxDD':>8s} | {'Sharpe':>7s} | {'final_equity':>14s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*14}")
    for name, res in results.items():
        for lbl in ["IS", "OOS"]:
            m = res.get(lbl)
            if m is None:
                print(f"  {name:12s} | {lbl:6s} | n/a")
            else:
                print(f"  {name:12s} | {lbl:6s} | {m['cagr']:>+8.1%} | {m['maxdd']:>+8.1%}"
                      f" | {m['sharpe']:>+7.3f} | {m['final_equity']:>14,.0f}")

    print(f"\n  DIFF SUMMARY  (zero_bull − base)")
    print(f"  {'period':6s} | {'ΔCAGR':>8s} | {'ΔMaxDD':>8s} | {'ΔSharpe':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for lbl in ["IS", "OOS"]:
        b = results.get("base",      {}).get(lbl)
        z = results.get("zero_bull", {}).get(lbl)
        if not (b and z):
            continue
        print(f"  {lbl:6s} | {z['cagr']-b['cagr']:>+8.1%}"
              f" | {z['maxdd']-b['maxdd']:>+8.1%}"
              f" | {z['sharpe']-b['sharpe']:>+8.3f}")


def run_grid_search() -> None:
    """Grid search: fraction_bear × max_pos → maximize CAGR×Sharpe, OOS MaxDD ≤ 10%."""
    sys.stdout.reconfigure(encoding="utf-8")
    import contextlib, io

    class _SilentBuf(io.StringIO):
        def reconfigure(self, **_): pass

    frac_list   = [0.02, 0.03, 0.04, 0.05, 0.06]
    maxpos_list = [10, 12, 15, 20]

    best_score = float("-inf")
    best_cfg:  dict = {}
    best_oos:  dict = {}

    for f in frac_list:
        for m in maxpos_list:
            cfg = {"bull": 0.0, "bear": f, "max_pos": m}
            with contextlib.redirect_stdout(_SilentBuf()):
                res = run_multi_strategy(frac_cfg=cfg)
            oos = res.get("OOS", {})
            if not oos:
                continue
            cagr, mdd, sh = oos["cagr"], oos["maxdd"], oos["sharpe"]
            score = float("-inf") if (mdd < -0.10) else (cagr * sh)
            if score > best_score:
                best_score = score
                best_cfg   = cfg
                best_oos   = oos

    print(f"\n[GRID SEARCH]  bear∈{[f'{v:.0%}' for v in frac_list]}  max_pos∈{maxpos_list}")
    print(f"  score = CAGR × Sharpe   制約: OOS MaxDD ≤ 10%   bull=固定0%")
    if not best_cfg:
        print("  全パターン制約違反"); return
    print(f"\n  best_fraction_bull : {best_cfg['bull']:.0%}")
    print(f"  best_fraction_bear : {best_cfg['bear']:.0%}")
    print(f"  best_max_pos       : {best_cfg['max_pos']}")
    print(f"\n  {'metric':12s} | value")
    print(f"  {'-'*12}-+----------")
    print(f"  {'OOS CAGR':12s} | {best_oos['cagr']:>+8.1%}")
    print(f"  {'OOS MaxDD':12s} | {best_oos['maxdd']:>+8.1%}")
    print(f"  {'OOS Sharpe':12s} | {best_oos['sharpe']:>+8.3f}")
    print(f"  {'OOS n_trades':12s} | {best_oos['n_trades']:>8,}")
    print(f"  {'score':12s} | {best_score:>+8.4f}")


def build_alpha_dataset() -> None:
    """Build data/trades_log.csv with score + score_new + forward_return from OHLCV."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as _np

    _OHLCV_DIR = BASE_DIR / "data" / "ohlcv"
    _OUT       = BASE_DIR / "data" / "trades_log.csv"
    _N_FWD     = 5

    files = sorted(_OHLCV_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"[BUILD] ERROR: OHLCV dir empty: {_OHLCV_DIR}")

    rows: list[pd.DataFrame] = []
    for _fp in files:
        try:
            _df = pd.read_csv(_fp, parse_dates=["date"])
            _df["close"] = pd.to_numeric(_df["close"], errors="coerce")
            _df = _df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(_df) < 42:
                continue

            # synthesize open/high/low/volume from close
            _df["open"]   = _df["close"].shift(1)
            _df["high"]   = _df[["close", "open"]].max(axis=1)
            _df["low"]    = _df[["close", "open"]].min(axis=1)
            _df["volume"] = 1_000_000.0

            # existing score: expanding Sharpe of ret_20d
            _ret20 = _df["close"] / _df["close"].shift(20) - 1
            _mu    = _ret20.expanding(min_periods=10).mean()
            _sd    = _ret20.expanding(min_periods=10).std()
            _df["score"] = (_mu / _sd).fillna(0.0)

            # score_new: body/range * volume with safety guard
            _s = (_df["close"] - _df["open"]) / (_df["high"] - _df["low"] + 1e-6) * _df["volume"]
            _s = _s.replace([_np.inf, -_np.inf], _np.nan)
            _lo, _hi = _s.quantile(0.001), _s.quantile(0.999)
            _s = _s.clip(_lo, _hi)
            _df["score_new"] = _s

            # score_test: 5-day z-score of close, winsorized
            if "score_test" not in _df.columns:
                _st = (_df["close"] - _df["close"].rolling(5).mean()) / (_df["close"].rolling(5).std() + 1e-6)
                _st = _st.replace([_np.inf, -_np.inf], _np.nan)
                _st_lo, _st_hi = _st.quantile(0.001), _st.quantile(0.999)
                _df["score_test"] = _st.clip(_st_lo, _st_hi)

            # forward return
            _df["forward_return"] = _df["close"].shift(-_N_FWD) / _df["close"] - 1

            _sub = _df[["date", "score", "score_new", "score_test", "forward_return"]].dropna()
            rows.append(_sub)
        except Exception:
            pass

    if not rows:
        sys.exit("[BUILD] ERROR: no data loaded")

    _out = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _out.to_csv(_OUT, index=False)
    print(f"[BUILD] {_OUT}  ({len(_out)} rows, {_out['date'].nunique()} dates)")


def _eval_one_score(
    df: "pd.DataFrame",
    score_col: str,
    n_quant: int,
    ic_min: float,
) -> "tuple[dict, bool]":
    """Compute IC_weighted/IC_tstat/spread/monotonic for one score column. Returns (metrics, pass)."""
    import numpy as _np, math as _math

    _sub = df.dropna(subset=[score_col, "forward_return"]).reset_index(drop=True)
    _ic_list: list[float] = []
    _n_list:  list[float] = []
    for _, _g in _sub.groupby("date"):
        _n = len(_g)
        if _n < 2:
            continue
        _r = float(_g[score_col].corr(_g["forward_return"]))
        if not _math.isnan(_r):
            _ic_list.append(_r)
            _n_list.append(float(_n))

    if not _ic_list:
        return {}, False

    _ic_arr = _np.array(_ic_list)
    _n_arr  = _np.array(_n_list)
    ic_w    = float((_ic_arr * _n_arr).sum() / _n_arr.sum())
    nd      = len(_ic_arr)
    ic_std  = float(_ic_arr.std(ddof=1)) if nd > 1 else float("nan")
    ic_t    = (float(_ic_arr.mean() / (ic_std / _np.sqrt(nd)))
               if (ic_std > 0 and nd > 1) else float("nan"))

    _sub["_rank"] = _sub[score_col].rank(method="first")
    _sub["_q"]    = pd.qcut(_sub["_rank"], q=n_quant, labels=False)
    q_rets        = _sub.groupby("_q", observed=True)["forward_return"].mean()
    spread        = float(q_rets.iloc[-1] - q_rets.iloc[0])
    mono          = bool(q_rets.is_monotonic_increasing)

    metrics = {
        "IC_weighted":    ic_w,
        "IC_tstat":       ic_t,
        "spread":         spread,
        "monotonic_flag": mono,
        "q_rets":         q_rets,
    }
    passed = (ic_w >= ic_min) and (spread > 0)
    return metrics, passed


def run_edge_eval() -> None:
    """Edge evaluation: weighted IC, t-stat, quantile returns, fail-fast, persist."""
    sys.stdout.reconfigure(encoding="utf-8")
    import math as _math
    from datetime import timezone as _tz, datetime as _dt

    _TRADES_LOG = BASE_DIR / "data" / "trades_log.csv"
    _EDGE_LOG   = BASE_DIR / "data" / "edge_eval.csv"
    _IC_MIN     = 0.02
    _N_QUANT    = 5

    if not _TRADES_LOG.exists():
        print(f"[EDGE EVAL] ERROR: not found: {_TRADES_LOG}"); sys.exit(1)

    df = pd.read_csv(_TRADES_LOG, parse_dates=["date"])
    if "forward_return" not in df.columns:
        print("[EDGE EVAL] ERROR: forward_return column missing"); sys.exit(1)

    score_cols = [c for c in df.columns if c == "score" or c.startswith("score_")]
    if not score_cols:
        print("[EDGE EVAL] ERROR: no score column found"); sys.exit(1)

    if len(df.dropna(subset=["forward_return"])) < _N_QUANT:
        print(f"[EDGE EVAL] ERROR: too few rows for {_N_QUANT} quantiles"); sys.exit(1)

    ts = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[EDGE EVAL]  {ts}")

    all_pass = True
    _log_rows: list[dict] = []

    for sc in score_cols:
        metrics, passed = _eval_one_score(df, sc, _N_QUANT, _IC_MIN)
        if not metrics:
            print(f"\n  [{sc}]  ERROR: no valid daily IC")
            all_pass = False
            continue

        ic_w  = metrics["IC_weighted"]
        ic_t  = metrics["IC_tstat"]
        sprd  = metrics["spread"]
        mono  = metrics["monotonic_flag"]
        q_rets = metrics["q_rets"]

        print(f"\n  [{sc}]")
        print(f"    IC_weighted  : {ic_w:>+.4f}")
        print(f"    IC_tstat     : {ic_t:>+.3f}" if not _math.isnan(ic_t) else "    IC_tstat     :    nan")
        print(f"    spread       : {sprd:>+.4f}")
        print(f"    monotonic    : {mono}")
        print(f"    quantile returns (Q0=lowest → Q4=highest):")
        for _q, _r in q_rets.items():
            print(f"      Q{int(_q)}: {_r:>+.4f}")

        if not passed:
            print(f"    EDGE BROKEN - {sc}")
            all_pass = False

        _row: dict = {
            "timestamp":    ts,
            "score_col":    sc,
            "IC_weighted":  round(ic_w, 6),
            "IC_tstat":     round(ic_t, 4) if not _math.isnan(ic_t) else float("nan"),
            "spread":       round(sprd, 6),
            "monotonic_flag": mono,
        }
        for _q, _r in q_rets.items():
            _row[f"q{int(_q)}_ret"] = round(float(_r), 6)
        _log_rows.append(_row)

    if not all_pass:
        print("\nEDGE BROKEN - STOP")
        _write_header = not _EDGE_LOG.exists()
        if _log_rows:
            pd.DataFrame(_log_rows).to_csv(_EDGE_LOG, mode="a", header=_write_header, index=False)
        sys.exit(1)

    _write_header = not _EDGE_LOG.exists()
    pd.DataFrame(_log_rows).to_csv(_EDGE_LOG, mode="a", header=_write_header, index=False)


def run_stability_check() -> None:
    """Stability check: split trades_log into 3 equal time periods, compute per-period Sharpe."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as _np, math as _math
    from datetime import timezone as _tz, datetime as _dt

    _TRADES_LOG    = BASE_DIR / "data" / "trades_log.csv"
    _STABILITY_LOG = BASE_DIR / "data" / "stability_eval.csv"
    _LABELS        = ["early", "mid", "late"]

    if not _TRADES_LOG.exists():
        print(f"[STABILITY] ERROR: not found: {_TRADES_LOG}"); sys.exit(1)

    df = (pd.read_csv(_TRADES_LOG, parse_dates=["date"])
            .dropna(subset=["forward_return"])
            .sort_values("date")
            .reset_index(drop=True))

    if df.empty:
        print("[STABILITY] ERROR: empty dataset"); sys.exit(1)

    # 3 equal time-span windows
    t_min  = df["date"].min()
    t_max  = df["date"].max()
    _span  = (t_max - t_min).days / 3
    _cuts  = [t_min + pd.Timedelta(days=int(_span * i)) for i in range(4)]
    _cuts[-1] = t_max  # exact boundary for last period

    _masks = [
        (df["date"] >= _cuts[0]) & (df["date"] <  _cuts[1]),
        (df["date"] >= _cuts[1]) & (df["date"] <  _cuts[2]),
        (df["date"] >= _cuts[2]) & (df["date"] <= _cuts[3]),
    ]

    _sharpes:  list[float] = []
    _n_days:   list[int]   = []
    _n_trades: list[int]   = []

    for mask in _masks:
        sub = df.loc[mask]
        _n_days.append(int(sub["date"].nunique()))
        _n_trades.append(len(sub))
        if len(sub) < 5 or sub["date"].nunique() < 2:
            _sharpes.append(float("nan"))
            continue
        daily = sub.groupby("date")["forward_return"].mean()
        _sd   = float(daily.std(ddof=1))
        _sharpes.append(
            float(daily.mean() / _sd * _np.sqrt(252)) if _sd > 0 else float("nan")
        )

    _valid      = [s for s in _sharpes if not _math.isnan(s)]
    sharpe_mean = float(_np.mean(_valid))         if _valid           else float("nan")
    sharpe_std  = float(_np.std(_valid, ddof=1))  if len(_valid) > 1  else float("nan")
    sharpe_min  = float(_np.min(_valid))          if _valid           else float("nan")

    # OLS slope of Sharpe across [early=0, mid=1, late=2]
    _x         = _np.array([0, 1, 2], dtype=float)
    _y         = _np.array([s if not _math.isnan(s) else 0.0 for s in _sharpes], dtype=float)
    slope      = float(_np.polyfit(_x, _y, 1)[0]) if len(_valid) >= 2 else float("nan")
    trend_flag = (not _math.isnan(slope)) and slope < -0.2

    # per-period MaxDD from daily equity curve
    maxdd_list: list[float] = []
    for _mask in _masks:
        _sub = df.loc[_mask]
        if _sub["date"].nunique() < 2:
            maxdd_list.append(float("nan"))
            continue
        _eq  = (1.0 + _sub.groupby("date")["forward_return"].mean().fillna(0.0)).cumprod()
        maxdd_list.append(float(((_eq - _eq.cummax()) / _eq.cummax()).min()))

    # period label column for regime aggregation
    df["_period"] = "early"
    df.loc[_masks[1], "_period"] = "mid"
    df.loc[_masks[2], "_period"] = "late"

    ts = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[STABILITY CHECK]  {ts}")
    print(f"  {'period':6s} | {'n_days':>6} | {'n_trades':>8} | {'Sharpe':>7} | {'MaxDD':>7}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    for lbl, nd, nt, sh, mdd in zip(_LABELS, _n_days, _n_trades, _sharpes, maxdd_list):
        _sh_str  = f"{sh:>+7.3f}"  if not _math.isnan(sh)  else "    n/a"
        _mdd_str = f"{mdd:>+7.3f}" if not _math.isnan(mdd) else "    n/a"
        print(f"  {lbl:6s} | {nd:>6} | {nt:>8} | {_sh_str} | {_mdd_str}")
    print(f"\n  sharpe_mean : {sharpe_mean:>+.3f}")
    print(f"  sharpe_std  : {sharpe_std:>+.3f}")
    print(f"  sharpe_min  : {sharpe_min:>+.3f}")
    print(f"  slope       : {slope:>+.3f}{'  ← declining' if trend_flag else ''}")

    # regime sub-aggregation (informational only — no FAIL impact)
    if "regime" in df.columns:
        print(f"\n  [REGIME SUB-AGGREGATION]")
        print(f"  {'period':6s} | {'regime':8s} | {'Sharpe':>7}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*7}")
        try:
            _rg = (df.groupby(["_period", "regime"])["forward_return"]
                     .apply(lambda r: float(r.mean() / r.std(ddof=1) * _np.sqrt(252))
                            if r.std(ddof=1) > 0 else float("nan"))
                     .reset_index(name="_sh"))
            for _, _row_rg in _rg.iterrows():
                _sh_str = f"{_row_rg['_sh']:>+7.3f}" if not _math.isnan(_row_rg["_sh"]) else "    n/a"
                print(f"  {str(_row_rg['_period']):6s} | {str(_row_rg['regime']):8s} | {_sh_str}")
        except Exception:
            print("  (regime aggregation unavailable)")
    else:
        print(f"\n  [REGIME SUB-AGGREGATION]  no 'regime' column in trades_log.csv")

    # fail-fast checks (priority order) — persist only on PASS
    _s2 = _sharpes[2] if len(_sharpes) > 2 and not _math.isnan(_sharpes[2]) else float("nan")
    if not _math.isnan(_s2) and _s2 < 0.5:
        print("FAIL: recent performance too weak")
        sys.exit(1)

    _dd0, _dd2 = maxdd_list[0], maxdd_list[2]
    if not (_math.isnan(_dd0) or _math.isnan(_dd2)):
        if _dd2 < _dd0 - 0.05:
            print("FAIL: drawdown deterioration")
            sys.exit(1)

    if (((not _math.isnan(sharpe_std)) and sharpe_std > 0.8)
            or ((not _math.isnan(sharpe_min)) and sharpe_min < 0)
            or trend_flag):
        print("FAIL: stability")
        sys.exit(1)

    print("  STABILITY PASS")

    _row = {
        "timestamp":    ts,
        "sharpe_early": round(_sharpes[0], 4) if not _math.isnan(_sharpes[0]) else float("nan"),
        "sharpe_mid":   round(_sharpes[1], 4) if not _math.isnan(_sharpes[1]) else float("nan"),
        "sharpe_late":  round(_sharpes[2], 4) if not _math.isnan(_sharpes[2]) else float("nan"),
        "sharpe_mean":  round(sharpe_mean, 4) if not _math.isnan(sharpe_mean) else float("nan"),
        "sharpe_std":   round(sharpe_std,  4) if not _math.isnan(sharpe_std)  else float("nan"),
        "sharpe_min":   round(sharpe_min,  4) if not _math.isnan(sharpe_min)  else float("nan"),
        "slope":        round(slope, 4)        if not _math.isnan(slope)       else float("nan"),
        "stability_pass": True,
    }
    _write_header = not _STABILITY_LOG.exists()
    pd.DataFrame([_row]).to_csv(_STABILITY_LOG, mode="a", header=_write_header, index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",        action="store_true")
    ap.add_argument("--noise-sweep",  action="store_true")
    ap.add_argument("--real-wf",      action="store_true")
    ap.add_argument("--real-wf-mom",  action="store_true")
    ap.add_argument("--diversify",    action="store_true")
    ap.add_argument("--multi",        action="store_true")
    ap.add_argument("--variants",     action="store_true")
    ap.add_argument("--grid",         action="store_true")
    ap.add_argument("--edge-eval",    action="store_true")
    ap.add_argument("--stability",    action="store_true")
    ap.add_argument("--alpha-build",  action="store_true")
    args = ap.parse_args()
    if args.alpha_build:
        build_alpha_dataset()
    elif args.real_wf_mom:
        run_real_wf_mom()
    elif args.real_wf:
        run_real_wf()
    elif args.diversify:
        run_diversification_eval()
    elif args.variants:
        run_variants()
    elif args.grid:
        run_grid_search()
    elif args.edge_eval:
        run_edge_eval()
    elif args.stability:
        run_stability_check()
    elif args.multi:
        run_multi_strategy()
    elif args.noise_sweep:
        noise_sweep()
    elif args.sweep:
        sweep()
    else:
        print("PIPELINE ENTRYPOINT TRIGGERED")
        import traceback as _tb
        from src.execution.live_pipeline import run_live_pipeline
        _csv = BASE_DIR / "data" / "latest_signals.csv"
        df = pd.read_csv(_csv)
        df["date"] = pd.to_datetime(df["date"])
        market_data = df
        strategies  = [{"strategy_id": "default", "df": df}]
        state: dict = {}
        date = str(df["date"].max().date())
        print(f"[DEBUG] columns: {list(df.columns)}")
        print(f"[DEBUG] rows: {len(df)}, date: {date}")
        print("[DEBUG] calling run_live_pipeline...")
        try:
            run_live_pipeline(market_data, strategies, state, date)
        except Exception:
            _tb.print_exc()
            sys.exit(1)
