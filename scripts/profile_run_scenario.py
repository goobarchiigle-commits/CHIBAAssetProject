from __future__ import annotations

import cProfile
import pstats
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.backtest.composite_alpha_bt as _bt
from src.backtest.rsr import calc_composite_alpha_matrix, calc_universe_rsr
from src.backtest.universe_builder import download_universe
from src.config_loader import load_strategy_config
from src.utils.memory import get_process_rss_mb

FULL_START = '2018-01-01'
OOS_START = '2025-01-01'
OOS_END = '2025-12-31'
FULL_END = OOS_END


def load_data():
    rsr_syms = _bt._load_rsr_universe()
    trade_syms = rsr_syms
    universe_raw = download_universe({**rsr_syms, **trade_syms}, start=FULL_START, end=FULL_END, verbose=False)
    topix_close = _bt._download_topix(FULL_START, FULL_END)
    rsr_df = calc_universe_rsr({sym: universe_raw[sym]['df']['Close'] for sym in rsr_syms if sym in universe_raw})
    alpha_df = calc_composite_alpha_matrix({sym: universe_raw[sym]['df']['Close'] for sym in trade_syms if sym in universe_raw}, window=_bt.COMP_ALPHA_WINDOW).shift(1)
    regime_df = _bt._calc_regime(topix_close)
    return universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close


def build_cfg():
    cfg = load_strategy_config()
    return replace(
        cfg,
        fujiko=replace(cfg.fujiko, mom_period=21, turtle_exit=55, min_rsr=75.0),
        portfolio=replace(cfg.portfolio, capital=3_000_000, max_positions=3),
        risk=replace(cfg.risk, min_hold_days=3),
    )


def run_case(dataset, cfg, precompute_signals: bool):
    universe_raw, rsr_df, alpha_df, regime_df, trade_syms, rsr_syms, topix_close = dataset
    return _bt.run_scenario(
        scenario='BASELINE',
        universe_raw=universe_raw,
        rsr_df=rsr_df,
        alpha_df=alpha_df,
        regime_df=regime_df,
        trade_syms=trade_syms,
        rsr_syms=rsr_syms,
        cfg=cfg,
        start=OOS_START,
        end=OOS_END,
        capital=float(cfg.portfolio.capital),
        verbose=False,
        min_hold=cfg.risk.min_hold_days,
        precompute_signals=precompute_signals,
        topix_close=topix_close,
    )


def profile_case(dataset, cfg, precompute_signals: bool):
    prof = cProfile.Profile()
    rss_before = get_process_rss_mb()
    t0 = time.perf_counter()
    prof.enable()
    result = run_case(dataset, cfg, precompute_signals=precompute_signals)
    prof.disable()
    elapsed = time.perf_counter() - t0
    rss_after = get_process_rss_mb()
    return prof, result, elapsed, rss_before, rss_after


def main() -> int:
    dataset = load_data()
    cfg = build_cfg()

    _, res_old, sec_old, rss0_old, rss1_old = profile_case(dataset, cfg, precompute_signals=False)
    prof_new, res_new, sec_new, rss0_new, rss1_new = profile_case(dataset, cfg, precompute_signals=True)

    print('=== run_scenario benchmark ===')
    print({
        'before_sec': round(sec_old, 3),
        'after_sec': round(sec_new, 3),
        'speedup_x': round(sec_old / sec_new, 2) if sec_new > 0 else None,
        'before_rss_delta_mb': None if rss0_old is None or rss1_old is None else round(rss1_old - rss0_old, 1),
        'after_rss_delta_mb': None if rss0_new is None or rss1_new is None else round(rss1_new - rss0_new, 1),
        'before_signal_calls': res_old.get('signal_calls'),
        'after_signal_calls': res_new.get('signal_calls'),
        'before_sharpe': res_old.get('sharpe'),
        'after_sharpe': res_new.get('sharpe'),
    })
    print('\n=== cProfile top 20 (optimized) ===')
    stats = pstats.Stats(prof_new)
    stats.sort_stats('cumulative')
    eq_old = res_old.get('equity_curve')
    eq_new = res_new.get('equity_curve')
    max_abs_diff = None
    if eq_old is not None and eq_new is not None:
        aligned_old, aligned_new = eq_old.align(eq_new, join='inner')
        if not aligned_old.empty:
            max_abs_diff = float((aligned_new - aligned_old).abs().max())
    print({
        'max_abs_diff': None if max_abs_diff is None else round(max_abs_diff, 6),
        'sharpe_diff': round(res_new.get('sharpe', 0.0) - res_old.get('sharpe', 0.0), 6),
    })
    stats.print_stats(20)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
