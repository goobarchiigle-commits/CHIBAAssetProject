# Production Candidate Integration — Deliverables (2026-06-27)

## Study47 Verification Results

| Case | Full IS CAGR | Full IS MaxDD | True OOS CAGR | ΔCAGR (IS) |
|------|-------------|--------------|--------------|-----------|
| A_BASELINE | +19.37% | -18.25% | +8.50% | — |
| B_ATR_EXT | +19.56% | -18.25% | +10.84% | +0.19pp |
| C_VOL_ADJ | +19.86% | -20.00% | +6.80% | +0.49pp |
| D_EQ_SCALE | +20.22% | -18.07% | +11.22% | +0.85pp |
| **E_COMBINED** | **+20.51%** | **-19.81%** | **+11.90%** | **+1.14pp** |

Gate check: G1=PASS (+1.14pp > 0), G2=PASS (ΔDD=-1.56pp > -2.0pp), G3=PASS (+3.40pp OOS).
Full IS ΔCAGR +1.14pp is lower than WF OOS +6.07pp (Study46) — expected behavior
(Full IS uses all data; WF OOS measures out-of-sample generalization).

---

## A. Code Diff Summary

### New files created:
```
src/research_candidate/__init__.py
src/research_candidate/vol_adj.py        — Study41 D_VOL_ADJ live computation
src/research_candidate/eq_scale_addon.py — Study45 D_EQ_SCALE live addon
src/research_candidate/atr_extension.py  — Study40 ATR Extension SELL filter
src/backtest/study47_production_candidate_verification.py
```

### Modified files:
```
src/configs/strategy.yaml  — research_candidates block (all enabled:false)
src/paths.py               — 3 new runtime state file constants
src/run_live_signal.py     — 3 injection points (all FAIL_OPEN, all default OFF)
```

### Injection points in run_live_signal.py:
1. **Before SignalBridge** (~line 2394): D_VOL_ADJ computes `_rc_max_pos`; replaces `MAX_POS` in `min(MAX_POS, MAX_OPEN_POSITIONS)`.
2. **After bridge.run()** (~line 2484): ATR Extension filters RSR-triggered SELL orders where `close > highest_close - 1×ATR20` and `pnl > 0`.
3. **After winner add-on** (~line 5600): D_EQ_SCALE addon generates BUY orders when `unrealized_gain >= 1×ATR20`.

### Zero change when disabled:
All 3 features are guarded by `getattr(cfg.research_candidates.feature.enabled, False)`.
When all features are disabled (default), code paths are never entered;
`order_objects` and `max_positions` remain bit-identical to current production.

---

## B. Parameter Registry

```yaml
# src/configs/strategy.yaml — research_candidates section
research_candidates:
  atr_extension:
    enabled: false              # Study40; default OFF
    atr_mult: 1.0               # defer threshold: highest_close - mult×ATR20
    max_defer_calendar_days: 7  # ~5 business days max deferral
  vol_adj:
    enabled: false              # Study41; default OFF
    topix_vol_threshold: 0.008  # TOPIX 20d std < 0.8% → expand
    calm_max_positions: 4       # max_positions on calm days
  eq_scale_addon:
    enabled: false              # Study45; default OFF
    atr_mult: 1.0               # trigger: gain >= mult × entry_ATR20
    size_frac: 0.25             # addon size = available_cash × 25%
```

Runtime state files (auto-created on first enable):
- `runtime/vol_adj_state.json` — daily regime decision log
- `runtime/atr_ext_defer_state.json` — deferred exit tracking
- `runtime/eq_scale_addon_state.json` — one-addon-per-lifecycle tracker

PARAMS_LOCKED values unchanged:
- max_positions = 3 (D_VOL_ADJ extends dynamically; does NOT change PARAMS_LOCKED)
- capital = ¥3,000,000
- min_rsr = 75.0, turtle_exit = 55d remain unchanged

---

## C. Migration Notes

### Cold restart safety:
- **ATR Extension**: deferred exits persist in `atr_ext_defer_state.json`. On cold restart,
  the state is re-loaded. If a position has been closed (removed from portfolio_state.json),
  its defer entry is purged by `_purge_exited()`. No phantom deferrals survive restart.
- **D_EQ_SCALE**: addon state keyed by `(symbol, entry_date)`. On cold restart, if the same
  position is still held, the `addon_done=True` flag prevents duplicate addon. If position
  was closed and re-opened (new `entry_date`), state resets and addon is eligible again.
- **D_VOL_ADJ**: stateless computation from TOPIX parquet. Cold restart recomputes from scratch.
  Daily result persisted to `vol_adj_state.json` for observability only.

### No changes to:
- signal_bridge.py (ASK_FIRST rule; ATR Extension implemented as post-filter)
- composite_alpha_bt.py (backtest engine; already has all study parameters)
- existing addon system (winner_confirmation.py + AddOnExecutionPolicy)
- portfolio_state.json schema (read-only; version 2 unchanged)
- ORDER_LOCK_FILE dedup (D_EQ_SCALE BUY orders flow through same lock checks)

### Deployment steps:
1. Set `research_candidates.atr_extension.enabled: false` (stays OFF)
2. Set `research_candidates.vol_adj.enabled: false` (stays OFF)
3. Set `research_candidates.eq_scale_addon.enabled: false` (stays OFF)
4. Run `run_live_signal.py --dry` — verify output identical to pre-integration
5. Enable one feature at a time per shadow plan below

---

## D. Production Readiness Checklist

### Code quality:
- [x] All 3 features default OFF; no behavior change when disabled
- [x] All 3 injection points are FAIL_OPEN (exceptions caught and logged)
- [x] No changes to signal_bridge.py (ASK_FIRST constraint respected)
- [x] ATR Extension only blocks RSR-triggered SELL (trailing stop / emergency pass through)
- [x] D_EQ_SCALE: one addon per position lifecycle (entry_date key prevents re-entry duplicates)
- [x] D_EQ_SCALE: qty rounded to 100-share unit (no fractional shares)
- [x] D_EQ_SCALE flows through existing ORDER_LOCK_FILE dedup checks
- [x] D_VOL_ADJ FAIL_OPEN: returns max_positions=3 on TOPIX data error
- [x] No leverage: D_EQ_SCALE size = available_cash × 0.25 (within cash balance)

### Test regression:
- Baseline: 22 failed, 10000 passed
- Post-integration: 23 failed, 9999 passed
- Delta: +1 failure = date-sensitive API auth test (kabu Station 401 on Saturday 2026-06-27)
  This is NOT caused by code changes (verified: no research_candidate references in failing tests)
- **No code-induced regressions**

### Verification matrix (Study47):
- [x] A_BASELINE matches S5: CAGR=+19.37% (reference)
- [x] C_VOL_ADJ True OOS = -1.70pp exactly matches Study46 B_VOL_ADJ True OOS ✓
- [x] D_EQ_SCALE True OOS = +2.72pp exactly matches Study46 C_EQ_SCALE True OOS ✓
- [x] E_COMBINED ΔCAGR = +1.14pp Full IS (positive direction confirmed)
- [x] E_COMBINED MaxDD ΔDD = -1.56pp (within 2pp tolerance)
- [x] E_COMBINED True OOS = +3.40pp (OOS improvement confirmed)

### Remaining risk:
- ATR Extension effect is small (+0.19pp Full IS, +2.34pp True OOS); verify it fires correctly in live
- D_VOL_ADJ True OOS is slightly negative (-1.70pp) — vol_adj works best in WF context (IS-trained)
- Addon fires rarely at ¥3M capital (4-5 times in Full IS, 14-16 in True OOS) — small sample

---

## E. 30-Day Shadow Deployment Plan

### Phase 1 — Baseline shadow (Day 1-5, all OFF)
- Goal: confirm no regression in dry-run output vs pre-integration
- Action: run `run_live_signal.py --dry` daily; compare signal counts and order lists
- Pass criterion: identical to current production dry-run output

### Phase 2 — D_EQ_SCALE shadow (Day 6-15)
- Enable: `research_candidates.eq_scale_addon.enabled: true`
- Monitor: `runtime/eq_scale_addon_state.json` — verify addon fires only when gain ≥ ATR
- Verify: addon BUY appears in dry-run output; ORDER_LOCK_FILE correctly blocks same-day duplicates
- Monitor: `logs/` for EQ_SCALE_ADDON log entries; no crash / no unhandled exceptions

### Phase 3 — D_VOL_ADJ shadow (Day 6-15, parallel)
- Enable: `research_candidates.vol_adj.enabled: true`
- Monitor: `runtime/vol_adj_state.json` — daily regime decision logged
- Verify: max_positions=4 appears on calm TOPIX days; 3 on normal days
- Monitor: signal count changes (should see more BUY candidates on max_pos=4 days)

### Phase 4 — ATR Extension shadow (Day 16-25)
- Enable: `research_candidates.atr_extension.enabled: true`
- Monitor: `runtime/atr_ext_defer_state.json` — deferred exits tracked
- Verify: RSR exit SELL orders blocked when position is near highest close
- Monitor: position hold days extend on deferred exits; no infinite loop (7-day hard cap enforced)

### Phase 5 — E_COMBINED shadow (Day 26-30)
- All 3 features enabled simultaneously
- Run full dry-run for 5 consecutive trading days
- Verify: no duplicate orders, no execution rule violations, no unexpected position concentration
- Pass criterion: 5/5 clean dry-runs → promote to limited live

### Limited live transition (after Day 30):
- Enable live execution with ¥3M capital (Study19-20 procedure)
- Monitor for 10 trading days before full activation
- Revert immediately if: MaxDD breach, duplicate order, OR unexpected addon behavior
