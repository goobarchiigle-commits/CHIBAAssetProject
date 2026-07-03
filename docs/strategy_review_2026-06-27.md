# Strategy Review — CHIBAAssetProject
**Date:** 2026-06-27 (⚠ 廃止 — 最新版は `docs/strategy_review_2026-06-28.md` を参照)

> **2026-06-28 UPDATE**: Integrity Fix (intersection→union) + Study52 採用変更により以下の数値は廃止。  
> - CAGR 20.51% → **12.37%** (IS 2018-2024、D_ATR_EQ)  
> - VOL_ADJ 採用 → **除外** (Study52 REJECT)  
> - Production 構成 E_COMBINED → **D_ATR_EQ** (ATR Extension + EQ_SCALE)  
> 詳細は `docs/strategy_review_2026-06-28.md`。

---

**Date:** 2026-06-27  
**Phase:** 2 (Shadow Deployment)  
**Research Program:** Study01–Study49 COMPLETE

---

## Executive Summary

Current Production System: **LIVE READY**

Research program completed 2026-06-27 with Study47 (Production Candidate Verification). All adopted features (ATR Extension, D_VOL_ADJ, D_EQ_SCALE Addon) have passed:

- Walk-Forward Validation (5-fold, 2020–2024 OOS)
- Production Equivalence Audit (Study48: 4/4 PASS)
- Operational Resilience Audit (Study49: 62/62 PASS)

**Production System Performance (Full IS 2018–2024):**

> ⚠ 2026-06-28 Integrity Fix: `common_dates.intersection` → `.union` により4055.T (IPO 2020-08-11) による期間圧縮バグを修正。20.51% は 2020-2024 の 4.27yr 計算値であり廃止。正値は以下。

| Metric | Value |
|--------|-------|
| CAGR | **+12.19%** |
| Sharpe | 0.565 |
| MaxDD | -20.11% |
| Calmar | 0.606 |
| Avg Exposure | 30.7% |
| Trades (IS) | 274 |
| 実際期間 | 2018-01-01 ~ 2024-12-31 (6.84yr) |

**True OOS 2025 (E_COMBINED):**

| Metric | Value |
|--------|-------|
| CAGR | +11.90% |
| Sharpe | 1.056 |
| MaxDD | -10.60% |
| Calmar | 1.123 |
| Avg Exposure | 38.5% |

**Walk-Forward (D_COMBINED, 5 folds 2020–2024):**

| Metric | Value |
|--------|-------|
| WF Pass | 5/5 |
| Avg OOS CAGR | +23.03% |
| Avg OOS MaxDD | -16.20% |
| Avg OOS Calmar | 1.696 |
| Avg OOS Exposure | 49.5% |

**WF CAGR improvement vs S5 Baseline: +6.07pp**

---

## 1. Current Architecture

### 1.1 Universe

- **Dynamic Universe:** RSR42 + bear-regime filtering (dyn_rsr42_bear_rs0)
- **Auto-Promote:** PROBATION → GRADUATED → LIVE_UNIVERSE closed-loop
- **Size:** 42 symbols nominal; adaptive under bear regime

### 1.2 Entry

- **Signal:** Turtle Breakout (55d high)
- **Filter:** RSR Ranking (min_rsr = 75.0)
- **Min Hold:** 3d

### 1.3 Exit

- **Primary:** RSR Exit (threshold = 70.0, Study04/Regime-Aware WF validated)
- **Secondary:** ATR Trailing Exit (50d lookback, 3×ATR band) ← S5 component
- **Enhancement:** ATR Extension (Study40, +0.30pp) — RSR SELL post-filter; suppresses premature RSR exit when ATR momentum intact

### 1.4 Capital Allocation

- **Base:** Equal-weight sizing (cash ÷ remaining candidates)
- **Vol-Adaptive Cap (D_VOL_ADJ, Study41, +2.03pp):**
  - TOPIX 20d daily return std < 0.8% → max_positions = 4
  - TOPIX 20d std ≥ 0.8% → max_positions = 3 (bear protection)
  - Calm days ratio: 21.3% of all trading days
- **Winner Add-On (D_EQ_SCALE, Study45, +3.38pp):**
  - Trigger: unrealized gain ≥ 1×ATR20
  - Size: cash × 25% (equity-scaled)
  - Limit: 1 add-on per position lifecycle
  - Max single weight cap: max_single_weight × 1.5

### 1.5 Risk Controls

- **Circuit Breaker:** max_dd_limit = -15% (warn only; no auto-exit)
- **Position Limit:** max_positions = 3 (base); vol-conditional 4
- **Concentration:** max_single_weight = 25%
- **Slippage:** 0.1% (required in all order logic)
- **Commission:** 0.055% (required in all order logic)
- **Duplicate Order Guard:** same-symbol same-side same-day block
- **Order Reconciliation:** broker truth reconciliation before live exec
- **State Recovery:** FAIL_OPEN on all research_candidate state files

### 1.6 Implementation Files

| Module | File | Role |
|--------|------|------|
| Vol-Adaptive Cap | `src/research_candidate/vol_adj.py` | TOPIX vol → effective_max_pos |
| Equity-Scale Addon | `src/research_candidate/eq_scale_addon.py` | gain≥1×ATR → addon BUY |
| ATR Extension | `src/research_candidate/atr_extension.py` | RSR SELL post-filter |
| Config | `src/configs/strategy.yaml` | research_candidates block (all default OFF) |
| State Paths | `src/paths.py` | EQ_SCALE_ADDON_STATE_FILE, ATR_EXT_DEFER_STATE_FILE, VOL_ADJ_STATE_FILE |
| Live Entry | `src/run_live_signal.py` | 3 injection points (all FAIL_OPEN) |

---

## 2. Adopted Studies (Study40–Study49)

### Study40 — ATR Extension (2026-06-26)

**Decision: ADOPTED**  
**Script:** `src/backtest/study40_exit_continuation_wf_202606.py`

| Metric | Value |
|--------|-------|
| WF Result | 5/5 (Policy A) |
| ΔCAGR (WF OOS avg) | +0.30pp |
| 2022 Seg Protection | -5.60% (unchanged) |
| MaxDD Impact | 0.00pp |
| Risk Level | LOW |

Mechanism: ATR Extension delays RSR-triggered SELL when the position's ATR momentum remains intact. Identified as the lowest-risk exit improvement from a 6-policy sweep.

Other policies: B (RSR Persistence) FAIL — 2022 degradation confirmed; D (Partial+Runner) FAIL — WF 1/5; C/F PASS but tied with A.

---

### Study41 — D_VOL_ADJ Position Cap (2026-06-26)

**Decision: ADOPTED**  
**Script:** `src/backtest/study41_position_cap_wf_202606.py`

| Metric | Value |
|--------|-------|
| WF Result | 4/5 |
| ΔCAGR (WF OOS avg) | +2.03pp |
| 2022 Seg Protection | -5.60% (unchanged) |
| MaxDD Impact | 0.00pp (vs A_CONTROL) |
| Idle Cash (Baseline) | 58.6% |
| Days at Max Positions | 60.7% |
| Missed Entries fwd20d | +3.37% avg (63.6% positive) |

Vol threshold: TOPIX 20d std = 0.8% (21.3% calm-day ratio).

Fixed cap expansion (C_4 through C_7) all FAIL — 2022 bear degradation. Vol-conditional approach is the only mechanism preserving 2022 non-degradation while capturing +2.03pp.

---

### Study42 — Capital Constraint Archaeology (2026-06-26)

**Decision: ANALYSIS ONLY**  
**Script:** `src/backtest/study42_capital_constraint_archaeology_202606.py`

| Case | Capital | Lot | CAGR | MaxDD | LotRej |
|------|---------|-----|------|-------|--------|
| A (baseline) | ¥3M | 100 | +20.14% | -15.71% | 20 |
| B (large cap) | ¥30M | 100 | +23.46% | -16.19% | 0 |
| D (lot=1) | ¥3M | 1 | +24.08% | -16.24% | 0 |
| E (unconstrained) | ¥300M | 1 | +24.15% | -16.24% | 0 |

Attribution:
- Capital effect (B-A): +3.32pp = 146.9% of gap
- Lot effect (D-A): +3.94pp = 174.3% of gap
- Combined (E-A): +4.01pp = **177.4% of gap** → **CAPITAL_LOT_DOMINANT**

Lot-rejected trades (n=20): fwd20d avg = +5.59%, WR=80% — real alpha cost. Cliff: ¥14–30M for lot_reject=0.

---

### Study43A — Capital Saturation Risk Audit (2026-06-26)

**Decision: ANALYSIS ONLY**  
**Script:** `src/backtest/study43a_capital_saturation_202606.py`

| Capital | CAGR | Sharpe | MaxDD% | Calmar | LotRej | MaxDD(¥) |
|---------|------|--------|--------|--------|--------|----------|
| ¥3M | +20.14% | 0.859 | -15.71% | 1.282 | 20 | ¥471k |
| ¥10M | +21.20% | 0.878 | -16.02% | 1.323 | 16 | — |
| ¥15M | +23.26% | 0.957 | -16.05% | 1.449 | 2 | — |
| ¥20M | +22.71% | 0.931 | -16.32% | 1.392 | 0 | — |
| **¥30M** | **+23.46%** | **0.957** | **-16.19%** | **1.449** | **0** | **¥4,857k** |
| ¥50M | +23.80% | 0.968 | -16.23% | 1.467 | 0 | — |

Capital Zones:
- **Minimum Efficient Capital:** ¥20M (lot_reject=0 first achieved)
- **Sweet Spot:** ¥15M–¥30M (Calmar=1.449, +3.12–3.32pp CAGR)
- **Saturation:** ¥30M+ (diminishing returns, ΔCAGR < 0.5pp per step)

**Critical:** MaxDD% is nearly unchanged (¥3M vs ¥30M: −0.48pp) but absolute loss is 10× (¥471k → ¥4,857k). ¥30M scale requires preparation for ¥4.86M maximum drawdown exposure.

---

### Study44 — Lot Cost Ratio Walk-Forward (2026-06-26)

**Decision: REJECTED / EXHAUSTED**  
**Script:** `src/backtest/study44_lot_cost_ratio_wf_202606.py`

| Case | Ratio | WF | ΔCAGR | Overall |
|------|-------|-----|-------|---------|
| A_BASELINE | None | 4/5 | — | BASELINE |
| B_030 | 0.30 | 4/5 | -1.70pp | FAIL |
| C_035 | 0.35 | 4/5 | -1.62pp | FAIL |
| D_040 | 0.40 | 4/5 | -1.62pp | FAIL |
| E_045 | 0.45 | 4/5 | -1.62pp | FAIL |

Root cause: S5 config (ATR trailing exit ON) has fast capital turnover → only 3 lot-rejections in Full IS (not 20; the 20-rejection figure from Study42 was APRIL_REPRO_A config-specific). OOS ratio admission fires = 1–2 events only — statistically insufficient. Seg1 (2020 COVID): single ratio-admitted trade caused -8.49pp.

C/D/E identical result: at ¥3M capital, no eligible stocks exist in 0.35–0.45 price range given cash constraints. Effective price ceiling saturates at ratio 0.35.

---

### Study45 — D_EQ_SCALE Addon Expansion Walk-Forward (2026-06-27)

**Decision: ADOPTED**  
**Script:** `src/backtest/study45_addon_expansion_wf_202606.py`

**Phase 1 — Idle Cash Attribution (Full IS 2018–2024):**

| Metric | Value |
|--------|-------|
| Avg idle cash (all idle days) | 97.6% of capital |
| Q1: idle cash % when winner >1×ATR present | 31.8% |
| Q2: idle days fraction with addable winner | 28.3% |
| Q3: deployable idle cash avg / all days | 31.0% |

**Phase 2 — Policy Walk-Forward:**

| Case | Policy | WF | avgCAGR | ΔCAGR | seg3_2022 | worstDD | Overall |
|------|--------|----|---------|-------|-----------|---------|---------|
| A_CONTROL | NONE | 4/5 | +19.27% | — | -5.60% | -20.93% | BASELINE |
| B_SINGLE | B | 4/5 | +21.99% | +2.72pp | -5.11% | -20.93% | PASS |
| C_PYRAMID | C | 4/5 | +21.42% | +2.15pp | -4.70% | -20.93% | PASS |
| **D_EQ_SCALE** | **D** | **5/5** | **+22.65%** | **+3.38pp** | **-2.65%** | -20.93% | **PASS★** |
| E_VOL_ADJ | E | 4/5 | +20.82% | +1.55pp | -5.60% | -20.93% | PASS |
| F_HYBRID | F | 4/5 | +21.38% | +2.11pp | -5.60% | -20.93% | PASS |

**D_EQ_SCALE OOS Fold Detail:**

| Seg | OOS Year | CAGR | ΔvsControl | Addons | WF |
|-----|----------|------|------------|--------|----|
| 1 | 2020 | +22.65% | +4.19pp | 2 | PASS |
| 2 | 2021 | +6.80% | +3.00pp | 1 | PASS |
| 3 | 2022 | -2.65% | +2.95pp | 1 | PASS |
| 4 | 2023 | +47.21% | +1.89pp | 4 | PASS |
| 5 | 2024 | +39.24% | +4.88pp | 6 | PASS |

Parameters: `addon_policy="D"`, `addon_atr_mult=1.0`, `addon_size_frac=0.25`, `addon_max_per_pos=1`

---

### Study46 — VolAdj × Addon Interaction Walk-Forward 2×2 Factorial (2026-06-27)

**Decision: ADDITIVE — Both features confirmed independent**  
**Script:** `src/backtest/study46_voladj_addon_interaction_wf_202606.py`

| Case | VOL_ADJ | Addon | WF | avgCAGR | ΔCAGR | seg3_2022 | Calmar | Overall |
|------|---------|-------|----|---------|-------|-----------|--------|---------|
| A_BASELINE | ✗ | ✗ | 4/5 | +16.96% | — | -5.60% | 1.522 | BASELINE |
| B_VOL_ADJ | ✓ | ✗ | 4/5 | +18.99% | +2.03pp | -5.60% | 1.449 | +VOL |
| C_EQ_SCALE | ✗ | ✓ | 5/5 | +20.74% | +3.78pp | -2.65% | 1.794 | +ADN |
| **D_COMBINED** | ✓ | ✓ | **5/5** | **+23.03%** | **+6.07pp** | **-2.65%** | **1.696** | **PASS★** |

**Interaction Analysis:**

| Source | ΔCAGR |
|--------|-------|
| VOL_ADJ alone | +2.03pp |
| EQ_SCALE alone | +3.78pp |
| Combined (measured) | +6.07pp |
| Expected additive | +5.81pp |
| **Interaction term** | **+0.26pp → ADDITIVE** |

Full IS interaction: −0.01pp (completely additive).  
True OOS 2025 interaction: +0.74pp (mild synergy — VOL_ADJ+Addon combination holds vs standalone).

Note: B_VOL_ADJ True OOS 2025 = +6.80% (below baseline +8.50%) — VOL_ADJ alone is counterproductive in 2025 low-vol environment. Addon (C_EQ_SCALE) is the primary 2025 driver (+11.22%).

---

### Study47 — Production Candidate Verification Matrix (2026-06-27)

**Decision: ALL GATES PASS — Research Phase COMPLETE**  
**Script:** `src/backtest/study47_production_candidate_verification.py`

| Case | Full IS CAGR | Full IS MaxDD | True OOS 2025 | ΔCAGR (IS) |
|------|-------------|---------------|---------------|------------|
| A_BASELINE | +19.37% | -18.25% | +8.50% | — |
| B_ATR_EXT | +19.56% | -18.25% | +10.84% | +0.19pp |
| C_VOL_ADJ | +19.86% | -20.00% | +6.80% | +0.49pp |
| D_EQ_SCALE | +20.22% | -18.07% | +11.22% | +0.85pp |
| **E_COMBINED** | **+20.51%** | **-19.81%** | **+11.90%** | **+1.14pp** |

Gate Results:
- G1 Full IS ΔCAGR > 0: **PASS** (+1.14pp)
- G2 ΔMaxDD > −2pp: **PASS** (−1.56pp)
- G3 True OOS ΔCAGR > −5pp: **PASS** (+3.40pp)

Cross-validation with Study46: C_VOL_ADJ = −1.70pp (matches B_VOL_ADJ True OOS exactly ✓); D_EQ_SCALE = +2.72pp (matches C_EQ_SCALE True OOS exactly ✓).

---

### Study48 — Production Equivalence Audit (2026-06-27)

**Decision: PASS (4/4)**  
**Script:** `src/backtest/study48_equivalence_audit_2026-06-27.json`

| Success Criterion | Result |
|-------------------|--------|
| SC1: ATR Extension exit diffs are intentional feature fires | PASS — all 14 fires profitable (min pnl = +0.1%) |
| SC2: Addon count difference ≤1 | PASS — E_PROD47=5, D_REF46=4 (Δ=1) |
| SC3: D_REF46 IS 2018–2024 matches Study46 D_COMBINED | PASS — 20.70% (Δ=0.00pp), n_trades=205, MaxDD=−19.81% |
| SC4: ATR Extension IS contribution positive | PASS — E_COMBINED vs D_REF46 Full IS = +0.05pp |

Period gap explanation: Study46 WF avg = 23.03% vs Study47 E_COMBINED Full IS 2018–2025 = 17.11%. Difference = 6.22pp = WF fold-selection effect + 2025 year drag (−3.6pp). No implementation bug.

---

### Study49 — Operational Resilience Audit (2026-06-27)

**Decision: READY — 62/62 PASS**  
**Script:** `src/backtest/study49_operational_resilience_2026-06-27.json`

| Section | Tests | Result |
|---------|-------|--------|
| 1. Restart Recovery | 11 | PASS — state persistence, partial write recovery |
| 2. State Corruption Recovery | 17 | PASS — invalid JSON / empty / partial / null / array → all FAIL_OPEN |
| 3. Rollback Recovery | 7 | PASS — Phase1→2→3→rollback idempotent |
| 4. Feature Flag Matrix | 7 | PASS — all 7 feature combinations confirmed |
| 5. Order ACK Reconciliation | 6 | PASS — engine/inflight/OrderInstruction/dup_guard |
| 6. Monitoring Completeness | 14 | PASS — all required fields daily+weekly |

Success Criteria:
- ✅ Zero state corruption (FAIL_OPEN confirmed)
- ✅ Zero duplicate addon (lifecycle state persistence confirmed)
- ✅ Zero unreconciled order (reconciliation engine + inflight confirmed)
- ✅ Successful rollback (idempotent confirmed)
- ✅ Successful restart recovery (state persistence confirmed)
- ✅ Complete monitoring coverage (daily + weekly all fields confirmed)

**Production Rollout Plan:**
```
Phase 1: eq_scale_addon=ON      → python tools/rollout_phase.py --phase 1
Phase 2: +vol_adj=ON            → python tools/rollout_phase.py --phase 2
Phase 3: +atr_extension=ON      → python tools/rollout_phase.py --phase 3
Rollback: all=OFF               → python tools/rollout_phase.py --rollback
```

**Monitoring (Task Scheduler registered):**
- Weekday 18:00: `rollout_monitor_daily.py`
- Friday 18:30: `rollout_monitor_weekly.py`

---

## 3. Capital Scaling Analysis

Based on Study43A full capital sweep (Full IS 2018–2024, lot=100, max_pos=3).

| Capital | CAGR | Sharpe | MaxDD% | Calmar | LotRej | MaxDD(¥) | WF Est CAGR |
|---------|------|--------|--------|--------|--------|----------|-------------|
| **¥3M** (current) | +20.14% | 0.859 | -15.71% | 1.282 | 20 | ¥471k | ~25–26% |
| ¥10M | +21.20% | 0.878 | -16.02% | 1.323 | 16 | — | — |
| ¥15M | +23.26% | 0.957 | -16.05% | 1.449 | 2 | — | — |
| **¥20M** | +22.71% | 0.931 | -16.32% | 1.392 | 0 | — | ~27–28% |
| **¥30M** | **+23.46%** | **0.957** | **-16.19%** | **1.449** | 0 | ¥4,857k | **~27–30%** |
| ¥50M | +23.80% | 0.968 | -16.23% | 1.467 | 0 | — | — |

WF CAGR estimates incorporate +6.07pp from D_COMBINED.

**Key findings:**
- ¥5M regression: ¥3M→¥5M CAGR −0.11pp (same lot constraint, adverse sizing change) — avoid ¥5M target
- Minimum Efficient Capital: **¥20M** (first lot_reject=0)
- Sweet Spot: **¥15M–¥30M** (Calmar=1.449, primary driver = lot constraint removal = 83% of gain)
- Saturation: ¥30M+ (ΔCAGR < 0.5pp per step)
- ¥15M > ¥20M anomaly (non-monotonic): ¥15M CAGR 23.26% > ¥20M 22.71% — lot=2 residual favorable coincidence

**Path to 30% (confirmed ceiling analysis):**

```
S5 Baseline                           = +19.37%
+ Study40 ATR Extension (WF 5/5)     = +0.30pp  →  19.67%
+ Study46 D_COMBINED (WF 5/5)        = +6.07pp  →  25.74%
  Current production ceiling @ ¥3M   = ~25–26%
+ Capital ¥20M (lot release)         ≈ +2.61pp  →  ~28.35%
+ Capital ¥30M (lot full release)    ≈ +3.32pp  →  ~29.06%
  Cash ceiling @ ¥30M                = ~27–30%
+ Leverage 1.3× @ ¥30M              × 1.3       →  ~38%
```

Cash-only ceiling: ~26% at ¥3M, ~28–30% at ¥20–30M. The only confirmed path beyond 30% is Leverage 1.3× at adequate capital scale.

---

## 4. Annual Performance Profile

Source: Study46 D_COMBINED Walk-Forward OOS per year (2020–2024); Study47 E_COMBINED True OOS (2025). Full IS aggregate 2018–2024 from Study47 E_COMBINED. Years 2018–2019 are training-period-only (always in IS across all 5 WF folds; no isolated OOS measurement).

### Current Production System (E_COMBINED)

#### Full IS Aggregate 2018–2024

| Metric | Value |
|--------|-------|
| CAGR | +20.51% |
| Sharpe | 0.880 |
| MaxDD | -19.81% |
| Calmar | 1.035 |
| Trades | 205 |
| Avg Exposure | 37.4% |
| Addons (IS) | 5 |

#### Walk-Forward OOS Annual (D_COMBINED, with +ATR Extension = E_COMBINED equivalent)

| Year | CAGR | MaxDD | Sharpe | Trades | Avg Exposure | Addons | Notes |
|------|------|-------|--------|--------|--------------|--------|-------|
| 2018 | — | — | — | — | — | — | IS only (all folds) |
| 2019 | — | — | — | — | — | — | IS only (all folds) |
| 2020 | +22.65% | -11.17% | 0.785 | 12 | 52.3% | 2 | COVID recovery |
| 2021 | +6.80% | -20.37% | 0.366 | 53 | 48.4% | 1 | Sideways/correction |
| 2022 | -2.65% | -20.93% | 0.057 | 43 | 50.3% | 1 | Bear market |
| 2023 | +49.10% | -13.49% | 1.766 | 55 | 51.7% | 3 | Strong bull |
| 2024 | +39.24% | -15.06% | 1.890 | 37 | 44.9% | 6 | Bull continuation |
| **2025** | **+11.90%** | **-10.60%** | **1.056** | **45** | **38.5%** | **16** | True OOS (E_COMBINED) |

**WF Summary (D_COMBINED, 5 folds 2020–2024):**

| Metric | Avg OOS |
|--------|---------|
| CAGR | +23.03% |
| MaxDD | -16.20% |
| Sharpe | 0.973 |
| Calmar | 1.696 |
| Exposure | 49.5% |
| Idle Cash | 56.8% |

#### S5 Baseline Annual (for comparison — A_BASELINE WF OOS)

| Year | CAGR | MaxDD | Trades | Avg Exposure |
|------|------|-------|--------|--------------|
| 2020 | +15.66% | -11.41% | 11 | 44.8% |
| 2021 | +6.08% | -20.60% | 54 | 47.9% |
| 2022 | -5.60% | -20.93% | 43 | 49.8% |
| 2023 | +37.59% | -9.17% | 50 | 45.1% |
| 2024 | +31.08% | -14.72% | 34 | 41.8% |
| 2025 (True OOS) | +8.50% | -8.07% | 44 | 29.4% |

**Year-over-year improvement (D_COMBINED vs A_BASELINE):**
- 2020: +7.0pp (+22.65% vs +15.66%)
- 2021: +0.7pp (+6.80% vs +6.08%)
- 2022: +2.95pp (−2.65% vs −5.60%)
- 2023: +11.5pp (+49.10% vs +37.59%)
- 2024: +8.2pp (+39.24% vs +31.08%)
- 2025: +3.4pp (+11.90% vs +8.50%) [True OOS]

---

## 5. Exposure Analysis

Source: Study46 WF OOS per-fold and True OOS 2025 data.

### Exposure Metrics

| Period | System | Avg Exposure | Avg Positions | Idle Cash | Days at Max Pos |
|--------|---------|-------------|---------------|-----------|-----------------|
| Full IS 2018–2024 | A_BASELINE | 34.9% | 2.42 | 100.6% | 60.3% |
| Full IS 2018–2024 | E_COMBINED | 37.4% | 2.52 | 98.6% | 55.1% |
| WF OOS avg 2020–2024 | A_BASELINE | 45.9% | 2.40 | 58.6% | — |
| WF OOS avg 2020–2024 | D_COMBINED | 49.5% | 2.45 | 56.8% | — |
| True OOS 2025 | A_BASELINE | 29.4% | 1.62 | 72.2% | 24.3% |
| True OOS 2025 | E_COMBINED | 38.5% | 1.70 | 63.0% | — |

### Per-Year Exposure (D_COMBINED WF OOS)

| Year | Avg Exposure | Avg Positions | Idle Cash | Days at Max Pos |
|------|-------------|---------------|-----------|-----------------|
| 2020 | 52.3% | 2.43 | 50.0% | 54.6% |
| 2021 | 48.4% | 2.71 | 56.7% | 69.0% |
| 2022 | 50.3% | 2.43 | 47.2% | 57.8% |
| 2023 | 51.7% | 2.61 | 62.1% | 54.1% |
| 2024 | 44.9% | 2.07 | 68.1% | 31.4% |

### Key Observations

- Average WF OOS exposure: 49.5% — strategy deploys approximately half of capital on average
- Idle cash (WF): 56.8% — opportunity for addon deployment exists on >28% of idle days
- D_VOL_ADJ calm-day ratio: 21.3% — 4th slot available on approximately 1-in-5 trading days
- Addon (D_EQ_SCALE) 2025 True OOS: 16 addons fired vs 5 in Full IS — addon frequency increases with portfolio gains

---

## 6. Risk Assessment

### 6.1 Known Market Risks

| Risk | Category | Mitigation Status |
|------|----------|-------------------|
| Bear market regime | Market | 2022 WF OOS = −2.65% (D_COMBINED); protected vs −5.60% baseline |
| Extended drawdown | Market | max_dd_limit=−15% warning gate; max observed WF OOS DD=−20.93% (2022) |
| Universe evolution | Market | Dynamic universe auto-promote/demote; RSR42 base filter |
| Liquidity deterioration | Market | min_lot=100 constraint; high-price stocks excluded at ¥3M |
| Correlation spike (bear) | Market | TOPIX vol gate suppresses 4th position during high-vol regimes |

### 6.2 Mitigated Implementation Risks

| Risk | Mitigation |
|------|-----------|
| State corruption | FAIL_OPEN on all 3 research_candidate state files |
| Restart mid-execution | Study49 SC1 — 11 restart tests PASS |
| Duplicate addon execution | Lifecycle state (1 addon per position lifecycle); Study49 SC5/SC6 |
| Unreconciled order | Broker truth reconciliation before each live session |
| Duplicate order submission | same-symbol same-side same-day block |
| Lookahead bias | TimeSeriesSplit-only CV; train_only scaler fit; pipeline mandatory |
| Overfit | Walk-forward 5-fold mandatory; min_trade_required=5; Sharpe >3.0 flags |
| Feature rollback failure | Phase1→2→3→rollback idempotent confirmed (Study49 SC3) |
| Monitoring blind spots | 14/14 monitoring fields confirmed daily+weekly (Study49 SC6) |

### 6.3 Capital Risk at Current Scale (¥3M)

| Metric | Value |
|--------|-------|
| Historical max WF OOS MaxDD | -20.93% (2022) |
| P95 estimated max loss | ¥140k (Study14 pre-live audit) |
| Worst year WF | 2022: −2.65% (D_COMBINED) |
| DD warning threshold | −15% (circuit breaker warning only) |

---

## 7. Operational Readiness

| Category | Status |
|----------|--------|
| **Research** | **COMPLETE** (Study01–Study49) |
| **Implementation** | **COMPLETE** (research_candidate package + strategy.yaml + run_live_signal.py 3 injection points) |
| **Equivalence Validation** | **COMPLETE** (Study48: 4/4 PASS; ΔCAGR=0.00pp vs reference) |
| **Operational Resilience** | **COMPLETE** (Study49: 62/62 PASS) |
| **Deployment Status** | **LIVE READY** → Shadow Deployment Active |

### Shadow Deployment Plan

| Phase | Days | Features Active | Action |
|-------|------|-----------------|--------|
| 1 | 1–5 | all OFF | baseline shadow (no new feature) |
| 2–3 | 6–15 | EQ_SCALE + VOL_ADJ | parallel shadow monitoring |
| 4 | 16–25 | + ATR Extension | ATR shadow |
| 5 | 26–30 | all ON (E_COMBINED) | full shadow → limited live assessment |

Enable command: `python tools/rollout_phase.py --phase {1|2|3|rollback}`

### Morning Routine (required)

1. API port check: 18080
2. Dry run (required before any live execution)
3. Signal summary in Japanese (required)
4. Live execution (allowed after dry run)
5. Execution log report: symbol, shares, amount, BUY/SELL
6. Drawdown monitoring: immediate warning if −15% reached

---

## Appendix A — Historical Evolution

### System Evolution Path

```
Pre-Study: RSR42 base system
│
├── Study01–08:  Entry signal sweep (all REJECT / CLOSED)
├── Study09:     Standalone RSR90 → S5 precursor confirmed
├── Study10–11:  Capital governance + standalone validation
├── Study12–16:  Pre-live audit / deployment readiness / authority gate
├── Study17–20:  Live sandbox → limited live activation
│
├── Study21–24:  Exit attribution / signal failure decomposition (CLOSED)
├── Study25–28:  Portfolio geometry / allocation (EXHAUSTED)
│
├── Study32:     ATR Risk Sizing → REMOVE (−8.8pp penalty removed → equal weight)
├── Study33:     Post-ATR Removal: new baseline C_PROD_MINUS_ATR confirmed
├── Study34:     Drawdown attribution: dynamic universe = primary DD source (69.4%)
├── Study35:     PROD_MINUS_ATR_MINUS_MTF WF → PROMOTE_D (MTF = alpha-destructive −1.56pp)
│
│   → S5 Baseline defined: CAGR=+19.37% / Sharpe=0.855 / MaxDD=−18.25%
│     Components: Dynamic Universe + Equal-weight sizing + ATR Trailing Exit + RSR Exit 70.0
│
├── Study36–38:  Engine forensics / APRIL_REPRO audit (no new alpha)
├── Study39:     Alpha expansion audit: research map closed; leverage = only >30% path
│
├── Study40:     ATR Extension → ADOPTED (+0.30pp, WF 5/5, LOW risk)
│               19.37% → 19.67%
│
├── Study41:     D_VOL_ADJ → ADOPTED (+2.03pp, WF 4/5)
│               19.67% → 21.70%
│
├── Study42:     Capital archaeology → CAPITAL_LOT_DOMINANT (analysis)
├── Study43A:    Capital saturation → ¥20M efficient, ¥30M sweet spot (analysis)
├── Study44:     Lot cost ratio → REJECTED (S5 config: 3 lot-rejects only, not 20)
│
├── Study45:     D_EQ_SCALE Addon → ADOPTED (+3.38pp, WF 5/5)
│               21.70% → ~25.08%
│
├── Study46:     Interaction audit → ADDITIVE (+0.26pp interaction bonus)
│               Combined WF effect: +6.07pp vs S5 baseline → 19.37% + 6.07 = 25.44%
│
├── Study47:     Production Candidate Verification → ALL GATES PASS
│               E_COMBINED Full IS: +20.51% / True OOS 2025: +11.90%
│               Research Phase → COMPLETE
│
├── Study48:     Production Equivalence Audit → PASS (4/4); Δ=0.00pp vs reference
│
└── Study49:     Operational Resilience Audit → READY (62/62 PASS)
                 → LIVE READY
```

### Feature Contribution Summary

| Feature | Study | Decision | WF ΔCAGR | Mechanism |
|---------|-------|----------|----------|-----------|
| Equal-weight sizing | Study32 | ADOPTED | +4.12pp (IS) | Replace ATR risk sizing |
| Dynamic Universe | Study33 | ADOPTED | +0.77pp (IS) | RSR42 + dyn bear filter |
| MTF Filter removal | Study35 | ADOPTED | +1.56pp (IS) | MTF was alpha-destructive |
| ATR Trailing Exit | S5 | ADOPTED | +0.30pp (IS) | S5 component |
| RSR Exit 70.0 | Study04/WF | ADOPTED | +2.72pp (WF) | RSR-based exit threshold |
| **ATR Extension** | **Study40** | **ADOPTED** | **+0.30pp** | RSR SELL post-filter |
| **D_VOL_ADJ** | **Study41** | **ADOPTED** | **+2.03pp** | TOPIX vol-conditional max_pos |
| **D_EQ_SCALE Addon** | **Study45** | **ADOPTED** | **+3.38pp** | Winner gain-based addon BUY |
| **Interaction bonus** | **Study46** | **CONFIRMED** | **+0.26pp** | VOL_ADJ × Addon additive |
| **Total (Study40–46)** | | | **+6.07pp** | WF OOS average |

### Rejected / Exhausted Research

| Theme | Studies | Outcome |
|-------|---------|---------|
| Entry signal improvement | Study01–08, Study22–24 | All REJECT / CLOSED |
| Portfolio geometry | Study25–28 | EXHAUSTED (best Calmar +0.091) |
| CAP state regime | Study08 series | REJECTED WF 3/5 |
| MSW / P4 leverage | Study13 series | REJECTED |
| Exit RSR threshold tuning | Study21, Study23 | CLOSED |
| MTF filter | Study33, Study35 | REMOVED (alpha-destructive) |
| CB bypass | Study38 | CLOSED (0% explanation of gap) |
| Lot cost ratio | Study44 | REJECTED / EXHAUSTED |
| Fixed position cap expansion | Study41 (C_4–C_7) | REJECTED (2022 degradation) |

---

## Appendix B — Final Audit 2026-06-27 (E_COMBINED 最終真値確定)

### B.1 数値出典整理

従来レポートの3数値はすべて **2020-08-11 始点** で計算されていた。  
原因: RSR42ユニバースに含まれる **4055.T (TDC SOFTWARE)** の上場日 = 2020-08-11。  
`composite_alpha_bt.py` の `common_dates = intersection of all active_syms' indices` により全42銘柄の共通日付が2020-08-11に制約される。

| 報告数値 | 実際の計算期間 | IS/OOS | Fixed/WF | 算出元 |
|------|------|------|------|------|
| **20.51%** | 2020-08-11〜2024-12-31 (4.27yr) | IS | Fixed | `study47_prod_candidate_2026-06-27.json` → `FULL_IS.E_COMBINED.cagr` |
| **17.11%** | 2020-08-11〜2025-12-31 (~5.46yr) | IS+2025 | Fixed | `study48_equivalence_audit_2026-06-27.json` → `section_A.E_PROD47.cagr` |
| **23.03%** | WF OOS avg (各年 2020-2024) | OOS | WF 5-fold | `study46_voladj_addon_interaction_wf_202606_2026-06-27.json` → `interaction_analysis.wf_oos_avg.D_combined` |

注: 20.51%/17.11%は2018-2019の低パフォーマンス期間が除外されており過大評価。  
23.03%はFold1 OOS=2020が 2020-08-11〜2020-12-31 の約100営業日分の年率換算のため統計的過大リスクあり。

---

### B.2 監査結果

| 項目 | 判定 | 詳細 |
|------|------|------|
| Future Leak | **PASS** | signal=close[i], 約定=open[i+1]。RSRはshift()で過去のみ参照。lookahead なし確認。 |
| Survivorship Bias | **WARNING** | RSR42 = TOPIX100から2018-2024実績でフィルタ選定 (universe.yaml明記)。4055.T はIPO後成績で採用。 |
| Selection Bias | **WARNING** | universe.yaml: 「同一期間で選定と評価→銘柄選択バイアスの可能性あり」と自己申告済み。 |
| Walk Forward | **PASS ⚠Fold1** | IS/OOS時系列分離は正しい。Fold1 IS = 2018-2019 実質空 (4055.T制約)、OOS = 4ヶ月分のみ。Fold2-5正常。 |
| Implementation Drift | **PASS** | Study48 SC3: D_REF46 IS 2018-2024 = 20.70% = Study46 D_COMBINED 完全一致 (Δ=0.00pp)。 |
| Execution Reality | **PASS** | 寄付始値約定・スリッページ0.1%・手数料0.055%・ギャップ=始値受け・全件実装確認。 |

---

### B.3 2020-08-11 開始の原因

**ユニバース制約 (4055.T 上場日制約)**

RSR42ユニバース (rsr_universe_42.csv) の **4055.T** のみ東証上場 = 2020-08-11。  
他41銘柄はすべて2015-11以前から上場。  
`common_dates = intersection` 演算により全バックテストが2020-08-11以前のデータを持たない。

---

### B.4 コロナショック DD (RSR41 真値)

4055.T 除外時のみ測定可能 (RSR42では期間外)。

| 指標 | 値 |
|------|------|
| Peak | ¥3,003,533 (2020-02-12) |
| Trough | ¥2,507,373 (2020-04-01) |
| **DD%** | **-16.52%** |

---

### B.5 正式成績 — RSR41 真値 (4055.T 除外, 41銘柄)

#### 2018-2024 vs 2018-2025 比較

| 指標 | 2018-2024 | 2018-2025 |
|------|------|------|
| 計算期間 | 2018-01-01〜2024-12-31 (6.84yr) | 2018-01-01〜2025-12-30 (7.81yr) |
| **CAGR** | **12.18%** | **11.03%** |
| MaxDD | -18.02% | -18.02% |
| Sharpe | 0.569 | 0.547 |
| Calmar | 0.676 | 0.612 |
| Trades | 267 | 316 |
| WinRate | 53.9% | 54.1% |
| AvgExp | 32.5% | 30.4% |

#### ベンチマーク比較 (2018-2024 / 2018-2025)

| | E_COMBINED RSR41 | TOPIX (1306.T) | Nikkei225 | E_COMBINED 超過 |
|------|------|------|------|------|
| CAGR 2018-2024 | 12.18% | 8.99% | 8.11% | +3.19pp / +4.07pp |
| CAGR 2018-2025 | 11.03% | 11.04% | 10.33% | -0.01pp / +0.70pp |
| MaxDD | -18.02% | -32.68% | -31.80% | **-14.7pp 改善** |

2025はベンチマーク超過がほぼゼロ。リスク調整後 (MaxDD-14.7pp改善) では引き続き優位。

#### 年次リターン (RSR41 2018-2025)

| 年 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|------|------|------|------|------|------|------|------|------|
| Return | -3.6% | +1.4% | +9.5% | +18.7% | +9.3% | +29.8% | +17.6% | +3.1% |
| YearMaxDD | -12.7% | -18.0% | -16.5% | -17.2% | -9.2% | -9.2% | -8.7% | -4.0% |

---

### B.6 最終採用構成

**E_COMBINED (Study47 Production Candidate) — 正式定義**

| コンポーネント | パラメータ | 実装ファイル |
|------|------|------|
| Dynamic Universe (dyn_rsr42_bear_rs0) | RSR42 + bear sector exclude | `strategy/universe.py` |
| RSR Exit | threshold = 70.0 | `composite_alpha_bt.py:1043` |
| ATR Trailing Stop | 3×ATR20 | `composite_alpha_bt.py:998-1009` |
| Equal Weight | sizing_mode="existing" | — |
| ATR Extension (exit_policy=A) | atr_mult=1.0, defer_days=5 | `research_candidate/atr_extension.py` |
| D_VOL_ADJ | topix_vol_threshold=0.008, calm_max_pos=4 | `research_candidate/vol_adj.py` |
| D_EQ_SCALE | atr_mult=1.0, size_frac=0.25, max_per_pos=1 | `research_candidate/eq_scale_addon.py` |

**真値サマリ:**

| | RSR42 (従来報告) | RSR41 (真値) | 差異原因 |
|------|------|------|------|
| 計算開始日 | 2020-08-11 | 2018-01-01 | 4055.T除外 |
| IS 2018-2024 CAGR | **20.51%** (= 2020-2024 の数値) | **12.18%** | 2018-2019 低迷期間の包含 |
| IS 2018-2025 CAGR | **17.11%** (= 2020-2025 の数値) | **11.03%** | 同上 |
| MaxDD | -19.81% | -18.02% | コロナショック含む |
| コロナショック | 期間外 | **-16.52%** | 測定可能 |

---

### B.7 出力ファイル一覧

| ファイル | 内容 | 行数 | 期間 |
|------|------|------|------|
| `reports/dashboard_data/equity_curve_full.csv` | RSR41 E_COMBINED 日次資産 | 1,967行 | 2018-01-01〜2025-12-30 |
| `reports/dashboard_data/trades_full.csv` | 全クローズトレード | 316行 | 2018〜2025 |
| `reports/dashboard_data/exposure_full.csv` | 日次エクスポージャー | 1,967行 | 2018-01-01〜2025-12-30 |
| `reports/strategy_dashboard_ecombined_full.png` | 最終ダッシュボード (TOPIX/Nikkei比較) | — | 2018〜2025 |
| `reports/dashboard_data/ecombined_full_summary.json` | 集計サマリーJSON | — | — |

---

*Document updated: 2026-06-27 (Appendix B: Final Audit)*  
*Based on: Study47 research_state.md / study46 JSON / study47 JSON / Study48–49 audit results*  
*Appendix B: E_COMBINED Final Audit — RSR41 True Value Determination*  
*Next milestone: 30-day shadow deployment completion → Phase 1 limited live activation*
