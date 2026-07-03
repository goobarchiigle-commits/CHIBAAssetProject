"""
portfolio/capital_efficiency.py

Capital Efficiency Enhancement Module
- Missed trade logging with forward-return fill (no lookahead)
- Rolling kNN-based expected alpha estimation (60-day window, decay half-life=20)
- Regime vol adjustment (smoothed z-score)
- CE-adjusted position sizing and entry score velocity

Leakage constraints:
  - kNN query uses only t < T observations
  - gap_pct uses today's observed open (known at EOD decision time)
  - forward_return_1d filled next day (t+1 close vs t close)
  - score_velocity uses RSR from vel_window days ago
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #
@dataclass
class _BufferEntry:
    date_idx:   int
    score:      float
    gap_pct:    float
    fill_prob:  float
    fwd_ret:    float


@dataclass
class _PendingEntry:
    symbol:    str
    score:     float
    gap_pct:   float
    fill_prob: float


# ------------------------------------------------------------------ #
# Core module
# ------------------------------------------------------------------ #
class CapitalEfficiencyModule:
    """
    Rolling kNN estimator for capital efficiency enhancement.

    All methods are designed to be called once per trading day in order:
      1. on_day_open(i, market_return)
      2. fill_forward_returns(i, close_today, close_prev)
      3. estimate_expected_alpha(i, score, gap_pct, fill_prob)   [per candidate]
      4. log_missed_trade(i, sym, score, gap_pct, fill_prob)     [per miss]

    Parameters
    ----------
    window      : rolling lookback days (default 60)
    half_life   : exponential decay half-life (default 20)
    k           : kNN neighbours (default 20)
    min_samples : minimum buffer size before estimating (default 10)
    clip_lo     : lower bound for expected_alpha (default -0.01)
    clip_hi     : upper bound for expected_alpha (default +0.02)
    vel_window  : RSR velocity look-back days (default 5)
    smoothing   : EMA alpha for output damping (default 0.7)
    """

    def __init__(
        self,
        window:      int   = 60,
        half_life:   int   = 20,
        k:           int   = 20,
        min_samples: int   = 10,
        clip_lo:     float = -0.01,
        clip_hi:     float =  0.02,
        vel_window:  int   = 5,
        smoothing:   float = 0.7,
    ) -> None:
        self.window      = window
        self.half_life   = half_life
        self.k           = k
        self.min_samples = min_samples
        self.clip_lo     = clip_lo
        self.clip_hi     = clip_hi
        self.vel_window  = vel_window
        self.smoothing   = smoothing

        # Rolling buffer of completed observations (filled forward returns)
        self._buf: deque[_BufferEntry] = deque()
        # date_idx -> list of pending (not yet filled) missed trades
        self._pending: dict[int, list[_PendingEntry]] = {}
        # Market absolute returns for vol z-score
        self._vol_hist: deque[float] = deque(maxlen=252)
        # EMA state for output smoothing
        self._prev_ea: float = 0.0
        # Per-day log for external inspection
        self.last_log: dict = {}

    # ---------------------------------------------------------------- #
    # Daily hooks (call in order each bar)
    # ---------------------------------------------------------------- #
    def on_day_open(self, date_idx: int, market_return: float) -> None:
        """Update market vol history. Call before any estimation on day i."""
        self._vol_hist.append(abs(float(market_return)))

    def fill_forward_returns(
        self,
        date_idx:   int,
        close_today: dict[str, float],
        close_prev:  dict[str, float],
    ) -> None:
        """
        Fill forward_return_1d for pending missed trades from (date_idx - 1).
        Adds filled entries to rolling buffer. Prunes entries outside window.
        """
        prev_idx = date_idx - 1
        if prev_idx in self._pending:
            for pe in self._pending[prev_idx]:
                prev_c = close_prev.get(pe.symbol, 0.0)
                cur_c  = close_today.get(pe.symbol, 0.0)
                if prev_c > 0 and cur_c > 0:
                    fwd = cur_c / prev_c - 1.0
                    self._buf.append(_BufferEntry(
                        date_idx  = prev_idx,
                        score     = pe.score,
                        gap_pct   = pe.gap_pct,
                        fill_prob = pe.fill_prob,
                        fwd_ret   = fwd,
                    ))
            del self._pending[prev_idx]

        # Prune buffer beyond rolling window
        cutoff = date_idx - self.window
        while self._buf and self._buf[0].date_idx < cutoff:
            self._buf.popleft()

    def log_missed_trade(
        self,
        date_idx: int,
        symbol:   str,
        score:    float,
        gap_pct:  float,
        fill_prob: float,
    ) -> None:
        """Record a skipped candidate. Forward return filled next bar."""
        pe = _PendingEntry(symbol=symbol, score=score, gap_pct=gap_pct, fill_prob=fill_prob)
        if date_idx not in self._pending:
            self._pending[date_idx] = []
        self._pending[date_idx].append(pe)

    # ---------------------------------------------------------------- #
    # Expected alpha estimation
    # ---------------------------------------------------------------- #
    def estimate_expected_alpha(
        self,
        date_idx:  int,
        score:     float,
        gap_pct:   float,
        fill_prob: float,
    ) -> tuple[float, float, str]:
        """
        Estimate expected 1-day alpha for a candidate using past missed trades.

        Returns
        -------
        (expected_alpha, confidence, regime_state)
        expected_alpha is clipped to [clip_lo, clip_hi].
        """
        n = len(self._buf)
        confidence = min(1.0, n / 100.0)

        if n < self.min_samples:
            self.last_log = {"ea": 0.0, "conf": confidence, "regime": "neutral", "n": n}
            return 0.0, confidence, "neutral"

        buf_list = list(self._buf)
        X = np.array([[b.score, b.gap_pct, b.fill_prob] for b in buf_list], dtype=np.float64)
        y = np.array([b.fwd_ret for b in buf_list], dtype=np.float64)
        ages = np.array([date_idx - b.date_idx for b in buf_list], dtype=np.float64)
        ages = np.maximum(ages, 0.0)

        # Exponential decay weights
        decay_w = np.power(0.5, ages / self.half_life)

        # Normalize features for distance
        feat_std = X.std(axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
        X_norm = X / feat_std
        q_norm = np.array([score, gap_pct, fill_prob], dtype=np.float64) / feat_std

        # kNN
        k_eff = min(self.k, n)
        dists = np.linalg.norm(X_norm - q_norm, axis=1)
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]

        dist_w = 1.0 / (dists[nn_idx] + 1e-4)
        combined_w = dist_w * decay_w[nn_idx]
        total_w = combined_w.sum()
        if total_w < 1e-12:
            raw_alpha = 0.0
        else:
            combined_w /= total_w
            raw_alpha = float(np.dot(combined_w, y[nn_idx]))

        # Confidence scaling
        raw_alpha *= confidence

        # Regime adjustment via market vol z-score
        regime_state = self._apply_regime(raw_alpha)
        regime_state, raw_alpha = regime_state  # unpack

        # EMA damping to prevent sudden jumps
        alpha_ema = self.smoothing * raw_alpha + (1.0 - self.smoothing) * self._prev_ea
        self._prev_ea = alpha_ema

        ea = float(np.clip(alpha_ema, self.clip_lo, self.clip_hi))
        self.last_log = {"ea": ea, "conf": confidence, "regime": regime_state, "n": n}
        return ea, confidence, regime_state

    def _apply_regime(self, alpha: float) -> tuple[str, float]:
        """Apply smoothed vol z-score regime adjustment. Returns (state, adjusted_alpha)."""
        if len(self._vol_hist) < 20:
            return "neutral", alpha
        arr = np.array(self._vol_hist)
        mu, sd = arr.mean(), arr.std()
        if sd < 1e-8:
            return "neutral", alpha
        vol_z = (arr[-1] - mu) / sd
        if vol_z > 0.5:
            return "high_vol", alpha * 1.3
        if vol_z < -0.5:
            return "low_vol", alpha * 0.7
        return "neutral", alpha

    # ---------------------------------------------------------------- #
    # Sizing helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def size_scale(expected_alpha: float) -> float:
        """
        Partial-execution scale factor replacing hard binary skip.
        scale = clip(1 + ea * 5, 0.3, 1.0)
        """
        return float(np.clip(1.0 + expected_alpha * 5.0, 0.3, 1.0))

    @staticmethod
    def alpha_adj(expected_alpha: float) -> float:
        """Weight adjustment: (1 + ea), used in CE weight computation."""
        return 1.0 + float(expected_alpha)

    def compute_ce_weights(
        self,
        scores:          dict[str, float],
        expected_alphas: dict[str, float],
        max_weight:      float = 0.40,
    ) -> dict[str, float]:
        """
        CE position weights = softmax(score) * (1 + expected_alpha), normalized.

        Parameters
        ----------
        scores          : {sym: composite_score}
        expected_alphas : {sym: expected_alpha}
        """
        if not scores:
            return {}
        syms = list(scores.keys())
        s_arr = np.array([scores[s] for s in syms], dtype=np.float64)
        a_arr = np.array([expected_alphas.get(s, 0.0) for s in syms], dtype=np.float64)

        # Numerically stable softmax
        s_arr -= s_arr.max()
        base_w = np.exp(s_arr)
        base_w /= base_w.sum()

        # Alpha adjustment and normalize
        adj_w = base_w * (1.0 + a_arr)
        total = adj_w.sum()
        if total <= 1e-12:
            adj_w = np.ones(len(syms)) / len(syms)
        else:
            adj_w /= total

        # Cap
        adj_w = np.clip(adj_w, 0.0, max_weight)
        t2 = adj_w.sum()
        if t2 > 1e-12:
            adj_w /= t2

        return {sym: float(w) for sym, w in zip(syms, adj_w)}

    # ---------------------------------------------------------------- #
    # Score velocity (static utility, no state needed)
    # ---------------------------------------------------------------- #
    @staticmethod
    def score_velocity(rsr_window: np.ndarray) -> float:
        """
        Compute normalized RSR velocity from a recent window array.

        Parameters
        ----------
        rsr_window : array of RSR values, shape (vel_window+1,) at minimum.
                     rsr_window[-1] = current, rsr_window[0] = vel_window days ago.

        Returns
        -------
        Normalized velocity clipped to [-1, 1].
        """
        if len(rsr_window) < 2:
            return 0.0
        cur  = float(rsr_window[-1])
        prev = float(rsr_window[0])
        denom = max(abs(prev), 1e-4)
        return float(np.clip((cur - prev) / denom, -1.0, 1.0))

    # ---------------------------------------------------------------- #
    # Convenience wrapper (stateless shortcut, no day-index context)
    # ---------------------------------------------------------------- #
    def estimate_alpha(
        self,
        score:     float,
        gap_pct:   float,
        fill_prob: float,
    ) -> float:
        """
        Stateless shortcut: estimate expected alpha without a date_idx.
        Uses len(buffer) as a synthetic date_idx so decay weights are neutral.
        Returns only the scalar expected_alpha (float).

        Suitable for standalone testing and live single-shot queries.
        """
        date_idx = len(self._buf)   # synthetic; all ages become 0 → full weight
        ea, _, _ = self.estimate_expected_alpha(date_idx, score, gap_pct, fill_prob)
        return ea


# ------------------------------------------------------------------ #
# Public alias
# ------------------------------------------------------------------ #
CapitalEfficiency = CapitalEfficiencyModule
