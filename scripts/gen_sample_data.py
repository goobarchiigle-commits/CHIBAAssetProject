"""Generate data/sample.csv: trend + sine + Gaussian noise."""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV  = BASE_DIR / "data" / "sample.csv"

TREND_START = 100.0
TREND_SLOPE = 0.05
SINE_AMP    = 12.0
SINE_PERIOD = 30.0
NOISE_STD   = 0.015   # 1.5% of price (stress test; was 0.3%)
FLIP_PROB   = 0.10    # regime disruption: sign-flip rate on returns
SEED        = 42
N_ROWS      = 253
START_DATE  = "2024-01-04"


def generate(seed: int = SEED, noise: float = NOISE_STD,
             flip_prob: float = FLIP_PROB) -> None:
    rng       = np.random.default_rng(seed)
    dates     = pd.bdate_range(START_DATE, periods=N_ROWS)
    i         = np.arange(N_ROWS, dtype=float)
    base      = TREND_START + TREND_SLOPE * i + SINE_AMP * np.sin(2 * np.pi * i / SINE_PERIOD)
    noise_arr = rng.normal(0.0, base * noise)
    close     = np.round(base + noise_arr, 2)
    if flip_prob > 0:
        rets            = np.diff(close) / close[:-1]
        flip_mask       = rng.random(len(rets)) < flip_prob
        rets[flip_mask] *= -1
        close[1:]       = close[0] * np.cumprod(1.0 + rets)
        close           = np.round(close, 2)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[GEN] {OUT_CSV}  ({len(df)} rows, noise={noise:.1%}, flip={flip_prob:.0%}, seed={seed})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",      type=int,   default=SEED)
    ap.add_argument("--noise",     type=float, default=NOISE_STD)
    ap.add_argument("--flip-prob", type=float, default=FLIP_PROB)
    args = ap.parse_args()
    generate(args.seed, args.noise, args.flip_prob)
