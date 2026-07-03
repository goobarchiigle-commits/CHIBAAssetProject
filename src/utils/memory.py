from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd


EXCLUDE_FLOAT64_COLS = {
    "equity_curve",
    "cumulative_return",
    "portfolio_value",
    "pnl_cumulative",
}


def optimize_backtest_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    float_cols  = ["open", "high", "low", "close", "atr", "mom21d"]
    volume_cols = ["volume"]

    lower_map = {str(col).lower(): col for col in df.columns}
    for col in float_cols:
        real_col = lower_map.get(col)
        if real_col is None or real_col in EXCLUDE_FLOAT64_COLS:
            continue
        df[real_col] = pd.to_numeric(df[real_col], downcast="float")

    for col in volume_cols:
        real_col = lower_map.get(col)
        if real_col is None:
            continue
        s = pd.to_numeric(df[real_col], errors="coerce")
        if s.isna().any():
            df[real_col] = s.astype("Int32")
        else:
            df[real_col] = s.astype(np.int32)

    if "code" in df.columns:
        if df["code"].isna().any():
            df["code"] = df["code"].astype("Int32")
        else:
            df["code"] = pd.to_numeric(df["code"], downcast="integer")

    if "sector" in df.columns and len(df) > 0:
        unique_ratio = df["sector"].nunique(dropna=False) / len(df)
        if unique_ratio < 0.5:
            df["sector"] = df["sector"].astype("category")

    return df


def log_optimization(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    label: str = "DataFrame",
    verbose: bool = False,
) -> None:
    if not verbose:
        return
    before_mb = before_df.memory_usage(deep=True).sum() / 1024**2
    after_mb = after_df.memory_usage(deep=True).sum() / 1024**2
    ratio = (after_mb / before_mb) if before_mb > 0 else 1.0

    print(f"[MEM] {label} before: {before_mb:.2f} MB")
    print(f"[MEM] {label} after : {after_mb:.2f} MB")
    print(f"[MEM] {label} saved : {before_mb - after_mb:.2f} MB")
    print(f"[MEM] {label} ratio : {ratio:.2%}")


def get_process_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024**2

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel32 = ctypes.WinDLL("Kernel32.dll")
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024**2
    except Exception:
        return None

    return None


def collect_and_log_process_memory(label: str, verbose: bool = True) -> float | None:
    gc.collect()
    rss_mb = get_process_rss_mb()
    if verbose and rss_mb is not None:
        print(f"[PROC MEM] {label}: {rss_mb:.1f} MB")
    return rss_mb
